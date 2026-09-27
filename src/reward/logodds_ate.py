"""LogOdds ATE Reward：P1 训练的唯一主 reward。

原理：
  对目标区域的 visual token 做 zero-out（反事实替换），
  观测模型输出 logits 中 yes/no 的 log-odds 变化量（ATE = Average Treatment Effect）。

  reward = |logodds(yes/no | V_orig) - logodds(yes/no | V_cf)|

  值越大 → 模型回答越依赖该视觉区域的信息（高视觉锚定度）
  值越小 → 模型仅靠文本先验回答（低视觉锚定度 / 幻觉风险）

接口设计：
  reward = LogOddsATEReward(model, processor, device)
  score  = reward.compute(image, question, target_bbox)
  batch  = reward.compute_batch(samples)  # 批量接口

注意：
  - 本 class 只负责 reward 计算，不负责训练逻辑
  - 每次调用需要 2 次前向（原始 + 反事实），单样本约 200-400ms（A100）
  - 不跟踪中间层（节省显存），只看 logits 层
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

# 确保能 import 同级模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_loader import load, num_layers
from visual_token_map import (
    compute_merge_ratio, bbox_to_token_indices,
    surrounding_indices, find_visual_range, to_absolute,
    reset_surrounding_meta,
)
from ced_core import (
    TokenReplacer, replacing, prepare_inputs, get_image_token_id,
    _collect_yes_no_token_ids, _logsumexp_logits,
)


class LogOddsATEReward:
    """LogOdds ATE Reward 计算器。

    单一职责：输入 (image, question, target_bbox) → 输出 reward 分数。
    """

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda:0",
        replace_mode: str = "zero",
    ):
        self.model = model
        self.processor = processor
        self.device = device

        # Token 替换器（第一层 transformer 入口 zero-out）
        self.rep = TokenReplacer()
        self.rep.mode = replace_mode
        self.rep.register(model)

        # 预计算 yes/no token 候选
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self._yes_ids, self._no_ids = _collect_yes_no_token_ids(tok)
        self._img_tid = get_image_token_id(processor)

        if not self._yes_ids or not self._no_ids:
            raise RuntimeError(
                "无法找到 yes/no token ids，请检查 tokenizer"
            )
        self.reward_name = "reward_logodds_ate"
        self.reward_scope = "sample_probe"
        self.reward_source = "ced"

    def cleanup(self):
        """移除 hook。"""
        self.rep.remove()

    @torch.no_grad()
    def compute(
        self,
        image: Image.Image,
        question: str,
        target_bbox: List[float],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        sur_ring: int = 2,
    ) -> Dict[str, Any]:
        """计算单个样本的 logodds_ate reward。

        Args:
            image: PIL Image
            question: 问题文本
            target_bbox: COCO 格式 [x, y, w, h]
            image_width/height: 图片原始尺寸（可选，默认从 image 取）
            sur_ring: surrounding 环带宽度

        Returns:
            dict with keys:
              - reward: float, 主 reward 值（logodds_ate）
              - logodds_orig: float, 原始 log-odds
              - logodds_replaced: float, 替换后 log-odds
              - prob_yes_orig: float, 原始 P(yes)
              - prob_yes_replaced: float, 替换后 P(yes)
              - n_tgt_tokens: int, 被替换的 token 数
              - n_vis_tokens: int, 总视觉 token 数
              - error: str (仅出错时)
        """
        iw = image_width or image.size[0]
        ih = image_height or image.size[1]

        try:
            inputs = prepare_inputs(self.processor, image, question, self.device)

            # 解析 grid 信息
            grid_thw = inputs.get("image_grid_thw")
            if grid_thw is None:
                return {"error": "no image_grid_thw", "reward": 0.0}
            g = grid_thw[0].tolist()
            gt_val, gh, gw = int(g[0]), int(g[1]), int(g[2])

            # 视觉 token 范围
            vs, ve = find_visual_range(inputs["input_ids"], self._img_tid)
            n_vis = ve - vs
            merge_ratio = compute_merge_ratio(gt_val * gh * gw, n_vis)

            # bbox → token 索引
            tgt_rel = bbox_to_token_indices(target_bbox, iw, ih, gh, gw, merge_ratio)
            if not tgt_rel:
                return {"error": "empty target tokens", "reward": 0.0}

            reset_surrounding_meta()
            sur_rel = surrounding_indices(tgt_rel, n_vis, gh, gw, merge_ratio, ring=sur_ring)
            tgt_abs = to_absolute(inputs["input_ids"], tgt_rel, self._img_tid)
            sur_abs = to_absolute(inputs["input_ids"], sur_rel, self._img_tid)

            if not tgt_abs:
                return {"error": "empty absolute target", "reward": 0.0}

            # Pass 1: 原始前向
            self.rep.active = False
            logits1 = self.model(**inputs).logits.float()

            # Pass 2: 反事实前向（zero-out 目标 token）
            self.rep.set(tgt_abs, sur_abs)
            with replacing(self.rep):
                logits2 = self.model(**inputs).logits.float()

            # 取 prompt 最后一个 token 位置的 logits
            T = logits1.shape[1]
            pos = T - 1
            lg1 = logits1[:, pos, :]
            lg2 = logits2[:, pos, :]

            # 计算 log-odds ATE
            ly1 = _logsumexp_logits(lg1, self._yes_ids)
            ln1 = _logsumexp_logits(lg1, self._no_ids)
            ly2 = _logsumexp_logits(lg2, self._yes_ids)
            ln2 = _logsumexp_logits(lg2, self._no_ids)

            lo_orig = (ly1 - ln1).mean().item()
            lo_repl = (ly2 - ln2).mean().item()
            ate = abs(lo_orig - lo_repl)

            # 辅助信息
            p1 = F.softmax(lg1, dim=-1)
            p2 = F.softmax(lg2, dim=-1)
            py1 = p1[:, self._yes_ids].sum(dim=-1).mean().item()
            py2 = p2[:, self._yes_ids].sum(dim=-1).mean().item()

            return {
                "reward": ate,
                "reward_name": self.reward_name,
                "reward_scope": self.reward_scope,
                "reward_source": self.reward_source,
                "logodds_orig": lo_orig,
                "logodds_replaced": lo_repl,
                "prob_yes_orig": py1,
                "prob_yes_replaced": py2,
                "n_tgt_tokens": len(tgt_abs),
                "n_vis_tokens": n_vis,
            }

        except Exception as e:
            return {
                "error": repr(e),
                "reward": 0.0,
                "reward_name": self.reward_name,
                "reward_scope": self.reward_scope,
                "reward_source": self.reward_source,
            }

    @torch.no_grad()
    def compute_batch(
        self,
        samples: List[Dict[str, Any]],
        image_root: str = "",
        sur_ring: int = 2,
    ) -> List[Dict[str, Any]]:
        """批量计算 reward。

        每个 sample 需包含: image_file, question, target_bbox
        可选: image_width, image_height, image (PIL Image 对象)

        注意：当前实现是逐样本串行（因为 Qwen3-VL 的视觉 token 数量
        随图片尺寸变化，不方便做真正的 batch 推理）。
        """
        results = []
        for s in samples:
            if "image" in s and isinstance(s["image"], Image.Image):
                img = s["image"]
            else:
                import os
                path = os.path.join(image_root, s["image_file"])
                img = Image.open(path).convert("RGB")

            r = self.compute(
                image=img,
                question=s["question"],
                target_bbox=s["target_bbox"],
                image_width=s.get("image_width"),
                image_height=s.get("image_height"),
                sur_ring=sur_ring,
            )
            # 附带样本元数据
            r["image_file"] = s.get("image_file", "")
            r["question"] = s["question"]
            r["target_bbox"] = s["target_bbox"]
            r["task_type"] = s.get("task_type", "unknown")
            r["gt_present"] = s.get("gt_present", None)
            r["pair_id"] = s.get("pair_id", None)
            results.append(r)
        return results


def build_reward(model_dir: str, device: str = "cuda:0",
                 dtype: str = "bfloat16") -> LogOddsATEReward:
    """便捷工厂函数：加载模型 + 构建 reward 计算器。"""
    processor, model, cfg = load(model_dir, device, dtype)
    return LogOddsATEReward(model, processor, device=device)
