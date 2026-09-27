"""Action-Dependent Log-Prob ATE Reward（v4 + main experiment routed evidence）。

=== v3 → v4 变更 ===

新增三种 reward mode，解决 hard gating 导致 group 内 reward 方差不足的问题：

1. reward_mode="legacy_hard_gate"
   - 完全保留 v3 旧逻辑（correct → clamp(delta,0,cap)，wrong → -1.0，other → -1.25）
   - 用于做 A/B 对照

2. reward_mode="raw_delta"
   - 不做 hard gating，直接返回连续 delta_logprob
   - 只做极轻微 clip 防 NaN/爆炸（±10.0）
   - 用于做 Naked Delta Test

3. reward_mode="soft_shaping"（默认）
   - correctness 只决定 bias / 大方向，连续信号负责排序
   - 新增 answer-conditioned yes/no margin shaping 项
   - wrong / other 不再是纯常数桶

=== soft_shaping 的核心思路 ===

主信号保持不变：
  delta_resp = mean_logprob(response | V_orig) - mean_logprob(response | V_cf)

新增 answer-conditioned yes/no margin：
  从 prompt 末位置（预测第一个答案 token 的 logits 位置）读取 yes/no logsumexp margin，
  然后根据 parsed_answer 做方向绑定：
    - parsed="yes"  → delta_ans_margin = +(margin_orig - margin_cf)
    - parsed="no"   → delta_ans_margin = -(margin_orig - margin_cf)
    - parsed="other" → delta_ans_margin = 0.0

  score_resp = tanh(delta_resp / tau_resp)
  score_ans  = tanh(delta_ans_margin / tau_ans)
  score_base = alpha_resp * score_resp + alpha_ans * score_ans

最终 reward 用 bias + scale * continuous_score 形式，而非纯 hard gate。

主实验补充：
  - 在保留旧 smoke reward_mode 的同时，新增 relative evidence margin
  - 新增三路正式 reward 字段：
      reward_main_correctness_only
      reward_main_additive_evidence
      reward_main_routed_gated_evidence
  - existence family 只保留 correctness-only 主训练路径，evidence 只做 probe/audit
"""

from __future__ import annotations
import sys
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from visual_token_map import (
    compute_merge_ratio, bbox_to_token_indices,
    surrounding_indices, find_visual_range, to_absolute,
    reset_surrounding_meta,
)
from ced_core import (
    TokenReplacer, replacing, prepare_inputs, get_image_token_id,
    extend_multimodal_inputs,
    _collect_yes_no_token_ids, _logsumexp_logits,
)
from yesno_utils import parse_yesno as strict_parse_yesno
from yesno_utils import parse_yesno_with_source as strict_parse_yesno_with_source
from yesno_utils import is_low_entropy_yesno_family
from answer_format_utils import (
    infer_task_family,
    task_aware_answer_score,
    task_family_allows_evidence_training,
)
from proposal_utils import build_intervention_proposals


# ─── 答案解析 ───

def parse_yesno(text: str) -> str:
    """鲁棒的 yes/no 解析。返回 'yes', 'no', 'other'。"""
    return strict_parse_yesno(text)


def _parse_yesno_with_source(text: str):
    """返回 (parsed_answer, parse_source)。
    parse_source 表示解析命中路径：'first_token', 'boundary_regex', 'uncertain', 'other'。
    用于在诊断中区分"首 token 直接命中"和"边界正则兜底"。
    """
    return strict_parse_yesno_with_source(text)


def check_correct(parsed: str, gt_answer: Optional[str],
                   gt_present: Optional[bool], task_type: str = "", question: str = "") -> Optional[bool]:
    """检查 parsed(yes/no) 是否正确。

    IMPORTANT:
      - 本 reward 的 correctness 只对 yes/no / existence 任务有定义。
      - counting/spatial/attribute 等 open-ended 任务即便带有 `gt_present`
        元数据，也不应被强行映射到 yes/no correctness。
    """
    family = infer_task_family(task_type=task_type, question=question)
    if family != "existence":
        return None

    expected = None
    if gt_present is not None:
        expected = "yes" if gt_present else "no"
    elif gt_answer:
        exp = parse_yesno(gt_answer)
        if exp != "other":
            expected = exp

    if expected is None:
        return None
    if parsed == "other":
        return False
    return parsed == expected


# ─── 主类 ───

class ActionLogProbATEReward:
    """Action-dependent reward（v4: 三种 reward mode + answer-conditioned soft shaping）。"""

    def __init__(
        self,
        model,
        processor,
        device: str = "cuda:0",
        replace_mode: str = "mean",
        # --- legacy 参数（legacy_hard_gate 模式使用）---
        reward_cap: float = 5.0,
        wrong_penalty: float = -1.0,
        unparseable_penalty: float = -1.25,
        max_response_tokens: int = 64,
        # --- v4 新参数 ---
        reward_mode: str = "soft_shaping",
        tau_resp: float = 0.20,
        tau_ans: float = 1.00,
        alpha_resp: float = 0.70,
        alpha_ans: float = 0.30,
        min_reward: float = -1.25,
        max_reward: float = 1.00,
        # --- 主实验参数：默认不新增复杂超参，只保留最少必要项 ---
        main_reward_mode: str = "routed_gated_evidence",
        negative_intervention_k: int = 3,
        evidence_eps: float = 0.10,
    ):
        assert reward_mode in ("legacy_hard_gate", "raw_delta", "soft_shaping"), \
            f"Unknown reward_mode: {reward_mode}"
        assert main_reward_mode in ("correctness_only", "additive_evidence", "routed_gated_evidence"), \
            f"Unknown main_reward_mode: {main_reward_mode}"

        self.model = model
        self.processor = processor
        self.device = device
        self.reward_cap = reward_cap
        self.wrong_penalty = wrong_penalty
        self.unparseable_penalty = unparseable_penalty
        self.max_response_tokens = max_response_tokens

        # v4: soft shaping 参数
        self.reward_mode = reward_mode
        self.tau_resp = tau_resp
        self.tau_ans = tau_ans
        self.alpha_resp = alpha_resp
        self.alpha_ans = alpha_ans
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.main_reward_mode = main_reward_mode
        self.negative_intervention_k = max(0, int(negative_intervention_k))
        self.evidence_eps = float(evidence_eps)

        self.rep = TokenReplacer()
        self.rep.mode = replace_mode
        self.rep.register(model)

        self._img_tid = get_image_token_id(processor)
        self._tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        # 预收集 yes/no token ids（复用 ced_core 的 _collect_yes_no_token_ids，不重复发明）
        self._yes_ids, self._no_ids = _collect_yes_no_token_ids(self._tok)
        self.reward_name = "reward_action_logprob_ate"
        self.reward_scope = "candidate"
        self.reward_source = "response_conditioned"
        self._warned_runtime_device_fallback = False

    def cleanup(self):
        self.rep.remove()

    def resolve_runtime_device(self) -> str:
        """Resolve an effective device at runtime.

        If caller requested CUDA but current process has no visible CUDA device,
        fall back to the model's own parameter device when possible, otherwise CPU.
        """
        requested = str(self.device or "cuda:0")
        if requested.startswith("cuda") and (not torch.cuda.is_available()):
            model_dev = torch.device("cpu")
            try:
                model_dev = next(self.model.parameters()).device
            except Exception:
                pass

            if getattr(model_dev, "type", "") == "cuda":
                effective = str(model_dev)
            else:
                effective = "cpu"

            if not self._warned_runtime_device_fallback:
                print(
                    f"[ActionLogProbATEReward] requested device={requested} but CUDA is unavailable; "
                    f"fallback to runtime device={effective}"
                )
                self._warned_runtime_device_fallback = True
            return effective

        return requested

    def _get_visual_targets(self, inputs, bbox, img_w, img_h, sur_ring=2):
        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is None:
            return None, None, 0, 0
        g = grid_thw[0].tolist()
        gt_val, gh, gw = int(g[0]), int(g[1]), int(g[2])
        vs, ve = find_visual_range(inputs["input_ids"], self._img_tid)
        n_vis = ve - vs
        mr = compute_merge_ratio(gt_val * gh * gw, n_vis)
        tgt_rel = bbox_to_token_indices(bbox, img_w, img_h, gh, gw, mr)
        if not tgt_rel:
            return None, None, 0, 0
        reset_surrounding_meta()
        sur_rel = surrounding_indices(tgt_rel, n_vis, gh, gw, mr, ring=sur_ring)
        tgt_abs = to_absolute(inputs["input_ids"], tgt_rel, self._img_tid)
        sur_abs = to_absolute(inputs["input_ids"], sur_rel, self._img_tid)
        return tgt_abs, sur_abs, len(tgt_abs), len(sur_abs)

    def _compute_yesno_margin(self, logits, pos: int) -> float:
        """从 logits 的 pos 位置提取 yes/no margin = logsumexp(yes) - logsumexp(no)。
        为什么用 logsumexp 而不是 argmax：覆盖多候选 token，更鲁棒。
        如果缺 yes/no token ids 就返回 0.0，不会把程序搞崩。
        """
        if not self._yes_ids or not self._no_ids:
            return 0.0
        lg = logits[:, pos, :]  # [B, V]
        ly = _logsumexp_logits(lg, self._yes_ids)
        ln = _logsumexp_logits(lg, self._no_ids)
        return (ly - ln).mean().item()

    @staticmethod
    def _mean_logprob_from_logits(
        logits: torch.Tensor,
        *,
        prompt_len: int,
        response_ids: List[int],
        runtime_device: str,
        scored_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        n_resp = len(response_ids)
        resp_ids_t = torch.tensor([response_ids], device=runtime_device, dtype=torch.long)
        lp = F.log_softmax(
            logits[:, prompt_len - 1: prompt_len - 1 + n_resp, :], dim=-1,
        )
        token_lp = lp.gather(-1, resp_ids_t.unsqueeze(-1)).squeeze(-1)
        if scored_indices is not None:
            idxs = [int(i) for i in scored_indices if 0 <= int(i) < n_resp]
            if idxs:
                token_lp = token_lp[:, idxs]
            else:
                idxs = []
        else:
            idxs = []
        token_lp_diff = token_lp.squeeze(0)
        return {
            "token_lp": token_lp,
            "mean_logprob": token_lp.mean().item() if token_lp.numel() > 0 else 0.0,
            "token_count": int(token_lp.shape[-1]) if token_lp.ndim >= 2 else 0,
            "indices_used": idxs,
            "token_lp_vector": token_lp_diff,
        }

    def _forward_with_intervention(
        self,
        *,
        extended_inputs: Dict[str, Any],
        abs_indices: List[int],
        sur_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """执行一次指定 intervention 的前向。"""
        if not abs_indices:
            return self.model(**extended_inputs).logits.float()
        self.rep.set(abs_indices, sur_indices or [])
        with replacing(self.rep):
            return self.model(**extended_inputs).logits.float()

    @staticmethod
    def _evidence_gate(relative_margin: float, tau_resp: float) -> float:
        if tau_resp <= 0:
            return 1.0 if relative_margin > 0 else 0.0
        return 0.5 * (1.0 + math.tanh(float(relative_margin) / float(tau_resp)))

    @staticmethod
    def _correctness_reward(
        *,
        answer_text: str,
        question: str,
        task_type: str,
        gt_answer: Optional[str],
        gt_present: Optional[bool],
    ) -> Dict[str, Any]:
        """主实验 correctness-only 主项。

        这里保持最小化设计：
        - 正确 = +1.0
        - 错误 = -0.25
        - 无 GT = 0.0
        """
        family = infer_task_family(task_type=task_type, question=question)
        gold = (gt_answer or "").strip()
        if (not gold) and gt_present is not None and is_low_entropy_yesno_family(family):
            gold = "yes" if bool(gt_present) else "no"
        has_gt = bool(gold)
        score = (
            float(task_aware_answer_score(answer_text, gold, task_type=task_type, question=question))
            if has_gt else 0.0
        )
        if not has_gt:
            reward = 0.0
            correct = None
        else:
            correct = bool(score >= 1.0)
            reward = 1.0 if correct else -0.25
        return {
            "correctness_score": score,
            "correctness_reward": reward,
            "correctness_known": has_gt,
            "correctness_is_correct": correct,
        }

    def _compose_main_rewards(
        self,
        *,
        task_family: str,
        correctness_reward: float,
        correctness_known: bool,
        relative_margin: float,
    ) -> Dict[str, Any]:
        """三种主实验 reward 模式统一在这里生成。"""
        evidence_allowed = task_family_allows_evidence_training(task_family)
        gate = self._evidence_gate(relative_margin, self.tau_resp)
        additive = float(correctness_reward)
        routed = float(correctness_reward)
        if evidence_allowed:
            additive = float(correctness_reward) + self.evidence_eps * float(relative_margin)
            routed = float(correctness_reward) * gate + self.evidence_eps * float(relative_margin)
        mode_to_value = {
            "correctness_only": float(correctness_reward),
            "additive_evidence": float(additive),
            "routed_gated_evidence": float(routed),
        }
        requested = (self.main_reward_mode or "routed_gated_evidence").strip().lower()
        effective = requested
        if requested in {"additive_evidence", "routed_gated_evidence"} and not evidence_allowed:
            effective = "correctness_only"
        active_value = mode_to_value.get(requested, mode_to_value["routed_gated_evidence"])
        if effective != requested:
            active_value = mode_to_value[effective]
        return {
            "reward_main_correctness_only": mode_to_value["correctness_only"],
            "reward_main_additive_evidence": mode_to_value["additive_evidence"],
            "reward_main_routed_gated_evidence": mode_to_value["routed_gated_evidence"],
            "main_reward_mode_requested": requested,
            "main_reward_mode_effective": effective,
            "main_reward_active": float(active_value),
            "main_reward_evidence_gate": float(gate),
            "main_reward_evidence_allowed": bool(evidence_allowed),
            "main_reward_correctness_known": bool(correctness_known),
        }

    def _soft_shaping_reward(self, score_base: float, is_correct, parsed: str):
        """soft_shaping 模式的最终 reward：bias + scale * continuous_score。

        为什么用这个结构而不是纯 hard gate：
          - hard gate 把不同 delta 压进同一个桶，导致 GRPO advantage 方差不足
          - bias + scale 让 correctness 只控制大方向，连续分数负责组内排序
          - wrong 样本也保留连续信号，不再是常数桶
        """
        if is_correct is True:
            # 答对了：正 bias + 只放大正信号（negative score 截断为 0，不让答对的被惩罚）
            reward_bias = 0.25
            reward_scale = 0.75
            reward = reward_bias + reward_scale * max(0.0, score_base)
        elif is_correct is False and parsed == "other":
            # 无法解析：强负 bias，保留微弱连续信号做排序
            reward_bias = -1.00
            reward_scale = 0.10
            reward = reward_bias + reward_scale * score_base
        elif is_correct is False:
            # 答错了（至少给出了 yes/no）：中等负 bias + 连续信号
            reward_bias = -0.60
            reward_scale = 0.20
            reward = reward_bias + reward_scale * score_base
        else:
            # correctness unknown：纯连续信号（不加 bias）
            reward_bias = 0.00
            reward_scale = 0.25
            reward = reward_bias + reward_scale * score_base

        reward_before_clip = reward
        reward = max(self.min_reward, min(self.max_reward, reward))
        return reward, reward_bias, reward_scale, reward_before_clip

    @torch.no_grad()
    def compute(
        self,
        image: Image.Image,
        question: str,
        target_bbox: List[float],
        response_text: str,
        # New: parse/label should prefer final answer text, not the full response.
        # This allows response_text to include "Evidence: ...\nFinal answer: ..." without
        # poisoning yes/no parsing or correctness.
        answer_text: Optional[str] = None,
        enable_yesno_margin: bool = True,
        gt_answer: Optional[str] = None,
        gt_present: Optional[bool] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        sur_ring: int = 2,
        response_text_source: str = "raw_generated_text",
        answer_text_source: str = "generated_text",
        token_indices_to_score: Optional[List[int]] = None,
        reward_logprob_scope: str = "full_response",
        strict_scope: bool = False,
        score_response_text: Optional[str] = None,
        reward_condition_on_prefix: bool = True,
        task_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """计算单样本的 action-dependent reward。支持 3 种 reward_mode。"""
        rsrc = (response_text_source or "").strip().lower()
        asrc = (answer_text_source or "").strip().lower()
        if rsrc == "generated_text":
            raise ValueError(
                "ActionLogProbATEReward contract violation: response_text must come from raw_generated_text, "
                "not generated_text(final_answer)"
            )
        if asrc == "raw_generated_text":
            raise ValueError(
                "ActionLogProbATEReward contract violation: answer_text should come from generated_text(final_answer), "
                "not raw_generated_text"
            )

        iw = image_width or image.size[0]
        ih = image_height or image.size[1]
        metadata = dict(metadata or {})

        parse_text = answer_text if answer_text is not None else response_text
        parsed, parse_source = _parse_yesno_with_source(parse_text)
        is_correct = check_correct(parsed, gt_answer, gt_present, task_type=task_type, question=question)
        task_family = infer_task_family(task_type=task_type, question=question, metadata=metadata)
        correctness_pack = self._correctness_reward(
            answer_text=parse_text,
            question=question,
            task_type=task_type,
            gt_answer=gt_answer,
            gt_present=gt_present,
        )

        try:
            t_start = time.time()
            runtime_device = self.resolve_runtime_device()
            reward_condition_on_prefix = bool(reward_condition_on_prefix)
            if reward_condition_on_prefix:
                scored_response_text = score_response_text if score_response_text is not None else response_text
            else:
                # Answer-only scoring path: isolate the final answer string so identical
                # final answers receive identical token logprob inputs, independent of
                # earlier reasoning prefix text.
                scored_response_text = answer_text if answer_text is not None else response_text
            response_ids = self._tok.encode(scored_response_text or "", add_special_tokens=False)
            if not response_ids:
                return self._make_result(
                    response_text, parsed, is_correct,
                    parse_source=parse_source,
                    error="empty_response_tokens",
                    score_response_text=scored_response_text,
                    task_family=task_family,
                )
            response_ids = response_ids[: self.max_response_tokens]
            n_resp = len(response_ids)

            inputs = prepare_inputs(self.processor, image, question, runtime_device)
            prompt_len = inputs["input_ids"].shape[1]

            requested_scope = (reward_logprob_scope or "full_response").strip().lower()
            if requested_scope not in {"full_response", "final_answer"}:
                requested_scope = "full_response"
            effective_scope = requested_scope
            scored_token_indices: Optional[List[int]] = None
            if requested_scope == "final_answer":
                if not reward_condition_on_prefix:
                    # The scored response is already isolated to the answer text, so the
                    # full isolated response equals the final-answer scope.
                    effective_scope = "answer_text_isolated"
                else:
                    idxs = [int(i) for i in (token_indices_to_score or []) if 0 <= int(i) < n_resp]
                    if idxs:
                        scored_token_indices = list(idxs)
                    elif strict_scope:
                        return self._make_result(
                            response_text,
                            parsed,
                            is_correct,
                            parse_source=parse_source,
                            error="missing_reward_token_span",
                            reward_response_text_source=response_text_source,
                            reward_answer_text_source=answer_text_source,
                            reward_logprob_scope_requested=requested_scope,
                            reward_logprob_scope_effective="invalid_final_answer",
                            reward_condition_on_prefix=reward_condition_on_prefix,
                            task_family=task_family,
                        )
                    else:
                        effective_scope = "full_response"

            proposal_bundle = build_intervention_proposals(
                inputs=inputs,
                img_tid=self._img_tid,
                target_bbox=target_bbox,
                image_width=iw,
                image_height=ih,
                metadata=metadata,
                sur_ring=sur_ring,
                negative_k=self.negative_intervention_k,
            )
            positive = proposal_bundle["positive"]
            negatives = proposal_bundle["negatives"]
            proposal_diag = proposal_bundle["diagnostics"]
            n_tgt = len(positive["absolute_indices"])
            n_sur = len(positive["surrounding_absolute_indices"])

            extended = extend_multimodal_inputs(inputs, response_ids, runtime_device)

            # 原始前向只做一次，后续相关/无关干预都复用相同的 extended inputs。
            self.rep.active = False
            logits_orig = self.model(**extended).logits.float()
            orig_stats = self._mean_logprob_from_logits(
                logits_orig,
                prompt_len=prompt_len,
                response_ids=response_ids,
                runtime_device=runtime_device,
                scored_indices=scored_token_indices,
            )

            logits_cf = self._forward_with_intervention(
                extended_inputs=extended,
                abs_indices=positive["absolute_indices"],
                sur_indices=positive["surrounding_absolute_indices"],
            )
            pos_stats = self._mean_logprob_from_logits(
                logits_cf,
                prompt_len=prompt_len,
                response_ids=response_ids,
                runtime_device=runtime_device,
                scored_indices=scored_token_indices,
            )

            n_scored_tokens = int(orig_stats["token_count"])
            mean_lp_orig = orig_stats["mean_logprob"]
            mean_lp_cf = pos_stats["mean_logprob"]
            s_pos = mean_lp_orig - mean_lp_cf

            neg_scores: List[float] = []
            neg_mean_logprobs: List[float] = []
            neg_area_fractions: List[float] = []
            for neg in negatives:
                logits_neg = self._forward_with_intervention(
                    extended_inputs=extended,
                    abs_indices=neg.get("absolute_indices", []),
                    sur_indices=neg.get("surrounding_absolute_indices", []),
                )
                neg_stats = self._mean_logprob_from_logits(
                    logits_neg,
                    prompt_len=prompt_len,
                    response_ids=response_ids,
                    runtime_device=runtime_device,
                    scored_indices=scored_token_indices,
                )
                neg_mean = neg_stats["mean_logprob"]
                neg_mean_logprobs.append(float(neg_mean))
                neg_scores.append(float(mean_lp_orig - neg_mean))
                neg_area_fractions.append(float(neg.get("proposal_area_fraction", float("nan"))))

            s_neg_mean = (sum(neg_scores) / len(neg_scores)) if neg_scores else 0.0
            # Paper Eq.3: m = tanh((s_ev - μ_non) / (σ_non + ε)).
            # Archived code used the unstandardized difference s_pos - mean(s_neg).
            if neg_scores:
                mu = float(s_neg_mean)
                if len(neg_scores) >= 2:
                    var = sum((float(x) - mu) ** 2 for x in neg_scores) / float(len(neg_scores))
                    sigma = math.sqrt(max(var, 0.0))
                else:
                    sigma = 0.0
                relative_margin = float(math.tanh((float(s_pos) - mu) / (sigma + 1e-6)))
            else:
                relative_margin = float(math.tanh(float(s_pos)))
            delta = relative_margin

            # ── 实现健康度诊断字段（v3 保留） ──
            token_lp_diff = orig_stats["token_lp_vector"] - pos_stats["token_lp_vector"]
            abs_diff = token_lp_diff.abs()
            mean_abs_diff = abs_diff.mean().item()
            max_abs_diff = abs_diff.max().item()

            # ── v4 新增：answer-conditioned yes/no margin ──
            # 位置 prompt_len-1 是预测第一个 answer token 的 logits 位置
            # 这里从已有的两次前向 logits 中直接读取，不引入额外的前向次数
            ans_margin_pos = prompt_len - 1
            if enable_yesno_margin:
                margin_orig = self._compute_yesno_margin(logits_orig, ans_margin_pos)
                margin_cf = self._compute_yesno_margin(logits_cf, ans_margin_pos)
                margin_delta = margin_orig - margin_cf

                # answer-conditioning：静态 yes/no margin 变成和 response 方向绑定的 shaping 项
                if parsed == "yes":
                    delta_ans_margin = margin_delta
                elif parsed == "no":
                    delta_ans_margin = -margin_delta
                else:
                    delta_ans_margin = 0.0
            else:
                margin_orig = 0.0
                margin_cf = 0.0
                margin_delta = 0.0
                delta_ans_margin = 0.0

            # tanh 归一化：让不同量纲的信号在可比范围内
            # tau 越小，信号越敏感（容易饱和）；越大，信号越线性
            score_resp = math.tanh(relative_margin / self.tau_resp) if self.tau_resp > 0 else 0.0
            score_ans = (
                math.tanh(delta_ans_margin / self.tau_ans)
                if (enable_yesno_margin and self.tau_ans > 0)
                else 0.0
            )
            score_base = self.alpha_resp * score_resp + self.alpha_ans * score_ans

            # Always compute supervised-shaped branch for explicit decomposition.
            supervised_shaped_reward, supervised_reward_bias, supervised_reward_scale, \
                supervised_reward_before_clip = self._soft_shaping_reward(
                    score_base, is_correct, parsed
                )
            main_reward_pack = self._compose_main_rewards(
                task_family=task_family,
                correctness_reward=float(correctness_pack["correctness_reward"]),
                correctness_known=bool(correctness_pack["correctness_known"]),
                relative_margin=float(relative_margin),
            )

            # ── 按 reward_mode 计算最终 reward ──
            if self.reward_mode == "legacy_hard_gate":
                # 完全保留 v3 旧逻辑（用于 A/B 对照）
                if is_correct is True:
                    reward = max(0.0, min(delta, self.reward_cap))
                elif is_correct is False and parsed == "other":
                    reward = self.unparseable_penalty
                elif is_correct is False:
                    reward = self.wrong_penalty
                else:
                    reward = max(0.0, min(delta, self.reward_cap)) * 0.5
                reward_bias = 0.0
                reward_scale = 0.0
                reward_before_clip = reward

            elif self.reward_mode == "raw_delta":
                # 裸 delta，只做极端值防护
                reward = max(-10.0, min(10.0, delta))
                if math.isnan(reward) or math.isinf(reward):
                    reward = 0.0
                reward_bias = 0.0
                reward_scale = 1.0
                reward_before_clip = reward

            elif self.reward_mode == "soft_shaping":
                reward, reward_bias, reward_scale, reward_before_clip = \
                    self._soft_shaping_reward(score_base, is_correct, parsed)

            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

            elapsed = time.time() - t_start

            return {
                # 主字段
                "reward": reward,
                "reward_name": self.reward_name,
                "reward_scope": self.reward_scope,
                "reward_source": self.reward_source,
                "delta_logprob": delta,
                "mean_logprob_orig": mean_lp_orig,
                "mean_logprob_cf": mean_lp_cf,
                "n_response_tokens": n_resp,
                "n_scored_response_tokens": n_scored_tokens,
                "parsed_answer": parsed,
                "is_correct": is_correct,
                "task_family": task_family,
                "response_text": response_text,
                "score_response_text": (score_response_text if score_response_text is not None else response_text),
                "reward_score_response_isolated": bool(score_response_text is not None and (score_response_text or "") != (response_text or "")) or (not reward_condition_on_prefix),
                "error": None,
                # v3 旧字段（向后兼容）
                "raw_reward_before_gating": delta,
                "raw_reward_after_gating": reward,
                "has_visual_targets": True,
                "n_visual_target_tokens": n_tgt,
                "n_surrounding_tokens": n_sur,
                "mean_abs_token_lp_diff": mean_abs_diff,
                "max_abs_token_lp_diff": max_abs_diff,
                # v4 新增字段
                "reward_mode": self.reward_mode,
                "main_reward_mode_requested": main_reward_pack["main_reward_mode_requested"],
                "main_reward_mode_effective": main_reward_pack["main_reward_mode_effective"],
                "main_reward_active": main_reward_pack["main_reward_active"],
                "reward_main_correctness_only": main_reward_pack["reward_main_correctness_only"],
                "reward_main_additive_evidence": main_reward_pack["reward_main_additive_evidence"],
                "reward_main_routed_gated_evidence": main_reward_pack["reward_main_routed_gated_evidence"],
                "main_reward_evidence_gate": main_reward_pack["main_reward_evidence_gate"],
                "main_reward_evidence_allowed": main_reward_pack["main_reward_evidence_allowed"],
                "main_reward_correctness_known": main_reward_pack["main_reward_correctness_known"],
                "correctness_score": correctness_pack["correctness_score"],
                "correctness_reward": correctness_pack["correctness_reward"],
                "correctness_known": correctness_pack["correctness_known"],
                "correctness_is_correct": correctness_pack["correctness_is_correct"],
                "s_pos": s_pos,
                "s_neg_mean": s_neg_mean,
                "s_neg_list": neg_scores,
                "relative_evidence_margin": relative_margin,
                "delta_ans_margin": delta_ans_margin,
                "yesno_margin_orig": margin_orig,
                "yesno_margin_cf": margin_cf,
                "score_resp": score_resp,
                "score_ans": score_ans,
                "score_base": score_base,
                "score_base_no_label": score_base,
                "supervised_shaped_reward": supervised_shaped_reward,
                "supervised_reward_bias": supervised_reward_bias,
                "supervised_reward_scale": supervised_reward_scale,
                "supervised_reward_before_clip": supervised_reward_before_clip,
                "reward_bias": reward_bias,
                "reward_scale": reward_scale,
                "reward_before_final_clip": reward_before_clip,
                "answer_margin_position": ans_margin_pos,
                "parse_source": parse_source,
                "reward_response_text_source": response_text_source,
                "reward_answer_text_source": answer_text_source,
                "reward_runtime_device": runtime_device,
                "reward_logprob_scope_requested": requested_scope,
                "reward_logprob_scope_effective": effective_scope,
                "reward_token_span_present": bool(scored_token_indices),
                "reward_condition_on_prefix": bool(reward_condition_on_prefix),
                "reward_semantics_name": ("response_conditioned_final_answer_ate" if reward_condition_on_prefix else "answer_only_final_answer_ate"),
                "n_negative_interventions": len(neg_scores),
                "extra_forward_passes": 1 + len(neg_scores),
                "reward_compute_time_sec": elapsed,
                "reward_cache_reuse_enabled": bool(proposal_diag.get("cache_reuse_enabled", True)),
                "proposal_source": proposal_diag.get("proposal_source"),
                "proposal_bbox_missing": proposal_diag.get("proposal_bbox_missing"),
                "proposal_fallback_used": proposal_diag.get("proposal_fallback_used"),
                "proposal_fallback_reason": proposal_diag.get("proposal_fallback_reason"),
                "proposal_area": proposal_diag.get("proposal_area"),
                "proposal_area_fraction": proposal_diag.get("proposal_area_fraction"),
                "proposal_token_count": proposal_diag.get("proposal_token_count"),
                "proposal_token_coverage_fraction": proposal_diag.get("proposal_token_coverage_fraction"),
                "negative_intervention_sources": proposal_diag.get("proposal_negative_sources", []),
                "negative_intervention_area_fractions": neg_area_fractions,
                "negative_intervention_mean_logprobs": neg_mean_logprobs,
            }

        except Exception as e:
            return self._make_result(
                response_text, parsed, is_correct,
                parse_source=parse_source,
                error=repr(e),
                reward_condition_on_prefix=reward_condition_on_prefix,
                task_family=task_family,
            )

    def _make_result(
        self,
        response_text,
        parsed,
        is_correct,
        parse_source="other",
        error=None,
        reward_response_text_source="raw_generated_text",
        reward_answer_text_source="generated_text",
        reward_logprob_scope_requested="full_response",
        reward_logprob_scope_effective="full_response",
        score_response_text=None,
        reward_condition_on_prefix=True,
        task_family: str = "other",
    ):
        """构造错误/空样本的返回 dict，key 结构与正常返回完全一致。"""
        if self.reward_mode == "legacy_hard_gate":
            if is_correct is True:
                reward = 0.0
            elif is_correct is False and parsed == "other":
                reward = self.unparseable_penalty
            elif is_correct is False:
                reward = self.wrong_penalty
            else:
                reward = 0.0
        elif self.reward_mode == "raw_delta":
            reward = 0.0
        elif self.reward_mode == "soft_shaping":
            if is_correct is True:
                reward = 0.25
            elif is_correct is False and parsed == "other":
                reward = -1.00
            elif is_correct is False:
                reward = -0.60
            else:
                reward = 0.0
        else:
            reward = 0.0

        return {
            "reward": reward,
            "reward_name": self.reward_name,
            "reward_scope": self.reward_scope,
            "reward_source": self.reward_source,
            "delta_logprob": 0.0,
            "mean_logprob_orig": 0.0,
            "mean_logprob_cf": 0.0,
            "n_response_tokens": 0,
            "n_scored_response_tokens": 0,
            "parsed_answer": parsed,
            "is_correct": is_correct,
            "task_family": task_family,
            "response_text": response_text,
            "score_response_text": (score_response_text if score_response_text is not None else response_text),
            "reward_score_response_isolated": bool(score_response_text is not None and (score_response_text or "") != (response_text or "")) or (not reward_condition_on_prefix),
            "error": error,
            "raw_reward_before_gating": 0.0,
            "raw_reward_after_gating": reward,
            "has_visual_targets": False,
            "n_visual_target_tokens": 0,
            "n_surrounding_tokens": 0,
            "mean_abs_token_lp_diff": 0.0,
            "max_abs_token_lp_diff": 0.0,
            "reward_mode": self.reward_mode,
            "delta_ans_margin": 0.0,
            "yesno_margin_orig": 0.0,
            "yesno_margin_cf": 0.0,
            "score_resp": 0.0,
            "score_ans": 0.0,
            "score_base": 0.0,
            "score_base_no_label": 0.0,
            "supervised_shaped_reward": 0.0,
            "supervised_reward_bias": 0.0,
            "supervised_reward_scale": 0.0,
            "supervised_reward_before_clip": 0.0,
            "reward_bias": 0.0,
            "reward_scale": 0.0,
            "reward_before_final_clip": reward,
            "answer_margin_position": 0,
            "parse_source": parse_source,
            "reward_response_text_source": reward_response_text_source,
            "reward_answer_text_source": reward_answer_text_source,
            "reward_logprob_scope_requested": reward_logprob_scope_requested,
            "reward_logprob_scope_effective": reward_logprob_scope_effective,
            "reward_token_span_present": False,
            "reward_condition_on_prefix": bool(reward_condition_on_prefix),
            "reward_semantics_name": ("response_conditioned_final_answer_ate" if reward_condition_on_prefix else "answer_only_final_answer_ate"),
            "main_reward_mode_requested": self.main_reward_mode,
            "main_reward_mode_effective": "correctness_only",
            "main_reward_active": 0.0,
            "reward_main_correctness_only": 0.0,
            "reward_main_additive_evidence": 0.0,
            "reward_main_routed_gated_evidence": 0.0,
            "main_reward_evidence_gate": 0.0,
            "main_reward_evidence_allowed": bool(task_family_allows_evidence_training(task_family)),
            "main_reward_correctness_known": False,
            "correctness_score": 0.0,
            "correctness_reward": 0.0,
            "correctness_known": False,
            "correctness_is_correct": None,
            "s_pos": 0.0,
            "s_neg_mean": 0.0,
            "s_neg_list": [],
            "relative_evidence_margin": 0.0,
            "n_negative_interventions": 0,
            "extra_forward_passes": 0,
            "reward_compute_time_sec": 0.0,
            "reward_cache_reuse_enabled": True,
            "proposal_source": "",
            "proposal_bbox_missing": False,
            "proposal_fallback_used": False,
            "proposal_fallback_reason": "",
            "proposal_area": 0.0,
            "proposal_area_fraction": 0.0,
            "proposal_token_count": 0,
            "proposal_token_coverage_fraction": 0.0,
            "negative_intervention_sources": [],
            "negative_intervention_area_fractions": [],
            "negative_intervention_mean_logprobs": [],
        }
