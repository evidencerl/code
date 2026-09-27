"""CED 核心：hook + token 替换 + JS/熵 计算。

修复记录（v2）：
  [FIX-1] TokenReplacer 支持三种替换模式：
          - "zero":  目标 token 置零（最强扰动，推荐首选）
          - "noise": 高斯噪声替换（VCD 风格）
          - "mean":  全局背景均值替换（原实现，扰动太弱）
  [FIX-2] 中间层度量修复：
          - 不再对 hidden states 做 softmax（hidden dim 不是概率分布）
          - 改用 cosine distance + L2 distance
          - 对 **目标区域 token** 取均值后度量，而不是只看最后一个 token

修复记录（v3）：
  [FIX-4] layer_X_js 字段名本质上装的是 cosine distance，已废弃。
          保留 layer_X_js 仅为兼容旧结果（值 = layer_X_tgt_cosine），
          新分析流程必须使用 layer_X_tgt_cosine / layer_X_tgt_l2 等真实字段名。
  [FIX-5] yes/no token 候选机制：考虑前导空格、大小写、分词差异，
          收集多候选 token id，用 logsumexp 聚合。
  [FIX-6] key_mode 观测模式扩展：
          - prompt_last: 只看 prompt 最后一个 token（原实现）
          - answer_first: 观测第一个答案 token 位置的 logits 变化
          - answer_span: 观测答案前 N 个 token，支持 mean/max 聚合
          - auto: existence 任务用 answer_first，其他用 prompt_last
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn.functional as F


# ─────── 度量 ───────

def js_div(p, q, eps: float = 1e-12):
    """对最后一维做对称 JS 距离，返回形状与 batch 相同的张量。"""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum(-1) + (q * (q / m).log()).sum(-1))


def ent(p, eps: float = 1e-12):
    """香农熵，单位：nat。"""
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(-1)


def kl_div(p, q, eps: float = 1e-12):
    """KL(p ‖ q)。"""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p / q).log()).sum(-1)


def cos_dist(a, b):
    """1 - cosine_similarity"""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


# ─────── 定位模型内部结构（兼容新旧 transformers） ───────

def _resolve_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _iter_wrapper_roots(model, max_depth: int = 6):
    """遍历常见 wrapper 解包路径，兼容 DDP / PEFT / HF 外层包装。"""
    queue = [(model, 0)]
    seen = set()
    while queue:
        node, depth = queue.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        if depth >= max_depth:
            continue
        for attr in ("module", "base_model", "model", "language_model"):
            try:
                child = getattr(node, attr)
            except (AttributeError, TypeError):
                continue
            if child is not None and child is not node:
                queue.append((child, depth + 1))


def _find_layers(model):
    candidate_paths = [
        "layers",
        "language_model.layers",
        "model.layers",
        "model.language_model.layers",
    ]
    for root in _iter_wrapper_roots(model):
        for path in candidate_paths:
            try:
                layers = _resolve_attr(root, path)
                if hasattr(layers, "__len__") and len(layers) > 0:
                    return layers
            except (AttributeError, TypeError):
                continue
    raise AttributeError(
        "Cannot find transformer layers. Tried: "
        "wrapper-unwrapped roots + layers / language_model.layers / "
        "model.layers / model.language_model.layers"
    )


def _find_embed(model):
    """找到 embedding 层（保留兜底）。"""
    candidate_paths = [
        "embed_tokens",
        "language_model.embed_tokens",
        "model.embed_tokens",
        "model.language_model.embed_tokens",
    ]
    for root in _iter_wrapper_roots(model):
        for path in candidate_paths:
            try:
                layer = _resolve_attr(root, path)
                if isinstance(layer, torch.nn.Module):
                    return layer
            except (AttributeError, TypeError):
                continue
    raise AttributeError("Cannot find embed_tokens layer")


# ─────── Hook：捕获 Hidden States ───────

class HiddenCapture:
    def __init__(self):
        self.data: Dict[int, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def register(self, model, layer_ids: List[int]):
        layers = _find_layers(model)
        for i in layer_ids:
            if i < 0 or i >= len(layers):
                continue

            def _hook(mod, inp, out, idx=i):
                hs = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(hs):
                    self.data[idx] = hs.detach()

            self._hooks.append(layers[i].register_forward_hook(_hook))

    def clear(self):
        self.data.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────── Hook：Token 替换（第一层 transformer 输入） ───────

class TokenReplacer:
    """在第一层 transformer 入口处替换 hidden_states 的 token 向量。

    [FIX-1] 支持三种替换模式：
      - "zero":  置零 → 最强扰动，信号最大
      - "noise": 高斯噪声（均值0，标准差取自目标token） → 中等扰动
      - "mean":  全局背景均值 → 最弱扰动（原实现）
    """

    def __init__(self):
        self._hook = None
        self.target: List[int] = []
        self.surround: List[int] = []
        self.active: bool = False
        self.mode: str = "zero"
        self.audit_mode: bool = False
        self._last_audit: Dict[str, Any] = {}

    def register(self, model):
        layers = _find_layers(model)
        first = layers[0]

        def _pre_hook(module, args):
            if (not self.active) or (not self.target):
                return
            if not args:
                return

            hs = args[0]
            if (not torch.is_tensor(hs)) or hs.dim() != 3:
                return

            B, T, C = hs.shape
            tgt = torch.as_tensor(self.target, device=hs.device, dtype=torch.long)
            tgt = tgt[(tgt >= 0) & (tgt < T)]
            if tgt.numel() == 0:
                return

            hs_new = hs.clone()

            if self.mode == "zero":
                hs_new[:, tgt, :] = 0.0

            elif self.mode == "noise":
                tgt_std = hs_new[:, tgt, :].std().item()
                noise = torch.randn_like(hs_new[:, tgt, :]) * max(tgt_std, 1e-6)
                hs_new[:, tgt, :] = noise

            elif self.mode == "mean":
                sur = torch.as_tensor(self.surround, device=hs.device, dtype=torch.long)
                sur = sur[(sur >= 0) & (sur < T)]
                if sur.numel() == 0:
                    bg = hs_new.mean(dim=1, keepdim=True)
                else:
                    bg = hs_new[:, sur, :].mean(dim=1, keepdim=True)
                hs_new[:, tgt, :] = bg

            else:
                raise ValueError(f"Unknown replacement mode: {self.mode}")

            # Audit: record before/after norms at replaced positions
            if self.audit_mode:
                with torch.no_grad():
                    before_norms = hs[:, tgt, :].norm(dim=-1)
                    after_norms = hs_new[:, tgt, :].norm(dim=-1)
                    self._last_audit = {
                        "replace_mode": self.mode,
                        "selected_positions": tgt.tolist(),
                        "n_positions": int(tgt.numel()),
                        "before_norm_mean": float(before_norms.mean().item()),
                        "before_norm_std": float(before_norms.std().item()),
                        "after_norm_mean": float(after_norms.mean().item()),
                        "after_norm_std": float(after_norms.std().item()),
                        "replacement_summary": {
                            "original_mean_abs": float(hs[:, tgt, :].abs().mean().item()),
                            "replaced_mean_abs": float(hs_new[:, tgt, :].abs().mean().item()),
                        },
                        "sequence_length": T,
                    }

            new_args = (hs_new,) + tuple(args[1:])
            return new_args

        self._hook = first.register_forward_pre_hook(_pre_hook)

    def set(self, target: List[int], surround: List[int]):
        self.target = list(target)
        self.surround = list(surround)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


@contextmanager
def replacing(rep: TokenReplacer):
    rep.active = True
    try:
        yield
    finally:
        rep.active = False


# ─────── [FIX-5] 稳健的 yes/no token 候选机制 ───────

def _collect_yes_no_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """收集 yes/no 的多候选 token id（考虑前导空格、大小写等分词差异）。

    局限性说明：
      - 这仍然只是 existence 场景的近似指标
      - 对于某些 tokenizer，"Yes" 可能被分成多个 token，这里只取首 token
      - 不能保证覆盖所有可能的编码方式，但已覆盖最常见的变体
    """
    yes_candidates = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
    no_candidates = ["no", " no", "No", " No", "NO", " NO"]

    yes_ids = set()
    no_ids = set()

    for w in yes_candidates:
        try:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                yes_ids.add(ids[0])
        except Exception:
            pass

    for w in no_candidates:
        try:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                no_ids.add(ids[0])
        except Exception:
            pass

    return sorted(yes_ids), sorted(no_ids)


def _logsumexp_logits(logits_1d: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
    """从 logits 向量中取指定 token ids，做 logsumexp 聚合。

    logsumexp 比 max 更平滑，等效于 log(sum(exp(logit_i)))，
    可以理解为"这些 token 联合出现的等效 logit"。
    """
    if not token_ids:
        return torch.tensor(float("-inf"), device=logits_1d.device)
    selected = logits_1d[..., token_ids]
    return torch.logsumexp(selected, dim=-1)


# ─────── [FIX-6] 辅助：拼接 answer token 到 inputs ───────

def extend_multimodal_inputs(
    inputs: Dict[str, torch.Tensor],
    extra_token_ids: List[int],
    device: str,
) -> Dict[str, torch.Tensor]:
    """安全地将 token ids 拼到多模态 inputs 末尾（teacher-forcing 用）。

    === 为什么不能浅拷贝整包 inputs 再只改 input_ids ===

    Qwen3-VL 的 processor 会生成与 seq_len 强绑定的张量，如 position_ids、
    rope_deltas、cache_position 等。如果浅拷贝整包 inputs 然后只扩展
    input_ids/attention_mask，这些旧长度的张量会原封不动传进 model.forward()，
    导致 shape mismatch crash（典型报错：
    "The shape of the mask [307] does not match ... [303]"）。

    === 为什么用白名单而不是黑名单 ===

    黑名单（只删 position_ids）不稳——Qwen3-VL/transformers 版本更新可能
    新增别的 seq-bound 键。白名单只保留已知安全的键，让模型前向自己根据
    新的 input_ids 长度动态计算 position_ids 等内部索引。

    === 为什么这比重跑 processor 更高效 ===

    重跑 processor 需要重新编码图像（视觉编码器前向），而这里只需要拼接
    几个 text token，图像侧的 pixel_values/image_grid_thw 完全不变。

    白名单保留的键：
      - input_ids: 扩展
      - attention_mask: 扩展（若存在）
      - pixel_values: 不变（图像特征，与 seq_len 无关）
      - image_grid_thw: 不变（grid 元信息，与 seq_len 无关）
      - mm_token_type_ids: Qwen3-VL mRoPE 需要，按文本 token=0 扩展

    显式丢弃/不透传的键（让模型前向自动重算）：
      - position_ids
      - rope_deltas
      - cache_position
      - token_type_ids（普通 BERT-style，不是 Qwen3-VL 的 mm_token_type_ids）
      - 以及任何其他与旧 seq_len 绑定的键
    """
    if not extra_token_ids:
        return inputs

    ids = inputs["input_ids"]
    B = ids.shape[0]

    # ── 白名单构造新 inputs ──
    new_inputs: Dict[str, torch.Tensor] = {}

    # 1. input_ids: 拼接 extra tokens
    extra = torch.tensor(
        [extra_token_ids], dtype=ids.dtype, device=device
    ).expand(B, -1)
    new_inputs["input_ids"] = torch.cat([ids, extra], dim=1)

    # 2. attention_mask: 拼接 ones（若存在）
    if "attention_mask" in inputs:
        mask = inputs["attention_mask"]
        extra_mask = torch.ones(
            B, len(extra_token_ids), dtype=mask.dtype, device=device
        )
        new_inputs["attention_mask"] = torch.cat([mask, extra_mask], dim=1)

    # 3. 图像侧张量: 原样保留（与 seq_len 无关）
    for k in ("pixel_values", "image_grid_thw", "image_sizes"):
        if k in inputs:
            new_inputs[k] = inputs[k]

    # 4. mm_token_type_ids: Qwen3-VL 多模态 RoPE 需要，必须与新 seq_len 对齐。
    # 追加的 response token 都是文本类型 0。
    if "mm_token_type_ids" in inputs:
        mm_ids = inputs["mm_token_type_ids"]
        extra_mm_ids = torch.zeros(
            B, len(extra_token_ids), dtype=mm_ids.dtype, device=device
        )
        new_inputs["mm_token_type_ids"] = torch.cat([mm_ids, extra_mm_ids], dim=1)

    # 不透传 position_ids / rope_deltas / cache_position / token_type_ids 等
    # → 让模型前向根据新 input_ids 长度自动计算

    assert new_inputs["input_ids"].shape[1] == (
        new_inputs.get("attention_mask", new_inputs["input_ids"]).shape[1]
    ), "input_ids 和 attention_mask 长度不一致"

    return new_inputs


def _append_answer_tokens_to_inputs(
    inputs: Dict[str, torch.Tensor],
    answer_token_ids: List[int],
    device: str,
) -> Dict[str, torch.Tensor]:
    """向后兼容包装：CEDComputer.compute() 的 answer_span 路径调用此函数。
    内部转调 extend_multimodal_inputs()。
    """
    return extend_multimodal_inputs(inputs, answer_token_ids, device)


# ─────── CED 计算器 ───────

class CEDComputer:
    def __init__(
        self,
        model,
        processor,
        layers: List[int],
        device: str = "cuda:0",
        replace_mode: str = "zero",
        audit_mode: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.layers = list(layers)
        self.audit_mode = audit_mode

        self.rep = TokenReplacer()
        self.rep.mode = replace_mode
        self.rep.audit_mode = audit_mode
        self.rep.register(model)

        self.cap1 = HiddenCapture()
        self.cap1.register(model, self.layers)

        self.cap2 = HiddenCapture()
        self.cap2.register(model, self.layers)

        # [FIX-5] 预计算 yes/no token 候选
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self._yes_ids, self._no_ids = _collect_yes_no_token_ids(tok)

    def cleanup(self):
        self.rep.remove()
        self.cap1.remove()
        self.cap2.remove()

    @torch.no_grad()
    def compute(
        self,
        inputs: Dict[str, torch.Tensor],
        tgt_abs: List[int],
        sur_abs: List[int],
        lambdas=(0.0, 0.1, 0.2),
        key_positions: Optional[List[int]] = None,
        key_mode: str = "prompt_last",
        answer_text: str = "",
        span_len: int = 3,
        span_reduce: str = "mean",
    ) -> Dict[str, Any]:
        """计算 logits 以及若干层 hidden 的 CED 指标。

        新增参数（v3）：
          key_mode: "prompt_last" | "answer_first" | "answer_span"
              prompt_last  = 原实现，只看 prompt 最后一个 token
              answer_first = teacher-force，观测第一个答案 token 位置的 logits
              answer_span  = teacher-force N 个答案 token，聚合
          answer_text: 答案文本（answer_first/answer_span 需要）
          span_len: answer_span 的 token 数（默认 3）
          span_reduce: answer_span 聚合方式 "mean" | "max"
        """
        if not tgt_abs:
            return {"error": "no_target_tokens"}

        # ── 计算观测位置 ──
        input_ids = inputs["input_ids"]
        T_orig = input_ids.shape[-1]
        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        key_meta: Dict[str, Any] = {"key_mode": key_mode}
        appended_ids: List[int] = []
        fallback_flags: List[str] = []
        fallback_reasons: List[str] = []

        # Audit: track multimodal input field presence
        has_position_ids = "position_ids" in inputs
        has_rope_deltas = "rope_deltas" in inputs
        has_cache_position = "cache_position" in inputs

        if key_positions is not None:
            # 向后兼容：直接传 key_positions
            computed_positions = list(key_positions)
            key_meta["key_mode"] = "explicit"
        elif key_mode == "prompt_last":
            computed_positions = [T_orig - 1]
        elif key_mode in ("answer_first", "answer_span"):
            if not answer_text:
                key_meta["key_mode"] = "prompt_last"
                key_meta["key_mode_fallback"] = "no_answer_text"
                fallback_flags.append("key_mode_fallback")
                fallback_reasons.append("no_answer_text")
                computed_positions = [T_orig - 1]
            else:
                answer_ids = tok.encode(answer_text, add_special_tokens=False)
                if not answer_ids:
                    key_meta["key_mode"] = "prompt_last"
                    key_meta["key_mode_fallback"] = "empty_answer_encoding"
                    fallback_flags.append("key_mode_fallback")
                    fallback_reasons.append("empty_answer_encoding")
                    computed_positions = [T_orig - 1]
                else:
                    if key_mode == "answer_first":
                        n_use = 1
                    else:
                        n_use = min(span_len, len(answer_ids))
                    key_meta["n_answer_tokens_used"] = n_use
                    key_meta["span_reduce"] = span_reduce if key_mode == "answer_span" else "single"
                    appended_ids = answer_ids[:n_use]
                    computed_positions = list(range(T_orig, T_orig + n_use))
        elif key_mode == "auto":
            # auto 应由上层 (verifier_reward) 解析后传入具体模式
            # 此处做兜底：根据 answer_text 长度选择
            if not answer_text:
                key_meta["key_mode"] = "prompt_last"
                computed_positions = [T_orig - 1]
            else:
                answer_ids = tok.encode(answer_text, add_special_tokens=False)
                if not answer_ids:
                    key_meta["key_mode"] = "prompt_last"
                    computed_positions = [T_orig - 1]
                elif len(answer_ids) == 1:
                    key_meta["key_mode"] = "answer_first"
                    key_meta["n_answer_tokens_used"] = 1
                    key_meta["span_reduce"] = "single"
                    appended_ids = answer_ids[:1]
                    computed_positions = [T_orig]
                else:
                    n_use = min(span_len, len(answer_ids))
                    key_meta["key_mode"] = "answer_span"
                    key_meta["n_answer_tokens_used"] = n_use
                    key_meta["span_reduce"] = span_reduce
                    appended_ids = answer_ids[:n_use]
                    computed_positions = list(range(T_orig, T_orig + n_use))
        else:
            raise ValueError(f"Unknown key_mode: {key_mode}")

        # 如果需要拼接 answer token
        if appended_ids:
            run_inputs = _append_answer_tokens_to_inputs(inputs, appended_ids, self.device)
        else:
            run_inputs = inputs

        # Pass 1: 原始
        self.cap1.clear()
        self.rep.active = False
        o1 = self.model(**run_inputs)
        logits1 = o1.logits.float()
        hs1 = {k: v.detach().clone() for k, v in self.cap1.data.items()}

        # Pass 2: 替换
        self.cap2.clear()
        self.rep.set(tgt_abs, sur_abs)
        with replacing(self.rep):
            o2 = self.model(**run_inputs)
        logits2 = o2.logits.float()
        hs2 = {k: v.detach().clone() for k, v in self.cap2.data.items()}

        B, T, V = logits1.shape

        t_list: List[int] = []
        for t in computed_positions:
            tt = t if t >= 0 else (T + t)
            if 0 <= tt < T:
                t_list.append(tt)
        if not t_list:
            return {"error": "no_valid_key_positions"}

        r: Dict[str, Any] = {"tkey_count": float(len(t_list))}
        r.update(key_meta)
        # 始终暴露 computed_positions（非仅 audit 模式）
        r["computed_positions"] = t_list

        # ── 逐位置计算 ──
        per_js, per_kl, per_cos = [], [], []
        per_h1, per_h2, per_penalty = [], [], []
        temp_per: Dict[float, List[float]] = {tau: [] for tau in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]}
        topks = [10, 50, 100, 500]
        topk_js_per: Dict[int, List[float]] = {k: [] for k in topks}
        topk_cos_per: Dict[int, List[float]] = {k: [] for k in topks}
        prob_cos_per: List[float] = []

        for tt in t_list:
            lg1 = logits1[:, tt, :]
            lg2 = logits2[:, tt, :]
            p1 = F.softmax(lg1, dim=-1)
            p2 = F.softmax(lg2, dim=-1)

            per_js.append(js_div(p1, p2).mean().item())
            per_h1.append(ent(p1).mean().item())
            per_h2.append(ent(p2).mean().item())
            per_kl.append(kl_div(p1, p2).mean().item())
            per_cos.append(cos_dist(lg1, lg2).mean().item())
            per_penalty.append(max(0.0, per_h2[-1] - per_h1[-1]))

            for tau in temp_per:
                pt1 = F.softmax(lg1 / tau, dim=-1)
                pt2 = F.softmax(lg2 / tau, dim=-1)
                temp_per[tau].append(js_div(pt1, pt2).mean().item())

            for k in topks:
                if k >= lg1.shape[-1]:
                    continue
                _, topk_idx = lg1.topk(k, dim=-1)
                p1_k = p1.gather(-1, topk_idx)
                p2_k = p2.gather(-1, topk_idx)
                p1_k = p1_k / p1_k.sum(-1, keepdim=True).clamp_min(1e-12)
                p2_k = p2_k / p2_k.sum(-1, keepdim=True).clamp_min(1e-12)
                topk_js_per[k].append(js_div(p1_k, p2_k).mean().item())

            prob_cos_per.append(cos_dist(p1, p2).mean().item())

            for k in topks:
                if k >= lg1.shape[-1]:
                    continue
                _, topk_idx = lg1.topk(k, dim=-1)
                l1_k = lg1.gather(-1, topk_idx)
                l2_k = lg2.gather(-1, topk_idx)
                topk_cos_per[k].append(cos_dist(l1_k, l2_k).mean().item())

        # ── 聚合 ──
        reduce = key_meta.get("span_reduce", "single")
        n_pos = len(t_list)

        def _agg(vals):
            if not vals:
                return 0.0
            if reduce == "max" and n_pos > 1:
                return max(vals)
            elif reduce == "mean" and n_pos > 1:
                return sum(vals) / len(vals)
            else:
                return sum(vals)  # single 或只有一个位置

        r["logits_js"] = _agg(per_js)
        r["logits_entropy_orig"] = _agg(per_h1)
        r["logits_entropy_replaced"] = _agg(per_h2)
        r["logits_entropy_delta"] = r["logits_entropy_replaced"] - r["logits_entropy_orig"]
        r["logits_kl"] = _agg(per_kl)
        r["logits_cosine_dist"] = _agg(per_cos)
        r["entropy_penalty_only"] = _agg(per_penalty)

        for lam in lambdas:
            r[f"ced_lambda_{lam:.2f}"] = r["logits_js"] + lam * r["entropy_penalty_only"]

        for tau, vals in temp_per.items():
            r[f"js_temp_{tau:.2f}"] = _agg(vals)

        for k in topks:
            if topk_js_per[k]:
                r[f"js_topk_{k}"] = _agg(topk_js_per[k])

        r["prob_cosine_dist"] = _agg(prob_cos_per)

        for k in topks:
            if topk_cos_per[k]:
                r[f"cos_topk_{k}"] = _agg(topk_cos_per[k])

        # ── [FIX-5] Log-Odds ATE（v5: 支持 span 聚合） ──
        if self._yes_ids and self._no_ids:
            per_logodds_ate: List[float] = []
            per_prob_yes_delta: List[float] = []

            for tt in t_list:
                lg1_t = logits1[:, tt, :]
                lg2_t = logits2[:, tt, :]

                ly1 = _logsumexp_logits(lg1_t, self._yes_ids)
                ln1 = _logsumexp_logits(lg1_t, self._no_ids)
                ly2 = _logsumexp_logits(lg2_t, self._yes_ids)
                ln2 = _logsumexp_logits(lg2_t, self._no_ids)

                lo_orig = (ly1 - ln1).mean().item()
                lo_repl = (ly2 - ln2).mean().item()
                per_logodds_ate.append(abs(lo_orig - lo_repl))

                py1 = F.softmax(lg1_t, dim=-1)[:, self._yes_ids].sum(dim=-1)
                py2 = F.softmax(lg2_t, dim=-1)[:, self._yes_ids].sum(dim=-1)
                per_prob_yes_delta.append((py1 - py2).abs().mean().item())

            r["logodds_ate"] = _agg(per_logodds_ate)
            r["prob_yes_delta"] = _agg(per_prob_yes_delta)
            # 保留最后位置的值作对照（兼容旧分析）
            r["logodds_ate_last"] = per_logodds_ate[-1]
            r["prob_yes_delta_last"] = per_prob_yes_delta[-1]
            r["n_yes_candidates"] = len(self._yes_ids)
            r["n_no_candidates"] = len(self._no_ids)

            # 最后位置的详细 logit 值（兼容旧字段）
            tt_last = t_list[-1]
            lg1_last = logits1[:, tt_last, :]
            lg2_last = logits2[:, tt_last, :]
            ly1 = _logsumexp_logits(lg1_last, self._yes_ids)
            ln1 = _logsumexp_logits(lg1_last, self._no_ids)
            ly2 = _logsumexp_logits(lg2_last, self._yes_ids)
            ln2 = _logsumexp_logits(lg2_last, self._no_ids)
            r["logit_yes_orig"] = ly1.mean().item()
            r["logit_no_orig"] = ln1.mean().item()
            r["logit_yes_replaced"] = ly2.mean().item()
            r["logit_no_replaced"] = ln2.mean().item()
            r["logodds_orig"] = (ly1 - ln1).mean().item()
            r["logodds_replaced"] = (ly2 - ln2).mean().item()

        # ── 中间层度量 ──
        tgt_abs_t = torch.as_tensor(tgt_abs, device=logits1.device, dtype=torch.long)
        tgt_abs_t = tgt_abs_t[(tgt_abs_t >= 0) & (tgt_abs_t < T)]

        # [FIX-LAYER-POS] 明确区分两个 "last" 位置：
        #   prompt_last_pos: 原 prompt 的最后一个位置（T_orig - 1）
        #   seq_last_pos: 完整序列（含 append 的 answer tokens）的最后位置（T - 1）
        # 当没有 append 时二者相同；有 append 时不同。
        prompt_last_pos = min(T_orig - 1, T - 1)
        seq_last_pos = T - 1

        for i in self.layers:
            if i not in hs1 or i not in hs2:
                continue
            h1 = hs1[i].float()
            h2 = hs2[i].float()

            if tgt_abs_t.numel() > 0:
                a_tgt = h1[:, tgt_abs_t, :].mean(dim=1)
                b_tgt = h2[:, tgt_abs_t, :].mean(dim=1)
                r[f"layer_{i}_tgt_cosine"] = cos_dist(a_tgt, b_tgt).mean().item()
                r[f"layer_{i}_tgt_l2"] = (a_tgt - b_tgt).norm(dim=-1).mean().item()

            # prompt_last: 原 prompt 最后一个 token 位置（与旧 layer_X_last_* 语义一致）
            a_prompt_last = h1[:, prompt_last_pos, :]
            b_prompt_last = h2[:, prompt_last_pos, :]
            r[f"layer_{i}_prompt_last_cosine"] = cos_dist(a_prompt_last, b_prompt_last).mean().item()
            r[f"layer_{i}_prompt_last_l2"] = (a_prompt_last - b_prompt_last).norm(dim=-1).mean().item()

            # seq_last: 完整序列最后一个 token 位置（append 后 answer 的最后位置）
            a_seq_last = h1[:, seq_last_pos, :]
            b_seq_last = h2[:, seq_last_pos, :]
            r[f"layer_{i}_seq_last_cosine"] = cos_dist(a_seq_last, b_seq_last).mean().item()
            r[f"layer_{i}_seq_last_l2"] = (a_seq_last - b_seq_last).norm(dim=-1).mean().item()

            # Backward compat: old layer_X_last_* aliases → prompt_last
            r[f"layer_{i}_last_cosine"] = r[f"layer_{i}_prompt_last_cosine"]
            r[f"layer_{i}_last_l2"] = r[f"layer_{i}_prompt_last_l2"]

            # ★ DEPRECATED: layer_X_js 实际是 cosine distance，不是 JS 散度 ★
            # 保留仅为兼容旧文件，新代码请用 layer_X_tgt_cosine
            r[f"layer_{i}_js"] = r.get(f"layer_{i}_tgt_cosine", 0.0)
            r[f"layer_{i}_cosine"] = r.get(f"layer_{i}_tgt_cosine", 0.0)

        # ── Audit 数据收集 ──
        if self.audit_mode:
            r["_audit"] = {
                "resolved_answer_text": answer_text,
                "resolved_answer_token_ids": appended_ids,
                "resolved_answer_token_count": len(appended_ids),
                "requested_key_mode": key_mode,
                "effective_key_mode": key_meta.get("key_mode", key_mode),
                "computed_positions": t_list,
                "appended_token_ids": appended_ids,
                "appended_token_count": len(appended_ids),
                "span_len": span_len if key_mode == "answer_span" else None,
                "actual_layers_tracked": list(self.layers),
                "requested_replace_mode": self.rep.mode,
                "actual_replace_mode": self.rep.mode,
                "T_orig": T_orig,
                "T_final": T,
                "prompt_last_pos": prompt_last_pos,
                "seq_last_pos": seq_last_pos,
                "target_positions_for_tgt_metrics": tgt_abs,
                "has_position_ids": has_position_ids,
                "has_rope_deltas": has_rope_deltas,
                "has_cache_position": has_cache_position,
                "fallback_flags": fallback_flags,
                "fallback_reasons": fallback_reasons,
                "replace_audit": dict(self.rep._last_audit) if self.rep._last_audit else None,
            }

        return r

    @torch.no_grad()
    def generate(self, inputs: Dict[str, torch.Tensor], max_new: int = 32) -> str:
        """给定 prepare_inputs 的结果，生成回答文本。"""
        self.rep.active = False
        from model_loader import generation_model
        g = generation_model(self.model).generate(**inputs, max_new_tokens=max_new)
        out = self.processor.batch_decode(
            g[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return out.strip()


# ─────── 工具函数 ───────

def expand_mm_inputs(inputs: dict, n: int) -> dict:
    """Repeat a single-example multimodal batch n times for batched generate.

    Qwen2.5-VL / Qwen3-VL pack visual tensors as concatenated patches
    (`pixel_values` / `image_grid_thw`), not a padded NCHW batch. Text
    tensors repeat on dim 0; visual tensors concatenate on dim 0.
    """
    if n <= 1:
        return inputs
    out = {}
    text_keys = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if key in text_keys or (value.dim() >= 2 and key.startswith("input")):
            out[key] = value.repeat(n, *([1] * (value.dim() - 1)))
        else:
            out[key] = torch.cat([value] * n, dim=0)
    return out


def prepare_inputs(processor, image, question: str, device: str):
    """封装 Qwen3-VL 的 chat 模板，返回送入 model 的张量 dict。"""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    # Qwen2.5-VL 可能带无用的 BERT-style token_type_ids；Qwen3-VL 的
    # mm_token_type_ids 必须保留，CED teacher-force 和 generate 都要它。
    inputs.pop("token_type_ids", None)
    return {k: v.to(device) for k, v in inputs.items()}


def get_image_token_id(processor) -> int:
    """推断视觉 token 在 input_ids 里的占位 token id。"""
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    for name in ["<|image_pad|>", "<|vision_pad|>", "<image>"]:
        try:
            tid = tok.convert_tokens_to_ids(name)
        except Exception:
            continue
        if tid != getattr(tok, "unk_token_id", -1):
            return tid
    assert hasattr(tok, "image_token_id"), "找不到 image token id"
    return tok.image_token_id
