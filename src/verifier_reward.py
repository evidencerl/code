"""Verifier reward computation.

This module now explicitly separates:
1) candidate-level rewards (valid rerank signals)
2) sample-level probes (diagnostics only, must NOT be used for rerank)

Compatibility:
- Legacy flat fields are still written (reward_logits_js, reward_mix_*, ...)
- New schema fields are added for explicit downstream filtering.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model_loader import load, num_layers
from ced_core import (
    CEDComputer, prepare_inputs, get_image_token_id,
)
from visual_token_map import (
    compute_merge_ratio,
    bbox_to_token_indices,
    surrounding_indices,
    find_visual_range,
    to_absolute,
    reset_surrounding_meta,
)
from yesno_utils import parse_yesno
from reward.action_logprob_ate import ActionLogProbATEReward
from reward.schema import (
    REWARD_SCHEMA_VERSION,
    get_all_reward_fields,
    get_candidate_reward_fields,
    get_sample_probe_fields,
    get_rerank_eligible_reward_fields,
    get_verifier_eligible_reward_fields,
    get_grpo_eligible_reward_fields,
    get_schema_summary,
    infer_active_reward_fields_from_records,
)
from dataset_adapters import open_sample_image, resolve_image_abspath


# ────────────────────────────────────────
# key_mode 自动解析
# ────────────────────────────────────────

ALL_KEY_MODES = ["prompt_last", "answer_first", "answer_span", "auto"]


def resolve_key_mode(
    requested: str,
    answer_text: str,
    processor,
    max_span_tokens: int = 4,
) -> dict:
    """将 requested key_mode 解析为 effective key_mode + 诊断信息。

    auto 规则：
      0 个答案 token → prompt_last
      1 个答案 token → answer_first
      ≥2 个答案 token → answer_span（最多取前 max_span_tokens 个）
    """
    tok = getattr(processor, "tokenizer", processor)

    answer_ids = []
    if answer_text:
        answer_ids = tok.encode(answer_text, add_special_tokens=False)

    n_answer_tokens = len(answer_ids)

    if requested != "auto":
        effective = requested
    elif n_answer_tokens == 0:
        effective = "prompt_last"
    elif n_answer_tokens == 1:
        effective = "answer_first"
    else:
        effective = "answer_span"

    # answer_span 用到的实际 span_len
    span_len = min(max_span_tokens, n_answer_tokens) if effective == "answer_span" else (
        1 if effective == "answer_first" else 0
    )

    return {
        "requested_key_mode": requested,
        "effective_key_mode": effective,
        "answer_token_len": n_answer_tokens,
        "span_len_used": span_len,
    }


# ────────────────────────────────────────
# Reward 定义：固定映射
# ────────────────────────────────────────

# Backward-compatible name: all flat reward-ish outputs written in candidate_scores.jsonl.
REWARD_CANDIDATES = get_all_reward_fields()
RERANK_ELIGIBLE_REWARDS = get_rerank_eligible_reward_fields(include_future=False)
VERIFIER_ELIGIBLE_REWARDS = get_verifier_eligible_reward_fields(include_future=False)
GRPO_ELIGIBLE_REWARDS = get_grpo_eligible_reward_fields(include_future=False)
SAMPLE_PROBE_FIELDS = get_sample_probe_fields()
CANDIDATE_REWARD_FIELDS = get_candidate_reward_fields(include_future=False)

# CED 输出 key -> reward name 的映射
_CED_TO_REWARD = {
    "logits_js": "reward_logits_js",
    "logits_cosine_dist": "reward_logits_cosine_dist",
    "layer_24_prompt_last_cosine": "reward_layer24_prompt_last_cosine",
}


def compute_rewards_from_ced(
    ced_metrics: Dict[str, Any],
    mix_alpha: float = 0.5,
    mix_beta: float = 0.5,
) -> Dict[str, Any]:
    """Extract reward fields from CED outputs with explicit schema split.

    Returns keys:
      - candidate_rewards: candidate-level rewards for rerank
      - sample_probes: sample-level diagnostics (may be constant within sample)
      - legacy_flat_rewards: flattened legacy fields for backward compatibility
    """
    rewards: Dict[str, float] = {}

    for ced_key, reward_name in _CED_TO_REWARD.items():
        val = ced_metrics.get(ced_key)
        rewards[reward_name] = float(val) if val is not None else float("nan")

    # 组合分数（保留写出，但会在 schema 中标记为 non-rerank）
    js = rewards.get("reward_logits_js", float("nan"))
    cos = rewards.get("reward_logits_cosine_dist", float("nan"))
    l24 = rewards.get("reward_layer24_prompt_last_cosine", float("nan"))

    if not (math.isnan(js) or math.isnan(l24)):
        rewards["reward_mix_js_promptlast"] = mix_alpha * js + mix_beta * l24
    else:
        rewards["reward_mix_js_promptlast"] = float("nan")

    if not (math.isnan(cos) or math.isnan(l24)):
        rewards["reward_mix_cosine_promptlast"] = mix_alpha * cos + mix_beta * l24
    else:
        rewards["reward_mix_cosine_promptlast"] = float("nan")

    candidate_rewards = {
        k: rewards.get(k, float("nan"))
        for k in CANDIDATE_REWARD_FIELDS
        if k in rewards
    }
    sample_probes = {
        k: rewards.get(k, float("nan"))
        for k in SAMPLE_PROBE_FIELDS
        if k in rewards
    }

    return {
        "candidate_rewards": candidate_rewards,
        "sample_probes": sample_probes,
        "legacy_flat_rewards": rewards,
    }


def _finite_or_nan(v: Any) -> float:
    try:
        fv = float(v)
    except Exception:
        return float("nan")
    if math.isnan(fv) or math.isinf(fv):
        return float("nan")
    return fv


# ────────────────────────────────────────
# 对候选答案计算 CED reward
# ────────────────────────────────────────

LEGACY_CED_DEFAULT_FIELDS = (
    "reward_logits_js",
    "reward_logits_cosine_dist",
    "reward_layer24_prompt_last_cosine",
    "reward_mix_js_promptlast",
    "reward_mix_cosine_promptlast",
)

ACTION_MAIN_DEFAULT_FIELDS = (
    "reward_action_logprob_ate",
    "reward_action_logprob_ate_supervised",
    "reward_action_logprob_ate_nolabel",
    "reward_main_correctness_only",
    "reward_main_additive_evidence",
    "reward_main_routed_gated_evidence",
)

@torch.no_grad()
def score_candidate(
    ced: CEDComputer,
    processor,
    img_tid: int,
    image: Image.Image,
    question: str,
    candidate_text: str,
    target_bbox: list,
    image_width: int,
    image_height: int,
    device: str,
    replace_mode: str,
    key_mode: str = "auto",
    lambdas: tuple = (0.0,),
    sur_ring: int = 2,
    mix_alpha: float = 0.5,
    mix_beta: float = 0.5,
    max_span_tokens: int = 4,
) -> Dict[str, Any]:
    """对单个候选答案计算所有固定 reward candidates。

    key_mode="auto" 会根据 candidate_text 的 token 长度自动选择：
      0 token → prompt_last
      1 token → answer_first
      ≥2 token → answer_span (最多 max_span_tokens)
    """
    # 解析 key_mode
    km_info = resolve_key_mode(
        key_mode, candidate_text, processor, max_span_tokens=max_span_tokens)
    effective_km = km_info["effective_key_mode"]
    span_len = km_info["span_len_used"] or 3

    inputs = prepare_inputs(processor, image, question, device)

    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        raise RuntimeError("image_grid_thw not found")
    g = grid_thw[0].tolist()
    gt_val, gh, gw = map(int, g)

    vs, ve = find_visual_range(inputs["input_ids"], img_tid)
    n_vis = ve - vs
    merge_ratio = compute_merge_ratio(gt_val * gh * gw, n_vis)

    tgt_rel = bbox_to_token_indices(target_bbox, image_width, image_height, gh, gw, merge_ratio)
    if not tgt_rel:
        raise RuntimeError("empty target token set")
    reset_surrounding_meta()
    sur_rel = surrounding_indices(tgt_rel, n_vis, gh, gw, merge_ratio, ring=sur_ring)
    if not sur_rel:
        raise RuntimeError("empty surround token set")
    tgt_abs = to_absolute(inputs["input_ids"], tgt_rel, img_tid)
    sur_abs = to_absolute(inputs["input_ids"], sur_rel, img_tid)

    ced_metrics = ced.compute(
        inputs, tgt_abs, sur_abs,
        lambdas=lambdas,
        key_mode=effective_km,
        answer_text=candidate_text,
        span_len=span_len,
        span_reduce="mean",
    )

    reward_pack = compute_rewards_from_ced(ced_metrics, mix_alpha, mix_beta)
    rewards = reward_pack["legacy_flat_rewards"]

    # 提取 ced_core 返回的真实 computed_positions
    computed_positions = ced_metrics.get("computed_positions",
                         ced_metrics.get("_audit", {}).get("computed_positions", []))

    return {
        **rewards,
        "candidate_rewards": reward_pack["candidate_rewards"],
        "sample_probes": reward_pack["sample_probes"],
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "rerank_eligible_reward_fields": RERANK_ELIGIBLE_REWARDS,
        "candidate_reward_fields": CANDIDATE_REWARD_FIELDS,
        "sample_probe_fields": SAMPLE_PROBE_FIELDS,
        "n_tgt_tokens": len(tgt_abs),
        "n_vis_tokens": n_vis,
        "replace_mode_used": replace_mode,
        # 新增诊断字段
        "requested_key_mode": km_info["requested_key_mode"],
        "effective_key_mode": km_info["effective_key_mode"],
        "answer_token_len": km_info["answer_token_len"],
        "computed_positions": computed_positions if computed_positions else [],
    }


@torch.no_grad()
def score_candidate_action_reward(
    action_reward: ActionLogProbATEReward,
    image: Image.Image,
    question: str,
    candidate_response_text: str,
    candidate_answer_text: str,
    task_type: str,
    target_bbox: list,
    image_width: int,
    image_height: int,
    gt_answer: Optional[str],
    gt_present: Optional[bool],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute ActionLogProbATEReward and map its main output into flat field.

    CRITICAL:
    - response_text MUST come from candidate's raw response (raw_generated_text if present)
    - parsing/correctness MUST come from extracted final answer (generated_text)
    - flat field name MUST be reward_action_logprob_ate
    """
    # Only enable yes/no margin shaping for actual yes/no tasks.
    # Why: counting/spatial/attribute would otherwise be forced into "parsed=other" buckets.
    # Here we treat "existence" and explicit yes/no ground-truth as yes/no tasks.
    is_yesno_task = (task_type == "existence") or (parse_yesno(gt_answer or "") != "other")

    r = action_reward.compute(
        image=image,
        question=question,
        target_bbox=target_bbox,
        response_text=candidate_response_text,
        answer_text=candidate_answer_text,
        enable_yesno_margin=bool(is_yesno_task),
        gt_answer=gt_answer,
        gt_present=gt_present if is_yesno_task else None,
        image_width=image_width,
        image_height=image_height,
        response_text_source="raw_generated_text",
        answer_text_source="generated_text",
        metadata=metadata,
    )

    return {
        # Backward-compatible alias: keep reward_action_logprob_ate but make semantics explicit.
        "reward_action_logprob_ate": _finite_or_nan(r.get("supervised_shaped_reward", r.get("reward"))),
        "reward_action_logprob_ate_supervised": _finite_or_nan(r.get("supervised_shaped_reward")),
        "reward_action_logprob_ate_nolabel": _finite_or_nan(r.get("score_base_no_label")),
        "reward_main_correctness_only": _finite_or_nan(r.get("reward_main_correctness_only")),
        "reward_main_additive_evidence": _finite_or_nan(r.get("reward_main_additive_evidence")),
        "reward_main_routed_gated_evidence": _finite_or_nan(r.get("reward_main_routed_gated_evidence")),
        # Keep a small set of debug fields (do NOT affect rerank filtering).
        "action_logprob_ate_reward_mode": r.get("reward_mode"),
        "action_logprob_ate_main_reward_mode_requested": r.get("main_reward_mode_requested"),
        "action_logprob_ate_main_reward_mode_effective": r.get("main_reward_mode_effective"),
        "action_logprob_ate_main_reward_active": _finite_or_nan(r.get("main_reward_active")),
        "action_logprob_ate_parsed_answer": r.get("parsed_answer"),
        "action_logprob_ate_is_correct": r.get("is_correct"),
        "action_logprob_ate_task_family": r.get("task_family"),
        "action_logprob_ate_correctness_score": _finite_or_nan(r.get("correctness_score")),
        "action_logprob_ate_correctness_reward": _finite_or_nan(r.get("correctness_reward")),
        "action_logprob_ate_delta_logprob": _finite_or_nan(r.get("delta_logprob")),
        "action_logprob_ate_s_pos": _finite_or_nan(r.get("s_pos")),
        "action_logprob_ate_s_neg_mean": _finite_or_nan(r.get("s_neg_mean")),
        "action_logprob_ate_relative_evidence_margin": _finite_or_nan(r.get("relative_evidence_margin")),
        "action_logprob_ate_delta_ans_margin": _finite_or_nan(r.get("delta_ans_margin")),
        "action_logprob_ate_score_base": _finite_or_nan(r.get("score_base")),
        "action_logprob_ate_score_base_no_label": _finite_or_nan(r.get("score_base_no_label")),
        "action_logprob_ate_supervised_shaped_reward": _finite_or_nan(r.get("supervised_shaped_reward")),
        "action_logprob_ate_supervised_reward_bias": _finite_or_nan(r.get("supervised_reward_bias")),
        "action_logprob_ate_supervised_reward_scale": _finite_or_nan(r.get("supervised_reward_scale")),
        "action_logprob_ate_main_reward_evidence_gate": _finite_or_nan(r.get("main_reward_evidence_gate")),
        "action_logprob_ate_main_reward_evidence_allowed": r.get("main_reward_evidence_allowed"),
        "action_logprob_ate_n_negative_interventions": r.get("n_negative_interventions"),
        "action_logprob_ate_extra_forward_passes": r.get("extra_forward_passes"),
        "action_logprob_ate_reward_compute_time_sec": _finite_or_nan(r.get("reward_compute_time_sec")),
        "action_logprob_ate_reward_cache_reuse_enabled": r.get("reward_cache_reuse_enabled"),
        "action_logprob_ate_proposal_source": r.get("proposal_source"),
        "action_logprob_ate_proposal_bbox_missing": r.get("proposal_bbox_missing"),
        "action_logprob_ate_proposal_fallback_used": r.get("proposal_fallback_used"),
        "action_logprob_ate_proposal_fallback_reason": r.get("proposal_fallback_reason"),
        "action_logprob_ate_proposal_area_fraction": _finite_or_nan(r.get("proposal_area_fraction")),
        "action_logprob_ate_proposal_token_count": r.get("proposal_token_count"),
        "action_logprob_ate_proposal_token_coverage_fraction": _finite_or_nan(r.get("proposal_token_coverage_fraction")),
        "action_logprob_ate_uses_ground_truth": bool(r.get("correctness_known", False)),
        "action_logprob_ate_label_conditioned": bool(r.get("correctness_known", False)),
        "reward_response_text_source": "raw_generated_text",
        "reward_answer_text_source": "generated_text",
        "parsing_text_source": "generated_text",
        "correctness_text_source": "generated_text",
        "action_logprob_ate_error": r.get("error"),
    }


def _backend_default_fields(reward_backend: str) -> List[str]:
    if reward_backend == "legacy_ced":
        return list(LEGACY_CED_DEFAULT_FIELDS)
    if reward_backend == "action_logprob_ate":
        return list(ACTION_MAIN_DEFAULT_FIELDS)
    if reward_backend == "both":
        return list(LEGACY_CED_DEFAULT_FIELDS) + list(ACTION_MAIN_DEFAULT_FIELDS)
    return []


def _apply_backend_error_defaults(rec: Dict[str, Any], reward_backend: str) -> None:
    """错误样本只补当前 backend 相关的核心字段，避免无关字段被 NaN 污染整行。"""
    for name in _backend_default_fields(reward_backend):
        rec.setdefault(name, 0.0)


def _core_field_status(rec: Dict[str, Any], reward_backend: str) -> str:
    """把 reward 记录状态显式化，避免下游只看到一堆 NaN。"""
    fields = _backend_default_fields(reward_backend)
    if rec.get("reward_error") or rec.get("action_logprob_ate_error"):
        return "error"
    if not fields:
        return "unknown_backend"
    finite = 0
    present = 0
    for name in fields:
        if name in rec:
            present += 1
            if not math.isnan(_finite_or_nan(rec.get(name))):
                finite += 1
    if present <= 0:
        return "not_computed"
    if finite == present:
        return "finite_ok"
    if finite > 0:
        return "partial_finite"
    return "all_nan"


# ────────────────────────────────────────
# 批量处理：读取 candidates.jsonl, 计算 reward
# ────────────────────────────────────────

def run_reward_scoring(
    candidates_file: str,
    model_dir: str,
    coco_image_dir: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    checkpoint_path: str = "",
    replace_mode: str = "zero",
    key_mode: str = "auto",
    mix_alpha: float = 0.5,
    mix_beta: float = 0.5,
    sur_ring: int = 2,
    reward_backend: str = "legacy_ced",
    # --- ActionLogProbATEReward params (only used when backend includes action_logprob_ate) ---
    reward_mode: str = "soft_shaping",
    max_response_tokens: int = 64,
    tau_resp: float = 0.20,
    tau_ans: float = 1.00,
    alpha_resp: float = 0.70,
    alpha_ans: float = 0.30,
    min_reward: float = -1.25,
    max_reward: float = 1.00,
    main_reward_mode: str = "routed_gated_evidence",
    negative_intervention_k: int = 1,
    evidence_eps: float = 0.10,
    output_path: str = "candidate_scores.jsonl",
) -> List[Dict]:
    """读取候选文件，为每个候选计算 reward，写入 output_path。"""
    assert reward_backend in ("legacy_ced", "action_logprob_ate", "both"), \
        f"Unknown reward_backend: {reward_backend}"

    processor, model, cfg = load(model_dir, device=device, dtype=dtype)
    checkpoint_info = {
        "checkpoint_loaded": False,
        "checkpoint_path": checkpoint_path,
    }
    if checkpoint_path:
        from mini_grpo_smoke import _setup_trainable

        payload = torch.load(checkpoint_path, map_location="cpu")
        extra = dict(payload.get("extra") or {})
        state = payload.get("trainable_state_dict") or {}
        model, _ = _setup_trainable(
            model,
            n_trainable_layers=int(extra.get("n_trainable_layers", 4)),
            use_lora=bool(extra.get("lora_applied", False)),
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()
        checkpoint_info = {
            "checkpoint_loaded": True,
            "checkpoint_path": checkpoint_path,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "checkpoint_extra": extra,
        }
    img_tid = get_image_token_id(processor)

    ced = None
    if reward_backend in ("legacy_ced", "both"):
        ced = CEDComputer(
            model, processor, layers=[24], device=device,
            replace_mode=replace_mode,
        )

    action_reward = None
    action_reward_config = None
    if reward_backend in ("action_logprob_ate", "both"):
        action_reward = ActionLogProbATEReward(
            model=model,
            processor=processor,
            device=device,
            replace_mode=replace_mode,
            reward_mode=reward_mode,
            max_response_tokens=max_response_tokens,
            tau_resp=tau_resp,
            tau_ans=tau_ans,
            alpha_resp=alpha_resp,
            alpha_ans=alpha_ans,
            min_reward=min_reward,
            max_reward=max_reward,
            main_reward_mode=main_reward_mode,
            negative_intervention_k=negative_intervention_k,
            evidence_eps=evidence_eps,
        )
        action_reward_config = {
            "reward_mode": reward_mode,
            "max_response_tokens": max_response_tokens,
            "tau_resp": tau_resp,
            "tau_ans": tau_ans,
            "alpha_resp": alpha_resp,
            "alpha_ans": alpha_ans,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "replace_mode": replace_mode,
            "main_reward_mode": main_reward_mode,
            "negative_intervention_k": negative_intervention_k,
            "evidence_eps": evidence_eps,
        }

    with open(candidates_file, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    # 过滤掉 error 记录
    candidates = [c for c in candidates if "error" not in c]
    print(f"[verifier_reward] scoring {len(candidates)} candidates "
          f"(replace={replace_mode}, key={key_mode})")

    # 图片缓存：避免重复加载
    _img_cache: Dict[str, Image.Image] = {}

    scored = []
    # Per-record schema payload: keep it small and purely config-oriented.
    # Run-active lists are written to the sidecar reward_schema.json after scoring.
    schema_base = {
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "backend_used": reward_backend,
        "replace_mode": replace_mode,
        "key_mode": key_mode,
        "mix_alpha": mix_alpha,
        "mix_beta": mix_beta,
        "sur_ring": sur_ring,
        "verifier_eligible_reward_fields": VERIFIER_ELIGIBLE_REWARDS,
        "grpo_eligible_reward_fields": GRPO_ELIGIBLE_REWARDS,
        "action_reward_config": action_reward_config,
        "checkpoint_info": checkpoint_info,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for cand in tqdm(candidates, desc="Scoring", ncols=80):
            rec = dict(cand)
            try:
                img_file = cand["image_file"]
                resolved_image_path = resolve_image_abspath(cand, image_root=coco_image_dir)
                if img_file not in _img_cache:
                    _img_cache[img_file] = open_sample_image(cand, image_root=coco_image_dir)
                    # 限制缓存大小
                    if len(_img_cache) > 200:
                        oldest = next(iter(_img_cache))
                        del _img_cache[oldest]

                image = _img_cache[img_file]
                iw, ih = image.size
                rec["resolved_image_path"] = resolved_image_path

                # Prefer explicit text fields so analysis never confuses reward inputs.
                rec.setdefault("candidate_answer_text", cand.get("generated_text", ""))
                if "raw_generated_text" in cand:
                    rec.setdefault("candidate_response_text", cand.get("raw_generated_text", ""))
                    rec.setdefault("reward_response_text_source", "raw_generated_text")
                else:
                    rec.setdefault("candidate_response_text", "")
                    rec.setdefault("reward_response_text_source", "missing_raw_generated_text")
                rec.setdefault("reward_answer_text_source", "generated_text")
                rec.setdefault("parsing_text_source", "generated_text")
                rec.setdefault("correctness_text_source", "generated_text")

                # --- legacy CED backend ---
                if ced is not None:
                    rewards = score_candidate(
                        ced, processor, img_tid, image,
                        cand["question"],
                        # CED 是 answer-conditioned key-mode，必须用最终答案文本。
                        cand.get("generated_text", ""),
                        cand.get("target_bbox"),
                        iw, ih, device,
                        replace_mode=replace_mode,
                        key_mode=key_mode,
                        lambdas=(0.0,),
                        sur_ring=sur_ring,
                        mix_alpha=mix_alpha,
                        mix_beta=mix_beta,
                    )
                    rec.update(rewards)
                    rec.setdefault("reward_response_text_source", "raw_generated_text")
                    rec.setdefault("reward_answer_text_source", "generated_text")
                    rec.setdefault("parsing_text_source", "generated_text")
                    rec.setdefault("correctness_text_source", "generated_text")

                # --- action_logprob_ate backend ---
                if action_reward is not None:
                    resp_text = cand.get("raw_generated_text")
                    ans_text = cand.get("generated_text", "")
                    if not isinstance(resp_text, str) or not resp_text.strip():
                        raise ValueError(
                            "Action reward contract violation: missing raw_generated_text; "
                            "generated_text(final_answer) cannot be used as response_text"
                        )
                    act = score_candidate_action_reward(
                        action_reward=action_reward,
                        image=image,
                        question=cand["question"],
                        candidate_response_text=resp_text,
                        candidate_answer_text=ans_text,
                        task_type=cand.get("task_type", ""),
                        target_bbox=cand.get("target_bbox"),
                        image_width=iw,
                        image_height=ih,
                        gt_answer=cand.get("gt_answer"),
                        gt_present=cand.get("gt_present"),
                        metadata=cand,
                    )
                    rec.update(act)

                rec.setdefault("reward_schema", schema_base)
            except Exception as e:
                rec["reward_error"] = repr(e)
                rec["resolved_image_path"] = resolve_image_abspath(cand, image_root=coco_image_dir)
                _apply_backend_error_defaults(rec, reward_backend)
                rec.setdefault("reward_schema", schema_base)

            rec["reward_backend_used"] = reward_backend
            rec["reward_record_status"] = _core_field_status(rec, reward_backend)

            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            scored.append(rec)

    if ced is not None:
        ced.cleanup()
    if action_reward is not None:
        action_reward.cleanup()

    # Sidecar schema for auditability / downstream tools.
    # IMPORTANT: must reflect *run-active* (finite) reward fields, not static registry.
    active_reward_fields = infer_active_reward_fields_from_records(scored, known_fields_only=True)
    schema_summary = get_schema_summary(
        include_future=False,
        available_fields=REWARD_CANDIDATES,
        active_fields=active_reward_fields,
    )
    schema_summary.update(schema_base)
    # Override: ensure the sidecar contains these explicit keys for downstream.
    schema_summary["available_reward_fields"] = active_reward_fields
    schema_summary["available_rerank_eligible_reward_fields"] = schema_summary.get(
        "rerank_eligible_reward_fields", []
    )
    schema_summary["available_verifier_eligible_reward_fields"] = schema_summary.get(
        "verifier_eligible_reward_fields", []
    )
    schema_summary["available_grpo_eligible_reward_fields"] = schema_summary.get(
        "grpo_eligible_reward_fields", []
    )

    out_dir = os.path.dirname(output_path) or "."
    base = os.path.basename(output_path)
    m = re.match(r"^candidate_scores_shard_(\d+)\.jsonl$", base)
    schema_filename = f"reward_schema_shard_{m.group(1)}.json" if m else "reward_schema.json"
    schema_path = os.path.join(out_dir, schema_filename)
    with open(schema_path, "w", encoding="utf-8") as sf:
        json.dump(schema_summary, sf, indent=2, ensure_ascii=False)

    n_ok = sum(1 for r in scored if "reward_error" not in r)
    core_fields = [f for f in ACTION_MAIN_DEFAULT_FIELDS if any(f in row for row in scored)]
    core_field_finite_rates = {}
    for field in core_fields:
        vals = [r.get(field) for r in scored if "reward_error" not in r and field in r]
        finite = sum(1 for v in vals if not math.isnan(_finite_or_nan(v)))
        core_field_finite_rates[field] = {
            "present": len(vals),
            "finite": finite,
            "finite_rate": (finite / max(1, len(vals))),
        }
    status_counter: Dict[str, int] = {}
    for row in scored:
        key = str(row.get("reward_record_status", "unknown"))
        status_counter[key] = status_counter.get(key, 0) + 1
    print(f"[verifier_reward] {n_ok}/{len(scored)} scored → {output_path}")
    if core_field_finite_rates:
        print(f"[verifier_reward] core_field_finite_rates={json.dumps(core_field_finite_rates, ensure_ascii=False)}")
    print(f"[verifier_reward] reward_record_status={json.dumps(status_counter, ensure_ascii=False)}")
    print(f"[verifier_reward] schema → {schema_path}")
    return scored


# ── 也支持读取现有 CED raw JSONL，直接提取 reward ──

def extract_rewards_from_raw(
    raw_file: str,
    mix_alpha: float = 0.5,
    mix_beta: float = 0.5,
) -> List[Dict]:
    """从已有 p0b_raw.jsonl 提取 reward 分数（不需要重新跑模型）。"""
    records = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rewards = compute_rewards_from_ced(rec, mix_alpha, mix_beta)
            rec.update(rewards)
            records.append(rec)
    return records


# ────────────────────────────────────────
# CLI
# ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Compute verifier rewards for candidates")
    ap.add_argument("--candidates_file", required=True,
                    help="candidates.jsonl from candidate_gen.py")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--coco_image_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--checkpoint_path", default="")
    ap.add_argument("--replace_mode", default="zero",
                    choices=["zero", "noise", "mean"])
    ap.add_argument("--key_mode", default="auto",
                    choices=["prompt_last", "answer_first", "answer_span", "auto"])
    ap.add_argument("--mix_alpha", type=float, default=0.5,
                    help="组合 reward 的 logits 权重")
    ap.add_argument("--mix_beta", type=float, default=0.5,
                    help="组合 reward 的 layer24 权重")
    ap.add_argument("--sur_ring", type=int, default=2)
    # Reward backend switch (default keeps legacy behavior).
    ap.add_argument("--reward_backend", default="legacy_ced",
                    choices=["legacy_ced", "action_logprob_ate", "both"],
                    help="Which reward backend(s) to compute")
    # ActionLogProbATEReward params (used when backend includes action_logprob_ate)
    ap.add_argument("--reward_mode", default="soft_shaping",
                    choices=["legacy_hard_gate", "raw_delta", "soft_shaping"])
    ap.add_argument("--max_response_tokens", type=int, default=64)
    ap.add_argument("--tau_resp", type=float, default=0.20)
    ap.add_argument("--tau_ans", type=float, default=1.00)
    ap.add_argument("--alpha_resp", type=float, default=0.70)
    ap.add_argument("--alpha_ans", type=float, default=0.30)
    ap.add_argument("--min_reward", type=float, default=-1.25)
    ap.add_argument("--max_reward", type=float, default=1.00)
    ap.add_argument("--main_reward_mode", default="routed_gated_evidence",
                    choices=["correctness_only", "additive_evidence", "routed_gated_evidence"])
    ap.add_argument("--negative_intervention_k", type=int, default=1)
    ap.add_argument("--evidence_eps", type=float, default=0.10)
    ap.add_argument("--output", default="results/verifier/candidate_scores.jsonl")
    # 多卡分片
    ap.add_argument("--shard_id", type=int, default=0,
                    help="当前分片 ID（0-based）")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="总分片数（>1 启用分片模式）")
    args = ap.parse_args()

    t0 = time.time()

    # 分片模式：只处理当前 shard 对应的候选
    if args.num_shards > 1:
        import math
        with open(args.candidates_file, "r", encoding="utf-8") as f:
            all_candidates = [json.loads(line) for line in f if line.strip()]
        all_candidates = [c for c in all_candidates if "error" not in c]
        total = len(all_candidates)
        shard_size = math.ceil(total / args.num_shards)
        start = args.shard_id * shard_size
        end = min(start + shard_size, total)
        shard_candidates = all_candidates[start:end]
        print(f"[verifier_reward] shard {args.shard_id}/{args.num_shards}: "
              f"{len(shard_candidates)}/{total} candidates (range [{start}, {end}))")

        # 写临时分片文件供 run_reward_scoring 读取
        shard_input = args.output + ".shard_input.jsonl"
        with open(shard_input, "w", encoding="utf-8") as f:
            for c in shard_candidates:
                f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")

        run_reward_scoring(
            candidates_file=shard_input,
            model_dir=args.model_dir,
            coco_image_dir=args.coco_image_dir,
            device=args.device,
            dtype=args.dtype,
            checkpoint_path=args.checkpoint_path,
            replace_mode=args.replace_mode,
            key_mode=args.key_mode,
            mix_alpha=args.mix_alpha,
            mix_beta=args.mix_beta,
            sur_ring=args.sur_ring,
            reward_backend=args.reward_backend,
            reward_mode=args.reward_mode,
            max_response_tokens=args.max_response_tokens,
            tau_resp=args.tau_resp,
            tau_ans=args.tau_ans,
            alpha_resp=args.alpha_resp,
            alpha_ans=args.alpha_ans,
            min_reward=args.min_reward,
            max_reward=args.max_reward,
            main_reward_mode=args.main_reward_mode,
            negative_intervention_k=args.negative_intervention_k,
            evidence_eps=args.evidence_eps,
            output_path=args.output,
        )
        # 清理临时文件
        try:
            os.remove(shard_input)
        except OSError:
            pass
    else:
        run_reward_scoring(
            candidates_file=args.candidates_file,
            model_dir=args.model_dir,
            coco_image_dir=args.coco_image_dir,
            device=args.device,
            dtype=args.dtype,
            checkpoint_path=args.checkpoint_path,
            replace_mode=args.replace_mode,
            key_mode=args.key_mode,
            mix_alpha=args.mix_alpha,
            mix_beta=args.mix_beta,
            sur_ring=args.sur_ring,
            reward_backend=args.reward_backend,
            reward_mode=args.reward_mode,
            max_response_tokens=args.max_response_tokens,
            tau_resp=args.tau_resp,
            tau_ans=args.tau_ans,
            alpha_resp=args.alpha_resp,
            alpha_ans=args.alpha_ans,
            min_reward=args.min_reward,
            max_reward=args.max_reward,
            main_reward_mode=args.main_reward_mode,
            negative_intervention_k=args.negative_intervention_k,
            evidence_eps=args.evidence_eps,
            output_path=args.output,
        )

    print(f"[verifier_reward] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
