"""多候选答案生成模块。

对每个样本用 sampling 生成 K 个候选答案。

本次修复的核心目标：缓解 candidate collapse，让 candidate-level rerank 至少“有空间可排”。

兼容性原则：
- 旧字段（generated_text/mean_logprob/sum_logprob/answer_length/behavior/correct...）保留
- 新增字段只做增量，不破坏旧 JSONL/CSV 的消费逻辑

关键新增：
1) task-aware generation prompt：Evidence + Final answer 结构化输出
2) Final answer 抽取：generated_text 用抽取后的最终答案；raw_generated_text 保留全输出
3) answer-only logprob：避免 Evidence 长度污染 logprob baseline
4) 分轮采样 + 去重聚合：dedup 默认按 raw_text，verifier 默认 variable-K（不做重复补齐）
5) min_unique_candidates 为硬约束：未达阈值会持续重采样直到达到上限轮次
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model_loader import load
from ced_core import prepare_inputs
from yesno_utils import parse_yesno
from answer_format_utils import (
    PROMPT_MODES,
    build_generation_prompt,
    extract_final_answer,
    infer_task_family,
    normalize_raw_text_for_dedup,
    normalize_final_answer_for_dedup,
    normalize_text_for_analysis,
)
from evaluator_utils import classify_open_ended_correctness, semantically_match_attribute
from dataset_adapters import open_sample_image, resolve_image_abspath


def _decode_with_token_char_spans(tok, token_ids: List[int]) -> (str, List[tuple]):
    """Decode token_ids and return (text, per_token_char_spans).

    We need this to estimate answer-only logprob using generate() scores without extra forward passes.
    The decoding is incremental so each token's appended substring span is tracked.
    """
    if not token_ids:
        return "", []

    spans: List[tuple] = []
    prev = ""
    for i in range(len(token_ids)):
        cur = tok.decode(
            token_ids[: i + 1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not cur.startswith(prev):
            # Extremely rare tokenizer edge; give up span alignment.
            return tok.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ), []
        spans.append((len(prev), len(cur)))
        prev = cur
    return prev, spans


def _token_indices_overlapping_char_span(spans: List[tuple], start: int, end: int) -> List[int]:
    out = []
    for i, (s, e) in enumerate(spans):
        if e <= start:
            continue
        if s >= end:
            continue
        out.append(i)
    return out
def _task_aware_decoding(task_type: str, temperature: float, top_p: float) -> Dict[str, float]:
    """Mild task-aware decoding tweaks to improve diversity without going wild."""
    tt = (task_type or "").strip().lower()
    t = float(temperature)
    p = float(top_p)
    if tt in ("existence", "spatial", "counting"):
        t = min(1.1, max(0.5, t + 0.15))
        p = min(0.98, max(0.85, p))
    elif tt == "attribute":
        # Keep close to default; attributes are often already diverse.
        t = min(0.9, max(0.5, t))
        p = min(0.98, max(0.85, p))
    return {"temperature": t, "top_p": p}


# ────────────────────────────────────────
# 辅助：获取 EOS / PAD token ids
# ────────────────────────────────────────

def _get_eos_ids(model) -> Set[int]:
    eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        return set(eos)
    elif eos is not None:
        return {eos}
    return set()


def _get_pad_id(processor, model) -> Optional[int]:
    tok = getattr(processor, "tokenizer", processor)
    pid = getattr(tok, "pad_token_id", None)
    if pid is None:
        pid = getattr(model.config, "pad_token_id", None)
    return pid


def _actual_gen_length(gen_ids: torch.Tensor, eos_ids: Set[int], pad_id: Optional[int]) -> int:
    """找到第一个 EOS / PAD 的位置，返回有效生成长度。"""
    for t in range(len(gen_ids)):
        tok = gen_ids[t].item()
        if tok in eos_ids or (pad_id is not None and tok == pad_id):
            return t
    return len(gen_ids)


def _extract_logprobs(
    scores, gen_ids: torch.Tensor, actual_len: int, cand_row: int = 0,
) -> List[float]:
    """从 generate 的 scores 中提取 per-token logprob。"""
    logprobs = []
    for step_i in range(min(actual_len, len(scores))):
        log_p = torch.log_softmax(scores[step_i][cand_row], dim=-1)
        logprobs.append(log_p[gen_ids[step_i].item()].item())
    return logprobs


def _build_candidate_dict(
    cand_idx: int,
    raw_text: str,
    final_answer: str,
    extraction_source: str,
    response_logprobs: List[float],
    final_answer_logprobs: List[float],
    n_tokens_generated: int,
    final_answer_logprob_span_source: str,
):
    yn = parse_yesno(final_answer)

    # Compatibility note:
    # - Historical meaning of mean_logprob/sum_logprob in this repo is: whole response logprob.
    # - We keep that behavior and add final_answer_* fields for answer-only baselines.
    resp_mean_lp = (
        sum(response_logprobs) / len(response_logprobs) if response_logprobs else float("nan")
    )
    resp_sum_lp = sum(response_logprobs) if response_logprobs else float("nan")
    ans_mean_lp = (
        sum(final_answer_logprobs) / len(final_answer_logprobs) if final_answer_logprobs else float("nan")
    )
    ans_sum_lp = sum(final_answer_logprobs) if final_answer_logprobs else float("nan")

    return {
        "candidate_id": cand_idx,
        # raw response for reward computation / audit
        "raw_generated_text": raw_text,
        # extracted final answer for correctness / rerank
        "generated_text": final_answer,
        # Unified text contract for downstream reward/verifier/training.
        "candidate_response_text": raw_text,
        "candidate_answer_text": final_answer,
        "candidate_response_text_source": "raw_generated_text",
        "candidate_answer_text_source": "generated_text",
        "final_answer_extraction_source": extraction_source,
        # Whole-response logprob (legacy-compatible)
        "mean_logprob": resp_mean_lp,
        "sum_logprob": resp_sum_lp,
        "response_mean_logprob": resp_mean_lp,
        # Answer-only logprob (new, preferred baseline)
        "final_answer_mean_logprob": ans_mean_lp,
        "final_answer_sum_logprob": ans_sum_lp,
        "final_answer_logprob_span_source": final_answer_logprob_span_source,
        # Length fields
        "answer_length": len((final_answer or "").split()),
        "answer_char_length": len(final_answer or ""),
        "response_length": len((raw_text or "").split()),
        "response_char_length": len(raw_text or ""),
        "is_yesno": yn != "other",
        "yesno_label": yn,
        "n_tokens_generated": n_tokens_generated,
    }


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _default_diversity_output_path(output_path: str) -> str:
    out_dir = os.path.dirname(output_path) or "."
    base = os.path.basename(output_path)
    m = re.match(r"^candidates_shard_(\d+)\.jsonl$", base)
    if m:
        return os.path.join(out_dir, f"candidate_diversity_shard_{m.group(1)}.json")
    return os.path.join(out_dir, "candidate_diversity.json")


def compute_candidate_diversity(
    records: List[Dict[str, Any]],
    warn_unique_ratio_lt: float = 0.5,
    warn_mode_fraction_gt: float = 0.8,
) -> Dict[str, Any]:
    """Compute lightweight diversity diagnostics to expose candidate collapse.

        This sidecar reports:
            - raw response unique (post-fill)
            - final answer unique (post-fill)
            - pre-fill unique count
            - post-fill candidate count
            - duplicate-filled rate
    """
    by_sample: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        sid = r.get("sample_id")
        if sid is None:
            continue
        by_sample.setdefault(sid, []).append(r)

    per_sample = []
    by_task: Dict[str, List[Dict[str, Any]]] = {}

    raw_field = "raw_generated_text"
    final_field = "generated_text"

    for sid, cands in sorted(by_sample.items(), key=lambda x: str(x[0])):
        task_type = cands[0].get("task_type", "unknown")
        texts_raw = [normalize_text_for_analysis(c.get(raw_field) or c.get(final_field, ""), task_type=task_type) for c in cands]
        texts_raw = [t for t in texts_raw if t != ""]
        post_count = len(cands)
        k = len(texts_raw)
        if post_count == 0:
            continue
        counts: Dict[str, int] = {}
        for t in texts_raw:
            counts[t] = counts.get(t, 0) + 1
        uniq = len(counts)
        mode_count = max(counts.values()) if counts else 0
        mode_fraction = mode_count / post_count if post_count else 0.0
        unique_ratio = uniq / post_count if post_count else 0.0

        # Final-answer diversity (generated_text)
        texts_final = [normalize_text_for_analysis(c.get(final_field, ""), task_type=task_type) for c in cands]
        texts_final = [t for t in texts_final if t != ""]
        kf = len(texts_final)
        counts_f: Dict[str, int] = {}
        for t in texts_final:
            counts_f[t] = counts_f.get(t, 0) + 1
        uniq_f = len(counts_f)
        mode_count_f = max(counts_f.values()) if counts_f else 0
        mode_frac_f = mode_count_f / post_count if post_count else 0.0
        unique_ratio_f = uniq_f / post_count if post_count else 0.0
        yesno_ratio = sum(1 for c in cands if c.get("is_yesno", False)) / len(cands) if cands else 0.0
        lens = [int(c.get("answer_length", 0)) for c in cands]
        mean_len = sum(lens) / len(lens) if lens else 0.0
        pre_fill_unique = int(cands[0].get("n_unique_candidates_before_fill", uniq))
        duplicate_filled = sum(1 for c in cands if bool(c.get("is_duplicate_candidate", False)))
        duplicate_fill_rate = (duplicate_filled / post_count) if post_count else 0.0

        row = {
            "sample_id": sid,
            "task_type": task_type,
            "n_candidates_post_fill": post_count,
            "n_unique_candidates_pre_fill": pre_fill_unique,
            "n_duplicate_filled": duplicate_filled,
            "duplicate_filled_rate": round(duplicate_fill_rate, 6),
            "duplicate_fill_applied": duplicate_filled > 0,
            # Raw-response diversity (compat: old sidecar looked at generated_text; now we use raw)
            "raw_text_field": raw_field,
            "n_nonempty_raw_response": k,
            "n_unique_raw_response": uniq,
            "raw_unique_ratio": round(unique_ratio, 6),
            "mode_fraction": round(mode_fraction, 6),
            "yesno_ratio": round(yesno_ratio, 6),
            "mean_answer_length": round(mean_len, 6),
            "all_same": uniq == 1,
            "low_unique_ratio": unique_ratio < warn_unique_ratio_lt,
            "high_mode_fraction": mode_fraction > warn_mode_fraction_gt,

            # Final-answer diversity (new)
            "final_text_field": final_field,
            "n_nonempty_final_answer": kf,
            "n_unique_final_answer": uniq_f,
            "final_unique_ratio": round(unique_ratio_f, 6),
            "final_mode_fraction": round(mode_frac_f, 6),
            "final_all_same": uniq_f == 1,
        }
        per_sample.append(row)
        by_task.setdefault(task_type, []).append(row)

    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {
                "n_samples": 0,
                "mean_unique_ratio": float("nan"),
                "median_unique_ratio": float("nan"),
                "pct_all_same": float("nan"),
                "pct_low_unique_ratio": float("nan"),
                "pct_high_mode_fraction": float("nan"),
            }
        urs = sorted(r["raw_unique_ratio"] for r in rows)
        mid = len(urs) // 2
        median = urs[mid] if len(urs) % 2 == 1 else (urs[mid - 1] + urs[mid]) / 2
        mean = sum(urs) / len(urs)
        pct_all_same = sum(1 for r in rows if r["all_same"]) / len(rows)
        pct_low_ur = sum(1 for r in rows if r["low_unique_ratio"]) / len(rows)
        pct_high_mode = sum(1 for r in rows if r["high_mode_fraction"]) / len(rows)
        return {
            "n_samples": len(rows),
            "mean_unique_ratio": round(mean, 6),
            "median_unique_ratio": round(median, 6),
            "pct_all_same": round(pct_all_same, 6),
            "pct_low_unique_ratio": round(pct_low_ur, 6),
            "pct_high_mode_fraction": round(pct_high_mode, 6),
        }

    global_summary = _agg(per_sample)
    per_task_summary = {tt: _agg(rows) for tt, rows in sorted(by_task.items())}

    warnings = []
    if isinstance(global_summary.get("pct_all_same"), (int, float)) and global_summary["pct_all_same"] > 0.2:
        warnings.append("high_all_same_rate")
    if isinstance(global_summary.get("mean_unique_ratio"), (int, float)) and global_summary["mean_unique_ratio"] < 0.7:
        warnings.append("low_mean_unique_ratio")

    return {
        "thresholds": {
            "warn_unique_ratio_lt": warn_unique_ratio_lt,
            "warn_mode_fraction_gt": warn_mode_fraction_gt,
        },
        "global": global_summary,
        "by_task_type": per_task_summary,
        "per_sample": per_sample,
        "warnings": warnings,
    }


# ────────────────────────────────────────
# 核心：批量并行生成（num_return_sequences）
# ────────────────────────────────────────

@torch.no_grad()
def generate_candidates(
    model,
    processor,
    image: Image.Image,
    question: str,
    device: str,
    num_candidates: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 32,
    task_type: str = "",
    do_sample: bool = True,
) -> List[Dict[str, Any]]:
    """为一个 (image, question) 并行生成 K 个候选答案。

    优先使用 num_return_sequences 一次生成 K 个（效率高），
    失败时回退到逐个生成。
    """
    inputs = prepare_inputs(processor, image, question, device)
    prompt_len = inputs["input_ids"].shape[-1]
    eos_ids = _get_eos_ids(model)
    pad_id = _get_pad_id(processor, model)

    try:
        return _generate_batch(
            model, processor, inputs, prompt_len,
            num_candidates, temperature, top_p, max_new_tokens,
            eos_ids, pad_id,
            task_type=task_type,
            do_sample=do_sample,
        )
    except Exception:
        return _generate_sequential(
            model, processor, inputs, prompt_len, device,
            num_candidates, temperature, top_p, max_new_tokens,
            eos_ids, pad_id,
            task_type=task_type,
            do_sample=do_sample,
        )


def _generate_batch(
    model, processor, inputs, prompt_len,
    num_candidates, temperature, top_p, max_new_tokens,
    eos_ids, pad_id,
    task_type: str = "",
    do_sample: bool = True,
) -> List[Dict[str, Any]]:
    """一次 generate 产生 K 个候选（num_return_sequences=K）。"""
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=bool(do_sample),
        num_return_sequences=num_candidates,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    outputs = model.generate(
        **gen_kwargs,
    )
    # outputs.sequences: [K, prompt_len + gen_len]
    # outputs.scores: tuple of [K, vocab_size], length = gen_steps

    tok = getattr(processor, "tokenizer", processor)
    candidates = []
    for cand_idx in range(num_candidates):
        gen_ids = outputs.sequences[cand_idx, prompt_len:]
        actual_len = _actual_gen_length(gen_ids, eos_ids, pad_id)
        gen_ids_trimmed = gen_ids[:actual_len]

        # Decode + extract final answer
        token_ids = [int(x) for x in gen_ids_trimmed.tolist()]
        raw_text, spans = _decode_with_token_char_spans(tok, token_ids)
        raw_text = (raw_text or "").strip()
        ext = extract_final_answer(raw_text, task_type=task_type)
        final_answer = (ext.get("final_answer") or "").strip()
        extraction_source = ext.get("source") or "fallback_raw"

        # Per-token logprobs from generate() scores
        logprobs = _extract_logprobs(
            outputs.scores, gen_ids, actual_len, cand_row=cand_idx)

        # Answer-only logprob: align by char span when tag exists.
        ans_lp = []
        span_source = "fallback_full_response"
        if spans and ext.get("char_span"):
            cs, ce = ext["char_span"]
            idxs = _token_indices_overlapping_char_span(spans, cs, ce)
            if idxs:
                ans_lp = [logprobs[i] for i in idxs if i < len(logprobs)]
                span_source = "tag_char_span"
            else:
                ans_lp = list(logprobs)
                span_source = "tag_span_align_failed"
        else:
            ans_lp = list(logprobs)

        candidates.append(
            _build_candidate_dict(
                cand_idx,
                raw_text=raw_text,
                final_answer=final_answer,
                extraction_source=extraction_source,
                response_logprobs=list(logprobs),
                final_answer_logprobs=ans_lp,
                n_tokens_generated=actual_len,
                final_answer_logprob_span_source=span_source,
            )
        )

    return candidates


def _generate_sequential(
    model, processor, inputs, prompt_len, device,
    num_candidates, temperature, top_p, max_new_tokens,
    eos_ids, pad_id,
    task_type: str = "",
    do_sample: bool = True,
) -> List[Dict[str, Any]]:
    """逐个生成（回退方案）。"""
    tok = getattr(processor, "tokenizer", processor)
    candidates = []
    for cand_idx in range(num_candidates):
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=bool(do_sample),
            output_scores=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        outputs = model.generate(
            **gen_kwargs,
        )
        gen_ids = outputs.sequences[0, prompt_len:]
        actual_len = _actual_gen_length(gen_ids, eos_ids, pad_id)
        gen_ids_trimmed = gen_ids[:actual_len]

        token_ids = [int(x) for x in gen_ids_trimmed.tolist()]
        raw_text, spans = _decode_with_token_char_spans(tok, token_ids)
        raw_text = (raw_text or "").strip()
        ext = extract_final_answer(raw_text, task_type=task_type)
        final_answer = (ext.get("final_answer") or "").strip()
        extraction_source = ext.get("source") or "fallback_raw"

        logprobs = _extract_logprobs(
            outputs.scores, gen_ids, actual_len, cand_row=0)

        ans_lp = []
        span_source = "fallback_full_response"
        if spans and ext.get("char_span"):
            cs, ce = ext["char_span"]
            idxs = _token_indices_overlapping_char_span(spans, cs, ce)
            if idxs:
                ans_lp = [logprobs[i] for i in idxs if i < len(logprobs)]
                span_source = "tag_char_span"
            else:
                ans_lp = list(logprobs)
                span_source = "tag_span_align_failed"
        else:
            ans_lp = list(logprobs)

        candidates.append(
            _build_candidate_dict(
                cand_idx,
                raw_text=raw_text,
                final_answer=final_answer,
                extraction_source=extraction_source,
                response_logprobs=list(logprobs),
                final_answer_logprobs=ans_lp,
                n_tokens_generated=actual_len,
                final_answer_logprob_span_source=span_source,
            )
        )

    return candidates


def classify_candidate(
    gen_text: str,
    gt_answer: str,
    gt_present: bool,
    task_type: str,
    answer_aliases: Optional[List[str]] = None,
) -> str:
    """判断候选答案的 correctness behavior。"""
    gt = (gt_answer or "").strip().lower()
    pred = (gen_text or "").strip().lower()
    gt_yn = parse_yesno(gt)
    pred_yn = parse_yesno(pred)

    if task_type == "existence" or gt_yn != "other":
        if pred_yn == "other":
            return "other"
        if gt_present:
            return "correct_positive" if pred_yn == "yes" else "miss"
        else:
            return "correct_negative" if pred_yn == "no" else "hallucination"

    # Open-ended tasks: use task-aware semantic evaluator instead of strict raw-string equality.
    is_ok = classify_open_ended_correctness(task_type, pred, gt)

    # Optional aliases from dataset (mainly attribute task), e.g. ["couch", "sofa"].
    if (not is_ok) and answer_aliases:
        if task_type == "attribute":
            is_ok = any(semantically_match_attribute(pred, a) for a in answer_aliases)
        else:
            is_ok = any(classify_open_ended_correctness(task_type, pred, a) for a in answer_aliases)

    return "correct_positive" if is_ok else ("miss" if gt_present else "hallucination")


def is_correct(behavior: str) -> bool:
    return behavior in ("correct_positive", "correct_negative")


# ────────────────────────────────────────
# 批量处理
# ────────────────────────────────────────

def run_candidate_generation(
    samples: List[Dict],
    model,
    processor,
    coco_image_dir: str,
    device: str,
    num_candidates: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 32,
    do_sample: bool = True,
    # New: prompt/sampling controls (defaults keep legacy behavior unless run.sh enables them)
    prompt_mode: str = "raw_question",
    task_aware_sampling: bool = False,
    candidate_batch_size: int = 0,
    min_unique_candidates: int = 0,
    max_sampling_rounds: int = 1,
    dedup_on: str = "raw_text",
    allow_duplicate_fill: bool = False,
    output_path: str = "candidates.jsonl",
    diversity_output_path: Optional[str] = None,
) -> List[Dict]:
    """对所有样本生成候选，写入 JSONL，返回 flat list of candidate records。"""
    all_records = []
    n_error_rows = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, s in enumerate(tqdm(samples, desc="CandGen", ncols=80)):
            sample_id = s.get("_global_idx", idx)
            try:
                # 复用主训练数据适配层的统一图片解析逻辑，避免 candidate 生成和训练阶段各自拼路径。
                resolved_image_path = resolve_image_abspath(s, image_root=coco_image_dir)
                image = open_sample_image(s, image_root=coco_image_dir)

                task_type = s.get("task_type", "existence")
                gt_answer = s.get("answer", s.get("visual_answer", ""))
                gt_present = s.get("gt_present", True)
                answer_aliases = s.get("answer_aliases", None)

                prompt = build_generation_prompt(s, prompt_mode=prompt_mode)
                use_task_aware = bool(task_aware_sampling)
                dec = _task_aware_decoding(task_type, temperature, top_p) if use_task_aware else {
                    "temperature": temperature,
                    "top_p": top_p,
                }

                # Resampling + dedup to mitigate collapse.
                batch_size = int(candidate_batch_size) if candidate_batch_size else int(num_candidates)
                rounds = int(max_sampling_rounds) if max_sampling_rounds else 1
                rounds = max(1, rounds)
                requested = max(1, int(num_candidates))
                min_unique_required = max(0, int(min_unique_candidates))
                min_unique_required = min(min_unique_required, requested)

                def _gen_one_round(n: int):
                    return generate_candidates(
                        model, processor, image, prompt, device,
                        num_candidates=n,
                        temperature=dec["temperature"],
                        top_p=dec["top_p"],
                        max_new_tokens=max_new_tokens,
                        task_type=task_type,
                        do_sample=do_sample,
                    )

                # Collect unique candidates.
                dedup_field = (dedup_on or "raw_text").strip().lower()
                if dedup_field not in ("final_answer", "raw_text"):
                    dedup_field = "raw_text"

                unique: List[Dict[str, Any]] = []
                seen: Dict[str, int] = {}

                rounds_executed = 0
                for r in range(rounds):
                    rounds_executed = r + 1
                    batch = _gen_one_round(batch_size)
                    for b in batch:
                        key_text = b.get("generated_text", "") if dedup_field == "final_answer" else b.get(
                            "raw_generated_text", b.get("generated_text", "")
                        )
                        if dedup_field == "raw_text":
                            dedup_semantics = "raw_text_light"
                            norm = normalize_raw_text_for_dedup(key_text)
                        else:
                            dedup_semantics = "final_answer_task_aware"
                            norm = normalize_final_answer_for_dedup(key_text, task_type=task_type)

                        seen_key = norm if norm else "__empty__"
                        if seen_key in seen:
                            continue
                        seen[seen_key] = len(unique)

                        b["dedup_key_raw"] = key_text
                        b["dedup_key_normalized"] = norm
                        b["dedup_semantics"] = dedup_semantics
                        b["sampling_round"] = r
                        unique.append(b)
                        if len(unique) >= requested:
                            break
                    if len(unique) >= requested:
                        break

                min_unique_satisfied = (
                    True if min_unique_required <= 0 else (len(unique) >= min_unique_required)
                )

                # Default verifier path is variable-K: no duplicate fill.
                # Optional legacy mode can force fixed-K via explicit duplicate fill.
                final_list: List[Dict[str, Any]] = []
                for i in range(min(requested, len(unique))):
                    c = dict(unique[i])
                    c["is_duplicate_candidate"] = False
                    c["duplicate_of_candidate_id"] = None
                    final_list.append(c)

                duplicate_fill_applied = False
                if allow_duplicate_fill and len(final_list) < requested:
                    duplicate_fill_applied = True
                    for i in range(len(final_list), requested):
                        if unique:
                            src_idx = (i - len(unique)) % len(unique)
                            src = dict(unique[src_idx])
                            src["is_duplicate_candidate"] = True
                            src["duplicate_of_candidate_id"] = src_idx
                            final_list.append(src)
                        else:
                            final_list.append({
                                "candidate_id": i,
                                "raw_generated_text": "",
                                "generated_text": "",
                                "dedup_key_raw": "",
                                "dedup_key_normalized": "",
                                "dedup_semantics": (
                                    "raw_text_light" if dedup_field == "raw_text" else "final_answer_task_aware"
                                ),
                                "final_answer_extraction_source": "fallback_raw",
                                "mean_logprob": float("nan"),
                                "sum_logprob": float("nan"),
                                "response_mean_logprob": float("nan"),
                                "final_answer_mean_logprob": float("nan"),
                                "final_answer_sum_logprob": float("nan"),
                                "final_answer_logprob_span_source": "fallback_full_response",
                                "answer_length": 0,
                                "answer_char_length": 0,
                                "response_length": 0,
                                "response_char_length": 0,
                                "is_yesno": False,
                                "yesno_label": "other",
                                "n_tokens_generated": 0,
                                "is_duplicate_candidate": True,
                                "duplicate_of_candidate_id": None,
                            })

                # Force candidate_id semantics: 0..K-1 in final written list.
                candidates = []
                for cid, c in enumerate(final_list):
                    c["candidate_id"] = cid
                    c["dedup_on"] = dedup_field
                    c.setdefault("dedup_key_raw", "")
                    c.setdefault("dedup_key_normalized", "")
                    c.setdefault(
                        "dedup_semantics",
                        "raw_text_light" if dedup_field == "raw_text" else "final_answer_task_aware",
                    )
                    c["n_sampling_rounds"] = rounds_executed
                    c["n_unique_candidates_before_fill"] = len(unique)
                    c["min_unique_candidates"] = min_unique_required
                    c["min_unique_satisfied"] = bool(min_unique_satisfied)
                    c["max_sampling_rounds"] = rounds
                    c["requested_num_candidates"] = requested
                    c["allow_duplicate_fill"] = bool(allow_duplicate_fill)
                    c["duplicate_fill_applied"] = bool(duplicate_fill_applied)
                    c["n_candidates_post_fill"] = len(final_list)
                    candidates.append(c)

                for cand in candidates:
                    beh = classify_candidate(
                        cand["generated_text"],
                        gt_answer,
                        gt_present,
                        task_type,
                        answer_aliases=answer_aliases,
                    )
                    rec = {
                        "sample_id": sample_id,
                        "image_file": s["image_file"],
                        "resolved_image_path": resolved_image_path,
                        "question": s["question"],
                        "task_family": infer_task_family(task_type=task_type, question=s.get("question", ""), metadata=s),
                        # New: inference prompt provenance (so reruns are auditable)
                        "generation_prompt_mode": prompt_mode,
                        "generation_prompt": prompt,
                        "gt_answer": gt_answer,
                        "gt_present": gt_present,
                        "task_type": task_type,
                        "target_bbox": s.get("target_bbox"),
                        "pair_id": s.get("pair_id"),
                        "answer_aliases": answer_aliases,
                        **cand,
                        "behavior": beh,
                        "correct": is_correct(beh),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    all_records.append(rec)

            except Exception as e:
                # 写一条 error 记录
                n_error_rows += 1
                rec = {
                    "sample_id": sample_id,
                    "image_file": s.get("image_file", ""),
                    "source_image_path": s.get("source_image_path", ""),
                    "question": s.get("question", ""),
                    "task_type": s.get("task_type", ""),
                    "task_family": infer_task_family(task_type=s.get("task_type", ""), question=s.get("question", ""), metadata=s),
                    "resolved_image_path": resolve_image_abspath(s, image_root=coco_image_dir),
                    "image_root": coco_image_dir,
                    "error": repr(e),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[candidate_gen] {len(all_records)} candidates, {n_error_rows} error rows → {output_path}"
    )

    # Diversity sidecar (helps diagnose candidate collapse).
    try:
        div_path = diversity_output_path or _default_diversity_output_path(output_path)
        div = compute_candidate_diversity(all_records)
        with open(div_path, "w", encoding="utf-8") as f:
            json.dump(div, f, indent=2, ensure_ascii=False)
        print(f"[candidate_gen] diversity → {div_path}")
    except Exception as e:
        print(f"[candidate_gen] diversity failed: {repr(e)}")

    return all_records


# ────────────────────────────────────────
# CLI
# ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Multi-candidate answer generation")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--vqa_file", required=True)
    ap.add_argument("--coco_image_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--checkpoint_path", default="",
                    help="可选：恢复 trainable checkpoint，用于主实验 benchmark/probe 评测")
    ap.add_argument("--output", default="results/verifier/candidates.jsonl")
    ap.add_argument("--diversity_output", default="",
                    help="多样性 sidecar 输出路径（默认根据 --output 推断）")
    ap.add_argument("--num_candidates", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--greedy", action="store_true",
                    help="关闭 sampling，改为确定性生成；benchmark_final 默认应使用它")
    # New: prompt/sampling controls (kept opt-in; run.sh will enable by default for verifier)
    ap.add_argument("--prompt_mode", default="raw_question", choices=list(PROMPT_MODES))
    ap.add_argument("--task_aware_sampling", action="store_true",
                    help="启用 task-aware prompt + mild task-aware decoding")
    ap.add_argument("--candidate_batch_size", type=int, default=0,
                    help="每轮 generate 的候选数（0=等于 num_candidates）")
    ap.add_argument("--min_unique_candidates", type=int, default=0,
                    help="最低可接受 unique 候选数（硬约束：达不到会继续重采样直到轮次上限）")
    ap.add_argument("--max_sampling_rounds", type=int, default=1,
                    help="分轮采样次数上限（>1 启用分轮去重聚合）")
    ap.add_argument("--dedup_on", default="raw_text", choices=["final_answer", "raw_text"],
                    help="去重依据：raw_text(默认) 或 final_answer")
    ap.add_argument("--allow_duplicate_fill", action="store_true",
                    help="显式开启固定-K重复补齐（默认关闭，verifier 主评估使用 variable-K）")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="最大样本数（0=全量）")
    # 多卡分片
    ap.add_argument("--shard_id", type=int, default=0,
                    help="当前分片 ID（0-based）")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="总分片数（>1 启用分片模式）")
    args = ap.parse_args()

    processor, model, cfg = load(args.model_dir, device=args.device, dtype=args.dtype)
    if args.checkpoint_path:
        from mini_grpo_smoke import _setup_trainable

        payload = torch.load(args.checkpoint_path, map_location="cpu")
        extra = dict(payload.get("extra") or {})
        state = payload.get("trainable_state_dict") or {}
        model, _ = _setup_trainable(
            model,
            n_trainable_layers=int(extra.get("n_trainable_layers", 4)),
            use_lora=bool(extra.get("lora_applied", False)),
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[candidate_gen] checkpoint_loaded path={args.checkpoint_path} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.eval()

    with open(args.vqa_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    for i, s in enumerate(samples):
        s.setdefault("_global_idx", i)

    if args.max_samples > 0:
        import random
        rng = random.Random(42)
        rng.shuffle(samples)
        samples = samples[:args.max_samples]

    # 多卡分片：取当前 shard 对应的子集
    if args.num_shards > 1:
        shard_size = math.ceil(len(samples) / args.num_shards)
        start = args.shard_id * shard_size
        end = min(start + shard_size, len(samples))
        samples = samples[start:end]
        print(f"[candidate_gen] shard {args.shard_id}/{args.num_shards}: "
              f"{len(samples)} samples (range [{start}, {end}))")

    print(f"[candidate_gen] {len(samples)} samples × {args.num_candidates} candidates")

    t0 = time.time()
    run_candidate_generation(
        samples, model, processor, args.coco_image_dir, args.device,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        do_sample=(not args.greedy),
        prompt_mode=args.prompt_mode,
        task_aware_sampling=args.task_aware_sampling,
        candidate_batch_size=args.candidate_batch_size,
        min_unique_candidates=args.min_unique_candidates,
        max_sampling_rounds=args.max_sampling_rounds,
        dedup_on=args.dedup_on,
        allow_duplicate_fill=args.allow_duplicate_fill,
        output_path=args.output,
        diversity_output_path=(args.diversity_output or None),
    )
    print(f"[candidate_gen] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
