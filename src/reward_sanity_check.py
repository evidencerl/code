"""Reward Sanity Check（v5）：兼容 v4 ActionLogProbATEReward 的三种 reward mode。

v5 变更:
  - 新增 --reward_mode 参数，透传给 ActionLogProbATEReward
  - Check C2 兼容 v4 新增字段（delta_ans_margin, score_base 等）
  - Check C2 summary 新增 soft_shaping 模式的分组统计
  - 保留 Check A, B, C1 不变（它们用 LogOddsATEReward，不受 reward_mode 影响）
"""

import argparse
import json
import math
import os
import sys
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent
_ROOT_DIR = _SRC_DIR.parent
for _p in (str(_ROOT_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_local_module(module_name: str, file_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"cannot create spec for {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from model_loader import load
except ModuleNotFoundError as e:
    if e.name != 'model_loader':
        raise
    _ml = _load_local_module('model_loader', _SRC_DIR / 'model_loader.py')
    load = _ml.load
from reward.logodds_ate import LogOddsATEReward
from reward.action_logprob_ate import ActionLogProbATEReward, parse_yesno
from answer_format_utils import (
    PROMPT_MODES,
    extract_final_answer,
    extract_final_answer_with_prefix,
    build_generation_prompt,
    normalize_text_for_analysis,
    task_aware_answer_score,
    infer_task_family,
)
from dataset_adapters import load_vg_brutal_as_main_schema
from dataset_adapters import open_sample_image, resolve_image_abspath, check_dataset_integrity
from ced_core import prepare_inputs, expand_mm_inputs


def load_vqa(path, max_samples=0, task_type="existence", seed=42):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    if task_type:
        data = [s for s in data if s.get("task_type") == task_type]
    if max_samples and len(data) > max_samples:
        random.seed(seed)
        data = random.sample(data, max_samples)
    return data


def load_dataset_main_schema(
    dataset_name: str,
    dataset_file: str,
    image_root: str,
    max_samples: int = 0,
    seed: int = 42,
    task_type: str = "",
    data_split: str = "all",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    name = (dataset_name or "legacy_vqa").strip().lower()
    if name == "vg_brutal":
        samples, summary = load_vg_brutal_as_main_schema(
            dataset_file=dataset_file,
            image_root=image_root,
            max_samples=max_samples,
            seed=seed,
            task_type=task_type,
            split=data_split,
            fill_image_size=True,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        return samples, summary

    samples = load_vqa(dataset_file, max_samples=max_samples, task_type=task_type, seed=seed)
    summary = {
        "dataset_name": "legacy_vqa",
        "dataset_file": dataset_file,
        "image_root": image_root,
        "valid_count": len(samples),
        "task_type_filter": task_type,
    }
    return samples, summary


def resolve_dataset_paths(args: argparse.Namespace) -> Dict[str, str]:
    dataset_name = (args.dataset_name or "legacy_vqa").strip().lower()
    if dataset_name not in {"legacy_vqa", "vg_brutal"}:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")

    if dataset_name == "vg_brutal":
        if not (args.dataset_file or "").strip():
            raise ValueError("dataset_name=vg_brutal requires --dataset_file")
        if not (args.image_root or "").strip():
            raise ValueError("dataset_name=vg_brutal requires --image_root")
        return {
            "dataset_name": dataset_name,
            "dataset_file": args.dataset_file,
            "image_root": args.image_root,
        }

    if not (args.vqa_file or "").strip():
        raise ValueError("dataset_name=legacy_vqa requires --vqa_file")
    if not (args.coco_image_dir or "").strip():
        raise ValueError("dataset_name=legacy_vqa requires --coco_image_dir")
    return {
        "dataset_name": dataset_name,
        "dataset_file": args.vqa_file,
        "image_root": args.coco_image_dir,
    }


def resolve_task_type_filter(dataset_name: str, task_type_arg: str) -> str:
    t = (task_type_arg or "").strip().lower()
    if t in ("", "all", "any", "*"):
        return "" if dataset_name == "vg_brutal" else "existence"
    return t


def resolve_checks(dataset_name: str, checks_arg: str) -> Tuple[set, Dict[str, List[str]]]:
    legacy_default_checks = ["a", "b", "c1", "c2", "d"]
    vg_brutal_default_checks = ["c2", "d"]

    selected_raw = (checks_arg or "").strip().lower()
    if not selected_raw:
        selected_raw = ",".join(vg_brutal_default_checks if dataset_name == "vg_brutal" else legacy_default_checks)

    checks = set()
    for c in selected_raw.split(","):
        c = c.strip().lower()
        if not c:
            continue
        if c == "c":
            checks.update(["c1", "c2"])
        else:
            checks.add(c)

    return checks, {
        "legacy_default_checks": legacy_default_checks,
        "vg_brutal_default_checks": vg_brutal_default_checks,
    }


def summarize_loaded_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    task_dist: Dict[str, int] = defaultdict(int)
    split_dist: Dict[str, int] = defaultdict(int)
    for s in samples:
        task_dist[(s.get("task_type") or "unknown")] += 1
        split_dist[(s.get("data_split") or "unknown")] += 1
    return {
        "loaded_sample_count": int(len(samples)),
        "task_type_distribution": dict(task_dist),
        "split_distribution": dict(split_dist),
    }


def apply_shard_filter(samples: List[Dict[str, Any]], shard_index: int = 0, num_shards: int = 1) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return list(samples)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard_index={shard_index} for num_shards={num_shards}")
    return [s for i, s in enumerate(samples) if (i % num_shards) == shard_index]


def _is_no_cuda_error(msg: str) -> bool:
    t = (msg or "").lower()
    return ("no cuda gpus are available" in t) or ("cuda" in t and "not available" in t)


def _bbox_area(bbox):
    return max(0.0, bbox[2]) * max(0.0, bbox[3])


@torch.no_grad()
def run_training_preflight_check(
    model,
    processor,
    reward_fn,
    samples: List[Dict[str, Any]],
    image_root: str,
    reward_backend: str,
    prompt_mode: str = "raw_question",
    answer_text_source: str = "final_answer",
    policy_logprob_scope: str = "final_answer",
    max_new_tokens: int = 32,
    group_size: int = 4,
    n_groups: int = 8,
    temperature: float = 1.0,
    top_p: float = 0.95,
    do_sample: bool = True,
    allow_prompt_side_reward_for_smoke: bool = False,
) -> Dict[str, Any]:
    """训练前小批量 preflight：检查组内 raw reward 方差与 reward contract。"""
    backend = (reward_backend or "").strip().lower()
    if backend in ("", "auto"):
        backend = "action_logprob_ate_auto"
    check_name = "check_d_group_reward_contract"
    reasons: List[str] = []

    if backend == "logodds_ate" and not allow_prompt_side_reward_for_smoke:
        return {
            "check_name": check_name,
            "passed": False,
            "reward_backend": backend,
            "reasons": [
                "prompt-side/sample-level reward 不适合作为默认 group reward",
                "reward_backend=logodds_ate is blocked by default",
            ],
            "prompt_side_reward_detected": True,
        }

    if not samples:
        return {
            "check_name": check_name,
            "passed": False,
            "reward_backend": backend,
            "reasons": ["no_samples_for_preflight"],
            "prompt_side_reward_detected": backend == "logodds_ate",
        }

    rng = random.Random(42)
    take = min(len(samples), max(1, n_groups))
    family_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        fam = infer_task_family(task_type=s.get("task_type", ""), question=s.get("question", ""))
        family_buckets[fam].append(s)

    if len(family_buckets) <= 1:
        chosen = rng.sample(samples, take) if len(samples) > take else list(samples)
    else:
        chosen = []
        fams = sorted(family_buckets.keys())
        base = max(1, take // max(1, len(fams)))
        remainder = max(0, take - base * len(fams))
        for i, fam in enumerate(fams):
            want = base + (1 if i < remainder else 0)
            bucket = family_buckets[fam]
            if len(bucket) <= want:
                chosen.extend(bucket)
            else:
                chosen.extend(rng.sample(bucket, want))
        if len(chosen) > take:
            chosen = chosen[:take]
        elif len(chosen) < take:
            pool = list(samples)
            while len(chosen) < take and pool:
                chosen.append(rng.choice(pool))
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    device = getattr(reward_fn, "device", "cuda:0")
    if hasattr(reward_fn, "resolve_runtime_device"):
        try:
            device = reward_fn.resolve_runtime_device()
        except Exception:
            device = getattr(reward_fn, "device", "cuda:0")

    group_records = []
    error_samples: List[Dict[str, Any]] = []
    group_error_types: Dict[str, int] = defaultdict(int)
    for s in chosen:
        try:
            img = open_sample_image(s, image_root=image_root)
            task_type = (s.get("task_type") or "existence").strip().lower()
            prompt = build_generation_prompt(s, prompt_mode=prompt_mode)
            inputs = prepare_inputs(processor, img, prompt, device)

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(do_sample),
                "use_cache": True,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
            eos_id = getattr(tok, "eos_token_id", None)
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id
            if eos_id is not None:
                gen_kwargs["eos_token_id"] = eos_id

            from model_loader import generation_model
            gen_model = generation_model(model)
            cfg = getattr(gen_model, "config", None)
            old_cache = getattr(cfg, "use_cache", None) if cfg is not None else None
            if cfg is not None:
                cfg.use_cache = True
            ckpt_on = bool(getattr(gen_model, "is_gradient_checkpointing", False))
            if ckpt_on and hasattr(gen_model, "gradient_checkpointing_disable"):
                gen_model.gradient_checkpointing_disable()
            try:
                expanded = expand_mm_inputs(inputs, int(group_size))
                with torch.inference_mode():
                    out = gen_model.generate(**expanded, **gen_kwargs)
                prompt_len = int(inputs["input_ids"].shape[-1])
                raw_responses = [
                    str(t).strip()
                    for t in processor.batch_decode(
                        out[:, prompt_len:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                ]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raw_responses = []
                with torch.inference_mode():
                    for _ in range(int(group_size)):
                        out = gen_model.generate(**inputs, **gen_kwargs)
                        raw_responses.append(
                            processor.batch_decode(
                                out[:, inputs["input_ids"].shape[-1]:],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )[0].strip()
                        )
            finally:
                if cfg is not None and old_cache is not None:
                    cfg.use_cache = old_cache
                if ckpt_on and hasattr(gen_model, "gradient_checkpointing_enable"):
                    try:
                        gen_model.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False}
                        )
                    except TypeError:
                        gen_model.gradient_checkpointing_enable()

            final_answers = []
            parsed_answers = []
            raw_rewards = []
            lengths = []
            parse_other = 0
            answer_scores = []
            open_form_n = 0
            open_form_extract_fail = 0
            open_form_parse_other = 0
            gt_ans = (s.get("answer") or "").strip()

            for rr in raw_responses:
                ext = extract_final_answer(rr, task_type=task_type)
                fa = (ext.get("final_answer") or "").strip()
                final_answers.append(fa)
                ans_text = fa if (answer_text_source or "").strip().lower() == "final_answer" else rr
                parsed = parse_yesno(ans_text)
                parsed_answers.append(parsed)
                if parsed == "other":
                    parse_other += 1
                lengths.append(len((rr or "").split()))
                answer_scores.append(task_aware_answer_score(
                    ans_text,
                    gt_ans,
                    task_type=s.get("task_type", task_type),
                    question=s.get("question", ""),
                ))

                if task_type != "existence":
                    open_form_n += 1
                    if (answer_text_source or "").strip().lower() == "final_answer":
                        src = ext.get("source") or "fallback_raw"
                        if not fa or src == "fallback_raw":
                            open_form_extract_fail += 1
                    if parsed == "other":
                        open_form_parse_other += 1

                effective_backend = backend
                if effective_backend == "action_logprob_ate_auto":
                    fam = infer_task_family(task_type=s.get("task_type", ""), question=s.get("question", ""))
                    gt_answer_norm = parse_yesno((s.get("answer") or "").strip())
                    has_supervision = (fam == "existence") and (
                        (s.get("gt_present") is not None) or (gt_answer_norm in ("yes", "no"))
                    )
                    effective_backend = "action_logprob_ate_supervised" if has_supervision else "action_logprob_ate_nolabel"

                if effective_backend in ("action_logprob_ate_supervised", "action_logprob_ate_nolabel", "response_conditioned_final_answer_ate", "response_conditioned_final_answer_ate_nolabel"):

                    r = reward_fn.compute(
                        image=img,
                        question=s["question"],
                        target_bbox=s["target_bbox"],
                        response_text=rr,
                        answer_text=ans_text,
                        gt_answer=s.get("answer") if effective_backend == "action_logprob_ate_supervised" else None,
                        gt_present=s.get("gt_present") if effective_backend == "action_logprob_ate_supervised" else None,
                        image_width=s.get("image_width"),
                        image_height=s.get("image_height"),
                        response_text_source="raw_generated_text",
                        answer_text_source=("generated_text" if (answer_text_source or "").strip().lower() == "final_answer" else "raw_response"),
                        task_type=task_type,
                    )
                    if r.get("error"):
                        raise RuntimeError(f"reward_fn_error:{r.get('error')}")
                    score = r.get("supervised_shaped_reward") if effective_backend == "action_logprob_ate_supervised" else r.get("score_base_no_label")
                    raw_rewards.append(float(score if score is not None else r.get("reward", 0.0)))
                elif backend == "logodds_ate":
                    r = reward_fn.compute(
                        image=img,
                        question=prompt,
                        target_bbox=s["target_bbox"],
                        image_width=s.get("image_width"),
                        image_height=s.get("image_height"),
                    )
                    raw_rewards.append(float(r.get("reward", 0.0)))
                else:
                    raise ValueError(f"Unknown backend in preflight: {backend}")

            raw_std = float(np.std(raw_rewards))
            raw_identical = bool(raw_std < 1e-8)
            parse_identical = len(set(parsed_answers)) == 1 and len(parsed_answers) > 1
            norm_final = [normalize_text_for_analysis(x, task_type=task_type) for x in final_answers]
            norm_final = [x for x in norm_final if x != ""]
            final_identical = len(set(norm_final)) == 1 and len(final_answers) > 1

            group_records.append({
                "raw_std": raw_std,
                "raw_identical": raw_identical,
                "parse_identical": parse_identical,
                "final_identical": final_identical,
                "reward_nonidentical": not raw_identical,
                "parse_other_rate": parse_other / max(1, len(parsed_answers)),
                "answer_score_mean": float(np.mean(answer_scores)) if answer_scores else 0.0,
                "total_reward_mean": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
                "open_form_n": int(open_form_n),
                "open_form_extract_fail_rate": (float(open_form_extract_fail) / max(1, open_form_n)) if open_form_n > 0 else 0.0,
                "open_form_parse_other_rate": (float(open_form_parse_other) / max(1, open_form_n)) if open_form_n > 0 else 0.0,
                "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            })
        except Exception as e:
            err_type = type(e).__name__
            group_error_types[err_type] += 1
            reasons.append(f"preflight_group_error:{err_type}:{str(e)}")
            error_samples.append({
                "pair_id": s.get("pair_id"),
                "image_id": s.get("image_id"),
                "image_file": s.get("image_file"),
                "source_image_path": s.get("source_image_path"),
                "image_root": image_root,
                "resolved_abspath": resolve_image_abspath(s, image_root=image_root),
                "question": s.get("question"),
                "error_type": err_type,
                "error_message": str(e),
            })

    if not group_records:
        infra_error_detected = bool(error_samples) and all(
            _is_no_cuda_error(es.get("error_message", "")) for es in error_samples
        )
        if infra_error_detected:
            reasons = ["infra_error:no_cuda_gpu"]
        return {
            "check_name": check_name,
            "passed": False,
            "reward_backend": backend,
            "reasons": reasons or ["no_valid_groups_in_preflight"],
            "prompt_side_reward_detected": backend == "logodds_ate",
            "error_samples": error_samples,
            "n_group_errors": int(len(error_samples)),
            "group_error_types": dict(group_error_types),
            "infra_error_detected": infra_error_detected,
        }

    raw_nonconstant_rate = float(sum(1 for g in group_records if not g["raw_identical"]) / len(group_records))
    mean_raw_std = float(np.mean([g["raw_std"] for g in group_records]))
    parse_identical_reward_nonidentical_ratio = float(
        sum(1 for g in group_records if g["parse_identical"] and g["reward_nonidentical"]) / max(1, sum(1 for g in group_records if g["parse_identical"]))
    )
    final_identical_reward_nonidentical_ratio = float(
        sum(1 for g in group_records if g["final_identical"] and g["reward_nonidentical"]) / max(1, sum(1 for g in group_records if g["final_identical"]))
    )

    parse_other_rate_mean = float(np.mean([g["parse_other_rate"] for g in group_records]))
    answer_score_mean = float(np.mean([g.get("answer_score_mean", 0.0) for g in group_records]))
    total_reward_mean = float(np.mean([g.get("total_reward_mean", 0.0) for g in group_records]))
    open_form_total_n = int(sum(g.get("open_form_n", 0) for g in group_records))
    open_form_extract_fail_rate = float(np.mean([
        g.get("open_form_extract_fail_rate", 0.0) for g in group_records if g.get("open_form_n", 0) > 0
    ])) if open_form_total_n > 0 else 0.0
    open_form_parse_other_rate = float(np.mean([
        g.get("open_form_parse_other_rate", 0.0) for g in group_records if g.get("open_form_n", 0) > 0
    ])) if open_form_total_n > 0 else 0.0
    mean_len = float(np.mean([g["mean_length"] for g in group_records]))
    mean_std = float(np.std([g["mean_length"] for g in group_records]))
    length_template_risk = bool(mean_std < 1e-6 or mean_len <= 0.0)

    reward_contract_ok = (
        (answer_text_source or "").strip().lower() == "final_answer"
        and (policy_logprob_scope or "").strip().lower() == "final_answer"
    )

    if backend == "logodds_ate":
        reasons.append("prompt_side_reward_detected")
        reasons.append("prompt-side/sample-level reward does not satisfy group ranking contract")
    if raw_nonconstant_rate < 0.2:
        reasons.append(f"raw_reward_nonconstant_group_rate_too_low:{raw_nonconstant_rate:.3f}")
    if parse_other_rate_mean > 0.8 and open_form_total_n == 0:
        reasons.append(f"high_parse_failure_dependence:{parse_other_rate_mean:.3f}")
    if length_template_risk:
        reasons.append("possible_template_or_length_dominance")
    if not reward_contract_ok:
        reasons.append("reward_contract_mismatch: answer_text_source/final_answer or policy_logprob_scope/final_answer expected")

    infra_error_detected = bool(error_samples) and all(
        _is_no_cuda_error(es.get("error_message", "")) for es in error_samples
    )
    if infra_error_detected and "infra_error:no_cuda_gpu" not in reasons:
        reasons.append("infra_error:no_cuda_gpu")

    return {
        "check_name": check_name,
        "passed": len(reasons) == 0,
        "reward_backend": backend,
        "reward_contract_ok": reward_contract_ok,
        "prompt_side_reward_detected": backend == "logodds_ate",
        "nonconstant_group_ratio": raw_nonconstant_rate,
        "raw_reward_nonconstant_group_rate": raw_nonconstant_rate,
        "mean_raw_reward_std": mean_raw_std,
        "answer_score_mean": answer_score_mean,
        "total_reward_mean": total_reward_mean,
        "parse_identical_but_reward_nonidentical_ratio": parse_identical_reward_nonidentical_ratio,
        "final_answer_identical_but_reward_nonidentical_ratio": final_identical_reward_nonidentical_ratio,
        "open_form_n": open_form_total_n,
        "open_form_extract_fail_rate": open_form_extract_fail_rate,
        "open_form_parse_other_rate": open_form_parse_other_rate,
        "error_samples": error_samples,
        "n_group_errors": int(len(error_samples)),
        "group_error_types": dict(group_error_types),
        "infra_error_detected": infra_error_detected,
        "reasons": reasons,
    }


# ============================================================
# Check A（不变）
# ============================================================

def check_a_distribution(reward_fn, samples, image_root, output_dir):
    print("\n" + "=" * 60)
    print("  Check A: Reward 数值分布 (LogOddsATEReward)")
    print("=" * 60)

    records = []
    for s in tqdm(samples, desc="Check A", ncols=80):
        try:
            img = open_sample_image(s, image_root=image_root)
            r = reward_fn.compute(img, s["question"], s["target_bbox"],
                                  s.get("image_width"), s.get("image_height"))
            bbox = s["target_bbox"]
            iw = s.get("image_width", img.size[0])
            ih = s.get("image_height", img.size[1])
            records.append({
                "reward": r["reward"],
                "bbox_area_ratio": (bbox[2] * bbox[3]) / max(1, iw * ih),
                "n_tgt_tokens": r.get("n_tgt_tokens", 0),
                "error": r.get("error"),
            })
        except Exception as e:
            records.append({"reward": 0.0, "error": repr(e)})

    with open(os.path.join(output_dir, "check_a_rewards.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in records if r.get("error") is None]
    rewards = [r["reward"] for r in ok]
    areas = [r["bbox_area_ratio"] for r in ok]
    n_toks = [r.get("n_tgt_tokens", 0) for r in ok]
    n_ok, n_nan = len(ok), sum(1 for v in rewards if np.isnan(v) or np.isinf(v))

    corr_a, corr_t = 0.0, 0.0
    if n_ok > 10:
        from scipy.stats import pearsonr
        corr_a, _ = pearsonr(areas, rewards)
        vt = [(t, r) for t, r in zip(n_toks, rewards) if t > 0]
        if len(vt) > 10:
            corr_t, _ = pearsonr([t for t, _ in vt], [r for _, r in vt])

    report = {"n_ok": n_ok, "n_nan_inf": n_nan,
              "reward_mean": float(np.mean(rewards)) if rewards else 0,
              "bbox_area_corr": corr_a, "n_tgt_tokens_corr": corr_t}
    issues = []
    if n_nan > 0:
        issues.append(f"{n_nan} NaN/Inf")
    if abs(corr_a) > 0.5:
        issues.append(f"面积相关 r={corr_a:.3f}")
    if abs(corr_t) > 0.5:
        issues.append(f"token数相关 r={corr_t:.3f}")
    report["passed"] = len(issues) == 0
    report["issues"] = issues

    print(f"\n  n={n_ok} mean={report['reward_mean']:.4f} "
          f"area_r={corr_a:.3f} tok_r={corr_t:.3f}")
    print(f"  {'✓' if report['passed'] else '✗ ' + '; '.join(issues)}")
    return report


# ============================================================
# Check B（v3 面积诊断，保留不变）
# ============================================================

def _random_bbox(img_w, img_h, seed=0):
    rng = random.Random(seed)
    ratio = rng.uniform(0.05, 0.15)
    side = (ratio * img_w * img_h) ** 0.5
    w = min(side, img_w * 0.5)
    h = min(side, img_h * 0.5)
    x = rng.uniform(0, max(1, img_w - w))
    y = rng.uniform(0, max(1, img_h - h))
    return [x, y, w, h]


def _same_image_wrong_bbox(samples, current, rng):
    cur_cat = current.get("target_category", "")
    cands = [s for s in samples
             if s.get("image_file") == current.get("image_file")
             and s.get("target_category", "") != cur_cat
             and s.get("gt_present") is True]
    if not cands:
        return None, None
    c = rng.choice(cands)
    return c["target_bbox"], c.get("target_category")


def _area_matched_wrong_bbox(img_w, img_h, correct_bbox, rng):
    c_area = max(1.0, correct_bbox[2] * correct_bbox[3])
    target_area = c_area * rng.uniform(0.8, 1.2)
    c_ratio = correct_bbox[2] / max(1.0, correct_bbox[3])
    w = min(math.sqrt(target_area * c_ratio), img_w * 0.9)
    h = min(target_area / max(1.0, w), img_h * 0.9)
    w, h = max(w, 1.0), max(h, 1.0)
    for _ in range(20):
        x = rng.uniform(0, max(1, img_w - w))
        y = rng.uniform(0, max(1, img_h - h))
        ix = max(0, min(x+w, correct_bbox[0]+correct_bbox[2]) - max(x, correct_bbox[0]))
        iy = max(0, min(y+h, correct_bbox[1]+correct_bbox[3]) - max(y, correct_bbox[1]))
        if ix * iy / max(1.0, w*h + c_area - ix*iy) < 0.3:
            return [x, y, w, h]
    cx = correct_bbox[0] + correct_bbox[2] / 2
    cy = correct_bbox[1] + correct_bbox[3] / 2
    x = max(0, img_w - w) if cx < img_w / 2 else 0
    y = max(0, img_h - h) if cy < img_h / 2 else 0
    return [x, y, w, h]


def _norm_reward(reward, n_tokens):
    n = max(1, n_tokens)
    return {"raw": reward, "per_sqrt_tok": reward / math.sqrt(n),
            "per_log_tok": reward / math.log(1 + n)}


def check_b_bbox_comparison(
    reward_fn, samples, image_root, output_dir,
    max_samples=200, seed=42, n_failure_export=50,
):
    """Check B: BBox 定位诊断（LogOddsATEReward，prompt-level 静态信号）。"""
    print("\n" + "=" * 60)
    print("  Check B: BBox Specificity (LogOddsATEReward, 静态)")
    print("=" * 60)
    rng = random.Random(seed)
    pos = [s for s in samples if s.get("gt_present")]
    if len(pos) > max_samples:
        pos = rng.sample(pos, max_samples)

    records = []
    for s in tqdm(pos, desc="Check B", ncols=80):
        try:
            img = open_sample_image(s, image_root=image_root)
            iw = s.get("image_width", img.size[0])
            ih = s.get("image_height", img.size[1])
            cb = s["target_bbox"]

            r_c = reward_fn.compute(img, s["question"], cb, iw, ih)
            c_rew, c_ntok = r_c["reward"], r_c.get("n_tgt_tokens", 0)

            rb = _random_bbox(iw, ih, hash(s["image_file"]) % 100000)
            r_r = reward_fn.compute(img, s["question"], rb, iw, ih)
            rand_rew, rand_ntok = r_r["reward"], r_r.get("n_tgt_tokens", 0)

            si_b, si_cat = _same_image_wrong_bbox(samples, s, rng)
            if si_b:
                r_si = reward_fn.compute(img, s["question"], si_b, iw, ih)
                si_rew, si_ntok = r_si["reward"], r_si.get("n_tgt_tokens", 0)
            else:
                si_rew = si_ntok = 0.0
                si_cat = None

            am_b = _area_matched_wrong_bbox(iw, ih, cb, rng)
            r_am = reward_fn.compute(img, s["question"], am_b, iw, ih)
            am_rew, am_ntok = r_am["reward"], r_am.get("n_tgt_tokens", 0)

            cn = _norm_reward(c_rew, c_ntok)
            rn = _norm_reward(rand_rew, rand_ntok)
            sn = _norm_reward(si_rew, si_ntok) if si_b else None
            an = _norm_reward(am_rew, am_ntok)

            rec = {
                "image_file": s["image_file"], "image_id": s.get("image_id"),
                "question": s["question"], "target_category": s.get("target_category"),
                "correct_bbox": cb, "correct_area": _bbox_area(cb),
                "correct_n_tokens": c_ntok, "correct_reward": c_rew,
                "random_reward": rand_rew, "random_n_tokens": rand_ntok,
                "correct_gt_random": c_rew > rand_rew,
                "correct_gt_random_sqrt": cn["per_sqrt_tok"] > rn["per_sqrt_tok"],
                "correct_gt_random_log": cn["per_log_tok"] > rn["per_log_tok"],
                "si_available": si_b is not None,
                "si_category": si_cat, "si_bbox": si_b,
                "si_area": _bbox_area(si_b) if si_b else None,
                "si_n_tokens": si_ntok if si_b else None,
                "si_reward": si_rew if si_b else None,
                "correct_gt_si": c_rew > si_rew if si_b else None,
                "correct_gt_si_sqrt": (cn["per_sqrt_tok"] > sn["per_sqrt_tok"]) if sn else None,
                "correct_gt_si_log": (cn["per_log_tok"] > sn["per_log_tok"]) if sn else None,
                "area_ratio_si": (_bbox_area(si_b) / max(1, _bbox_area(cb))) if si_b else None,
                "token_ratio_si": (si_ntok / max(1, c_ntok)) if si_b else None,
                "am_bbox": am_b, "am_area": _bbox_area(am_b),
                "am_n_tokens": am_ntok, "am_reward": am_rew,
                "correct_gt_am": c_rew > am_rew,
                "correct_gt_am_sqrt": cn["per_sqrt_tok"] > an["per_sqrt_tok"],
                "correct_gt_am_log": cn["per_log_tok"] > an["per_log_tok"],
            }
            records.append(rec)
        except Exception as e:
            records.append({"error": repr(e)})

    with open(os.path.join(output_dir, "check_b_bbox_diagnosis.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid = [r for r in records if "error" not in r]
    si_valid = [r for r in valid if r["si_available"]]
    n, n_si = len(valid), len(si_valid)
    if n == 0:
        return {"passed": False, "issues": ["无有效样本"], "n_valid": 0}

    def _rate(recs, key):
        vs = [r[key] for r in recs if r.get(key) is not None]
        return sum(1 for v in vs if v) / max(1, len(vs)), len(vs)

    gt_rand, _ = _rate(valid, "correct_gt_random")
    gt_si, _ = _rate(si_valid, "correct_gt_si")
    gt_am, _ = _rate(valid, "correct_gt_am")
    gt_rand_s, _ = _rate(valid, "correct_gt_random_sqrt")
    gt_si_s, _ = _rate(si_valid, "correct_gt_si_sqrt")
    gt_am_s, _ = _rate(valid, "correct_gt_am_sqrt")
    gt_rand_l, _ = _rate(valid, "correct_gt_random_log")
    gt_si_l, _ = _rate(si_valid, "correct_gt_si_log")
    gt_am_l, _ = _rate(valid, "correct_gt_am_log")

    c_areas = np.array([r["correct_area"] for r in valid])
    si_areas = np.array([r["si_area"] for r in si_valid]) if si_valid else np.array([])
    si_bigger_rate = float((si_areas > c_areas[:len(si_areas)]).mean()) if len(si_areas) > 0 else 0

    print(f"\n  n={n} (同图wrong={n_si})")
    print(f"\n  原始:     random={gt_rand:.3f}  same_img={gt_si:.3f}  area_match={gt_am:.3f}")
    print(f"  /sqrt(t): random={gt_rand_s:.3f}  same_img={gt_si_s:.3f}  area_match={gt_am_s:.3f}")
    print(f"  /log(t):  random={gt_rand_l:.3f}  same_img={gt_si_l:.3f}  area_match={gt_am_l:.3f}")
    if len(si_areas) > 0:
        print(f"\n  same_image 面积>correct: {si_bigger_rate:.1%}")

    for label, pool, key, bkey in [
        ("same_image", si_valid, "correct_gt_si", "si_reward"),
        ("area_matched", valid, "correct_gt_am", "am_reward"),
    ]:
        fails = sorted(
            [r for r in pool if r.get(key) is False],
            key=lambda r: (r.get(bkey, 0) - r.get("correct_reward", 0)), reverse=True,
        )[:n_failure_export]
        if fails:
            p = os.path.join(output_dir, f"check_b_failures_{label}.jsonl")
            with open(p, "w") as f:
                for r in fails:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  失败导出: {p} ({len(fails)})")

    issues = []
    if gt_rand < 0.5:
        issues.append(f"correct>random = {gt_rand:.3f}")
    if gt_am < 0.5:
        issues.append(f"correct>area_matched = {gt_am:.3f}")

    report = {
        "n_valid": n, "n_same_image": n_si,
        "rate_correct_gt_random": gt_rand,
        "rate_correct_gt_same_image": gt_si,
        "rate_correct_gt_area_matched": gt_am,
        "rate_correct_gt_random_sqrt": gt_rand_s,
        "rate_correct_gt_same_image_sqrt": gt_si_s,
        "rate_correct_gt_area_matched_sqrt": gt_am_s,
        "si_area_bigger_rate": si_bigger_rate,
        "passed": len(issues) == 0, "issues": issues,
    }
    if gt_si < 0.5 and n_si > 0:
        report["warning_same_image"] = f"correct>same_image={gt_si:.3f}（面积偏置）"
    print(f"  {'✓' if report['passed'] else '✗ ' + '; '.join(issues)}")
    return report


# ============================================================
# Check C1（不变）
# ============================================================

def check_c1_question_wording(reward_fn, samples, image_root, output_dir,
                               max_samples=100, seed=42):
    print("\n" + "=" * 60)
    print("  Check C1: Question Wording Sensitivity (prompt-level)")
    print("=" * 60)
    rng = random.Random(seed)
    pos = [s for s in samples if s.get("gt_present")]
    if len(pos) > max_samples:
        pos = rng.sample(pos, max_samples)
    garbage_qs = [
        "Answer yes or no.", "Is there something in this image? Answer yes or no.",
        "abc def ghi. Answer yes or no.", "What is the meaning of life? Answer yes or no.",
    ]
    records = []
    for s in tqdm(pos, desc="Check C1", ncols=80):
        try:
            img = open_sample_image(s, image_root=image_root)
            iw, ih = s.get("image_width", img.size[0]), s.get("image_height", img.size[1])
            r_real = reward_fn.compute(img, s["question"], s["target_bbox"], iw, ih)
            garb = [reward_fn.compute(img, gq, s["target_bbox"], iw, ih)["reward"] for gq in garbage_qs]
            records.append({"reward_real": r_real["reward"], "max_garbage": max(garb),
                           "real_gt": r_real["reward"] > max(garb)})
        except Exception:
            pass
    n = len(records)
    if n == 0:
        return {"passed": False, "issues": ["无有效样本"], "n_valid": 0}
    rate = sum(1 for r in records if r["real_gt"]) / n
    report = {"n_valid": n, "rate_real_gt_garbage": rate,
              "passed": rate >= 0.5, "issues": [] if rate >= 0.5 else [f"real>garbage={rate:.3f}"]}
    print(f"\n  real>garbage: {rate:.3f}")
    print(f"  {'✓' if report['passed'] else '✗ ' + report['issues'][0]}")
    return report


# ============================================================
# Check C2（v5: 兼容 v4 ActionLogProbATEReward 三种 mode）
# ============================================================


def _wrap_structured_response(text: str, structured_response_style: str = "raw_response") -> str:
    """Wrap a short answer / garbage response into a structured two-line response.

    - raw_response: return text as-is
    - task_aware_v1: return:
        Evidence:
        Final answer: <text>
    """
    style = (structured_response_style or "raw_response").strip().lower()
    t = (text or "").strip()
    if style == "task_aware_v1":
        return f"Evidence:\nFinal answer: {t}" if t else "Evidence:\nFinal answer:"
    return t


def _select_answer_text_for_sanity(
    response_text_used: str,
    answer_text_source: str,
) -> Dict[str, Any]:
    part = extract_final_answer_with_prefix(response_text_used)
    extracted = part.get("final_answer") or ""
    extracted_src = part.get("extraction_source") or "fallback_raw"

    src = (answer_text_source or "raw_response").strip().lower()
    if src == "final_answer":
        answer_used = extracted
    else:
        answer_used = response_text_used

    return {
        "answer_text_used": answer_used,
        "extracted_final_answer": extracted,
        "final_answer_extraction_source": extracted_src,
    }

def check_c2_response_hacking(
    action_reward_fn, samples, image_root, output_dir,
    max_samples=100, seed=42,
    prompt_mode: str = "raw_question",
    answer_text_source: str = "raw_response",
    structured_response_style: str = "raw_response",
):
    """Check C2: Response hacking detection (action-level)。

    v5 变更：
    - 兼容 v4 新增字段（delta_ans_margin, score_base, reward_mode 等）
    - 新增 soft_shaping 模式的分组统计
    - 保留旧的 gating 检查逻辑
    """
    print("\n" + "=" * 60)
    print(f"  Check C2: Response Hacking (action-level, reward_mode={action_reward_fn.reward_mode})")
    print(f"  stable_io: prompt_mode={prompt_mode} answer_text_source={answer_text_source} structured_response_style={structured_response_style}")
    print("=" * 60)

    rng = random.Random(seed)
    if len(samples) > max_samples:
        samples = rng.sample(samples, max_samples)

    garbage_unparseable = ["the the the", "abc def ghi", "42", "", "yes no yes no"]
    garbage_refusal = ["I cannot determine.", "not sure", "maybe"]

    records = []
    debug_records = []
    top_level_errors: List[str] = []

    for s in tqdm(samples, desc="Check C2", ncols=80):
        try:
            img = open_sample_image(s, image_root=image_root)
            gt_present = s.get("gt_present")
            task_type = (s.get("task_type") or "").strip().lower()
            gt_answer_text = (s.get("answer") or "").strip()
            gt_yn = parse_yesno(gt_answer_text)
            is_yesno_task = (task_type == "existence") or (gt_yn in ("yes", "no"))
            if is_yesno_task:
                if gt_yn in ("yes", "no"):
                    correct_answer = gt_yn
                elif gt_present is not None:
                    correct_answer = "yes" if gt_present else "no"
                else:
                    correct_answer = "yes"
            else:
                correct_answer = gt_answer_text if gt_answer_text else "unknown"

            if is_yesno_task:
                if correct_answer == "yes":
                    garbage_parseable_wrong = ["no", "no no no no"]
                else:
                    garbage_parseable_wrong = ["yes", "yes yes yes yes"]
            else:
                garbage_parseable_wrong = ["irrelevant", "wrong object"]

            response_text_used = _wrap_structured_response(correct_answer, structured_response_style)
            ans_sel = _select_answer_text_for_sanity(response_text_used, answer_text_source)
            r_correct = action_reward_fn.compute(
                img, s["question"], s["target_bbox"],
                response_text=response_text_used,
                answer_text=ans_sel["answer_text_used"],
                gt_answer=s.get("answer"), gt_present=gt_present,
                image_width=s.get("image_width"),
                image_height=s.get("image_height"),
                response_text_source="raw_generated_text",
                answer_text_source=("generated_text" if (answer_text_source or "").strip().lower() == "final_answer" else "raw_response"),
                task_type=task_type,
            )
            correct_has_error = r_correct.get("error") is not None

            group_results = {}
            for group_name, group_list in [
                ("unparseable", garbage_unparseable),
                ("parseable_wrong", garbage_parseable_wrong),
                ("refusal", garbage_refusal),
            ]:
                g_results = []
                for gr in group_list:
                    response_text_used = _wrap_structured_response(gr, structured_response_style)
                    ans_sel = _select_answer_text_for_sanity(response_text_used, answer_text_source)
                    r_g = action_reward_fn.compute(
                        img, s["question"], s["target_bbox"],
                        response_text=response_text_used,
                        answer_text=ans_sel["answer_text_used"],
                        gt_answer=s.get("answer"), gt_present=gt_present,
                        image_width=s.get("image_width"),
                        image_height=s.get("image_height"),
                        response_text_source="raw_generated_text",
                        answer_text_source=("generated_text" if (answer_text_source or "").strip().lower() == "final_answer" else "raw_response"),
                        task_type=task_type,
                    )
                    g_results.append({
                        "response": gr,
                        "response_text_used": response_text_used,
                        "answer_text_used": ans_sel.get("answer_text_used"),
                        "extracted_final_answer": ans_sel.get("extracted_final_answer"),
                        "final_answer_extraction_source": ans_sel.get("final_answer_extraction_source"),
                        "reward": r_g["reward"],
                        "delta_logprob": r_g["delta_logprob"],
                        "parsed_answer": r_g["parsed_answer"],
                        "is_correct": r_g["is_correct"],
                        "error": r_g.get("error"),
                        "has_visual_targets": r_g.get("has_visual_targets", False),
                        "mean_abs_token_lp_diff": r_g.get("mean_abs_token_lp_diff", 0),
                        # v4 新增字段
                        "delta_ans_margin": r_g.get("delta_ans_margin", 0),
                        "score_base": r_g.get("score_base", 0),
                        "score_resp": r_g.get("score_resp", 0),
                        "score_ans": r_g.get("score_ans", 0),
                        "reward_mode": r_g.get("reward_mode"),
                        "reward_bias": r_g.get("reward_bias", 0),
                        "reward_scale": r_g.get("reward_scale", 0),
                    })
                group_results[group_name] = g_results

            def _group_stats(glist):
                valid_g = [g for g in glist if g.get("error") is None]
                error_g = [g for g in glist if g.get("error") is not None]
                if not valid_g:
                    return {"n_valid": 0, "n_error": len(error_g),
                            "max_reward": None, "max_delta": None,
                            "mean_reward": None, "mean_delta": None,
                            "mean_score_base": None, "mean_delta_ans_margin": None}
                rews = [g["reward"] for g in valid_g]
                deltas = [g["delta_logprob"] for g in valid_g]
                sbs = [g.get("score_base", 0) for g in valid_g]
                dams = [g.get("delta_ans_margin", 0) for g in valid_g]
                return {
                    "n_valid": len(valid_g), "n_error": len(error_g),
                    "max_reward": max(rews), "max_delta": max(deltas),
                    "mean_reward": float(np.mean(rews)),
                    "mean_delta": float(np.mean(deltas)),
                    "mean_score_base": float(np.mean(sbs)),
                    "mean_delta_ans_margin": float(np.mean(dams)),
                }

            group_stats = {gn: _group_stats(gl) for gn, gl in group_results.items()}

            rec = {
                "image_file": s["image_file"],
                "gt_present": gt_present,
                "correct_answer": correct_answer,
                "response_text_used": response_text_used,
                "answer_text_used": ans_sel.get("answer_text_used"),
                "extracted_final_answer": ans_sel.get("extracted_final_answer"),
                "final_answer_extraction_source": ans_sel.get("final_answer_extraction_source"),
                "correct_reward": r_correct["reward"],
                "correct_delta": r_correct["delta_logprob"],
                "correct_error": r_correct.get("error"),
                "correct_has_visual_targets": r_correct.get("has_visual_targets", False),
                "correct_mean_abs_diff": r_correct.get("mean_abs_token_lp_diff", 0),
                # v4 新增
                "correct_score_base": r_correct.get("score_base", 0),
                "correct_delta_ans_margin": r_correct.get("delta_ans_margin", 0),
                "reward_mode": r_correct.get("reward_mode"),
            }
            for gn, gs in group_stats.items():
                for k, v in gs.items():
                    rec[f"{gn}_{k}"] = v

            up_stats = group_stats["unparseable"]
            if up_stats["max_reward"] is not None:
                rec["gating_ok_unparseable"] = up_stats["max_reward"] <= 0
            else:
                rec["gating_ok_unparseable"] = None

            if up_stats["max_delta"] is not None and not correct_has_error:
                rec["delta_ok_unparseable"] = r_correct["delta_logprob"] > up_stats["max_delta"]
            else:
                rec["delta_ok_unparseable"] = None

            records.append(rec)

            debug_records.append({
                "image_file": s["image_file"],
                "gt_present": gt_present,
                "response_text_used": response_text_used,
                "answer_text_used": ans_sel.get("answer_text_used"),
                "extracted_final_answer": ans_sel.get("extracted_final_answer"),
                "final_answer_extraction_source": ans_sel.get("final_answer_extraction_source"),
                "correct_result": {k: v for k, v in r_correct.items()
                                   if k != "response_text"},
                "garbage_groups": {
                    gn: [{k: v for k, v in g.items()} for g in gl]
                    for gn, gl in group_results.items()
                },
            })

        except Exception as e:
            top_level_errors.append(repr(e))
            records.append({"error": repr(e)})

    with open(os.path.join(output_dir, "check_c2_response_hacking.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "check_c2_debug.jsonl"), "w") as f:
        for r in debug_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid = [r for r in records if "error" not in r]
    n = len(valid)
    if n == 0:
        return {"passed": False, "issues": ["无有效样本"], "n_valid": 0}

    n_correct_error = sum(1 for r in valid if r.get("correct_error") is not None)
    n_correct_no_vis = sum(1 for r in valid if not r.get("correct_has_visual_targets", True))
    total_garb_errors = sum(
        r.get(f"{gn}_n_error", 0) for r in valid for gn in ["unparseable", "parseable_wrong", "refusal"]
    )
    total_garb_valid = sum(
        r.get(f"{gn}_n_valid", 0) for r in valid for gn in ["unparseable", "parseable_wrong", "refusal"]
    )
    internal_error_rate = (n_correct_error + total_garb_errors) / max(1, n + total_garb_valid + total_garb_errors)

    up_gating_vals = [r["gating_ok_unparseable"] for r in valid if r.get("gating_ok_unparseable") is not None]
    up_delta_vals = [r["delta_ok_unparseable"] for r in valid if r.get("delta_ok_unparseable") is not None]

    gating_ok_rate = sum(1 for v in up_gating_vals if v) / max(1, len(up_gating_vals))
    delta_ok_rate = sum(1 for v in up_delta_vals if v) / max(1, len(up_delta_vals))

    group_summary = {}
    for gn in ["unparseable", "parseable_wrong", "refusal"]:
        rews = [r[f"{gn}_mean_reward"] for r in valid if r.get(f"{gn}_mean_reward") is not None]
        deltas = [r[f"{gn}_mean_delta"] for r in valid if r.get(f"{gn}_mean_delta") is not None]
        sbs = [r[f"{gn}_mean_score_base"] for r in valid if r.get(f"{gn}_mean_score_base") is not None]
        dams = [r[f"{gn}_mean_delta_ans_margin"] for r in valid if r.get(f"{gn}_mean_delta_ans_margin") is not None]
        errs = sum(r.get(f"{gn}_n_error", 0) for r in valid)
        group_summary[gn] = {
            "mean_reward": float(np.mean(rews)) if rews else None,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "mean_score_base": float(np.mean(sbs)) if sbs else None,
            "mean_delta_ans_margin": float(np.mean(dams)) if dams else None,
            "total_internal_errors": errs,
        }

    correct_deltas = [r["correct_delta"] for r in valid if r.get("correct_error") is None]
    mean_correct_delta = float(np.mean(correct_deltas)) if correct_deltas else 0

    # ── v5 新增：soft_shaping 模式的分组统计 ──
    soft_shaping_stats = {}
    reward_mode = action_reward_fn.reward_mode
    if reward_mode == "soft_shaping":
        # 按 correctness 分组统计
        correct_samples = [r for r in valid if r.get("correct_error") is None]
        correct_score_bases = [r.get("correct_score_base", 0) for r in correct_samples]
        correct_rewards = [r.get("correct_reward", 0) for r in correct_samples]
        correct_dam = [r.get("correct_delta_ans_margin", 0) for r in correct_samples]

        # "correct" = 正确回答的样本；"wrong" = parseable_wrong 组平均；"other" = unparseable 组平均
        pw_rewards = [r.get("parseable_wrong_mean_reward") for r in valid
                      if r.get("parseable_wrong_mean_reward") is not None]
        pw_sbs = [r.get("parseable_wrong_mean_score_base") for r in valid
                  if r.get("parseable_wrong_mean_score_base") is not None]
        up_rewards = [r.get("unparseable_mean_reward") for r in valid
                      if r.get("unparseable_mean_reward") is not None]
        up_sbs = [r.get("unparseable_mean_score_base") for r in valid
                  if r.get("unparseable_mean_score_base") is not None]

        soft_shaping_stats = {
            "correct_mean_score_base": float(np.mean(correct_score_bases)) if correct_score_bases else None,
            "correct_mean_reward": float(np.mean(correct_rewards)) if correct_rewards else None,
            "correct_mean_delta_ans_margin": float(np.mean(correct_dam)) if correct_dam else None,
            "correct_score_base_positive_rate": (
                sum(1 for v in correct_score_bases if v > 0) / max(1, len(correct_score_bases))
            ) if correct_score_bases else None,
            "wrong_mean_score_base": float(np.mean(pw_sbs)) if pw_sbs else None,
            "wrong_mean_reward": float(np.mean(pw_rewards)) if pw_rewards else None,
            "other_mean_score_base": float(np.mean(up_sbs)) if up_sbs else None,
            "other_mean_reward": float(np.mean(up_rewards)) if up_rewards else None,
            "wrong_overall_negative": bool(np.mean(pw_rewards) < 0) if pw_rewards else None,
            "other_overall_negative": bool(np.mean(up_rewards) < 0) if up_rewards else None,
        }

    report = {
        "n_valid": n,
        "reward_mode": reward_mode,
        "internal_error_rate": internal_error_rate,
        "n_correct_error": n_correct_error,
        "n_correct_no_visual_targets": n_correct_no_vis,
        "total_garbage_internal_errors": total_garb_errors,
        "gating_ok_rate_unparseable": gating_ok_rate,
        "delta_ok_rate_unparseable": delta_ok_rate,
        "mean_correct_delta": mean_correct_delta,
        "group_summary": group_summary,
    }

    no_cuda_infra_error = False
    if top_level_errors and all(_is_no_cuda_error(x) for x in top_level_errors):
        no_cuda_infra_error = True
    if n_correct_error == n and n > 0 and total_garb_errors > 0:
        all_internal_msgs: List[str] = []
        for r in valid:
            ce = r.get("correct_error")
            if ce is not None:
                all_internal_msgs.append(str(ce))
        if all_internal_msgs and all(_is_no_cuda_error(x) for x in all_internal_msgs):
            no_cuda_infra_error = True

    if soft_shaping_stats:
        report["soft_shaping_stats"] = soft_shaping_stats

    issues = []
    if no_cuda_infra_error:
        issues.append("infra error: no CUDA GPUs are available; C2 cannot run on current host")
    elif internal_error_rate > 0.3:
        issues.append(
            f"implementation error rate = {internal_error_rate:.1%} > 30%. "
            f"C2 结论不可信——失败主要是实现问题，不是信号问题。"
        )
    elif reward_mode == "legacy_hard_gate":
        # 旧模式下保留原有 gating 检查
        if gating_ok_rate < 0.8 and len(up_gating_vals) > 5:
            issues.append(f"unparseable gating 拦截率 = {gating_ok_rate:.1%} < 80%")
    elif reward_mode == "soft_shaping":
        # soft shaping 模式：检查 wrong/other 总体是否保持负 reward
        if soft_shaping_stats.get("wrong_overall_negative") is False:
            issues.append("soft_shaping: wrong 组 mean_reward >= 0（应该为负）")
        if soft_shaping_stats.get("other_overall_negative") is False:
            issues.append("soft_shaping: other/unparseable 组 mean_reward >= 0（应该为负）")

    report["passed"] = len(issues) == 0
    report["issues"] = issues
    report["infra_error_detected"] = no_cuda_infra_error

    print(f"\n  n={n}  reward_mode={reward_mode}  internal_error_rate={internal_error_rate:.1%}")
    print(f"  correct: n_error={n_correct_error} n_no_vis={n_correct_no_vis} "
          f"mean_delta={mean_correct_delta:.4f}")
    for gn, gs in group_summary.items():
        mr = f"{gs['mean_reward']:.4f}" if gs['mean_reward'] is not None else "N/A"
        md = f"{gs['mean_delta']:.4f}" if gs['mean_delta'] is not None else "N/A"
        msb = f"{gs['mean_score_base']:.4f}" if gs.get('mean_score_base') is not None else "N/A"
        print(f"  {gn:18s}: mean_reward={mr}  mean_delta={md}  score_base={msb}  errors={gs['total_internal_errors']}")
    print(f"  unparseable gating OK: {gating_ok_rate:.1%}  delta OK: {delta_ok_rate:.1%}")

    if soft_shaping_stats:
        print(f"\n  ── soft_shaping 分组统计 ──")
        print(f"  correct: mean_score_base={soft_shaping_stats.get('correct_mean_score_base', 'N/A')}")
        print(f"           mean_reward={soft_shaping_stats.get('correct_mean_reward', 'N/A')}")
        print(f"           score_base>0 rate={soft_shaping_stats.get('correct_score_base_positive_rate', 'N/A')}")
        print(f"           mean_delta_ans_margin={soft_shaping_stats.get('correct_mean_delta_ans_margin', 'N/A')}")
        print(f"  wrong:   mean_score_base={soft_shaping_stats.get('wrong_mean_score_base', 'N/A')}")
        print(f"           mean_reward={soft_shaping_stats.get('wrong_mean_reward', 'N/A')}")
        print(f"           overall_negative={soft_shaping_stats.get('wrong_overall_negative', 'N/A')}")
        print(f"  other:   mean_score_base={soft_shaping_stats.get('other_mean_score_base', 'N/A')}")
        print(f"           mean_reward={soft_shaping_stats.get('other_mean_reward', 'N/A')}")
        print(f"           overall_negative={soft_shaping_stats.get('other_overall_negative', 'N/A')}")

    print(f"  {'✓' if report['passed'] else '✗ ' + '; '.join(issues)}")
    return report


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Reward Sanity Check (v5)")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--vqa_file", default="")
    ap.add_argument("--coco_image_dir", default="")
    ap.add_argument("--dataset_name", default="legacy_vqa", choices=["legacy_vqa", "vg_brutal"])
    ap.add_argument("--dataset_file", default="", help="vg_brutal 数据文件路径")
    ap.add_argument("--image_root", default="", help="vg_brutal 图片根目录")
    ap.add_argument("--task_type", default="",
                    help="task_type 过滤；空字符串/any/all 表示不过滤")
    ap.add_argument("--data_split", default="all", choices=["all", "train", "val", "probe"],
                    help="VG brutal 数据 split 过滤")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--allow_missing_images", action="store_true",
                    help="允许图片路径不完整时继续执行（默认关闭，VG brutal 会 fail fast）")
    ap.add_argument("--integrity_max_check", type=int, default=0,
                    help="integrity 检查样本数上限；0 表示全量检查")
    ap.add_argument("--output_dir", default="results/sanity")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--max_samples_b", type=int, default=200)
    ap.add_argument("--max_samples_c", type=int, default=100)
    ap.add_argument("--n_failure_export", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_shards", type=int, default=1, help="并行分片总数；>1 时对加载后的样本做 index%num_shards 分片")
    ap.add_argument("--shard_index", type=int, default=0, help="当前分片编号，范围 [0, num_shards)")
    ap.add_argument("--checks", default="",
                    help="留空时按 dataset_name 选择默认模板；可选: a,b,c1,c2,d; 'c'='c1,c2'")
    # stable I/O controls (align with mini_grpo_smoke semantics)
    ap.add_argument("--prompt_mode", default="raw_question",
                    choices=list(PROMPT_MODES),
                    help="兼容 mini_grpo 的 prompt_mode（本脚本不生成，仅用于对齐配置/日志）")
    ap.add_argument("--answer_text_source", default="final_answer",
                    choices=["raw_response", "final_answer"],
                    help="C2 中 compute(..., answer_text=...) 绑定到 raw_response 或 extracted final_answer")
    ap.add_argument("--structured_response_style", default="raw_response",
                    choices=["raw_response", "task_aware_v1"],
                    help="C2 构造的 response_text 是否包装成 Evidence+Final answer 两行结构")
    # v5 新增：reward_mode 参数
    ap.add_argument("--reward_mode", default="soft_shaping",
                    choices=["legacy_hard_gate", "raw_delta", "soft_shaping"],
                    help="ActionLogProbATEReward 的 reward mode（仅影响 C2）")
    ap.add_argument("--reward_backend", default="action_logprob_ate_auto",
                    help="奖励后端选择：auto/action_logprob_ate_auto/action_logprob_ate_supervised/action_logprob_ate_nolabel/response_conditioned_final_answer_ate/logodds_ate；response_conditioned_final_answer_ate 是 no-label backend 的语义化别名；auto 仅对 existence/yes-no 样本使用 supervised")
    ap.add_argument("--tau_resp", type=float, default=0.20)
    ap.add_argument("--tau_ans", type=float, default=1.00)
    ap.add_argument("--alpha_resp", type=float, default=0.70)
    ap.add_argument("--alpha_ans", type=float, default=0.30)
    ap.add_argument("--min_reward", type=float, default=-1.25)
    ap.add_argument("--max_reward", type=float, default=1.00)
    args = ap.parse_args()

    try:
        resolved_paths = resolve_dataset_paths(args)
    except ValueError as e:
        ap.error(str(e))

    dataset_name = resolved_paths["dataset_name"]
    dataset_file = resolved_paths["dataset_file"]
    image_root = resolved_paths["image_root"]
    task_type_filter = resolve_task_type_filter(dataset_name=dataset_name, task_type_arg=args.task_type)
    checks, check_templates = resolve_checks(dataset_name=dataset_name, checks_arg=args.checks)

    if args.train_ratio < 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) > 1.0:
        ap.error("Invalid split ratios: require train_ratio>=0, val_ratio>=0, train_ratio+val_ratio<=1.0")

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    print("=" * 60)
    resolved_reward_backend = (args.reward_backend or "").strip().lower()
    if resolved_reward_backend in ("", "auto"):
        resolved_reward_backend = "action_logprob_ate_auto"
    elif resolved_reward_backend in {"response_conditioned_final_answer_ate", "response_conditioned_final_answer_ate_nolabel"}:
        resolved_reward_backend = "action_logprob_ate_nolabel"

    print(f"  Reward Sanity Check (v5) checks={sorted(checks)} reward_mode={args.reward_mode}")
    print(f"  dataset={dataset_name} dataset_file={dataset_file} image_root={image_root} task_type_filter={task_type_filter or 'ALL'}")
    print(f"  data_split={args.data_split} train_ratio={args.train_ratio} val_ratio={args.val_ratio}")
    print(f"  stable_io: prompt_mode={args.prompt_mode} answer_text_source={args.answer_text_source} structured_response_style={args.structured_response_style}")
    print("=" * 60)

    t0 = time.time()
    processor, model, cfg = load(args.model_dir, args.device, args.dtype)
    print(f"[INFO] 模型加载 ({time.time()-t0:.1f}s)")

    need_logodds = any(c in checks for c in ("a", "b", "c1"))
    logodds_reward = LogOddsATEReward(model, processor, device=args.device) if need_logodds else None
    action_reward = None
    if "c2" in checks or "d" in checks:
        action_reward = ActionLogProbATEReward(
            model, processor, device=args.device,
            reward_mode=args.reward_mode,
            tau_resp=args.tau_resp,
            tau_ans=args.tau_ans,
            alpha_resp=args.alpha_resp,
            alpha_ans=args.alpha_ans,
            min_reward=args.min_reward,
            max_reward=args.max_reward,
        )

    samples, dataset_summary = load_dataset_main_schema(
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        image_root=image_root,
        max_samples=0,
        seed=args.seed,
        task_type=task_type_filter,
        data_split=args.data_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"[INFO] loaded_samples_before_shard={len(samples)}")
    if args.num_shards > 1:
        samples = apply_shard_filter(samples, shard_index=args.shard_index, num_shards=args.num_shards)
        dataset_summary = dict(dataset_summary)
        dataset_summary["num_shards"] = int(args.num_shards)
        dataset_summary["shard_index"] = int(args.shard_index)
        dataset_summary["valid_count_in_shard"] = int(len(samples))
        print(f"[INFO] shard_filter applied: shard={args.shard_index}/{args.num_shards} loaded_samples_after_shard={len(samples)}")
    else:
        print(f"[INFO] loaded_samples={len(samples)}")

    loaded_summary = summarize_loaded_samples(samples)
    resolved_dataset_config = {
        "dataset_name": dataset_name,
        "dataset_file": dataset_file,
        "image_root": image_root,
        "data_split": args.data_split,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "task_type_filter": task_type_filter or "all",
        "loaded_sample_count": loaded_summary["loaded_sample_count"],
        "task_type_distribution": loaded_summary["task_type_distribution"],
        "split_distribution": loaded_summary["split_distribution"],
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
    }

    with open(os.path.join(args.output_dir, "resolved_dataset_config.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_dataset_config, f, indent=2, ensure_ascii=False)

    print(f"[INFO] pair_id_unique={dataset_summary.get('pair_id_unique', 'N/A')}")
    print(f"[INFO] task_type_distribution={loaded_summary['task_type_distribution']}")
    print(f"[INFO] split_distribution={loaded_summary['split_distribution']}")

    dataset_integrity = None
    if dataset_name == "vg_brutal":
        dataset_integrity = check_dataset_integrity(
            samples=samples,
            image_root=image_root,
            max_check=args.integrity_max_check,
        )
        with open(os.path.join(args.output_dir, "dataset_integrity.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_integrity, f, indent=2, ensure_ascii=False)

        ok_rate = dataset_integrity.get("image_path_resolve_ok_rate", 0.0)
        print(
            f"[INFO] integrity: checked={dataset_integrity.get('checked_samples')} "
            f"ok_rate={ok_rate:.3f} missing={dataset_integrity.get('image_path_resolve_missing_count')}"
        )
        if dataset_integrity.get("bad_path_examples"):
            print(f"[INFO] integrity bad_path_examples(head): {dataset_integrity['bad_path_examples'][:3]}")
        if ok_rate < 1.0 and not args.allow_missing_images:
            raise RuntimeError(
                "Dataset integrity check failed: image_path_resolve_ok_rate < 1.0. "
                "Use --allow_missing_images to override."
            )

    with open(os.path.join(args.output_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)

    reports = {}
    all_passed = True

    if "a" in checks:
        r = check_a_distribution(logodds_reward,
                                 random.sample(samples, min(args.max_samples, len(samples))),
                                 image_root, args.output_dir)
        reports["check_a"] = r
        if not r["passed"]: all_passed = False

    if "b" in checks:
        r = check_b_bbox_comparison(logodds_reward, samples, image_root,
                                     args.output_dir, args.max_samples_b, args.seed,
                                     args.n_failure_export)
        reports["check_b"] = r
        if not r["passed"]: all_passed = False

    if "c1" in checks:
        r = check_c1_question_wording(logodds_reward, samples, image_root,
                                       args.output_dir, args.max_samples_c, args.seed)
        reports["check_c1"] = r
        if not r["passed"]: all_passed = False

    if "c2" in checks and action_reward:
        r = check_c2_response_hacking(action_reward, samples, image_root,
                                       args.output_dir, args.max_samples_c, args.seed,
                                       prompt_mode=args.prompt_mode,
                                       answer_text_source=args.answer_text_source,
                                       structured_response_style=args.structured_response_style)
        reports["check_c2"] = r
        if not r["passed"]: all_passed = False

    if "d" in checks and action_reward:
        preflight = run_training_preflight_check(
            model=model,
            processor=processor,
            reward_fn=action_reward,
            samples=samples,
            image_root=image_root,
            reward_backend=resolved_reward_backend,
            prompt_mode=args.prompt_mode,
            answer_text_source=args.answer_text_source,
            policy_logprob_scope="final_answer",
            max_new_tokens=32,
            group_size=4,
            n_groups=min(8, max(3, args.max_samples_c // 20 if args.max_samples_c > 0 else 3)),
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            allow_prompt_side_reward_for_smoke=False,
        )
        reports["check_d_group_reward_contract"] = preflight
        if not preflight.get("passed", False):
            all_passed = False

    reports["all_passed"] = all_passed
    reports["dataset_name"] = dataset_name
    reports["dataset_file"] = dataset_file
    reports["image_root"] = image_root
    reports["task_type_filter"] = task_type_filter
    reports["data_split"] = args.data_split
    reports["train_ratio"] = args.train_ratio
    reports["val_ratio"] = args.val_ratio
    reports["resolved_dataset_config"] = resolved_dataset_config
    reports["check_templates"] = check_templates
    reports["dataset_summary"] = dataset_summary
    if dataset_integrity is not None:
        reports["dataset_integrity"] = dataset_integrity
    reports["elapsed_sec"] = time.time() - t0
    reports["reward_mode"] = args.reward_mode
    reports["reward_backend"] = resolved_reward_backend
    reports["num_shards"] = int(args.num_shards)
    reports["shard_index"] = int(args.shard_index)

    if dataset_name == "vg_brutal":
        c2 = reports.get("check_c2", {})
        d = reports.get("check_d_group_reward_contract", {})
        reports["vg_brutal_open_form_summary"] = {
            "reward_backend": resolved_reward_backend,
            "open_form_answer_score_rule": "task_aware_answer_score(answer_text, gt_answer, task_type, question); final_answer preferred when configured",
            "parse_fail_rate": d.get("open_form_parse_other_rate", None),
            "anti_hacking_passed": c2.get("passed", None),
            "preflight_passed": d.get("passed", None),
        }

    rpt_path = os.path.join(args.output_dir, "sanity_report.json")
    with open(rpt_path, "w") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False, default=str)

    results_path = os.path.join(args.output_dir, "sanity_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for k, v in reports.items():
            if k in ("all_passed", "elapsed_sec", "reward_mode"):
                continue
            f.write(json.dumps({"check": k, "result": v}, ensure_ascii=False, default=str) + "\n")

    print(f"\n{'=' * 60}")
    print(f"  {'✓ 全部通过' if all_passed else '✗ 有 check 未通过'}")
    print(f"  报告: {rpt_path}  耗时: {time.time()-t0:.0f}s")
    print("=" * 60)

    if logodds_reward:
        logodds_reward.cleanup()
    if action_reward:
        action_reward.cleanup()


if __name__ == "__main__":
    main()
