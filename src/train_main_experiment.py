"""NeurIPS 主实验训练入口。

设计原则：
1. 不重写现有训练器，复用 mini_grpo_smoke 里已经验证过的 GRPO 逻辑。
2. 但主入口与 smoke 彻底分开，默认数据/路由/reward/eval 都按主实验 contract 走。
3. 训练前默认先做 offline rerank gate；若 routed_gated_evidence 在离线排序上无优势，会显式告警。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

_EARLY_CUDA_HOLD: List[Any] = []


def _occupy_visible_cuda(reason: str) -> None:
    """Keep the train PID on nvidia-smi compute-apps during dataset/config load.

    ICLR idle needs used<1024MiB and no compute UUID for 30s. Occupy here
    immediately after importing torch, before the heavy trainer imports.
    """
    _EARLY_CUDA_HOLD.clear()
    if not torch.cuda.is_available():
        return
    n = int(torch.cuda.device_count())
    # 2 GiB/visible GPU: fails used<1024 and registers a compute UUID.
    for i in range(n):
        _EARLY_CUDA_HOLD.append(
            torch.zeros((512, 1024, 1024), device=f"cuda:{i}", dtype=torch.float32)
        )
    print(
        f"[train_main] early_cuda_occupy n_visible={n} reason={reason}",
        flush=True,
    )


def _release_early_cuda() -> None:
    _EARLY_CUDA_HOLD.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _early_occupy_enabled() -> bool:
    return str(os.environ.get("EVIDENCE_RL_EARLY_OCCUPY", "0")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


if _early_occupy_enabled():
    _occupy_visible_cuda("import_torch")

from answer_format_utils import (
    infer_task_family,
    task_family_allows_evidence_training,
)
from candidate_gen import run_candidate_generation
from candidate_gen import compute_candidate_diversity
from dataset_adapters import check_dataset_integrity
from mini_grpo_smoke import (
    BalancedFamilyCoverageSampler,
    MiniGRPO,
    _compute_group_diagnostics,
    _flush_fsync,
    _safe_close_file,
    _parse_csv_lower_list,
    _setup_trainable,
    build_family_buckets,
    load_train_dataset,
    resolve_dataset_paths,
    sample_from_family_buckets,
    summarize_loaded_samples,
)
from model_loader import load
from reward.action_logprob_ate import ActionLogProbATEReward
from reward_sanity_check import run_training_preflight_check
from rerank_eval import run_rerank_eval
from verifier_reward import run_reward_scoring
from runtime_parallel import (
    child_env_for_gpu_group,
    merge_jsonl_files,
    runtime_parallel_summary,
    split_visible_devices,
    write_jsonl_rows,
)
from reward.schema import get_schema_summary, infer_active_reward_fields_from_records


def _resolve_split_alias(split_name: str) -> str:
    raw = (split_name or "train").strip().lower()
    if raw == "test":
        return "probe"
    return raw


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") in {"0", ""}


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    if not _is_rank0():
        return
    parent = os.path.dirname(path)
    if parent:
        _ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    if not _is_rank0():
        return
    print(f"[main_train][{_ts()}] {msg}", flush=True)


def _read_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _resolve_effective_train_budget(
    *,
    args: argparse.Namespace,
    family_buckets: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    active_sizes = {
        fam: len(rows)
        for fam, rows in sorted((family_buckets or {}).items())
        if rows
    }
    if not active_sizes:
        raise RuntimeError("cannot resolve train budget: empty active family buckets")

    min_family_size = min(active_sizes.values())
    balanced_epoch_size = int(len(active_sizes) * min_family_size)
    budget_mode = str(args.train_budget_mode or "balanced_epochs").strip().lower()
    if budget_mode == "balanced_epochs":
        effective_steps = int(math.ceil(float(args.train_balanced_epochs) * float(balanced_epoch_size)))
    elif budget_mode == "steps":
        effective_steps = int(args.n_steps)
    else:
        raise ValueError(f"Unsupported train_budget_mode: {args.train_budget_mode}")

    if effective_steps <= 0:
        raise RuntimeError(f"resolved effective_train_steps must be positive, got {effective_steps}")

    return {
        "train_budget_mode": budget_mode,
        "configured_n_steps": int(args.n_steps),
        "train_balanced_epochs": float(args.train_balanced_epochs),
        "effective_train_steps": int(effective_steps),
        "sampling_mode": str(args.sampling_mode or "balanced_no_replacement").strip().lower(),
        "active_families": list(active_sizes.keys()),
        "family_sizes": active_sizes,
        "min_family_size": int(min_family_size),
        "balanced_epoch_size": int(balanced_epoch_size),
        "expected_draws_per_family_per_epoch": int(min_family_size),
        "nominal_draw_coverage_rate": float(effective_steps) / float(max(1, sum(active_sizes.values()))),
    }


def _run_logged_subprocess(
    *,
    label: str,
    cmd: List[str],
    env: Dict[str, str],
    log_dir: str,
) -> Dict[str, Any]:
    """运行一个子进程，并把 stdout/stderr 分别落盘。"""
    _ensure_dir(log_dir)
    stdout_path = os.path.join(log_dir, f"{label}.stdout.log")
    stderr_path = os.path.join(log_dir, f"{label}.stderr.log")
    t0 = time.time()
    with open(stdout_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
        proc = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
    return {
        "label": label,
        "command": cmd,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "returncode": int(proc.returncode),
        "stdout_log": stdout_path,
        "stderr_log": stderr_path,
        "elapsed_sec": round(time.time() - t0, 3),
    }


def _run_logged_subprocess_batch(job_specs: List[Dict[str, Any]], log_dir: str) -> List[Dict[str, Any]]:
    """同一批 shard 并行启动，确保多卡真的同时工作。"""
    _ensure_dir(log_dir)
    items: List[Dict[str, Any]] = []
    for job in job_specs:
        label = str(job["label"])
        cmd = list(job["cmd"])
        env = dict(job["env"])
        stdout_path = os.path.join(log_dir, f"{label}.stdout.log")
        stderr_path = os.path.join(log_dir, f"{label}.stderr.log")
        out_f = open(stdout_path, "w", encoding="utf-8")
        err_f = open(stderr_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        items.append({
            "label": label,
            "command": cmd,
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            "proc": proc,
            "stdout_f": out_f,
            "stderr_f": err_f,
            "stdout_log": stdout_path,
            "stderr_log": stderr_path,
            "t0": time.time(),
        })

    results: List[Dict[str, Any]] = []
    for item in items:
        rc = item["proc"].wait()
        item["stdout_f"].close()
        item["stderr_f"].close()
        results.append({
            "label": item["label"],
            "command": item["command"],
            "cuda_visible_devices": item["cuda_visible_devices"],
            "returncode": int(rc),
            "stdout_log": item["stdout_log"],
            "stderr_log": item["stderr_log"],
            "elapsed_sec": round(time.time() - item["t0"], 3),
        })
    return results


def _write_merged_reward_schema(scored_rows: List[Dict[str, Any]], output_dir: str) -> str:
    """分片合并后补一份总 schema，供 rerank/analysis 继续复用。"""
    active_reward_fields = infer_active_reward_fields_from_records(scored_rows, known_fields_only=True)
    schema_summary = get_schema_summary(
        include_future=False,
        active_fields=active_reward_fields,
        available_fields=active_reward_fields,
    )
    schema_path = os.path.join(output_dir, "reward_schema.json")
    _save_json(schema_path, schema_summary)
    return schema_path


def _run_candidate_generation_sharded(
    *,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    image_root: str,
    rerank_dir: str,
    candidates_path: str,
) -> Dict[str, Any]:
    """按可见 GPU 分片跑 candidate generation；单卡时回退到原有 in-process 逻辑。"""
    gpu_groups = split_visible_devices(len(samples))
    log_dir = _ensure_dir(os.path.join(rerank_dir, "logs"))
    temp_samples_path = os.path.join(rerank_dir, "val_samples_for_rerank.jsonl")
    write_jsonl_rows(temp_samples_path, samples)

    if not gpu_groups:
        _log("candidate_generation single_process mode")
        processor, model, _cfg = load(args.model_dir, device=args.device, dtype=args.dtype)
        model.eval()
        run_candidate_generation(
            samples,
            model,
            processor,
            image_root,
            args.device,
            num_candidates=args.rerank_num_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            prompt_mode=args.prompt_mode,
            task_aware_sampling=True,
            candidate_batch_size=args.rerank_candidate_batch_size,
            min_unique_candidates=args.rerank_min_unique_candidates,
            max_sampling_rounds=args.rerank_max_sampling_rounds,
            dedup_on="raw_text",
            allow_duplicate_fill=False,
            output_path=candidates_path,
        )
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        merged_rows = _read_jsonl_rows(candidates_path)
        diversity_path = os.path.join(rerank_dir, "candidate_diversity.json")
        _save_json(diversity_path, compute_candidate_diversity(merged_rows))
        return {
            "mode": "single_process",
            "gpu_groups": [],
            "shard_runs": [],
            "diversity_path": diversity_path,
        }

    _log(f"candidate_generation multi_gpu_sharded groups={[','.join(g) for g in gpu_groups]}")
    jobs: List[Dict[str, Any]] = []
    shard_paths: List[str] = []
    for shard_id, gpu_group in enumerate(gpu_groups):
        shard_output = os.path.join(rerank_dir, f"candidates_shard_{shard_id}.jsonl")
        shard_diversity = os.path.join(rerank_dir, f"candidate_diversity_shard_{shard_id}.json")
        cmd = [
            sys.executable, "-u", "src/candidate_gen.py",
            "--model_dir", args.model_dir,
            "--vqa_file", temp_samples_path,
            "--coco_image_dir", image_root,
            "--device", "cuda:0",
            "--dtype", args.dtype,
            "--output", shard_output,
            "--diversity_output", shard_diversity,
            "--num_candidates", str(args.rerank_num_candidates),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--max_new_tokens", str(args.max_new_tokens),
            "--prompt_mode", args.prompt_mode,
            "--task_aware_sampling",
            "--candidate_batch_size", str(args.rerank_candidate_batch_size),
            "--min_unique_candidates", str(args.rerank_min_unique_candidates),
            "--max_sampling_rounds", str(args.rerank_max_sampling_rounds),
            "--dedup_on", "raw_text",
            "--num_shards", str(len(gpu_groups)),
            "--shard_id", str(shard_id),
        ]
        env = child_env_for_gpu_group(gpu_group)
        jobs.append({
            "label": f"candidate_gen_shard_{shard_id}",
            "cmd": cmd,
            "env": env,
        })
        shard_paths.append(shard_output)
    shard_runs = _run_logged_subprocess_batch(jobs, log_dir)
    for run_info in shard_runs:
        if run_info["returncode"] != 0:
            raise RuntimeError(
                f"candidate generation shard failed: label={run_info['label']} "
                f"cuda={run_info['cuda_visible_devices']} stderr={run_info['stderr_log']}"
            )

    merge_jsonl_files(shard_paths, candidates_path)
    merged_rows = _read_jsonl_rows(candidates_path)
    _log(f"candidate_generation merged candidate_rows={len(merged_rows)} path={candidates_path}")
    diversity_path = os.path.join(rerank_dir, "candidate_diversity.json")
    _save_json(diversity_path, compute_candidate_diversity(merged_rows))
    return {
        "mode": "multi_gpu_sharded",
        "gpu_groups": [",".join(g) for g in gpu_groups],
        "shard_runs": shard_runs,
        "diversity_path": diversity_path,
    }


def _run_reward_scoring_sharded(
    *,
    args: argparse.Namespace,
    image_root: str,
    rerank_dir: str,
    candidates_path: str,
    scored_path: str,
) -> Dict[str, Any]:
    """按可见 GPU 分片跑 reward scoring；单卡时回退到原有 in-process 逻辑。"""
    gpu_groups = split_visible_devices(max(1, len(_read_jsonl_rows(candidates_path))))
    log_dir = _ensure_dir(os.path.join(rerank_dir, "logs"))

    if not gpu_groups:
        _log("reward_scoring single_process mode")
        scored_rows = run_reward_scoring(
            candidates_file=candidates_path,
            model_dir=args.model_dir,
            coco_image_dir=image_root,
            device=args.device,
            dtype=args.dtype,
            replace_mode=args.replace_mode,
            key_mode="auto",
            mix_alpha=0.5,
            mix_beta=0.5,
            sur_ring=args.sur_ring,
            reward_backend="action_logprob_ate",
            reward_mode="soft_shaping",
            max_response_tokens=args.max_response_tokens,
            tau_resp=args.tau_resp,
            tau_ans=args.tau_ans,
            alpha_resp=args.alpha_resp,
            alpha_ans=args.alpha_ans,
            min_reward=args.min_reward,
            max_reward=args.max_reward,
            main_reward_mode=args.reward_mode,
            negative_intervention_k=args.negative_intervention_k,
            evidence_eps=args.evidence_eps,
            output_path=scored_path,
        )
        return {
            "mode": "single_process",
            "gpu_groups": [],
            "shard_runs": [],
            "scored_rows": len(scored_rows),
        }

    _log(f"reward_scoring multi_gpu_sharded groups={[','.join(g) for g in gpu_groups]}")
    jobs: List[Dict[str, Any]] = []
    shard_paths: List[str] = []
    for shard_id, gpu_group in enumerate(gpu_groups):
        shard_output = os.path.join(rerank_dir, f"candidate_scores_shard_{shard_id}.jsonl")
        cmd = [
            sys.executable, "-u", "src/verifier_reward.py",
            "--candidates_file", candidates_path,
            "--model_dir", args.model_dir,
            "--coco_image_dir", image_root,
            "--device", "cuda:0",
            "--dtype", args.dtype,
            "--replace_mode", args.replace_mode,
            "--key_mode", "auto",
            "--mix_alpha", "0.5",
            "--mix_beta", "0.5",
            "--sur_ring", str(args.sur_ring),
            "--reward_backend", "action_logprob_ate",
            "--reward_mode", "soft_shaping",
            "--max_response_tokens", str(args.max_response_tokens),
            "--tau_resp", str(args.tau_resp),
            "--tau_ans", str(args.tau_ans),
            "--alpha_resp", str(args.alpha_resp),
            "--alpha_ans", str(args.alpha_ans),
            "--min_reward", str(args.min_reward),
            "--max_reward", str(args.max_reward),
            "--main_reward_mode", args.reward_mode,
            "--negative_intervention_k", str(args.negative_intervention_k),
            "--evidence_eps", str(args.evidence_eps),
            "--num_shards", str(len(gpu_groups)),
            "--shard_id", str(shard_id),
            "--output", shard_output,
        ]
        env = child_env_for_gpu_group(gpu_group)
        jobs.append({
            "label": f"reward_scoring_shard_{shard_id}",
            "cmd": cmd,
            "env": env,
        })
        shard_paths.append(shard_output)
    shard_runs = _run_logged_subprocess_batch(jobs, log_dir)
    for run_info in shard_runs:
        if run_info["returncode"] != 0:
            raise RuntimeError(
                f"reward scoring shard failed: label={run_info['label']} "
                f"cuda={run_info['cuda_visible_devices']} stderr={run_info['stderr_log']}"
            )

    merge_jsonl_files(shard_paths, scored_path)
    scored_rows = _read_jsonl_rows(scored_path)
    _log(f"reward_scoring merged scored_rows={len(scored_rows)} path={scored_path}")
    _write_merged_reward_schema(scored_rows, rerank_dir)
    return {
        "mode": "multi_gpu_sharded",
        "gpu_groups": [",".join(g) for g in gpu_groups],
        "shard_runs": shard_runs,
        "scored_rows": len(scored_rows),
    }


def _compact_log_entry(info: Dict[str, Any]) -> Dict[str, Any]:
    drop = {
        "responses", "raw_responses", "final_answers", "final_answer_extraction_sources",
        "answer_text_used", "policy_prefix_text", "policy_target_text",
        "policy_scope_effective", "policy_scope_fallback_reason", "final_answer_char_span",
        "_reward_details", "score_bases", "pre_kl_rewards", "final_rewards",
        "kl_penalties", "answer_anchor_rewards", "raw_rewards_before_anchor",
        "raw_rewards_after_hard_veto", "hard_template_veto_applied",
        "hard_template_veto_reasons",
    }
    return {k: v for k, v in info.items() if k not in drop}


def _save_trainable_checkpoint(model, output_dir: str, step: int, extra: Optional[Dict[str, Any]] = None) -> str:
    """只保存可训练参数，避免在主实验里写出整模型权重。"""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
    if not _is_rank0():
        return ckpt_path
    raw = model.module if hasattr(model, "module") else model
    _ensure_dir(ckpt_dir)
    state = {
        "step": int(step),
        "trainable_state_dict": {
            name: p.detach().cpu()
            for name, p in raw.named_parameters()
            if p.requires_grad
        },
        "extra": dict(extra or {}),
    }
    torch.save(state, ckpt_path)
    return ckpt_path


def _summarize_main_telemetry(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [m for m in all_metrics if "error" not in m]
    group_diag = _compute_group_diagnostics(all_metrics)
    if not valid:
        return {
            "group_diagnostics": group_diag,
            "task_family_stats": {},
            "answer_distribution": {},
        }

    # 主实验默认输出 reviewer 关心的 telemetry，避免只停留在 smoke 诊断。
    yes_count = 0
    no_count = 0
    neg_rates = []
    resp_lens = []
    answer_counter = Counter()
    area_vals = []
    reward_vals = []
    family_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "zero_variance_fallback_rate": 0.0,
        "raw_reward_nonconstant_group_rate": 0.0,
        "final_answer_identical_rate": 0.0,
        "mean_unique_final_answers": 0.0,
    })

    for row in valid:
        fam = row.get("task_family") or infer_task_family(
            task_type=row.get("task_type", ""),
            question=row.get("question", ""),
        )
        fs = family_stats[fam]
        fs["count"] += 1
        fs["zero_variance_fallback_rate"] += 1.0 if row.get("used_zero_variance_fallback") else 0.0
        fs["raw_reward_nonconstant_group_rate"] += 0.0 if row.get("raw_reward_identical") else 1.0
        fs["final_answer_identical_rate"] += 1.0 if row.get("final_answer_identical") else 0.0
        fs["mean_unique_final_answers"] += float(row.get("n_unique_final_answers", 0.0))

        neg_rates.append(float(row.get("negation_rate", 0.0)))
        resp_lens.append(float(row.get("output_length_mean", 0.0)))
        for ans in row.get("final_answers", []) or []:
            key = " ".join((ans or "").strip().lower().split())
            if key:
                answer_counter[key] += 1
        for parsed in row.get("parsed_answers", []) or []:
            if parsed == "yes":
                yes_count += 1
            elif parsed == "no":
                no_count += 1

        for rd in row.get("_reward_details", []) or []:
            area = rd.get("proposal_area_fraction")
            reward = rd.get("reward")
            if isinstance(area, (int, float)) and isinstance(reward, (int, float)):
                if not (np.isnan(area) or np.isnan(reward)):
                    area_vals.append(float(area))
                    reward_vals.append(float(reward))

    for fam, stats in family_stats.items():
        denom = float(max(1, stats["count"]))
        stats["zero_variance_fallback_rate"] /= denom
        stats["raw_reward_nonconstant_group_rate"] /= denom
        stats["final_answer_identical_rate"] /= denom
        stats["mean_unique_final_answers"] /= denom

    corr = float("nan")
    if len(area_vals) >= 3 and np.std(area_vals) > 1e-12 and np.std(reward_vals) > 1e-12:
        corr = float(np.corrcoef(np.asarray(area_vals), np.asarray(reward_vals))[0, 1])

    return {
        "group_diagnostics": group_diag,
        "zero_variance_fallback_rate": group_diag.get("used_zero_variance_fallback_rate", 0.0),
        "raw_reward_nonconstant_group_rate": group_diag.get("raw_reward_nonconstant_group_rate", 0.0),
        "same_final_answer_diff_raw_reward": group_diag.get("same_final_answer_diff_raw_reward_group_rate", 0.0),
        "final_answer_identical_rate": group_diag.get("final_answer_identical_group_rate", 0.0),
        "mean_unique_final_answers": group_diag.get("mean_unique_final_answers_per_group", 0.0),
        "negation_rate": float(np.mean(neg_rates)) if neg_rates else 0.0,
        "yes_rate": yes_count / max(1, yes_count + no_count),
        "no_rate": no_count / max(1, yes_count + no_count),
        "avg_response_length": float(np.mean(resp_lens)) if resp_lens else 0.0,
        "intervention_area_vs_reward_correlation": corr,
        "task_family_stats": dict(family_stats),
        "answer_distribution": dict(answer_counter.most_common(20)),
    }


def _run_offline_rerank(
    *,
    args: argparse.Namespace,
    val_samples: List[Dict[str, Any]],
    image_root: str,
    output_dir: str,
) -> Dict[str, Any]:
    rerank_dir = _ensure_dir(os.path.join(output_dir, "offline_rerank"))
    candidates_path = os.path.join(rerank_dir, "candidates.jsonl")
    scored_path = os.path.join(rerank_dir, "candidate_scores.jsonl")
    _log(
        f"offline_rerank start val_samples={len(val_samples)} output_dir={rerank_dir} "
        f"reward_mode={args.reward_mode}"
    )
    candidate_stage = _run_candidate_generation_sharded(
        samples=val_samples,
        args=args,
        image_root=image_root,
        rerank_dir=rerank_dir,
        candidates_path=candidates_path,
    )
    reward_stage = _run_reward_scoring_sharded(
        args=args,
        image_root=image_root,
        rerank_dir=rerank_dir,
        candidates_path=candidates_path,
        scored_path=scored_path,
    )
    candidate_rows = _read_jsonl_rows(candidates_path)
    valid_candidate_rows = [r for r in candidate_rows if "error" not in r]
    candidate_error_rows = [r for r in candidate_rows if "error" in r]
    scored_rows = _read_jsonl_rows(scored_path)
    valid_scored_rows = [r for r in scored_rows if "error" not in r and "reward_error" not in r]
    _log(
        f"offline_rerank collected candidates={len(candidate_rows)} valid_candidates={len(valid_candidate_rows)} "
        f"scored_rows={len(scored_rows)} valid_scored_rows={len(valid_scored_rows)}"
    )

    if not valid_candidate_rows or not valid_scored_rows:
        warning = (
            "offline rerank produced zero valid candidates or zero valid scored rows; "
            "likely image resolution / candidate generation failure on val split"
        )
        error_examples = []
        for row in candidate_error_rows[:8]:
            error_examples.append({
                "sample_id": row.get("sample_id"),
                "image_file": row.get("image_file"),
                "source_image_path": row.get("source_image_path"),
                "resolved_image_path": row.get("resolved_image_path"),
                "error": row.get("error"),
            })
        gate_summary = {
            "warning": warning,
            "should_continue": bool(args.allow_rerank_warning_continue),
            "reward_mode": args.reward_mode,
            "offline_rerank_dir": rerank_dir,
            "candidate_generation": candidate_stage,
            "reward_scoring": reward_stage,
            "candidate_count": len(candidate_rows),
            "valid_candidate_count": len(valid_candidate_rows),
            "candidate_error_count": len(candidate_error_rows),
            "scored_row_count": len(scored_rows),
            "valid_scored_row_count": len(valid_scored_rows),
            "candidate_error_examples": error_examples,
        }
        rerank_results = {
            "main_experiment_offline_compare": {
                "table": [],
                "available_reward_modes": [],
                "warning": warning,
            },
            "empty_offline_rerank": {
                "candidate_count": len(candidate_rows),
                "valid_candidate_count": len(valid_candidate_rows),
                "candidate_error_count": len(candidate_error_rows),
                "scored_row_count": len(scored_rows),
                "valid_scored_row_count": len(valid_scored_rows),
                "candidate_error_examples": error_examples,
            },
        }
        _save_json(os.path.join(rerank_dir, "offline_rerank_gate.json"), gate_summary)
        _log(f"offline_rerank warning={warning}")
        if not args.allow_rerank_warning_continue:
            raise RuntimeError(f"offline rerank gate blocked training: {warning}")
        return {
            "rerank_dir": rerank_dir,
            "rerank_results": rerank_results,
            "gate_summary": gate_summary,
        }

    rerank_results = run_rerank_eval(
        scored_path,
        rerank_dir,
        candidates_file=candidates_path,
        require_min_unique_satisfied=True,
    )
    gate_warning = rerank_results.get("main_experiment_offline_compare", {}).get("warning", "")
    gate_summary = {
        "warning": gate_warning,
        "should_continue": (not gate_warning) or bool(args.allow_rerank_warning_continue),
        "reward_mode": args.reward_mode,
        "offline_rerank_dir": rerank_dir,
        "candidate_generation": candidate_stage,
        "reward_scoring": reward_stage,
    }
    _save_json(os.path.join(rerank_dir, "offline_rerank_gate.json"), gate_summary)
    _log(f"offline_rerank done gate_warning={gate_warning or 'none'}")
    if gate_warning and (not args.allow_rerank_warning_continue):
        raise RuntimeError(f"offline rerank gate blocked training: {gate_warning}")
    return {
        "rerank_dir": rerank_dir,
        "rerank_results": rerank_results,
        "gate_summary": gate_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="NeurIPS main experiment training entry")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--dataset_name", default="vg_brutal", choices=["legacy_vqa", "vg_brutal"])
    ap.add_argument("--dataset_file", default="")
    ap.add_argument("--image_root", default="")
    ap.add_argument("--vqa_file", default="")
    ap.add_argument("--coco_image_dir", default="")
    ap.add_argument("--output_dir", default="results/main_experiments/default_run")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=256)
    ap.add_argument("--max_probe_samples", type=int, default=256)
    ap.add_argument("--task_family_filter", default="counting,attribute,spatial")
    ap.add_argument("--probe_task_family_filter", default="existence")
    ap.add_argument("--train_budget_mode", default="steps", choices=["balanced_epochs", "steps"])
    ap.add_argument("--train_balanced_epochs", type=float, default=2.0)
    ap.add_argument("--sampling_mode", default="balanced_no_replacement",
                    choices=["balanced_no_replacement", "balanced_with_replacement"])
    ap.add_argument("--n_steps", type=int, default=2000)
    ap.add_argument("--group_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_coeff", type=float, default=0.01)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--max_response_tokens", type=int, default=64)
    ap.add_argument("--n_trainable_layers", type=int, default=4)
    ap.add_argument("--lora", dest="use_lora", action="store_true")
    ap.add_argument("--no_lora", dest="use_lora", action="store_false")
    ap.set_defaults(use_lora=False)
    ap.add_argument("--disable_ref_kl", action="store_true")
    ap.add_argument("--save_interval", type=int, default=100)
    ap.add_argument("--checkpoint_interval", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--prompt_mode", default="short_evidence_v1",
                    choices=["raw_question", "task_aware_v1", "short_evidence_v1", "final_answer_only_v1"])
    ap.add_argument("--answer_text_source", default="final_answer", choices=["raw_response", "final_answer"])
    ap.add_argument("--policy_logprob_scope", default="final_answer", choices=["full_response", "final_answer"])
    ap.add_argument("--reward_logprob_scope", default="final_answer", choices=["full_response", "final_answer"])
    ap.add_argument("--strict_final_answer_scope", action="store_true", default=True)
    ap.add_argument("--invalid_final_answer_penalty", type=float, default=-0.75)
    ap.add_argument("--reward_condition_on_prefix", action="store_true", default=True)
    ap.add_argument("--reward_mode", default="routed_gated_evidence",
                    choices=["correctness_only", "additive_evidence", "routed_gated_evidence"])
    ap.add_argument("--replace_mode", default="mean", choices=["zero", "noise", "mean"])
    ap.add_argument("--sur_ring", type=int, default=2)
    ap.add_argument("--negative_intervention_k", type=int, default=3)
    ap.add_argument("--evidence_eps", type=float, default=0.10)
    ap.add_argument("--tau_resp", type=float, default=0.20)
    ap.add_argument("--tau_ans", type=float, default=1.00)
    ap.add_argument("--alpha_resp", type=float, default=0.70)
    ap.add_argument("--alpha_ans", type=float, default=0.30)
    ap.add_argument("--min_reward", type=float, default=-1.25)
    ap.add_argument("--max_reward", type=float, default=1.00)
    ap.add_argument("--rerank_num_candidates", type=int, default=6)
    ap.add_argument("--rerank_candidate_batch_size", type=int, default=0)
    ap.add_argument("--rerank_min_unique_candidates", type=int, default=2)
    ap.add_argument("--rerank_max_sampling_rounds", type=int, default=3)
    ap.add_argument("--skip_offline_rerank", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--allow_rerank_warning_continue", action="store_true")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        os.environ["AUTO_MULTI_GPU"] = "0"
        os.environ["MODEL_DEVICE_MAP"] = "local_rank"
        _log(f"ddp world={world_size} local_rank={local_rank} device={args.device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if _early_occupy_enabled() and not _EARLY_CUDA_HOLD:
        _occupy_visible_cuda("before_dataset_and_model_load")

    output_dir = _ensure_dir(args.output_dir)
    dataset_cfg = resolve_dataset_paths(args)
    dataset_name = dataset_cfg["dataset_name"]
    dataset_file = dataset_cfg["dataset_file"]
    image_root = dataset_cfg["image_root"]

    train_family_filter = _parse_csv_lower_list(args.task_family_filter)
    probe_family_filter = _parse_csv_lower_list(args.probe_task_family_filter)
    _log(
        f"start output_dir={output_dir} reward_mode={args.reward_mode} "
        f"runtime_parallel={json.dumps(runtime_parallel_summary(), ensure_ascii=False)}"
    )

    train_samples, train_summary, _ = load_train_dataset(
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        image_root=image_root,
        task_type="",
        data_split=_resolve_split_alias(args.train_split),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples=args.max_train_samples,
        seed=args.seed,
        task_type_allowlist=train_family_filter,
    )
    val_samples, val_summary, _ = load_train_dataset(
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        image_root=image_root,
        task_type="",
        data_split=_resolve_split_alias(args.val_split),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples=args.max_val_samples,
        seed=args.seed,
        task_type_allowlist=train_family_filter,
    )
    probe_samples, probe_summary, _ = load_train_dataset(
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        image_root=image_root,
        task_type="",
        data_split=_resolve_split_alias(args.test_split),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples=args.max_probe_samples,
        seed=args.seed,
        task_type_allowlist=probe_family_filter,
    )

    if not train_samples:
        raise RuntimeError("main experiment train set is empty after task_family_filter")
    if not val_samples and not args.skip_offline_rerank:
        raise RuntimeError("offline rerank requires non-empty val set")

    dataset_integrity = check_dataset_integrity(train_samples, image_root=image_root, max_check=0)
    family_buckets = build_family_buckets(train_samples)
    train_budget = _resolve_effective_train_budget(
        args=args,
        family_buckets=family_buckets,
    )
    dataset_summary = {
        "dataset_name": dataset_name,
        "dataset_file": dataset_file,
        "image_root": image_root,
        "train": dict(train_summary),
        "val": dict(val_summary),
        "probe": dict(probe_summary),
        "train_loaded_summary": summarize_loaded_samples(train_samples),
        "train_task_family_distribution": {k: len(v) for k, v in family_buckets.items()},
        "train_budget": train_budget,
        "dataset_integrity": dataset_integrity,
        "task_family_filter": train_family_filter,
        "probe_task_family_filter": probe_family_filter,
        "split_alias": {"test": _resolve_split_alias(args.test_split)},
        "runtime_parallel": runtime_parallel_summary(),
    }
    _save_json(os.path.join(output_dir, "dataset_summary.json"), dataset_summary)
    _log(
        f"dataset loaded train={len(train_samples)} val={len(val_samples)} probe={len(probe_samples)} "
        f"train_families={json.dumps(dataset_summary['train_task_family_distribution'], ensure_ascii=False)} "
        f"train_budget={json.dumps(train_budget, ensure_ascii=False)}"
    )

    offline_rerank_payload: Dict[str, Any] = {}
    if not args.skip_offline_rerank:
        offline_rerank_payload = _run_offline_rerank(
            args=args,
            val_samples=val_samples,
            image_root=image_root,
            output_dir=output_dir,
        )

    if args.skip_train:
        summary = {
            "mode": "main_experiment_rerank_only",
            "reward_mode": args.reward_mode,
            "task_family_filter": train_family_filter,
            "dataset_summary": dataset_summary,
            "offline_rerank": offline_rerank_payload,
            "runtime_parallel": runtime_parallel_summary(),
        }
        _save_json(os.path.join(output_dir, "main_experiment_summary.json"), summary)
        _log(f"rerank-only summary -> {os.path.join(output_dir, 'main_experiment_summary.json')}")
        return

    _release_early_cuda()
    processor, model, _cfg = load(args.model_dir, device=args.device, dtype=args.dtype)
    model, lora_applied = _setup_trainable(
        model,
        n_trainable_layers=args.n_trainable_layers,
        use_lora=bool(args.use_lora),
    )
    if args.use_lora and not lora_applied:
        raise RuntimeError("LoRA was requested (--lora) but was not applied")
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    initial_params = None
    use_ref_kl = not args.disable_ref_kl
    if use_ref_kl and not lora_applied:
        initial_params = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }

    reward_fn = ActionLogProbATEReward(
        model=model,
        processor=processor,
        device=args.device,
        replace_mode=args.replace_mode,
        reward_mode="soft_shaping",
        max_response_tokens=args.max_response_tokens,
        tau_resp=args.tau_resp,
        tau_ans=args.tau_ans,
        alpha_resp=args.alpha_resp,
        alpha_ans=args.alpha_ans,
        min_reward=args.min_reward,
        max_reward=args.max_reward,
        main_reward_mode=args.reward_mode,
        negative_intervention_k=args.negative_intervention_k,
        evidence_eps=args.evidence_eps,
    )

    preflight = run_training_preflight_check(
        model=model,
        processor=processor,
        reward_fn=reward_fn,
        samples=train_samples,
        image_root=image_root,
        reward_backend="action_logprob_ate_nolabel",
        prompt_mode=args.prompt_mode,
        answer_text_source=args.answer_text_source,
        policy_logprob_scope=args.policy_logprob_scope,
        max_new_tokens=args.max_new_tokens,
        group_size=args.group_size,
        n_groups=min(8, max(3, train_budget["effective_train_steps"])),
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        allow_prompt_side_reward_for_smoke=False,
    )
    _save_json(os.path.join(output_dir, "preflight_summary.json"), preflight)
    _log(
        f"preflight passed={preflight.get('passed', False)} "
        f"output={os.path.join(output_dir, 'preflight_summary.json')}"
    )
    if not preflight.get("passed", False):
        raise RuntimeError("main experiment preflight failed")

    trainer = MiniGRPO(
        model,
        processor,
        reward_fn,
        reward_backend="action_logprob_ate_nolabel",
        device=args.device,
        lr=args.lr,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        kl_coeff=args.kl_coeff,
        lora_applied=lora_applied,
        use_ref_kl=use_ref_kl,
        initial_params=initial_params,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        prompt_mode=args.prompt_mode,
        answer_text_source=args.answer_text_source,
        policy_logprob_scope=args.policy_logprob_scope,
        reward_logprob_scope=args.reward_logprob_scope,
        strict_final_answer_scope=args.strict_final_answer_scope,
        invalid_final_answer_penalty=args.invalid_final_answer_penalty,
        reward_condition_on_prefix=args.reward_condition_on_prefix,
        reward_mix_mode="main_experiment",
        main_reward_mode=args.reward_mode,
        allow_ground_truth_for_main_rewards=True,
        task_type="",
    )

    log_path = os.path.join(output_dir, "training_log.jsonl")
    sample_path = os.path.join(output_dir, "sample_outputs.jsonl")
    all_metrics: List[Dict[str, Any]] = []
    effective_train_steps = int(train_budget["effective_train_steps"])
    sampler: BalancedFamilyCoverageSampler | None = None
    if train_budget["sampling_mode"] in {"balanced_no_replacement", "balanced_with_replacement"}:
        sampler = BalancedFamilyCoverageSampler(
            family_buckets=family_buckets,
            rng=random,
            sampling_mode=train_budget["sampling_mode"],
        )
    progress_interval = max(10, min(200, max(1, effective_train_steps // 50)))
    _log(
        f"training start effective_train_steps={effective_train_steps} configured_n_steps={args.n_steps} "
        f"group_size={args.group_size} sampling_mode={train_budget['sampling_mode']} "
        f"progress_interval={progress_interval} checkpoint_interval={args.checkpoint_interval}"
    )
    _ensure_dir(os.path.dirname(log_path))
    _ensure_dir(os.path.dirname(sample_path))
    log_f = open(log_path, "w", encoding="utf-8")
    sample_f = open(sample_path, "w", encoding="utf-8")
    try:
        for step in range(effective_train_steps):
            if sampler is not None:
                sample = sampler.sample()
            else:
                sample = sample_from_family_buckets(
                    train_samples,
                    family_buckets=family_buckets,
                    rng=random,
                    balanced=True,
                )
            info = trainer.train_step(sample, image_root)
            info["step"] = step
            info["stage"] = "main_experiment"
            all_metrics.append(info)

            log_f.write(json.dumps(_compact_log_entry(info), ensure_ascii=False) + "\n")
            _flush_fsync(log_f)

            if step % args.save_interval == 0:
                sample_f.write(json.dumps({
                    "step": step,
                    "question": info.get("question"),
                    "task_family": info.get("task_family"),
                    "final_answers": info.get("final_answers", []),
                    "parsed_answers": info.get("parsed_answers", []),
                    "rewards": info.get("rewards", []),
                    "advantages": info.get("advantages", []),
                }, ensure_ascii=False) + "\n")
                _flush_fsync(sample_f)

            if args.checkpoint_interval > 0 and ((step + 1) % args.checkpoint_interval == 0):
                ckpt_path = _save_trainable_checkpoint(
                    model,
                    output_dir,
                    step + 1,
                    extra={
                        "reward_mode": args.reward_mode,
                        "lora_applied": bool(lora_applied),
                        "n_trainable_layers": int(args.n_trainable_layers),
                    },
                )
                _log(f"checkpoint_saved step={step + 1} path={ckpt_path}")

            if (step == 0) or ((step + 1) % progress_interval == 0) or ((step + 1) == effective_train_steps):
                mean_reward = float(np.mean(info.get("rewards", []) or [0.0]))
                unique_answers = info.get("n_unique_final_answers", 0)
                _log(
                    f"training_progress step={step + 1}/{effective_train_steps} "
                    f"task_family={info.get('task_family', '')} mean_reward={mean_reward:.4f} "
                    f"unique_final_answers={unique_answers} zero_var_fallback={bool(info.get('used_zero_variance_fallback'))}"
                )
    finally:
        _safe_close_file(sample_f)
        _safe_close_file(log_f)

    final_ckpt = _save_trainable_checkpoint(
        model,
        output_dir,
        effective_train_steps,
        extra={
            "reward_mode": args.reward_mode,
            "final": True,
            "lora_applied": bool(lora_applied),
            "n_trainable_layers": int(args.n_trainable_layers),
        },
    )

    telemetry = _summarize_main_telemetry(all_metrics)
    summary = {
        "mode": "main_experiment_train",
        "reward_mode": args.reward_mode,
        "task_family_filter": train_family_filter,
        "probe_task_family_filter": probe_family_filter,
        "n_steps": effective_train_steps,
        "configured_n_steps": int(args.n_steps),
        "train_budget": train_budget,
        "final_checkpoint": final_ckpt,
        "dataset_summary": dataset_summary,
        "offline_rerank": offline_rerank_payload,
        "telemetry": telemetry,
        "runtime_parallel": runtime_parallel_summary(),
    }
    _save_json(os.path.join(output_dir, "main_experiment_summary.json"), summary)
    merged_dir = os.path.join(output_dir, "merged_hf")
    try:
        processor.save_pretrained(merged_dir)
        model.save_pretrained(merged_dir, safe_serialization=True)
        summary["merged_model_dir"] = merged_dir
        _save_json(os.path.join(output_dir, "main_experiment_summary.json"), summary)
        _log(f"merged_hf -> {merged_dir}")
    except Exception as exc:
        _log(f"merged_hf failed: {type(exc).__name__}: {exc}")

    _log(f"summary -> {os.path.join(output_dir, 'main_experiment_summary.json')}")


if __name__ == "__main__":
    main()
