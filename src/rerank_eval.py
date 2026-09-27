"""Rerank / Selection Evaluation 模块。

对每道题的 K 个候选答案，按不同选择策略 rerank，比较 accuracy。

选择策略（schema 驱动）：
    1. select_by_logprob                     — 取 mean_logprob 最高的候选
    2. select_by_reward_logits_js            — 取 logits JS 最高的候选（若在本 run 可用）
    3. select_by_reward_logits_cosine_dist   — 取 logits cosine dist 最高（若在本 run 可用）

明确禁止参与 rerank 的项：
    - sample-level probe（如 reward_layer24_prompt_last_cosine）
    - 与 base reward 排序等价的 fake mix（如 reward_mix_*）
    - runtime 检测为 constant-within-sample 的 reward

另外增加两个 baseline:
  - random:  随机选（期望 accuracy）
  - oracle:  如果任一候选正确，就算对（正确答案覆盖率上界）
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))
from yesno_utils import parse_yesno
from answer_format_utils import infer_task_family
from reward.schema import (
    REWARD_SCHEMA_VERSION,
    get_verifier_eligible_reward_fields,
    get_training_reward_diagnostic_fields,
    reward_to_strategy_map,
    get_schema_summary,
    infer_active_reward_fields_from_records,
)

# 一致性校验时默认检查的字段
FIELDS_TO_CHECK = ("generated_text", "correct", "behavior")


def _load_reward_schema_sidecar(scored_file: str) -> Optional[Dict[str, Any]]:
    """Load reward_schema.json next to scored_file if present."""
    try:
        schema_path = os.path.join(os.path.dirname(scored_file) or ".", "reward_schema.json")
        if os.path.isfile(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


# ────────────────────────────────────────
# 策略定义（schema 驱动）
# ────────────────────────────────────────

# Always keep logprob strategy name as baseline rerank comparator.
# The actual score key is resolved at runtime:
#   final_answer_mean_logprob (preferred) → mean_logprob (fallback)
BASE_STRATEGY_NAME = "select_by_logprob"
MAIN_EXPERIMENT_STRATEGIES = {
    "correctness_only": "select_by_reward_main_correctness_only",
    "additive_evidence": "select_by_reward_main_additive_evidence",
    "routed_gated_evidence": "select_by_reward_main_routed_gated_evidence",
}


def _is_finite_number(v: Any) -> bool:
    try:
        fv = float(v)
    except Exception:
        return False
    return not (math.isnan(fv) or math.isinf(fv))


def _normalize_answer_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _resolve_logprob_score_key(records: List[Dict[str, Any]]) -> str:
    """Prefer answer-only logprob when available.

    Why: Evidence + Final answer 会让整段 response logprob 被解释长度污染。
    """
    for r in records:
        if _is_finite_number(r.get("final_answer_mean_logprob")):
            return "final_answer_mean_logprob"
    return "mean_logprob"


def _non_duplicate_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prefer explicit non-duplicate candidates for subset diagnostics."""
    kept = [c for c in cands if not bool(c.get("is_duplicate_candidate", False))]
    return kept if kept else list(cands)


def _informative_sample_ids(groups: Dict[Any, List[Dict]]) -> set:
    """Informative subset: within-sample >=2 unique normalized raw responses (duplicate-filtered)."""
    ids = set()
    for sid, cands in groups.items():
        cands_eff = _non_duplicate_candidates(cands)
        texts = [_normalize_answer_text(c.get("raw_generated_text", c.get("generated_text", ""))) for c in cands_eff]
        texts = [t for t in texts if t]
        if len(set(texts)) >= 2:
            ids.add(sid)
    return ids


def _strong_informative_sample_ids(groups: Dict[Any, List[Dict]]) -> set:
    """Stronger subset (optional): informative + correctness not all identical."""
    ids = set()
    for sid, cands in groups.items():
        cands_eff = _non_duplicate_candidates(cands)
        texts = [_normalize_answer_text(c.get("raw_generated_text", c.get("generated_text", ""))) for c in cands_eff]
        texts = [t for t in texts if t]
        if len(set(texts)) < 2:
            continue
        cs = [bool(_is_correct(c)) for c in cands_eff]
        if any(cs) and not all(cs):
            ids.add(sid)
    return ids


def _top1_wrong_but_oracle_right_ids(groups: Dict[Any, List[Dict]]) -> set:
    ids = set()
    for sid, cands in groups.items():
        if not cands:
            continue
        cands_eff = _non_duplicate_candidates(cands)
        top1 = next((c for c in cands if c.get("candidate_id") == 0), cands[0])
        top1_correct = bool(_is_correct(top1))
        oracle_correct = any(_is_correct(c) for c in cands_eff)
        if (not top1_correct) and oracle_correct:
            ids.add(sid)
    return ids


def _sample_min_unique_status(cands: List[Dict[str, Any]]) -> Optional[bool]:
    vals = [c.get("min_unique_satisfied") for c in cands if "min_unique_satisfied" in c]
    if not vals:
        return None
    # Be conservative on inconsistent records: any False means sample is unsatisfied.
    return all(bool(v) for v in vals)


def _select_best(candidates: List[Dict], score_key: str) -> Optional[Dict]:
    """从候选中选出 score_key 最大的，跳过 nan。"""
    valid = []
    for c in candidates:
        v = c.get(score_key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            valid.append((float(v), c))
    if not valid:
        return None
    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1]


def _classify_behavior(cand: Dict) -> str:
    """从候选记录推断 behavior，兼容已有的 behavior 字段。"""
    return cand.get("behavior", "other")


def _is_correct(cand: Dict) -> bool:
    if "correct" in cand:
        return bool(cand["correct"])
    return _classify_behavior(cand) in ("correct_positive", "correct_negative")


def _build_main_experiment_comparison(strategy_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """提取主实验三种 reward mode 的统一离线比较摘要。"""
    table = []
    available = {}
    for mode, strat in MAIN_EXPERIMENT_STRATEGIES.items():
        data = strategy_results.get(strat)
        if not data:
            continue
        row = {
            "reward_mode": mode,
            "strategy": strat,
            "accuracy": data.get("accuracy", 0.0),
            "correct": data.get("correct", 0),
            "total_samples": data.get("total_samples", 0),
            "informative_accuracy": data.get("informative_accuracy", 0.0),
            "informative_correct": data.get("informative_correct", 0),
            "informative_total_samples": data.get("informative_total_samples", 0),
            "strong_informative_accuracy": data.get("strong_informative_accuracy", 0.0),
        }
        available[mode] = row
        table.append(row)

    warning = ""
    routed = available.get("routed_gated_evidence")
    corr = available.get("correctness_only")
    addi = available.get("additive_evidence")
    if routed and corr and addi:
        gated_best_gain = float(routed["accuracy"]) - max(float(corr["accuracy"]), float(addi["accuracy"]))
        gated_inf_gain = float(routed["informative_accuracy"]) - max(
            float(corr["informative_accuracy"]),
            float(addi["informative_accuracy"]),
        )
        if gated_best_gain <= 0.0 and gated_inf_gain <= 0.0:
            warning = (
                "routed_gated_evidence has no offline rerank advantage over "
                "correctness_only/additive_evidence on both overall and informative subsets"
            )
    return {
        "table": table,
        "available_reward_modes": [row["reward_mode"] for row in table],
        "warning": warning,
    }


# ────────────────────────────────────────
# 辅助：MD5 + 行数
# ────────────────────────────────────────

def _md5_file(path: str) -> str:
    """计算文件 MD5，文件不存在或出错则返回 'n/a'。"""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "n/a"


def _count_jsonl_lines(path: str) -> int:
    """统计 jsonl 行数（忽略空行）。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return -1


# ────────────────────────────────────────
# 一致性校验
# ────────────────────────────────────────

def validate_candidate_consistency(
    candidates_file: str,
    scored_file: str,
    check_fields: tuple = FIELDS_TO_CHECK,
) -> None:
    """按 (sample_id, candidate_id) 对齐，校验关键字段在两个文件中完全一致。

    发现任意不一致时直接打印详情并 sys.exit(1)。
    This is intentionally strict: silent mismatches cause mixed-run artifacts.
    """
    with open(candidates_file, "r", encoding="utf-8") as f:
        cands = [json.loads(line) for line in f if line.strip()]
    with open(scored_file, "r", encoding="utf-8") as f:
        scored = [json.loads(line) for line in f if line.strip()]

    # 建立 candidates 索引
    cand_index: Dict[tuple, Dict] = {}
    for c in cands:
        sid = c.get("sample_id", c.get("index"))
        cid = c.get("candidate_id")
        if sid is not None and cid is not None:
            cand_index[(sid, cid)] = c

    errors: List[str] = []
    n_checked = 0
    for s in scored:
        if "error" in s or "reward_error" in s or s.get("action_logprob_ate_error") or s.get("reward_record_status") == "error":
            continue
        sid = s.get("sample_id", s.get("index"))
        cid = s.get("candidate_id")
        if sid is None or cid is None:
            continue
        key = (sid, cid)
        if key not in cand_index:
            errors.append(
                f"  (sample_id={sid!r}, candidate_id={cid!r}) "
                f"存在于 candidate_scores 但不在 candidates 中"
            )
            continue
        n_checked += 1
        orig = cand_index[key]
        for field in check_fields:
            if field not in s and field not in orig:
                continue  # 两边都没有，不算不一致
            v_scored = s.get(field)
            v_cands = orig.get(field)
            if v_scored != v_cands:
                errors.append(
                    f"  (sample_id={sid!r}, candidate_id={cid!r}) "
                    f"field={field!r}: "
                    f"candidates={v_cands!r} vs candidate_scores={v_scored!r}"
                )
                if len(errors) >= 20:  # 防止日志爆炸
                    break
        if len(errors) >= 20:
            break

    if errors:
        msg = (
            f"[rerank_eval] FATAL: candidates.jsonl 与 candidate_scores.jsonl 不一致！\n"
            f"  candidates:       {candidates_file}\n"
            f"  candidate_scores: {scored_file}\n"
            f"  这通常意味着两个文件来自不同的 run（shard 污染或跳步执行）。\n"
            f"  发现 {len(errors)} 处不一致（最多显示 20 条）:\n"
            + "\n".join(errors)
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    print(
        f"[rerank_eval] ✓ candidate consistency OK "
        f"({n_checked} records checked across "
        f"{len(cands)} candidates / {len(scored)} scored)"
    )


def verify_selected_vs_results(
    selected_records: List[Dict],
    strategy_results: Dict[str, Any],
) -> None:
    """从 selected_candidates 逐策略重新统计 accuracy，与 strategy_results 严格比对。

    不一致时 sys.exit(1) — 保证 selected.jsonl 和 rerank_results.json 总是同一次写出的。
    """
    by_strategy: Dict[str, List[Dict]] = defaultdict(list)
    for rec in selected_records:
        strat = rec.get("strategy")
        if strat:
            by_strategy[strat].append(rec)

    errors: List[str] = []
    for strat, records in by_strategy.items():
        if strat not in strategy_results:
            continue
        n_total = len(records)
        n_correct = sum(1 for r in records if r.get("correct", False))
        recount_acc = round(n_correct / n_total, 4) if n_total > 0 else 0.0
        expected_acc = strategy_results[strat].get("accuracy")
        if expected_acc is None:
            continue
        if abs(recount_acc - float(expected_acc)) > 1e-4:
            errors.append(
                f"  strategy={strat!r}: "
                f"recount acc={recount_acc} ({n_correct}/{n_total}) "
                f"≠ rerank_results acc={expected_acc}"
            )

    if errors:
        msg = (
            f"[rerank_eval] FATAL: selected_candidates_by_strategy 与 rerank_results 不一致！\n"
            f"  这表明内存中的 selected_records 与写入 JSON 的数值不同步（代码 bug）。\n"
            f"  发现 {len(errors)} 处不一致:\n"
            + "\n".join(errors)
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    print(f"[rerank_eval] ✓ selected vs rerank_results consistency OK ({len(by_strategy)} strategies)")


# ────────────────────────────────────────
# Provenance / Manifest
# ────────────────────────────────────────

def write_verifier_manifest(
    output_dir: str,
    run_id: str,
    candidates_file: Optional[str],
    scored_file: str,
    n_samples: int,
    n_candidate_rows: int,
    extra: Optional[Dict] = None,
) -> Dict[str, Any]:
    """写出 verifier_manifest.json，包含本次 run 的完整 provenance。"""
    # 计算每样本候选数分布（仅统计 scored_file，因为它继承了 candidates 字段）
    cands_per_sample: Dict[Any, int] = defaultdict(int)
    try:
        with open(scored_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec.get("sample_id", rec.get("index"))
                if sid is not None:
                    cands_per_sample[sid] += 1
    except Exception:
        pass
    counts = sorted(cands_per_sample.values()) if cands_per_sample else []
    n_k = len(counts)
    cand_dist = {
        "min": counts[0] if counts else None,
        "max": counts[-1] if counts else None,
        "median": counts[n_k // 2] if counts else None,
        "mean": round(sum(counts) / n_k, 2) if counts else None,
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "candidates_path": os.path.abspath(candidates_file) if candidates_file else None,
        "candidate_scores_path": os.path.abspath(scored_file),
        "candidates_md5": _md5_file(candidates_file) if candidates_file else None,
        "candidate_scores_md5": _md5_file(scored_file),
        "n_samples": n_samples,
        "n_candidate_rows": n_candidate_rows,
        "n_candidates_per_sample": cand_dist,
    }
    if extra:
        manifest.update(extra)

    manifest_path = os.path.join(output_dir, "verifier_manifest.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[rerank_eval] manifest  → {manifest_path}")
    return manifest


# ────────────────────────────────────────
# 按 sample_id 分组
# ────────────────────────────────────────

def group_by_sample(records: List[Dict]) -> Dict[Any, List[Dict]]:
    """按 sample_id 分组。"""
    groups = defaultdict(list)
    for r in records:
        sid = r.get("sample_id", r.get("index"))
        if sid is not None:
            groups[sid].append(r)
    return dict(groups)


def build_strategy_map(
    reward_keys_for_rerank: List[str],
    logprob_score_key: str,
) -> Dict[str, str]:
    """Build runtime strategy map.

    Only rewards in reward_keys_for_rerank are added as rerank strategies.
    This prevents sample-level probes / fake-mix rewards from entering rerank.
    """
    mapping = {BASE_STRATEGY_NAME: logprob_score_key}
    reward_to_strategy = reward_to_strategy_map(include_future=False)
    for rk in reward_keys_for_rerank:
        strat = reward_to_strategy.get(rk)
        if strat:
            mapping[strat] = rk
    return mapping


def build_training_diagnostic_strategy_map(
    reward_keys_for_diagnostic: List[str],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for rk in reward_keys_for_diagnostic:
        mapping[f"diagnostic_select_by_{rk}"] = rk
    return mapping


# ────────────────────────────────────────
# 评估
# ────────────────────────────────────────

def evaluate_strategy(
    groups: Dict[Any, List[Dict]],
    strategy: str,
    score_key: str,
) -> Dict[str, Any]:
    """对一个策略做整体 + task-wise 评估。"""

    total = 0
    correct = 0
    behavior_counts = defaultdict(int)
    task_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    family_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    selected_records = []
    total_length = 0
    total_yesno = 0

    for sid, cands in groups.items():
        selected = _select_best(cands, score_key)
        if selected is None:
            continue

        total += 1
        beh = _classify_behavior(selected)
        behavior_counts[beh] += 1
        c = _is_correct(selected)
        if c:
            correct += 1

        tt = selected.get("task_type", "unknown")
        fam = selected.get("task_family") or infer_task_family(
            task_type=selected.get("task_type", ""),
            question=selected.get("question", ""),
            metadata=selected,
        )
        task_stats[tt]["total"] += 1
        if c:
            task_stats[tt]["correct"] += 1
        family_stats[fam]["total"] += 1
        if c:
            family_stats[fam]["correct"] += 1

        total_length += selected.get("answer_length", 0)
        if selected.get("is_yesno", False):
            total_yesno += 1

        selected_records.append({
            "sample_id": sid,
            "strategy": strategy,
            "selected_candidate_id": selected.get("candidate_id"),
            "selected_text": selected.get("generated_text", ""),
            "selected_score": selected.get(score_key),
            "selected_score_key": score_key,
            "correct": c,
            "behavior": beh,
            "task_type": tt,
            "task_family": fam,
        })

    accuracy = correct / total if total > 0 else 0.0
    task_accuracy = {}
    for tt, ts in task_stats.items():
        task_accuracy[tt] = {
            "accuracy": ts["correct"] / ts["total"] if ts["total"] > 0 else 0.0,
            "correct": ts["correct"],
            "total": ts["total"],
        }
    family_accuracy = {}
    for fam, fs in family_stats.items():
        family_accuracy[fam] = {
            "accuracy": fs["correct"] / fs["total"] if fs["total"] > 0 else 0.0,
            "correct": fs["correct"],
            "total": fs["total"],
        }

    return {
        "strategy": strategy,
        "score_key": score_key,
        "total_samples": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "behavior_breakdown": dict(behavior_counts),
        "avg_answer_length": round(total_length / total, 2) if total > 0 else 0,
        "yesno_ratio": round(total_yesno / total, 4) if total > 0 else 0,
        "task_accuracy": task_accuracy,
        "task_family_accuracy": family_accuracy,
        "selected_records": selected_records,
    }


def evaluate_baselines(groups: Dict[Any, List[Dict]]) -> Dict[str, Dict]:
    """计算 random / oracle / top1 baseline。"""
    rng = random.Random(42)
    results = {}

    # Random baseline
    total = correct_rand = 0
    correct_top1 = 0
    correct_oracle = 0
    task_rand = defaultdict(lambda: {"total": 0, "correct": 0})
    task_top1 = defaultdict(lambda: {"total": 0, "correct": 0})
    task_oracle = defaultdict(lambda: {"total": 0, "correct": 0})
    fam_rand = defaultdict(lambda: {"total": 0, "correct": 0})
    fam_top1 = defaultdict(lambda: {"total": 0, "correct": 0})
    fam_oracle = defaultdict(lambda: {"total": 0, "correct": 0})

    for sid, cands in groups.items():
        if not cands:
            continue
        total += 1
        tt = cands[0].get("task_type", "unknown")
        fam = cands[0].get("task_family") or infer_task_family(
            task_type=cands[0].get("task_type", ""),
            question=cands[0].get("question", ""),
            metadata=cands[0],
        )

        # Random
        pick = rng.choice(cands)
        c = _is_correct(pick)
        if c:
            correct_rand += 1
        task_rand[tt]["total"] += 1
        if c:
            task_rand[tt]["correct"] += 1
        fam_rand[fam]["total"] += 1
        if c:
            fam_rand[fam]["correct"] += 1

        # Top1 (first candidate, candidate_id=0)
        top1 = next((c for c in cands if c.get("candidate_id") == 0), cands[0])
        c1 = _is_correct(top1)
        if c1:
            correct_top1 += 1
        task_top1[tt]["total"] += 1
        if c1:
            task_top1[tt]["correct"] += 1
        fam_top1[fam]["total"] += 1
        if c1:
            fam_top1[fam]["correct"] += 1

        # Oracle
        any_correct = any(_is_correct(c) for c in cands)
        if any_correct:
            correct_oracle += 1
        task_oracle[tt]["total"] += 1
        if any_correct:
            task_oracle[tt]["correct"] += 1
        fam_oracle[fam]["total"] += 1
        if any_correct:
            fam_oracle[fam]["correct"] += 1

    def _fmt(name, n_correct, task_data, family_data):
        ta = {}
        for tt, ts in task_data.items():
            ta[tt] = {
                "accuracy": ts["correct"] / ts["total"] if ts["total"] > 0 else 0,
                "correct": ts["correct"], "total": ts["total"],
            }
        fa = {}
        for fam, fs in family_data.items():
            fa[fam] = {
                "accuracy": fs["correct"] / fs["total"] if fs["total"] > 0 else 0,
                "correct": fs["correct"], "total": fs["total"],
            }
        return {
            "strategy": name,
            "total_samples": total,
            "correct": n_correct,
            "accuracy": round(n_correct / total, 4) if total > 0 else 0,
            "task_accuracy": ta,
            "task_family_accuracy": fa,
        }

    results["random"] = _fmt("random", correct_rand, task_rand, fam_rand)
    results["top1"] = _fmt("top1", correct_top1, task_top1, fam_top1)
    results["oracle"] = _fmt("oracle", correct_oracle, task_oracle, fam_oracle)
    return results


def _pairwise_win_loss(
    a_correct: Dict[Any, bool],
    b_correct: Dict[Any, bool],
    sample_ids: Optional[set] = None,
) -> Dict[str, int]:
    """Pairwise win/loss based on correctness.

    win:  A correct, B wrong
    loss: A wrong, B correct
    tie:  both correct or both wrong
    """
    n_win = n_loss = n_tie = 0
    keys = sample_ids if sample_ids is not None else set(a_correct.keys())
    for sid in keys:
        if sid not in a_correct or sid not in b_correct:
            continue
        a = bool(a_correct[sid])
        b = bool(b_correct[sid])
        if a and not b:
            n_win += 1
        elif (not a) and b:
            n_loss += 1
        else:
            n_tie += 1
    return {"n_win": n_win, "n_loss": n_loss, "n_tie": n_tie}


# ────────────────────────────────────────
# Reward 退化检查（重排前 sanity check）
# ────────────────────────────────────────

def reward_degeneracy_check(
    groups: Dict[Any, List[Dict]],
    reward_keys: Optional[List[str]] = None,
    nonconstant_threshold: float = 0.3,
    winner_not_candidate0_threshold: float = 0.1,
) -> Dict[str, Any]:
    """检查 reward 是否对 candidate 敏感。

    返回:
      per_reward:
        {reward_name: {
          samplewise_nonconstant_rate,
          winner_not_candidate0_rate,    # 基于真实 candidate_id 字段（修复版）
          winner_not_first_record_rate,  # 基于记录在组内的先后顺序（参考用）
          n_samples_evaluated,
          is_degenerate,
        }}
      selection_degeneracy_flag: True 表示所有 reward 都高度退化，rerank 无意义

    修复说明：
      旧实现用列表索引 best_idx != 0 来近似 "winner 不是 candidate 0"，
      实际上算的是 "winner 不是组内第一条记录"（winner_not_first_record_rate）。
      两者在候选列表无序时完全不同。本实现明确分开两个指标。
      is_degenerate 判断使用正确的 winner_not_candidate0_rate。
    """
    if reward_keys is None:
        reward_keys = get_verifier_eligible_reward_fields(include_future=False)

    per_reward = {}
    any_ok = False

    for rkey in reward_keys:
        n_samples = 0
        n_nonconstant = 0
        n_winner_not_candidate0 = 0
        n_winner_not_first_record = 0

        for sid, cands in groups.items():
            # 收集 (值, candidate_id, 记录在列表中的下标)
            valid_triples: List[tuple] = []
            for rec_idx, c in enumerate(cands):
                v = c.get(rkey)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    valid_triples.append((float(v), c.get("candidate_id"), rec_idx))

            if len(valid_triples) < 2:
                continue
            n_samples += 1

            vals = [t[0] for t in valid_triples]

            # 检查组内是否存在不同值
            if max(vals) - min(vals) > 1e-9:
                n_nonconstant += 1

            # 找最高分的那条记录
            best_pos = max(range(len(valid_triples)), key=lambda i: valid_triples[i][0])
            winner_cid = valid_triples[best_pos][1]   # 真实 candidate_id
            winner_rec_idx = valid_triples[best_pos][2]  # 列表位置

            # winner_not_candidate0_rate：winner 的 candidate_id 不为 0（正确指标）
            if winner_cid is not None and winner_cid != 0:
                n_winner_not_candidate0 += 1

            # winner_not_first_record_rate：winner 不是列表里第一条（参考指标）
            if winner_rec_idx != 0:
                n_winner_not_first_record += 1

        nc_rate = n_nonconstant / n_samples if n_samples else 0.0
        wc0_rate = n_winner_not_candidate0 / n_samples if n_samples else 0.0
        wfr_rate = n_winner_not_first_record / n_samples if n_samples else 0.0

        # 退化判定：使用正确的 winner_not_candidate0_rate（而非旧版的 first_record 指标）
        is_degenerate = (
            nc_rate < nonconstant_threshold
            and wc0_rate < winner_not_candidate0_threshold
        )

        per_reward[rkey] = {
            "samplewise_nonconstant_rate": round(nc_rate, 4),
            "winner_not_candidate0_rate": round(wc0_rate, 4),
            "winner_not_first_record_rate": round(wfr_rate, 4),
            "n_samples_evaluated": n_samples,
            "is_degenerate": is_degenerate,
        }

        if not is_degenerate:
            any_ok = True

    selection_degeneracy_flag = not any_ok

    return {
        "per_reward": per_reward,
        "selection_degeneracy_flag": selection_degeneracy_flag,
        "nonconstant_threshold": nonconstant_threshold,
        "winner_not_candidate0_threshold": winner_not_candidate0_threshold,
    }


# ────────────────────────────────────────
# 综合入口
# ────────────────────────────────────────

def run_rerank_eval(
    scored_file: str,
    output_dir: str,
    candidates_file: Optional[str] = None,
    run_id: str = "",
    require_min_unique_satisfied: bool = True,
) -> Dict[str, Any]:
    """运行全部 rerank 评估，输出 JSON + CSV。

    Args:
        scored_file:     candidate_scores.jsonl 路径
        output_dir:      输出目录
        candidates_file: candidates.jsonl 路径（可选）。
                         若提供，则在 rerank 前做严格一致性校验：
                         两个文件必须来自同一次 pipeline run。
        run_id:          本次 run 的唯一 ID，写入 manifest；留空则自动生成。
    """
    with open(scored_file, "r", encoding="utf-8") as f:
        all_scored_records = [json.loads(line) for line in f if line.strip()]

    # ── Step 0: 一致性校验（必须在过滤前，用完整文件做 MD5/行数 provenance）──
    if candidates_file and os.path.isfile(candidates_file):
        validate_candidate_consistency(candidates_file, scored_file)
    elif candidates_file:
        print(f"[rerank_eval] WARNING: candidates_file 指定但不存在: {candidates_file}",
              file=sys.stderr)

    records = [
        r for r in all_scored_records
        if "error" not in r
        and "reward_error" not in r
        and (not r.get("action_logprob_ate_error"))
        and r.get("reward_record_status") != "error"
    ]

    groups_all = group_by_sample(records)
    pre_filter_n_samples = len(groups_all)

    missing_min_unique_samples: List[Any] = []
    groups = groups_all
    if require_min_unique_satisfied:
        filtered = {}
        for sid, cands in groups_all.items():
            ok = _sample_min_unique_status(cands)
            if ok is None:
                missing_min_unique_samples.append(sid)
                continue
            if ok:
                filtered[sid] = cands
        if missing_min_unique_samples:
            raise RuntimeError(
                "require_min_unique_satisfied=True but some samples have no min_unique_satisfied field; "
                f"count={len(missing_min_unique_samples)}"
            )
        groups = filtered

    n_samples = len(groups)
    excluded_by_min_unique = pre_filter_n_samples - n_samples
    n_candidate_rows = len(all_scored_records)  # 含 error 行，用于 manifest
    print(
        f"[rerank_eval] {len(records)} valid candidates, {n_samples} samples "
        f"(prefilter={pre_filter_n_samples}, excluded_by_min_unique={excluded_by_min_unique}, "
        f"require_min_unique_satisfied={require_min_unique_satisfied})"
    )
    if n_samples <= 0:
        raise RuntimeError("No samples left in main evaluation set after min_unique filtering")

    # ── Subsets: overall vs informative (candidate diversity exists) ──
    informative_ids = _informative_sample_ids(groups)
    strong_ids = _strong_informative_sample_ids(groups)
    informative_groups = {sid: c for sid, c in groups.items() if sid in informative_ids}
    strong_groups = {sid: c for sid, c in groups.items() if sid in strong_ids}
    print(
        f"[rerank_eval] informative subset: {len(informative_groups)}/{n_samples} samples "
        f"(strong={len(strong_groups)}/{n_samples})"
    )

    # Resolve logprob baseline key.
    logprob_score_key = _resolve_logprob_score_key(records)

    # ── Resolve run-active reward fields and schema summary ──
    sidecar = _load_reward_schema_sidecar(scored_file)
    if sidecar:
        schema_summary = sidecar
        verifier_reward_pool = list(
            sidecar.get("available_verifier_eligible_reward_fields")
            or sidecar.get("verifier_eligible_reward_fields")
            or sidecar.get("available_rerank_eligible_reward_fields")
            or sidecar.get("rerank_eligible_reward_fields")
            or []
        )
        training_reward_diag_pool = list(
            sidecar.get("available_training_reward_diagnostic_fields")
            or []
        )
    else:
        # Fallback: infer active fields (finite-only) from records.
        active_fields = infer_active_reward_fields_from_records(records, known_fields_only=True)
        available_fields = sorted({k for r in records for k in r.keys()})
        schema_summary = get_schema_summary(
            include_future=False,
            available_fields=available_fields,
            active_fields=active_fields,
        )
        verifier_reward_pool = list(
            schema_summary.get("available_verifier_eligible_reward_fields")
            or schema_summary.get("verifier_eligible_reward_fields")
            or []
        )
        training_reward_diag_pool = list(
            get_training_reward_diagnostic_fields(
                include_future=False,
                available_fields=active_fields,
            )
        )

    # Enforce verifier-only reward pool even for legacy sidecars.
    verifier_allowed = set(
        get_verifier_eligible_reward_fields(
            include_future=False,
            available_fields=verifier_reward_pool if verifier_reward_pool else None,
        )
    )
    if verifier_allowed:
        verifier_reward_pool = [rk for rk in verifier_reward_pool if rk in verifier_allowed]

    training_reward_diag_pool = [
        rk for rk in training_reward_diag_pool
        if rk not in verifier_reward_pool
    ]

    # ── Reward 退化检查（overall + informative） ──
    degeneracy = reward_degeneracy_check(groups, reward_keys=verifier_reward_pool)
    degeneracy_inf = reward_degeneracy_check(informative_groups, reward_keys=verifier_reward_pool)
    training_degeneracy = reward_degeneracy_check(groups, reward_keys=training_reward_diag_pool)
    training_degeneracy_inf = reward_degeneracy_check(informative_groups, reward_keys=training_reward_diag_pool)
    if degeneracy["selection_degeneracy_flag"]:
        print("[rerank_eval] ⚠ WARNING: ALL rewards are degenerate — "
              "candidates within each sample have near-identical scores. "
              "Rerank results will be meaningless.")
    else:
        ok_rewards = [k for k, v in degeneracy["per_reward"].items()
                      if not v["is_degenerate"]]
        print(f"[rerank_eval] degeneracy check: {len(ok_rewards)}/{len(degeneracy['per_reward'])} "
              "rewards are candidate-sensitive ✓")

    os.makedirs(output_dir, exist_ok=True)

    # Baselines
    baselines = evaluate_baselines(groups)
    baselines_inf = evaluate_baselines(informative_groups)
    baselines_strong = evaluate_baselines(strong_groups)

    twbor_ids = _top1_wrong_but_oracle_right_ids(groups)
    twbor_groups = {sid: c for sid, c in groups.items() if sid in twbor_ids}
    baselines_twbor = evaluate_baselines(twbor_groups)

    # Strategy evaluations (strictly forbid constant-within-sample rewards)
    constant_blocked = []
    rerank_reward_fields = []
    for rk in verifier_reward_pool:
        deg = degeneracy.get("per_reward", {}).get(rk, {})
        if float(deg.get("samplewise_nonconstant_rate", 0.0)) <= 0.0:
            constant_blocked.append(rk)
        else:
            rerank_reward_fields.append(rk)

    training_constant_blocked = []
    training_rerank_reward_fields = []
    for rk in training_reward_diag_pool:
        deg = training_degeneracy.get("per_reward", {}).get(rk, {})
        if float(deg.get("samplewise_nonconstant_rate", 0.0)) <= 0.0:
            training_constant_blocked.append(rk)
        else:
            training_rerank_reward_fields.append(rk)

    strategy_map = build_strategy_map(
        rerank_reward_fields,
        logprob_score_key=logprob_score_key,
    )
    training_strategy_map = build_training_diagnostic_strategy_map(training_rerank_reward_fields)
    strategy_results = {}
    all_selected: List[Dict] = []
    for strat, score_key in strategy_map.items():
        result = evaluate_strategy(groups, strat, score_key)
        result_inf = evaluate_strategy(informative_groups, strat, score_key)
        result_strong = evaluate_strategy(strong_groups, strat, score_key)
        result_twbor = evaluate_strategy(twbor_groups, strat, score_key)
        # Add informative subset metrics without breaking old keys.
        result["informative_total_samples"] = result_inf.get("total_samples", 0)
        result["informative_correct"] = result_inf.get("correct", 0)
        result["informative_accuracy"] = result_inf.get("accuracy", 0.0)
        result["informative_task_accuracy"] = result_inf.get("task_accuracy", {})
        result["strong_informative_total_samples"] = result_strong.get("total_samples", 0)
        result["strong_informative_correct"] = result_strong.get("correct", 0)
        result["strong_informative_accuracy"] = result_strong.get("accuracy", 0.0)
        result["top1_wrong_but_oracle_right_total_samples"] = result_twbor.get("total_samples", 0)
        result["top1_wrong_but_oracle_right_correct"] = result_twbor.get("correct", 0)
        result["top1_wrong_but_oracle_right_accuracy"] = result_twbor.get("accuracy", 0.0)

        strategy_results[strat] = result
        sel_records = result.pop("selected_records", [])
        for rec in sel_records:
            rec["pool"] = "verifier"
        all_selected.extend(sel_records)

    training_strategy_results = {}
    for strat, score_key in training_strategy_map.items():
        result = evaluate_strategy(groups, strat, score_key)
        result_inf = evaluate_strategy(informative_groups, strat, score_key)
        result_strong = evaluate_strategy(strong_groups, strat, score_key)
        result_twbor = evaluate_strategy(twbor_groups, strat, score_key)
        result["informative_total_samples"] = result_inf.get("total_samples", 0)
        result["informative_correct"] = result_inf.get("correct", 0)
        result["informative_accuracy"] = result_inf.get("accuracy", 0.0)
        result["informative_task_accuracy"] = result_inf.get("task_accuracy", {})
        result["strong_informative_total_samples"] = result_strong.get("total_samples", 0)
        result["strong_informative_correct"] = result_strong.get("correct", 0)
        result["strong_informative_accuracy"] = result_strong.get("accuracy", 0.0)
        result["top1_wrong_but_oracle_right_total_samples"] = result_twbor.get("total_samples", 0)
        result["top1_wrong_but_oracle_right_correct"] = result_twbor.get("correct", 0)
        result["top1_wrong_but_oracle_right_accuracy"] = result_twbor.get("accuracy", 0.0)
        training_strategy_results[strat] = result
        sel_records = result.pop("selected_records", [])
        for rec in sel_records:
            rec["pool"] = "training_reward_diagnostic"
        all_selected.extend(sel_records)

    main_experiment_compare = _build_main_experiment_comparison(strategy_results)
    if main_experiment_compare.get("warning"):
        print(f"[rerank_eval] ⚠ WARNING: {main_experiment_compare['warning']}")

    print(
        f"[rerank_eval] verifier strategies: total={len(strategy_map)} "
        f"(logprob + {len(rerank_reward_fields)} reward-based, blocked_constant={len(constant_blocked)})"
    )
    print(
        f"[rerank_eval] training diagnostic strategies: total={len(training_strategy_map)} "
        f"(reward-based={len(training_rerank_reward_fields)}, blocked_constant={len(training_constant_blocked)})"
    )

    # ── 事后一致性校验：selected 重新统计值 == strategy_results ──
    verify_selected_vs_results(
        [r for r in all_selected if r.get("pool") == "verifier"],
        strategy_results,
    )

    # ── 写 reward_degeneracy.json ──
    deg_path = os.path.join(output_dir, "reward_degeneracy.json")
    with open(deg_path, "w", encoding="utf-8") as f:
        out = dict(degeneracy)
        out["informative"] = degeneracy_inf
        out["training_reward_diagnostic"] = training_degeneracy
        out["training_reward_diagnostic_informative"] = training_degeneracy_inf
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[rerank_eval] degeneracy → {deg_path}")

    # ── 构建 rerank_summary（也包含 provenance run_id） ──
    _run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rerank_summary: Dict[str, Any] = {
        "run_id": _run_id,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "candidates_path": os.path.abspath(candidates_file) if candidates_file else None,
        "candidate_scores_path": os.path.abspath(scored_file),
        "subset_definitions": {
            "main_evaluation_filter": {
                "definition": "sample-level filter before all baselines/subsets/strategies",
                "require_min_unique_satisfied": bool(require_min_unique_satisfied),
                "prefilter_total_samples": pre_filter_n_samples,
                "postfilter_total_samples": n_samples,
                "excluded_samples_by_min_unique": excluded_by_min_unique,
                "missing_min_unique_satisfied_field_samples": len(missing_min_unique_samples),
            },
            "informative": {
                "definition": "within-sample unique normalized raw_generated_text >= 2 after filtering explicit duplicates",
                "total_samples": len(informative_groups),
            },
            "strong_informative": {
                "definition": "informative + candidate correctness not all identical (after duplicate filtering)",
                "total_samples": len(strong_groups),
            },
            "top1_wrong_but_oracle_right": {
                "definition": "samples where top1 is wrong but oracle over non-duplicate candidates is correct",
                "total_samples": len(twbor_groups),
            },
        },
        "logprob_baseline": {
            "strategy": BASE_STRATEGY_NAME,
            "score_key_used": logprob_score_key,
        },
        "reward_schema": schema_summary,
        "rerank_strategy_registry": {
            "strategy_to_score_key": strategy_map,
            "reward_fields_used_for_rerank": rerank_reward_fields,
            "reward_fields_blocked_constant_within_sample": constant_blocked,
            "reward_pool_semantics": "verifier_eligible_only",
        },
        "training_reward_diagnostic_registry": {
            "strategy_to_score_key": training_strategy_map,
            "reward_fields_used_for_rerank": training_rerank_reward_fields,
            "reward_fields_blocked_constant_within_sample": training_constant_blocked,
            "reward_pool_semantics": "training_reward_diagnostic_only",
        },
        "main_experiment_offline_compare": main_experiment_compare,
        "reward_degeneracy": degeneracy,
        "reward_degeneracy_informative": degeneracy_inf,
        "reward_degeneracy_training_reward_diagnostic": training_degeneracy,
        "reward_degeneracy_training_reward_diagnostic_informative": training_degeneracy_inf,
        "sample_filtering": {
            "require_min_unique_satisfied": bool(require_min_unique_satisfied),
            "prefilter_total_samples": pre_filter_n_samples,
            "postfilter_total_samples": n_samples,
            "excluded_samples_by_min_unique": excluded_by_min_unique,
            "excluded_ratio_by_min_unique": (
                round(excluded_by_min_unique / pre_filter_n_samples, 4)
                if pre_filter_n_samples > 0 else 0.0
            ),
            "missing_min_unique_satisfied_field_samples": len(missing_min_unique_samples),
        },
        "baselines": baselines,
        "baselines_informative": baselines_inf,
        "baselines_strong_informative": baselines_strong,
        "baselines_top1_wrong_but_oracle_right": baselines_twbor,
        "strategies": {},
    }
    for strat, result in strategy_results.items():
        rerank_summary["strategies"][strat] = {
            k: v for k, v in result.items() if k != "selected_records"
        }
        rerank_summary["strategies"][strat]["pool"] = "verifier"
    for strat, result in training_strategy_results.items():
        rerank_summary["strategies"][strat] = {
            k: v for k, v in result.items() if k != "selected_records"
        }
        rerank_summary["strategies"][strat]["pool"] = "training_reward_diagnostic"

    # ── Pairwise win/loss vs baselines (overall + informative) ──
    strat_correct: Dict[str, Dict[Any, bool]] = defaultdict(dict)
    for rec in all_selected:
        sid = rec.get("sample_id")
        sname = rec.get("strategy")
        if sid is None or not sname:
            continue
        strat_correct[sname][sid] = bool(rec.get("correct", False))

    top1_correct: Dict[Any, bool] = {}
    for sid, cands in groups.items():
        if not cands:
            continue
        top1 = next((c for c in cands if c.get("candidate_id") == 0), cands[0])
        top1_correct[sid] = bool(_is_correct(top1))

    pairwise = {}
    for strat in sorted(strategy_map.keys()):
        if strat not in strat_correct:
            continue
        vs_top1 = _pairwise_win_loss(strat_correct[strat], top1_correct)
        vs_lp = {}
        if BASE_STRATEGY_NAME in strat_correct and strat != BASE_STRATEGY_NAME:
            vs_lp = _pairwise_win_loss(strat_correct[strat], strat_correct[BASE_STRATEGY_NAME])

        pairwise[strat] = {
            "overall": {
                "vs_top1": vs_top1,
                "vs_select_by_logprob": vs_lp,
            },
            "informative": {
                "vs_top1": _pairwise_win_loss(
                    strat_correct[strat], top1_correct, sample_ids=informative_ids
                ),
                "vs_select_by_logprob": (
                    _pairwise_win_loss(
                        strat_correct[strat],
                        strat_correct[BASE_STRATEGY_NAME],
                        sample_ids=informative_ids,
                    )
                    if (BASE_STRATEGY_NAME in strat_correct and strat != BASE_STRATEGY_NAME)
                    else {}
                ),
            },
        }
    rerank_summary["pairwise_win_loss"] = pairwise

    rerank_summary["strategy_pools"] = {
        "verifier": sorted(strategy_map.keys()),
        "training_reward_diagnostic": sorted(training_strategy_map.keys()),
    }

    # ── 写 rerank_results.json ──
    with open(os.path.join(output_dir, "rerank_results.json"), "w", encoding="utf-8") as f:
        json.dump(rerank_summary, f, indent=2, ensure_ascii=False)

    # ── 写 rerank_results.csv (策略 × 指标) ──
    csv_path = os.path.join(output_dir, "rerank_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "pool", "accuracy", "correct", "total",
                         "informative_accuracy", "informative_correct", "informative_total",
                         "strong_informative_accuracy", "strong_informative_correct", "strong_informative_total",
                         "top1_wrong_but_oracle_right_accuracy", "top1_wrong_but_oracle_right_correct", "top1_wrong_but_oracle_right_total",
                         "avg_answer_length", "yesno_ratio",
                         "correct_positive", "correct_negative", "miss", "hallucination"])
        for bname, bdata in baselines.items():
            b_inf = baselines_inf.get(bname, {})
            b_strong = baselines_strong.get(bname, {})
            b_twbor = baselines_twbor.get(bname, {})
            writer.writerow([
                bname, "baseline", bdata["accuracy"], bdata["correct"], bdata["total_samples"],
                b_inf.get("accuracy", ""), b_inf.get("correct", ""), b_inf.get("total_samples", ""),
                b_strong.get("accuracy", ""), b_strong.get("correct", ""), b_strong.get("total_samples", ""),
                b_twbor.get("accuracy", ""), b_twbor.get("correct", ""), b_twbor.get("total_samples", ""),
                "", "", "", "", "", "",
            ])
        for strat, result in strategy_results.items():
            bd = result.get("behavior_breakdown", {})
            writer.writerow([
                strat, "verifier", result["accuracy"], result["correct"], result["total_samples"],
                result.get("informative_accuracy", ""),
                result.get("informative_correct", ""),
                result.get("informative_total_samples", ""),
                result.get("strong_informative_accuracy", ""),
                result.get("strong_informative_correct", ""),
                result.get("strong_informative_total_samples", ""),
                result.get("top1_wrong_but_oracle_right_accuracy", ""),
                result.get("top1_wrong_but_oracle_right_correct", ""),
                result.get("top1_wrong_but_oracle_right_total_samples", ""),
                result.get("avg_answer_length", ""),
                result.get("yesno_ratio", ""),
                bd.get("correct_positive", 0),
                bd.get("correct_negative", 0),
                bd.get("miss", 0),
                bd.get("hallucination", 0),
            ])
        for strat, result in training_strategy_results.items():
            bd = result.get("behavior_breakdown", {})
            writer.writerow([
                strat, "training_reward_diagnostic", result["accuracy"], result["correct"], result["total_samples"],
                result.get("informative_accuracy", ""),
                result.get("informative_correct", ""),
                result.get("informative_total_samples", ""),
                result.get("strong_informative_accuracy", ""),
                result.get("strong_informative_correct", ""),
                result.get("strong_informative_total_samples", ""),
                result.get("top1_wrong_but_oracle_right_accuracy", ""),
                result.get("top1_wrong_but_oracle_right_correct", ""),
                result.get("top1_wrong_but_oracle_right_total_samples", ""),
                result.get("avg_answer_length", ""),
                result.get("yesno_ratio", ""),
                bd.get("correct_positive", 0),
                bd.get("correct_negative", 0),
                bd.get("miss", 0),
                bd.get("hallucination", 0),
            ])

    main_csv_path = os.path.join(output_dir, "main_reward_mode_comparison.csv")
    with open(main_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "reward_mode", "strategy", "accuracy", "correct", "total_samples",
            "informative_accuracy", "informative_correct", "informative_total_samples",
            "strong_informative_accuracy",
        ])
        for row in main_experiment_compare.get("table", []):
            writer.writerow([
                row.get("reward_mode"),
                row.get("strategy"),
                row.get("accuracy"),
                row.get("correct"),
                row.get("total_samples"),
                row.get("informative_accuracy"),
                row.get("informative_correct"),
                row.get("informative_total_samples"),
                row.get("strong_informative_accuracy"),
            ])

    # ── 写 taskwise_results.csv ──
    tw_path = os.path.join(output_dir, "taskwise_results.csv")
    with open(tw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "task_type", "accuracy", "correct", "total"])
        for source_name, source_data in [("baselines", baselines),
                                         ("strategies", strategy_results)]:
            for name, data in source_data.items():
                ta = data.get("task_accuracy", {})
                for tt, tdata in sorted(ta.items()):
                    writer.writerow([
                        name, tt,
                        tdata.get("accuracy", 0),
                        tdata.get("correct", 0),
                        tdata.get("total", 0),
                    ])

    # ── 写 selected_candidates_by_strategy.jsonl ──
    sel_path = os.path.join(output_dir, "selected_candidates_by_strategy.jsonl")
    with open(sel_path, "w", encoding="utf-8") as f:
        for rec in all_selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── 写 verifier_manifest.json ──
    write_verifier_manifest(
        output_dir=output_dir,
        run_id=_run_id,
        candidates_file=candidates_file,
        scored_file=scored_file,
        n_samples=n_samples,
        n_candidate_rows=n_candidate_rows,
    )

    print(f"[rerank_eval] results  → {output_dir}/rerank_results.json")
    print(f"[rerank_eval] csv      → {csv_path}")
    print(f"[rerank_eval] main csv → {main_csv_path}")
    print(f"[rerank_eval] taskwise → {tw_path}")
    print(f"[rerank_eval] selected → {sel_path}")

    return rerank_summary


# ────────────────────────────────────────
# CLI
# ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Rerank evaluation of candidate answers")
    ap.add_argument("--scored_file", required=True,
                    help="candidate_scores.jsonl from verifier_reward.py")
    ap.add_argument("--candidates_file", default="",
                    help="candidates.jsonl（可选，提供时做严格一致性校验）")
    ap.add_argument("--output_dir", default="results/verifier")
    ap.add_argument("--run_id", default="",
                    help="本次 run 的唯一标识，写入 verifier_manifest.json")
    ap.add_argument(
        "--require_min_unique_satisfied",
        dest="require_min_unique_satisfied",
        action="store_true",
        default=True,
        help="主评估集合仅保留 min_unique_satisfied=True 的样本（默认开启）",
    )
    ap.add_argument(
        "--allow_min_unique_unsatisfied",
        dest="require_min_unique_satisfied",
        action="store_false",
        help="关闭 min_unique 硬过滤（仅用于兼容分析，不建议用于 verifier 主结论）",
    )
    args = ap.parse_args()

    run_rerank_eval(
        args.scored_file,
        args.output_dir,
        candidates_file=args.candidates_file or None,
        run_id=args.run_id,
        require_min_unique_satisfied=args.require_min_unique_satisfied,
    )


if __name__ == "__main__":
    main()
