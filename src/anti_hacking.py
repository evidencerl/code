"""Anti-hacking / sanity-check 分析模块。

检测 reward 是否在钻空子：
1. reward 与 answer length 的相关性
2. reward 与 yes/no 风格答案的相关性
3. reward 在不同 task_type 上的分布
4. reward 高分候选是否真的更可能 correct
5. 各策略选出的答案是否更短/更模板化
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from answer_format_utils import infer_task_family
from reward.schema import get_verifier_eligible_reward_fields, REWARD_SCHEMA_VERSION
from style_bias_utils import compute_style_bias_features


def _normalize_answer_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _informative_sample_ids(records: List[Dict[str, Any]]) -> set:
    """Informative subset: within-sample unique normalized raw responses >= 2 after duplicate filtering."""
    by_sample: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        sid = r.get("sample_id", r.get("index"))
        if sid is None:
            continue
        by_sample[sid].append(r)

    ids = set()
    for sid, cands in by_sample.items():
        cands_eff = [c for c in cands if not bool(c.get("is_duplicate_candidate", False))]
        if not cands_eff:
            cands_eff = cands
        texts = [_normalize_answer_text(c.get("raw_generated_text", c.get("generated_text", ""))) for c in cands_eff]
        texts = [t for t in texts if t]
        if len(set(texts)) >= 2:
            ids.add(sid)
    return ids


def _choose_length_field(records: List[Dict[str, Any]]) -> str:
    """Prefer response_length when available.

    Note: action_logprob_ate 使用 raw response_text，长度投机应看 response 长度。
    """
    for r in records:
        if "response_length" in r:
            return "response_length"
    return "answer_length"


def _style_feature_view(record: Dict[str, Any]) -> Dict[str, Any]:
    answer_text = record.get("generated_text") or record.get("final_answer") or record.get("answer_text") or record.get("raw_generated_text") or ""
    raw_response = record.get("raw_generated_text") or record.get("response_text") or answer_text
    parsed = record.get("parsed_answer") or record.get("parse") or ""
    task_type = record.get("task_type") or ""
    is_parse_fail = bool(record.get("is_parse_fail", False))
    return compute_style_bias_features(
        answer_text,
        raw_response=raw_response,
        parsed_answer=parsed,
        task_type=task_type,
        is_parse_fail=is_parse_fail,
    )


def _load_reward_schema_sidecar(scored_file: str) -> Dict[str, Any]:
    try:
        schema_path = os.path.join(os.path.dirname(scored_file) or ".", "reward_schema.json")
        if os.path.isfile(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}


def _safe_corr(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation, nan-safe."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 3:
        return float("nan")
    n = len(pairs)
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] ** 2 for p in pairs)
    syy = sum(p[1] ** 2 for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    denom = math.sqrt(max(n * sxx - sx * sx, 0)) * math.sqrt(max(n * syy - sy * sy, 0))
    if denom < 1e-15:
        return float("nan")
    return (n * sxy - sx * sy) / denom


def _safe_mean(xs: List[float]) -> float:
    valid = [x for x in xs if not math.isnan(x)]
    return sum(valid) / len(valid) if valid else float("nan")


def _safe_std(xs: List[float]) -> float:
    valid = [x for x in xs if not math.isnan(x)]
    if len(valid) < 2:
        return float("nan")
    m = sum(valid) / len(valid)
    var = sum((x - m) ** 2 for x in valid) / (len(valid) - 1)
    return math.sqrt(var)


def run_anti_hacking(
    scored_file: str,
    output_dir: str,
    rerank_results: Dict = None,
) -> Dict[str, Any]:
    """运行 anti-hacking 分析。"""
    with open(scored_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [
        r for r in records
        if "error" not in r
        and "reward_error" not in r
        and (not r.get("action_logprob_ate_error"))
        and r.get("reward_record_status") != "error"
    ]

    os.makedirs(output_dir, exist_ok=True)
    summary: Dict[str, Any] = {}
    summary["run_id"] = rerank_results.get("run_id") if rerank_results else None
    summary["candidates_path"] = rerank_results.get("candidates_path") if rerank_results else None
    summary["candidate_scores_path"] = rerank_results.get("candidate_scores_path") if rerank_results else os.path.abspath(scored_file)

    # Only analyze rewards that are truly used for rerank.
    if rerank_results:
        reward_names = rerank_results.get("rerank_strategy_registry", {}).get(
            "reward_fields_used_for_rerank", []
        )
    else:
        reward_names = []
    if not reward_names:
        sidecar = _load_reward_schema_sidecar(scored_file)
        reward_names = list(
            sidecar.get("available_rerank_eligible_reward_fields")
            or sidecar.get("rerank_eligible_reward_fields")
            or []
        )
    if not reward_names:
        reward_names = get_verifier_eligible_reward_fields(include_future=False)

    summary["reward_schema_version"] = REWARD_SCHEMA_VERSION
    summary["analyzed_reward_fields"] = reward_names

    length_field = _choose_length_field(records)
    summary["length_field_used"] = length_field

    # 主实验默认 telemetry：统一补齐 family / yes-no / intervention 维度。
    task_family_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "avg_response_length": 0.0,
        "negation_rate": 0.0,
        "proposal_fallback_rate": 0.0,
        "bbox_missing_rate": 0.0,
    })
    yes_count = 0
    no_count = 0
    intervention_area = []
    intervention_reward = []
    for r in records:
        fam = r.get("task_family") or infer_task_family(
            task_type=r.get("task_type", ""),
            question=r.get("question", ""),
            metadata=r,
        )
        stats = task_family_stats[fam]
        stats["count"] += 1
        stats["avg_response_length"] += float(r.get("response_length", r.get("answer_length", 0.0)))
        feat = _style_feature_view(r)
        stats["negation_rate"] += 1.0 if feat.get("has_negation") else 0.0
        stats["proposal_fallback_rate"] += 1.0 if r.get("action_logprob_ate_proposal_fallback_used") else 0.0
        stats["bbox_missing_rate"] += 1.0 if r.get("action_logprob_ate_proposal_bbox_missing") else 0.0
        yn = (r.get("yesno_label") or "").strip().lower()
        if yn == "yes":
            yes_count += 1
        elif yn == "no":
            no_count += 1
        area = r.get("action_logprob_ate_proposal_area_fraction")
        reward = r.get("action_logprob_ate_main_reward_active")
        try:
            area_f = float(area)
            reward_f = float(reward)
            if not (math.isnan(area_f) or math.isnan(reward_f)):
                intervention_area.append(area_f)
                intervention_reward.append(reward_f)
        except Exception:
            pass

    for fam, stats in task_family_stats.items():
        denom = float(max(1, stats["count"]))
        stats["avg_response_length"] /= denom
        stats["negation_rate"] /= denom
        stats["proposal_fallback_rate"] /= denom
        stats["bbox_missing_rate"] /= denom

    summary["task_family_stats"] = dict(task_family_stats)
    summary["yes_rate"] = yes_count / max(1, yes_count + no_count)
    summary["no_rate"] = no_count / max(1, yes_count + no_count)
    summary["negation_rate"] = round(_safe_mean([_style_feature_view(r).get("has_negation", False) * 1.0 for r in records]), 6)
    summary["avg_response_length"] = round(_safe_mean([float(r.get("response_length", r.get("answer_length", 0))) for r in records]), 6)
    summary["zero_variance_fallback_rate"] = round(_safe_mean([1.0 if r.get("used_zero_variance_fallback") else 0.0 for r in records]), 6)
    summary["raw_reward_nonconstant_group_rate"] = round(_safe_mean([1.0 if not r.get("raw_reward_identical", False) else 0.0 for r in records]), 6)
    summary["same_final_answer_diff_raw_reward"] = round(_safe_mean([1.0 if r.get("same_final_answer_diff_raw_reward") else 0.0 for r in records]), 6)
    summary["final_answer_identical_rate"] = round(_safe_mean([1.0 if r.get("final_answer_identical") else 0.0 for r in records]), 6)
    summary["intervention_area_vs_reward_correlation"] = round(_safe_corr(intervention_area, intervention_reward), 6)

    # ────── 1. reward 与 length 的相关性 ──────
    length_corr = {}
    for rname in reward_names:
        rewards = [float(r.get(rname, float("nan"))) for r in records]
        lengths = [float(r.get(length_field, 0)) for r in records]
        length_corr[rname] = round(_safe_corr(rewards, lengths), 4)
    summary["reward_length_correlation"] = length_corr

    # ────── 2. reward 与 yes/no 风格的关系 ──────
    yesno_bias = {}
    for rname in reward_names:
        yn_rewards = []
        non_yn_rewards = []
        for r in records:
            v = r.get(rname)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if r.get("is_yesno", False):
                yn_rewards.append(float(v))
            else:
                non_yn_rewards.append(float(v))
        yesno_bias[rname] = {
            "yesno_mean": round(_safe_mean(yn_rewards), 6),
            "non_yesno_mean": round(_safe_mean(non_yn_rewards), 6),
            "yesno_count": len(yn_rewards),
            "non_yesno_count": len(non_yn_rewards),
            "gap": round(_safe_mean(yn_rewards) - _safe_mean(non_yn_rewards), 6)
                  if yn_rewards and non_yn_rewards else float("nan"),
        }
    summary["reward_yesno_bias"] = yesno_bias

    # ────── 3. reward 在不同 task_type 的分布 ──────
    task_dist: Dict[str, Dict[str, Dict]] = {}
    for rname in reward_names:
        by_task: Dict[str, List[float]] = defaultdict(list)
        for r in records:
            v = r.get(rname)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            tt = r.get("task_type", "unknown")
            by_task[tt].append(float(v))
        task_dist[rname] = {}
        for tt, vals in sorted(by_task.items()):
            task_dist[rname][tt] = {
                "mean": round(_safe_mean(vals), 6),
                "std": round(_safe_std(vals), 6),
                "count": len(vals),
            }
    summary["reward_task_distribution"] = task_dist

    # ────── 4. reward 高分候选 vs 低分候选的 correctness ──────
    high_low_correct = {}
    for rname in reward_names:
        scored_pairs = []
        for r in records:
            v = r.get(rname)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            c = r.get("correct", False)
            scored_pairs.append((float(v), bool(c)))
        if len(scored_pairs) < 10:
            high_low_correct[rname] = {"insufficient_data": True}
            continue
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        n = len(scored_pairs)
        top_q = scored_pairs[:n // 4]
        bot_q = scored_pairs[-(n // 4):]
        top_acc = sum(1 for _, c in top_q if c) / len(top_q) if top_q else 0
        bot_acc = sum(1 for _, c in bot_q if c) / len(bot_q) if bot_q else 0
        high_low_correct[rname] = {
            "top_25pct_accuracy": round(top_acc, 4),
            "bottom_25pct_accuracy": round(bot_acc, 4),
            "lift": round(top_acc - bot_acc, 4),
            "top_count": len(top_q),
            "bottom_count": len(bot_q),
        }
    summary["reward_correctness_lift"] = high_low_correct

    warnings = {
        "negative_correctness_lift_rewards": [],
        "large_negative_lift_rewards": [],
    }
    for rname, stats in high_low_correct.items():
        if stats.get("insufficient_data"):
            continue
        lift = stats.get("lift")
        if isinstance(lift, (int, float)) and lift < 0:
            warnings["negative_correctness_lift_rewards"].append(rname)
        if isinstance(lift, (int, float)) and lift <= -0.05:
            warnings["large_negative_lift_rewards"].append(rname)
    summary["warnings"] = warnings

    # ────── Informative subset: 同题至少 2 个 unique final answer ──────
    info_ids = _informative_sample_ids(records)
    info_records = [r for r in records if r.get("sample_id", r.get("index")) in info_ids]
    info_length_field = _choose_length_field(info_records)

    info_length_corr = {}
    for rname in reward_names:
        rewards = [float(r.get(rname, float("nan"))) for r in info_records]
        lengths = [float(r.get(info_length_field, 0)) for r in info_records]
        info_length_corr[rname] = round(_safe_corr(rewards, lengths), 4)

    info_yesno_bias = {}
    for rname in reward_names:
        yn_rewards = []
        non_yn_rewards = []
        for r in info_records:
            v = r.get(rname)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if r.get("is_yesno", False):
                yn_rewards.append(float(v))
            else:
                non_yn_rewards.append(float(v))
        info_yesno_bias[rname] = {
            "yesno_mean": round(_safe_mean(yn_rewards), 6),
            "non_yesno_mean": round(_safe_mean(non_yn_rewards), 6),
            "yesno_count": len(yn_rewards),
            "non_yesno_count": len(non_yn_rewards),
            "gap": round(_safe_mean(yn_rewards) - _safe_mean(non_yn_rewards), 6)
                  if yn_rewards and non_yn_rewards else float("nan"),
        }

    info_task_dist: Dict[str, Dict[str, Dict]] = {}
    for rname in reward_names:
        by_task: Dict[str, List[float]] = defaultdict(list)
        for r in info_records:
            v = r.get(rname)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            tt = r.get("task_type", "unknown")
            by_task[tt].append(float(v))
        info_task_dist[rname] = {}
        for tt, vals in sorted(by_task.items()):
            info_task_dist[rname][tt] = {
                "mean": round(_safe_mean(vals), 6),
                "std": round(_safe_std(vals), 6),
                "count": len(vals),
            }

    info_high_low_correct = {}
    for rname in reward_names:
        scored_pairs = []
        for r in info_records:
            v = r.get(rname)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            c = r.get("correct", False)
            scored_pairs.append((float(v), bool(c)))
        if len(scored_pairs) < 10:
            info_high_low_correct[rname] = {"insufficient_data": True}
            continue
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        n = len(scored_pairs)
        top_q = scored_pairs[:n // 4]
        bot_q = scored_pairs[-(n // 4):]
        top_acc = sum(1 for _, c in top_q if c) / len(top_q) if top_q else 0
        bot_acc = sum(1 for _, c in bot_q if c) / len(bot_q) if bot_q else 0
        info_high_low_correct[rname] = {
            "top_25pct_accuracy": round(top_acc, 4),
            "bottom_25pct_accuracy": round(bot_acc, 4),
            "lift": round(top_acc - bot_acc, 4),
            "top_count": len(top_q),
            "bottom_count": len(bot_q),
        }

    info_warnings = {
        "negative_correctness_lift_rewards": [],
        "large_negative_lift_rewards": [],
    }
    for rname, stats in info_high_low_correct.items():
        if stats.get("insufficient_data"):
            continue
        lift = stats.get("lift")
        if isinstance(lift, (int, float)) and lift < 0:
            info_warnings["negative_correctness_lift_rewards"].append(rname)
        if isinstance(lift, (int, float)) and lift <= -0.05:
            info_warnings["large_negative_lift_rewards"].append(rname)

    summary["informative_subset"] = {
        "definition": "within-sample unique normalized raw_generated_text >= 2 after filtering explicit duplicates",
        "total_samples": len(info_ids),
        "total_candidates": len(info_records),
        "length_field_used": info_length_field,
        "reward_length_correlation": info_length_corr,
        "reward_yesno_bias": info_yesno_bias,
        "reward_task_distribution": info_task_dist,
        "reward_correctness_lift": info_high_low_correct,
        "warnings": info_warnings,
    }

    # ────── 4.5 style / nuisance buckets ──────
    style_bucket_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    style_flags = {
        "has_negation": lambda f: bool(f.get("has_negation", False)),
        "is_abstention_like": lambda f: bool(f.get("is_abstention_like", False)),
        "is_template_like": lambda f: bool(f.get("is_template_like", False)),
        "is_numeric_short": lambda f: bool(f.get("is_numeric_short", False)),
    }
    style_views = [_style_feature_view(r) for r in records]
    for rname in reward_names:
        style_bucket_summary[rname] = {}
        vals = []
        for rec, feat in zip(records, style_views):
            v = rec.get(rname)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            vals.append((float(v), bool(rec.get("correct", False)), feat))
        for flag_name, pred in style_flags.items():
            pos = [(rv, ok) for rv, ok, feat in vals if pred(feat)]
            neg = [(rv, ok) for rv, ok, feat in vals if not pred(feat)]
            style_bucket_summary[rname][flag_name] = {
                "positive_count": len(pos),
                "negative_count": len(neg),
                "positive_reward_mean": round(_safe_mean([x for x, _ in pos]), 6),
                "negative_reward_mean": round(_safe_mean([x for x, _ in neg]), 6),
                "positive_correct_rate": round(_safe_mean([1.0 if c else 0.0 for _, c in pos]), 6),
                "negative_correct_rate": round(_safe_mean([1.0 if c else 0.0 for _, c in neg]), 6),
            }
    summary["reward_style_bucket_summary"] = style_bucket_summary

    # ────── 5. 各策略选出答案的长度/模板化分析 ──────
    # 从 rerank_results 或重新计算
    if rerank_results and "strategies" in rerank_results:
        strategy_template_risk = {}
        for strat, sdata in rerank_results["strategies"].items():
            avg_len = sdata.get("avg_answer_length", 0)
            yn_ratio = sdata.get("yesno_ratio", 0)
            # 启发式风险标记：答案平均长度 < 2 词 且 yesno 比例 > 0.8
            risk_short = avg_len < 2
            risk_template = yn_ratio > 0.85
            strategy_template_risk[strat] = {
                "avg_answer_length": avg_len,
                "yesno_ratio": yn_ratio,
                "risk_too_short": risk_short,
                "risk_too_template": risk_template,
                "overall_risk": risk_short or risk_template,
            }
        summary["strategy_template_risk"] = strategy_template_risk

    # ────── 写 JSON ──────
    json_path = os.path.join(output_dir, "anti_hacking_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ────── 写 reward_correlations.csv ──────
    csv_path = os.path.join(output_dir, "reward_correlations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "reward_name",
            "corr_with_length",
            "yesno_mean", "non_yesno_mean", "yesno_gap",
            "top25_accuracy", "bottom25_accuracy", "correctness_lift",
        ])
        for rname in reward_names:
            lc = length_corr.get(rname, float("nan"))
            yb = yesno_bias.get(rname, {})
            hlc = high_low_correct.get(rname, {})
            writer.writerow([
                rname,
                lc,
                yb.get("yesno_mean", ""),
                yb.get("non_yesno_mean", ""),
                yb.get("gap", ""),
                hlc.get("top_25pct_accuracy", ""),
                hlc.get("bottom_25pct_accuracy", ""),
                hlc.get("lift", ""),
            ])

    print(f"[anti_hacking] summary → {json_path}")
    print(f"[anti_hacking] correlations → {csv_path}")

    # Additive: correlations for informative subset
    try:
        info_csv_path = os.path.join(output_dir, "reward_correlations_informative.csv")
        with open(info_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "reward_name",
                "corr_with_length",
                "yesno_mean", "non_yesno_mean", "yesno_gap",
                "top25_accuracy", "bottom25_accuracy", "correctness_lift",
            ])
            for rname in reward_names:
                lc = info_length_corr.get(rname, float("nan"))
                yb = info_yesno_bias.get(rname, {})
                hlc = info_high_low_correct.get(rname, {})
                writer.writerow([
                    rname,
                    lc,
                    yb.get("yesno_mean", ""),
                    yb.get("non_yesno_mean", ""),
                    yb.get("gap", ""),
                    hlc.get("top_25pct_accuracy", ""),
                    hlc.get("bottom_25pct_accuracy", ""),
                    hlc.get("lift", ""),
                ])
        print(f"[anti_hacking] correlations (informative) → {info_csv_path}")
    except Exception:
        pass
    return summary


# ────────────────────────────────────────
# CLI
# ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Anti-hacking analysis for verifier rewards")
    ap.add_argument("--scored_file", required=True)
    ap.add_argument("--output_dir", default="results/verifier")
    ap.add_argument("--rerank_results", default="",
                    help="rerank_results.json (optional, for template risk)")
    args = ap.parse_args()

    rerank = None
    if args.rerank_results and os.path.isfile(args.rerank_results):
        with open(args.rerank_results) as f:
            rerank = json.load(f)

    run_anti_hacking(args.scored_file, args.output_dir, rerank_results=rerank)


if __name__ == "__main__":
    main()
