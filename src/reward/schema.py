"""Unified reward schema for verifier/rerank pipeline.

This module defines a static registry ("supported rewards") and provides helpers
to derive *run-active* reward fields (computed + finite in the current output).

Design goals:
1) Separate candidate-level rewards from sample-level probes.
2) Explicitly mark rerank-eligible rewards.
3) Keep compatibility with legacy field names while preventing misuse.
4) Prevent "future" placeholders from silently entering evaluation.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Iterable

import math

REWARD_SCHEMA_VERSION = "v6_main_experiment_task_routed_reward"

MAIN_EXPERIMENT_REWARD_MODES: Dict[str, Dict[str, Any]] = {
    "correctness_only": {
        "uses_evidence": False,
        "default_for_family": [],
        "allowed_families": ["counting", "attribute", "spatial", "existence", "other"],
        "description": "只使用 correctness 奖励，不引入视觉证据分数",
    },
    "additive_evidence": {
        "uses_evidence": True,
        "default_for_family": [],
        "allowed_families": ["counting", "attribute", "spatial"],
        "description": "在 correctness 上直接叠加 relative evidence margin",
    },
    "routed_gated_evidence": {
        "uses_evidence": True,
        "default_for_family": ["counting", "attribute", "spatial"],
        "allowed_families": ["counting", "attribute", "spatial"],
        "description": "默认主方法：task-routed + correctness-anchored gated evidence",
    },
}

# NOTE:
# - scope="candidate": candidate-specific and should vary within sample.
# - scope="sample_probe": sample-level diagnostics; must NOT drive rerank.
REWARD_SCHEMA: Dict[str, Dict[str, Any]] = {
    "reward_logits_js": {
        "scope": "candidate",
        "source": "ced",
        "family": "counterfactual",
        "rerank_eligible": True,
        "verifier_eligible": True,
        "grpo_eligible": False,
        "uses_ground_truth": False,
        "label_conditioned": False,
        "verifier_only": True,
        "training_only": False,
        "strategy": "select_by_reward_logits_js",
        "description": "CED logits JS divergence at answer-conditioned key positions",
        "response_conditioned": False,
        "candidate_dependent": True,
        "default_training_reward": False,
        "diagnostic_only": False,
        "unsafe_as_verifier": False,
        "unsafe_as_group_reward": True,
    },
    "reward_logits_cosine_dist": {
        "scope": "candidate",
        "source": "ced",
        "family": "counterfactual",
        "rerank_eligible": True,
        "verifier_eligible": True,
        "grpo_eligible": False,
        "uses_ground_truth": False,
        "label_conditioned": False,
        "verifier_only": True,
        "training_only": False,
        "strategy": "select_by_reward_logits_cosine_dist",
        "description": "CED logits cosine distance at answer-conditioned key positions",
        "response_conditioned": False,
        "candidate_dependent": True,
        "default_training_reward": False,
        "diagnostic_only": False,
        "unsafe_as_verifier": False,
        "unsafe_as_group_reward": True,
    },
    "reward_layer24_prompt_last_cosine": {
        "scope": "sample_probe",
        "source": "ced",
        "family": "probe",
        "rerank_eligible": False,
        "verifier_eligible": False,
        "grpo_eligible": False,
        "uses_ground_truth": False,
        "label_conditioned": False,
        "verifier_only": False,
        "training_only": False,
        "strategy": None,
        "description": "Layer24 prompt_last cosine probe (sample-level, often constant-within-sample)",
        "exclude_reason": "sample_level_probe",
        "response_conditioned": False,
        "candidate_dependent": False,
        "default_training_reward": False,
        "diagnostic_only": True,
        "unsafe_as_verifier": True,
        "unsafe_as_group_reward": True,
    },
    "reward_mix_js_promptlast": {
        "scope": "sample_probe",
        "source": "derived",
        "family": "mix",
        "rerank_eligible": False,
        "verifier_eligible": False,
        "grpo_eligible": False,
        "uses_ground_truth": False,
        "label_conditioned": False,
        "verifier_only": False,
        "training_only": False,
        "strategy": None,
        "description": "alpha*js + beta*prompt_last_probe",
        "exclude_reason": "fake_mix_rank_equivalent_when_probe_constant",
        "response_conditioned": False,
        "candidate_dependent": False,
        "default_training_reward": False,
        "diagnostic_only": True,
        "unsafe_as_verifier": True,
        "unsafe_as_group_reward": True,
    },
    "reward_mix_cosine_promptlast": {
        "scope": "sample_probe",
        "source": "derived",
        "family": "mix",
        "rerank_eligible": False,
        "verifier_eligible": False,
        "grpo_eligible": False,
        "uses_ground_truth": False,
        "label_conditioned": False,
        "verifier_only": False,
        "training_only": False,
        "strategy": None,
        "description": "alpha*cosine + beta*prompt_last_probe",
        "exclude_reason": "fake_mix_rank_equivalent_when_probe_constant",
        "response_conditioned": False,
        "candidate_dependent": False,
        "default_training_reward": False,
        "diagnostic_only": True,
        "unsafe_as_verifier": True,
        "unsafe_as_group_reward": True,
    },
    "reward_action_logprob_ate": {
        "scope": "candidate",
        "source": "response_conditioned",
        "family": "action",
        "rerank_eligible": False,
        "verifier_eligible": False,
        "grpo_eligible": False,
        "uses_ground_truth": True,
        "label_conditioned": True,
        "verifier_only": False,
        "training_only": True,
        "strategy": None,
        "description": "Legacy alias; maps to supervised action reward semantics and is not the default training reward",
        "exclude_reason": "ambiguous_alias_not_for_default_training",
        "response_conditioned": True,
        "candidate_dependent": True,
        "default_training_reward": False,
        "diagnostic_only": True,
        "unsafe_as_verifier": True,
        "unsafe_as_group_reward": True,
    },
    "reward_action_logprob_ate_supervised": {
        "scope": "candidate",
        "source": "response_conditioned",
        "family": "action",
        "rerank_eligible": False,
        "verifier_eligible": False,
        "grpo_eligible": True,
        "uses_ground_truth": True,
        "label_conditioned": True,
        "verifier_only": False,
        "training_only": True,
        "strategy": "select_by_reward_action_logprob_ate_supervised",
        "description": "Action reward with supervised shaping and correctness bias",
        "exclude_reason": "label_conditioned_training_reward",
        "response_conditioned": True,
        "candidate_dependent": True,
        "default_training_reward": False,
        "diagnostic_only": False,
        "unsafe_as_verifier": True,
        "unsafe_as_group_reward": False,
    },
    "reward_action_logprob_ate_nolabel": {
        "scope": "candidate",
        "source": "response_conditioned",
        "family": "action",
        "rerank_eligible": False,
        "verifier_eligible": False,
        "grpo_eligible": True,
        "uses_ground_truth": False,
        "label_conditioned": False,
        "verifier_only": False,
        "training_only": True,
        "strategy": "select_by_reward_action_logprob_ate_nolabel",
        "description": "Response-conditioned final-answer ATE (no-label continuous signal; score_base_no_label)",
        "exclude_reason": "training_reward_not_verifier",
        "response_conditioned": True,
        "candidate_dependent": True,
        "default_training_reward": True,
        "diagnostic_only": False,
        "unsafe_as_verifier": True,
        "unsafe_as_group_reward": False,
    },
    "reward_main_correctness_only": {
        "scope": "candidate",
        "source": "main_experiment",
        "family": "task_routed",
        "rerank_eligible": True,
        "verifier_eligible": True,
        "grpo_eligible": True,
        "uses_ground_truth": True,
        "label_conditioned": True,
        "verifier_only": False,
        "training_only": False,
        "strategy": "select_by_reward_main_correctness_only",
        "description": "Main experiment correctness-only reward",
        "response_conditioned": False,
        "candidate_dependent": True,
        "default_training_reward": False,
        "diagnostic_only": False,
        "unsafe_as_verifier": False,
        "unsafe_as_group_reward": False,
    },
    "reward_main_additive_evidence": {
        "scope": "candidate",
        "source": "main_experiment",
        "family": "task_routed",
        "rerank_eligible": True,
        "verifier_eligible": True,
        "grpo_eligible": True,
        "uses_ground_truth": True,
        "label_conditioned": True,
        "verifier_only": False,
        "training_only": False,
        "strategy": "select_by_reward_main_additive_evidence",
        "description": "Main experiment additive correctness + evidence reward",
        "response_conditioned": True,
        "candidate_dependent": True,
        "default_training_reward": False,
        "diagnostic_only": False,
        "unsafe_as_verifier": False,
        "unsafe_as_group_reward": False,
    },
    "reward_main_routed_gated_evidence": {
        "scope": "candidate",
        "source": "main_experiment",
        "family": "task_routed",
        "rerank_eligible": True,
        "verifier_eligible": True,
        "grpo_eligible": True,
        "uses_ground_truth": True,
        "label_conditioned": True,
        "verifier_only": False,
        "training_only": False,
        "strategy": "select_by_reward_main_routed_gated_evidence",
        "description": "Main experiment default routed-gated evidence reward",
        "response_conditioned": True,
        "candidate_dependent": True,
        "default_training_reward": True,
        "diagnostic_only": False,
        "unsafe_as_verifier": False,
        "unsafe_as_group_reward": False,
    },
}


def _is_finite_number(v: Any) -> bool:
    try:
        fv = float(v)
    except Exception:
        return False
    return not (math.isnan(fv) or math.isinf(fv))


def get_all_reward_fields() -> List[str]:
    return list(REWARD_SCHEMA.keys())


def get_main_experiment_reward_modes() -> List[str]:
    return list(MAIN_EXPERIMENT_REWARD_MODES.keys())


def get_default_main_experiment_reward_mode() -> str:
    return "routed_gated_evidence"


def get_main_experiment_reward_mode_schema() -> Dict[str, Dict[str, Any]]:
    return dict(MAIN_EXPERIMENT_REWARD_MODES)


def get_candidate_reward_fields(include_future: bool = False) -> List[str]:
    out = []
    for name, meta in REWARD_SCHEMA.items():
        if meta.get("scope") != "candidate":
            continue
        if meta.get("future") and not include_future:
            continue
        out.append(name)
    return out


def get_sample_probe_fields() -> List[str]:
    return [n for n, m in REWARD_SCHEMA.items() if m.get("scope") == "sample_probe"]


def get_rerank_eligible_reward_fields(
    include_future: bool = False,
    available_fields: Optional[List[str]] = None,
) -> List[str]:
    out = []
    avail = set(available_fields) if available_fields is not None else None
    for name, meta in REWARD_SCHEMA.items():
        if not meta.get("rerank_eligible", False):
            continue
        if meta.get("future") and not include_future:
            continue
        if avail is not None and name not in avail:
            continue
        out.append(name)
    return out


def get_verifier_eligible_reward_fields(
    include_future: bool = False,
    available_fields: Optional[List[str]] = None,
) -> List[str]:
    out = []
    avail = set(available_fields) if available_fields is not None else None
    for name, meta in REWARD_SCHEMA.items():
        if not meta.get("verifier_eligible", False):
            continue
        if meta.get("future") and not include_future:
            continue
        if avail is not None and name not in avail:
            continue
        out.append(name)
    return out


def get_grpo_eligible_reward_fields(
    include_future: bool = False,
    available_fields: Optional[List[str]] = None,
) -> List[str]:
    out = []
    avail = set(available_fields) if available_fields is not None else None
    for name, meta in REWARD_SCHEMA.items():
        if not meta.get("grpo_eligible", False):
            continue
        if meta.get("future") and not include_future:
            continue
        if avail is not None and name not in avail:
            continue
        out.append(name)
    return out


def get_training_reward_diagnostic_fields(
    include_future: bool = False,
    available_fields: Optional[List[str]] = None,
) -> List[str]:
    out = []
    avail = set(available_fields) if available_fields is not None else None
    for name, meta in REWARD_SCHEMA.items():
        if not bool(meta.get("training_only", False)):
            continue
        if meta.get("future") and not include_future:
            continue
        if avail is not None and name not in avail:
            continue
        out.append(name)
    return out


def get_supported_reward_fields() -> List[str]:
    """All reward fields theoretically supported by this codebase (static registry)."""
    return get_all_reward_fields()


def get_supported_rerank_eligible_reward_fields(include_future: bool = False) -> List[str]:
    return get_rerank_eligible_reward_fields(include_future=include_future)


def get_supported_verifier_eligible_reward_fields(include_future: bool = False) -> List[str]:
    return get_verifier_eligible_reward_fields(include_future=include_future)


def get_supported_grpo_eligible_reward_fields(include_future: bool = False) -> List[str]:
    return get_grpo_eligible_reward_fields(include_future=include_future)


def infer_active_reward_fields_from_records(
    records: Iterable[Dict[str, Any]],
    known_fields_only: bool = True,
) -> List[str]:
    """Infer run-active reward fields from output records.

    "Active" means: field exists AND at least one record contains a finite value.
    This is stronger than "field appears in JSON" (which may be NaN placeholders).

    Args:
        records: iterable of JSON dicts (e.g. candidate_scores.jsonl records)
        known_fields_only: if True, only consider fields registered in REWARD_SCHEMA.
    """
    active = set()
    registry = set(REWARD_SCHEMA.keys())
    for r in records:
        keys = r.keys() if not known_fields_only else [k for k in r.keys() if k in registry]
        for k in keys:
            if k in active:
                continue
            if _is_finite_number(r.get(k)):
                active.add(k)
    return sorted(active)


def strategy_to_reward_map(include_future: bool = False) -> Dict[str, str]:
    mapping = {}
    for name, meta in REWARD_SCHEMA.items():
        strat = meta.get("strategy")
        if not strat or not meta.get("rerank_eligible", False):
            continue
        if meta.get("future") and not include_future:
            continue
        mapping[strat] = name
    return mapping


def reward_to_strategy_map(include_future: bool = False) -> Dict[str, str]:
    return {v: k for k, v in strategy_to_reward_map(include_future=include_future).items()}


def get_schema_summary(
    include_future: bool = False,
    available_fields: Optional[List[str]] = None,
    active_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    present = set(available_fields) if available_fields is not None else None
    active = set(active_fields) if active_fields is not None else present

    rerank_fields = get_rerank_eligible_reward_fields(
        include_future=include_future,
        available_fields=list(active) if active is not None else None,
    )
    verifier_fields = get_verifier_eligible_reward_fields(
        include_future=include_future,
        available_fields=list(active) if active is not None else None,
    )
    grpo_fields = get_grpo_eligible_reward_fields(
        include_future=include_future,
        available_fields=list(active) if active is not None else None,
    )
    training_diag_fields = get_training_reward_diagnostic_fields(
        include_future=include_future,
        available_fields=list(active) if active is not None else None,
    )
    probes = [
        f for f in get_sample_probe_fields()
        if present is None or f in present
    ]

    excluded = {}
    for name, meta in REWARD_SCHEMA.items():
        if present is not None and name not in present:
            continue
        if name in rerank_fields:
            continue
        excluded[name] = {
            "scope": meta.get("scope"),
            "exclude_reason": meta.get("exclude_reason", "not_rerank_eligible"),
            "description": meta.get("description", ""),
        }

    supported_candidate = get_candidate_reward_fields(include_future=include_future)
    supported_probes = get_sample_probe_fields()
    supported_rerank = get_supported_rerank_eligible_reward_fields(include_future=include_future)
    supported_verifier = get_supported_verifier_eligible_reward_fields(include_future=include_future)
    supported_grpo = get_supported_grpo_eligible_reward_fields(include_future=include_future)

    eligibility = {
        name: {
            "uses_ground_truth": bool(meta.get("uses_ground_truth", False)),
            "label_conditioned": bool(meta.get("label_conditioned", False)),
            "verifier_eligible": bool(meta.get("verifier_eligible", False)),
            "grpo_eligible": bool(meta.get("grpo_eligible", False)),
            "verifier_only": bool(meta.get("verifier_only", False)),
            "training_only": bool(meta.get("training_only", False)),
            "scope": meta.get("scope"),
            "family": meta.get("family"),
            "source": meta.get("source"),
            "strategy": meta.get("strategy"),
            "response_conditioned": bool(meta.get("response_conditioned", False)),
            "candidate_dependent": bool(meta.get("candidate_dependent", meta.get("scope") == "candidate")),
            "default_training_reward": bool(meta.get("default_training_reward", False)),
            "diagnostic_only": bool(meta.get("diagnostic_only", False)),
            "unsafe_as_verifier": bool(meta.get("unsafe_as_verifier", False)),
            "unsafe_as_group_reward": bool(meta.get("unsafe_as_group_reward", False)),
        }
        for name, meta in REWARD_SCHEMA.items()
    }

    return {
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "main_experiment_reward_modes": get_main_experiment_reward_mode_schema(),
        "default_main_experiment_reward_mode": get_default_main_experiment_reward_mode(),

        # Static registry (supported-by-code) lists.
        "supported_reward_fields": get_supported_reward_fields(),
        "supported_candidate_reward_fields": supported_candidate,
        "supported_sample_probe_fields": supported_probes,
        "supported_rerank_eligible_reward_fields": supported_rerank,
        "supported_verifier_eligible_reward_fields": supported_verifier,
        "supported_grpo_eligible_reward_fields": supported_grpo,

        # Backward-compatible keys (older scripts may expect these names).
        "candidate_reward_fields": supported_candidate,
        "sample_probe_fields": probes,
        "rerank_eligible_reward_fields": rerank_fields,
        "verifier_eligible_reward_fields": verifier_fields,
        "grpo_eligible_reward_fields": grpo_fields,
        "excluded_from_rerank": excluded,
        "reward_eligibility": eligibility,

        # Run-active info (may be absent if caller didn't provide active_fields).
        "available_reward_fields": sorted(active) if active is not None else None,
        "available_rerank_eligible_reward_fields": rerank_fields,
        "available_verifier_eligible_reward_fields": verifier_fields,
        "available_grpo_eligible_reward_fields": grpo_fields,
        "available_training_reward_diagnostic_fields": training_diag_fields,
    }
