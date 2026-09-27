from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from answer_format_utils import normalize_answer_for_exact_match

_NEGATION_RE = re.compile(
    r"\b(?:no|not|none|never|without|cannot|can't|cant|unable|neither|n't)\b",
    flags=re.IGNORECASE,
)

_ABSTENTION_PATTERNS = [
    re.compile(p, flags=re.IGNORECASE)
    for p in [
        r"\b(?:cannot|can't|cant)\s+(?:determine|tell|infer|know)\b",
        r"\bnot\s+(?:visible|clear|enough information|sure)\b",
        r"\b(?:unclear|unknown|unsure|hard to tell)\b",
        r"\b(?:cannot|can't|cant)\s+be\s+seen\b",
        r"\bnot\s+shown\b",
        r"\b(?:n/?a|not applicable)\b",
    ]
]

_TEMPLATE_SLOT_PATTERNS = [
    re.compile(p, flags=re.IGNORECASE)
    for p in [
        r"<[^>]+>",
        r"\[[^\]]+\]",
        r"\b(?:short\s*answer|very\s*short|your\s*answer|fill\s+in|placeholder)\b",
        r"\b(?:integer\s+or\s+short\s+phrase|left\s*/\s*right\s*/\s*top\s*/\s*bottom)\b",
        r"\b(?:evidence\s*:\s*\.\.\.|final\s*answer\s*:\s*\.\.\.)\b",
    ]
]

_SCHEMA_LABEL_RE = re.compile(r"(?im)^\s*(evidence|final\s*answer)\s*[:：]\s*(.*)$")


_PLACEHOLDER_VALUES = {
    "...",
    "…",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "unsure",
    "unclear",
    "not applicable",
    "answer",
    "text",
    "short answer",
    "very short",
    "yes/no",
    "yes or no",
    "integer or short phrase",
    "left/right/top/bottom",
    "short attribute phrase",
}


def _is_placeholder_value(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    if t in _PLACEHOLDER_VALUES:
        return True
    if re.search(r"<[^>]+>|\[[^\]]+\]", t):
        return True
    return False


def _extract_schema_field_values(raw_response: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for m in _SCHEMA_LABEL_RE.finditer(raw_response or ""):
        key = re.sub(r"\s+", " ", (m.group(1) or "").strip().lower())
        fields[key] = (m.group(2) or "").strip()
    return fields


def _schema_evidence_status(raw_response: str) -> Dict[str, bool]:
    fields = _extract_schema_field_values(raw_response or "")
    has_evidence_field = "evidence" in fields
    evidence_text = (fields.get("evidence") or "").strip()
    evidence_missing = has_evidence_field and (not evidence_text)
    evidence_placeholder = has_evidence_field and _is_placeholder_value(evidence_text)
    return {
        "has_evidence_field": bool(has_evidence_field),
        "evidence_missing": bool(evidence_missing),
        "evidence_placeholder": bool(evidence_placeholder),
        "evidence_text": evidence_text,
    }


def _looks_like_template_echo(answer_text: str, raw_response: str) -> bool:
    ans = (answer_text or "").strip()
    raw = (raw_response or "").strip()
    probe = raw or ans
    if not probe:
        return False

    if ans and any(p.search(ans) for p in _TEMPLATE_SLOT_PATTERNS):
        return True
    if raw and re.search(r"<[^>]+>|\[[^\]]+\]", raw):
        return True

    fields = _extract_schema_field_values(raw)
    final_val = fields.get("final answer", fields.get("answer", ""))
    evidence_val = fields.get("evidence", "")

    if final_val and _is_placeholder_value(final_val):
        return True
    if len(fields) >= 2 and not ans and evidence_val and _is_placeholder_value(evidence_val):
        return True
    if ans and _is_placeholder_value(ans):
        return True
    return False


_RELATION_TASK_HINTS = {
    "spatial", "relation", "cooccur_bait", "multihop_cross", "action_state", "occlusion"
}
_ATTRIBUTE_TASK_HINTS = {"attribute", "attr_conjunction", "finegrained"}
_COUNT_TASK_HINTS = {"counting", "count_exclusion"}
_YESNO_TASK_HINTS = {"existence", "binary", "yesno"}


def _word_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])


def target_answer_length_range(task_type: str) -> Tuple[int, int]:
    tt = (task_type or "").strip().lower()
    if tt in _YESNO_TASK_HINTS:
        return 1, 2
    if tt in _COUNT_TASK_HINTS:
        return 1, 2
    if tt in _ATTRIBUTE_TASK_HINTS:
        return 1, 4
    if tt in _RELATION_TASK_HINTS:
        return 2, 6
    return 1, 6


def _compute_hard_template_veto(
    *,
    answer_text: str,
    raw_response: str,
    is_template_like: bool,
    is_abstention_like: bool,
    is_parse_fail: bool,
) -> Dict[str, Any]:
    ans = (answer_text or "").strip()
    raw = (raw_response or "").strip()
    fields = _extract_schema_field_values(raw)
    final_val = fields.get("final answer", fields.get("answer", ""))
    evidence_val = fields.get("evidence", "")
    final_placeholder = bool(final_val) and _is_placeholder_value(final_val)
    evidence_placeholder = bool(evidence_val) and _is_placeholder_value(evidence_val)
    answer_placeholder = _is_placeholder_value(ans)
    evidence_placeholder_only = evidence_placeholder and (not final_placeholder) and bool(ans) and (not answer_placeholder)

    reasons = []
    if is_abstention_like and (answer_placeholder or final_placeholder or not ans):
        reasons.append("abstention_like")
    if is_template_like and answer_placeholder:
        reasons.append("placeholder_answer")
    if final_placeholder or ((not ans) and evidence_placeholder):
        reasons.append("placeholder_schema")
    if is_parse_fail and (answer_placeholder or final_placeholder or ((not ans) and evidence_placeholder)):
        reasons.append("parse_fail_placeholder")

    if evidence_placeholder_only:
        reasons = []

    if not reasons:
        return {
            "hard_template_veto": False,
            "hard_template_veto_reason": "",
            "hard_template_veto_reasons": [],
        }

    return {
        "hard_template_veto": True,
        "hard_template_veto_reason": reasons[0],
        "hard_template_veto_reasons": reasons,
    }


def compute_style_bias_features(
    answer_text: str,
    *,
    raw_response: str = "",
    parsed_answer: str = "",
    task_type: str = "",
    is_parse_fail: bool = False,
) -> Dict[str, Any]:
    ans = (answer_text or "").strip()
    raw = (raw_response or "").strip()
    norm = normalize_answer_for_exact_match(ans or raw)
    length_tokens = _word_count(ans)
    lo, hi = target_answer_length_range(task_type)
    length_excess = max(0, length_tokens - hi) + max(0, lo - length_tokens)
    has_negation = bool(_NEGATION_RE.search(ans or raw))
    is_abstention_like = any(p.search(ans or raw) for p in _ABSTENTION_PATTERNS)
    is_template_like = _looks_like_template_echo(ans, raw)
    is_numeric_short = bool(re.fullmatch(r"[-+]?\d+", norm)) and length_tokens <= 2
    is_yesno_short = norm in {"yes", "no"} and length_tokens <= 2
    evidence_status = _schema_evidence_status(raw)
    hard_veto = _compute_hard_template_veto(
        answer_text=ans,
        raw_response=raw,
        is_template_like=is_template_like,
        is_abstention_like=is_abstention_like,
        is_parse_fail=is_parse_fail,
    )

    return {
        "task_type": (task_type or "").strip().lower(),
        "normalized_answer": norm,
        "length_tokens": int(length_tokens),
        "target_length_lo": int(lo),
        "target_length_hi": int(hi),
        "length_excess": float(length_excess),
        "has_negation": bool(has_negation),
        "is_abstention_like": bool(is_abstention_like),
        "is_template_like": bool(is_template_like),
        "is_numeric_short": bool(is_numeric_short),
        "is_yesno_short": bool(is_yesno_short),
        "has_evidence_field": bool(evidence_status["has_evidence_field"]),
        "evidence_missing": bool(evidence_status["evidence_missing"]),
        "evidence_placeholder": bool(evidence_status["evidence_placeholder"]),
        "parsed_answer": (parsed_answer or "").strip().lower(),
        "is_parse_fail": bool(is_parse_fail),
        "hard_template_veto": bool(hard_veto["hard_template_veto"]),
        "hard_template_veto_reason": hard_veto["hard_template_veto_reason"],
        "hard_template_veto_reasons": hard_veto["hard_template_veto_reasons"],
    }


def compute_style_penalty(
    features: Dict[str, Any],
    *,
    length_coef: float = 0.03,
    abstention_coef: float = 0.10,
    template_coef: float = 0.15,
    parse_fail_coef: float = 0.10,
    negation_coef: float = 0.0,
    evidence_missing_coef: float = 0.06,
    evidence_placeholder_coef: float = 0.08,
    max_penalty: float = 0.35,
) -> Dict[str, Any]:
    neg_ok = bool(features.get("has_negation", False)) and not bool(features.get("is_yesno_short", False))
    components = {
        "length": float(length_coef) * float(features.get("length_excess", 0.0)),
        "abstention": float(abstention_coef) if bool(features.get("is_abstention_like", False)) else 0.0,
        "template": float(template_coef) if bool(features.get("is_template_like", False)) else 0.0,
        "parse_fail": float(parse_fail_coef) if bool(features.get("is_parse_fail", False)) else 0.0,
        "negation": float(negation_coef) if neg_ok else 0.0,
        "evidence_missing": float(evidence_missing_coef) if bool(features.get("evidence_missing", False)) else 0.0,
        "evidence_placeholder": float(evidence_placeholder_coef) if bool(features.get("evidence_placeholder", False)) else 0.0,
    }
    raw_penalty = sum(components.values())
    penalty = min(float(max_penalty), max(0.0, float(raw_penalty)))
    return {
        "penalty": penalty,
        "raw_penalty": raw_penalty,
        "components": components,
    }
