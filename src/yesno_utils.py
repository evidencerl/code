"""Strict yes/no parsing helpers shared across reward and analysis scripts."""

from __future__ import annotations

import re
from typing import Tuple

# existence/yes-no 任务在主实验里是低熵 probe family，不进入 evidence 主训练 reward。
LOW_ENTROPY_TASK_FAMILIES = {"existence"}

_LEADING_TRAILING_PUNCT = " \t\r\n\"'`([{<)]}>.,!?;:-_/\\"

_FIRST_TOKEN_RE = re.compile(r"^\s*([^\s]+)")
_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)

# Refusal / uncertainty phrases should stay in "other" rather than leaking into no/yes.
_UNCERTAIN_PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in [
        r"\bnot sure\b",
        r"\bmaybe\b",
        r"\bunknown\b",
        r"\bunclear\b",
        r"\bcannot determine\b",
        r"\bcan'?t determine\b",
        r"\bunable to determine\b",
        r"\bcannot tell\b",
        r"\bcan'?t tell\b",
        r"\bnot enough information\b",
        r"\binsufficient information\b",
        r"\bi do not know\b",
        r"\bdon't know\b",
        r"\bdo not know\b",
        r"\bunsure\b",
    ]
]


def _normalize_token(token: str) -> str:
    return token.strip(_LEADING_TRAILING_PUNCT).lower()


def _first_token(text: str) -> str:
    match = _FIRST_TOKEN_RE.search(text or "")
    if not match:
        return ""
    return _normalize_token(match.group(1))


def is_uncertain_yesno(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    return any(p.search(lowered) for p in _UNCERTAIN_PATTERNS)


def parse_yesno_with_source(text: str) -> Tuple[str, str]:
    """Return ('yes'|'no'|'other', parse_source)."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return "other", "other"

    first = _first_token(lowered)
    if first == "yes":
        return "yes", "first_token"
    if first == "no":
        return "no", "first_token"

    if is_uncertain_yesno(lowered):
        return "other", "uncertain"

    has_yes = bool(_YES_RE.search(lowered))
    has_no = bool(_NO_RE.search(lowered))
    if has_yes and not has_no:
        return "yes", "boundary_regex"
    if has_no and not has_yes:
        return "no", "boundary_regex"
    return "other", "other"


def parse_yesno(text: str) -> str:
    return parse_yesno_with_source(text)[0]


def is_low_entropy_yesno_family(task_family: str) -> bool:
    return (task_family or "").strip().lower() in LOW_ENTROPY_TASK_FAMILIES
