"""Task-aware candidate evaluator helpers.

These helpers reduce false negatives caused by strict string equality,
especially for attribute/category answers.
"""

from __future__ import annotations

import re
from typing import Optional

from answer_format_utils import infer_task_family

_ARTICLES = {"a", "an", "the"}

_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_SYNONYM_MAP = {
    "bike": "bicycle",
    "bikes": "bicycle",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "tv": "television",
    "television": "television",
    "sofa": "couch",
    "couch": "couch",
    "cellphone": "phone",
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "phone": "phone",
    "aeroplane": "airplane",
    "airplane": "airplane",
    "plane": "airplane",
}


def _normalize_text(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_token(token: str) -> str:
    tok = _normalize_text(token)
    if tok in _ARTICLES:
        return ""
    if tok.endswith("s") and len(tok) > 3:
        tok = tok[:-1]
    return _SYNONYM_MAP.get(tok, tok)


def canonicalize_label(text: str) -> str:
    t = _normalize_text(text)
    if not t:
        return ""
    toks = [_normalize_token(x) for x in t.split()]
    toks = [x for x in toks if x and x not in _ARTICLES]
    return " ".join(toks)


def parse_count(text: str) -> Optional[int]:
    t = canonicalize_label(text)
    if not t:
        return None

    # 1) direct digits
    m = re.search(r"\d+", t)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None

    # 2) first token as word number
    first = t.split()[0]
    if first in _NUM_WORDS:
        return _NUM_WORDS[first]

    return None


def semantically_match_attribute(pred: str, gold: str) -> bool:
    p = canonicalize_label(pred)
    g = canonicalize_label(gold)
    if not p or not g:
        return False
    if p == g:
        return True

    # containment fallback: "small dog" vs "dog"
    p_set = set(p.split())
    g_set = set(g.split())
    if g_set.issubset(p_set) or p_set.issubset(g_set):
        return True

    return False


def classify_open_ended_correctness(task_type: str, pred_text: str, gold_text: str) -> bool:
    tt = infer_task_family(task_type=task_type)

    if tt == "counting":
        p_cnt = parse_count(pred_text)
        g_cnt = parse_count(gold_text)
        return p_cnt is not None and g_cnt is not None and p_cnt == g_cnt

    if tt == "attribute":
        return semantically_match_attribute(pred_text, gold_text)

    # spatial / other open-ended
    return canonicalize_label(pred_text) == canonicalize_label(gold_text)
