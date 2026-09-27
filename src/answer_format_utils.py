"""Shared prompt + answer formatting utilities.

This module is intentionally lightweight: it only contains pure helper functions
that are shared by `candidate_gen.py` and `mini_grpo_smoke.py`.

Goals:
- Keep prompt building consistent across scripts.
- Extract a stable `final_answer` from structured generation outputs.
- Keep dedup-time normalization and analysis-time normalization explicitly separated.

Non-goals:
- Do NOT move training/generation pipelines here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple

from yesno_utils import parse_yesno


PROMPT_MODES = ("raw_question", "task_aware_v1", "short_evidence_v1", "final_answer_only_v1")

# 主实验统一只认这 5 个 family。不要再向下游暴露 generic/misc 之类的松散标签。
TASK_FAMILIES = ("counting", "attribute", "spatial", "existence", "other")
EVIDENCE_TRAINING_TASK_FAMILIES = ("counting", "attribute", "spatial")
PROBE_ONLY_TASK_FAMILIES = ("existence",)


_TASK_FAMILY_ALIASES = {
    # canonical families
    "existence": "existence",
    "yesno": "existence",
    "counting": "counting",
    "count": "counting",
    "spatial": "spatial",
    "location": "spatial",
    "attribute": "attribute",
    "description": "attribute",
    # vg_brutal task names
    "count_exclusion": "counting",
    "spatial_chain": "spatial",
    "occlusion": "spatial",
    "multihop_cross": "spatial",
    "attr_conjunction": "attribute",
    "finegrained": "attribute",
    "action_state": "attribute",
    # 这类 failure-mode / hallucination 任务不进入主训练，统一落到 other/existence probe 分流。
    "cooccur_bait": "other",
}


_YESNO_PREFIX_RE = re.compile(
    r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|would|should|will|won't|can't)\b",
    flags=re.I,
)
_COUNT_PREFIX_RE = re.compile(r"^\s*(how\s+many|what\s+number\s+of)\b", flags=re.I)
_SPATIAL_PREFIX_RE = re.compile(
    r"^\s*(where\b|what\s+is\s+located\b|what\s+is\s+between\b|what\s+is\s+behind\b|"
    r"what\s+is\s+in\s+front\s+of\b|what\s+is\s+occluding\b|which\s+object\s+is\b)",
    flags=re.I,
)
_ATTRIBUTE_PREFIX_RE = re.compile(
    r"^\s*(describe\b|what\s+color\b|what\s+kind\b|what\s+material\b|what\s+state\b|"
    r"what\s+condition\b|what\s+does\s+the\b)",
    flags=re.I,
)


def normalize_task_family_label(value: str = "") -> str:
    """把外部传入的 family/task_type 标签到主实验统一 family 集合。"""
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    mapped = _TASK_FAMILY_ALIASES.get(raw)
    if mapped:
        return mapped
    if raw in TASK_FAMILIES:
        return raw
    if "count" in raw:
        return "counting"
    if any(k in raw for k in ("attr", "fine", "state", "action", "desc", "color", "material")):
        return "attribute"
    if any(k in raw for k in ("spatial", "loc", "occlusion", "between", "relation", "behind", "front")):
        return "spatial"
    if any(k in raw for k in ("exist", "presence", "yesno", "hallucination", "probe")):
        return "existence"
    return "other"


def _iter_metadata_family_hints(metadata: Optional[Dict[str, Any]]) -> Iterable[str]:
    if not isinstance(metadata, dict):
        return []
    hints = []
    for key in (
        "task_family",
        "family",
        "benchmark_task_family",
        "benchmark_family",
        "task_type",
        "benchmark_name",
    ):
        v = metadata.get(key)
        if isinstance(v, str) and v.strip():
            hints.append(v)
    return hints


def infer_task_family(
    task_type: str = "",
    question: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """主实验统一 task family 路由入口。

    设计约束：
    1. 优先使用数据侧显式标签（task_family/task_type/metadata），避免临时猜测。
    2. 只有在显式标签缺失时，才回退到 question surface heuristic。
    3. 输出必须落在固定集合：counting/attribute/spatial/existence/other。
    """
    tt = (task_type or "").strip().lower()
    q = (question or "").strip()

    for hint in _iter_metadata_family_hints(metadata):
        fam = normalize_task_family_label(hint)
        if fam:
            return fam

    fam = normalize_task_family_label(tt)
    if fam:
        return fam

    if q:
        if _COUNT_PREFIX_RE.match(q):
            return "counting"
        if _YESNO_PREFIX_RE.match(q):
            return "existence"
        if _SPATIAL_PREFIX_RE.match(q):
            return "spatial"
        if _ATTRIBUTE_PREFIX_RE.match(q):
            return "attribute"

    return "other"


def task_family_allows_evidence_training(family: str) -> bool:
    return normalize_task_family_label(family) in EVIDENCE_TRAINING_TASK_FAMILIES


def task_family_is_probe_only(family: str) -> bool:
    return normalize_task_family_label(family) in PROBE_ONLY_TASK_FAMILIES


def task_family_is_main_training(family: str) -> bool:
    return task_family_allows_evidence_training(family)


def canonical_final_answer_prefix(prompt_mode: str = "raw_question") -> str:
    mode = (prompt_mode or "raw_question").strip().lower()
    if mode in {"task_aware_v1", "short_evidence_v1", "final_answer_only_v1"}:
        return "Final answer: "
    return ""


def _format_bbox_text(sample: Dict[str, Any]) -> str:
    bbox = sample.get("target_bbox") or sample.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return ""
    try:
        x, y, w, h = [float(v) for v in bbox]
    except Exception:
        return ""

    def _fmt(v: float) -> str:
        iv = int(round(v))
        return str(iv) if abs(v - iv) < 1e-6 else f"{v:.1f}"

    return f"[x={_fmt(x)}, y={_fmt(y)}, w={_fmt(w)}, h={_fmt(h)}]"


def render_question_text(sample: Dict[str, Any]) -> str:
    """Render the question text fed into generation.

    Some VG-brutal rows still contain a literal ``[bbox]`` placeholder.
    Leaving that token untouched creates an artificial ambiguity: the model is
    told to reason about a region, but the region spec is omitted. We replace it
    with the actual target bbox when available.
    """
    q = (sample.get("question") or "").strip()
    if not q:
        return q
    bbox_text = _format_bbox_text(sample)
    if bbox_text and "[bbox]" in q:
        q = q.replace("[bbox]", bbox_text)
    return q


def build_generation_prompt(sample: Dict[str, Any], prompt_mode: str = "raw_question") -> str:
    """Build inference-time prompt.

    `task_aware_v1` / `short_evidence_v1` use an explicit short-evidence channel
    before the final answer. This widens the controllable action space without
    turning generation into free-form CoT.

    `final_answer_only_v1` keeps a strict answer-only protocol for ablations.
    """
    q = render_question_text(sample)
    if prompt_mode == "raw_question" or not q:
        return q

    family = infer_task_family(sample.get("task_type") or "", q)

    if family == "existence":
        task_instr = "For yes/no questions, the final answer must be exactly yes or no."
    elif family == "counting":
        task_instr = "For counting questions, the final answer must be a single integer or number phrase."
    elif family == "spatial":
        task_instr = "For spatial questions, the final answer must be a short noun phrase or location phrase."
    elif family == "attribute":
        task_instr = "For attribute or description questions, the final answer must be a concise attribute phrase."
    else:
        task_instr = "Keep the final answer short, specific, and directly grounded in the image."

    mode = (prompt_mode or "raw_question").strip().lower()
    if mode == "final_answer_only_v1":
        return (
            "Answer the question using the image. Do not output chain-of-thought. "
            "Use exactly this one-line format:\n"
            "Final answer: <answer only>\n"
            f"{task_instr}\n\n"
            f"Question: {q}"
        )

    return (
        "Answer the question using the image. Do not output chain-of-thought. "
        "First write one very short evidence phrase grounded in the image (1-8 words, no full sentence), "
        "then write the final answer. Do not add any extra lines. Use exactly this two-line format:\n"
        "Evidence: <1-8 grounded words>\n"
        "Final answer: <answer only>\n"
        f"{task_instr}\n\n"
        f"Question: {q}"
    )

_FINAL_TAG_LINE_RE = re.compile(r"(?im)^\s*final\s*answer\s*[:：]\s*(?P<ans>[^\n\r]*)")
_ANSWER_TAG_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:：]\s*(?P<ans>[^\n\r]*)")
_EVIDENCE_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:：]\s*(?P<ev>.*)$")
_TAG_ONLY_LINE_RE = re.compile(r"(?im)^\s*(final\s*answer|answer)\s*[:：]\s*$")


def _clean_answer_line_keep_span(raw_line: str, abs_start: int) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Clean a single-line answer candidate while preserving its char span.

    Cleaning is intentionally conservative:
    - Trim surrounding whitespace
    - Strip a single leading bullet marker ('-' or '*')
    - Strip a single pair of surrounding backticks

    Returns (cleaned_text, (start,end)) where span is in the original full text.
    If span cannot be determined, returns (cleaned_text, None).
    """
    if raw_line is None:
        return "", None

    line = raw_line
    if not line:
        return "", None

    stripped = line.strip()
    if not stripped:
        return "", None

    idx0 = line.find(stripped)
    if idx0 < 0:
        idx0 = 0
    cur_start = abs_start + idx0
    cur_text = stripped

    m = re.match(r"^([-*])\s+", cur_text)
    if m:
        cur_start += m.end()
        cur_text = cur_text[m.end():]

    if len(cur_text) >= 2 and cur_text[0] == "`" and cur_text[-1] == "`":
        cur_start += 1
        cur_text = cur_text[1:-1]

    cleaned = cur_text.strip()
    if not cleaned:
        return "", None

    rel = stripped.find(cleaned)
    if rel < 0:
        return cleaned, (cur_start, cur_start + len(cleaned))
    s = abs_start + idx0 + rel
    e = s + len(cleaned)
    return cleaned, (s, e)


def _is_short_answer_candidate(s: str, max_chars: int = 80, max_words: int = 12) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    if "\n" in t or "\r" in t:
        return False
    if len(t) > max_chars:
        return False
    if re.search(r"\b(evidence|final\s*answer|question)\b\s*[:：]", t, flags=re.I):
        return False
    words = [w for w in t.split() if w]
    return len(words) <= max_words


def _line_char_ranges(raw: str):
    pos = 0
    for line in raw.splitlines(keepends=True):
        pure = line.rstrip("\r\n")
        yield pure, pos, pos + len(pure)
        pos += len(line)
    if raw and not raw.endswith(("\n", "\r")):
        return


def _extract_tag_then_next_short_line(raw: str) -> Optional[Dict[str, Any]]:
    lines = list(_line_char_ranges(raw))
    for i, (line, start, _end) in enumerate(lines):
        if not _TAG_ONLY_LINE_RE.match(line or ""):
            continue
        for j in range(i + 1, len(lines)):
            nxt, ns, _ne = lines[j]
            cleaned, span = _clean_answer_line_keep_span(nxt, abs_start=ns)
            if cleaned and _is_short_answer_candidate(cleaned):
                src = "final_answer_next_line" if "final" in line.lower() else "answer_next_line"
                return {"final_answer": cleaned, "source": src, "char_span": span}
            if (nxt or "").strip():
                break
    return None


def _extract_last_short_line(raw: str) -> Optional[Dict[str, Any]]:
    lines = list(_line_char_ranges(raw))
    for line, start, _end in reversed(lines):
        stripped = (line or "").strip()
        if not stripped:
            continue
        if re.match(r"^(evidence|question)\s*[:：]", stripped, flags=re.I):
            continue
        cleaned, span = _clean_answer_line_keep_span(line, abs_start=start)
        if cleaned and _is_short_answer_candidate(cleaned, max_chars=60, max_words=10):
            return {"final_answer": cleaned, "source": "last_short_line", "char_span": span}
        break
    return None


def _extract_single_line_short(raw: str) -> Optional[Dict[str, Any]]:
    if "\n" in raw or "\r" in raw:
        return None
    cleaned, span = _clean_answer_line_keep_span(raw, abs_start=0)
    if cleaned and _is_short_answer_candidate(cleaned):
        return {"final_answer": cleaned, "source": "single_line_short", "char_span": span}
    return None


def extract_final_answer(raw_text: str, task_type: str = "") -> Dict[str, Any]:
    """Extract final answer from model output.

    Returns a dict (kept compatible with `candidate_gen.py`):
      - final_answer: extracted string (may fall back to raw)
      - source: structured_tag | fallback_raw | single_line_short | ...
      - char_span: (start,end) within raw_text for extracted content, or None

    NOTE: `task_type` is currently unused by the extraction logic itself; it is
    preserved to keep call sites stable and to leave room for future heuristics.
    """
    raw = (raw_text or "").strip()
    if not raw:
        return {"final_answer": "", "source": "fallback_raw", "char_span": None}

    m = _FINAL_TAG_LINE_RE.search(raw)
    if m:
        ans_raw = m.group("ans")
        cleaned, span = _clean_answer_line_keep_span(ans_raw, abs_start=m.start("ans"))
        if cleaned and _is_short_answer_candidate(cleaned):
            return {"final_answer": cleaned, "source": "structured_tag", "char_span": span}

    m = _ANSWER_TAG_LINE_RE.search(raw)
    if m:
        ans_raw = m.group("ans")
        cleaned, span = _clean_answer_line_keep_span(ans_raw, abs_start=m.start("ans"))
        if cleaned and _is_short_answer_candidate(cleaned):
            return {"final_answer": cleaned, "source": "answer_tag", "char_span": span}

    tagged_next = _extract_tag_then_next_short_line(raw)
    if tagged_next is not None:
        return tagged_next

    lines = raw.splitlines()
    if len(lines) >= 2 and _EVIDENCE_LINE_RE.match(lines[0] or ""):
        second = lines[1]
        nl_pos = raw.find("\n")
        if nl_pos >= 0:
            abs_start = nl_pos + 1
            cleaned, span = _clean_answer_line_keep_span(second, abs_start=abs_start)
            if cleaned and _is_short_answer_candidate(cleaned, max_chars=60, max_words=10):
                return {"final_answer": cleaned, "source": "two_line_evidence", "char_span": span}

    single_line = _extract_single_line_short(raw)
    if single_line is not None:
        return single_line

    last_short = _extract_last_short_line(raw)
    if last_short is not None:
        return last_short

    return {"final_answer": raw, "source": "fallback_raw", "char_span": None}


def extract_final_answer_with_prefix(raw_response: str, task_type: str = "") -> Dict[str, Any]:
    """Extract final answer and also return prefix text before that answer.

    Returns:
      - final_answer
      - extraction_source
      - char_span: (start,end) within raw_response for the extracted *clean* answer
      - prefix_text_before_answer: raw_response[:start] if span is available else ""

    This helper is intended for span-conditioned answer-only logprob.
    """
    raw = (raw_response or "")
    ext = extract_final_answer(raw, task_type=task_type)
    fa = (ext.get("final_answer") or "").strip()
    src = ext.get("source") or "fallback_raw"
    span = ext.get("char_span")
    prefix = ""
    if isinstance(span, (tuple, list)) and len(span) == 2 and span[0] is not None:
        try:
            s0 = int(span[0])
            if 0 <= s0 <= len(raw):
                prefix = raw[:s0]
        except Exception:
            prefix = ""
    return {
        "final_answer": fa,
        "extraction_source": src,
        "char_span": span,
        "prefix_text_before_answer": prefix,
    }


def normalize_raw_text_for_dedup(text: str) -> str:
    """Light normalization for raw-response dedup keys.

    IMPORTANT:
    - This must not collapse semantically different responses into yes/no or numbers.
    - Intended only for `dedup_on=raw_text` in candidate generation.
    """
    return _normalize_text(text)


def normalize_final_answer_for_dedup(text: str, task_type: str = "") -> str:
    """Task-aware normalization for final-answer dedup keys.

    Intended for `dedup_on=final_answer` where semantic collapse is expected.
    """
    s = _normalize_text(text)
    family = infer_task_family(task_type=task_type)
    if not s:
        return ""

    if family == "existence":
        yn = parse_yesno(s)
        return yn if yn in ("yes", "no") else s

    if family == "counting":
        m = re.search(r"(-?\d+)", s)
        if m:
            return m.group(1)

    return s


def normalize_text_for_analysis(text: str, task_type: str = "") -> str:
    """Analysis-time normalization for collapse/diversity diagnostics."""
    return normalize_final_answer_for_dedup(text, task_type=task_type)


def normalize_candidate_text(text: str, task_type: str = "") -> str:
    """Backward-compatible alias for analysis-time normalization.

    Keep old API for existing callsites; new code should use:
      - normalize_raw_text_for_dedup
      - normalize_final_answer_for_dedup
      - normalize_text_for_analysis
    """
    return normalize_text_for_analysis(text, task_type=task_type)


_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def normalize_answer_for_exact_match(text: str) -> str:
    """Normalize answer text for robust exact-match scoring.

    Rules:
    - lower-case + strip
    - remove punctuation
    - collapse whitespace
    - simple number normalization (word->digit, leading-zero integer cleanup)
    """
    s = (text or "").strip().lower()
    if not s:
        return ""

    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    toks = [t for t in s.split() if t]
    out = []
    for t in toks:
        if t in _NUMBER_WORDS:
            out.append(_NUMBER_WORDS[t])
            continue
        if re.fullmatch(r"[-+]?\d+", t):
            try:
                out.append(str(int(t)))
                continue
            except Exception:
                pass
        out.append(t)
    return " ".join(out)


def normalized_exact_match(pred: str, gold: str) -> float:
    """Return 1.0 if normalized strings are exactly equal, else 0.0."""
    p = normalize_answer_for_exact_match(pred)
    g = normalize_answer_for_exact_match(gold)
    if not p or not g:
        return 0.0
    return 1.0 if p == g else 0.0


def parse_count_answer(text: str) -> Optional[int]:
    """Parse a counting answer into an integer when possible."""
    norm = normalize_answer_for_exact_match(text)
    if not norm:
        return None
    if re.fullmatch(r"[-+]?\d+", norm):
        try:
            return int(norm)
        except Exception:
            return None
    m = re.search(r"([-+]?\d+)", norm)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def answer_matches_expected_family_format(answer_text: str, family: str) -> bool:
    fam = (family or "").strip().lower()
    if fam == "existence":
        return parse_yesno(answer_text) in {"yes", "no"}
    if fam == "counting":
        return parse_count_answer(answer_text) is not None
    return False


def family_safe_slice_decision(
    *,
    question: str = "",
    task_type: str = "",
    gt_answer: str = "",
    allowed_families: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Conservative family slice decision used by smoke-stage RL.

    We first infer the family from question/task_type, then require the GT answer
    to match that family's expected answer format. By default only counting and
    existence are kept.
    """
    family = infer_task_family(task_type=task_type, question=question)
    allowed = tuple((allowed_families or ("counting", "existence")))
    if family not in allowed:
        return {"keep": False, "family": family, "reason": "family_not_allowed"}
    if family == "counting":
        ok = answer_matches_expected_family_format(gt_answer, family)
        return {"keep": ok, "family": family, "reason": "ok" if ok else "counting_gt_not_integer"}
    if family == "existence":
        ok = answer_matches_expected_family_format(gt_answer, family)
        return {"keep": ok, "family": family, "reason": "ok" if ok else "existence_gt_not_yesno"}
    return {"keep": False, "family": family, "reason": "unsupported_family"}


def task_aware_answer_score(pred: str, gold: str, *, task_type: str = "", question: str = "") -> float:
    """Family-aware exact-match score for smoke/sanity diagnostics and anchors."""
    family = infer_task_family(task_type=task_type, question=question)
    if family == "existence":
        p = parse_yesno(pred)
        g = parse_yesno(gold)
        if p not in {"yes", "no"} or g not in {"yes", "no"}:
            return 0.0
        return 1.0 if p == g else 0.0
    if family == "counting":
        p = parse_count_answer(pred)
        g = parse_count_answer(gold)
        if p is None or g is None:
            return 0.0
        return 1.0 if p == g else 0.0
    return normalized_exact_match(pred, gold)


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())
