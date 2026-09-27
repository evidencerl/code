from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def _to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "yes", "y", "1"):
            return True
        if t in ("false", "no", "n", "0"):
            return False
    return None


def _normalize_bbox(v: Any) -> Optional[List[float]]:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    out = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            return None
    return out


def _resolve_image_file(image_path: str, image_root: str) -> str:
    if not image_path:
        return ""

    p = os.path.normpath(str(image_path))
    if os.path.isabs(p):
        return p

    root = os.path.normpath(image_root) if image_root else ""
    if root:
        full = os.path.normpath(os.path.join(root, p))
        if os.path.exists(full):
            return p

    if os.path.exists(p):
        return p

    return p


def _resolve_image_abspath(image_file: str, image_root: str) -> Optional[str]:
    if not image_file:
        return None
    if os.path.isabs(image_file):
        return image_file if os.path.exists(image_file) else None

    cands = []
    if image_root:
        cands.append(os.path.join(image_root, image_file))
    cands.append(image_file)

    for c in cands:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c
    return None


def _iter_image_path_candidates(image_ref: str, image_root: str) -> List[str]:
    if not image_ref:
        return []

    ref = os.path.normpath(str(image_ref))
    root = os.path.normpath(str(image_root)) if image_root else ""

    cands: List[str] = []
    if os.path.isabs(ref):
        cands.append(ref)
    else:
        if root:
            cands.append(os.path.normpath(os.path.join(root, ref)))

            # Avoid duplicated prefix when root already points to val2017 and ref is val2017/xxx.jpg.
            root_base = os.path.basename(root.rstrip(os.sep))
            if root_base and ref.startswith(root_base + os.sep):
                stripped = ref[len(root_base) + 1:]
                cands.append(os.path.normpath(os.path.join(root, stripped)))

        cands.append(ref)

    # De-duplicate while preserving order.
    seen = set()
    uniq = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def resolve_image_abspath(sample: Dict[str, Any], image_root: str) -> Optional[str]:
    refs = [
        sample.get("image_file"),
        sample.get("source_image_path"),
        sample.get("image_path"),
    ]
    for r in refs:
        if not r:
            continue
        for cand in _iter_image_path_candidates(str(r), image_root=image_root):
            if os.path.exists(cand):
                return cand
    return None


def open_sample_image(sample: Dict[str, Any], image_root: str) -> Image.Image:
    resolved = resolve_image_abspath(sample, image_root=image_root)
    if not resolved:
        ctx = {
            "pair_id": sample.get("pair_id"),
            "image_id": sample.get("image_id"),
            "image_file": sample.get("image_file"),
            "source_image_path": sample.get("source_image_path"),
            "image_root": image_root,
            "resolved_abspath": None,
        }
        raise FileNotFoundError(f"image_not_found: {json.dumps(ctx, ensure_ascii=False)}")

    try:
        return Image.open(resolved).convert("RGB")
    except Exception as e:
        ctx = {
            "pair_id": sample.get("pair_id"),
            "image_id": sample.get("image_id"),
            "image_file": sample.get("image_file"),
            "source_image_path": sample.get("source_image_path"),
            "image_root": image_root,
            "resolved_abspath": resolved,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }
        raise RuntimeError(f"image_open_failed: {json.dumps(ctx, ensure_ascii=False)}") from e


def check_dataset_integrity(samples: List[Dict[str, Any]], image_root: str, max_check: int = 0) -> Dict[str, Any]:
    total = len(samples)
    checked = samples if max_check <= 0 else samples[: max_check]

    ok = 0
    bad_examples: List[Dict[str, Any]] = []
    for s in checked:
        resolved = resolve_image_abspath(s, image_root=image_root)
        if resolved:
            ok += 1
        else:
            if len(bad_examples) < 20:
                bad_examples.append({
                    "pair_id": s.get("pair_id"),
                    "image_id": s.get("image_id"),
                    "image_file": s.get("image_file"),
                    "source_image_path": s.get("source_image_path"),
                    "image_root": image_root,
                    "resolved_abspath": resolved,
                })

    pair_counter = Counter([s.get("pair_id") for s in samples])
    pair_size_dist = Counter(pair_counter.values())
    pair_size_dist = {str(k): int(v) for k, v in sorted(pair_size_dist.items(), key=lambda x: int(x[0]))}

    task_counter = Counter([s.get("task_type", "unknown") for s in samples])
    split_counter = Counter([s.get("data_split", "unknown") for s in samples])

    return {
        "total_samples": int(total),
        "checked_samples": int(len(checked)),
        "image_path_resolve_ok_count": int(ok),
        "image_path_resolve_missing_count": int(len(checked) - ok),
        "image_path_resolve_ok_rate": float(ok) / float(max(1, len(checked))),
        "bad_path_examples": bad_examples,
        "pair_id_unique": int(len(pair_counter)),
        "pair_id_size_distribution": pair_size_dist,
        "task_type_distribution": dict(task_counter),
        "split_distribution": dict(split_counter),
    }


def _split_from_image_id(image_id: Any, train_ratio: float, val_ratio: float) -> str:
    key = str(image_id)
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    frac = int(h[:8], 16) / float(0xFFFFFFFF)
    if frac < train_ratio:
        return "train"
    if frac < train_ratio + val_ratio:
        return "val"
    return "probe"


def adapt_sample_to_main_schema(
    raw: Dict[str, Any],
    idx: int,
    image_root: str = "",
    fill_image_size: bool = True,
    image_size_cache: Optional[Dict[str, Tuple[Optional[int], Optional[int]]]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Optional[Dict[str, Any]]:
    pair_id = str(raw.get("pair_id") or f"vg_pair_{idx}")
    sample_uid = str(raw.get("sample_uid") or f"{pair_id}:{idx}")
    image_id = raw.get("image_id", -1)
    image_path = str(raw.get("image_path") or raw.get("image_file") or "")
    image_file = _resolve_image_file(image_path, image_root=image_root)

    bbox = _normalize_bbox(raw.get("target_bbox", raw.get("bbox")))
    if bbox is None:
        return None

    question = str(raw.get("question") or "").strip()
    answer = str(raw.get("answer") or "").strip()
    if not question or not answer:
        return None

    gt_present = _to_bool(raw.get("gt_present"))

    image_width = raw.get("image_width")
    image_height = raw.get("image_height")

    if fill_image_size and (not image_width or not image_height):
        cache = image_size_cache if image_size_cache is not None else {}
        if image_file in cache:
            iw, ih = cache[image_file]
        else:
            iw, ih = None, None
            abs_path = _resolve_image_abspath(image_file, image_root=image_root)
            if abs_path:
                try:
                    with Image.open(abs_path) as img:
                        iw, ih = img.size
                except Exception:
                    iw, ih = None, None
            cache[image_file] = (iw, ih)
        image_width = image_width or iw
        image_height = image_height or ih

    task_type = str(raw.get("task_type") or "unknown").strip().lower()
    split = _split_from_image_id(image_id, train_ratio=train_ratio, val_ratio=val_ratio)

    out = {
        "pair_id": pair_id,
        "sample_uid": sample_uid,
        "image_id": image_id,
        "image_file": image_file,
        "image_width": int(image_width) if image_width else None,
        "image_height": int(image_height) if image_height else None,
        "task_type": task_type,
        "question": question,
        "answer": answer,
        "target_bbox": bbox,
        "gt_present": gt_present,
        "dataset_name": "vg_brutal",
        "data_split": split,
        "source_image_path": image_path,
    }
    return out


def load_vg_brutal_as_main_schema(
    dataset_file: str,
    image_root: str = "",
    max_samples: int = 0,
    seed: int = 42,
    task_type: str = "",
    split: str = "all",
    fill_image_size: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if train_ratio < 0 or val_ratio < 0 or (train_ratio + val_ratio) > 1.0:
        raise ValueError(
            f"Invalid split ratios: train_ratio={train_ratio}, val_ratio={val_ratio}. "
            "Require train_ratio>=0, val_ratio>=0, and train_ratio+val_ratio<=1.0"
        )

    raw_rows: List[Dict[str, Any]] = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_rows.append(json.loads(line))

    size_cache: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    adapted: List[Dict[str, Any]] = []
    dropped = 0
    for i, row in enumerate(raw_rows):
        x = adapt_sample_to_main_schema(
            row,
            idx=i,
            image_root=image_root,
            fill_image_size=fill_image_size,
            image_size_cache=size_cache,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        if x is None:
            dropped += 1
            continue
        adapted.append(x)

    tt = (task_type or "").strip().lower()
    if tt and tt not in ("all", "any", "*"):
        adapted = [s for s in adapted if (s.get("task_type") or "") == tt]

    sp = (split or "all").strip().lower()
    if sp in ("train", "val", "probe"):
        adapted = [s for s in adapted if s.get("data_split") == sp]

    if max_samples and len(adapted) > max_samples:
        rng = random.Random(seed)
        adapted = rng.sample(adapted, max_samples)

    pair_counter = Counter([s.get("pair_id") for s in adapted])
    pair_size_dist = Counter(pair_counter.values())
    pair_size_dist = {str(k): int(v) for k, v in sorted(pair_size_dist.items(), key=lambda x: int(x[0]))}

    task_counter = Counter([s.get("task_type", "unknown") for s in adapted])
    split_counter = Counter([s.get("data_split", "unknown") for s in adapted])
    gt_type_counter = Counter([type(s.get("gt_present")).__name__ for s in adapted])

    image_resolve_ok = 0
    for s in adapted:
        if _resolve_image_abspath(s.get("image_file", ""), image_root=image_root):
            image_resolve_ok += 1

    summary: Dict[str, Any] = {
        "dataset_name": "vg_brutal",
        "dataset_file": dataset_file,
        "image_root": image_root,
        "requested_split": sp,
        "requested_task_type_filter": tt or "all",
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "probe_ratio": float(1.0 - train_ratio - val_ratio),
        "raw_count": len(raw_rows),
        "valid_count": len(adapted),
        "dropped_count": int(dropped),
        "pair_id_unique": len(pair_counter),
        "pair_id_size_distribution": pair_size_dist,
        "pair_id_exactly_two_ratio": (
            float(pair_size_dist.get("2", 0)) / float(max(1, len(pair_counter)))
        ),
        "task_type_distribution": dict(task_counter),
        "split_distribution": dict(split_counter),
        "gt_present_type_distribution": dict(gt_type_counter),
        "image_path_resolve_ok_count": int(image_resolve_ok),
        "image_path_resolve_ok_rate": float(image_resolve_ok) / float(max(1, len(adapted))),
        "has_image_size_count": int(sum(1 for s in adapted if s.get("image_width") and s.get("image_height"))),
    }

    return adapted, summary
