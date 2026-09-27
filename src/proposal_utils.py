"""主实验 proposal / evidence region 构造工具。

目标：
1. 显式整理 proposal 来源，避免 bbox/fallback 逻辑散落在 reward/training 脚本里。
2. 为 relative evidence margin 提供统一的正向干预 / 随机负向干预采样接口。
3. 输出 reviewer 关心的 proposal 质量诊断字段。
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from visual_token_map import (
    compute_merge_ratio,
    bbox_to_token_indices,
    surrounding_indices,
    find_visual_range,
    to_absolute,
    reset_surrounding_meta,
)


def _normalize_bbox(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    out = []
    for v in bbox:
        try:
            out.append(float(v))
        except Exception:
            return None
    if out[2] <= 0 or out[3] <= 0:
        return None
    return out


def _clip_bbox_to_image(bbox: List[float], image_width: int, image_height: int) -> Optional[List[float]]:
    x, y, w, h = [float(v) for v in bbox]
    x = max(0.0, min(x, float(image_width)))
    y = max(0.0, min(y, float(image_height)))
    max_w = max(0.0, float(image_width) - x)
    max_h = max(0.0, float(image_height) - y)
    w = max(0.0, min(w, max_w))
    h = max(0.0, min(h, max_h))
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def _center_fallback_bbox(image_width: int, image_height: int) -> List[float]:
    """bbox 缺失时使用保守中心框，确保流程能跑且日志明确标记 fallback。"""
    w = max(1.0, float(image_width) * 0.35)
    h = max(1.0, float(image_height) * 0.35)
    x = max(0.0, (float(image_width) - w) * 0.5)
    y = max(0.0, (float(image_height) - h) * 0.5)
    return [x, y, w, h]


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-9)


def _seed_from_metadata(metadata: Optional[Dict[str, Any]]) -> int:
    if not isinstance(metadata, dict):
        return 0
    base = (
        metadata.get("sample_uid")
        or metadata.get("pair_id")
        or metadata.get("sample_id")
        or metadata.get("image_id")
        or "proposal_seed"
    )
    s = hashlib.md5(str(base).encode("utf-8")).hexdigest()[:8]
    return int(s, 16)


def resolve_primary_proposal_bbox(
    *,
    target_bbox: Any,
    image_width: int,
    image_height: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """解析主 proposal bbox，并把 fallback 原因显式写回诊断。"""
    metadata = metadata or {}
    bbox_candidates = [
        ("metadata.proposal_bbox", metadata.get("proposal_bbox")),
        ("metadata.target_bbox", metadata.get("target_bbox")),
        ("arg.target_bbox", target_bbox),
        ("metadata.bbox", metadata.get("bbox")),
    ]
    for source, raw_bbox in bbox_candidates:
        norm = _normalize_bbox(raw_bbox)
        if norm is None:
            continue
        clipped = _clip_bbox_to_image(norm, image_width=image_width, image_height=image_height)
        if clipped is not None:
            return {
                "bbox": clipped,
                "proposal_source": source,
                "proposal_fallback_used": False,
                "proposal_bbox_missing": False,
                "proposal_fallback_reason": "",
            }

    fallback = _center_fallback_bbox(image_width=image_width, image_height=image_height)
    return {
        "bbox": fallback,
        "proposal_source": "fallback.center_bbox",
        "proposal_fallback_used": True,
        "proposal_bbox_missing": True,
        "proposal_fallback_reason": "missing_or_invalid_bbox",
    }


def _sample_negative_bbox(
    *,
    primary_bbox: List[float],
    image_width: int,
    image_height: int,
    rng: random.Random,
    max_tries: int = 24,
) -> Optional[List[float]]:
    """采样与正向 proposal 同尺度、低重叠的随机框。"""
    x, y, w, h = [float(v) for v in primary_bbox]
    if w <= 0 or h <= 0 or w > image_width or h > image_height:
        return None
    max_x = max(0.0, float(image_width) - w)
    max_y = max(0.0, float(image_height) - h)
    for _ in range(max_tries):
        cand = [
            rng.uniform(0.0, max_x) if max_x > 0 else 0.0,
            rng.uniform(0.0, max_y) if max_y > 0 else 0.0,
            w,
            h,
        ]
        if _bbox_iou(primary_bbox, cand) <= 0.05:
            return cand
    return None


def _token_stats(
    *,
    bbox: List[float],
    image_width: int,
    image_height: int,
    n_tokens: int,
    n_vis_tokens: int,
) -> Dict[str, Any]:
    area = max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))
    total_area = max(1.0, float(image_width) * float(image_height))
    return {
        "proposal_area": area,
        "proposal_area_fraction": area / total_area,
        "proposal_token_count": int(n_tokens),
        "proposal_token_coverage_fraction": float(n_tokens) / float(max(1, n_vis_tokens)),
    }


def build_intervention_proposals(
    *,
    inputs: Dict[str, Any],
    img_tid: int,
    target_bbox: Any,
    image_width: int,
    image_height: int,
    metadata: Optional[Dict[str, Any]] = None,
    sur_ring: int = 2,
    negative_k: int = 1,
) -> Dict[str, Any]:
    """构造主实验所需的 proposal bundle。

    返回：
    - positive: 相关干预 proposal
    - negatives: 随机/无关干预 proposal 列表
    - diagnostics: proposal 质量统计
    """
    metadata = metadata or {}
    primary = resolve_primary_proposal_bbox(
        target_bbox=target_bbox,
        image_width=image_width,
        image_height=image_height,
        metadata=metadata,
    )
    bbox = primary["bbox"]

    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        raise RuntimeError("image_grid_thw not found while building proposals")
    gt_val, gh, gw = [int(v) for v in grid_thw[0].tolist()]
    vs, ve = find_visual_range(inputs["input_ids"], img_tid)
    n_vis = ve - vs
    merge_ratio = compute_merge_ratio(gt_val * gh * gw, n_vis)

    def _build_one(bbox_value: List[float], source: str) -> Optional[Dict[str, Any]]:
        rel = bbox_to_token_indices(bbox_value, image_width, image_height, gh, gw, merge_ratio)
        if not rel:
            return None
        reset_surrounding_meta()
        sur_rel = surrounding_indices(rel, n_vis, gh, gw, merge_ratio, ring=sur_ring)
        abs_idx = to_absolute(inputs["input_ids"], rel, img_tid)
        sur_abs = to_absolute(inputs["input_ids"], sur_rel, img_tid)
        if not abs_idx:
            return None
        out = {
            "bbox": [float(v) for v in bbox_value],
            "proposal_source": source,
            "relative_indices": list(rel),
            "absolute_indices": list(abs_idx),
            "surrounding_absolute_indices": list(sur_abs),
        }
        out.update(
            _token_stats(
                bbox=bbox_value,
                image_width=image_width,
                image_height=image_height,
                n_tokens=len(abs_idx),
                n_vis_tokens=n_vis,
            )
        )
        return out

    positive = _build_one(bbox, primary["proposal_source"])
    positive_fallback_reason = primary["proposal_fallback_reason"]
    if positive is None:
        fallback_bbox = _center_fallback_bbox(image_width=image_width, image_height=image_height)
        positive = _build_one(fallback_bbox, "fallback.center_bbox_tokenized")
        primary["proposal_fallback_used"] = True
        primary["proposal_bbox_missing"] = True
        positive_fallback_reason = positive_fallback_reason or "bbox_tokenization_empty"
    if positive is None:
        raise RuntimeError("failed_to_build_positive_proposal")

    rng = random.Random(_seed_from_metadata(metadata))
    negatives: List[Dict[str, Any]] = []
    for neg_idx in range(max(0, int(negative_k))):
        neg_bbox = _sample_negative_bbox(
            primary_bbox=positive["bbox"],
            image_width=image_width,
            image_height=image_height,
            rng=rng,
        )
        neg = None
        neg_source = f"random_bbox_neg_{neg_idx}"
        if neg_bbox is not None:
            neg = _build_one(neg_bbox, neg_source)
        if neg is None:
            # bbox 采样失败时退回到随机 token 子集，保证 relative margin 最低可跑。
            pos_rel = set(positive["relative_indices"])
            candidates = [i for i in range(n_vis) if i not in pos_rel]
            rng.shuffle(candidates)
            take = candidates[: len(positive["relative_indices"])]
            if take:
                abs_idx = to_absolute(inputs["input_ids"], take, img_tid)
                reset_surrounding_meta()
                sur_rel = surrounding_indices(take, n_vis, gh, gw, merge_ratio, ring=sur_ring)
                sur_abs = to_absolute(inputs["input_ids"], sur_rel, img_tid)
                neg = {
                    "bbox": None,
                    "proposal_source": f"random_token_subset_neg_{neg_idx}",
                    "relative_indices": list(take),
                    "absolute_indices": list(abs_idx),
                    "surrounding_absolute_indices": list(sur_abs),
                    "proposal_area": float("nan"),
                    "proposal_area_fraction": float("nan"),
                    "proposal_token_count": int(len(abs_idx)),
                    "proposal_token_coverage_fraction": float(len(abs_idx)) / float(max(1, n_vis)),
                }
        if neg is not None:
            negatives.append(neg)

    diagnostics = {
        "proposal_source": positive["proposal_source"],
        "proposal_bbox_missing": bool(primary["proposal_bbox_missing"]),
        "proposal_fallback_used": bool(primary["proposal_fallback_used"]),
        "proposal_fallback_reason": positive_fallback_reason,
        "proposal_area": positive.get("proposal_area"),
        "proposal_area_fraction": positive.get("proposal_area_fraction"),
        "proposal_token_count": positive.get("proposal_token_count"),
        "proposal_token_coverage_fraction": positive.get("proposal_token_coverage_fraction"),
        "proposal_negative_count": len(negatives),
        "proposal_negative_sources": [n.get("proposal_source") for n in negatives],
        "n_visual_tokens": int(n_vis),
        "merge_ratio": float(merge_ratio),
        "cache_reuse_enabled": True,
    }

    return {
        "positive": positive,
        "negatives": negatives,
        "diagnostics": diagnostics,
    }
