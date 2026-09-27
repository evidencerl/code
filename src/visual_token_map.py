"""bbox → visual token index 映射 & 相关工具。

Qwen3-VL 实际行为：
  - `image_grid_thw` 给出视觉 grid 的 (T, H, W)
  - 视觉编码器输出经过 **2×2 空间 merge**（spatial_merge_size=2）
  - merge 后的 token 排列：merged_h = grid_h // 2, merged_w = grid_w // 2
  - grid cell (gy, gx) 对应 merged token = (gy // 2) * merged_w + (gx // 2)
  - input_ids 里的视觉 token 数 ≈ T * merged_h * merged_w

修复记录（v2）：
  [FIX-3] bbox_to_token_indices: 原来用 grid_linear_index // merge_ratio 做线性
          整除，这对 2×2 空间 merge 是错的。改为正确的 2D 空间 merge。

修复记录（v3）：
  [FIX-7] surrounding_indices: 原来返回"所有非目标视觉 token"，实际是全局背景
          而非局部环带。改为真正的 ring-based 局部 surrounding 实现。
          支持 ring=1,2,...，当局部环带为空时 fallback 并记录。
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any


def compute_merge_ratio(grid_total: int, n_visual_tokens: int) -> int:
    """根据 grid 总数和视觉 token 数推 merge_ratio。"""
    assert n_visual_tokens > 0, "n_visual_tokens must be > 0"
    raw = grid_total / float(n_visual_tokens)
    mr = int(round(raw))
    if mr <= 0:
        mr = 1
    if grid_total % n_visual_tokens != 0:
        approx = n_visual_tokens * mr
        rel_err = abs(approx - grid_total) / float(max(1, grid_total))
        if rel_err > 0.05:
            raise ValueError(f"merge_ratio inconsistency: grid_total={grid_total}, "
                             f"n_vis={n_visual_tokens}, mr={mr}, rel_err={rel_err:.3f}")
    return mr


def _infer_spatial_merge_size(merge_ratio: int) -> Tuple[int, int]:
    """从 merge_ratio 推断空间 merge 的 (sh, sw)。"""
    sq = int(math.isqrt(merge_ratio))
    if sq * sq == merge_ratio:
        return sq, sq
    for sh in range(sq, 0, -1):
        if merge_ratio % sh == 0:
            return sh, merge_ratio // sh
    return 1, merge_ratio


def _bbox_to_grid_indices(
    bbox, img_w: int, img_h: int, grid_h: int, grid_w: int
) -> List[Tuple[int, int]]:
    """COCO bbox (x,y,w,h) → 覆盖到的 grid cell 坐标 (gy, gx) 列表。"""
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return []
    x2, y2 = x + w, y + h
    cell_w = img_w / float(grid_w)
    cell_h = img_h / float(grid_h)

    coords: List[Tuple[int, int]] = []
    for gy in range(grid_h):
        cy = (gy + 0.5) * cell_h
        if cy < y or cy > y2:
            continue
        for gx in range(grid_w):
            cx = (gx + 0.5) * cell_w
            if cx < x or cx > x2:
                continue
            coords.append((gy, gx))

    if not coords:
        cx = x + w / 2.0
        cy = y + h / 2.0
        gx = min(grid_w - 1, max(0, int(cx / cell_w)))
        gy = min(grid_h - 1, max(0, int(cy / cell_h)))
        coords.append((gy, gx))
    return coords


def bbox_to_token_indices(
    bbox, img_w: int, img_h: int, grid_h: int, grid_w: int, merge_ratio: int
) -> List[int]:
    """COCO bbox → 视觉 token 相对索引（0..n_vis-1）。

    [FIX-3] 正确的 2D 空间 merge 映射。
    """
    grid_coords = _bbox_to_grid_indices(bbox, img_w, img_h, grid_h, grid_w)
    if not grid_coords:
        return []

    sh, sw = _infer_spatial_merge_size(max(1, int(merge_ratio)))
    merged_w = grid_w // sw

    toks = sorted({(gy // sh) * merged_w + (gx // sw) for gy, gx in grid_coords})
    return toks


def _token_to_merged_coords(
    token_idx: int, merged_h: int, merged_w: int
) -> Tuple[int, int]:
    """merged token 线性索引 → (my, mx) 坐标。"""
    my = token_idx // merged_w
    mx = token_idx % merged_w
    return my, mx


def surrounding_indices(
    target_tokens: List[int],
    n_visual_tokens: int,
    grid_h: int,
    grid_w: int,
    merge_ratio: int,
    ring: int = 2,
) -> List[int]:
    """[FIX-7] 获取目标区域外围的局部环带 token 索引。

    实现：
      1. 将 target_tokens 转为 merged grid 上的 (my, mx) 坐标集合
      2. 对每个 ring 层级 r (1..ring)，收集距离目标区域 Chebyshev 距离 = r 的 token
      3. 返回所有环带层级的并集
      4. 若局部环带为空（目标覆盖整个 grid），fallback 到全局背景

    返回值增加一个 surrounding_meta 属性（通过返回 list 子类或外部调用方式获取），
    但为了兼容性，函数签名不变，改用 module-level 变量传递 fallback 信息。

    参数:
      target_tokens: 目标区域的 merged token 相对索引列表
      n_visual_tokens: 总视觉 token 数
      grid_h, grid_w: 原始 grid 尺寸（merge 前）
      merge_ratio: 空间 merge 比率
      ring: 环带宽度（Chebyshev 距离，默认 2）
    """
    if n_visual_tokens <= 0:
        _surrounding_meta.update({"method": "empty", "fallback": True, "ring_used": 0})
        return []

    sh, sw = _infer_spatial_merge_size(max(1, int(merge_ratio)))
    merged_h = grid_h // sh
    merged_w = grid_w // sw

    tgt_set = set()
    tgt_coords = set()
    for t in target_tokens:
        t = int(t)
        if 0 <= t < n_visual_tokens:
            tgt_set.add(t)
            tgt_coords.add(_token_to_merged_coords(t, merged_h, merged_w))

    if not tgt_coords:
        _surrounding_meta.update({"method": "global_fallback", "fallback": True, "ring_used": 0})
        return list(range(n_visual_tokens))

    # 收集环带
    ring_tokens = set()
    for r in range(1, ring + 1):
        for my, mx in tgt_coords:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dy) != r and abs(dx) != r:
                        continue  # 只取恰好在第 r 圈的
                    ny, nx = my + dy, mx + dx
                    if 0 <= ny < merged_h and 0 <= nx < merged_w:
                        idx = ny * merged_w + nx
                        if idx not in tgt_set and 0 <= idx < n_visual_tokens:
                            ring_tokens.add(idx)

    if ring_tokens:
        _surrounding_meta.update({
            "method": "ring",
            "fallback": False,
            "ring_used": ring,
            "n_ring_tokens": len(ring_tokens),
        })
        return sorted(ring_tokens)

    # 环带为空（目标太大，覆盖了整个 grid）→ fallback 到全局非目标
    global_bg = [i for i in range(n_visual_tokens) if i not in tgt_set]
    if global_bg:
        _surrounding_meta.update({
            "method": "global_fallback",
            "fallback": True,
            "ring_used": ring,
            "reason": "ring_empty_target_too_large",
        })
        return global_bg

    # 极端情况：目标覆盖全部 token → 用全部 token（含自身）
    _surrounding_meta.update({
        "method": "all_tokens_fallback",
        "fallback": True,
        "ring_used": ring,
        "reason": "target_covers_all",
    })
    return list(range(n_visual_tokens))


# Module-level meta dict，调用 surrounding_indices 后可读取
_surrounding_meta: Dict[str, Any] = {}


def get_surrounding_meta() -> Dict[str, Any]:
    """获取上次 surrounding_indices() 调用的元信息（fallback 等）。"""
    return dict(_surrounding_meta)


def reset_surrounding_meta():
    """重置 surrounding meta。"""
    _surrounding_meta.clear()


def find_visual_range(input_ids, image_token_id: int) -> Tuple[int, int]:
    """返回 input_ids 中视觉 token 的 [start, end)。"""
    ids = input_ids.squeeze().tolist()
    pos = [i for i, t in enumerate(ids) if t == image_token_id]
    assert pos, f"image_token_id={image_token_id} not found in input_ids"
    return pos[0], pos[-1] + 1


def to_absolute(input_ids, rel_indices, image_token_id) -> List[int]:
    """相对（visual 段）索引 → input_ids 绝对位置。"""
    vs, ve = find_visual_range(input_ids, image_token_id)
    n = ve - vs
    out: List[int] = []
    for i in rel_indices:
        i = int(i)
        if 0 <= i < n:
            out.append(vs + i)
    return out
