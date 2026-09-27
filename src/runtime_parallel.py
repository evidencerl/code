"""运行时多卡/分片辅助工具。

设计目标：
1. 统一解析 CUDA_VISIBLE_DEVICES，避免每个入口各自手写一遍。
2. 只做轻量单机多卡辅助，不引入新的训练框架。
3. 让 rerank / eval / paper bundle 能在保持现有目录结构的前提下自动吃满可见卡。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


def auto_multi_gpu_enabled() -> bool:
    """统一自动多卡开关。

    默认开启；如需快速回退，可设 AUTO_MULTI_GPU=0。
    """
    raw = str(os.environ.get("AUTO_MULTI_GPU", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def parse_visible_cuda_devices() -> List[str]:
    """解析当前进程可见的 GPU 列表。

    优先尊重 CUDA_VISIBLE_DEVICES；若未设置，则回退到 torch.cuda.device_count()。
    """
    raw = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if raw:
        items = [x.strip() for x in raw.split(",") if x.strip()]
        if all(item.isdigit() for item in items):
            return items
    if not torch.cuda.is_available():
        return []
    try:
        return [str(i) for i in range(int(torch.cuda.device_count()))]
    except Exception:
        return []


def split_visible_devices(n_groups: int) -> List[List[str]]:
    """把可见 GPU 尽量均匀切成 n_groups 份。

    说明：
    - groups 数不会超过可见 GPU 数；
    - 返回空列表表示当前无需/无法做多卡分组；
    - 使用 round-robin 分配，避免最后一组拿不到卡。
    """
    devices = parse_visible_cuda_devices()
    if (not auto_multi_gpu_enabled()) or len(devices) <= 1:
        return []
    n_groups = max(1, min(int(n_groups), len(devices)))
    groups: List[List[str]] = [[] for _ in range(n_groups)]
    for idx, dev in enumerate(devices):
        groups[idx % n_groups].append(dev)
    return [g for g in groups if g]


def child_env_for_gpu_group(gpu_group: List[str]) -> Dict[str, str]:
    """为子进程构造隔离后的 CUDA_VISIBLE_DEVICES。"""
    env = dict(os.environ)
    if gpu_group:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_group)
    return env


def runtime_parallel_summary() -> Dict[str, Any]:
    """把当前进程的多卡运行状态写成统一摘要。"""
    visible = parse_visible_cuda_devices()
    return {
        "auto_multi_gpu": bool(auto_multi_gpu_enabled()),
        "cuda_visible_devices": ",".join(visible),
        "visible_gpu_count": int(len(visible)),
        "multi_gpu_active": bool(auto_multi_gpu_enabled() and len(visible) > 1),
    }


def read_jsonl_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl_rows(path: str | Path, rows: Iterable[Dict[str, Any]]) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return str(path)


def merge_jsonl_files(paths: Iterable[str | Path], output_path: str | Path) -> str:
    """按给定顺序拼接 JSONL 分片。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for path in paths:
            if not path or not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line.rstrip("\n") + "\n")
    return str(output_path)


def shard_slice_bounds(total: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    """与现有 CLI 分片逻辑保持一致的范围切分。"""
    num_shards = max(1, int(num_shards))
    shard_id = max(0, int(shard_id))
    shard_size = int(math.ceil(float(total) / float(num_shards)))
    start = shard_id * shard_size
    end = min(start + shard_size, total)
    return start, end
