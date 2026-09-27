#!/usr/bin/env python3
"""Download the 1312 Visual Genome JPEGs used by vg_brutal_pairs.jsonl."""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = "Mozilla/5.0 Evidence-RL-repro"
MIRRORS = [
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{name}",
    "https://cs.stanford.edu/people/rak248/VG_100K/{name}",
]


def fetch_one(name: str, out_dir: Path, retries: int = 4) -> tuple[str, str]:
    dest = out_dir / name
    if dest.is_file() and dest.stat().st_size > 1000:
        return name, "exists"
    tmp = dest.with_suffix(".jpg.part")
    last = ""
    for url_tmpl in MIRRORS:
        url = url_tmpl.format(name=name)
        for _ in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                if len(data) < 1000:
                    raise RuntimeError(f"too_small:{len(data)}")
                tmp.write_bytes(data)
                tmp.replace(dest)
                return name, f"ok:{url}"
            except Exception as exc:
                last = repr(exc)
                time.sleep(0.4)
    return name, f"fail:{last}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    names = [ln.strip() for ln in Path(args.names_file).read_text().splitlines() if ln.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(fetch_one, name, out_dir) for name in names]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            if status.startswith("fail"):
                fail += 1
                print(f"[fail] {name} {status}", flush=True)
            else:
                ok += 1
            if i % 50 == 0 or i == len(names):
                print(f"[progress] {i}/{len(names)} ok={ok} fail={fail}", flush=True)
    print(f"[done] ok={ok} fail={fail} out={out_dir}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
