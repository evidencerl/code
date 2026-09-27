#!/usr/bin/env python3
"""Merge a LoRA/trainable training checkpoint with base model into a standalone Hugging Face model directory.

Usage:
    python scripts/merge_checkpoint.py \
        --base Qwen/Qwen3.5-9B \
        --checkpoint checkpoints/qwen35_9b_answer_ced/step_002000.pt \
        --dest checkpoints/qwen35_9b_answer_ced/merged_hf \
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def _src_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def merge(base: Path, checkpoint: Path, dest: Path, device: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("trainable_state_dict") or {}
    extra = dict(payload.get("extra") or {})
    if not state:
        raise SystemExit(f"no trainable_state_dict in {checkpoint}")
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "MERGE_COMPLETE.json"
    if marker.is_file() and (dest / "config.json").is_file():
        print(f"[merge] already complete at {dest}", flush=True)
        return

    print(f"[merge] Loading base model from {base}...")
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if device != "cpu" else None,
        trust_remote_code=True,
    )
    use_lora = bool(extra.get("lora_applied"))
    n_trainable_layers = int(extra.get("n_trainable_layers", 4))
    if use_lora:
        src = _src_dir()
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from mini_grpo_smoke import _setup_trainable

        print(f"[merge] Setting up LoRA model structure...")
        model, _ = _setup_trainable(
            model, n_trainable_layers=n_trainable_layers, use_lora=True
        )
        incompatible = model.load_state_dict(state, strict=False)
        print(f"[merge] Merging LoRA weights into base model weights...")
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        model.eval()
        n_frozen = len(incompatible.missing_keys)
    else:
        named = dict(model.named_parameters())
        missing = [key for key in state if key not in named]
        if missing:
            raise SystemExit(f"checkpoint keys not in base model: {missing[:8]}")
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise SystemExit(f"unexpected keys: {incompatible.unexpected_keys[:8]}")
        named = dict(model.named_parameters())
        for key, tensor in state.items():
            current = named[key].detach().cpu()
            if current.shape != tensor.shape or not torch.equal(current, tensor.cpu()):
                raise SystemExit(f"overlay mismatch: {key}")
        n_frozen = len(incompatible.missing_keys)

    print(f"[merge] Saving full model to {dest}...")
    processor.save_pretrained(dest)
    model.save_pretrained(dest, safe_serialization=True)
    record = {
        "status": "ok",
        "base": str(base),
        "checkpoint": str(checkpoint),
        "step": payload.get("step"),
        "n_trainable_tensors": len(state),
        "n_frozen_left_untouched": n_frozen,
        "lora_merged": use_lora,
        "extra": extra,
    }
    marker.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[merge] Successfully saved merged checkpoint to {dest}!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA/trainable checkpoint into full HF directory")
    parser.add_argument("--base", required=True, help="Path or HuggingFace ID of base model")
    parser.add_argument("--checkpoint", required=True, help="Path to step_XXXXXX.pt checkpoint")
    parser.add_argument("--dest", required=True, help="Destination directory for merged model")
    parser.add_argument("--device", default="cpu", help="Device for merge operations (default: cpu)")
    args = parser.parse_args()
    merge(Path(args.base), Path(args.checkpoint), Path(args.dest), args.device)


if __name__ == "__main__":
    main()
