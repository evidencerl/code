"""Optional NVIDIA training kernels for Evidence-RL.

The paper trainer stays HuggingFace + GRPO. This module is a thin shell:

- Transformer Engine: fused LN / FP8 later, if installed.
- FlashAttention-2: preferred attention (wired in model_loader).
- NeMo-RL: not used as the CED reward loop. CED needs teacher-forced
  logprobs and visual-token replace hooks; those stay in hhj-train.

Importing this file must never fail if NVIDIA packages are absent.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def transformer_engine_status() -> Dict[str, Any]:
    try:
        import transformer_engine.pytorch as te  # noqa: F401

        version = getattr(__import__("transformer_engine"), "__version__", "unknown")
        return {"available": True, "version": version}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def flash_attn_status() -> Dict[str, Any]:
    try:
        import flash_attn

        return {"available": True, "version": getattr(flash_attn, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def nemo_rl_status() -> Dict[str, Any]:
    try:
        import nemo_rl  # type: ignore

        return {"available": True, "version": getattr(nemo_rl, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def recommended_attn_implementation() -> str:
    override = str(os.environ.get("EVIDENCE_RL_ATTN", "")).strip()
    if override:
        return override
    fa = flash_attn_status()
    if fa.get("available"):
        return "flash_attention_2"
    return "sdpa"


def runtime_report() -> Dict[str, Any]:
    return {
        "attn": recommended_attn_implementation(),
        "flash_attn": flash_attn_status(),
        "transformer_engine": transformer_engine_status(),
        "nemo_rl": nemo_rl_status(),
        "ced_backend": "huggingface_grpo",
        "nemo_rl_used_for_ced": False,
    }


def maybe_te_dtype(default: Optional[Any] = None) -> Optional[Any]:
    """Placeholder: FP8/TE dtype is opt-in and off unless EVIDENCE_RL_TE=1."""
    if str(os.environ.get("EVIDENCE_RL_TE", "0")).strip().lower() in {"0", "false", "no", "off"}:
        return default
    status = transformer_engine_status()
    if not status.get("available"):
        return default
    return default
