"""Load the paper backbones for Evidence-RL.

Cross-backbone recipe (arXiv:2608.08021): Qwen2.5-VL-3B, Qwen2.5-VL-7B,
and Qwen3-VL-8B-Instruct. Class is selected from config.json.
"""

from __future__ import annotations

import json
import os

import torch
from transformers import AutoConfig, AutoProcessor


def _auto_multi_gpu_enabled() -> bool:
    raw = str(os.environ.get("AUTO_MULTI_GPU", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _visible_gpu_count() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _resolve_device_map_strategy(requested_device: str):
    override = str(os.environ.get("MODEL_DEVICE_MAP", "")).strip().lower()
    if override == "local_rank":
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        return {"": lr}, f"ddp_local_rank={lr}"
    if override in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return override, f"env_override:{override}"

    n_gpu = _visible_gpu_count()
    if n_gpu <= 1:
        return {"": 0}, f"single_visible_gpu_full_replica n={n_gpu}"
    if requested_device.startswith("cuda") and _auto_multi_gpu_enabled() and n_gpu > 1:
        return "balanced", f"auto_multi_gpu_visible_count={n_gpu}"
    return "auto", f"default_single_process_visible_count={n_gpu}"


def _read_model_type(model_dir: str) -> tuple[str, list]:
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return str(raw.get("model_type", "")).lower(), list(raw.get("architectures") or [])


def _resolve_model_class(model_dir: str):
    model_type, architectures = _read_model_type(model_dir)
    arch = " ".join(str(x) for x in architectures).lower()
    blob = f"{model_type} {arch}"

    if "qwen2_5" in blob or "qwen2.5" in blob:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration, "qwen2_5_vl"
    if "qwen3_5" in blob or "qwen3.5" in blob:
        try:
            from transformers import Qwen3_5ForConditionalGeneration

            return Qwen3_5ForConditionalGeneration, "qwen3_5"
        except Exception:
            pass
    if "qwen3" in blob:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration, "qwen3_vl"

    raise ValueError(
        f"Unsupported Evidence-RL backbone in {model_dir}: "
        f"model_type={model_type} architectures={architectures}"
    )


def load(model_dir: str, device="cuda:0", dtype="bfloat16"):
    assert os.path.isfile(f"{model_dir}/config.json"), f"模型不存在: {model_dir}"

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    device_map, strategy_reason = _resolve_device_map_strategy(str(device or "cuda:0"))
    visible_gpu_count = _visible_gpu_count()
    model_cls, family = _resolve_model_class(model_dir)
    print(
        f"[model_loader] family={family} cls={model_cls.__name__} "
        f"requested_device={device} visible_gpu_count={visible_gpu_count} "
        f"device_map={device_map} reason={strategy_reason}"
    )
    load_kw = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    attn_impl = str(os.environ.get("EVIDENCE_RL_ATTN", "flash_attention_2")).strip() or "flash_attention_2"
    try:
        model = model_cls.from_pretrained(
            model_dir,
            attn_implementation=attn_impl,
            **load_kw,
        )
        print(f"[model_loader] attn_implementation={attn_impl}", flush=True)
    except Exception as exc:
        print(
            f"[model_loader] attn={attn_impl} failed ({type(exc).__name__}: {exc}); "
            "fallback sdpa",
            flush=True,
        )
        model = model_cls.from_pretrained(
            model_dir,
            attn_implementation="sdpa",
            **load_kw,
        )
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        print("[model_loader] gradient_checkpointing enabled", flush=True)
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return processor, model, cfg


def generation_model(model):
    """Inference entry. DDP does not expose generate or config."""
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def find_decoder_layers(model):
    """Locate LLM decoder layers across Qwen2.5-VL / Qwen3-VL wrappers."""
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.model.language_model.layers,
        lambda m: m.language_model.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    return None


def num_layers(cfg, model=None) -> int:
    for sub in [cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)]:
        if sub and hasattr(sub, "num_hidden_layers"):
            return int(sub.num_hidden_layers)
    if model is not None:
        layers = find_decoder_layers(model)
        if layers is not None:
            return len(layers)
    raise AttributeError("Cannot determine num_hidden_layers")
