"""Mini-GRPO Smoke Test（v5）：采样参数可配 + 分层诊断 + collapse_layer。

=== v4→v5 变更 ===

【修复】生成采样参数从硬编码 temp=0.7/top_p=0.9 改为 CLI 可配：
  --temperature (默认 1.0), --top_p (默认 0.95), --greedy
  短回答（yes/no）用低温度会导致 95%+ 组内文本完全相同（坍缩问题的根源）。

【新增】collapse_layer 诊断：
  step_info 中新增 collapse_layer 字段，标识坍缩最早从哪一层出现
  (text → parse → raw_delta → score_base → pre_kl → final)。
  final summary 输出 collapse_layer_distribution。

【新增】--debug_dump_mode collapse_only：
  只 dump 有 collapse 的 group，便于快速排查。

=== v3→v4 变更 ===

【新增1】reward_mode 参数透传
  CLI 新增 --reward_mode, --tau_resp, --tau_ans, --alpha_resp, --alpha_ans,
  --min_reward, --max_reward，并透传给 ActionLogProbATEReward。

【新增2】分层诊断统计
  在 final summary 中输出：
  A. 文本层: exact_match_group_rate, mean_unique_responses_per_group
  B. parse 层: parse_identical_group_rate, correctness_identical_group_rate
  C. reward 分层: raw_delta_identical_group_rate, score_base_identical_group_rate,
     pre_kl_reward_identical_group_rate, final_reward_identical_group_rate
  D. 方差: mean_reward_std_within_group, n_groups_with_all_same_text 等

【新增3】debug dump
  --debug_dump_groups N 控制前 N 个 group 的详细 jsonl 导出。

=== 保留 v3 的 RL 数学修复 ===
  - 负 advantage 参与 policy gradient
  - KL 在 reward 侧而非 loss 侧
  - 零方差 fallback
"""

import argparse
import json
import math
import os
import sys
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent
_ROOT_DIR = _SRC_DIR.parent
for _p in (str(_ROOT_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_local_module(module_name: str, file_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"cannot create spec for {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from model_loader import load, num_layers
except ModuleNotFoundError as e:
    if e.name != 'model_loader':
        raise
    _ml = _load_local_module('model_loader', _SRC_DIR / 'model_loader.py')
    load = _ml.load
    num_layers = _ml.num_layers
from reward.action_logprob_ate import ActionLogProbATEReward
from reward.logodds_ate import LogOddsATEReward
from ced_core import prepare_inputs, get_image_token_id, extend_multimodal_inputs, expand_mm_inputs
from answer_format_utils import (
    build_generation_prompt,
    extract_final_answer,
    extract_final_answer_with_prefix,
    infer_task_family,
    normalize_final_answer_for_dedup,
    normalize_raw_text_for_dedup,
    normalize_text_for_analysis,
    task_aware_answer_score,
)
from style_bias_utils import compute_style_bias_features, compute_style_penalty
from dataset_adapters import load_vg_brutal_as_main_schema, open_sample_image, check_dataset_integrity
from reward_sanity_check import run_training_preflight_check
from yesno_utils import parse_yesno


def _flush_fsync(f) -> None:
    try:
        f.flush()
    except (FileNotFoundError, OSError):
        return
    try:
        os.fsync(f.fileno())
    except (FileNotFoundError, OSError):
        pass


def _safe_close_file(f) -> None:
    if f is None:
        return
    try:
        f.close()
    except (FileNotFoundError, OSError):
        pass


def _decode_with_token_char_spans(tok, token_ids: List[int]) -> (str, List[tuple]):
    """Decode token_ids and return (text, per_token_char_spans).

    This is used to align a character span (in decoded text) to token indices,
    enabling span-conditioned answer-only logprob.
    """
    if not token_ids:
        return "", []

    spans: List[tuple] = []
    prev = ""
    for i in range(len(token_ids)):
        cur = tok.decode(
            token_ids[: i + 1],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        if not cur.startswith(prev):
            return tok.decode(
                token_ids,
                skip_special_tokens=True, clean_up_tokenization_spaces=False,
            ), []
        spans.append((len(prev), len(cur)))
        prev = cur
    return prev, spans


def _token_indices_overlapping_char_span(spans: List[tuple], start: int, end: int) -> List[int]:
    out = []
    for i, (s, e) in enumerate(spans):
        if e <= start:
            continue
        if s >= end:
            continue
        out.append(i)
    return out


def _repetition_rate(text: str) -> float:
    words = text.strip().split()
    if len(words) < 2:
        return 0.0
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return 1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def _task_aware_parse_label(answer_text: str, task_type: str, question: str = "") -> str:
    """Monitoring-only parser label.

    For yes/no tasks we keep the old yes/no/other semantics.
    For open-form tasks, use a normalized short answer string instead of forcing
    everything into the useless "other" bucket.
    """
    family = infer_task_family(task_type=task_type, question=question)
    text = (answer_text or "").strip()
    if not text:
        return "__empty__"
    if family == "existence":
        return parse_yesno(text)
    norm = normalize_text_for_analysis(text, task_type=task_type)
    return norm if norm else "__empty__"


def _is_yesno_monitor_task(task_type: str, question: str = "") -> bool:
    return infer_task_family(task_type=task_type, question=question) == "existence"


def _setup_trainable(model, n_trainable_layers: int = 4, use_lora: bool = False):
    # Appendix A.1 default: last four LLM layers. LoRA (r=16, α=32) is only
    # for the Qwen3.5-9B isolation table.
    if use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config)
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model.parameters())
            print(f"[GRPO] LoRA: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M trainable")
            return model, True
        except ImportError:
            print("[GRPO] peft 未安装，fallback 到冻结策略")
        except Exception as e:
            print(f"[GRPO] LoRA 失败: {e}")

    for param in model.parameters():
        param.requires_grad = False
    from model_loader import find_decoder_layers
    layers = find_decoder_layers(model)
    if layers is not None and len(layers) > n_trainable_layers:
        for layer in layers[-n_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    for name, param in model.named_parameters():
        if "lm_head" in name:
            param.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    print(f"[GRPO] last-{n_trainable_layers}-layers: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M trainable")
    return model, False


def load_train_dataset(
    dataset_name: str,
    dataset_file: str,
    image_root: str,
    task_type: str,
    data_split: str,
    train_ratio: float,
    val_ratio: float,
    max_samples: int,
    seed: int,
    task_type_allowlist: Optional[List[str]] = None,
):
    name = (dataset_name or "legacy_vqa").strip().lower()
    task_filter = (task_type or "").strip().lower()
    if task_filter in ("all", "any", "*"):
        task_filter = ""
    allowlist = _parse_csv_lower_list(",".join(task_type_allowlist or []))

    if name == "vg_brutal":
        samples, summary = load_vg_brutal_as_main_schema(
            dataset_file=dataset_file,
            image_root=image_root,
            max_samples=0,
            seed=seed,
            task_type=task_filter,
            split=data_split,
            fill_image_size=True,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    else:
        with open(dataset_file, "r") as f:
            all_samples = [json.loads(l) for l in f if l.strip()]
        samples = [
            s for s in all_samples
            if (not task_filter) or ((s.get("task_type") or "").strip().lower() == task_filter)
        ]
        summary = {
            "dataset_name": "legacy_vqa",
            "dataset_file": dataset_file,
            "image_root": image_root,
            "task_type_filter": task_filter,
            "valid_count": len(samples),
        }

    if allowlist:
        samples = [
            s for s in samples
            if (
                ((s.get("task_type") or "").strip().lower() in allowlist)
                or (infer_task_family(task_type=s.get("task_type", ""), question=s.get("question", "")) in allowlist)
            )
        ]
        summary = dict(summary)
        summary["task_type_allowlist"] = allowlist
        summary["task_type_allowlist_match_mode"] = "task_type_or_family"
        summary["valid_count_after_allowlist"] = len(samples)

    if max_samples and len(samples) > max_samples:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_samples)

    task_filter_label = task_filter if task_filter else (",".join(allowlist) if allowlist else "")
    return samples, summary, task_filter_label


def resolve_dataset_paths(args: argparse.Namespace) -> Dict[str, str]:
    dataset_name = (args.dataset_name or "legacy_vqa").strip().lower()
    if dataset_name not in {"legacy_vqa", "vg_brutal"}:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")

    if dataset_name == "vg_brutal":
        if not (args.dataset_file or "").strip():
            raise ValueError("dataset_name=vg_brutal requires --dataset_file")
        if not (args.image_root or "").strip():
            raise ValueError("dataset_name=vg_brutal requires --image_root")
        return {
            "dataset_name": dataset_name,
            "dataset_file": args.dataset_file,
            "image_root": args.image_root,
        }

    if not (args.vqa_file or "").strip():
        raise ValueError("dataset_name=legacy_vqa requires --vqa_file")
    if not (args.coco_image_dir or "").strip():
        raise ValueError("dataset_name=legacy_vqa requires --coco_image_dir")
    return {
        "dataset_name": dataset_name,
        "dataset_file": args.vqa_file,
        "image_root": args.coco_image_dir,
    }


def resolve_task_type_filter(dataset_name: str, task_type_arg: str) -> str:
    t = (task_type_arg or "").strip().lower()
    if t in ("", "auto"):
        return "" if dataset_name == "vg_brutal" else "existence"
    if t in ("all", "any", "*"):
        return ""
    return t


def _parse_csv_lower_list(raw: str) -> List[str]:
    items: List[str] = []
    for part in (raw or "").split(","):
        t = (part or "").strip().lower()
        if not t or t in items:
            continue
        items.append(t)
    return items


def summarize_loaded_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    task_dist: Dict[str, int] = {}
    split_dist: Dict[str, int] = {}
    for s in samples:
        tt = (s.get("task_type") or "unknown")
        sp = (s.get("data_split") or "unknown")
        task_dist[tt] = task_dist.get(tt, 0) + 1
        split_dist[sp] = split_dist.get(sp, 0) + 1
    return {
        "loaded_sample_count": int(len(samples)),
        "task_type_distribution": task_dist,
        "split_distribution": split_dist,
    }


def build_family_buckets(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        fam = infer_task_family(task_type=s.get("task_type", ""), question=s.get("question", ""))
        buckets[fam].append(s)
    return {k: list(v) for k, v in sorted(buckets.items()) if v}


def sample_from_family_buckets(
    samples: List[Dict[str, Any]],
    family_buckets: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    rng: Optional[random.Random] = None,
    balanced: bool = True,
) -> Dict[str, Any]:
    rr = rng if rng is not None else random
    if not samples:
        raise ValueError("cannot sample from empty sample list")
    buckets = family_buckets or {}
    if (not balanced) or len(buckets) <= 1:
        return rr.choice(samples)
    fam = rr.choice(list(buckets.keys()))
    return rr.choice(buckets[fam])


class BalancedFamilyCoverageSampler:
    """Family-balanced sampler with optional no-replacement epochs.

    One balanced epoch draws `min_family_size` samples from every active family.
    Larger families keep advancing through their shuffled queues across epochs,
    so repeated epochs increase unique coverage before reshuffling exhausted queues.
    """

    def __init__(
        self,
        family_buckets: Dict[str, List[Dict[str, Any]]],
        *,
        rng: Optional[random.Random] = None,
        sampling_mode: str = "balanced_no_replacement",
    ):
        self.rng = rng if rng is not None else random.Random()
        self.sampling_mode = str(sampling_mode or "balanced_no_replacement").strip().lower()
        self.family_buckets = {
            fam: list(rows)
            for fam, rows in sorted((family_buckets or {}).items())
            if rows
        }
        if not self.family_buckets:
            raise ValueError("BalancedFamilyCoverageSampler requires non-empty family buckets")
        self.families = list(self.family_buckets.keys())
        self.family_sizes = {fam: len(rows) for fam, rows in self.family_buckets.items()}
        self.min_family_size = min(self.family_sizes.values())
        self.balanced_epoch_size = int(len(self.families) * self.min_family_size)
        self._family_cursor = 0
        self._epoch_index = 0
        self._epoch_remaining = {fam: self.min_family_size for fam in self.families}
        self._queues = {fam: list(rows) for fam, rows in self.family_buckets.items()}
        self._positions = {fam: 0 for fam in self.families}
        if self.sampling_mode == "balanced_no_replacement":
            for fam in self.families:
                self.rng.shuffle(self._queues[fam])

    def _reset_epoch(self) -> None:
        self._epoch_index += 1
        self._epoch_remaining = {fam: self.min_family_size for fam in self.families}
        self._family_cursor = 0

    def _next_active_family(self) -> str:
        for _ in range(len(self.families)):
            fam = self.families[self._family_cursor % len(self.families)]
            self._family_cursor = (self._family_cursor + 1) % len(self.families)
            if self._epoch_remaining[fam] > 0:
                return fam
        self._reset_epoch()
        return self._next_active_family()

    def _draw_no_replacement(self, family: str) -> Dict[str, Any]:
        queue = self._queues[family]
        pos = self._positions[family]
        if pos >= len(queue):
            queue = list(self.family_buckets[family])
            self.rng.shuffle(queue)
            self._queues[family] = queue
            self._positions[family] = 0
            pos = 0
        sample = queue[pos]
        self._positions[family] = pos + 1
        return sample

    def sample(self) -> Dict[str, Any]:
        fam = self._next_active_family()
        self._epoch_remaining[fam] -= 1
        if self.sampling_mode == "balanced_no_replacement":
            return self._draw_no_replacement(fam)
        if self.sampling_mode == "balanced_with_replacement":
            return self.rng.choice(self.family_buckets[fam])
        raise ValueError(f"Unsupported sampling_mode: {self.sampling_mode}")

    def summary(self) -> Dict[str, Any]:
        return {
            "sampling_mode": self.sampling_mode,
            "families": list(self.families),
            "family_sizes": dict(self.family_sizes),
            "min_family_size": int(self.min_family_size),
            "balanced_epoch_size": int(self.balanced_epoch_size),
            "expected_draws_per_family_per_epoch": int(self.min_family_size),
        }


# ─── 分层诊断工具函数 ───

def _is_identical_floats(vals, atol=1e-8):
    """判断一组浮点数是否"全相同"（使用 max-min < atol 而非严格相等）。"""
    if len(vals) < 2:
        return True
    return (max(vals) - min(vals)) < atol


def _safe_np_corr(xs, ys):
    try:
        if len(xs) < 3 or len(ys) < 3:
            return float("nan")
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return float("nan")


def _same_answer_diff_reward_stats(
    final_answers: List[str],
    rewards: List[float],
    *,
    task_type: str,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    norm_finals = [normalize_text_for_analysis(a, task_type=task_type) for a in final_answers]
    groups: Dict[str, List[float]] = {}
    for ans, reward in zip(norm_finals, rewards):
        if not ans:
            continue
        groups.setdefault(ans, []).append(float(reward))

    total_pairs = 0
    diff_pairs = 0
    max_reward_spread = 0.0
    n_answer_groups = 0
    for vals in groups.values():
        if len(vals) < 2:
            continue
        n_answer_groups += 1
        spread = max(vals) - min(vals)
        max_reward_spread = max(max_reward_spread, spread)
        total_pairs += len(vals) * (len(vals) - 1) // 2
        if spread > atol:
            diff_pairs += len(vals) * (len(vals) - 1) // 2

    return {
        "has_same_answer_pair": total_pairs > 0,
        "same_answer_pair_count": int(total_pairs),
        "same_answer_diff_reward_pair_count": int(diff_pairs),
        "same_answer_diff_reward_pair_rate": (float(diff_pairs) / float(total_pairs)) if total_pairs > 0 else 0.0,
        "same_answer_diff_reward": bool(diff_pairs > 0),
        "same_answer_reward_max_spread": float(max_reward_spread),
        "same_answer_group_count": int(n_answer_groups),
    }


class MiniGRPO:
    """GRPO 训练器（v4: 分层诊断 + soft shaping 支持）。"""

    def __init__(
        self, model, processor,
        reward_fn,
        reward_backend: str = "response_conditioned_final_answer_ate",
        device: str = "cuda:0",
        lr: float = 1e-5,
        group_size: int = 4,
        max_new_tokens: int = 32,
        kl_coeff: float = 0.01,
        lora_applied: bool = False,
        use_ref_kl: bool = True,
        initial_params: Optional[Dict[str, torch.Tensor]] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        prompt_mode: str = "short_evidence_v1",
        answer_text_source: str = "final_answer",
        policy_logprob_scope: str = "final_answer",
        reward_logprob_scope: str = "final_answer",
        strict_final_answer_scope: bool = True,
        invalid_final_answer_penalty: float = -0.75,
        reward_condition_on_prefix: bool = True,
        reward_mix_mode: str = "family_default",
        main_reward_mode: str = "routed_gated_evidence",
        counting_answer_coef: float = 1.0,
        counting_visual_coef: float = 0.15,
        existence_answer_coef: float = 1.0,
        existence_visual_coef: float = 0.0,
        answer_anchor_coef: float = 0.20,
        allow_ground_truth_for_main_rewards: bool = False,
        task_type: str = "existence",
        allow_prompt_side_reward_for_smoke: bool = False,
        prompt_side_constant_group_stop: int = 3,
        reward_style_debias: bool = True,
        style_debias_length_coef: float = 0.03,
        style_debias_abstention_coef: float = 0.10,
        style_debias_template_coef: float = 0.15,
        style_debias_parse_fail_coef: float = 0.10,
        style_debias_negation_coef: float = 0.0,
        style_debias_evidence_missing_coef: float = 0.06,
        style_debias_evidence_placeholder_coef: float = 0.08,
        style_debias_max_penalty: float = 0.35,
        hard_template_veto: bool = True,
        hard_template_veto_cap: float = -0.50,
        hard_template_veto_extra_penalty: float = 0.25,
    ):
        self.model = model
        self.processor = processor
        self.reward_fn = reward_fn
        self.reward_backend = reward_backend
        self.device = device
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.kl_coeff = kl_coeff
        self.lora_applied = lora_applied
        self.use_ref_kl = use_ref_kl
        self.initial_params = initial_params
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.prompt_mode = prompt_mode
        self.answer_text_source = answer_text_source
        self.policy_logprob_scope = policy_logprob_scope
        self.reward_logprob_scope = reward_logprob_scope
        self.strict_final_answer_scope = bool(strict_final_answer_scope)
        self.invalid_final_answer_penalty = float(invalid_final_answer_penalty)
        self.reward_condition_on_prefix = bool(reward_condition_on_prefix)
        self.reward_mix_mode = str(reward_mix_mode or "family_default").strip().lower()
        self.main_reward_mode = str(main_reward_mode or "routed_gated_evidence").strip().lower()
        self.counting_answer_coef = float(counting_answer_coef)
        self.counting_visual_coef = float(counting_visual_coef)
        self.existence_answer_coef = float(existence_answer_coef)
        self.existence_visual_coef = float(existence_visual_coef)
        self.answer_anchor_coef = float(answer_anchor_coef)
        self.allow_ground_truth_for_main_rewards = bool(allow_ground_truth_for_main_rewards)
        self.task_type = task_type
        self.allow_prompt_side_reward_for_smoke = bool(allow_prompt_side_reward_for_smoke)
        self.prompt_side_constant_group_stop = int(max(1, prompt_side_constant_group_stop))
        self.reward_style_debias = bool(reward_style_debias)
        self.style_debias_length_coef = float(style_debias_length_coef)
        self.style_debias_abstention_coef = float(style_debias_abstention_coef)
        self.style_debias_template_coef = float(style_debias_template_coef)
        self.style_debias_parse_fail_coef = float(style_debias_parse_fail_coef)
        self.style_debias_negation_coef = float(style_debias_negation_coef)
        self.style_debias_evidence_missing_coef = float(style_debias_evidence_missing_coef)
        self.style_debias_evidence_placeholder_coef = float(style_debias_evidence_placeholder_coef)
        self.style_debias_max_penalty = float(style_debias_max_penalty)
        self.hard_template_veto = bool(hard_template_veto)
        self.hard_template_veto_cap = float(hard_template_veto_cap)
        self.hard_template_veto_extra_penalty = float(hard_template_veto_extra_penalty)
        self.batched_generate = str(os.environ.get("EVIDENCE_RL_BATCHED_GENERATE", "1")).strip().lower() not in {
            "0", "false", "no", "off",
        }
        self.paper_uniform_sample = str(os.environ.get("EVIDENCE_RL_PAPER_SAMPLE", "1")).strip().lower() not in {
            "0", "false", "no", "off",
        }
        try:
            self.gen_batch_size = max(1, int(os.environ.get("EVIDENCE_RL_GEN_BATCH", "32")))
        except ValueError:
            self.gen_batch_size = 32
        self._logged_generate_mode = False
        self._prompt_side_identical_group_streak = 0
        self._prompt_side_identical_group_total = 0
        self._n_groups_seen = 0
        self._n_groups_raw_reward_nonconstant = 0
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=0.01,
        )
        self._tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    def _candidate_sampling_kwargs(self, task_type: str, candidate_idx: int, attempt_idx: int = 0) -> Dict[str, Any]:
        if not self.do_sample:
            return {}

        family = infer_task_family(task_type=task_type, question="")
        temperature = float(self.temperature)
        top_p = float(self.top_p)

        # Short-answer families collapse easily; keep them slightly hotter by default.
        if family in {"existence", "counting"}:
            temperature = max(temperature, 1.10)
            top_p = max(top_p, 0.97)

        temp_offsets = [0.00, 0.12, -0.06, 0.20, -0.12, 0.28, -0.18, 0.34]
        top_p_offsets = [0.00, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04]
        off_i = candidate_idx % len(temp_offsets)
        temperature += temp_offsets[off_i] + 0.10 * float(attempt_idx)
        top_p += top_p_offsets[off_i] + 0.01 * float(attempt_idx)

        temperature = self._clip(temperature, 0.85, 1.65)
        top_p = self._clip(top_p, 0.90, 0.995)
        return {
            "temperature": temperature,
            "top_p": top_p,
        }

    @staticmethod
    def _answer_anchor_gate_value(visual_signal: float) -> float:
        if not math.isfinite(float(visual_signal)):
            return 0.0
        v = float(visual_signal)
        if v <= -0.10:
            return 0.0
        if v >= 0.25:
            return 1.0
        return max(0.0, min(1.0, (v + 0.10) / 0.35))


    def _reward_mix_coefs(self, family: str) -> Tuple[float, float]:
        fam = (family or "").strip().lower()
        if self.reward_mix_mode == "legacy_anchor":
            return 0.0, 1.0
        if fam == "counting":
            return self.counting_answer_coef, self.counting_visual_coef
        if fam == "existence":
            return self.existence_answer_coef, self.existence_visual_coef
        return 1.0, 0.0

    def _compose_training_reward(
        self,
        *,
        family: str,
        ans_score: float,
        visual_signal: float,
        legacy_anchor_gate_value: float,
    ) -> Dict[str, float]:
        if self.reward_mix_mode == "legacy_anchor":
            anchor_reward = self.answer_anchor_coef * float(ans_score) * float(legacy_anchor_gate_value)
            total = float(visual_signal) + float(anchor_reward)
            return {
                "training_answer_coef": 0.0,
                "training_visual_coef": 1.0,
                "training_answer_reward": float(anchor_reward),
                "training_visual_reward": float(visual_signal),
                "training_reward_total": float(total),
                "answer_anchor_reward": float(anchor_reward),
            }

        ans_coef, vis_coef = self._reward_mix_coefs(family)
        answer_reward = float(ans_coef) * float(ans_score)
        visual_reward = float(vis_coef) * float(visual_signal)
        total = answer_reward + visual_reward
        return {
            "training_answer_coef": float(ans_coef),
            "training_visual_coef": float(vis_coef),
            "training_answer_reward": float(answer_reward),
            "training_visual_reward": float(visual_reward),
            "training_reward_total": float(total),
            "answer_anchor_reward": 0.0,
        }

    def _compose_main_experiment_reward(
        self,
        *,
        reward_detail: Dict[str, Any],
        family: str,
    ) -> Dict[str, float]:
        """主实验 reward contract：直接消费 action_logprob_ate 输出的三路主 reward。"""
        mode = self.main_reward_mode
        field_map = {
            "correctness_only": "reward_main_correctness_only",
            "additive_evidence": "reward_main_additive_evidence",
            "routed_gated_evidence": "reward_main_routed_gated_evidence",
        }
        active_field = field_map.get(mode, "reward_main_routed_gated_evidence")
        total = float(reward_detail.get(active_field, 0.0))
        correctness_reward = float(reward_detail.get("correctness_reward", 0.0))
        evidence_part = total - correctness_reward
        evidence_allowed = bool(reward_detail.get("main_reward_evidence_allowed", False))
        return {
            "training_answer_coef": 1.0,
            "training_visual_coef": 1.0 if evidence_allowed else 0.0,
            "training_answer_reward": correctness_reward,
            "training_visual_reward": evidence_part,
            "training_reward_total": total,
            "answer_anchor_reward": 0.0,
            "training_reward_mode": mode,
            "training_reward_mode_effective": reward_detail.get("main_reward_mode_effective", mode),
            "training_reward_active_field": active_field,
        }

    def _resolve_task_type(self, sample: Dict[str, Any]) -> str:
        tt = (self.task_type or "").strip().lower()
        if tt:
            return tt
        return (sample.get("task_type") or "existence").strip().lower()

    @staticmethod
    def _sample_has_supervision(sample: Dict[str, Any]) -> bool:
        """Return whether auto-backend should use supervised shaping for this sample.

        Only existence/yes-no style tasks are allowed to route into the
        supervised yes/no branch. Open-form tasks such as counting may carry
        `gt_present` as dataset metadata, but that signal is semantically
        incompatible with yes/no correctness parsing and would otherwise poison
        reward routing.
        """
        family = infer_task_family(
            task_type=(sample.get("task_type") or ""),
            question=(sample.get("question") or ""),
        )
        if family != "existence":
            return False

        if sample.get("gt_present") is not None:
            return True

        gt_answer = (sample.get("answer") or "").strip()
        return parse_yesno(gt_answer) in {"yes", "no"}

    def _resolve_effective_reward_backend(self, sample: Dict[str, Any]) -> str:
        backend = (self.reward_backend or "").strip().lower()
        if backend in ("", "auto", "action_logprob_ate_auto"):
            return "action_logprob_ate_supervised" if self._sample_has_supervision(sample) else "action_logprob_ate_nolabel"
        if backend in {"response_conditioned_final_answer_ate", "response_conditioned_final_answer_ate_nolabel"}:
            return "action_logprob_ate_nolabel"
        return backend

    @staticmethod
    def _target_min_unique_final_answers(task_type: str, n: int) -> int:
        family = infer_task_family(task_type=task_type, question="")
        if family in {"existence", "counting"} and n > 1:
            return min(2, int(n))
        return 0

    def _select_answer_text(self, raw_response: str, final_answer: str) -> str:
        if (self.answer_text_source or "").strip().lower() == "final_answer":
            return (final_answer or "").strip()
        return (raw_response or "").strip()


    def _select_policy_scope_and_span(
        self,
        raw_response: str,
        effective_task_type: str,
    ) -> Dict[str, Any]:
        """Resolve effective policy scope and token indices to score."""
        requested = (self.policy_logprob_scope or "").strip().lower()
        rr = (raw_response or "").strip()

        def _scope_failure(reason: str, span=None):
            if self.strict_final_answer_scope:
                return {
                    "policy_scope_effective": "invalid_final_answer",
                    "policy_scope_fallback_reason": reason,
                    "policy_prefix_text": "",
                    "policy_target_text": "",
                    "final_answer_char_span": span,
                    "token_indices_to_score": None,
                    "scope_valid": False,
                }
            return {
                "policy_scope_effective": "full_response",
                "policy_scope_fallback_reason": reason,
                "policy_prefix_text": "",
                "policy_target_text": rr,
                "final_answer_char_span": span,
                "token_indices_to_score": None,
                "scope_valid": True,
            }

        if requested != "final_answer":
            return {
                "policy_scope_effective": "full_response",
                "policy_scope_fallback_reason": "",
                "policy_prefix_text": "",
                "policy_target_text": rr,
                "final_answer_char_span": None,
                "token_indices_to_score": None,
                "scope_valid": True,
            }

        part = extract_final_answer_with_prefix(rr, task_type=effective_task_type)
        fa = (part.get("final_answer") or "").strip()
        span = part.get("char_span")
        prefix = part.get("prefix_text_before_answer") or ""
        src = part.get("extraction_source") or "fallback_raw"

        if not fa:
            return _scope_failure("empty_final_answer", span if span else None)

        if not (isinstance(span, (tuple, list)) and len(span) == 2):
            return _scope_failure(f"missing_char_span(source={src})")

        try:
            cs, ce = int(span[0]), int(span[1])
        except Exception:
            return _scope_failure(f"invalid_char_span(source={src})")

        resp_ids = self._tok.encode(rr, add_special_tokens=False)
        _, spans = _decode_with_token_char_spans(self._tok, resp_ids[: self.max_new_tokens])
        if not spans:
            return _scope_failure("token_char_alignment_failed", (cs, ce))

        idxs = _token_indices_overlapping_char_span(spans, cs, ce)
        if not idxs:
            return _scope_failure("empty_token_span_for_answer", (cs, ce))

        return {
            "policy_scope_effective": "final_answer",
            "policy_scope_fallback_reason": "",
            "policy_prefix_text": prefix,
            "policy_target_text": fa,
            "final_answer_char_span": (cs, ce),
            "token_indices_to_score": idxs,
            "scope_valid": True,
        }

    def _compute_reward_detail(
        self,
        image,
        prompt: str,
        sample: Dict[str, Any],
        response_text: str,
        answer_text: str,
        reward_token_indices_to_score: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        backend_requested = (self.reward_backend or "").strip().lower()
        backend = self._resolve_effective_reward_backend(sample)
        if backend in ("action_logprob_ate_supervised", "action_logprob_ate_nolabel", "response_conditioned_final_answer_ate", "response_conditioned_final_answer_ate_nolabel"):
            kwargs = dict(
                image=image,
                question=sample["question"],
                target_bbox=sample["target_bbox"],
                response_text=response_text,
                answer_text=answer_text,
                image_width=sample.get("image_width"),
                image_height=sample.get("image_height"),
                response_text_source="raw_generated_text",
                answer_text_source="generated_text" if (self.answer_text_source or "").strip().lower() == "final_answer" else "raw_response",
                token_indices_to_score=reward_token_indices_to_score,
                reward_logprob_scope=self.reward_logprob_scope,
                strict_scope=self.strict_final_answer_scope,
                reward_condition_on_prefix=self.reward_condition_on_prefix,
                task_type=(sample.get("task_type") or ""),
                metadata=sample,
            )

            if backend == "action_logprob_ate_supervised" or self.allow_ground_truth_for_main_rewards:
                kwargs["gt_answer"] = sample.get("answer")
                kwargs["gt_present"] = sample.get("gt_present")
                gt_passed_to_compute = True
            else:
                # No-label backend must not pass GT into reward compute path.
                gt_passed_to_compute = False

            r = self.reward_fn.compute(**kwargs)

            if backend == "action_logprob_ate_nolabel":
                if "score_base_no_label" not in r:
                    raise ValueError("action_logprob_ate_nolabel requires score_base_no_label in reward output")
                used_reward = float(r["score_base_no_label"])
                used_field = "score_base_no_label"
                used_sem = ("response_conditioned_final_answer_ate" if self.reward_condition_on_prefix else "answer_only_final_answer_ate")
                uses_gt = False
            else:
                if "supervised_shaped_reward" not in r:
                    raise ValueError("action_logprob_ate_supervised requires supervised_shaped_reward in reward output")
                used_reward = float(r["supervised_shaped_reward"])
                used_field = "supervised_shaped_reward"
                used_sem = ("response_conditioned_supervised_ate" if self.reward_condition_on_prefix else "answer_only_supervised_ate")
                uses_gt = True

            r["reward"] = used_reward
            r["reward_backend"] = backend
            r["reward_backend_effective"] = backend
            r["reward_backend_requested"] = backend_requested
            r["used_reward_field"] = used_field
            r["used_reward_semantics"] = used_sem
            r["uses_ground_truth"] = uses_gt
            r["uses_ground_truth_for_training_step"] = uses_gt
            r["ground_truth_passed_to_reward_compute"] = gt_passed_to_compute
            r["reward_scope_semantics"] = ("response_conditioned" if self.reward_condition_on_prefix else "answer_only_isolated")
            r["response_conditioned"] = bool(self.reward_condition_on_prefix)
            r["reward_condition_on_prefix"] = bool(self.reward_condition_on_prefix)
            r["candidate_dependent_reward"] = True
            r["candidate_dependent"] = True
            r["safe_for_group_ranking"] = True
            return r

        if backend == "logodds_ate":
            r = self.reward_fn.compute(
                image=image,
                question=prompt,
                target_bbox=sample["target_bbox"],
                image_width=sample.get("image_width"),
                image_height=sample.get("image_height"),
            )
            r["parsed_answer"] = "other"
            r["is_correct"] = None
            r["reward"] = float(r.get("reward", 0.0))
            r["reward_backend"] = backend
            r["reward_backend_effective"] = backend
            r["reward_backend_requested"] = backend_requested
            r["used_reward_field"] = "reward"
            r["used_reward_semantics"] = "prompt_side_logodds_ate"
            r["uses_ground_truth"] = False
            r["uses_ground_truth_for_training_step"] = False
            r["ground_truth_passed_to_reward_compute"] = False
            r["reward_scope_semantics"] = "prompt_side_sample_level"
            r["response_conditioned"] = False
            r["candidate_dependent_reward"] = False
            r["candidate_dependent"] = False
            r["safe_for_group_ranking"] = False
            return r

        raise ValueError(f"Unknown reward_backend: {self.reward_backend}")

    @torch.no_grad()
    def _generation_pad_ids(self):
        tok = self._tok
        eos = getattr(tok, "eos_token_id", None)
        pad = getattr(tok, "pad_token_id", None)
        if pad is None:
            pad = eos
        return pad, eos

    def _uniform_gen_kwargs(self) -> Dict[str, Any]:
        pad_id, eos_id = self._generation_pad_ids()
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "use_cache": True,
        }
        if pad_id is not None:
            kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            kwargs["eos_token_id"] = eos_id
        if self.do_sample:
            kwargs["temperature"] = float(self.temperature)
            kwargs["top_p"] = float(self.top_p)
        return kwargs

    def _generate_chunk(self, inputs, bsz: int, gen_kwargs: Dict[str, Any]) -> List[str]:
        if bsz <= 0:
            return []
        try:
            expanded = expand_mm_inputs(inputs, bsz)
            with torch.inference_mode():
                from model_loader import generation_model
                out = generation_model(self.model).generate(**expanded, **gen_kwargs)
            prompt_len = int(inputs["input_ids"].shape[-1])
            texts = self.processor.batch_decode(
                out[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return [str(t).strip() for t in texts]
        except torch.cuda.OutOfMemoryError:
            if bsz <= 1:
                raise
            torch.cuda.empty_cache()
            mid = max(1, bsz // 2)
            print(f"[GRPO] generate OOM at bsz={bsz}, retry {mid}+{bsz-mid}", flush=True)
            return self._generate_chunk(inputs, mid, gen_kwargs) + self._generate_chunk(
                inputs, bsz - mid, gen_kwargs
            )

    def _generate_responses(self, inputs, n, task_type: str = ""):
        self.reward_fn.rep.active = False

        family = infer_task_family(task_type=task_type, question="")
        min_unique_final_answers = self._target_min_unique_final_answers(task_type=task_type, n=n)
        short_answer_family = family in {"existence", "counting"}
        max_draws = max(1, int(n))
        if self.do_sample and short_answer_family:
            max_draws = max(int(n) * 4, int(n) + 4)

        if not self._logged_generate_mode:
            print(
                f"[GRPO] generate batched={int(self.batched_generate)} "
                f"gen_batch={self.gen_batch_size} paper_sample={int(self.paper_uniform_sample)} "
                f"use_cache=1 group={int(n)} max_draws={max_draws}",
                flush=True,
            )
            self._logged_generate_mode = True

        from model_loader import generation_model
        gen_model = generation_model(self.model)
        ckpt_on = bool(getattr(gen_model, "is_gradient_checkpointing", False))
        if ckpt_on and hasattr(gen_model, "gradient_checkpointing_disable"):
            gen_model.gradient_checkpointing_disable()
        cfg = getattr(gen_model, "config", None)
        old_cache = getattr(cfg, "use_cache", None) if cfg is not None else None
        if cfg is not None:
            cfg.use_cache = True

        candidate_pool: List[Dict[str, Any]] = []
        try:
            if self.batched_generate:
                remaining = int(max_draws)
                while remaining > 0:
                    bsz = min(int(self.gen_batch_size), remaining)
                    if self.paper_uniform_sample:
                        gen_kwargs = self._uniform_gen_kwargs()
                    else:
                        # Preserve archived per-slot temperature jitter, but still batch
                        # copies that share identical kwargs.
                        gen_kwargs = dict(self._uniform_gen_kwargs())
                        jitter = self._candidate_sampling_kwargs(
                            task_type=task_type,
                            candidate_idx=len(candidate_pool),
                            attempt_idx=len(candidate_pool) // max(1, int(n)),
                        )
                        gen_kwargs.update(jitter)
                    texts = self._generate_chunk(inputs, bsz, gen_kwargs)
                    for text in texts:
                        ext = extract_final_answer(text, task_type=task_type)
                        final_answer = (ext.get("final_answer") or "").strip()
                        candidate_pool.append({
                            "raw_text": text,
                            "raw_norm": normalize_raw_text_for_dedup(text),
                            "final_answer": final_answer,
                            "final_norm": normalize_final_answer_for_dedup(final_answer, task_type=task_type),
                        })
                    remaining -= bsz
                    if not (self.do_sample and short_answer_family):
                        if len(candidate_pool) >= int(n):
                            break
                        continue
                    uniq_final = {c["final_norm"] for c in candidate_pool if c.get("final_norm")}
                    uniq_raw = {c["raw_norm"] for c in candidate_pool if c.get("raw_norm")}
                    if (
                        len(candidate_pool) >= int(n)
                        and len(uniq_final) >= int(min_unique_final_answers)
                        and len(uniq_raw) >= min(int(n), 2)
                    ):
                        break
            else:
                for draw_idx in range(max_draws):
                    slot_idx = draw_idx % max(1, int(n))
                    attempt_idx = draw_idx // max(1, int(n))
                    gen_kwargs = self._uniform_gen_kwargs()
                    if not self.paper_uniform_sample:
                        gen_kwargs.update(
                            self._candidate_sampling_kwargs(
                                task_type=task_type,
                                candidate_idx=slot_idx,
                                attempt_idx=attempt_idx,
                            )
                        )
                    out = gen_model.generate(**inputs, **gen_kwargs)
                    text = self.processor.batch_decode(
                        out[:, inputs["input_ids"].shape[-1]:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                    ext = extract_final_answer(text, task_type=task_type)
                    final_answer = (ext.get("final_answer") or "").strip()
                    candidate_pool.append({
                        "raw_text": text,
                        "raw_norm": normalize_raw_text_for_dedup(text),
                        "final_answer": final_answer,
                        "final_norm": normalize_final_answer_for_dedup(final_answer, task_type=task_type),
                    })
                    if not (self.do_sample and short_answer_family):
                        if len(candidate_pool) >= int(n):
                            break
                        continue
                    uniq_final = {c["final_norm"] for c in candidate_pool if c.get("final_norm")}
                    uniq_raw = {c["raw_norm"] for c in candidate_pool if c.get("raw_norm")}
                    if len(candidate_pool) >= int(n) and len(uniq_final) >= int(min_unique_final_answers) and len(uniq_raw) >= min(int(n), 2):
                        break
        finally:
            if cfg is not None and old_cache is not None:
                cfg.use_cache = old_cache
            if ckpt_on and hasattr(gen_model, "gradient_checkpointing_enable"):
                try:
                    gen_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    gen_model.gradient_checkpointing_enable()

        selected: List[Dict[str, Any]] = []
        used_idx = set()
        seen_raw = set()
        seen_final = set()

        if min_unique_final_answers > 0:
            for idx, cand in enumerate(candidate_pool):
                final_norm = cand.get("final_norm") or ""
                if not final_norm or final_norm in seen_final:
                    continue
                selected.append(cand)
                used_idx.add(idx)
                seen_final.add(final_norm)
                if cand.get("raw_norm"):
                    seen_raw.add(cand["raw_norm"])
                if len(seen_final) >= int(min_unique_final_answers) or len(selected) >= int(n):
                    break

        for mode in ("prefer_new_raw", "prefer_new_final", "fallback_any"):
            if len(selected) >= int(n):
                break
            for idx, cand in enumerate(candidate_pool):
                if idx in used_idx:
                    continue
                raw_norm = cand.get("raw_norm") or ""
                final_norm = cand.get("final_norm") or ""
                take = False
                if mode == "prefer_new_raw":
                    take = bool(raw_norm) and raw_norm not in seen_raw
                elif mode == "prefer_new_final":
                    take = bool(final_norm) and final_norm not in seen_final
                else:
                    take = True
                if not take:
                    continue

                selected.append(cand)
                used_idx.add(idx)
                if raw_norm:
                    seen_raw.add(raw_norm)
                if final_norm:
                    seen_final.add(final_norm)
                if len(selected) >= int(n):
                    break

        while len(selected) < int(n):
            if candidate_pool:
                selected.append(candidate_pool[len(selected) % len(candidate_pool)])
            else:
                selected.append({"raw_text": "", "raw_norm": "", "final_answer": "", "final_norm": ""})

        return [cand.get("raw_text", "") for cand in selected[: int(n)]]

    def _generate_group(self, inputs, n: int, task_type: str = "") -> List[str]:
        """One GRPO group. Under DDP each rank samples its own slice, then the group is gathered."""
        import torch.distributed as dist

        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if world <= 1:
            return self._generate_responses(inputs, n, task_type=task_type)
        rank = dist.get_rank()
        local_n = n // world + (1 if rank < (n % world) else 0)
        seed = int(torch.initial_seed() % (2**31)) + rank * 10007 + int(getattr(self, "_ddp_gen_round", 0))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self._ddp_gen_round = int(getattr(self, "_ddp_gen_round", 0)) + 1
        local = self._generate_responses(inputs, local_n, task_type=task_type)
        gathered: List[Optional[List[str]]] = [None] * world
        dist.all_gather_object(gathered, local)
        merged: List[str] = []
        for part in gathered:
            merged.extend(part or [])
        return merged[: int(n)]

    def _compute_response_logprobs(
        self,
        inputs,
        response_text,
        with_grad: bool = True,
        token_indices_to_score: Optional[List[int]] = None,
    ):
        """Mean log-prob under current model.

        If token_indices_to_score is provided, only those continuation tokens are
        averaged (span-conditioned scoring).

        Returns (mean_lp, n_scored_tokens).
        """
        response_ids = self._tok.encode(response_text, add_special_tokens=False)
        if not response_ids:
            input_dev = inputs.get("input_ids").device if isinstance(inputs, dict) and ("input_ids" in inputs) else self.device
            z = torch.tensor(0.0, device=input_dev)
            return (z if with_grad else 0.0), 0
        response_ids = response_ids[:self.max_new_tokens]
        n_resp = len(response_ids)

        fwd_inputs = extend_multimodal_inputs(inputs, response_ids, self.device)

        if with_grad:
            logits = self.model(**fwd_inputs).logits.float()
        else:
            with torch.no_grad():
                logits = self.model(**fwd_inputs).logits.float()

        prompt_len = inputs["input_ids"].shape[1]
        B = logits.shape[0]
        resp_logits = logits[:, prompt_len - 1: prompt_len - 1 + n_resp, :]
        log_probs = F.log_softmax(resp_logits, dim=-1)
        resp_ids_t = torch.tensor(
            [response_ids], dtype=torch.long, device=logits.device
        ).expand(B, -1)
        token_lps = log_probs.gather(-1, resp_ids_t.unsqueeze(-1)).squeeze(-1)

        if token_indices_to_score is not None:
            idxs = [i for i in token_indices_to_score if 0 <= i < n_resp]
            if not idxs:
                z = torch.tensor(0.0, device=logits.device)
                return (z if with_grad else 0.0), 0
            token_lps = token_lps[:, idxs]
            n_scored = len(idxs)
        else:
            n_scored = n_resp

        mean_lp = token_lps.mean()
        return (mean_lp if with_grad else mean_lp.item()), n_scored

    @torch.no_grad()
    def _get_reference_logprob(self, inputs, response_text, token_indices_to_score: Optional[List[int]] = None):
        """Reference model log-prob (no grad). Returns (float, n_tok)."""
        if self.lora_applied and hasattr(self.model, "disable_adapter_layers"):
            self.model.disable_adapter_layers()
            try:
                lp, n = self._compute_response_logprobs(
                    inputs,
                    response_text,
                    with_grad=False,
                    token_indices_to_score=token_indices_to_score,
                )
            finally:
                self.model.enable_adapter_layers()
            return lp, n

        if self.initial_params is not None:
            saved = {}
            try:
                for name, p in self.model.named_parameters():
                    if p.requires_grad and name in self.initial_params:
                        saved[name] = p.data.clone()
                        p.data.copy_(self.initial_params[name])
                lp, n = self._compute_response_logprobs(
                    inputs,
                    response_text,
                    with_grad=False,
                    token_indices_to_score=token_indices_to_score,
                )
            finally:
                for name, p in self.model.named_parameters():
                    if name in saved:
                        p.data.copy_(saved[name])
            return lp, n

        return 0.0, 0

    def train_step(self, sample: Dict[str, Any], image_root: str) -> Dict[str, Any]:
        image = open_sample_image(sample, image_root=image_root)

        # Build stable prompt (compatible with legacy raw_question).
        effective_task_type = self._resolve_task_type(sample)
        sample_for_prompt = dict(sample)
        sample_for_prompt["task_type"] = effective_task_type
        prompt = build_generation_prompt(sample_for_prompt, prompt_mode=self.prompt_mode)

        inputs = prepare_inputs(self.processor, image, prompt, self.device)

        # ── 1. 生成 G 个回答 ──
        self.model.eval()
        raw_responses = self._generate_group(inputs, self.group_size, task_type=effective_task_type)

        # ── 1.5 抽取 final answer + 统一 scope（用于 parse/logprob/reward scope/诊断） ──
        final_answers: List[str] = []
        final_sources: List[str] = []
        scope_infos: List[Dict[str, Any]] = []
        policy_scope_effective: List[str] = []
        policy_scope_fallback_reason: List[str] = []
        policy_prefix_texts: List[str] = []
        policy_target_texts: List[str] = []
        final_answer_char_spans: List[Any] = []
        invalid_final_answer_count = 0
        for rr in raw_responses:
            ext = extract_final_answer(rr, task_type=effective_task_type)
            fa = (ext.get("final_answer") or "").strip()
            src = ext.get("source") or "fallback_raw"
            final_answers.append(fa)
            final_sources.append(src)

            scope_info = self._select_policy_scope_and_span(rr, effective_task_type)
            scope_infos.append(scope_info)
            policy_scope_effective.append(scope_info.get("policy_scope_effective", "full_response"))
            policy_scope_fallback_reason.append(scope_info.get("policy_scope_fallback_reason", ""))
            policy_prefix_texts.append(scope_info.get("policy_prefix_text", ""))
            policy_target_texts.append(scope_info.get("policy_target_text", ""))
            final_answer_char_spans.append(scope_info.get("final_answer_char_span"))
            if not bool(scope_info.get("scope_valid", True)):
                invalid_final_answer_count += 1

        # Parse/correctness should explicitly bind to selected text.
        parse_on_text = (self.answer_text_source or "raw_response").strip().lower()

        # ── 2. 计算每个回答的 raw reward（无梯度）──
        reward_details = []
        raw_rewards = []
        answer_texts_used: List[str] = []
        gt_answer = sample.get("answer", "")
        answer_scores = []
        answer_anchor_rewards: List[float] = []
        for rr, fa, scope_info in zip(raw_responses, final_answers, scope_infos):
            answer_text = self._select_answer_text(rr, fa)
            answer_texts_used.append(answer_text)
            idxs = scope_info.get("token_indices_to_score")
            r = self._compute_reward_detail(
                image=image,
                prompt=prompt,
                sample=sample,
                response_text=rr,
                answer_text=answer_text,
                reward_token_indices_to_score=idxs,
            )
            if not bool(scope_info.get("scope_valid", True)):
                r["reward"] = min(float(r.get("reward", 0.0)), self.invalid_final_answer_penalty)
                r["invalid_final_answer_scope"] = True
                r["invalid_final_answer_reason"] = scope_info.get("policy_scope_fallback_reason", "")
            else:
                r["invalid_final_answer_scope"] = False
                r["invalid_final_answer_reason"] = ""

            family = infer_task_family(task_type=sample.get("task_type", ""), question=sample.get("question", ""))
            visual_signal = float(
                r.get("relative_evidence_margin", r.get("score_base_no_label", r.get("score_base", r.get("reward", 0.0))))
            )
            answer_anchor_gate_value = self._answer_anchor_gate_value(visual_signal)
            answer_anchor_visual_gate_passed = bool(answer_anchor_gate_value > 0.0)
            ans_score = float(
                r.get(
                    "correctness_score",
                    task_aware_answer_score(
                        answer_text,
                        gt_answer,
                        task_type=sample.get("task_type", ""),
                        question=sample.get("question", ""),
                    ),
                )
            )
            answer_scores.append(ans_score)
            if self.reward_mix_mode == "main_experiment":
                reward_mix = self._compose_main_experiment_reward(
                    reward_detail=r,
                    family=family,
                )
            else:
                reward_mix = self._compose_training_reward(
                    family=family,
                    ans_score=ans_score,
                    visual_signal=visual_signal,
                    legacy_anchor_gate_value=answer_anchor_gate_value,
                )
            answer_anchor_rewards.append(float(reward_mix.get("answer_anchor_reward", 0.0)))
            r["answer_anchor_family"] = family
            r["answer_anchor_family_enabled"] = bool(family in {"counting", "existence"})
            r["answer_anchor_visual_signal"] = float(visual_signal)
            r["answer_anchor_gate_value"] = float(answer_anchor_gate_value)
            r["answer_anchor_visual_gate_passed"] = bool(answer_anchor_visual_gate_passed)
            r["answer_anchor_score"] = float(ans_score)
            r["answer_anchor_reward"] = float(reward_mix.get("answer_anchor_reward", 0.0))
            r["training_reward_mix_mode"] = self.reward_mix_mode
            r.update(reward_mix)

            raw_reward_i = float(reward_mix["training_reward_total"])
            raw_rewards.append(raw_reward_i)
            reward_details.append(r)

        raw_rewards_before_anchor = [float(d.get("reward", 0.0)) for d in reward_details]
        raw_rewards_before_style_debias = list(raw_rewards)
        style_features: List[Dict[str, Any]] = []
        style_penalties: List[float] = []
        style_penalty_components: List[Dict[str, float]] = []
        hard_template_veto_applied: List[bool] = []
        hard_template_veto_reasons: List[str] = []
        for i, (rr, fa, src, ans_text, rd) in enumerate(zip(raw_responses, final_answers, final_sources, answer_texts_used, reward_details)):
            is_parse_fail = False
            if parse_on_text == "final_answer":
                is_parse_fail = (not fa) or (src == "fallback_raw")
            elif effective_task_type == "existence":
                is_parse_fail = (rd.get("parsed_answer") == "other")
            feats = compute_style_bias_features(
                ans_text,
                raw_response=rr,
                parsed_answer=rd.get("parsed_answer", ""),
                task_type=effective_task_type,
                is_parse_fail=is_parse_fail,
            )
            style_features.append(feats)
            pen = compute_style_penalty(
                feats,
                length_coef=self.style_debias_length_coef,
                abstention_coef=self.style_debias_abstention_coef,
                template_coef=self.style_debias_template_coef,
                parse_fail_coef=self.style_debias_parse_fail_coef,
                negation_coef=self.style_debias_negation_coef,
                evidence_missing_coef=self.style_debias_evidence_missing_coef,
                evidence_placeholder_coef=self.style_debias_evidence_placeholder_coef,
                max_penalty=self.style_debias_max_penalty,
            )
            style_penalties.append(float(pen["penalty"]))
            style_penalty_components.append(dict(pen["components"]))

            veto_flag = bool(feats.get("hard_template_veto", False))
            actual_veto_applied = bool(self.hard_template_veto and veto_flag)
            hard_template_veto_applied.append(actual_veto_applied)
            hard_template_veto_reasons.append(str(feats.get("hard_template_veto_reason", "")))
            if actual_veto_applied:
                raw_rewards[i] = min(
                    float(raw_rewards[i]) - self.hard_template_veto_extra_penalty,
                    self.hard_template_veto_cap,
                )
                rd["hard_template_veto_applied"] = True
                rd["hard_template_veto_reason"] = feats.get("hard_template_veto_reason", "")
                rd["hard_template_veto_reasons"] = feats.get("hard_template_veto_reasons", [])
            else:
                rd["hard_template_veto_applied"] = False
                rd["hard_template_veto_reason"] = ""
                rd["hard_template_veto_reasons"] = []

        raw_rewards_after_hard_veto = list(raw_rewards)
        if self.reward_style_debias and style_penalties:
            raw_rewards = [float(r) - float(p) for r, p in zip(raw_rewards, style_penalties)]

        # ── 3. 计算 KL penalties（无梯度）并合成 final rewards ──
        kl_penalties = []
        self.model.eval()
        for rr, scope_info in zip(raw_responses, scope_infos):
            rr_text = (rr or "").strip()
            eff_scope = scope_info["policy_scope_effective"]
            idxs = scope_info.get("token_indices_to_score")

            if (not bool(scope_info.get("scope_valid", True))) or (not rr_text.strip()):
                kl_i = 0.0
            elif self.use_ref_kl and self.kl_coeff > 0:
                if eff_scope == "final_answer" and idxs is not None:
                    policy_lp, _ = self._compute_response_logprobs(
                        inputs, rr_text, with_grad=False, token_indices_to_score=idxs
                    )
                    ref_lp, _ = self._get_reference_logprob(
                        inputs, rr_text, token_indices_to_score=idxs
                    )
                else:
                    policy_lp, _ = self._compute_response_logprobs(inputs, rr_text, with_grad=False)
                    ref_lp, _ = self._get_reference_logprob(inputs, rr_text)
                kl_i = policy_lp - ref_lp
            else:
                kl_i = 0.0
            kl_penalties.append(float(kl_i))

        raw_rewards_arr = np.array(raw_rewards)
        raw_reward_std = float(raw_rewards_arr.std())
        raw_reward_identical = bool(raw_reward_std < 1e-8)

        self._n_groups_seen += 1
        if raw_reward_identical:
            self._prompt_side_identical_group_total += 1
        else:
            self._n_groups_raw_reward_nonconstant += 1

        prompt_side_reward_blocked = False
        if (self.reward_backend or "").strip().lower() == "logodds_ate":
            if raw_reward_identical:
                self._prompt_side_identical_group_streak += 1
            else:
                self._prompt_side_identical_group_streak = 0
            if self._prompt_side_identical_group_streak >= self.prompt_side_constant_group_stop:
                prompt_side_reward_blocked = True
                raise RuntimeError(
                    "prompt-side/sample-level reward (logodds_ate) produced near-constant raw rewards "
                    f"for {self._prompt_side_identical_group_streak} consecutive groups; "
                    "abort smoke training because it is unsafe as default group reward"
                )
        kl_penalties_arr = np.array(kl_penalties)
        final_rewards = raw_rewards_arr - self.kl_coeff * kl_penalties_arr
        # ── 4. 组内标准化得 advantage ──
        reward_std = float(final_rewards.std())
        all_identical = bool(reward_std < 1e-8)
        mean_r = final_rewards.mean()

        used_zero_var_fallback = False
        if reward_std < 1e-4:
            used_zero_var_fallback = True
            advantages = final_rewards - mean_r
        else:
            advantages = (final_rewards - mean_r) / reward_std

        n_zero_adv = int(np.sum(np.abs(advantages) < 1e-6))
        n_positive_adv = int(np.sum(advantages > 1e-6))
        n_negative_adv = int(np.sum(advantages < -1e-6))

        # ── 5. Policy gradient ──
        self.model.train()
        total_policy_loss = None
        n_in_loss = 0

        for rr, adv, scope_info in zip(raw_responses, advantages, scope_infos):
            if not rr.strip():
                continue

            rr_text = (rr or "").strip()
            eff_scope = scope_info["policy_scope_effective"]
            idxs = scope_info.get("token_indices_to_score")
            if not bool(scope_info.get("scope_valid", True)):
                continue

            if eff_scope == "final_answer" and idxs is not None:
                policy_lp, n_tok = self._compute_response_logprobs(
                    inputs, rr_text, with_grad=True, token_indices_to_score=idxs
                )
            else:
                policy_lp, n_tok = self._compute_response_logprobs(inputs, rr_text, with_grad=True)
            if n_tok == 0:
                continue

            p_loss = -adv * policy_lp
            total_policy_loss = p_loss if total_policy_loss is None else (total_policy_loss + p_loss)
            n_in_loss += 1

        if n_in_loss > 0:
            avg_loss = total_policy_loss / n_in_loss
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0,
            )
            self.optimizer.step()
        else:
            avg_loss = torch.tensor(0.0, device=inputs["input_ids"].device)

        # ── 6. 监控指标 ──
        # Backward-compatible alias: keep `responses` as raw responses.
        responses = list(raw_responses)

        lengths = [len(r.split()) for r in raw_responses]
        # parsed_answers should reflect the text used for parsing (answer_text_source).
        parsed_answers = [
            _task_aware_parse_label(ans_text, effective_task_type, sample.get("question") or "")
            for ans_text in answer_texts_used
        ]
        yn_counts = Counter(parsed_answers)
        total_resp = len(responses)
        answer_score_mean = float(np.mean(answer_scores)) if answer_scores else 0.0
        task_aware_correct_rate = answer_score_mean
        label_correct_flags = [d.get("is_correct", False) for d in reward_details]
        label_correct_rate = sum(1 for c in label_correct_flags if c is True) / max(1, total_resp)
        yesno_monitor = _is_yesno_monitor_task(effective_task_type, sample.get("question") or "")
        negation_rate = float(np.mean([1.0 if f.get("has_negation") else 0.0 for f in style_features])) if style_features else 0.0
        abstention_rate = float(np.mean([1.0 if f.get("is_abstention_like") else 0.0 for f in style_features])) if style_features else 0.0
        template_rate = float(np.mean([1.0 if f.get("is_template_like") else 0.0 for f in style_features])) if style_features else 0.0
        numeric_short_rate = float(np.mean([1.0 if f.get("is_numeric_short") else 0.0 for f in style_features])) if style_features else 0.0
        mean_length_excess = float(np.mean([float(f.get("length_excess", 0.0)) for f in style_features])) if style_features else 0.0
        mean_style_penalty = float(np.mean(style_penalties)) if style_penalties else 0.0
        raw_reward_length_corr = _safe_np_corr(raw_rewards_before_style_debias, [float(f.get("length_tokens", 0.0)) for f in style_features])
        debiased_reward_length_corr = _safe_np_corr(raw_rewards, [float(f.get("length_tokens", 0.0)) for f in style_features])
        extraction_fail_rate = float(
            sum(1 for fa, src in zip(final_answers, final_sources) if (not fa) or src == "fallback_raw") / max(1, total_resp)
        )
        if parse_on_text == "final_answer":
            parse_fail_rate = extraction_fail_rate
        elif yesno_monitor:
            parse_fail_rate = yn_counts.get("other", 0) / max(1, total_resp)
        else:
            parse_fail_rate = float(sum(1 for x in parsed_answers if x == "__empty__") / max(1, total_resp))

        # ── v4 新增：分层诊断指标 ──
        # A. 文本层
        unique_raw = set(raw_responses)
        exact_match_all = len(unique_raw) == 1 and total_resp > 1
        n_unique_responses = len(unique_raw)

        # Final-answer layer (new): normalize for existence/counting to avoid superficial diffs.
        norm_finals = [normalize_text_for_analysis(a, task_type=effective_task_type) for a in final_answers]
        norm_finals = [x for x in norm_finals if x != ""]
        n_unique_final_answers = len(set(norm_finals)) if norm_finals else 0
        final_answer_identical = (n_unique_final_answers == 1 and total_resp > 1)

        # B. parse 层 (parse_on_text)
        parse_identical = len(set(parsed_answers)) == 1 and total_resp > 1
        correctness_vals = answer_scores
        correctness_identical = len(set(str(c) for c in correctness_vals)) == 1 and total_resp > 1

        # C. reward 分层
        delta_logprobs = [d.get("delta_logprob", 0.0) for d in reward_details]
        score_bases = [d.get("score_base", 0.0) for d in reward_details]
        pre_kl_rewards = raw_rewards  # pre-KL = raw reward from reward_fn
        final_rewards_list = final_rewards.tolist()

        raw_delta_identical = _is_identical_floats(delta_logprobs)
        score_base_identical = _is_identical_floats(score_bases)
        pre_kl_reward_identical = _is_identical_floats(pre_kl_rewards)
        final_reward_identical = _is_identical_floats(final_rewards_list)
        same_answer_raw_stats = _same_answer_diff_reward_stats(
            final_answers,
            pre_kl_rewards,
            task_type=effective_task_type,
        )
        same_answer_final_stats = _same_answer_diff_reward_stats(
            final_answers,
            final_rewards_list,
            task_type=effective_task_type,
        )

        # collapse_layer: 最早出现 "全相同" 的层级
        collapse_layer = None
        for _cl_name, _cl_flag in [
            ("text/raw_response", exact_match_all),
            ("final_answer", final_answer_identical),
            ("parse", parse_identical),
            ("raw_delta", raw_delta_identical),
            ("score_base", score_base_identical),
            ("pre_kl", pre_kl_reward_identical),
            ("final", final_reward_identical),
        ]:
            if _cl_flag:
                collapse_layer = _cl_name
                break

        reward_contract_ok = (
            (self.answer_text_source or "").strip().lower() == "final_answer"
            and (self.policy_logprob_scope or "").strip().lower() == "final_answer"
            and (self.reward_logprob_scope or "").strip().lower() == "final_answer"
            and bool(self.strict_final_answer_scope)
        )
        raw_reward_nonconstant_group_rate = (
            float(self._n_groups_raw_reward_nonconstant) / float(max(1, self._n_groups_seen))
        )
        task_family = infer_task_family(
            task_type=effective_task_type,
            question=sample.get("question", ""),
        )

        step_info = {
            "mean_raw_reward": float(raw_rewards_arr.mean()),
            "raw_reward_std_within_group": raw_reward_std,
            "raw_reward_identical": raw_reward_identical,
            "n_groups_raw_reward_identical": int(self._prompt_side_identical_group_total),
            "raw_reward_nonconstant_group_rate": raw_reward_nonconstant_group_rate,
            "mean_ref_kl_penalty": float(kl_penalties_arr.mean()),
            "mean_final_reward": float(final_rewards.mean()),
            "mean_reward": float(final_rewards.mean()),  # 兼容旧字段
            "std_reward": float(reward_std),
            "reward_std_within_group": float(reward_std),
            "total_reward_mean": float(final_rewards.mean()),
            "ced_score_mean": float(raw_rewards_arr.mean()),
            "answer_score": answer_score_mean,
            "answer_score_mean": answer_score_mean,
            "task_aware_correct_rate_within_group": task_aware_correct_rate,
            "label_correct_rate_within_group": label_correct_rate,
            "all_rewards_identical": all_identical,
            "used_zero_variance_fallback": used_zero_var_fallback,
            "advantages": advantages.tolist(),
            "n_zero_advantage": n_zero_adv,
            "n_positive_advantage": n_positive_adv,
            "n_negative_advantage": n_negative_adv,
            "n_in_loss": n_in_loss,
            "loss": avg_loss.item(),
            "mean_policy_loss": (total_policy_loss / n_in_loss).item() if n_in_loss > 0 else 0.0,
            "mean_kl": float(np.mean(kl_penalties)),
            "mean_total_loss": avg_loss.item(),
            "output_length_mean": float(np.mean(lengths)),
            "output_length_std": float(np.std(lengths)),
            "yes_rate": (yn_counts.get("yes", 0) / total_resp) if yesno_monitor else 0.0,
            "no_rate": (yn_counts.get("no", 0) / total_resp) if yesno_monitor else 0.0,
            "other_rate": (yn_counts.get("other", 0) / total_resp) if yesno_monitor else 0.0,
            "repetition_rate": float(np.mean([_repetition_rate(r) for r in responses])),
            "format_error_rate": extraction_fail_rate if parse_on_text == "final_answer" else ((yn_counts.get("other", 0) / total_resp) if yesno_monitor else parse_fail_rate),
            "parse_fail_rate": parse_fail_rate,
            "negation_rate": negation_rate,
            "abstention_rate": abstention_rate,
            "template_rate": template_rate,
            "numeric_short_rate": numeric_short_rate,
            "mean_length_excess": mean_length_excess,
            "style_debias_applied": self.reward_style_debias,
            "mean_style_penalty": mean_style_penalty,
            "mean_answer_anchor_reward": float(np.mean(answer_anchor_rewards)) if answer_anchor_rewards else 0.0,
            "answer_anchor_rewards": answer_anchor_rewards,
            "raw_rewards_before_anchor": raw_rewards_before_anchor,
            "raw_rewards_before_style_debias": raw_rewards_before_style_debias,
            "raw_rewards_after_hard_veto": raw_rewards_after_hard_veto,
            "style_penalties": style_penalties,
            "style_features": style_features,
            "style_penalty_components": style_penalty_components,
            "hard_template_veto_enabled": self.hard_template_veto,
            "hard_template_veto_cap": self.hard_template_veto_cap,
            "hard_template_veto_extra_penalty": self.hard_template_veto_extra_penalty,
            "hard_template_veto_rate": float(np.mean([1.0 if x else 0.0 for x in hard_template_veto_applied])) if hard_template_veto_applied else 0.0,
            "hard_template_veto_count": int(sum(1 for x in hard_template_veto_applied if x)),
            "hard_template_veto_applied": hard_template_veto_applied,
            "hard_template_veto_reasons": hard_template_veto_reasons,
            "raw_reward_length_corr": raw_reward_length_corr,
            "debiased_reward_length_corr": debiased_reward_length_corr,
            "final_answer_extraction_fail_rate": extraction_fail_rate,
            "parsed_answers": parsed_answers,
            "correct_rate_within_group": task_aware_correct_rate,
            "responses": responses,
            "raw_responses": raw_responses,
            "final_answers": final_answers,
            "final_answer_extraction_sources": final_sources,
            # New stable I/O debug fields (requested)
            "answer_text_used": answer_texts_used,
            "policy_prefix_text": policy_prefix_texts,
            "policy_target_text": policy_target_texts,
            "policy_scope_effective": policy_scope_effective,
            "policy_scope_fallback_reason": policy_scope_fallback_reason,
            "final_answer_char_span": final_answer_char_spans,
            "rewards": raw_rewards,
            "question": sample["question"],
            "generation_prompt": prompt,
            "prompt_mode": self.prompt_mode,
            "task_type": effective_task_type,
            "task_family": task_family,
            "reward_backend": (reward_details[0].get("reward_backend_effective") if reward_details else self.reward_backend),
            "reward_backend_requested": self.reward_backend,
            "reward_backend_effective": (reward_details[0].get("reward_backend_effective") if reward_details else self.reward_backend),
            "response_text_source": "raw_generated_text",
            "used_reward_field": reward_details[0].get("used_reward_field") if reward_details else "",
            "used_reward_semantics": reward_details[0].get("used_reward_semantics") if reward_details else "",
            "reward_scope_semantics": reward_details[0].get("reward_scope_semantics") if reward_details else "",
            "response_conditioned": bool(reward_details[0].get("response_conditioned")) if reward_details else False,
            "candidate_dependent_reward": bool(reward_details[0].get("candidate_dependent_reward")) if reward_details else False,
            "safe_for_group_ranking": bool(reward_details[0].get("safe_for_group_ranking")) if reward_details else False,
            "uses_ground_truth": bool(reward_details[0].get("uses_ground_truth")) if reward_details else False,
            "uses_ground_truth_for_training_step": bool(
                reward_details[0].get("uses_ground_truth_for_training_step")
            ) if reward_details else False,
            "ground_truth_passed_to_reward_compute": bool(
                reward_details[0].get("ground_truth_passed_to_reward_compute")
            ) if reward_details else False,
            "answer_text_source": self.answer_text_source,
            "answer_text_source_effective": "final_answer" if parse_on_text == "final_answer" else "raw_response",
            "policy_logprob_scope": self.policy_logprob_scope,
            "reward_logprob_scope": self.reward_logprob_scope,
            "strict_final_answer_scope": self.strict_final_answer_scope,
            "invalid_final_answer_penalty": self.invalid_final_answer_penalty,
            "invalid_final_answer_count": invalid_final_answer_count,
            "invalid_final_answer_rate": float(invalid_final_answer_count) / float(max(1, total_resp)),
            "parse_on_text": parse_on_text,
            "prompt_side_reward_blocked": prompt_side_reward_blocked,
            "reward_contract_ok": reward_contract_ok,
            "image_file": sample["image_file"],
            "gt_present": sample.get("gt_present"),
            "gt_answer": sample.get("answer"),
            "delta_logprobs": delta_logprobs,
            # v4 分层诊断
            "exact_match_all": exact_match_all,
            "n_unique_responses": n_unique_responses,
            "n_unique_final_answers": n_unique_final_answers,
            "final_answer_identical": final_answer_identical,
            "parse_identical": parse_identical,
            "correctness_identical": correctness_identical,
            "raw_delta_identical": raw_delta_identical,
            "score_base_identical": score_base_identical,
            "pre_kl_reward_identical": pre_kl_reward_identical,
            "final_reward_identical": final_reward_identical,
            "same_final_answer_diff_raw_reward": same_answer_raw_stats["same_answer_diff_reward"],
            "same_final_answer_diff_raw_reward_pair_rate": same_answer_raw_stats["same_answer_diff_reward_pair_rate"],
            "same_final_answer_raw_reward_max_spread": same_answer_raw_stats["same_answer_reward_max_spread"],
            "same_final_answer_diff_final_reward": same_answer_final_stats["same_answer_diff_reward"],
            "same_final_answer_diff_final_reward_pair_rate": same_answer_final_stats["same_answer_diff_reward_pair_rate"],
            "same_final_answer_final_reward_max_spread": same_answer_final_stats["same_answer_reward_max_spread"],
            "collapse_layer": collapse_layer,
            "score_bases": score_bases,
            "pre_kl_rewards": pre_kl_rewards,
            "final_rewards": final_rewards_list,
            "kl_penalties": kl_penalties,
            # 完整 reward details 用于 debug dump
            "_reward_details": reward_details,
        }

        if all_identical:
            step_info["_warning"] = "ALL_REWARDS_IDENTICAL"

        return step_info


def _compute_group_diagnostics(all_metrics):
    """从全量训练 step 指标中汇总分层诊断统计（用于 final summary）。"""
    valid = [m for m in all_metrics if "error" not in m]
    n = len(valid)
    if n == 0:
        return {}

    diag = {}

    # A. 文本层
    diag["exact_match_group_rate"] = sum(
        1 for m in valid if m.get("exact_match_all")) / n
    unique_counts = [m.get("n_unique_responses", 0) for m in valid]
    diag["mean_unique_responses_per_group"] = float(np.mean(unique_counts)) if unique_counts else 0

    # A2. final answer 层
    diag["final_answer_identical_group_rate"] = sum(
        1 for m in valid if m.get("final_answer_identical")) / n
    fa_unique_counts = [m.get("n_unique_final_answers", 0) for m in valid]
    diag["mean_unique_final_answers_per_group"] = float(np.mean(fa_unique_counts)) if fa_unique_counts else 0

    # B. parse 层
    diag["parse_identical_group_rate"] = sum(
        1 for m in valid if m.get("parse_identical")) / n
    diag["correctness_identical_group_rate"] = sum(
        1 for m in valid if m.get("correctness_identical")) / n

    # C. reward 分层
    diag["raw_delta_identical_group_rate"] = sum(
        1 for m in valid if m.get("raw_delta_identical")) / n
    diag["score_base_identical_group_rate"] = sum(
        1 for m in valid if m.get("score_base_identical")) / n
    diag["pre_kl_reward_identical_group_rate"] = sum(
        1 for m in valid if m.get("pre_kl_reward_identical")) / n
    diag["final_reward_identical_group_rate"] = sum(
        1 for m in valid if m.get("final_reward_identical")) / n

    # D. 对齐诊断 + 方差诊断
    diag["same_final_answer_diff_raw_reward_group_rate"] = sum(
        1 for m in valid if m.get("same_final_answer_diff_raw_reward")) / n
    diag["same_final_answer_diff_final_reward_group_rate"] = sum(
        1 for m in valid if m.get("same_final_answer_diff_final_reward")) / n
    diag["mean_same_final_answer_diff_raw_reward_pair_rate"] = float(np.mean([
        float(m.get("same_final_answer_diff_raw_reward_pair_rate", 0.0)) for m in valid
    ]))
    diag["mean_same_final_answer_diff_final_reward_pair_rate"] = float(np.mean([
        float(m.get("same_final_answer_diff_final_reward_pair_rate", 0.0)) for m in valid
    ]))
    diag["mean_same_final_answer_raw_reward_max_spread"] = float(np.mean([
        float(m.get("same_final_answer_raw_reward_max_spread", 0.0)) for m in valid
    ]))
    diag["mean_same_final_answer_final_reward_max_spread"] = float(np.mean([
        float(m.get("same_final_answer_final_reward_max_spread", 0.0)) for m in valid
    ]))
    diag["mean_invalid_final_answer_rate"] = float(np.mean([
        float(m.get("invalid_final_answer_rate", 0.0)) for m in valid
    ]))

    diag["used_zero_variance_fallback_rate"] = sum(
        1 for m in valid if m.get("used_zero_variance_fallback")) / n
    reward_stds = [m.get("reward_std_within_group", 0) for m in valid]
    diag["mean_reward_std_within_group"] = float(np.mean(reward_stds))
    raw_reward_stds = [m.get("raw_reward_std_within_group", 0) for m in valid]
    diag["mean_raw_reward_std_within_group"] = float(np.mean(raw_reward_stds)) if raw_reward_stds else 0.0
    diag["raw_reward_identical_group_rate"] = sum(
        1 for m in valid if m.get("raw_reward_identical")
    ) / n
    diag["raw_reward_nonconstant_group_rate"] = sum(
        1 for m in valid if not m.get("raw_reward_identical")
    ) / n
    diag["n_groups_raw_reward_identical"] = sum(
        1 for m in valid if m.get("raw_reward_identical")
    )
    diag["prompt_side_reward_blocked"] = any(bool(m.get("prompt_side_reward_blocked", False)) for m in valid)
    diag["reward_contract_ok_rate"] = sum(1 for m in valid if bool(m.get("reward_contract_ok", False))) / n
    final_reward_stds = [float(np.std(m.get("final_rewards", [])))
                         for m in valid if m.get("final_rewards")]
    diag["mean_final_reward_std_within_group"] = float(np.mean(final_reward_stds)) if final_reward_stds else 0

    diag["n_groups_with_all_same_text"] = sum(
        1 for m in valid if m.get("exact_match_all"))
    # 相同 parse 但不同 text
    diag["n_groups_with_same_parse_but_diff_text"] = sum(
        1 for m in valid if m.get("parse_identical") and not m.get("exact_match_all"))
    # 不同 text 但同 final reward
    diag["n_groups_with_diff_text_but_same_final_reward"] = sum(
        1 for m in valid
        if not m.get("exact_match_all") and m.get("final_reward_identical"))

    # collapse_layer 分布
    cl_counts = Counter(m.get("collapse_layer") for m in valid)
    diag["collapse_layer_distribution"] = dict(cl_counts)

    # E. style / nuisance 聚合（用于最终 summary）
    def _mean_key(k: str) -> float:
        vals = [float(m.get(k, 0.0)) for m in valid]
        return float(np.mean(vals)) if vals else 0.0

    diag["mean_task_aware_correct_rate"] = _mean_key("task_aware_correct_rate_within_group")
    diag["mean_label_correct_rate"] = _mean_key("label_correct_rate_within_group")
    diag["mean_negation_rate"] = _mean_key("negation_rate")
    diag["mean_abstention_rate"] = _mean_key("abstention_rate")
    diag["mean_template_rate"] = _mean_key("template_rate")
    diag["mean_numeric_short_rate"] = _mean_key("numeric_short_rate")
    diag["mean_style_penalty"] = _mean_key("mean_style_penalty")
    diag["mean_answer_anchor_reward"] = _mean_key("mean_answer_anchor_reward")
    diag["mean_hard_template_veto_rate"] = _mean_key("hard_template_veto_rate")
    diag["mean_raw_reward_length_corr"] = _mean_key("raw_reward_length_corr")
    diag["mean_debiased_reward_length_corr"] = _mean_key("debiased_reward_length_corr")

    return diag


def _write_debug_dump(all_metrics, n_groups, output_path, mode="all"):
    """把 group 的样本级详细信息写入 jsonl。
    mode:
      "all"           — 前 n_groups 个 group
      "collapse_only" — 只 dump 有 collapse_layer 的 group（最多 n_groups 个）
    """
    valid = [m for m in all_metrics if "error" not in m]
    if mode == "collapse_only":
        dump_groups = [m for m in valid if m.get("collapse_layer") is not None][:n_groups]
    else:
        dump_groups = valid[:n_groups]

    with open(output_path, "w") as f:
        for gi, m in enumerate(dump_groups):
            raw_responses = m.get("raw_responses") or m.get("responses", [])
            final_answers = m.get("final_answers", [])
            final_sources = m.get("final_answer_extraction_sources", [])
            answer_texts_used = m.get("answer_text_used", [])
            policy_prefix_texts = m.get("policy_prefix_text", [])
            policy_target_texts = m.get("policy_target_text", [])
            policy_scope_effective = m.get("policy_scope_effective", [])
            policy_scope_fallback_reason = m.get("policy_scope_fallback_reason", [])
            final_answer_char_spans = m.get("final_answer_char_span", [])
            reward_details = m.get("_reward_details", [])
            final_rewards = m.get("final_rewards", [])
            kl_penalties = m.get("kl_penalties", [])
            advantages = m.get("advantages", [])

            group_record = {
                "step_idx": m.get("step", gi),
                "group_idx": gi,
                "question": m.get("question"),
                "generation_prompt": m.get("generation_prompt"),
                "prompt_mode": m.get("prompt_mode"),
                "task_type": m.get("task_type"),
                "answer_text_source": m.get("answer_text_source"),
                "policy_logprob_scope": m.get("policy_logprob_scope"),
                "reward_logprob_scope": m.get("reward_logprob_scope"),
                "strict_final_answer_scope": m.get("strict_final_answer_scope"),
                "parse_on_text": m.get("parse_on_text"),
                "image_file": m.get("image_file"),
                "gt_answer": m.get("gt_answer"),
                "gt_present": m.get("gt_present"),
                # group 级别诊断
                "exact_match_all": m.get("exact_match_all"),
                "parse_identical": m.get("parse_identical"),
                "final_answer_identical": m.get("final_answer_identical"),
                "final_reward_identical": m.get("final_reward_identical"),
                "raw_delta_identical": m.get("raw_delta_identical"),
                "collapse_layer": m.get("collapse_layer"),
                "samples": [],
            }

            for si, rr in enumerate(raw_responses):
                rd = reward_details[si] if si < len(reward_details) else {}
                fa = final_answers[si] if si < len(final_answers) else ""
                src = final_sources[si] if si < len(final_sources) else ""
                sample_record = {
                    # Keep legacy key but make it explicit.
                    "response_text": rr,
                    "raw_response_text": rr,
                    "final_answer": fa,
                    "final_answer_extraction_source": src,
                    "final_answer_char_span": final_answer_char_spans[si] if si < len(final_answer_char_spans) else None,
                    "parsed_answer": rd.get("parsed_answer"),
                    "is_correct": rd.get("is_correct"),
                    "answer_text_used": answer_texts_used[si] if si < len(answer_texts_used) else None,
                    "policy_scope_effective": policy_scope_effective[si] if si < len(policy_scope_effective) else None,
                    "policy_scope_fallback_reason": policy_scope_fallback_reason[si] if si < len(policy_scope_fallback_reason) else None,
                    "policy_prefix_text": policy_prefix_texts[si] if si < len(policy_prefix_texts) else None,
                    "policy_target_text": policy_target_texts[si] if si < len(policy_target_texts) else None,
                    "reward_logprob_scope_effective": rd.get("reward_logprob_scope_effective"),
                    "reward_token_span_present": rd.get("reward_token_span_present"),
                    "invalid_final_answer_scope": rd.get("invalid_final_answer_scope"),
                    "invalid_final_answer_reason": rd.get("invalid_final_answer_reason"),
                    "answer_anchor_reward": rd.get("answer_anchor_reward", 0),
                    "hard_template_veto_applied": rd.get("hard_template_veto_applied", False),
                    "hard_template_veto_reason": rd.get("hard_template_veto_reason", ""),
                    "delta_logprob": rd.get("delta_logprob", 0),
                    "delta_ans_margin": rd.get("delta_ans_margin", 0),
                    "score_resp": rd.get("score_resp", 0),
                    "score_ans": rd.get("score_ans", 0),
                    "score_base": rd.get("score_base", 0),
                    "reward_before_kl": rd.get("reward", 0),
                    "final_reward": final_rewards[si] if si < len(final_rewards) else 0,
                    "ref_kl": kl_penalties[si] if si < len(kl_penalties) else 0,
                    "advantage": advantages[si] if si < len(advantages) else 0,
                    "reward_bias": rd.get("reward_bias", 0),
                    "reward_scale": rd.get("reward_scale", 0),
                    "reward_mode": rd.get("reward_mode"),
                    "parse_source": rd.get("parse_source"),
                    "error": rd.get("error"),
                }
                group_record["samples"].append(sample_record)

            f.write(json.dumps(group_record, ensure_ascii=False) + "\n")

    return len(dump_groups)


def main():
    ap = argparse.ArgumentParser(description="Mini-GRPO Smoke Test (v4)")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--vqa_file", default="")
    ap.add_argument("--coco_image_dir", default="")
    ap.add_argument("--dataset_name", default="legacy_vqa", choices=["legacy_vqa", "vg_brutal"])
    ap.add_argument("--dataset_file", default="", help="vg_brutal 数据文件路径")
    ap.add_argument("--image_root", default="", help="vg_brutal 图片根目录")
    ap.add_argument("--data_split", default="train", choices=["all", "train", "val", "probe"],
                    help="训练数据 split 过滤；VG brutal 默认 train")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--allow_missing_images", action="store_true",
                    help="允许图片路径不完整时继续执行（默认关闭，VG brutal 会在 preflight 前 fail fast）")
    ap.add_argument("--integrity_max_check", type=int, default=0,
                    help="integrity 检查样本数上限；0 表示全量检查")
    ap.add_argument("--output_dir", default="results/grpo_smoke")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_coeff", type=float, default=0.01)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--n_trainable_layers", type=int, default=4)
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--disable_ref_kl", action="store_true")
    ap.add_argument("--reward_cap", type=float, default=5.0)
    ap.add_argument("--wrong_penalty", type=float, default=-1.0)
    ap.add_argument("--unparseable_penalty", type=float, default=-1.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_interval", type=int, default=10)
    ap.add_argument("--max_train_samples", type=int, default=1000)
    # v6: stable I/O controls (defaults keep legacy behavior)
    ap.add_argument("--prompt_mode", default="short_evidence_v1",
                    choices=["raw_question", "task_aware_v1", "short_evidence_v1", "final_answer_only_v1"],
                    help="prompt 构造模式：short_evidence_v1=短 Evidence + Final answer；task_aware_v1=较完整的 Evidence+Final answer；final_answer_only_v1=只输出 Final answer；raw_question=兼容旧行为")
    ap.add_argument("--answer_text_source", default="final_answer",
                    choices=["raw_response", "final_answer"],
                    help="reward 的 answer_text/parse 绑定到哪段文本")
    ap.add_argument("--policy_logprob_scope", default="final_answer",
                    choices=["full_response", "final_answer"],
                    help="policy/ref logprob/KL/优势诊断的作用域")
    ap.add_argument("--reward_logprob_scope", default="final_answer",
                    choices=["full_response", "final_answer"],
                    help="reward 中 delta_logprob 的计分作用域；默认与 policy 对齐")
    ap.add_argument("--strict_final_answer_scope", dest="strict_final_answer_scope", action="store_true",
                    help="当 final answer 抽取/span 对齐失败时，不再静默回退到 full_response")
    ap.add_argument("--no_strict_final_answer_scope", dest="strict_final_answer_scope", action="store_false",
                    help="允许 final answer scope 失败时回退到 full_response（默认关闭）")
    ap.set_defaults(strict_final_answer_scope=True)
    ap.add_argument("--invalid_final_answer_penalty", type=float, default=-0.75,
                    help="strict final-answer scope 失败时的额外惩罚上限")
    ap.add_argument("--reward_condition_on_prefix", dest="reward_condition_on_prefix", action="store_true",
                    help="reward 计分时保留完整 response prefix 条件（默认开启）")
    ap.add_argument("--no_reward_condition_on_prefix", dest="reward_condition_on_prefix", action="store_false",
                    help="reward 计分时隔离 final answer 文本，去掉前缀 reasoning 条件")
    ap.set_defaults(reward_condition_on_prefix=True)
    ap.add_argument("--task_type", default="auto",
                    help="训练 task_type；auto=按 dataset_name 选默认（vg_brutal->all, legacy_vqa->existence）")
    ap.add_argument("--task_type_allowlist", default="",
                    help="可选：对 smoke 训练样本再做一次 task_type 白名单过滤（逗号分隔）")
    ap.add_argument("--disable_family_balanced_sampling", action="store_true",
                    help="关闭按 task family 均衡采样（默认开启，用于避免 counting/yes-no 某一类独占 smoke）")
    ap.add_argument("--reward_backend", default="auto",
                    choices=["auto", "response_conditioned_final_answer_ate", "action_logprob_ate_supervised", "action_logprob_ate_nolabel", "logodds_ate"],
                    help="训练 reward backend；auto/response_conditioned_final_answer_ate 仅对 existence/yes-no 样本在有 GT 时走 supervised，其余任务退回 no-label")
    ap.add_argument("--allow_prompt_side_reward_for_smoke", action="store_true",
                    help="危险开关：允许在 smoke 中使用 prompt-side reward(logodds_ate) 做 group reward")
    ap.add_argument("--prompt_side_constant_group_stop", type=int, default=3,
                    help="当 prompt-side raw reward 连续多少个 group 近似常数时强制中止")
    # v4 新增参数
    ap.add_argument("--reward_mode", default="soft_shaping",
                    choices=["legacy_hard_gate", "raw_delta", "soft_shaping"],
                    help="reward 计算模式（默认: soft_shaping）")
    ap.add_argument("--tau_resp", type=float, default=0.20,
                    help="delta_resp 的 tanh 温度参数")
    ap.add_argument("--tau_ans", type=float, default=1.00,
                    help="delta_ans_margin 的 tanh 温度参数")
    ap.add_argument("--alpha_resp", type=float, default=0.70,
                    help="score_resp 权重")
    ap.add_argument("--alpha_ans", type=float, default=0.30,
                    help="score_ans 权重")
    ap.add_argument("--min_reward", type=float, default=-1.25,
                    help="soft_shaping 最终 reward 下界")
    ap.add_argument("--max_reward", type=float, default=1.00,
                    help="soft_shaping 最终 reward 上界")
    ap.add_argument("--disable_reward_style_debias", action="store_true",
                    help="关闭 reward nuisance-style debias（默认开启）")
    ap.add_argument("--style_debias_length_coef", type=float, default=0.03,
                    help="长度偏离目标区间的 nuisance 扣分系数（不是长度奖励）")
    ap.add_argument("--style_debias_abstention_coef", type=float, default=0.10,
                    help="abstention-like 模板的 nuisance 扣分系数")
    ap.add_argument("--style_debias_template_coef", type=float, default=0.15,
                    help="模板化/占位符输出的 nuisance 扣分系数")
    ap.add_argument("--style_debias_parse_fail_coef", type=float, default=0.10,
                    help="parse/extraction fail 的 nuisance 扣分系数")
    ap.add_argument("--style_debias_negation_coef", type=float, default=0.0,
                    help="显式 negation 风格项的 nuisance 扣分系数（默认 0，避免误伤合法 no）")
    ap.add_argument("--style_debias_evidence_missing_coef", type=float, default=0.06,
                    help="Evidence 行为空时的 nuisance 扣分系数（仅在结构化输出时生效）")
    ap.add_argument("--style_debias_evidence_placeholder_coef", type=float, default=0.08,
                    help="Evidence 行为占位/模板值时的 nuisance 扣分系数")
    ap.add_argument("--style_debias_max_penalty", type=float, default=0.35,
                    help="单个 candidate 的最大 style nuisance 扣分")
    ap.add_argument("--reward_mix_mode", default="family_default", choices=["family_default", "legacy_anchor", "main_experiment"],
                    help="训练 reward 混合模式：family_default=按任务族拆 answer/visual 主辅项；legacy_anchor=保留旧的视觉主项+GT anchor；main_experiment=直接消费主实验三路 reward")
    ap.add_argument("--main_reward_mode", default="routed_gated_evidence",
                    choices=["correctness_only", "additive_evidence", "routed_gated_evidence"],
                    help="当 reward_mix_mode=main_experiment 时启用的正式主实验 reward mode")
    ap.add_argument("--counting_answer_coef", type=float, default=1.0,
                    help="counting 任务的正确性主奖励系数")
    ap.add_argument("--counting_visual_coef", type=float, default=0.15,
                    help="counting 任务的视觉辅助奖励系数")
    ap.add_argument("--existence_answer_coef", type=float, default=1.0,
                    help="existence 任务的正确性主奖励系数")
    ap.add_argument("--existence_visual_coef", type=float, default=0.0,
                    help="existence 任务的视觉辅助奖励系数；默认 0，避免 yes/no no-label reward 主导")
    ap.add_argument("--answer_anchor_coef", type=float, default=0.20,
                    help="旧版 legacy_anchor 模式下的 GT anchor 系数；family_default 模式下不再作为主训练奖励")
    ap.add_argument("--disable_hard_template_veto", action="store_true",
                    help="关闭对模板/占位/abstention 输出的硬 veto")
    ap.add_argument("--hard_template_veto_cap", type=float, default=-0.50,
                    help="命中硬 veto 时，raw reward 会被压到不高于这个值")
    ap.add_argument("--hard_template_veto_extra_penalty", type=float, default=0.25,
                    help="命中硬 veto 时，在 cap 之前额外再减一小段")
    ap.add_argument("--debug_dump_groups", type=int, default=5,
                    help="debug dump 前 N 个 group 的详细信息")
    ap.add_argument("--debug_dump_path", default="",
                    help="debug dump 输出路径（默认: output_dir/debug_groups.jsonl）")
    ap.add_argument("--debug_dump_mode", default="all",
                    choices=["all", "collapse_only"],
                    help="debug dump 模式：all=前N个 group，collapse_only=只 dump 有 collapse 的 group")
    # v5 新增：生成采样参数
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="生成采样温度（默认 1.0；旧版硬编码 0.7 会导致短回答坍缩）")
    ap.add_argument("--top_p", type=float, default=0.95,
                    help="nucleus sampling top_p（默认 0.95）")
    ap.add_argument("--greedy", action="store_true",
                    help="关闭采样，使用 greedy decoding（用于对照实验）")
    args = ap.parse_args()

    try:
        resolved_paths = resolve_dataset_paths(args)
    except ValueError as e:
        ap.error(str(e))

    dataset_name = resolved_paths["dataset_name"]
    dataset_file = resolved_paths["dataset_file"]
    image_root = resolved_paths["image_root"]

    if args.train_ratio < 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) > 1.0:
        ap.error("Invalid split ratios: require train_ratio>=0, val_ratio>=0, train_ratio+val_ratio<=1.0")

    task_type_filter = resolve_task_type_filter(dataset_name=dataset_name, task_type_arg=args.task_type)
    task_type_allowlist = _parse_csv_lower_list(args.task_type_allowlist)
    reward_backend = (args.reward_backend or "").strip().lower()
    if reward_backend in ("", "auto"):
        reward_backend = "action_logprob_ate_auto"

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("  Mini-GRPO Smoke Test (v5: 采样参数可配 + 分层诊断)")
    print(f"  steps={args.n_steps} group={args.group_size} lr={args.lr} "
          f"kl={args.kl_coeff}")
    print(f"  reward_mode={args.reward_mode}")
    print(f"  reward_backend={reward_backend}")
    print(f"  dataset={dataset_name} dataset_file={dataset_file} image_root={image_root}")
    print(f"  data_split={args.data_split} train_ratio={args.train_ratio} val_ratio={args.val_ratio}")
    print(f"  task_type_filter={task_type_filter or 'all'}")
    if task_type_allowlist:
        print(f"  task_type_allowlist={','.join(task_type_allowlist)}")
    print(f"  family_balanced_sampling={not args.disable_family_balanced_sampling}")
    print(f"  ref_kl={'ON' if not args.disable_ref_kl else 'OFF'}")
    do_sample = not args.greedy
    print(f"  sampling: do_sample={do_sample} temperature={args.temperature} top_p={args.top_p}")
    print(f"  prompt_mode={args.prompt_mode} answer_text_source={args.answer_text_source} policy_logprob_scope={args.policy_logprob_scope} reward_logprob_scope={args.reward_logprob_scope}")
    print(f"  strict_final_answer_scope={args.strict_final_answer_scope} invalid_final_answer_penalty={args.invalid_final_answer_penalty} reward_condition_on_prefix={args.reward_condition_on_prefix}")
    print(f"  reward_mix_mode={args.reward_mix_mode} main_reward_mode={args.main_reward_mode} counting(ans={args.counting_answer_coef},vis={args.counting_visual_coef}) existence(ans={args.existence_answer_coef},vis={args.existence_visual_coef}) legacy_answer_anchor_coef={args.answer_anchor_coef}")
    print(f"  reward_style_debias={not args.disable_reward_style_debias} length_coef={args.style_debias_length_coef} abstain_coef={args.style_debias_abstention_coef} template_coef={args.style_debias_template_coef} parse_fail_coef={args.style_debias_parse_fail_coef} negation_coef={args.style_debias_negation_coef} evidence_missing_coef={args.style_debias_evidence_missing_coef} evidence_placeholder_coef={args.style_debias_evidence_placeholder_coef}")
    print(f"  hard_template_veto={not args.disable_hard_template_veto} cap={args.hard_template_veto_cap} extra_penalty={args.hard_template_veto_extra_penalty}")
    if args.reward_mode == "soft_shaping":
        print(f"  tau_resp={args.tau_resp} tau_ans={args.tau_ans} "
              f"alpha_resp={args.alpha_resp} alpha_ans={args.alpha_ans}")
    if dataset_name == "vg_brutal":
        print(f"  当前 smoke 使用的是 VG brutal 的 {args.data_split} split")
    print("=" * 60)

    t0 = time.time()
    processor, model, cfg = load(args.model_dir, args.device, args.dtype)
    print(f"[INFO] 模型加载 ({time.time()-t0:.1f}s)")

    model, lora_applied = _setup_trainable(
        model, n_trainable_layers=args.n_trainable_layers,
        use_lora=not args.no_lora,
    )

    initial_params = None
    use_ref_kl = not args.disable_ref_kl
    if use_ref_kl and not lora_applied:
        print("[GRPO] 保存初始参数用于 reference KL...")
        initial_params = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        mb = sum(p.numel() for p in initial_params.values()) * 2 / 1e6
        print(f"  保存 {mb:.0f} MB")

    if reward_backend == "logodds_ate" and not args.allow_prompt_side_reward_for_smoke:
        raise ValueError(
            "prompt-side/sample-level reward 不适合作为默认 group reward: reward_backend=logodds_ate 已被阻断。"
            "如需仅做危险对照实验，请显式传入 --allow_prompt_side_reward_for_smoke"
        )

    if reward_backend == "logodds_ate":
        allowed = {"existence"}
        if (task_type_filter or "").strip().lower() not in allowed:
            raise ValueError(
                "reward_backend=logodds_ate 仅支持 yes/no 类任务（当前强制 task_type=existence）"
            )
        reward_fn = LogOddsATEReward(
            model=model,
            processor=processor,
            device=args.device,
            replace_mode="zero",
        )
    else:
        reward_fn = ActionLogProbATEReward(
            model, processor, device=args.device,
            reward_cap=args.reward_cap,
            wrong_penalty=args.wrong_penalty,
            unparseable_penalty=args.unparseable_penalty,
            reward_mode=args.reward_mode,
            tau_resp=args.tau_resp,
            tau_ans=args.tau_ans,
            alpha_resp=args.alpha_resp,
            alpha_ans=args.alpha_ans,
            min_reward=args.min_reward,
            max_reward=args.max_reward,
            main_reward_mode=args.main_reward_mode,
        )

    train_samples, dataset_summary, train_task = load_train_dataset(
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        image_root=image_root,
        task_type=task_type_filter,
        data_split=args.data_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples=args.max_train_samples,
        seed=args.seed,
        task_type_allowlist=task_type_allowlist,
    )

    if not train_samples:
        raise RuntimeError(
            "No training samples left after task filtering/allowlist. "
            f"task_type={task_type_filter or 'all'} allowlist={task_type_allowlist or []}"
        )

    loaded_summary = summarize_loaded_samples(train_samples)
    family_buckets = build_family_buckets(train_samples)
    family_dist = {k: len(v) for k, v in family_buckets.items()}
    dataset_summary = dict(dataset_summary)
    dataset_summary.update(loaded_summary)
    dataset_summary["task_family_distribution"] = family_dist
    dataset_summary["family_balanced_sampling"] = (not args.disable_family_balanced_sampling)
    resolved_dataset_config = {
        "dataset_name": dataset_name,
        "dataset_file": dataset_file,
        "image_root": image_root,
        "data_split": args.data_split,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "task_type_filter": train_task or "all",
        "task_type_allowlist": task_type_allowlist,
        "pair_id_unique": dataset_summary.get("pair_id_unique", None),
        "loaded_sample_count": loaded_summary["loaded_sample_count"],
        "task_type_distribution": loaded_summary["task_type_distribution"],
        "split_distribution": loaded_summary["split_distribution"],
        "task_family_distribution": family_dist,
        "family_balanced_sampling": (not args.disable_family_balanced_sampling),
    }
    with open(os.path.join(args.output_dir, "resolved_dataset_config.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_dataset_config, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] pair_id_unique={dataset_summary.get('pair_id_unique', 'N/A')}")
    print(f"[INFO] task_type_distribution={loaded_summary['task_type_distribution']}")
    print(f"[INFO] split_distribution={loaded_summary['split_distribution']}")
    if family_dist:
        print(f"[INFO] task_family_distribution={family_dist}")

    dataset_integrity = None
    if dataset_name == "vg_brutal":
        dataset_integrity = check_dataset_integrity(
            samples=train_samples,
            image_root=image_root,
            max_check=args.integrity_max_check,
        )
        with open(os.path.join(args.output_dir, "dataset_integrity.json"), "w", encoding="utf-8") as f:
            json.dump(dataset_integrity, f, indent=2, ensure_ascii=False)

        ok_rate = dataset_integrity.get("image_path_resolve_ok_rate", 0.0)
        print(
            f"[INFO] integrity: checked={dataset_integrity.get('checked_samples')} "
            f"ok_rate={ok_rate:.3f} missing={dataset_integrity.get('image_path_resolve_missing_count')}"
        )
        if dataset_integrity.get("bad_path_examples"):
            print(f"[INFO] integrity bad_path_examples(head): {dataset_integrity['bad_path_examples'][:5]}")
        if ok_rate < 1.0 and not args.allow_missing_images:
            raise RuntimeError(
                "Dataset integrity check failed before preflight: image_path_resolve_ok_rate < 1.0. "
                "Use --allow_missing_images to override."
            )

    n_pos = sum(1 for s in train_samples if s.get("gt_present"))
    n_neg = sum(1 for s in train_samples if not s.get("gt_present"))
    print(f"[INFO] 训练样本: {len(train_samples)} (pos={n_pos}, neg={n_neg}) task_filter={train_task or 'ALL'} split={args.data_split}")
    if task_type_allowlist:
        print(f"[INFO] smoke allowlist={task_type_allowlist}")

    preflight = run_training_preflight_check(
        model=model,
        processor=processor,
        reward_fn=reward_fn,
        samples=train_samples,
        image_root=image_root,
        reward_backend=reward_backend,
        prompt_mode=args.prompt_mode,
        answer_text_source=args.answer_text_source,
        policy_logprob_scope=args.policy_logprob_scope,
        max_new_tokens=args.max_new_tokens,
        group_size=args.group_size,
        n_groups=min(8, max(3, args.n_steps)),
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=do_sample,
        allow_prompt_side_reward_for_smoke=args.allow_prompt_side_reward_for_smoke,
    )
    print(f"[PRECHECK] {preflight.get('check_name')} passed={preflight.get('passed')}")
    if not preflight.get("passed", False):
        err_samples = preflight.get("error_samples", []) or []
        if err_samples:
            print(f"[PRECHECK] error_samples(head): {err_samples[:5]}")
        if preflight.get("infra_error_detected", False):
            raise RuntimeError(
                "Preflight blocked by infrastructure: no CUDA GPU is available on current host/session. "
                "Please run on a GPU node or fix CUDA visibility, then rerun smoke."
            )
        raise RuntimeError(
            "Preflight failed before entering training loop: "
            + "; ".join(preflight.get("reasons", ["unknown_reason"]))
        )

    grpo = MiniGRPO(
        model, processor, reward_fn,
        reward_backend=reward_backend,
        device=args.device,
        lr=args.lr, group_size=args.group_size,
        max_new_tokens=args.max_new_tokens, kl_coeff=args.kl_coeff,
        lora_applied=lora_applied, use_ref_kl=use_ref_kl,
        initial_params=initial_params,
        temperature=args.temperature, top_p=args.top_p,
        do_sample=do_sample,
        prompt_mode=args.prompt_mode,
        answer_text_source=args.answer_text_source,
        policy_logprob_scope=args.policy_logprob_scope,
        reward_logprob_scope=args.reward_logprob_scope,
        strict_final_answer_scope=args.strict_final_answer_scope,
        invalid_final_answer_penalty=args.invalid_final_answer_penalty,
        reward_condition_on_prefix=args.reward_condition_on_prefix,
        reward_mix_mode=args.reward_mix_mode,
        main_reward_mode=args.main_reward_mode,
        counting_answer_coef=args.counting_answer_coef,
        counting_visual_coef=args.counting_visual_coef,
        existence_answer_coef=args.existence_answer_coef,
        existence_visual_coef=args.existence_visual_coef,
        answer_anchor_coef=args.answer_anchor_coef,
        allow_ground_truth_for_main_rewards=(args.reward_mix_mode == "main_experiment"),
        task_type=(task_type_filter if task_type_filter and (not task_type_allowlist) else ""),
        allow_prompt_side_reward_for_smoke=args.allow_prompt_side_reward_for_smoke,
        prompt_side_constant_group_stop=args.prompt_side_constant_group_stop,
        reward_style_debias=not args.disable_reward_style_debias,
        style_debias_length_coef=args.style_debias_length_coef,
        style_debias_abstention_coef=args.style_debias_abstention_coef,
        style_debias_template_coef=args.style_debias_template_coef,
        style_debias_parse_fail_coef=args.style_debias_parse_fail_coef,
        style_debias_negation_coef=args.style_debias_negation_coef,
        style_debias_evidence_missing_coef=args.style_debias_evidence_missing_coef,
        style_debias_evidence_placeholder_coef=args.style_debias_evidence_placeholder_coef,
        style_debias_max_penalty=args.style_debias_max_penalty,
        hard_template_veto=not args.disable_hard_template_veto,
        hard_template_veto_cap=args.hard_template_veto_cap,
        hard_template_veto_extra_penalty=args.hard_template_veto_extra_penalty,
    )

    log_path = os.path.join(args.output_dir, "training_log.jsonl")
    sample_path = os.path.join(args.output_dir, "sample_outputs.jsonl")
    all_metrics = []
    window_size = 10
    consecutive_zero_gradient = 0

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    log_f = open(log_path, "w")
    sample_f = open(sample_path, "w")
    try:
        for step in range(args.n_steps):
            sample = sample_from_family_buckets(
                train_samples,
                family_buckets=family_buckets,
                rng=random,
                balanced=(not args.disable_family_balanced_sampling),
            )
            try:
                info = grpo.train_step(sample, image_root)
            except Exception as e:
                print(f"[WARN] step {step}: {e}")
                info = {"error": repr(e), "mean_reward": 0.0,
                        "n_in_loss": 0, "all_rewards_identical": True,
                        "n_positive_advantage": 0, "n_negative_advantage": 0}

            info["step"] = step
            all_metrics.append(info)

            if info.get("n_in_loss", 0) == 0:
                consecutive_zero_gradient += 1
            else:
                consecutive_zero_gradient = 0

            # 日志中不写 responses/rewards/_reward_details 等大字段
            log_entry = {k: v for k, v in info.items()
                         if k not in ("responses", "rewards", "delta_logprobs",
                                      "raw_responses", "final_answers", "final_answer_extraction_sources",
                                      "answer_text_used", "policy_prefix_text", "policy_target_text",
                                      "policy_scope_effective", "policy_scope_fallback_reason",
                                      "final_answer_char_span",
                                      "_reward_details", "score_bases",
                                      "pre_kl_rewards", "final_rewards",
                                      "kl_penalties", "answer_anchor_rewards",
                                      "raw_rewards_before_anchor", "raw_rewards_after_hard_veto",
                                      "hard_template_veto_applied", "hard_template_veto_reasons")}
            log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            _flush_fsync(log_f)

            if step % args.save_interval == 0:
                sample_f.write(json.dumps({
                    "step": step,
                    "question": info.get("question"),
                    "image_file": info.get("image_file"),
                    "task_family": info.get("task_family"),
                    "gt_present": info.get("gt_present"),
                    "responses": info.get("responses", []),
                    "raw_responses": info.get("raw_responses", info.get("responses", [])),
                    "final_answers": info.get("final_answers", []),
                    "final_answer_extraction_sources": info.get("final_answer_extraction_sources", []),
                    "final_answer_char_span": info.get("final_answer_char_span", []),
                    "answer_text_used": info.get("answer_text_used", []),
                    "policy_scope_effective": info.get("policy_scope_effective", []),
                    "policy_scope_fallback_reason": info.get("policy_scope_fallback_reason", []),
                    "policy_prefix_text": info.get("policy_prefix_text", []),
                    "policy_target_text": info.get("policy_target_text", []),
                    "rewards": info.get("rewards", []),
                    "advantages": info.get("advantages", []),
                    "parsed_answers": info.get("parsed_answers", []),
                    "mean_kl": info.get("mean_kl", 0),
                    "used_zero_variance_fallback": info.get("used_zero_variance_fallback"),
                    "answer_anchor_rewards": info.get("answer_anchor_rewards", []),
                    "hard_template_veto_applied": info.get("hard_template_veto_applied", []),
                    "hard_template_veto_reasons": info.get("hard_template_veto_reasons", []),
                    # v4 新增
                    "score_bases": info.get("score_bases", []),
                    "delta_logprobs": info.get("delta_logprobs", []),
                    "exact_match_all": info.get("exact_match_all"),
                    "final_answer_identical": info.get("final_answer_identical"),
                    "parse_identical": info.get("parse_identical"),
                    "final_reward_identical": info.get("final_reward_identical"),
                }, ensure_ascii=False) + "\n")
                _flush_fsync(sample_f)

            if step % 10 == 0:
                recent = all_metrics[-window_size:]
                avg_r = np.mean([m.get("mean_final_reward", m.get("mean_reward", 0)) for m in recent])
                avg_kl = np.mean([m.get("mean_kl", 0) for m in recent])
                avg_correct = np.mean([m.get("task_aware_correct_rate_within_group", m.get("answer_score", 0)) for m in recent])
                avg_ans = np.mean([m.get("answer_score", 0) for m in recent])
                avg_style_pen = np.mean([m.get("mean_style_penalty", 0) for m in recent])
                avg_template = np.mean([m.get("template_rate", 0) for m in recent])
                avg_ced = np.mean([m.get("ced_score_mean", m.get("mean_raw_reward", 0)) for m in recent])
                avg_rstd = np.mean([m.get("reward_std_within_group", m.get("std_reward", 0)) for m in recent])
                avg_len = np.mean([m.get("output_length_mean", 0) for m in recent])
                avg_parse_fail = np.mean([m.get("parse_fail_rate", 0) for m in recent])
                n_identical = sum(1 for m in recent if m.get("final_reward_identical", m.get("all_rewards_identical")))
                avg_pos = np.mean([m.get("n_positive_advantage", 0) for m in recent])
                avg_neg = np.mean([m.get("n_negative_advantage", 0) for m in recent])
                n_fallback = sum(1 for m in recent if m.get("used_zero_variance_fallback"))
                n_text_same = sum(1 for m in recent if m.get("exact_match_all"))

                warn = ""
                if n_identical >= len(recent) * 0.8:
                    warn += " ⚠IDENTICAL"
                if consecutive_zero_gradient >= 10:
                    warn += " ⚠ZERO_GRAD"
                if n_fallback > len(recent) * 0.5:
                    warn += " ⚠VAR_FALLBACK"
                if n_text_same > len(recent) * 0.5:
                    warn += " ⚠TEXT_REPEAT"

                print(f"  step {step:4d}/{args.n_steps}  "
                      f"reward={avg_r:.3f}  kl={avg_kl:.4f}  "
                        f"rstd={avg_rstd:.3f}  ans={avg_ans:.2f}  ced={avg_ced:.3f}  "
                        f"task_em={avg_correct:.2f} parse_fail={avg_parse_fail:.2f} len={avg_len:.1f} "
                      f"style_pen={avg_style_pen:.2f} template={avg_template:.2f}  "
                      f"+adv={avg_pos:.1f} -adv={avg_neg:.1f}  "
                      f"identical={n_identical}/{len(recent)}"
                      f"  txt_same={n_text_same}{warn}")
    finally:
        _safe_close_file(sample_f)
        _safe_close_file(log_f)

    # ── debug dump ──
    debug_path = args.debug_dump_path or os.path.join(args.output_dir, "debug_groups.jsonl")
    n_dumped = _write_debug_dump(all_metrics, args.debug_dump_groups, debug_path,
                                 mode=args.debug_dump_mode)
    print(f"\n[INFO] debug dump: {debug_path} ({n_dumped} groups, mode={args.debug_dump_mode})")

    # ── 分层诊断报告 ──
    print("\n" + "=" * 60)
    print("  训练诊断 (v4: 分层)")
    print("=" * 60)

    valid = [m for m in all_metrics if "error" not in m]
    issues = []
    group_diag = _compute_group_diagnostics(all_metrics)

    if valid:
        reward_ts = [m.get("mean_final_reward", m.get("mean_reward", 0)) for m in valid]
        ident_rate = group_diag.get("final_reward_identical_group_rate", 0)
        fallback_rate = group_diag.get("used_zero_variance_fallback_rate", 0)
        avg_pos = np.mean([m.get("n_positive_advantage", 0) for m in valid])
        avg_neg = np.mean([m.get("n_negative_advantage", 0) for m in valid])
        avg_kl = np.mean([m.get("mean_kl", 0) for m in valid])

        print(f"\n  ── 文本层 ──")
        print(f"  exact_match_group_rate:         {group_diag.get('exact_match_group_rate', 0):.1%}")
        print(f"  mean_unique_responses_per_group: {group_diag.get('mean_unique_responses_per_group', 0):.2f}")

        print(f"\n  ── parse 层 ──")
        print(f"  parse_identical_group_rate:       {group_diag.get('parse_identical_group_rate', 0):.1%}")
        print(f"  correctness_identical_group_rate: {group_diag.get('correctness_identical_group_rate', 0):.1%}")

        print(f"\n  ── reward 分层 ──")
        print(f"  raw_delta_identical_group_rate:    {group_diag.get('raw_delta_identical_group_rate', 0):.1%}")
        print(f"  score_base_identical_group_rate:   {group_diag.get('score_base_identical_group_rate', 0):.1%}")
        print(f"  pre_kl_reward_identical_group_rate:{group_diag.get('pre_kl_reward_identical_group_rate', 0):.1%}")
        print(f"  final_reward_identical_group_rate: {group_diag.get('final_reward_identical_group_rate', 0):.1%}")
        print(f"  same_final_answer_diff_raw_reward: {group_diag.get('same_final_answer_diff_raw_reward_group_rate', 0):.1%}")
        print(f"  same_final_answer_diff_final_reward:{group_diag.get('same_final_answer_diff_final_reward_group_rate', 0):.1%}")

        print(f"\n  ── 方差/fallback ──")
        print(f"  zero-var fallback rate:         {fallback_rate:.1%}")
        print(f"  mean_reward_std_within_group:   {group_diag.get('mean_reward_std_within_group', 0):.4f}")
        print(f"  mean_final_reward_std:          {group_diag.get('mean_final_reward_std_within_group', 0):.4f}")
        print(f"  n_groups_all_same_text:         {group_diag.get('n_groups_with_all_same_text', 0)}")
        print(f"  n_same_parse_diff_text:         {group_diag.get('n_groups_with_same_parse_but_diff_text', 0)}")
        print(f"  n_diff_text_same_reward:        {group_diag.get('n_groups_with_diff_text_but_same_final_reward', 0)}")

        print(f"\n  ── style / nuisance ──")
        print(f"  mean task-aware EM:            {group_diag.get('mean_task_aware_correct_rate', 0):.3f}")
        print(f"  mean label-correct rate:       {group_diag.get('mean_label_correct_rate', 0):.3f}")
        print(f"  mean negation rate:            {group_diag.get('mean_negation_rate', 0):.3f}")
        print(f"  mean abstention rate:          {group_diag.get('mean_abstention_rate', 0):.3f}")
        print(f"  mean template rate:            {group_diag.get('mean_template_rate', 0):.3f}")
        print(f"  mean numeric-short rate:       {group_diag.get('mean_numeric_short_rate', 0):.3f}")
        print(f"  mean style penalty:            {group_diag.get('mean_style_penalty', 0):.3f}")
        print(f"  mean answer-anchor reward:     {group_diag.get('mean_answer_anchor_reward', 0):.3f}")
        print(f"  mean hard-template-veto rate:  {group_diag.get('mean_hard_template_veto_rate', 0):.3f}")
        print(f"  mean raw reward/len corr:      {group_diag.get('mean_raw_reward_length_corr', 0):.3f}")
        print(f"  mean debiased reward/len corr: {group_diag.get('mean_debiased_reward_length_corr', 0):.3f}")
        print(f"  mean invalid-final-answer rate:{group_diag.get('mean_invalid_final_answer_rate', 0):.3f}")

        # collapse_layer 分布
        cl_dist = group_diag.get("collapse_layer_distribution", {})
        if cl_dist:
            print(f"\n  ── collapse_layer 分布 ──")
            for layer_name in ["text/raw_response", "final_answer", "parse", "raw_delta", "score_base", "pre_kl", "final", None]:
                cnt = cl_dist.get(layer_name, 0)
                if cnt > 0:
                    lbl = layer_name if layer_name else "no_collapse"
                    print(f"  {lbl:20s} {cnt:4d} ({cnt/len(valid):.0%})")

        print(f"\n  ── 其他 ──")
        print(f"  avg +adv/step: {avg_pos:.2f}  avg -adv/step: {avg_neg:.2f}")
        print(f"  avg KL: {avg_kl:.4f}")

        if ident_rate > 0.5:
            issues.append(f"final_reward identical rate = {ident_rate:.1%}")
        if avg_neg < 0.01:
            issues.append(f"avg negative advantage = {avg_neg:.2f}（模型几乎没被抑制）")
        if not args.disable_ref_kl and abs(avg_kl) < 1e-8:
            issues.append("KL 始终 0，reference model 可能未工作")
        if group_diag.get("same_final_answer_diff_raw_reward_group_rate", 0) > 0.10:
            issues.append(f"same-final-answer diff raw-reward rate = {group_diag.get('same_final_answer_diff_raw_reward_group_rate', 0):.1%}")
        if group_diag.get("mean_invalid_final_answer_rate", 0) > 0.05:
            issues.append(f"invalid final-answer scope rate = {group_diag.get('mean_invalid_final_answer_rate', 0):.1%}")

        if group_diag.get("mean_template_rate", 0) > 0.40 and group_diag.get("mean_task_aware_correct_rate", 0) < 0.10:
            issues.append(f"template rate too high = {group_diag.get('mean_template_rate', 0):.1%} with low task-aware EM")
        if group_diag.get("mean_abstention_rate", 0) > 0.40 and group_diag.get("mean_task_aware_correct_rate", 0) < 0.10:
            issues.append(f"abstention rate too high = {group_diag.get('mean_abstention_rate', 0):.1%} with low task-aware EM")
        raw_len_corr = group_diag.get("mean_raw_reward_length_corr", float("nan"))
        debiased_len_corr = group_diag.get("mean_debiased_reward_length_corr", float("nan"))
        if not np.isnan(raw_len_corr) and abs(raw_len_corr) > 0.30:
            issues.append(f"raw reward/length corr too strong = {raw_len_corr:.2f}")
        if not np.isnan(debiased_len_corr) and abs(debiased_len_corr) > 0.20:
            issues.append(f"debiased reward/length corr still strong = {debiased_len_corr:.2f}")
        if len(reward_ts) >= 20:
            first = np.mean(reward_ts[:len(reward_ts)//2])
            second = np.mean(reward_ts[len(reward_ts)//2:])
            if second > first * 2:
                issues.append(f"reward 飙升 {first:.3f}→{second:.3f}")

    verdict = "可以继续" if not issues else "需要检查"
    diagnosis = {
        "n_steps": args.n_steps,
        "n_errors": len(all_metrics) - len(valid),
        "reward_backend": reward_backend,
        "reward_mode": args.reward_mode,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": do_sample,
        "answer_text_source": args.answer_text_source,
        "reward_logprob_scope": args.reward_logprob_scope,
        "strict_final_answer_scope": args.strict_final_answer_scope,
        "invalid_final_answer_penalty": args.invalid_final_answer_penalty,
        "reward_condition_on_prefix": args.reward_condition_on_prefix,
        "reward_mix_mode": args.reward_mix_mode,
        "main_reward_mode": args.main_reward_mode,
        "counting_answer_coef": args.counting_answer_coef,
        "counting_visual_coef": args.counting_visual_coef,
        "existence_answer_coef": args.existence_answer_coef,
        "existence_visual_coef": args.existence_visual_coef,
        "answer_anchor_coef": args.answer_anchor_coef,
        "hard_template_veto": (not args.disable_hard_template_veto),
        "hard_template_veto_cap": args.hard_template_veto_cap,
        "hard_template_veto_extra_penalty": args.hard_template_veto_extra_penalty,
        "reward_style_debias": (not args.disable_reward_style_debias),
        "style_debias_length_coef": args.style_debias_length_coef,
        "style_debias_abstention_coef": args.style_debias_abstention_coef,
        "style_debias_template_coef": args.style_debias_template_coef,
        "style_debias_parse_fail_coef": args.style_debias_parse_fail_coef,
        "style_debias_negation_coef": args.style_debias_negation_coef,
        "style_debias_max_penalty": args.style_debias_max_penalty,
        "policy_logprob_scope": args.policy_logprob_scope,
        "dataset_name": dataset_name,
        "data_split": args.data_split,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "task_type_filter": train_task,
        "task_type_allowlist": task_type_allowlist,
        "resolved_dataset_config": resolved_dataset_config,
        "dataset_file": dataset_file,
        "image_root": image_root,
        "dataset_summary": dataset_summary,
        "dataset_integrity": dataset_integrity,
        "response_text_source": "raw_generated_text",
        "answer_text_source_effective": args.answer_text_source,
        "reward_contract_ok": (
            (args.answer_text_source or "").strip().lower() == "final_answer"
            and (args.policy_logprob_scope or "").strip().lower() == "final_answer"
            and (args.reward_logprob_scope or "").strip().lower() == "final_answer"
            and bool(args.strict_final_answer_scope)
        ),
        "prompt_side_reward_blocked": (reward_backend == "logodds_ate" and not args.allow_prompt_side_reward_for_smoke),
        "preflight": preflight,
        "used_reward_field": (valid[0].get("used_reward_field") if valid else ""),
        "used_reward_semantics": (valid[0].get("used_reward_semantics") if valid else ""),
        "uses_ground_truth": (bool(valid[0].get("uses_ground_truth")) if valid else False),
        "uses_ground_truth_for_training_step": (
            bool(valid[0].get("uses_ground_truth_for_training_step")) if valid else False
        ),
        "ground_truth_passed_to_reward_compute": (
            bool(valid[0].get("ground_truth_passed_to_reward_compute")) if valid else False
        ),
        "n_groups": len(valid),
        "mean_raw_reward": (float(np.mean([m.get("mean_raw_reward", 0.0) for m in valid])) if valid else None),
        "mean_final_reward": (float(np.mean([m.get("mean_final_reward", 0.0) for m in valid])) if valid else None),
        "mean_answer_score": (float(np.mean([m.get("answer_score_mean", 0.0) for m in valid])) if valid else None),
        "mean_answer_anchor_reward": (float(np.mean([m.get("mean_answer_anchor_reward", 0.0) for m in valid])) if valid else None),
        "mean_invalid_final_answer_rate": (float(np.mean([m.get("invalid_final_answer_rate", 0.0) for m in valid])) if valid else None),
        "same_final_answer_diff_raw_reward_group_rate": group_diag.get("same_final_answer_diff_raw_reward_group_rate", 0.0),
        "same_final_answer_diff_final_reward_group_rate": group_diag.get("same_final_answer_diff_final_reward_group_rate", 0.0),
        "issues": issues,
        "verdict": verdict,
        "lora_applied": lora_applied,
        "ref_kl_enabled": use_ref_kl,
        "elapsed_sec": time.time() - t0,
        # v4 分层诊断
        "group_diagnostics": group_diag,
    }
    diag_path = os.path.join(args.output_dir, "smoke_diagnosis.json")
    with open(diag_path, "w") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)

    print(f"\n  判定: {'✓' if not issues else '⚠'} {verdict}")
    for iss in issues:
        print(f"  ✗ {iss}")
    print(f"\n  诊断: {diag_path}")
    print(f"  debug dump: {debug_path}")
    reward_fn.cleanup()


if __name__ == "__main__":
    main()
