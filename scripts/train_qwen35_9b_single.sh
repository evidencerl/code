#!/usr/bin/env bash
# Evidence-RL: Single-GPU Training for Qwen3.5-9B
#
# Usage:
#   bash scripts/train_qwen35_9b_single.sh --gpu 0
#   bash scripts/train_qwen35_9b_single.sh --gpu 0 --steps 2000
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/src:$ROOT"

GPU="${GPU:-0}"
MODEL_DIR="${MODEL_DIR:-Qwen/Qwen3.5-9B}"
DATA_FILE="${DATA_FILE:-$ROOT/data/vg_brutal_pairs.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-$ROOT/data/visual_genome}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/qwen35_9b_answer_ced_single}"
N_STEPS="${N_STEPS:-2000}"
GROUP_SIZE="${GROUP_SIZE:-32}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --model) MODEL_DIR="$2"; shift 2 ;;
    --data-file) DATA_FILE="$2"; shift 2 ;;
    --image-root) IMAGE_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --steps) N_STEPS="$2"; shift 2 ;;
    --group-size) GROUP_SIZE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash scripts/train_qwen35_9b_single.sh [--gpu 0] [--model PATH] [--output-dir PATH] [--steps 2000]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

echo "=========================================================="
echo "Starting Evidence-RL Training on Qwen3.5-9B (Single GPU)"
echo "GPU:                 $GPU"
echo "Base Model:          $MODEL_DIR"
echo "Dataset:             $DATA_FILE"
echo "Image Root:          $IMAGE_ROOT"
echo "Output Directory:    $OUTPUT_DIR"
echo "Steps:               $N_STEPS"
echo "GRPO Group Size:     $GROUP_SIZE"
echo "=========================================================="

mkdir -p "$OUTPUT_DIR"

python "$ROOT/src/train_main_experiment.py" \
  --device "cuda:0" \
  --model_dir "$MODEL_DIR" \
  --dataset_name "vg_brutal" \
  --dataset_file "$DATA_FILE" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --dtype bfloat16 \
  --reward_mode routed_gated_evidence \
  --task_family_filter counting,attribute,spatial \
  --probe_task_family_filter existence \
  --train_budget_mode steps \
  --n_steps "$N_STEPS" \
  --group_size "$GROUP_SIZE" \
  --sampling_mode balanced_no_replacement \
  --replace_mode mean \
  --negative_intervention_k 3 \
  --evidence_eps 0.10 \
  --tau_resp 0.20 \
  --alpha_resp 0.70 \
  --alpha_ans 0.30 \
  --min_reward -1.25 \
  --max_reward 1.00 \
  --lr 1e-5 \
  --kl_coeff 0.01 \
  --n_trainable_layers 4 \
  --lora \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_new_tokens 32 \
  --max_response_tokens 64 \
  --prompt_mode short_evidence_v1 \
  --save_interval 50 \
  --checkpoint_interval 200 \
  --skip_offline_rerank

echo "Training complete! Final checkpoint saved in $OUTPUT_DIR."
echo "To merge LoRA weights with the base model, run:"
echo "  python scripts/merge_checkpoint.py --base $MODEL_DIR --checkpoint $OUTPUT_DIR/checkpoints/step_${N_STEPS}.pt --dest $OUTPUT_DIR/merged_hf"
