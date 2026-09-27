#!/usr/bin/env bash
# Create the Evidence-RL conda prefix (self-contained for open-source).
# Online machine (platform 2 / laptop): builds the env and a wheelhouse.
# Offline machine (platform 1): install from that wheelhouse.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${EVIDENCE_RL_CONDA_PREFIX:-/input0/hhj/envs/evidence_rl}"
WHEELHOUSE="${EVIDENCE_RL_WHEELHOUSE:-/input0/hhj/wheelhouse/evidence_rl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
MODE="${1:-online}"  # online | offline

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$PREFIX")" "$WHEELHOUSE"

if [[ ! -x "$PREFIX/bin/python" ]]; then
  conda create --yes --prefix "$PREFIX" "python=${PYTHON_VERSION}" pip git cmake ninja packaging
fi

PY="$PREFIX/bin/python"
PIP="$PREFIX/bin/pip"

"$PY" -m pip install --upgrade pip setuptools wheel

install_torch() {
  # Pin to the cluster-validated CUDA 12.6 stack so flash-attn wheels match
  # and a laptop clone can pip-install from the wheelhouse without solving.
  "$PIP" install \
    "torch==${TORCH_VER:-2.8.0}" \
    "torchvision==${VISION_VER:-0.23.0}" \
    "torchaudio==${AUDIO_VER:-2.8.0}" \
    --index-url "$TORCH_INDEX"
}

install_flash_attn() {
  # Prefer a prebuilt wheel in the wheelhouse; otherwise build.
  if ls "$WHEELHOUSE"/flash_attn-*.whl >/dev/null 2>&1; then
    "$PIP" install --no-deps "$WHEELHOUSE"/flash_attn-*.whl
  else
    "$PIP" install flash-attn --no-build-isolation || \
      echo "WARN: flash-attn install failed; model_loader will fall back to sdpa"
  fi
}

install_transformer_engine() {
  if ls "$WHEELHOUSE"/transformer_engine*.whl >/dev/null 2>&1; then
    "$PIP" install --no-deps "$WHEELHOUSE"/transformer_engine*.whl || true
  else
    "$PIP" install transformer-engine[pytorch] || \
      echo "WARN: transformer-engine not installed (optional)"
  fi
}

case "$MODE" in
  online)
    install_torch
    "$PIP" download -d "$WHEELHOUSE" \
      "torch==${TORCH_VER:-2.8.0}" \
      "torchvision==${VISION_VER:-0.23.0}" \
      "torchaudio==${AUDIO_VER:-2.8.0}" \
      --index-url "$TORCH_INDEX"
    "$PIP" install -r "$ROOT/requirements.txt"
    "$PIP" download -d "$WHEELHOUSE" -r "$ROOT/requirements.txt"
    install_flash_attn
    install_transformer_engine
    bash "$ROOT/scripts/vendor_nvidia_infra.sh" || true
    ;;
  offline)
    "$PIP" install --no-index --find-links "$WHEELHOUSE" torch torchvision torchaudio
    "$PIP" install --no-index --find-links "$WHEELHOUSE" -r "$ROOT/requirements.txt"
    install_flash_attn
    install_transformer_engine
    ;;
  *)
    echo "Usage: $0 [online|offline]" >&2
    exit 2
    ;;
esac

"$PY" - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("torch missing", e)
try:
    import flash_attn
    print("flash_attn", getattr(flash_attn, "__version__", "?"))
except Exception as e:
    print("flash_attn missing", e)
try:
    import transformer_engine
    print("transformer_engine", getattr(transformer_engine, "__version__", "?"))
except Exception as e:
    print("transformer_engine missing (optional)", e)
PY

echo "[ok] conda prefix=$PREFIX"
echo "[ok] wheelhouse=$WHEELHOUSE"
echo "Use: $PREFIX/bin/python"
