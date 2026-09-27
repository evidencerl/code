#!/usr/bin/env bash
# Vendor NVIDIA training infra next to Evidence-RL so a clone can reproduce
# without hunting URLs. Does not rewrite the CED/GRPO trainer to NeMo-RL.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="${EVIDENCE_RL_VENDOR:-$ROOT/third_party}"
mkdir -p "$VENDOR"

clone_or_update() {
  local owner_repo="$1"  # e.g. NVIDIA/TransformerEngine
  local dest="$2"
  local pin="${3:-}"
  local mirrors=(
    "https://github.com/${owner_repo}.git"
    "https://gitclone.com/github.com/${owner_repo}.git"
    "https://ghproxy.net/https://github.com/${owner_repo}.git"
    "https://mirror.ghproxy.com/https://github.com/${owner_repo}.git"
  )
  if [[ -d "$dest/.git" ]]; then
    git -C "$dest" fetch --depth 1 origin || true
    return 0
  fi
  local url
  for url in "${mirrors[@]}"; do
    echo "[vendor] clone $url -> $dest"
    if git clone --depth 1 "$url" "$dest"; then
      if [[ -n "$pin" ]]; then
        git -C "$dest" fetch --depth 1 origin "$pin" || true
        git -C "$dest" checkout --detach "$pin" || true
      fi
      return 0
    fi
    rm -rf "$dest"
  done
  echo "WARN: could not clone $owner_repo; pip wheels in create_conda_env.sh still install flash-attn / transformer-engine" >&2
  return 0
}

clone_or_update NVIDIA/TransformerEngine "$VENDOR/TransformerEngine"
clone_or_update NVIDIA-NeMo/RL "$VENDOR/NeMo-RL"
clone_or_update Dao-AILab/flash-attention "$VENDOR/flash-attention"

cat > "$VENDOR/README.md" <<'EOF'
# Vendored NVIDIA infra

| Tree | Role in Evidence-RL |
|------|---------------------|
| TransformerEngine | Optional fused kernels / FP8. Not required for the paper GRPO loop. |
| flash-attention | Preferred attention. `model_loader` uses `flash_attention_2` with SDPA fallback. |
| NeMo-RL | Reference GRPO/RL infrastructure. CED reward (visual-token replace + routed gate) stays in `hhj-train`. Do not swap the paper trainer for NeMo-RL unless a reproduction explicitly re-implements those hooks. |

Install via `scripts/create_conda_env.sh`. Training entry remains
`scripts/launch_paper_train.sh` with `PYTHON=/path/to/envs/evidence_rl/bin/python`.
EOF

echo "[ok] vendored NVIDIA trees under $VENDOR"
ls -d "$VENDOR"/*
