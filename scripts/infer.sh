#!/usr/bin/env bash
# Single-image inference using the selected epoch-03 adapter by default.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$QDE_PROJECT_ROOT/configs/rtx3090_24gb.env"

qde_enable_log "inference"
qde_require_backend
qde_require_model
"$QDE_PYTHON" "$QDE_PROJECT_ROOT/scripts/enhance.py" \
  --project-root "$QDE_PROJECT_ROOT" \
  --steps "$QDE_INFERENCE_STEPS" \
  --guidance-scale "$QDE_GUIDANCE_SCALE" \
  --blocks-to-swap "$QDE_BLOCKS_TO_SWAP" \
  --seed "$QDE_SEED" \
  "$@"
