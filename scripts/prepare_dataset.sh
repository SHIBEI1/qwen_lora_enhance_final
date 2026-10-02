#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$QDE_PROJECT_ROOT/configs/rtx3090_24gb.env"

qde_enable_log "prepare_dataset"
qde_require_python
"$QDE_PYTHON" "$QDE_PROJECT_ROOT/scripts/validate_dataset.py" --project-root "$QDE_PROJECT_ROOT"
"$QDE_PYTHON" "$QDE_PROJECT_ROOT/scripts/prepare_musubi_metadata.py" \
  --project-root "$QDE_PROJECT_ROOT" \
  --resolution "$QDE_TRAIN_RESOLUTION" \
  --control-resolution "$QDE_CONTROL_RESOLUTION"
