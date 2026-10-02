#!/usr/bin/env bash
# Download one official Qwen snapshot. It is used for both LoRA training and inference.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -gt 0 ]]; then
  echo "Usage: bash scripts/download_models.sh" >&2
  exit 2
fi

qde_enable_log "model_download"
qde_require_python
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
mkdir -p "$QDE_PROJECT_ROOT/models/hf" "$QDE_PROJECT_ROOT/.cache/huggingface"

QDE_AVAILABLE_KB="$(df -Pk "$QDE_PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
QDE_REQUIRED_KB=$((100 * 1024 * 1024))
if (( QDE_AVAILABLE_KB < QDE_REQUIRED_KB )); then
  echo "Need at least 100GiB free before downloading the Qwen snapshot; available: ${QDE_AVAILABLE_KB}KiB" >&2
  exit 2
fi

export HF_HOME="$QDE_PROJECT_ROOT/.cache/huggingface"
export HF_HUB_DOWNLOAD_TIMEOUT="180"
QDE_DOWNLOAD_WORKERS="${QDE_DOWNLOAD_WORKERS:-4}"
QDE_MODEL_REVISION="6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"

# Prefer Hugging Face itself, but make the download usable on training hosts
# whose outbound route only permits the widely used HF mirror.  Callers may
# always override this selection with HF_ENDPOINT before invoking the script.
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  if curl -fsS --connect-timeout 8 --max-time 15 \
    "https://huggingface.co/api/models/Qwen/Qwen-Image-Edit-2511" >/dev/null; then
    export HF_ENDPOINT="https://huggingface.co"
  elif curl -fsS --connect-timeout 8 --max-time 15 \
    "https://hf-mirror.com/api/models/Qwen/Qwen-Image-Edit-2511" >/dev/null; then
    export HF_ENDPOINT="https://hf-mirror.com"
  else
    echo "Neither Hugging Face nor the configured mirror is reachable from this host." >&2
    exit 1
  fi
fi
echo "Downloading Qwen weights through: $HF_ENDPOINT"

"$QDE_CONDA_ROOT/envs/$QDE_ENV_NAME/bin/hf" download Qwen/Qwen-Image-Edit-2511 \
  --revision "$QDE_MODEL_REVISION" \
  --local-dir "$QDE_MODEL_DIR" \
  --max-workers "$QDE_DOWNLOAD_WORKERS"

# The pinned LoRA backend obtains its compatible tokenizer and image processor
# from these original Qwen repository IDs. Cache their small non-weight assets so
# the subsequent cache/train/infer stages do not need an incidental network fetch.
"$QDE_CONDA_ROOT/envs/$QDE_ENV_NAME/bin/hf" download Qwen/Qwen-Image \
  --revision "75e0b4be04f60ec59a75f475837eced720f823b6" \
  --include "tokenizer/*" \
  --max-workers 2
"$QDE_CONDA_ROOT/envs/$QDE_ENV_NAME/bin/hf" download Qwen/Qwen-Image-Edit \
  --revision "ac7f9318f633fc4b5778c59367c8128225f1e3de" \
  --include "processor/*" \
  --max-workers 2

qde_require_model
"$QDE_PYTHON" "$QDE_PROJECT_ROOT/scripts/verify_model.py" --project-root "$QDE_PROJECT_ROOT"
QDE_MODEL_DIR="$QDE_MODEL_DIR" "$QDE_PYTHON" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["QDE_MODEL_DIR"])
files = sorted(path for path in root.rglob("*.safetensors") if path.is_file())
size_gib = sum(path.stat().st_size for path in files) / 1024**3
print(f"Verified {len(files)} safetensors files ({size_gib:.2f} GiB) in {root}")
PY

echo "Official Qwen-Image-Edit-2511 snapshot is ready: $QDE_MODEL_DIR"
