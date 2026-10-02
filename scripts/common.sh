#!/usr/bin/env bash
# Shared, portable paths for every project entry point.
set -euo pipefail

QDE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDE_CONDA_ROOT="${QDE_CONDA_ROOT:-$HOME/miniforge3}"
QDE_ENV_NAME="${QDE_ENV_NAME:-myenv}"
QDE_PYTHON="${QDE_PYTHON:-$QDE_CONDA_ROOT/envs/$QDE_ENV_NAME/bin/python}"
QDE_ACCELERATE="${QDE_ACCELERATE:-$QDE_CONDA_ROOT/envs/$QDE_ENV_NAME/bin/accelerate}"

# Keep the small tokenizer/processor snapshots alongside the project.  This
# makes train, cache and inference independent of a user's global HF cache.
export HF_HOME="$QDE_PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Workflows whose output is piped through timestamp_tee.py must remain
# line-buffered so the terminal and its corresponding log stay in sync.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONDONTWRITEBYTECODE=1

QDE_MUSUBI_DIR="$QDE_PROJECT_ROOT/third_party/musubi-tuner"
export PYTHONPATH="$QDE_PROJECT_ROOT:$QDE_MUSUBI_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
QDE_MODEL_DIR="$QDE_PROJECT_ROOT/models/hf/Qwen-Image-Edit-2511"
QDE_DIT="$QDE_MODEL_DIR/transformer/diffusion_pytorch_model-00001-of-00005.safetensors"
QDE_VAE="$QDE_MODEL_DIR/vae/diffusion_pytorch_model.safetensors"
QDE_TEXT_ENCODER="$QDE_MODEL_DIR/text_encoder/model-00001-of-00004.safetensors"

QDE_ARTIFACTS_DIR="$QDE_PROJECT_ROOT/artifacts"
QDE_LOG_DIR="${QDE_LOG_DIR:-$QDE_PROJECT_ROOT/logs}"
QDE_DATASET_TRAIN_TOML="$QDE_ARTIFACTS_DIR/dataset/train.toml"
QDE_DATASET_VAL_TOML="$QDE_ARTIFACTS_DIR/dataset/val.toml"
QDE_DATASET_TEST_TOML="$QDE_ARTIFACTS_DIR/dataset/test.toml"
QDE_OUTPUT_DIR="${QDE_OUTPUT_DIR:-$QDE_PROJECT_ROOT/outputs/lora}"

qde_require_file() {
  local qde_path="$1"
  if [[ ! -s "$qde_path" ]]; then
    echo "Required file is missing or empty: $qde_path" >&2
    exit 2
  fi
}

qde_require_python() {
  if [[ ! -x "$QDE_PYTHON" ]]; then
    cat >&2 <<EOF
Python environment not found: $QDE_PYTHON
Run: bash $QDE_PROJECT_ROOT/scripts/setup_environment.sh
EOF
    exit 2
  fi
}

qde_require_backend() {
  if [[ ! -f "$QDE_MUSUBI_DIR/src/musubi_tuner/qwen_image_train_network.py" ]]; then
    echo "Musubi Tuner backend is missing: $QDE_MUSUBI_DIR" >&2
    exit 2
  fi
}

qde_require_model() {
  local qde_shard
  for qde_shard in {00001..00005}; do
    qde_require_file "$QDE_MODEL_DIR/transformer/diffusion_pytorch_model-${qde_shard}-of-00005.safetensors"
  done
  for qde_shard in {00001..00004}; do
    qde_require_file "$QDE_MODEL_DIR/text_encoder/model-${qde_shard}-of-00004.safetensors"
  done
  qde_require_file "$QDE_VAE"
}

# Redirect one top-level workflow to a fixed, timestamped log.  Nested project
# scripts inherit QDE_LOG_ACTIVE and keep writing into their caller's log so a
# training invocation has one complete terminal transcript.
qde_enable_log() {
  local qde_log_name="${1:?A fixed log name is required}"
  if [[ "${QDE_LOG_ACTIVE:-}" == "1" ]]; then
    return 0
  fi
  qde_require_python
  mkdir -p "$QDE_LOG_DIR"
  local qde_log_path="$QDE_LOG_DIR/${qde_log_name}.log"
  export QDE_LOG_ACTIVE=1
  export QDE_LOG_FILE="$qde_log_path"
  exec > >("$QDE_PYTHON" "$QDE_PROJECT_ROOT/scripts/timestamp_tee.py" --log-file "$qde_log_path") 2>&1
  printf 'Started %s. Log file: %s\n' "$qde_log_name" "$qde_log_path"
}
