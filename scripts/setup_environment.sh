#!/usr/bin/env bash
# Idempotent environment bootstrap. It intentionally installs only below $HOME.
set -euo pipefail

QDE_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDE_CONDA_ROOT="${QDE_CONDA_ROOT:-$HOME/miniforge3}"
QDE_ENV_NAME="${QDE_ENV_NAME:-myenv}"
QDE_ENV_PATH="$QDE_CONDA_ROOT/envs/$QDE_ENV_NAME"
QDE_INSTALLER="$HOME/.cache/Miniforge3-Linux-x86_64.sh"
QDE_BACKEND_DIR="$QDE_PROJECT_ROOT/third_party/musubi-tuner"
QDE_BACKEND_TAG="v0.2.15"

if [[ ! -x "$QDE_CONDA_ROOT/bin/conda" ]]; then
  mkdir -p "$HOME/.cache"
  curl --fail --location --retry 5 --retry-delay 3 \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    --output "$QDE_INSTALLER"
  bash "$QDE_INSTALLER" -b -p "$QDE_CONDA_ROOT"
fi

if [[ ! -x "$QDE_ENV_PATH/bin/python" ]]; then
  "$QDE_CONDA_ROOT/bin/mamba" create --yes --name "$QDE_ENV_NAME" --override-channels --channel conda-forge \
    python=3.11 pip setuptools wheel
fi

if [[ -e "$QDE_BACKEND_DIR" && ! -d "$QDE_BACKEND_DIR/.git" ]]; then
  echo "Refusing to overwrite non-git backend path: $QDE_BACKEND_DIR" >&2
  exit 2
fi
mkdir -p "$(dirname "$QDE_BACKEND_DIR")"
if [[ ! -d "$QDE_BACKEND_DIR/.git" ]]; then
  git clone --depth 1 --branch "$QDE_BACKEND_TAG" https://github.com/kohya-ss/musubi-tuner.git "$QDE_BACKEND_DIR"
fi
bash "$QDE_PROJECT_ROOT/scripts/ensure_musubi_cache_key_patch.sh" "$QDE_BACKEND_DIR"

QDE_PYTHON="$QDE_ENV_PATH/bin/python"
"$QDE_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1
"$QDE_PYTHON" -m pip install -r "$QDE_PROJECT_ROOT/environment/requirements-lock.txt"
"$QDE_PYTHON" -m pip install --no-deps --editable "$QDE_BACKEND_DIR"

"$QDE_PYTHON" - <<'PY'
import torch
print(f"python environment: {torch.__file__}")
print(f"torch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
PY

echo "Environment ready. Activate manually with: source $QDE_CONDA_ROOT/bin/activate $QDE_ENV_NAME"
