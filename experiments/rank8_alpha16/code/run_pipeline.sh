#!/usr/bin/env bash
set -euo pipefail
QDE_EXP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QDE_PYTHON=/home/liu/miniforge3/envs/myenv/bin/python
QDE_PROJECT_ROOT=/mnt/Disk1/kl/qwen_diffusion_enhance
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$QDE_EXP_ROOT/logs"
"$QDE_PYTHON" "$QDE_EXP_ROOT/code/rank_ablation.py" run 2>&1 |
  "$QDE_PYTHON" "$QDE_PROJECT_ROOT/scripts/timestamp_tee.py" --log-file "$QDE_EXP_ROOT/logs/pipeline.log"
