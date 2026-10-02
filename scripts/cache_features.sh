#!/usr/bin/env bash
# Cache VAE latents and Qwen2.5-VL outputs before LoRA training.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

QDE_SPLIT="${1:-train}"
case "$QDE_SPLIT" in
  train) QDE_DATASET_TOML="$QDE_DATASET_TRAIN_TOML" ;;
  val) QDE_DATASET_TOML="$QDE_DATASET_VAL_TOML" ;;
  test) QDE_DATASET_TOML="$QDE_DATASET_TEST_TOML" ;;
  *) echo "Usage: bash scripts/cache_features.sh [train|val|test]" >&2; exit 2 ;;
esac

qde_enable_log "cache_${QDE_SPLIT}"
qde_require_python
qde_require_backend
qde_require_model
bash "$QDE_PROJECT_ROOT/scripts/ensure_musubi_cache_key_patch.sh"
if [[ ! -f "$QDE_DATASET_TOML" ]]; then
  bash "$QDE_PROJECT_ROOT/scripts/prepare_dataset.sh"
fi

"$QDE_PYTHON" "$QDE_MUSUBI_DIR/src/musubi_tuner/qwen_image_cache_latents.py" \
  --dataset_config "$QDE_DATASET_TOML" \
  --vae "$QDE_VAE" \
  --model_version edit-2511 \
  --batch_size 1 \
  --skip_existing

"$QDE_PYTHON" "$QDE_MUSUBI_DIR/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py" \
  --dataset_config "$QDE_DATASET_TOML" \
  --text_encoder "$QDE_TEXT_ENCODER" \
  --model_version edit-2511 \
  --batch_size 1 \
  --fp8_vl \
  --skip_existing

echo "Feature caches are ready for split: $QDE_SPLIT"
