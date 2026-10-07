#!/usr/bin/env bash
# Start the single-model Qwen-Image-Edit-2511 DiT LoRA experiment.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$QDE_PROJECT_ROOT/configs/rtx3090_24gb.env"

qde_enable_log "train"
qde_require_python
qde_require_backend
qde_require_model
bash "$QDE_PROJECT_ROOT/scripts/prepare_dataset.sh"
bash "$QDE_PROJECT_ROOT/scripts/cache_features.sh" train
"$QDE_PYTHON" "$QDE_PROJECT_ROOT/scripts/preflight.py" \
  --project-root "$QDE_PROJECT_ROOT" \
  --require-model \
  --blocks-to-swap "$QDE_BLOCKS_TO_SWAP"

mkdir -p "$QDE_OUTPUT_DIR"
"$QDE_ACCELERATE" launch \
  --config_file "$QDE_PROJECT_ROOT/configs/accelerate_single_gpu.yaml" \
  --num_cpu_threads_per_process 1 \
  --mixed_precision bf16 \
  "$QDE_MUSUBI_DIR/src/musubi_tuner/qwen_image_train_network.py" \
  --dit "$QDE_DIT" \
  --vae "$QDE_VAE" \
  --text_encoder "$QDE_TEXT_ENCODER" \
  --dataset_config "$QDE_DATASET_TRAIN_TOML" \
  --model_version edit-2511 \
  --sdpa \
  --mixed_precision bf16 \
  --timestep_sampling shift \
  --weighting_scheme none \
  --discrete_flow_shift 2.2 \
  --optimizer_type adamw8bit \
  --learning_rate "$QDE_LEARNING_RATE" \
  --gradient_checkpointing \
  --blocks_to_swap "$QDE_BLOCKS_TO_SWAP" \
  --fp8_base \
  --fp8_scaled \
  --max_data_loader_n_workers "$QDE_WORKERS" \
  --persistent_data_loader_workers \
  --gradient_accumulation_steps 1 \
  --network_module networks.lora_qwen_image \
  --network_dim "$QDE_LORA_DIM" \
  --network_alpha "$QDE_LORA_ALPHA" \
  --network_dropout "$QDE_LORA_DROPOUT" \
  --max_train_epochs "$QDE_EPOCHS" \
  --save_every_n_epochs "$QDE_SAVE_EVERY_EPOCHS" \
  --seed "$QDE_SEED" \
  --output_dir "$QDE_OUTPUT_DIR" \
  --output_name qwen_image_enhance_lora

echo "Training finished. Adapter weights are under: $QDE_OUTPUT_DIR"
