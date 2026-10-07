#!/usr/bin/env python3
"""Enhance one low-quality image with a trained Qwen DiT LoRA.

This CLI deliberately accepts no user prompt. The image is the only task input;
the quality-preserving instruction is imported internally from constants.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image


def _rounded_dimensions(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid input size: {width}x{height}")
    scale = min(1.0, math.sqrt(max_pixels / (width * height)))
    scaled_w = max(16, int(round(width * scale / 16.0)) * 16)
    scaled_h = max(16, int(round(height * scale / 16.0)) * 16)
    return scaled_w, scaled_h


def _newest_image(directory: Path) -> Path:
    supported = {".png", ".jpg", ".jpeg", ".webp"}
    candidates = [path for path in directory.rglob("*") if path.suffix.lower() in supported]
    if not candidates:
        raise FileNotFoundError(f"Qwen inference produced no image under {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser, root: Path, log) -> int:
    # Musubi's FlowMatch scheduler constructs its timestep spacing from at
    # least two samples.  A one-step request emits invalid numerical values.
    if args.steps < 2:
        parser.error("--steps must be at least 2 for Qwen-Image-Edit inference")

    # The backend obtains its tokenizer and processor through Hugging Face
    # identifiers.  Pin those lookups to the project's downloaded cache so a
    # copied project does not silently fall back to a global cache or network.
    project_hf_cache = root / ".cache" / "huggingface"
    if not project_hf_cache.is_dir():
        parser.error(f"Missing project-local Hugging Face cache: {project_hf_cache}; run scripts/download_models.sh")
    os.environ["HF_HOME"] = str(project_hf_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from qwen_enhance.constants import FIXED_ENHANCEMENT_PROMPT, MODEL_VERSION, NEGATIVE_PROMPT
    from qwen_enhance.paths import component_paths
    from qwen_enhance.runtime_logging import run_logged_command

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    lora_path = Path(args.lora).expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input image does not exist: {input_path}")
    if not lora_path.is_file() or lora_path.stat().st_size == 0:
        parser.error(f"LoRA adapter does not exist or is empty: {lora_path}")
    components = component_paths(root)
    missing = [name for name, path in components.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        parser.error(f"Missing official Qwen component(s): {', '.join(missing)}; run scripts/download_models.sh")
    backend = root / "third_party" / "musubi-tuner" / "src" / "musubi_tuner" / "qwen_image_generate_image.py"
    if not backend.is_file():
        parser.error(f"Missing backend: {backend}; run scripts/setup_environment.sh")

    with Image.open(input_path) as source:
        source = source.convert("RGB")
        original_size = source.size
    generation_width, generation_height = _rounded_dimensions(*original_size, args.max_pixels)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_temporary:
        work_dir = output_path.parent / f".{output_path.stem}_qwen_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup_context = None
    else:
        cleanup_context = tempfile.TemporaryDirectory(prefix="qwen_enhance_", dir=output_path.parent)
        work_dir = Path(cleanup_context.name)

    try:
        command = [
            sys.executable,
            str(backend),
            "--dit",
            str(components["dit"]),
            "--vae",
            str(components["vae"]),
            "--text_encoder",
            str(components["text_encoder"]),
            "--model_version",
            MODEL_VERSION,
            "--control_image_path",
            str(input_path),
            "--prompt",
            FIXED_ENHANCEMENT_PROMPT,
            "--negative_prompt",
            NEGATIVE_PROMPT,
            "--image_size",
            str(generation_height),
            str(generation_width),
            "--infer_steps",
            str(args.steps),
            "--guidance_scale",
            str(args.guidance_scale),
            "--attn_mode",
            # In this Musubi release Qwen's implementation maps the PyTorch
            # scaled-dot-product path to `torch`; the parser accepts `sdpa`
            # but the Qwen attention layout table does not implement that
            # alias.
            "torch",
            "--blocks_to_swap",
            str(args.blocks_to_swap),
            "--fp8",
            "--fp8_scaled",
            "--resize_control_to_official_size",
            "--append_original_name",
            "--lora_weight",
            str(lora_path),
            "--lora_multiplier",
            "1.0",
            "--save_path",
            str(work_dir),
            "--output_type",
            "images",
            "--seed",
            str(args.seed),
        ]
        print("Launching Qwen inference with one image input and the fixed internal prompt.")
        run_logged_command(command, log)
        generated_path = _newest_image(work_dir)
        with Image.open(generated_path) as generated:
            generated = generated.convert("RGB")
            if generated.size != original_size:
                generated = generated.resize(original_size, Image.Resampling.LANCZOS)
            generated.save(output_path)
        metadata = {
            "input": str(input_path),
            "output": str(output_path),
            "lora": str(lora_path),
            "model_version": MODEL_VERSION,
            "prompt": FIXED_ENHANCEMENT_PROMPT,
            "original_size": list(original_size),
            "generation_size": [generation_width, generation_height],
            "steps": args.steps,
            "seed": args.seed,
            "guidance_scale": args.guidance_scale,
            "blocks_to_swap": args.blocks_to_swap,
        }
        output_path.with_suffix(output_path.suffix + ".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Enhanced image saved to: {output_path}")
    finally:
        if cleanup_context is not None:
            cleanup_context.cleanup()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Only user-facing model input: one low-quality image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora", required=True, help="Trained .safetensors LoRA adapter")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--blocks-to-swap", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=1_048_576)
    parser.add_argument("--keep-temporary", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Project root does not exist: {root}")
    sys.path.insert(0, str(root))
    from qwen_enhance.runtime_logging import timestamped_log

    with timestamped_log(root, "inference") as log:
        return _run(args, parser, root, log)


if __name__ == "__main__":
    raise SystemExit(main())
