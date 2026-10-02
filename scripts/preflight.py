#!/usr/bin/env python3
"""Report whether the portable project is ready for the selected Qwen LoRA profile."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _mem_gib() -> float | None:
    info = Path("/proc/meminfo")
    if not info.is_file():
        return None
    values = {}
    for line in info.read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = value.strip().split()[0]
    return int(values["MemTotal"]) / 1024**2


def _run(args: argparse.Namespace, root: Path) -> int:
    from qwen_enhance.paths import component_paths

    report: dict[str, object] = {
        "project_root": str(root),
        "python": sys.version,
        "ram_gib": _mem_gib(),
        "disk_free_gib": shutil.disk_usage(root).free / 1024**3,
        "blocks_to_swap": args.blocks_to_swap,
        "checks": {},
        "warnings": [],
        "errors": [],
    }
    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
    warnings: list[str] = report["warnings"]  # type: ignore[assignment]
    errors: list[str] = report["errors"]  # type: ignore[assignment]

    checks["dataset_manifests"] = all((root / "dataset" / "manifests" / f"{name}.jsonl").is_file() for name in ("train", "val", "test"))
    checks["musubi_backend"] = (root / "third_party" / "musubi-tuner" / "src" / "musubi_tuner" / "qwen_image_train_network.py").is_file()
    components = component_paths(root)
    model = components["dit"].parents[1]
    required_model_files = {
        **{
            f"dit_{index:05d}": model
            / "transformer"
            / f"diffusion_pytorch_model-{index:05d}-of-00005.safetensors"
            for index in range(1, 6)
        },
        **{
            f"text_encoder_{index:05d}": model
            / "text_encoder"
            / f"model-{index:05d}-of-00004.safetensors"
            for index in range(1, 5)
        },
        "vae": components["vae"],
    }
    checks["model_components"] = {
        name: path.is_file() and path.stat().st_size > 0 for name, path in required_model_files.items()
    }
    if args.require_model and not all(checks["model_components"].values()):  # type: ignore[union-attr]
        errors.append("Official Qwen model shards are incomplete; run scripts/download_models.sh")

    try:
        import torch

        checks["torch"] = torch.__version__
        checks["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            checks["gpu"] = properties.name
            checks["vram_gib"] = round(properties.total_memory / 1024**3, 2)
            checks["bf16_supported"] = torch.cuda.is_bf16_supported()
        else:
            errors.append("PyTorch cannot see a CUDA GPU")
    except Exception as exc:  # import failure is actionable in the report
        errors.append(f"PyTorch check failed: {type(exc).__name__}: {exc}")

    ram_gib = report["ram_gib"]
    if args.blocks_to_swap > 0 and isinstance(ram_gib, float) and ram_gib < 64:
        warnings.append(
            "Block swap is enabled but host RAM is below the backend's 64GiB recommendation; "
            "use the 512 profile and watch RAM during the first epoch."
        )
    # A 24 GB RTX 3090 reports about 23.56 GiB to PyTorch; leave a small unit
    # conversion margin so the intended hardware does not raise a false alarm.
    if isinstance(checks.get("vram_gib"), float) and checks["vram_gib"] < 23:
        warnings.append("VRAM is below the RTX 3090 profile; reduce resolution and/or increase block swap.")

    report["ok"] = not errors
    report_path = root / "artifacts" / "reports" / "preflight.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--blocks-to-swap", type=int, default=int(os.environ.get("QDE_BLOCKS_TO_SWAP", "16")))
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Project root does not exist: {root}")
    sys.path.insert(0, str(root))
    from qwen_enhance.runtime_logging import timestamped_log

    with timestamped_log(root, "preflight"):
        return _run(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
