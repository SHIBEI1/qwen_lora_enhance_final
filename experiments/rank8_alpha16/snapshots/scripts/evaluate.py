#!/usr/bin/env python3
"""Generate and score paired-image enhancement results by split and task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rgb_array(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image)


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser, root: Path, log) -> int:
    if args.steps < 2:
        parser.error("--steps must be at least 2 for Qwen-Image-Edit inference")
    from qwen_enhance.runtime_logging import run_logged_command

    dataset_root = root / "dataset"
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.is_file():
            parser.error(f"Manifest does not exist: {manifest_path}")
        records = _load_records(manifest_path)
        dataset_label = args.label or manifest_path.stem.removesuffix("_source")
    else:
        manifest_path = dataset_root / "manifests" / f"{args.split}.jsonl"
        records = _load_records(manifest_path)
        dataset_label = args.label or args.split
    if args.limit is not None:
        records = records[: args.limit]
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "outputs" / "evaluation" / dataset_label
    output_dir.mkdir(parents=True, exist_ok=True)
    enhance_script = root / "scripts" / "enhance.py"
    lora = Path(args.lora).expanduser().resolve()
    if not lora.is_file():
        parser.error(f"LoRA file does not exist: {lora}")

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        item_id = str(record["id"])
        task = str(record["task"])
        prediction = output_dir / task / f"{item_id}.png"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        if not (args.skip_existing and prediction.is_file()):
            command = [
                sys.executable,
                str(enhance_script),
                "--project-root",
                str(root),
                "--input",
                str(dataset_root / record["input"]),
                "--output",
                str(prediction),
                "--lora",
                str(lora),
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
            ]
            print(f"[{index}/{len(records)}] {task}: {item_id}")
            run_logged_command(command, log)
        target_path = dataset_root / record["target"]
        target = _rgb_array(target_path)
        prediction_array = _rgb_array(prediction, size=(target.shape[1], target.shape[0]))
        rows.append(
            {
                "id": item_id,
                "task": task,
                "input": record["input"],
                "target": record["target"],
                "prediction": str(prediction),
                "psnr": float(peak_signal_noise_ratio(target, prediction_array, data_range=255)),
                "ssim": float(structural_similarity(target, prediction_array, channel_axis=-1, data_range=255)),
            }
        )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)
    summary = {
        "dataset_label": dataset_label,
        "manifest": str(manifest_path),
        "count": len(rows),
        "lora": str(lora),
        "steps": args.steps,
        "seed": args.seed,
        "overall": {
            "psnr": mean(float(row["psnr"]) for row in rows) if rows else None,
            "ssim": mean(float(row["ssim"]) for row in rows) if rows else None,
        },
        "by_task": {
            task: {
                "count": len(task_rows),
                "psnr": mean(float(row["psnr"]) for row in task_rows),
                "ssim": mean(float(row["ssim"]) for row in task_rows),
            }
            for task, task_rows in sorted(grouped.items())
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_image_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id", "task", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Curated split when --manifest is omitted")
    parser.add_argument("--manifest", default=None, help="Optional JSONL records such as artifacts/pilot/val_pilot_source.jsonl")
    parser.add_argument("--label", default=None, help="Optional output/report label; defaults to the selected split or manifest stem")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-name", default="evaluation", help="Fixed log basename without .log")
    parser.add_argument("--log-dir", default=None, help="Optional directory for this run's fixed log")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Project root does not exist: {root}")
    sys.path.insert(0, str(root))
    from qwen_enhance.runtime_logging import timestamped_log

    prior_log_dir = os.environ.get("QDE_LOG_DIR")
    if args.log_dir:
        os.environ["QDE_LOG_DIR"] = str(Path(args.log_dir).expanduser().resolve())
    try:
        with timestamped_log(root, args.log_name) as log:
            return _run(args, parser, root, log)
    finally:
        if prior_log_dir is None:
            os.environ.pop("QDE_LOG_DIR", None)
        else:
            os.environ["QDE_LOG_DIR"] = prior_log_dir


if __name__ == "__main__":
    raise SystemExit(main())
