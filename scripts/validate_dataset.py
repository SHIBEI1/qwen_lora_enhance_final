#!/usr/bin/env python3
"""Validate portable paired-image manifests before caching or training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


def _project_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _safe_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Manifest path must be relative to dataset/: {relative_path!r}")
    resolved = (dataset_root / candidate).resolve()
    if dataset_root not in (resolved, *resolved.parents):
        raise ValueError(f"Manifest path escapes dataset/: {relative_path!r}")
    return resolved


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            records.append(record)
    return records


def _image_info(path: Path) -> tuple[str, int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.mode, image.width, image.height
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable image {path}: {exc}") from exc


def validate(root: Path, splits: list[str], enforce_expected_counts: bool) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    from qwen_enhance.constants import EXPECTED_SPLIT_COUNTS, EXPECTED_TASK_COUNTS

    dataset_root = root / "dataset"
    manifests_root = dataset_root / "manifests"
    errors: list[str] = []
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    image_owners: dict[Path, list[str]] = defaultdict(list)
    report: dict[str, Any] = {
        "project_root": str(root),
        "splits": {},
        "errors": errors,
        "scope": "Manifest paths, readable RGB pairs, dimensions and counts; not pixel-content independence",
        "known_limitations": [
            "Frozen low-light data has known cross-split content duplicates; see dataset/DATASET_CARD.md"
        ],
    }

    for split in splits:
        manifest_path = manifests_root / f"{split}.jsonl"
        if not manifest_path.is_file():
            errors.append(f"Missing manifest: {manifest_path}")
            continue
        try:
            records = _read_manifest(manifest_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        records_by_split[split] = records
        ids: set[str] = set()
        task_counts: Counter[str] = Counter()
        split_info = {"count": len(records), "task_counts": task_counts, "images_checked": 0}
        report["splits"][split] = split_info

        for index, record in enumerate(records, start=1):
            label = f"{split}[{index}]"
            required = {"id", "split", "task", "input", "target", "fixed_prompt"}
            missing = sorted(required.difference(record))
            if missing:
                errors.append(f"{label} missing required fields: {missing}")
                continue
            if record["split"] != split:
                errors.append(f"{label} has split={record['split']!r}, expected {split!r}")
            item_id = str(record["id"])
            if item_id in ids:
                errors.append(f"Duplicate id in {split}: {item_id}")
            ids.add(item_id)
            task_counts[str(record["task"])] += 1
            try:
                input_path = _safe_dataset_path(dataset_root, str(record["input"]))
                target_path = _safe_dataset_path(dataset_root, str(record["target"]))
                if input_path == target_path:
                    errors.append(f"{label} input and target point to the same file")
                for role, image_path in (("input", input_path), ("target", target_path)):
                    if not image_path.is_file():
                        errors.append(f"{label} missing {role} image: {image_path}")
                        continue
                    mode, width, height = _image_info(image_path)
                    if mode != "RGB":
                        errors.append(f"{label} {role} is {mode}, expected RGB: {image_path}")
                    if width <= 0 or height <= 0:
                        errors.append(f"{label} {role} has invalid size {width}x{height}")
                    image_owners[image_path].append(f"{split}:{item_id}:{role}")
                    split_info["images_checked"] += 1
                if input_path.is_file() and target_path.is_file():
                    _, input_w, input_h = _image_info(input_path)
                    _, target_w, target_h = _image_info(target_path)
                    if (input_w, input_h) != (target_w, target_h):
                        errors.append(
                            f"{label} input/target size mismatch: {input_w}x{input_h} vs {target_w}x{target_h}"
                        )
            except ValueError as exc:
                errors.append(f"{label}: {exc}")

        split_info["task_counts"] = dict(sorted(task_counts.items()))
        if enforce_expected_counts and split in EXPECTED_SPLIT_COUNTS:
            if len(records) != EXPECTED_SPLIT_COUNTS[split]:
                errors.append(
                    f"{split} count={len(records)}, expected {EXPECTED_SPLIT_COUNTS[split]}"
                )
            if dict(task_counts) != EXPECTED_TASK_COUNTS[split]:
                errors.append(
                    f"{split} task counts={dict(task_counts)}, expected {EXPECTED_TASK_COUNTS[split]}"
                )

    cross_split_collisions: dict[str, list[str]] = {}
    for image_path, owners in image_owners.items():
        owner_splits = {owner.split(":", 1)[0] for owner in owners}
        if len(owner_splits) > 1:
            cross_split_collisions[str(image_path)] = owners
    if cross_split_collisions:
        errors.append(f"Found {len(cross_split_collisions)} image files shared by different splits")
    report["cross_split_image_collisions"] = cross_split_collisions
    report["ok"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--report", default=None, help="Defaults to artifacts/reports/dataset_validation.json")
    args = parser.parse_args()

    root = _project_root(args.project_root)
    report = validate(root, args.splits, enforce_expected_counts=not args.allow_count_mismatch)
    report_path = Path(args.report).expanduser().resolve() if args.report else root / "artifacts" / "reports" / "dataset_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        print(f"Dataset validation failed; details written to {report_path}", file=sys.stderr)
        return 1
    print(f"Dataset validation passed; report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
