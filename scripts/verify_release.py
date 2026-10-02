#!/usr/bin/env python3
"""Verify release assets; create a checksum inventory only during packaging."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def packaged_files(root: Path) -> list[Path]:
    excluded_top = {"artifacts", "outputs", "logs"}
    excluded_parts = {".git", "__pycache__", ".pytest_cache", ".locks"}
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] in excluded_top or excluded_parts.intersection(relative.parts):
            continue
        if path.name == "release_checksums.sha256" or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--create-manifest", action="store_true", help="Packaging only: record all current release files")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    from qwen_enhance.constants import DEFAULT_LORA_RELATIVE_PATH, DEFAULT_LORA_SHA256
    from verify_model import EXPECTED_SHA256

    manifest = root / "release_checksums.sha256"
    errors: list[str] = []
    symlinks = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_symlink()]
    if symlinks:
        errors.append(f"Release contains symbolic links: {symlinks}")
    digests: dict[str, str] = {}
    if args.create_manifest:
        paths = packaged_files(root)
        for index, path in enumerate(paths, 1):
            relative = path.relative_to(root).as_posix()
            if path.stat().st_size > 100 * 1024 * 1024:
                print(f"Hashing {index}/{len(paths)}: {relative}", flush=True)
            digests[relative] = sha256(path)
    else:
        if not manifest.is_file():
            errors.append(f"Missing checksum inventory: {manifest}")
        else:
            for index, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
                expected, relative = line.split("  ", 1)
                path = root / relative
                if path.resolve().is_relative_to(root) and path.is_file():
                    print(f"Verifying {index}: {relative}", flush=True) if path.stat().st_size > 100 * 1024 * 1024 else None
                    actual = sha256(path)
                    digests[relative] = actual
                    if actual != expected:
                        errors.append(f"Checksum mismatch: {relative}")
                else:
                    errors.append(f"Missing or unsafe release path: {relative}")
    for relative, expected in EXPECTED_SHA256.items():
        key = f"models/hf/Qwen-Image-Edit-2511/{relative}"
        if digests.get(key) != expected:
            errors.append(f"Official model hash mismatch: {key}")
    if digests.get(DEFAULT_LORA_RELATIVE_PATH) != DEFAULT_LORA_SHA256:
        errors.append("Selected epoch-03 adapter hash mismatch")

    from safetensors import safe_open
    adapter_metadata: dict[str, str] = {}
    adapter_path = root / DEFAULT_LORA_RELATIVE_PATH
    if adapter_path.is_file():
        with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
            adapter_metadata = adapter.metadata() or {}
            if adapter_metadata.get("ss_epoch") != "3" or adapter_metadata.get("ss_steps") != "1920":
                errors.append("Adapter header is not epoch 3 / step 1920")
    if args.create_manifest and not errors:
        manifest.write_text("".join(f"{value}  {key}\n" for key, value in sorted(digests.items())), encoding="utf-8")

    report = {
        "ok": not errors,
        "files_verified": len(digests),
        "adapter_epoch": adapter_metadata.get("ss_epoch"),
        "adapter_step": adapter_metadata.get("ss_steps"),
        "symlinks": symlinks,
        "errors": errors,
    }
    report_path = root / "artifacts/reports/release_integrity.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
