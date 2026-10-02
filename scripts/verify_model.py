#!/usr/bin/env python3
"""Verify the exact Qwen-Image-Edit-2511 weight files used by this project."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


# SHA-256 values published for Qwen/Qwen-Image-Edit-2511 revision
# 6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9.  Keeping these in the project
# makes a copied, offline project independently auditable.
EXPECTED_SHA256 = {
    "text_encoder/model-00001-of-00004.safetensors": "d725335e4ea2399be706469e4b8807716a8fa64bd03468252e9f7acf2415fee4",
    "text_encoder/model-00002-of-00004.safetensors": "b1830db6908dcc76df3a71492acbcf2b8cac130114cf1f3c2d9edae8de8c6de3",
    "text_encoder/model-00003-of-00004.safetensors": "09c1807c6d00d7cab94f7db39d4c02ebb8537225ccde383861ac48db97945aa6",
    "text_encoder/model-00004-of-00004.safetensors": "5dd068336d14d45ffb43cef374d286cc6ba9d8741b028f90a7d040d847961f4a",
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors": "2a0c30c9ba44a5f11c21ca139e37951430bbde814ff4e0b5b1a68b80530e7a1a",
    "transformer/diffusion_pytorch_model-00002-of-00005.safetensors": "54ec249b07b4376e19cf16b764054f03ca03ae2cfbd9939453e2085f4e9bd259",
    "transformer/diffusion_pytorch_model-00003-of-00005.safetensors": "c55157843525653161e8f6af5acc670ba3aceff04284f7cf657199d24d065e16",
    "transformer/diffusion_pytorch_model-00004-of-00005.safetensors": "ffcfb5a4895702635890a67bad183591e0ae515d794bdcb26e217b27a7f6d12d",
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors": "2b2556b736629e10a5a0dfa14606f2057f4f81c2ba53f94103682c7ac42d4940",
    "vae/diffusion_pytorch_model.safetensors": "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344",
}
MODEL_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    model_root = root / "models" / "hf" / "Qwen-Image-Edit-2511"
    results: dict[str, dict[str, object]] = {}
    valid = True
    for relative_path, expected in EXPECTED_SHA256.items():
        path = model_root / relative_path
        if not path.is_file():
            results[relative_path] = {"ok": False, "error": "missing"}
            valid = False
            continue
        actual = digest(path)
        ok = actual == expected
        results[relative_path] = {
            "ok": ok,
            "size_bytes": path.stat().st_size,
            "sha256": actual,
        }
        valid &= ok

    report = {
        "model": "Qwen/Qwen-Image-Edit-2511",
        "revision": MODEL_REVISION,
        "model_root": str(model_root),
        "ok": valid,
        "files": results,
    }
    report_path = root / "artifacts" / "reports" / "model_integrity.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
