"""Portable project-root and model-path discovery."""

from __future__ import annotations

from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Find the root without depending on a machine-specific absolute path."""
    candidate = (start or Path(__file__)).resolve()
    for directory in (candidate, *candidate.parents):
        if (directory / "dataset" / "manifests").is_dir() and (directory / "scripts").is_dir():
            return directory
    raise RuntimeError("Could not locate project root containing dataset/manifests and scripts")


def official_model_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "models" / "hf" / "Qwen-Image-Edit-2511"


def component_paths(root: Path | None = None) -> dict[str, Path]:
    model = official_model_dir(root)
    return {
        "dit": model / "transformer" / "diffusion_pytorch_model-00001-of-00005.safetensors",
        "vae": model / "vae" / "diffusion_pytorch_model.safetensors",
        "text_encoder": model / "text_encoder" / "model-00001-of-00004.safetensors",
    }
