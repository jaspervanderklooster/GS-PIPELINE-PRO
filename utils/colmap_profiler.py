"""Heuristic preset selection for COLMAP jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:
    Image = None

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _image_files(images_dir: Path) -> list[Path]:
    return [p for p in sorted(images_dir.glob("*")) if p.is_file() and p.suffix.lower() in VALID_EXTS]


def choose_preset(jobdir: Path) -> dict[str, Any]:
    """Choose preset from image count and max side heuristics."""
    images_dir = jobdir / "staging" / "frames"
    files = _image_files(images_dir)
    num_images = len(files)
    max_side = 0
    if Image is not None:
        for p in files:
            try:
                with Image.open(p) as im:
                    max_side = max(max_side, max(im.size))
            except Exception:
                continue
    if num_images >= 4000 or max_side >= 4000:
        preset, reason = "standard_safe", "large dataset"
    elif num_images >= 2000 or max_side >= 3500:
        preset, reason = "hq_safe", "mid-large dataset"
    elif num_images <= 1000 and max_side <= 3000:
        preset, reason = "hq", "small dataset"
    else:
        preset, reason = "standard_safe", "default safe band"
    return {
        "preset": preset,
        "reason": reason,
        "note": "heuristic",
        "num_images": num_images,
        "max_side": max_side,
    }
