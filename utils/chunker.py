"""Chunk helpers for large COLMAP jobs."""
from __future__ import annotations

import shutil
from pathlib import Path


def chunk_images(input_dir: Path, out_dir: Path, chunk_size: int = 500, overlap: int = 50) -> list[Path]:
    files = sorted([p for p in input_dir.glob("*") if p.is_file()])
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    step = max(1, chunk_size - overlap)
    idx = 0
    chunk_id = 0
    while idx < len(files):
        chunk_id += 1
        chunk_dir = out_dir / f"chunk_{chunk_id:03d}" / "images"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for f in files[idx: idx + chunk_size]:
            target = chunk_dir / f.name
            if not target.exists():
                shutil.copy2(str(f), str(target))
        chunks.append(chunk_dir.parent)
        idx += step
    return chunks


def merge_colmap_models(models_dirs: list[Path], output_dir: Path) -> Path:
    """Minimal merge: pick first available fused.ply."""
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = output_dir / "fused.ply"
    for m in models_dirs:
        src = m / "workspace" / "fused.ply"
        if src.exists() and src.stat().st_size > 0:
            shutil.copy2(str(src), str(merged))
            return merged
    raise RuntimeError("No chunk fused.ply found to merge")
