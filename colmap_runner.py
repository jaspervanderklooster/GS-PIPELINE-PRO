from __future__ import annotations

"""COLMAP runner tuned for GS_PIPELINE."""

import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

COLMAP_PRESET_ARGS = {
    "standard": {
        "SiftExtraction.peak_threshold": 0.01,
        "SiftExtraction.edge_threshold": 10,
        "PatchMatchStereo.geom_consistency": True,
        "PatchMatchStereo.num_iterations": 4,
        "PatchMatchStereo.window_radius": 3,
        "PatchMatchStereo.num_samples": 12,
        "PatchMatchStereo.filter_min_ncc": 0.35,
        "StereoFusion.max_image_size": 2800,
        "StereoFusion.min_num_pixels": 2,
        "StereoFusion.max_reproj_error": 2.0,
        "StereoFusion.geom_consistency": True,
        "SiftExtraction.max_num_features": 8192,
    },
    "standard_safe": {
        "SiftExtraction.peak_threshold": 0.02,
        "SiftExtraction.edge_threshold": 12,
        "PatchMatchStereo.geom_consistency": True,
        "PatchMatchStereo.num_iterations": 2,
        "PatchMatchStereo.window_radius": 2,
        "PatchMatchStereo.num_samples": 8,
        "PatchMatchStereo.filter_min_ncc": 0.4,
        "StereoFusion.max_image_size": 2500,
        "StereoFusion.min_num_pixels": 2,
        "StereoFusion.max_reproj_error": 2.5,
        "StereoFusion.geom_consistency": True,
        "SiftExtraction.max_num_features": 4096,
    },
    "hq_safe": {
        "SiftExtraction.peak_threshold": 0.009,
        "SiftExtraction.edge_threshold": 9,
        "PatchMatchStereo.geom_consistency": True,
        "PatchMatchStereo.num_iterations": 4,
        "PatchMatchStereo.window_radius": 3,
        "PatchMatchStereo.num_samples": 12,
        "PatchMatchStereo.filter_min_ncc": 0.35,
        "StereoFusion.max_image_size": 3000,
        "StereoFusion.min_num_pixels": 1,
        "StereoFusion.max_reproj_error": 2.0,
        "StereoFusion.geom_consistency": True,
        "SiftExtraction.max_num_features": 8192,
    },
    "hq": {
        "SiftExtraction.peak_threshold": 0.0075,
        "SiftExtraction.edge_threshold": 8,
        "PatchMatchStereo.geom_consistency": True,
        "PatchMatchStereo.num_iterations": 8,
        "PatchMatchStereo.window_radius": 5,
        "PatchMatchStereo.num_samples": 24,
        "PatchMatchStereo.filter_min_ncc": 0.30,
        "StereoFusion.max_image_size": 3500,
        "StereoFusion.min_num_pixels": 1,
        "StereoFusion.max_reproj_error": 2.5,
        "StereoFusion.geom_consistency": True,
        "SiftExtraction.max_num_features": 16000,
    },
}


def _env_colmap_bin() -> Optional[str]:
    return os.environ.get("COLMAP_BIN")


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("RUN:", " ".join(shlex.quote(x) for x in cmd))
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, shell=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "Command failed "
            f"(rc={res.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{(res.stdout or '').strip()}\n"
            f"stderr:\n{(res.stderr or '').strip()}"
        )


def run_colmap(
    image_dir: Path,
    workspace: Path,
    preset: str = "standard",
    input_type: str = "photoset",
    colmap_bin: Optional[str] = None,
    force_cpu: bool = False,
) -> Path:
    """Execute COLMAP pipeline and return fused pointcloud (PLY) path."""
    if preset not in COLMAP_PRESET_ARGS:
        raise ValueError(f"Unknown preset: {preset}")

    colmap_bin = colmap_bin or _env_colmap_bin() or "colmap"
    workspace = Path(workspace)
    image_dir = Path(image_dir)

    if not image_dir.exists() or not any(image_dir.glob("*")):
        raise RuntimeError(f"Image directory empty or missing: {image_dir}")

    preset_args = COLMAP_PRESET_ARGS[preset]

    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    dense_dir = workspace / "dense"
    fused_ply = workspace / "fused.ply"

    if db_path.exists():
        print("Existing database found; removing to ensure reproducible run.")
        db_path.unlink()

    feat_cmd = [colmap_bin, "feature_extractor", "--database_path", str(db_path), "--image_path", str(image_dir)]
    for k, v in preset_args.items():
        if k.startswith("SiftExtraction."):
            feat_cmd.extend([f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)])
    if force_cpu:
        # Verify flag names with local COLMAP version; adjust if needed.
        feat_cmd.extend(["--SiftExtraction.use_gpu", "false"])
    _run(feat_cmd, cwd=workspace)

    matcher = "sequential_matcher" if input_type == "video" else "exhaustive_matcher"
    match_cmd = [colmap_bin, matcher, "--database_path", str(db_path)]
    _run(match_cmd, cwd=workspace)

    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
    ]
    _run(mapper_cmd, cwd=workspace)

    model_dir = None
    for candidate in sorted(sparse_dir.iterdir()) if sparse_dir.exists() else []:
        if candidate.is_dir():
            model_dir = candidate
            break
    if model_dir is None:
        model_dir = sparse_dir / "0"
    if not model_dir.exists():
        model_dir = sparse_dir
    print("Using sparse model directory:", model_dir)

    dense_dir.mkdir(parents=True, exist_ok=True)
    pms_cmd = [colmap_bin, "patch_match_stereo", "--workspace_path", str(workspace)]
    for k, v in preset_args.items():
        if k.startswith("PatchMatchStereo."):
            pms_cmd.extend([f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)])
    if force_cpu:
        # Verify flag names with local COLMAP version; adjust if needed.
        pms_cmd.extend(["--PatchMatchStereo.use_gpu", "false"])
    _run(pms_cmd, cwd=workspace)

    fusion_cmd = [colmap_bin, "stereo_fusion", "--workspace_path", str(workspace), "--output_path", str(fused_ply)]
    for k, v in preset_args.items():
        if k.startswith("StereoFusion."):
            fusion_cmd.extend([f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)])
    _run(fusion_cmd, cwd=workspace)

    if not fused_ply.exists():
        raise RuntimeError("Fusion did not produce fused.ply as expected.")

    print("COLMAP pipeline finished. Fused pointcloud:", fused_ply)
    return fused_ply


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COLMAP runner for GS_PIPELINE")
    parser.add_argument("--image_dir", required=True, help="Path with images (or frames)")
    parser.add_argument("--workspace", required=True, help="Workspace folder")
    parser.add_argument("--preset", default="standard", choices=list(COLMAP_PRESET_ARGS.keys()), help="quality preset")
    parser.add_argument("--input_type", default="photoset", choices=["photoset", "video"], help="input type; affects matcher")
    parser.add_argument("--colmap_bin", default=None, help="Optional explicit COLMAP binary path")
    args = parser.parse_args()

    try:
        run_colmap(Path(args.image_dir), Path(args.workspace), preset=args.preset, input_type=args.input_type, colmap_bin=args.colmap_bin)
    except Exception as e:
        print("ERROR:", e)
        raise
