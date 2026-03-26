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
        "StereoFusion.max_image_size": 2000,
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
        "StereoFusion.max_image_size": 3200,
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
        "StereoFusion.max_image_size": 4500,
        "StereoFusion.min_num_pixels": 1,
        "StereoFusion.max_reproj_error": 2.5,
        "StereoFusion.geom_consistency": True,
        "SiftExtraction.max_num_features": 12288,
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


def _colmap_help_text(colmap_bin: str, command: str) -> str:
    """Return help text for `colmap <command> --help`. On error return empty string."""
    try:
        res = subprocess.run([colmap_bin, command, "--help"], capture_output=True, text=True)
        return (res.stdout or "") + (res.stderr or "")
    except Exception:
        return ""


def _colmap_has_flag(colmap_bin: str, command: str, flag: str) -> bool:
    help_txt = _colmap_help_text(colmap_bin, command)
    return flag in help_txt


def _maybe_extend_with_preset_args(cmd: List[str], preset_args: Dict[str, object], colmap_bin: str, colmap_cmd: str, prefix: str) -> None:
    """Append preset args from preset_args that start with prefix, but only if COLMAP help declares the flag."""
    help_txt = _colmap_help_text(colmap_bin, colmap_cmd)
    for k, v in preset_args.items():
        if not k.startswith(prefix):
            continue
        flag = f"--{k}"
        if flag in help_txt:
            cmd.extend([flag, str(v).lower() if isinstance(v, bool) else str(v)])
        else:
            # silently skip unsupported options but log for visibility
            print(f"Note: COLMAP '{colmap_cmd}' does not support {flag}; skipping.")


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
    stereo_dir = workspace / "stereo"
    fused_ply = workspace / "fused.ply"

    if db_path.exists():
        print("Existing database found; removing to ensure reproducible run.")
        db_path.unlink()

    #
    # FEATURE EXTRACTION
    #
    feat_cmd = [colmap_bin, "feature_extractor", "--database_path", str(db_path), "--image_path", str(image_dir)]
    # add SiftExtraction.* options only if supported by the binary
    _maybe_extend_with_preset_args(feat_cmd, preset_args, colmap_bin, "feature_extractor", "SiftExtraction.")
    # GPU disabling for SIFT: prefer FeatureExtraction.use_gpu, fallback to SiftExtraction.use_gpu
    if force_cpu:
        if _colmap_has_flag(colmap_bin, "feature_extractor", "--FeatureExtraction.use_gpu"):
            feat_cmd.extend(["--FeatureExtraction.use_gpu", "false"])
        elif _colmap_has_flag(colmap_bin, "feature_extractor", "--SiftExtraction.use_gpu"):
            feat_cmd.extend(["--SiftExtraction.use_gpu", "false"])
        else:
            print("Warning: no GPU-disable flag found for feature_extractor; continuing without forcing CPU for SIFT extraction.")
    _run(feat_cmd, cwd=workspace)

    #
    # MATCHING
    #
    matcher = "sequential_matcher" if input_type == "video" else "exhaustive_matcher"
    match_cmd = [colmap_bin, matcher, "--database_path", str(db_path)]
    _run(match_cmd, cwd=workspace)

    #
    # MAPPER
    #
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

    # find model directory (sparse/0 or sparse/<first-dir>), fallback to sparse if needed
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

    #
    # BUILD STEREO WORKSPACE
    #
    stereo_dir.mkdir(parents=True, exist_ok=True)
    max_img_size = preset_args.get("StereoFusion.max_image_size", 2000)
    undist_cmd = [
        colmap_bin,
        "image_undistorter",
        "--image_path",
        str(image_dir),
        "--input_path",
        str(model_dir),
        "--output_path",
        str(stereo_dir),
        "--output_type",
        "COLMAP",
        "--max_image_size",
        str(max_img_size),
    ]
    _run(undist_cmd, cwd=workspace)

    #
    # PATCH MATCH STEREO (dense)
    #
    pms_cmd = [colmap_bin, "patch_match_stereo", "--workspace_path", str(stereo_dir)]
    # add patchmatch options only if supported
    _maybe_extend_with_preset_args(pms_cmd, preset_args, colmap_bin, "patch_match_stereo", "PatchMatchStereo.")
    if force_cpu:
        # If COLMAP supports an explicit use_gpu flag, prefer that.
        if _colmap_has_flag(colmap_bin, "patch_match_stereo", "--PatchMatchStereo.use_gpu"):
            pms_cmd.extend(["--PatchMatchStereo.use_gpu", "false"])
        elif _colmap_has_flag(colmap_bin, "patch_match_stereo", "--PatchMatchStereo.gpu_index"):
            pms_cmd.extend(["--PatchMatchStereo.gpu_index", "-1"])
        else:
            print("Warning: no GPU-disable option found for patch_match_stereo; continuing without forcing CPU for PatchMatch.")
    _run(pms_cmd, cwd=workspace)

    #
    # STEREO FUSION
    #
    fusion_cmd = [colmap_bin, "stereo_fusion", "--workspace_path", str(stereo_dir), "--output_path", str(fused_ply)]
    # add stereo fusion args only if supported by the binary
    _maybe_extend_with_preset_args(fusion_cmd, preset_args, colmap_bin, "stereo_fusion", "StereoFusion.")
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