# colmap_runner.py
"""COLMAP runner tuned for GS_PIPELINE-PRO.

Features:
- Preset mappings for SIFT / PatchMatch / Fusion args.
- Probes supported COLMAP CLI flags (via `colmap <cmd> --help`) and only passes compatible flags.
- Builds stereo workspace with image_undistorter before dense steps.
- Supports force_cpu: uses explicit COLMAP flags when available, otherwise falls back to
  disabling CUDA for the process via CUDA_VISIBLE_DEVICES=''.
- Raises informative RuntimeError with stdout/stderr when a step fails.
"""

from __future__ import annotations
import os
import shlex
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --- Presets: human -> COLMAP parameter mapping ---
COLMAP_PRESET_ARGS: Dict[str, Dict[str, object]] = {
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


# -- Helpers -----------------------------------------------------------
def _env_colmap_bin() -> Optional[str]:
    return os.environ.get("COLMAP_BIN")


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    """Run command, print stdout/stderr, raise RuntimeError if exit != 0."""
    print("RUN:", " ".join(shlex.quote(x) for x in cmd))
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, shell=False, env=env, capture_output=True, text=True)
    out = res.stdout or ""
    err = res.stderr or ""
    if out.strip():
        print("stdout:\n" + out[-4000:])
    if err.strip():
        print("stderr:\n" + err[-4000:])
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(cmd)}\nstdout:\n{out}\nstderr:\n{err}")


def probe_supported_flags(colmap_bin: str, command: str) -> Set[str]:
    """Return a set of supported flags (without leading --) for a COLMAP command by parsing --help."""
    try:
        p = subprocess.run([colmap_bin, command, "--help"], capture_output=True, text=True, check=False)
        help_text = (p.stdout or "") + "\n" + (p.stderr or "")
    except Exception:
        help_text = ""
    # find patterns like --Something.option or --flag-name
    matches = set(re.findall(r"--[A-Za-z0-9_.-]+", help_text))
    # normalize without leading --
    return {m.lstrip("-") for m in matches}


def _format_flag_value(value: object) -> str:
    # bools should be lowercase true/false
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _supports(flags: Set[str], key: str) -> bool:
    # key e.g. 'SiftExtraction.max_num_features' -> check direct presence
    return key in flags


# -- Main runner -------------------------------------------------------
def run_colmap(
    image_dir: Path,
    workspace: Path,
    preset: str = "standard",
    input_type: str = "photoset",
    colmap_bin: Optional[str] = None,
    force_cpu: bool = False,
) -> Path:
    """Execute COLMAP pipeline and return fused pointcloud (PLY) path.

    Args:
        image_dir: path to images (frames).
        workspace: path where COLMAP will write database, sparse/dense outputs.
        preset: one of keys in COLMAP_PRESET_ARGS.
        input_type: 'photoset' or 'video' (affects matcher).
        colmap_bin: explicit colmap binary path or None to use COLMAP_BIN env or 'colmap'.
        force_cpu: if True, attempt to force CPU-only run (try flags first, then env fallback).
    """
    if preset not in COLMAP_PRESET_ARGS:
        raise ValueError(f"Unknown preset: {preset}")

    colmap_bin = colmap_bin or _env_colmap_bin() or "colmap"
    image_dir = Path(image_dir)
    workspace = Path(workspace)

    if not image_dir.exists() or not any(image_dir.glob("*")):
        raise RuntimeError(f"Image directory empty or missing: {image_dir}")

    preset_args = COLMAP_PRESET_ARGS[preset]

    # Make workspace structure
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    dense_dir = workspace / "dense"
    fused_ply = workspace / "fused.ply"

    # If database exists, remove to ensure reproducible run
    if db_path.exists():
        print("Existing database found; removing to ensure reproducible run.")
        try:
            db_path.unlink()
        except Exception:
            pass

    # Probe supported flags for relevant commands
    feat_flags = probe_supported_flags(colmap_bin, "feature_extractor")
    match_flags = probe_supported_flags(colmap_bin, "exhaustive_matcher") | probe_supported_flags(colmap_bin, "sequential_matcher")
    mapper_flags = probe_supported_flags(colmap_bin, "mapper")
    undistort_flags = probe_supported_flags(colmap_bin, "image_undistorter")
    pms_flags = probe_supported_flags(colmap_bin, "patch_match_stereo")
    fusion_flags = probe_supported_flags(colmap_bin, "stereo_fusion")

    # We will optionally set an env override when we force CPU via CUDA_VISIBLE_DEVICES=''
    env_override: Optional[dict] = None

    # ---------------- feature_extractor ----------------
    feat_cmd: List[str] = [colmap_bin, "feature_extractor", "--database_path", str(db_path), "--image_path", str(image_dir)]
    # add Sift/FeatureExtraction args if supported
    for k, v in preset_args.items():
        if k.startswith("SiftExtraction.") or k.startswith("FeatureExtraction."):
            # only add if supported by this colmap binary
            if _supports(feat_flags, k):
                feat_cmd.extend([f"--{k}", _format_flag_value(v)])
            else:
                print(f"Note: COLMAP 'feature_extractor' does not support --{k}; skipping.")
    # handle force_cpu for feature extractor
    if force_cpu:
        # prefer FeatureExtraction.use_gpu, fallback to SiftExtraction.use_gpu
        if _supports(feat_flags, "FeatureExtraction.use_gpu"):
            feat_cmd.extend(["--FeatureExtraction.use_gpu", "false"])
            print("Forcing CPU: added --FeatureExtraction.use_gpu false")
        elif _supports(feat_flags, "SiftExtraction.use_gpu"):
            feat_cmd.extend(["--SiftExtraction.use_gpu", "false"])
            print("Forcing CPU: added --SiftExtraction.use_gpu false")
        else:
            # no known flag; we'll use env fallback
            env_override = os.environ.copy()
            env_override["CUDA_VISIBLE_DEVICES"] = ""
            print("Warning: no GPU-disable option found for feature_extractor; using CUDA_VISIBLE_DEVICES='' to force CPU.")

    _run(feat_cmd, cwd=workspace, env=env_override)

    # ---------------- matcher ----------------
    matcher = "sequential_matcher" if input_type == "video" else "exhaustive_matcher"
    match_cmd: List[str] = [colmap_bin, matcher, "--database_path", str(db_path)]
    _run(match_cmd, cwd=workspace, env=env_override)

    # ---------------- mapper ----------------
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd: List[str] = [
        colmap_bin,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
    ]
    # No preset flags for mapper in our mapping - but keep for completeness:
    for k, v in preset_args.items():
        if k.startswith("Mapper.") or k.startswith("Mapping."):
            if _supports(mapper_flags, k):
                mapper_cmd.extend([f"--{k}", _format_flag_value(v)])
            else:
                print(f"Note: COLMAP 'mapper' does not support --{k}; skipping.")
    _run(mapper_cmd, cwd=workspace, env=env_override)

    # Find sparse model directory
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

    # ---------------- image undistorter -> stereo workspace ----------------
    stereo_dir = workspace / "stereo"
    stereo_dir.mkdir(parents=True, exist_ok=True)
    undistort_cmd: List[str] = [
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
    ]
    # add max_image_size if available in preset
    if "StereoFusion.max_image_size" in preset_args:
        key = "StereoFusion.max_image_size"
        # image_undistorter accepts --max_image_size
        if _supports(undistort_flags, "max_image_size") or True:
            undistort_cmd.extend(["--max_image_size", _format_flag_value(preset_args[key])])
    _run(undistort_cmd, cwd=workspace, env=env_override)

    # ---------------- patch_match_stereo ----------------
    pms_cmd: List[str] = [colmap_bin, "patch_match_stereo", "--workspace_path", str(workspace)]
    for k, v in preset_args.items():
        if k.startswith("PatchMatchStereo."):
            if _supports(pms_flags, k):
                pms_cmd.extend([f"--{k}", _format_flag_value(v)])
            else:
                print(f"Note: COLMAP 'patch_match_stereo' does not support --{k}; skipping.")
    # force_cpu handling for patch_match_stereo
    if force_cpu:
        if _supports(pms_flags, "PatchMatchStereo.use_gpu"):
            pms_cmd.extend(["--PatchMatchStereo.use_gpu", "false"])
            print("Forcing CPU: added --PatchMatchStereo.use_gpu false")
        elif _supports(pms_flags, "PatchMatchStereo.gpu_index"):
            # set gpu_index to -1 (commonly used to trigger CPU-only), only if supported
            pms_cmd.extend(["--PatchMatchStereo.gpu_index", "-1"])
            print("Forcing CPU: added --PatchMatchStereo.gpu_index -1")
        else:
            # no patch_match flag: ensure env_override is set (reuse existing or create)
            if env_override is None:
                env_override = os.environ.copy()
                env_override["CUDA_VISIBLE_DEVICES"] = ""
            print("Warning: no GPU-disable option found for patch_match_stereo; using CUDA_VISIBLE_DEVICES='' to force CPU.")

    _run(pms_cmd, cwd=workspace, env=env_override)

    # ---------------- stereo_fusion ----------------
    fusion_cmd: List[str] = [colmap_bin, "stereo_fusion", "--workspace_path", str(workspace), "--output_path", str(fused_ply)]
    for k, v in preset_args.items():
        if k.startswith("StereoFusion."):
            if _supports(fusion_flags, k):
                fusion_cmd.extend([f"--{k}", _format_flag_value(v)])
            else:
                print(f"Note: COLMAP 'stereo_fusion' does not support --{k}; skipping.")
    _run(fusion_cmd, cwd=workspace, env=env_override)

    if not fused_ply.exists():
        raise RuntimeError("Fusion did not produce fused.ply as expected.")

    print("COLMAP pipeline finished. Fused pointcloud:", fused_ply)
    return fused_ply


# CLI support
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COLMAP runner for GS_PIPELINE-PRO")
    parser.add_argument("--image_dir", required=True, help="Path with images (or frames)")
    parser.add_argument("--workspace", required=True, help="Workspace folder")
    parser.add_argument("--preset", default="standard", choices=list(COLMAP_PRESET_ARGS.keys()), help="quality preset")
    parser.add_argument("--input_type", default="photoset", choices=["photoset", "video"], help="input type; affects matcher")
    parser.add_argument("--colmap_bin", default=None, help="Optional explicit COLMAP binary path")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU-only run (try flags then env fallback)")
    args = parser.parse_args()

    try:
        run_colmap(
            Path(args.image_dir),
            Path(args.workspace),
            preset=args.preset,
            input_type=args.input_type,
            colmap_bin=args.colmap_bin,
            force_cpu=args.force_cpu,
        )
    except Exception as e:
        print("ERROR:", e)
        raise