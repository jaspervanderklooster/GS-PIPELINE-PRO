"""COLMAP runner tuned for GS_PIPELINE-PRO.

Goals:
- Keep HQ visibly better than standard, but not so aggressive that it falls over constantly.
- Prefer safe GPU fallbacks before any CPU-only attempt.
- Try multiple sparse strategies for drone-like photo sets before giving up.
- Write a registration summary so the worker can reason about actual registered images.
- Fail early on catastrophically weak sparse reconstructions instead of wasting time on dense.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set

COLMAP_PRESET_ARGS: Dict[str, Dict[str, object]] = {
    "standard": {
        "SiftExtraction.peak_threshold": 0.012,
        "SiftExtraction.edge_threshold": 10,
        "SiftExtraction.max_num_features": 8192,
        "SiftMatching.guided_matching": True,
        "PatchMatchStereo.geom_consistency": True,
        "PatchMatchStereo.num_iterations": 4,
        "PatchMatchStereo.window_radius": 3,
        "PatchMatchStereo.num_samples": 12,
        "PatchMatchStereo.filter_min_ncc": 0.35,
        "StereoFusion.max_image_size": 2800,
        "StereoFusion.min_num_pixels": 2,
        "StereoFusion.max_reproj_error": 2.0,
        "StereoFusion.geom_consistency": True,
        "Mapper.multiple_models": False,
    },
    "standard_safe": {
        "SiftExtraction.peak_threshold": 0.018,
        "SiftExtraction.edge_threshold": 11,
        "SiftExtraction.max_num_features": 6144,
        "SiftMatching.guided_matching": True,
        "PatchMatchStereo.geom_consistency": True,
        "PatchMatchStereo.num_iterations": 2,
        "PatchMatchStereo.window_radius": 2,
        "PatchMatchStereo.num_samples": 8,
        "PatchMatchStereo.filter_min_ncc": 0.40,
        "StereoFusion.max_image_size": 2200,
        "StereoFusion.min_num_pixels": 2,
        "StereoFusion.max_reproj_error": 2.5,
        "StereoFusion.geom_consistency": True,
        "Mapper.multiple_models": False,
    },
    "hq_safe": {
        "SiftExtraction.peak_threshold": 0.010,
        "SiftExtraction.edge_threshold": 9,
        "SiftExtraction.max_num_features": 9216,
        "SiftMatching.guided_matching": True,
        "PatchMatchStereo.geom_consistency": True,
        "PatchMatchStereo.num_iterations": 4,
        "PatchMatchStereo.window_radius": 3,
        "PatchMatchStereo.num_samples": 12,
        "PatchMatchStereo.filter_min_ncc": 0.34,
        "StereoFusion.max_image_size": 3200,
        "StereoFusion.min_num_pixels": 1,
        "StereoFusion.max_reproj_error": 2.1,
        "StereoFusion.geom_consistency": True,
        "Mapper.multiple_models": False,
    },
    "hq": {
        "SiftExtraction.peak_threshold": 0.0085,
        "SiftExtraction.edge_threshold": 8,
        "SiftExtraction.max_num_features": 10240,
        "SiftMatching.guided_matching": True,
        "PatchMatchStereo.geom_consistency": True,
        "PatchMatchStereo.num_iterations": 5,
        "PatchMatchStereo.window_radius": 4,
        "PatchMatchStereo.num_samples": 16,
        "PatchMatchStereo.filter_min_ncc": 0.32,
        "StereoFusion.max_image_size": 3600,
        "StereoFusion.min_num_pixels": 1,
        "StereoFusion.max_reproj_error": 2.2,
        "StereoFusion.geom_consistency": True,
        "Mapper.multiple_models": False,
    },
}

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _env_colmap_bin() -> Optional[str]:
    return os.environ.get("COLMAP_BIN")


def _run(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    print("RUN:", " ".join(shlex.quote(x) for x in cmd))
    kwargs = {
        "cwd": str(cwd) if cwd else None,
        "shell": False,
        "capture_output": True,
        "text": True,
    }
    if env is not None:
        kwargs["env"] = env
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        res = subprocess.run(cmd, **kwargs)
    except TypeError as exc:
        if "env" in str(exc).lower() and "env" in kwargs:
            kwargs.pop("env", None)
            res = subprocess.run(cmd, **kwargs)
        else:
            raise
    out = res.stdout or ""
    err = res.stderr or ""
    if out.strip():
        print("stdout:\n" + out[-4000:])
    if err.strip():
        print("stderr:\n" + err[-4000:])
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(cmd)}\nstdout:\n{out}\nstderr:\n{err}")


def probe_supported_flags(colmap_bin: str, command: str) -> Set[str]:
    try:
        p = subprocess.run([colmap_bin, command, "--help"], capture_output=True, text=True, check=False)
        help_text = (p.stdout or "") + "\n" + (p.stderr or "")
    except Exception:
        help_text = ""
    matches = set(re.findall(r"--[A-Za-z0-9_.-]+", help_text))
    return {m.lstrip("-") for m in matches}


def _command_available(colmap_bin: str, command: str) -> bool:
    try:
        p = subprocess.run([colmap_bin, command, "--help"], capture_output=True, text=True, check=False)
        text = ((p.stdout or "") + "\n" + (p.stderr or "")).lower()
        if "unknown command" in text or "unrecognized" in text or "invalid choice" in text:
            return False
        return bool(text.strip())
    except Exception:
        return False


def _format_flag_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _supports(flags: Set[str], key: str) -> bool:
    return key in flags


def _list_input_images(image_dir: Path) -> list[Path]:
    images = []
    for p in sorted(image_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            images.append(p)
    return images


def _extract_number(name: str) -> Optional[int]:
    matches = re.findall(r"(\d+)", name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def _looks_ordered_capture(images: list[Path]) -> bool:
    if len(images) < 8:
        return False
    nums = []
    for p in images:
        n = _extract_number(p.stem)
        if n is not None:
            nums.append(n)
    if len(nums) < max(6, int(len(images) * 0.7)):
        return False
    ascending = 0
    for a, b in zip(nums, nums[1:]):
        if b > a:
            ascending += 1
    return ascending >= max(5, int((len(nums) - 1) * 0.75))


def _count_registered_images_from_text_model(images_txt: Path) -> int:
    if not images_txt.exists():
        return 0
    meaningful = []
    for line in images_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        meaningful.append(stripped)
    if not meaningful:
        return 0
    return len(meaningful) // 2


def _convert_model_to_text(workspace: Path, model_dir: Path, colmap_bin: str, subdir_name: str) -> tuple[int, Path, str]:
    text_dir = workspace / subdir_name
    if text_dir.exists():
        shutil.rmtree(text_dir, ignore_errors=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            colmap_bin,
            "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(text_dir),
            "--output_type", "TXT",
        ]
        _run(cmd, cwd=workspace)
        count = _count_registered_images_from_text_model(text_dir / "images.txt")
        return count, text_dir, "ok" if count > 0 else "empty"
    except Exception as exc:
        print(f"Warning: failed to convert model to text: {exc}")
        return 0, text_dir, f"failed: {exc}"


def _choose_model_dir(sparse_root: Path) -> Path:
    for candidate in sorted(sparse_root.iterdir()) if sparse_root.exists() else []:
        if candidate.is_dir():
            return candidate
    fallback = sparse_root / "0"
    if fallback.exists():
        return fallback
    return sparse_root


def _registration_good_enough(registered: int, total_images: int) -> bool:
    if total_images <= 0:
        return registered > 0
    ratio = registered / float(total_images)
    if registered < min(30, total_images):
        return False
    if total_images >= 100 and ratio < 0.12:
        return False
    return True


def _registration_too_weak_for_dense(registered: int, total_images: int) -> bool:
    if registered <= 0:
        return True
    if total_images <= 0:
        return registered < 10
    ratio = registered / float(total_images)
    return registered < min(10, total_images) or ratio < 0.03


def _default_matcher_order(input_type: str, ordered_capture: bool, colmap_bin: str) -> list[str]:
    if input_type == "video":
        desired = ["sequential_matcher", "exhaustive_matcher"]
    else:
        if ordered_capture:
            desired = ["sequential_matcher", "spatial_matcher", "exhaustive_matcher"]
        else:
            desired = ["spatial_matcher", "exhaustive_matcher", "sequential_matcher"]
    return [cmd for cmd in desired if _command_available(colmap_bin, cmd)]


def _matcher_order_from_env(input_type: str, ordered_capture: bool, colmap_bin: str) -> list[str]:
    env_value = (os.environ.get("GS_COLMAP_MATCHERS") or "").strip()
    if not env_value:
        return _default_matcher_order(input_type, ordered_capture, colmap_bin)
    raw = []
    for part in env_value.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if not p.endswith("_matcher"):
            p += "_matcher"
        raw.append(p)
    result = []
    for cmd in raw:
        if _command_available(colmap_bin, cmd):
            result.append(cmd)
    return result or _default_matcher_order(input_type, ordered_capture, colmap_bin)


def _build_feature_extractor_cmd(
    colmap_bin: str,
    db_path: Path,
    image_dir: Path,
    feat_flags: Set[str],
    preset_args: dict,
    *,
    ordered_capture: bool,
    input_type: str,
    force_cpu: bool,
) -> tuple[list[str], Optional[dict]]:
    env_override: Optional[dict] = None
    cmd: List[str] = [
        colmap_bin,
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
    ]
    if (input_type == "video" or ordered_capture) and _supports(feat_flags, "ImageReader.single_camera"):
        cmd.extend(["--ImageReader.single_camera", "true"])
    for k, v in preset_args.items():
        if k.startswith(("SiftExtraction.", "FeatureExtraction.")) and _supports(feat_flags, k):
            cmd.extend([f"--{k}", _format_flag_value(v)])
    if force_cpu:
        if _supports(feat_flags, "FeatureExtraction.use_gpu"):
            cmd.extend(["--FeatureExtraction.use_gpu", "false"])
        elif _supports(feat_flags, "SiftExtraction.use_gpu"):
            cmd.extend(["--SiftExtraction.use_gpu", "false"])
        else:
            env_override = os.environ.copy()
            env_override["CUDA_VISIBLE_DEVICES"] = ""
    return cmd, env_override


def _build_match_cmd(
    colmap_bin: str,
    matcher_cmd: str,
    db_path: Path,
    match_flags: Set[str],
    preset_args: dict,
    *,
    ordered_capture: bool,
    force_cpu: bool,
    env_override: Optional[dict],
) -> tuple[list[str], Optional[dict]]:
    cmd: List[str] = [colmap_bin, matcher_cmd, "--database_path", str(db_path)]
    for k, v in preset_args.items():
        if k.startswith("SiftMatching.") and _supports(match_flags, k):
            cmd.extend([f"--{k}", _format_flag_value(v)])
    if matcher_cmd == "sequential_matcher":
        if _supports(match_flags, "SequentialMatching.overlap"):
            cmd.extend(["--SequentialMatching.overlap", "15" if ordered_capture else "10"])
        if ordered_capture and _supports(match_flags, "SequentialMatching.quadratic_overlap"):
            cmd.extend(["--SequentialMatching.quadratic_overlap", "true"])
    elif matcher_cmd == "spatial_matcher":
        if _supports(match_flags, "SpatialMatching.max_num_neighbors"):
            cmd.extend(["--SpatialMatching.max_num_neighbors", "30" if ordered_capture else "20"])
        if _supports(match_flags, "SpatialMatching.ignore_z"):
            cmd.extend(["--SpatialMatching.ignore_z", "false"])
    if force_cpu:
        if _supports(match_flags, "SiftMatching.use_gpu"):
            cmd.extend(["--SiftMatching.use_gpu", "false"])
        elif env_override is None:
            env_override = os.environ.copy()
            env_override["CUDA_VISIBLE_DEVICES"] = ""
    return cmd, env_override


def _build_mapper_cmd(
    colmap_bin: str,
    db_path: Path,
    image_dir: Path,
    output_path: Path,
    mapper_flags: Set[str],
    preset_args: dict,
    *,
    ordered_capture: bool,
) -> list[str]:
    cmd: List[str] = [
        colmap_bin,
        "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(output_path),
    ]
    for k, v in preset_args.items():
        if k.startswith(("Mapper.", "Mapping.")) and _supports(mapper_flags, k):
            cmd.extend([f"--{k}", _format_flag_value(v)])
    stabilization_flags = {
        "Mapper.min_model_size": 10,
        "Mapper.abs_pose_min_num_inliers": 20,
        "Mapper.init_min_num_inliers": 80 if ordered_capture else 60,
        "Mapper.ba_refine_principal_point": False,
    }
    for k, v in stabilization_flags.items():
        if _supports(mapper_flags, k):
            cmd.extend([f"--{k}", _format_flag_value(v)])
    return cmd


def _write_registration_summary(workspace: Path, summary: dict) -> None:
    (workspace / "registration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_colmap(
    image_dir: Path,
    workspace: Path,
    preset: str = "standard",
    input_type: str = "photoset",
    colmap_bin: Optional[str] = None,
    force_cpu: bool = False,
) -> Path:
    if preset not in COLMAP_PRESET_ARGS:
        raise ValueError(f"Unknown preset: {preset}")

    colmap_bin = colmap_bin or _env_colmap_bin() or "colmap"
    image_dir = Path(image_dir)
    workspace = Path(workspace)

    images = _list_input_images(image_dir)
    if not image_dir.exists() or not images:
        raise RuntimeError(f"Image directory empty or missing: {image_dir}")

    total_images = len(images)
    ordered_capture = _looks_ordered_capture(images)
    preset_args = COLMAP_PRESET_ARGS[preset]

    workspace.mkdir(parents=True, exist_ok=True)
    base_db = workspace / "database_base.db"
    sparse_root = workspace / "sparse"
    fused_ply = workspace / "fused.ply"
    if sparse_root.exists():
        shutil.rmtree(sparse_root, ignore_errors=True)
    sparse_root.mkdir(parents=True, exist_ok=True)
    for old_db in workspace.glob("database_*.db"):
        old_db.unlink(missing_ok=True)
    for old_tmp in workspace.glob("*_txt"):
        if old_tmp.is_dir():
            shutil.rmtree(old_tmp, ignore_errors=True)

    feat_flags = probe_supported_flags(colmap_bin, "feature_extractor")
    mapper_flags = probe_supported_flags(colmap_bin, "mapper")
    undistort_flags = probe_supported_flags(colmap_bin, "image_undistorter")
    pms_flags = probe_supported_flags(colmap_bin, "patch_match_stereo")
    fusion_flags = probe_supported_flags(colmap_bin, "stereo_fusion")

    feature_cmd, env_override = _build_feature_extractor_cmd(
        colmap_bin,
        base_db,
        image_dir,
        feat_flags,
        preset_args,
        ordered_capture=ordered_capture,
        input_type=input_type,
        force_cpu=force_cpu,
    )
    _run(feature_cmd, cwd=workspace, env=env_override)

    matcher_order = _matcher_order_from_env(input_type, ordered_capture, colmap_bin)
    if not matcher_order:
        raise RuntimeError("No supported COLMAP matcher commands available.")
    print(f"Sparse strategy order: {matcher_order} (ordered_capture={ordered_capture}, total_images={total_images})")

    attempt_summaries = []
    best_attempt: Optional[dict] = None
    selected_attempt: Optional[dict] = None

    for idx, matcher_cmd in enumerate(matcher_order, start=1):
        db_path = workspace / f"database_{idx:02d}_{matcher_cmd}.db"
        shutil.copy2(base_db, db_path)
        model_root = sparse_root / f"{idx:02d}_{matcher_cmd}"
        if model_root.exists():
            shutil.rmtree(model_root, ignore_errors=True)
        model_root.mkdir(parents=True, exist_ok=True)

        match_flags = probe_supported_flags(colmap_bin, matcher_cmd)
        match_cmd, match_env = _build_match_cmd(
            colmap_bin,
            matcher_cmd,
            db_path,
            match_flags,
            preset_args,
            ordered_capture=ordered_capture,
            force_cpu=force_cpu,
            env_override=env_override,
        )
        _run(match_cmd, cwd=workspace, env=match_env)

        mapper_cmd = _build_mapper_cmd(
            colmap_bin,
            db_path,
            image_dir,
            model_root,
            mapper_flags,
            preset_args,
            ordered_capture=ordered_capture,
        )
        _run(mapper_cmd, cwd=workspace, env=match_env)

        model_dir = _choose_model_dir(model_root)
        registered, text_dir, status = _convert_model_to_text(
            workspace,
            model_dir,
            colmap_bin,
            f"_{idx:02d}_{matcher_cmd}_txt",
        )
        ratio = round((registered / total_images), 4) if total_images else 0.0
        attempt = {
            "attempt_index": idx,
            "matcher": matcher_cmd,
            "model_dir": str(model_dir),
            "text_model_dir": str(text_dir),
            "registered_images": registered,
            "input_images": total_images,
            "registration_ratio": ratio,
            "text_status": status,
        }
        attempt_summaries.append(attempt)
        print(
            f"Sparse attempt {idx}/{len(matcher_order)} matcher={matcher_cmd} -> "
            f"registered={registered}/{total_images} ({ratio:.2%})"
        )

        if best_attempt is None or registered > int(best_attempt.get("registered_images") or 0):
            best_attempt = attempt

        if _registration_good_enough(registered, total_images):
            selected_attempt = attempt
            break

    if selected_attempt is None:
        selected_attempt = best_attempt

    summary = {
        "status": "ok" if selected_attempt else "failed",
        "preset": preset,
        "input_type": input_type,
        "ordered_capture": ordered_capture,
        "force_cpu": force_cpu,
        "input_images": total_images,
        "attempts": attempt_summaries,
        "selected_attempt": selected_attempt,
        "registered_images": int(selected_attempt.get("registered_images") or 0) if selected_attempt else 0,
        "selected_matcher": selected_attempt.get("matcher") if selected_attempt else None,
    }
    _write_registration_summary(workspace, summary)

    if not selected_attempt:
        raise RuntimeError("No sparse reconstruction attempt produced a usable model.")

    selected_model_dir = Path(str(selected_attempt["model_dir"]))
    selected_registered = int(selected_attempt.get("registered_images") or 0)

    if _registration_too_weak_for_dense(selected_registered, total_images):
        raise RuntimeError(
            "Sparse reconstruction too weak for dense stage: "
            f"best matcher={selected_attempt.get('matcher')} registered {selected_registered}/{total_images} images."
        )

    stereo_dir = workspace / "stereo"
    if stereo_dir.exists():
        shutil.rmtree(stereo_dir, ignore_errors=True)
    stereo_dir.mkdir(parents=True, exist_ok=True)

    undistort_cmd: List[str] = [
        colmap_bin,
        "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(selected_model_dir),
        "--output_path", str(stereo_dir),
        "--output_type", "COLMAP",
    ]
    if "StereoFusion.max_image_size" in preset_args and (_supports(undistort_flags, "max_image_size") or True):
        undistort_cmd.extend(["--max_image_size", _format_flag_value(preset_args["StereoFusion.max_image_size"])])
    _run(undistort_cmd, cwd=workspace, env=env_override)

    pms_cmd: List[str] = [colmap_bin, "patch_match_stereo", "--workspace_path", str(stereo_dir)]
    for k, v in preset_args.items():
        if k.startswith("PatchMatchStereo.") and _supports(pms_flags, k):
            pms_cmd.extend([f"--{k}", _format_flag_value(v)])
    if force_cpu:
        if _supports(pms_flags, "PatchMatchStereo.use_gpu"):
            pms_cmd.extend(["--PatchMatchStereo.use_gpu", "false"])
        elif _supports(pms_flags, "PatchMatchStereo.gpu_index"):
            pms_cmd.extend(["--PatchMatchStereo.gpu_index", "-1"])
        else:
            if env_override is None:
                env_override = os.environ.copy()
                env_override["CUDA_VISIBLE_DEVICES"] = ""
    _run(pms_cmd, cwd=stereo_dir, env=env_override)

    fusion_cmd: List[str] = [
        colmap_bin,
        "stereo_fusion",
        "--workspace_path", str(stereo_dir),
        "--output_path", str(fused_ply),
    ]
    for k, v in preset_args.items():
        if k.startswith("StereoFusion.") and _supports(fusion_flags, k):
            fusion_cmd.extend([f"--{k}", _format_flag_value(v)])
    _run(fusion_cmd, cwd=stereo_dir, env=env_override)

    if not fused_ply.exists():
        raise RuntimeError("Fusion did not produce fused.ply as expected.")

    print(
        "COLMAP pipeline finished. "
        f"Selected matcher={selected_attempt.get('matcher')} registered={selected_registered}/{total_images}. "
        f"Fused pointcloud: {fused_ply}"
    )
    return fused_ply


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="COLMAP runner for GS_PIPELINE-PRO")
    parser.add_argument("--image_dir", required=True, help="Path with images (or frames)")
    parser.add_argument("--workspace", required=True, help="Workspace folder")
    parser.add_argument("--preset", default="standard", choices=list(COLMAP_PRESET_ARGS.keys()), help="quality preset")
    parser.add_argument("--input_type", default="photoset", choices=["photoset", "video"], help="input type; affects matcher")
    parser.add_argument("--colmap_bin", default=None, help="Optional explicit COLMAP binary path")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU-only run (last resort)")
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
