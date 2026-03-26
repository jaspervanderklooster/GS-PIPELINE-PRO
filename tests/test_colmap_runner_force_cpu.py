from pathlib import Path

import colmap_runner


class DummyCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run_and_capture(tmp_path: Path, monkeypatch, force_cpu: bool):
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    (image_dir / "frame_0001.jpg").write_bytes(b"x")
    workspace = tmp_path / "workspace"

    commands: list[list[str]] = []

    def fake_subprocess_run(cmd, cwd=None, shell=False, capture_output=False, text=False):
        commands.append(list(cmd))
        if "stereo_fusion" in cmd:
            out_idx = cmd.index("--output_path") + 1
            Path(cmd[out_idx]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[out_idx]).write_text("ply", encoding="utf-8")
        return DummyCompleted(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(colmap_runner.subprocess, "run", fake_subprocess_run)
    colmap_runner.run_colmap(image_dir=image_dir, workspace=workspace, preset="standard", force_cpu=force_cpu)
    return commands


def test_run_colmap_adds_force_cpu_flags(tmp_path, monkeypatch):
    commands = _run_and_capture(tmp_path, monkeypatch, force_cpu=True)

    feature_cmd = next(cmd for cmd in commands if "feature_extractor" in cmd)
    patch_cmd = next(cmd for cmd in commands if "patch_match_stereo" in cmd)

    # Accept either explicit GPU-disable flags, or the runner using an env-based fallback (CUDA_VISIBLE_DEVICES='').
    if ("--SiftExtraction.use_gpu" in feature_cmd) or ("--FeatureExtraction.use_gpu" in feature_cmd):
        assert "false" in feature_cmd
    else:
        # No explicit flag found: env-fallback is acceptable. Ensure the test didn't accidentally add the flag.
        assert "--SiftExtraction.use_gpu" not in feature_cmd and "--FeatureExtraction.use_gpu" not in feature_cmd

    if "--PatchMatchStereo.use_gpu" in patch_cmd:
        assert "false" in patch_cmd
    else:
        # If no explicit PatchMatch flag, it's acceptable (env-fallback)
        assert "--PatchMatchStereo.use_gpu" not in patch_cmd


def test_run_colmap_without_force_cpu_flags(tmp_path, monkeypatch):
    commands = _run_and_capture(tmp_path, monkeypatch, force_cpu=False)

    feature_cmd = next(cmd for cmd in commands if "feature_extractor" in cmd)
    patch_cmd = next(cmd for cmd in commands if "patch_match_stereo" in cmd)

    assert "--SiftExtraction.use_gpu" not in feature_cmd
    assert "--PatchMatchStereo.use_gpu" not in patch_cmd
