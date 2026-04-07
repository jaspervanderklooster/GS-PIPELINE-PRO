import json
from pathlib import Path

import pytest

import worker


def _setup_job(tmp_path: Path, preset: str = "hq"):
    job_folder = tmp_path / "job_001"
    job_folder.mkdir(parents=True)
    job_path = job_folder / "job.json"
    job = {
        "job_id": "job_001",
        "preset": preset,
        "preset_used": preset,
        "input": {"type": "photoset", "tag": "demo", "owner": "tester", "counts": {}},
        "artifacts": {},
        "state": "ready_for_training",
    }
    job_path.write_text(json.dumps(job), encoding="utf-8")

    image_dir = job_folder / "staging" / "frames"
    image_dir.mkdir(parents=True)
    (image_dir / "frame_0001.jpg").write_bytes(b"x")

    workspace = job_folder / "colmap"
    workspace.mkdir(parents=True)
    return job_folder, job_path, image_dir, workspace


def test_colmap_fallback_succeeds_on_cpu(tmp_path, monkeypatch):
    job_folder, job_path, image_dir, workspace = _setup_job(tmp_path)
    monkeypatch.setattr(worker, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(worker, "STATUS_META_DIR", tmp_path / "status_meta")

    calls = []

    def fake_run_colmap_pipeline(**kwargs):
        calls.append((kwargs["preset"], kwargs.get("force_cpu", False)))
        if len(calls) < 3:
            raise RuntimeError("CUDA out of memory")
        fused = kwargs["workspace"] / "fused.ply"
        fused.write_text("ply", encoding="utf-8")
        return fused

    monkeypatch.setattr(worker, "run_colmap_pipeline", fake_run_colmap_pipeline)

    fused = worker.run_colmap_with_fallback(job_path, image_dir, workspace, "hq")
    job = json.loads(job_path.read_text(encoding="utf-8"))

    assert fused == workspace / "fused.ply"
    assert calls == [("hq", False), ("hq_safe", False), ("hq_safe", True)]
    assert job["preset_used"] == "hq_safe"
    assert "fallback" in (job.get("result_summary") or "").lower()


def test_colmap_fallback_raises_after_all_retries(tmp_path, monkeypatch):
    _, job_path, image_dir, workspace = _setup_job(tmp_path)
    monkeypatch.setattr(worker, "OUTBOX", tmp_path / "outbox")
    monkeypatch.setattr(worker, "STATUS_META_DIR", tmp_path / "status_meta")

    def always_fail(**kwargs):
        raise RuntimeError("CUDAError: out of memory")

    monkeypatch.setattr(worker, "run_colmap_pipeline", always_fail)

    with pytest.raises(RuntimeError, match="fallback geprobeerd"):
        worker.run_colmap_with_fallback(job_path, image_dir, workspace, "standard")
