#!/usr/bin/env python3
"""Simple smoke check for COLMAP fallback flow without real COLMAP."""

import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import worker


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        job_folder = root / "job"
        job_folder.mkdir()
        job_path = job_folder / "job.json"
        job_path.write_text(
            json.dumps(
                {
                    "job_id": "smoke",
                    "preset": "hq",
                    "preset_used": "hq",
                    "input": {"type": "photoset", "tag": "smoke", "owner": "smoke", "counts": {}},
                    "artifacts": {},
                    "state": "ready_for_training",
                }
            ),
            encoding="utf-8",
        )

        frames = job_folder / "staging" / "frames"
        frames.mkdir(parents=True)
        (frames / "frame_0001.jpg").write_bytes(b"x")

        workspace = job_folder / "colmap"
        workspace.mkdir()

        worker.OUTBOX = root / "outbox"
        worker.STATUS_META_DIR = root / "status_meta"

        calls = {"n": 0}

        def fake_run(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("CUDA out of memory")
            fused = kwargs["workspace"] / "fused.ply"
            fused.write_text("ply", encoding="utf-8")
            return fused

        worker.run_colmap_pipeline = fake_run
        fused = worker.run_colmap_with_fallback(job_path, frames, workspace, "hq")
        if not fused.exists() or calls["n"] != 3:
            return 1
        latest = worker.ensure_job_shape(worker.load_json(job_path))
        if latest.get("preset_used") != "standard_safe":
            print(f"expected preset_used=standard_safe, got {latest.get('preset_used')}")
            return 1
        worker.set_state(job_path, latest, "colmap_done")
        latest = worker.ensure_job_shape(worker.load_json(job_path))
        if latest.get("preset_used") != "standard_safe":
            print(f"set_state overwrote preset_used: {latest.get('preset_used')}")
            return 1

    print("smoke_colmap_fallback: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
