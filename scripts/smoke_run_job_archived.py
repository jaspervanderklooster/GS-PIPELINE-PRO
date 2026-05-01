#!/usr/bin/env python3
"""Smoke check run_job final status when process_job archives the job folder."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
run_job_path = repo / "scripts" / "run_job.py"


def load_run_job_module():
    spec = importlib.util.spec_from_file_location("run_job_smoke_target", run_job_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {run_job_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_job(job_folder: Path, state: str = "queued") -> None:
    job_folder.mkdir(parents=True, exist_ok=True)
    (job_folder / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_folder.name,
                "state": state,
                "preset": "standard",
                "input": {"tag": "smoke", "owner": "smoke", "counts": {}},
                "artifacts": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    run_job = load_run_job_module()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        success_job = root / "processing" / "success_job"
        write_job(success_job)

        def fake_success_process_job(job_folder: Path) -> dict:
            job = json.loads((job_folder / "job.json").read_text(encoding="utf-8"))
            archive_dest = root / "archive" / "done" / job_folder.name
            archive_dest.parent.mkdir(parents=True, exist_ok=True)
            job["state"] = "done"
            job["result"] = "success"
            job["result_summary"] = "archived success smoke"
            job["delivery"] = {
                "finalized": True,
                "archive_status": "archived",
                "archive_note": str(archive_dest),
            }
            (job_folder / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
            shutil.move(str(job_folder), str(archive_dest))
            return job

        run_job.process_job = fake_success_process_job
        rc = run_job.main(["run_job.py", str(success_job)])
        if rc != 0:
            print(f"expected archived success rc=0, got {rc}")
            return 1
        if success_job.exists():
            print("expected original success job folder to be archived")
            return 1

        failed_job = root / "processing" / "failed_job"
        write_job(failed_job)

        def fake_failed_process_job(job_folder: Path) -> dict:
            job = json.loads((job_folder / "job.json").read_text(encoding="utf-8"))
            job["state"] = "failed"
            job["result"] = "failed"
            job["result_summary"] = "failed smoke"
            return job

        run_job.process_job = fake_failed_process_job
        rc = run_job.main(["run_job.py", str(failed_job)])
        if rc != 1:
            print(f"expected failed rc=1, got {rc}")
            return 1

    print("smoke_run_job_archived: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
