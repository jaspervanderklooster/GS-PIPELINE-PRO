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
sys.path.insert(0, str(repo))
run_job_path = repo / "scripts" / "run_job.py"

import worker  # noqa: E402


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

        delivery_job = root / "processing" / "delivery_success"
        write_job(delivery_job, state="done")
        artifact = delivery_job / "final" / "model.ply"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("ply", encoding="utf-8")
        delivery_data = json.loads((delivery_job / "job.json").read_text(encoding="utf-8"))
        delivery_data["result"] = "success"
        delivery_data["result_summary"] = "delivery smoke"
        delivery_data["result_artifact"] = str(artifact)
        (delivery_job / "job.json").write_text(json.dumps(delivery_data, indent=2), encoding="utf-8")

        worker.OUTBOX = root / "outbox"
        worker.ARCHIVE_DONE = root / "archive" / "done"
        worker.ARCHIVE_FAILED = root / "archive" / "failed"
        job = worker.ensure_job_shape(worker.load_json(delivery_job / "job.json"))
        worker.finalize_terminal_job(delivery_job, delivery_job / "job.json", job)

        outbox_dir = worker.OUTBOX / "smoke" / "smoke" / "delivery_success"
        delivered_job_path = outbox_dir / "job.json"
        if not delivered_job_path.exists():
            print("expected delivered job.json")
            return 1
        delivered = worker.ensure_job_shape(worker.load_json(delivered_job_path))
        delivered_artifact = outbox_dir / "model.ply"
        if delivered.get("result_artifact") != str(delivered_artifact):
            print(f"expected delivered artifact path, got {delivered.get('result_artifact')}")
            return 1
        delivery = delivered.get("delivery") or {}
        for key in ["finalized", "outbox_path", "delivered_at", "archive_status", "archive_note"]:
            if not delivery.get(key):
                print(f"missing delivery metadata {key}: {delivery}")
                return 1
        if delivery.get("archive_status") != "archived" or delivery.get("outbox_path") != str(outbox_dir):
            print(f"unexpected delivery metadata: {delivery}")
            return 1
        summary = (outbox_dir / "summary.txt").read_text(encoding="utf-8")
        if str(delivered_artifact) not in summary or str(artifact) in summary:
            print("summary did not point to delivered artifact")
            return 1

    print("smoke_run_job_archived: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
