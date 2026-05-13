#!/usr/bin/env python3
"""Smoke checks for preprocessing safety without COLMAP or LichtFeld."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preprocessor
import worker


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        job_folder = root / "job"
        raw = job_folder / "input_raw" / "photos"
        raw.mkdir(parents=True)
        source = raw / "a.jpg"
        source.write_bytes(b"not a real image")

        staged = preprocessor._prepare_staging_from_photos(job_folder, raw)
        staged_file = job_folder / "staging" / "frames" / "frame_000001.jpg"
        if staged != 1 or not staged_file.exists() or not source.exists():
            print("photoset staging did not preserve input_raw")
            return 1

        report = worker._preprocess_safety_report(
            raw,
            {"photos": worker.PREPROCESS_MAX_PHOTOS_STANDARD + 1, "videos": 0, "zips": 0},
            "standard",
            "after_unzip",
        )
        if report.get("classification") != "too_large_for_safe_preprocessing":
            print(f"expected unsafe report, got {report}")
            return 1

        job = {"input": {"counts": {}}, "assessment": {}}
        worker._record_preprocess_safety(job, report)
        if "after_unzip" not in job.get("assessment", {}).get("worker_preprocess_safety", {}):
            print("safety report was not recorded by stage")
            return 1

        lf_cfg = {"max_width": 4500}
        scaling = {"rules": []}
        configured, capped = worker._cap_lichtfeld_max_width(lf_cfg, scaling)
        if configured != 4500 or capped != worker.LICHTFELD_MAX_WIDTH_LIMIT or lf_cfg["max_width"] != 4096:
            print(f"LichtFeld max-width cap failed: configured={configured} capped={capped} cfg={lf_cfg}")
            return 1

        meta_path = root / "status_meta" / "owner" / "project.json"
        original_replace = Path.replace
        replace_calls = {"n": 0}

        def flaky_replace(self, target):
            if Path(target) == meta_path and replace_calls["n"] < 2:
                replace_calls["n"] += 1
                raise PermissionError("simulated status lock")
            return original_replace(self, target)

        Path.replace = flaky_replace
        try:
            worker.save_status_meta_json_atomic(meta_path, {"ok": True}, replace_attempts=3, retry_delay=0)
        finally:
            Path.replace = original_replace
        if replace_calls["n"] != 2 or json.loads(meta_path.read_text(encoding="utf-8")).get("ok") is not True:
            print("status meta retry write failed")
            return 1

        original_status_file = worker.status_file
        original_status_meta = worker.status_meta
        try:
            worker.status_file = lambda job: (_ for _ in ()).throw(PermissionError("blocked status file"))
            worker.status_meta = lambda job: (_ for _ in ()).throw(PermissionError("blocked status meta"))
            worker.write_user_status({"job_id": "status-smoke", "input": {"owner": "owner", "tag": "project"}}, "In behandeling")
        finally:
            worker.status_file = original_status_file
            worker.status_meta = original_status_meta

    print("smoke_preprocess_safety: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
