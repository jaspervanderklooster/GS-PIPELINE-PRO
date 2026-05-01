#!/usr/bin/env python3
"""Smoke checks for preprocessing safety without COLMAP or LichtFeld."""

from __future__ import annotations

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

    print("smoke_preprocess_safety: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
