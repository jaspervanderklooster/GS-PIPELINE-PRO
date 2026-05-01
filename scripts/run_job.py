# scripts/run_job.py
# Run exactly one full job in a subprocess, then exit.
from __future__ import annotations

from pathlib import Path
import json
import sys
import traceback

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from worker import process_job, ensure_job_shape, load_json  # noqa: E402

try:
    from PIL import Image
except Exception:
    Image = None


def main(argv):
    if len(argv) < 2:
        print("Usage: python scripts/run_job.py <job_folder>", file=sys.stderr)
        return 2

    job_folder = Path(argv[1])
    job_path = job_folder / "job.json"
    if not job_path.exists():
        print("job.json not found in", job_folder, file=sys.stderr)
        return 3

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        print(f"Starting full job {job.get('job_id')} in {job_folder}")
        print(f"WORKER_MODULE_ROOT: {repo}")
        if Image is not None:
            print(f"PIL available: {getattr(Image, '__version__', 'unknown')}")
        else:
            print("Pillow not available in subprocess")

        process_job(job_folder)

        latest = ensure_job_shape(load_json(job_path))
        state = latest.get("state")
        print(f"JOB STATE after process_job: {state}")
        print(f"JOB RESULT: {latest.get('result')}")
        print(f"JOB SUMMARY: {latest.get('result_summary')}")
        return 0 if state == "done" else 1
    except Exception as exc:
        traceback.print_exc()
        print("JOB FAILED:", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
