"""Minimal watcher dispatch utilities for subprocess jobs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_job.py"


def dispatch_job(jobdir: str | Path) -> int:
    jobdir = str(jobdir)
    venv_py_windows = Path(REPO_ROOT) / ".venv" / "Scripts" / "python.exe"
    venv_py_unix = Path(REPO_ROOT) / ".venv" / "bin" / "python"
    if venv_py_windows.exists():
        python_exe = str(venv_py_windows)
    elif venv_py_unix.exists():
        python_exe = str(venv_py_unix)
    else:
        python_exe = sys.executable

    env = os.environ.copy()
    if Path(python_exe).parent.exists():
        venv_bin = str(Path(python_exe).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    job_path = Path(jobdir) / "job.json"
    job = {}
    if job_path.exists():
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except Exception:
            job = {}

    worker_log = Path(jobdir) / "worker_subprocess.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    with worker_log.open("a", encoding="utf-8") as fh:
        fh.write(f"Using python executable: {python_exe}\n")
        p = subprocess.Popen(
            [python_exe, str(RUNNER_SCRIPT), jobdir],
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
        )
        rc = p.wait()
        if job:
            fh.write(f"Job {job.get('job_id', Path(jobdir).name)} exited rc={rc}\n")
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python watcher.py <jobdir>")
        raise SystemExit(2)
    raise SystemExit(dispatch_job(sys.argv[1]))
