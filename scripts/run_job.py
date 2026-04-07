# scripts/run_job.py
# Generic job runner: invoked by watcher to perform a single job in a subprocess.
from pathlib import Path
import sys
import json
import traceback

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

# worker exposes run_colmap_with_fallback(...) — same call as the smoke test
from worker import run_colmap_with_fallback
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
    job = json.loads(job_path.read_text(encoding="utf-8"))
    # Determine input and workspace
    image_dir = Path(job.get("input", {}).get("primary") or "")
    workspace = job_folder / "colmap_workspace"
    requested_preset = job.get("preset", "standard")
    try:
        print(f"Starting job {job.get('job_id')} image_dir={image_dir} workspace={workspace} preset={requested_preset}")
        if Image is not None:
            print(f"PIL available: {getattr(Image, '__version__', 'unknown')}")
        else:
            print("Pillow not available — skipping reencode")
        fused = run_colmap_with_fallback(job_path, image_dir, workspace, requested_preset=requested_preset)
        print("JOB DONE. fused:", fused)
        return 0
    except Exception as e:
        traceback.print_exc()
        print("JOB FAILED:", e)
        return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv))
