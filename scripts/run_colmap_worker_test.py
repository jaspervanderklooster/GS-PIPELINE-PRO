# scripts/run_colmap_worker_test.py
from pathlib import Path
import json, traceback, sys, os

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

# importeer worker functie (verwacht run_colmap_with_fallback)
from worker import run_colmap_with_fallback

def write_job(job_folder: Path):
    job_folder.mkdir(parents=True, exist_ok=True)
    job_path = job_folder / "job.json"
    imgs = list((repo / "tmp_colmap_test" / "images").glob("*.jpg"))
    job = {
        "job_id": job_folder.name,
        "input": {
            "primary": str((repo / "tmp_colmap_test" / "images").resolve()),
            "type": "photoset",
            "counts": {"photos": len(imgs)}
        },
        "preset": "hq",   # hoog zodat fallback getest kan worden
        "artifacts": {}
    }
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job_path

def main():
    try:
        job_folder = repo / "processing" / "smoke_test_job"
        if not job_folder.exists():
            job_folder.mkdir(parents=True, exist_ok=True)
        job_path = write_job(job_folder)
        image_dir = repo / "tmp_colmap_test" / "images"
        workspace = repo / "processing" / "smoke_test_job" / "colmap_workspace"
        print("Job path:", job_path)
        print("Image dir:", image_dir)
        print("Workspace:", workspace)
        fused = run_colmap_with_fallback(job_path, image_dir, workspace, requested_preset="hq")
        print("run_colmap_with_fallback returned fused:", fused)
    except Exception as e:
        traceback.print_exc()
        print("FAILED:", e)

if __name__ == "__main__":
    main()
