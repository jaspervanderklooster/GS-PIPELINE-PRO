import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from preprocessor import (
    preprocess_photoset,
    extract_frames_from_video,
    unzip_if_needed,
    detect_counts,
    detect_input_type,
)
from colmap_runner import run_colmap as run_colmap_pipeline
from utils.config import get_config

# Ensure helper subprocesses launched by preprocessing (notably dedupe_headless.py)
# use the exact same interpreter as this worker subprocess.
os.environ.setdefault("PYTHON", sys.executable)

cfg = get_config()
GS_ROOT = Path(cfg.get("GS_ROOT", r"D:\GS_PIPELINE"))
PROCESSING = Path(cfg.get("PROCESSING", str(GS_ROOT / "processing")))
ARCHIVE_DONE = Path(cfg.get("ARCHIVE_DONE", str(GS_ROOT / "archive" / "done")))
ARCHIVE_FAILED = Path(cfg.get("ARCHIVE_FAILED", str(GS_ROOT / "archive" / "failed")))
OUTPUT = Path(cfg.get("OUTPUT", str(GS_ROOT / "output")))
OUTBOX = Path(cfg.get("OUTBOX", str(GS_ROOT / "outbox")))
LOGS = Path(cfg.get("LOG_DIR", str(GS_ROOT / "logs")))
TEMP = Path(cfg.get("TEMP", str(GS_ROOT / "temp")))
STATUS_META_DIR = TEMP / "status_meta"
CLEANUP_QUEUE = Path(cfg.get("CLEANUP_QUEUE", str(GS_ROOT / "cleanup_queue")))

LICHTFELD_EXE = Path(cfg.get("LIGHTFELD_BIN", r"C:\LichtFeld-Studio\build\Release\LichtFeld-Studio.exe"))
COLMAP_BIN = cfg.get("COLMAP_BIN")

POLL_SECONDS = 5
MIN_REGISTERED_IMAGES = int(cfg.get("COLMAP_MIN_REGISTERED_IMAGES", 30))
CLEANUP_AFTER_DAYS = 7
MAX_LOG_TAIL = 6000
STATUS_RETENTION_HOURS = 24
CPU_FALLBACK_MAX_IMAGES_STANDARD = int(cfg.get("COLMAP_CPU_FALLBACK_MAX_IMAGES_STANDARD", 450))
CPU_FALLBACK_MAX_IMAGES_HQ = int(cfg.get("COLMAP_CPU_FALLBACK_MAX_IMAGES_HQ", 700))
LOW_REGISTRATION_WARNING_RATIO = float(cfg.get("COLMAP_LOW_REGISTRATION_WARNING_RATIO", 0.55))
LICHTFELD_MAX_WIDTH_LIMIT = 4096
PREPROCESS_MAX_PHOTOS_STANDARD = int(cfg.get("PREPROCESS_MAX_PHOTOS_STANDARD", 1200))
PREPROCESS_MAX_PHOTOS_HQ = int(cfg.get("PREPROCESS_MAX_PHOTOS_HQ", 800))
PREPROCESS_MAX_BYTES_STANDARD = int(cfg.get("PREPROCESS_MAX_BYTES_STANDARD", 50 * 1024**3))
PREPROCESS_MAX_BYTES_HQ = int(cfg.get("PREPROCESS_MAX_BYTES_HQ", 30 * 1024**3))

TERMINAL_STATES = {"done", "failed"}
VALID_STATES = {
    "queued",
    "preprocessing",
    "ready_for_training",
    "colmap_running",
    "colmap_done",
    "lichtfeld_running",
    "done",
    "failed",
}

DEFAULT_LF_PRESETS = {
    "standard": {
        "iter": 22000,
        "strategy": "mcmc",
        "tile_mode": 1,
        "resize_factor": "auto",
        "max_width": 3200,
        "max_cap": 650000,
        "extra_flags": [],
    },
    "hq": {
        "iter": 28000,
        "strategy": "mcmc",
        "tile_mode": 1,
        "resize_factor": "auto",
        "max_width": 3520,
        "max_cap": 850000,
        "extra_flags": ["--enable-mip"],
    },
}

_raw_presets = cfg.get("PRESETS") or cfg.get("LF_PRESETS") or {}
if "lichtfeld_presets" in _raw_presets:
    PRESETS = _raw_presets.get("lichtfeld_presets") or {}
else:
    PRESETS = _raw_presets
_PRESETS_WARNING_EMITTED = False


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def now_dt() -> datetime:
    return datetime.now().astimezone()


def parse_iso(value: str | None):
    try:
        return datetime.fromisoformat(value or "")
    except Exception:
        return None


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json_atomic(p: Path, data: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def log(job_folder: Path, msg: str):
    lp = job_folder / "worker.log"
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a", encoding="utf-8") as f:
        f.write(f"[{iso_now()}] {msg}\n")


def safe_name(value: str, fallback: str = "project") -> str:
    text = "".join(c for c in (value or "").strip() if c.isalnum() or c in "_-")
    return text or fallback


def cmd_pretty(cmd: list[str]) -> str:
    return " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd])


def run_cmd(job_folder: Path, cmd: list[str]) -> tuple[int, str, str]:
    log(job_folder, "RUN: " + cmd_pretty(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout or ""
    err = r.stderr or ""
    if out.strip():
        log(job_folder, "stdout:\n" + out[-MAX_LOG_TAIL:])
    if err.strip():
        log(job_folder, "stderr:\n" + err[-MAX_LOG_TAIL:])
    return r.returncode, out, err


def ensure_dirs():
    for p in [PROCESSING, ARCHIVE_DONE, ARCHIVE_FAILED, OUTPUT, OUTBOX, LOGS, TEMP, STATUS_META_DIR, CLEANUP_QUEUE]:
        p.mkdir(parents=True, exist_ok=True)


def list_jobs_sorted() -> list[Path]:
    jobs = [p for p in PROCESSING.iterdir() if p.is_dir() and (p / "job.json").exists()]
    jobs.sort(key=lambda p: p.name.lower())
    return jobs


def owner_of(job: dict) -> str:
    return safe_name(str(job.get("owner") or job.get("submitted_by") or job.get("input", {}).get("owner") or "unknown"), "unknown")


def tag_of(job: dict) -> str:
    return safe_name(str(job.get("input", {}).get("tag") or job.get("job_id") or "project"), "project")


def status_file(job: dict) -> Path:
    owner_dir = OUTBOX / owner_of(job)
    owner_dir.mkdir(parents=True, exist_ok=True)
    return owner_dir / f"{tag_of(job)}_status.txt"


def status_meta(job: dict) -> Path:
    return STATUS_META_DIR / owner_of(job) / f"{tag_of(job)}.json"


def write_user_status(job: dict, status: str, *, reason: str | None = None,
                      explanation: str | None = None, advice: str | None = None,
                      note: str | None = None, terminal: bool = False):
    project = str(job.get("input", {}).get("tag") or job.get("job_id") or "project")
    lines = [f"Project: {project}", f"Status: {status}"]
    if reason:
        lines.append(f"Reden: {reason}")
    if explanation:
        lines.append(f"Uitleg: {explanation}")
    if advice:
        lines.append(f"Advies: {advice}")
    if note:
        lines.append(f"Opmerking: {note}")
    lines.append(f"Laatste update: {now_dt().strftime('%Y-%m-%d %H:%M')}")
    status_file(job).write_text("\n".join(lines), encoding="utf-8")
    save_json_atomic(status_meta(job), {
        "owner": owner_of(job),
        "project": project,
        "status": status,
        "updated_at": iso_now(),
        "terminal": terminal,
        "remove_after": (now_dt() + timedelta(hours=STATUS_RETENTION_HOURS)).isoformat(timespec="seconds") if terminal else None,
    })


def cleanup_expired_status_files():
    if not STATUS_META_DIR.exists():
        return
    for meta_path in STATUS_META_DIR.rglob("*.json"):
        try:
            meta = load_json(meta_path)
            if not meta.get("terminal"):
                continue
            remove_after = parse_iso(meta.get("remove_after"))
            if not remove_after or now_dt() < remove_after:
                continue
            owner_dir = OUTBOX / safe_name(str(meta.get("owner", "")), "unknown")
            sf = owner_dir / f"{safe_name(str(meta.get('project', 'project')))}_status.txt"
            sf.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        except Exception:
            continue


def translate_failure(stage: str, error: str) -> tuple[str, str, str]:
    text = (error or "").lower()
    if "too large for safe preprocessing" in text or "te groot voor veilige preprocessing" in text:
        return (
            "De dataset is te groot voor veilige automatische preprocessing.",
            "Na uitpakken of voorbereiden bleek de input groter dan de veilige limieten.",
            "Splits de dataset op, gebruik minder beelden of kies een korter videofragment.",
        )
    if "fallback geprobeerd" in text or "fallback exhausted" in text:
        return (
            "De reconstructie faalde ook na veilige terugval-instellingen.",
            "We hebben automatisch lichtere varianten geprobeerd, maar zonder stabiel resultaat.",
            "Probeer minder beelden of lagere kwaliteit; neem contact op als je wilt dat we de logs analyseren.",
        )
    if "memory" in text or "cuda" in text or "out of memory" in text:
        return (
            "De verwerking vroeg meer geheugen dan nu beschikbaar is.",
            "Deze dataset of kwaliteitsinstelling is te zwaar voor de huidige machine-instelling.",
            "Gebruik minder input of kies een lichtere instelling. Neem contact op als je wilt dat we meekijken.",
        )
    if "registered only" in text or "colmap" in text or "registration ratio" in text:
        return (
            "De camera-locaties konden niet betrouwbaar worden bepaald.",
            "Er was te weinig overlap of te weinig bruikbaar beeldmateriaal om de opname goed uit te lijnen.",
            "Gebruik meer overlap, scherpere beelden of een korter en consistenter fragment.",
        )
    if "empty artifact" in text or "lichtfeld" in text:
        return (
            "De Gaussian Splat-training kon niet goed worden afgerond.",
            "De trainingsstap leverde geen bruikbaar eindresultaat op.",
            "Probeer een lichtere kwaliteitsinstelling of een compactere dataset.",
        )
    if stage == "preprocessing":
        return (
            "De input kon niet goed worden voorbereid voor verwerking.",
            "De aangeleverde bestanden waren niet bruikbaar genoeg voor een veilige automatische start.",
            "Controleer het bronmateriaal en probeer het opnieuw.",
        )
    return (
        "De verwerking kon niet worden afgerond.",
        "Tijdens de automatische pipeline trad een fout op waardoor het resultaat niet veilig kon worden opgeleverd.",
        "Probeer het opnieuw of vraag iemand om mee te kijken.",
    )


def status_for_worker_state(job: dict, state: str) -> tuple[str, str | None]:
    assessment = job.get("assessment", {}) or {}
    note = None
    if assessment.get("classification") == "heavy_but_allowed":
        note = assessment.get("message") or "Deze dataset is groter dan aanbevolen. De kans op mislukken is verhoogd."
    mapping = {
        "queued": "Upload ontvangen",
        "preprocessing": "In behandeling",
        "ready_for_training": "In behandeling",
        "colmap_running": "Camera locaties bepalen",
        "colmap_done": "In behandeling",
        "lichtfeld_running": "Gaussian Splat wordt getraind",
        "done": "Resultaat klaar",
        "failed": "Mislukt",
    }
    return mapping.get(state, "In behandeling"), note


def ensure_job_shape(job: dict) -> dict:
    job.setdefault("job_id", "")
    job.setdefault("state", "queued")
    job.setdefault("error", None)
    job.setdefault("created_at", iso_now())
    job.setdefault("started_at", None)
    job.setdefault("finished_at", None)
    job.setdefault("failed_at", None)
    job.setdefault("failed_stage", None)
    job.setdefault("result", None)
    job.setdefault("result_summary", None)
    job.setdefault("result_artifact", None)
    job.setdefault("registered_images", None)
    job.setdefault("preset_used", job.get("preset", "standard"))
    job.setdefault("delivery", {})
    job.setdefault("artifacts", {})
    job.setdefault("input", {})
    job["input"].setdefault("counts", {})
    job["input"].setdefault("tag", safe_name(job["job_id"], "project"))
    job.setdefault("assessment", {})
    job.setdefault("meta", {})
    job["meta"].setdefault("fallback_history", [])
    return job


def persist_job(job_path: Path, job: dict):
    save_json_atomic(job_path, job)


def set_state(job_path: Path, job: dict, new_state: str, *, note: Optional[str] = None):
    if new_state not in VALID_STATES:
        raise RuntimeError(f"Invalid worker state: {new_state}")
    previous = job.get("state", "queued")
    job["state"] = new_state
    job["preset_used"] = job.get("preset_used") or job.get("preset") or "standard"
    if not job.get("started_at") and new_state != "queued":
        job["started_at"] = iso_now()
    if new_state in TERMINAL_STATES:
        job["finished_at"] = iso_now()
    persist_job(job_path, job)
    human_status, human_note = status_for_worker_state(job, new_state)
    write_user_status(job, human_status, note=note or human_note, terminal=new_state in TERMINAL_STATES)
    log(job_path.parent, f"State {previous} -> {new_state}" + (f" ({note})" if note else ""))


def fail_job(job_path: Path, job: dict, stage: str, exc: Exception):
    msg = str(exc)
    job["error"] = msg
    job["failed_stage"] = stage
    job["failed_at"] = iso_now()
    job["result"] = "failed"
    job["result_summary"] = f"Stage '{stage}' failed: {msg}"
    reason, explanation, advice = translate_failure(stage, msg)
    set_state(job_path, job, "failed", note=stage)
    write_user_status(job, "Mislukt", reason=reason, explanation=explanation, advice=advice, terminal=True)
    log(job_path.parent, f"FAILED at {stage}: {msg}")


def frames_dir(job_folder: Path) -> Path:
    d = job_folder / "staging" / "frames"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lichtfeld_out_dir(job_id: str) -> Path:
    d = OUTPUT / job_id / "lichtfeld"
    d.mkdir(parents=True, exist_ok=True)
    return d


def outbox_project_dir(owner: str, tag: str) -> Path:
    d = OUTBOX / owner / safe_name(tag)
    d.mkdir(parents=True, exist_ok=True)
    return d


def archive_dir_for_state(state: str) -> Path:
    return ARCHIVE_DONE if state == "done" else ARCHIVE_FAILED


def resolve_effective_preset(job: dict) -> tuple[str, dict, dict]:
    global _PRESETS_WARNING_EMITTED
    requested = str(job.get("preset") or "standard").strip().lower()
    if requested == "good":
        requested = "standard"
    elif requested == "high":
        requested = "hq"
    source_presets = PRESETS or DEFAULT_LF_PRESETS
    if not PRESETS and not _PRESETS_WARNING_EMITTED:
        print("WARNING: PRESETS ontbreekt in config; fallback naar interne defaults. Vul config/presets.yaml in.")
        _PRESETS_WARNING_EMITTED = True
    if requested not in source_presets:
        requested = "standard"
    base_defaults = {
        "iter": 22000,
        "strategy": "mcmc",
        "tile_mode": 1,
        "resize_factor": "auto",
        "max_width": 3200,
        "max_cap": 650000,
        "extra_flags": [],
    }
    cfg = {**base_defaults, **dict(source_presets[requested])}
    scaling = {
        "requested": requested,
        "effective": requested,
        "photos": int(job.get("input", {}).get("counts", {}).get("photos") or 0),
        "registered_images": int(job.get("registered_images") or 0),
        "rules": [],
    }
    image_count = scaling["registered_images"] or scaling["photos"]
    assessment = job.get("assessment", {}) or {}
    if assessment.get("classification") == "heavy_but_allowed":
        cfg["iter"] = min(cfg["iter"], 32000 if requested == "hq" else 22000)
        cfg["max_cap"] = min(cfg["max_cap"], 900000 if requested == "hq" else 650000)
        scaling["effective"] = f"{requested}_safe_assessed"
        scaling["rules"].append("dataset-assessment: veilige afvlakking op zware set")
    if requested == "hq":
        if image_count >= 300:
            cfg["iter"] = min(cfg["iter"], 30000)
            cfg["max_cap"] = min(cfg["max_cap"], 900000)
            scaling["effective"] = "high_safe_300"
            scaling["rules"].append(">=300 beelden: high afgevlakt")
        if image_count >= 450:
            cfg["iter"] = min(cfg["iter"], 28000)
            cfg["max_cap"] = min(cfg["max_cap"], 850000)
            scaling["effective"] = "high_safe_450"
            scaling["rules"].append(">=450 beelden: extra afvlakking")
    _cap_lichtfeld_max_width(cfg, scaling)
    return requested, cfg, scaling


def _cap_lichtfeld_max_width(lf_cfg: dict, scaling: dict | None = None) -> tuple[int, int]:
    configured = int(lf_cfg.get("max_width") or 0)
    capped = min(configured, LICHTFELD_MAX_WIDTH_LIMIT) if configured > 0 else LICHTFELD_MAX_WIDTH_LIMIT
    lf_cfg["max_width"] = capped
    if configured > LICHTFELD_MAX_WIDTH_LIMIT and scaling is not None:
        scaling["max_width_capped_from"] = configured
        scaling["max_width_limit"] = LICHTFELD_MAX_WIDTH_LIMIT
        scaling.setdefault("rules", []).append(f"max-width capped: {configured} -> {LICHTFELD_MAX_WIDTH_LIMIT}")
    return configured, capped


def map_colmap_preset(value: str) -> str:
    p = (value or "").strip().lower()
    alias = {
        "high": "hq",
        "good": "standard",
    }
    p = alias.get(p, p)
    if p in {"standard", "standard_safe", "hq_safe", "hq"}:
        return p
    return "standard"


# --- COLMAP helpers -------------------------------------------------
def _is_gpu_related_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    markers = (
        "cuda", "out of memory", "memory", "out of mem", "cudaerror",
        "allocation", "failed to allocate", "memory exhausted"
    )
    return any(marker in text for marker in markers)


def _estimate_prepared_image_count(job: dict, image_dir: Path) -> int:
    try:
        files = [p for p in image_dir.glob("frame_*") if p.is_file()]
        if files:
            return len(files)
    except Exception:
        pass
    counts = job.get("input", {}).get("counts", {})
    return int(counts.get("prepared_images") or counts.get("photos") or 0)


def _folder_total_bytes(folder: Path) -> int:
    total = 0
    if not folder.exists():
        return total
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def _preprocess_safety_report(src: Path, counts: dict, preset: str, stage: str, prepared_images: int | None = None) -> dict:
    normalized = map_colmap_preset(preset)
    max_photos = PREPROCESS_MAX_PHOTOS_HQ if normalized == "hq" else PREPROCESS_MAX_PHOTOS_STANDARD
    max_bytes = PREPROCESS_MAX_BYTES_HQ if normalized == "hq" else PREPROCESS_MAX_BYTES_STANDARD
    photos = int(prepared_images if prepared_images is not None else counts.get("photos") or 0)
    total_bytes = _folder_total_bytes(src)
    reasons = []
    if photos > max_photos:
        reasons.append(f"{photos} images exceeds safe limit {max_photos}")
    if total_bytes > max_bytes:
        reasons.append(f"{total_bytes} bytes exceeds safe limit {max_bytes}")
    return {
        "stage": stage,
        "classification": "too_large_for_safe_preprocessing" if reasons else "normal",
        "reason": "; ".join(reasons) if reasons else None,
        "counts": dict(counts or {}),
        "prepared_images": prepared_images,
        "total_bytes": total_bytes,
        "preset": normalized,
        "limits": {"photos": max_photos, "bytes": max_bytes},
    }


def _record_preprocess_safety(job: dict, report: dict):
    job.setdefault("assessment", {})
    safety = job["assessment"].setdefault("worker_preprocess_safety", {})
    if not isinstance(safety, dict) or "classification" in safety:
        safety = {}
    safety[str(report.get("stage") or "unknown")] = report
    job["assessment"]["worker_preprocess_safety"] = safety
    job.setdefault("input", {}).setdefault("counts", {})
    job["input"]["counts"]["total_bytes"] = int(report.get("total_bytes") or 0)


def _cpu_fallback_allowed(job: dict, image_dir: Path, normalized_preset: str) -> tuple[bool, str]:
    image_count = _estimate_prepared_image_count(job, image_dir)
    limit = CPU_FALLBACK_MAX_IMAGES_HQ if normalized_preset == "hq" else CPU_FALLBACK_MAX_IMAGES_STANDARD
    if image_count <= 0:
        return True, "image count unknown"
    if image_count > limit:
        return False, f"CPU fallback skipped: {image_count} prepared images exceeds safe threshold {limit}."
    return True, f"CPU fallback allowed for {image_count} prepared images (limit={limit})."


def _count_registered_images_from_text_model(images_txt: Path) -> int:
    if not images_txt.exists():
        return 0
    meaningful = []
    for line in images_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        meaningful.append(stripped)
    if not meaningful:
        return 0
    return len(meaningful) // 2


def _read_registration_summary(colmap_workspace: Path) -> dict:
    summary_path = colmap_workspace / "registration_summary.json"
    summary: dict = {}
    if summary_path.exists():
        try:
            summary = load_json(summary_path)
        except Exception:
            summary = {}

    text_model_dir = None
    for candidate in [
        summary.get("text_model_dir") if isinstance(summary, dict) else None,
        str(colmap_workspace / "_model_txt"),
    ]:
        if candidate:
            p = Path(candidate)
            if p.exists():
                text_model_dir = p
                break

    registered = int(summary.get("registered_images") or 0) if isinstance(summary, dict) else 0
    if registered <= 0 and text_model_dir is not None:
        registered = _count_registered_images_from_text_model(text_model_dir / "images.txt")

    return {
        "path": str(summary_path),
        "exists": summary_path.exists(),
        "registered_images": registered,
        "raw": summary if isinstance(summary, dict) else {},
        "text_model_dir": str(text_model_dir) if text_model_dir else None,
    }


def run_colmap_with_fallback(job_path: Path, image_dir: Path, workspace: Path, requested_preset: str) -> Path:
    """
    GPU/memory fallback only.
    Sparse strategy retries now belong inside colmap_runner.py itself.
    Ordered attempts here:
      hq -> hq_safe -> standard_safe -> standard_safe CPU (only if workload is sane)
      standard -> standard_safe -> standard_safe CPU (only if workload is sane)
    """
    job_folder = job_path.parent
    job = ensure_job_shape(load_json(job_path))
    normalized_preset = map_colmap_preset(requested_preset)

    try:
        import colmap_runner  # type: ignore
        available_presets = set(colmap_runner.COLMAP_PRESET_ARGS.keys())
    except Exception:
        available_presets = {"standard", "standard_safe", "hq", "hq_safe"}

    safe_variant = f"{normalized_preset}_safe" if f"{normalized_preset}_safe" in available_presets else "standard_safe"

    def gpu_free_mb() -> int:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode != 0:
                return 0
            vals = [int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
            return max(vals) if vals else 0
        except Exception:
            return 0

    def record_attempt(preset_name: str, force_cpu: bool, success: bool, note: str = ""):
        latest = ensure_job_shape(load_json(job_path))
        meta = latest.setdefault("meta", {})
        hist = meta.setdefault("fallback_history", [])
        hist.append({
            "attempt_index": len(hist) + 1,
            "preset": preset_name,
            "force_cpu": bool(force_cpu),
            "ts": iso_now(),
            "success": bool(success),
            "note": note,
            "gpu_free_mb": gpu_free_mb(),
        })
        latest["meta"] = meta
        save_json_atomic(job_path, latest)

    def _attempt(preset_name: str, force_cpu: bool = False) -> Path:
        log(job_folder, f"COLMAP attempt: preset={preset_name}, force_cpu={force_cpu}")
        return run_colmap_pipeline(
            image_dir=image_dir,
            workspace=workspace,
            preset=preset_name,
            input_type=job.get("input", {}).get("type", "photoset"),
            colmap_bin=COLMAP_BIN or os.environ.get("COLMAP_BIN") or None,
            force_cpu=force_cpu,
        )

    attempts: list[tuple[str, bool]] = []
    attempts.append((normalized_preset, False))
    if safe_variant != normalized_preset:
        attempts.append((safe_variant, False))
    if safe_variant != "standard_safe":
        attempts.append(("standard_safe", False))

    allowed_cpu, cpu_reason = _cpu_fallback_allowed(job, image_dir, normalized_preset)
    log(job_folder, cpu_reason)
    if allowed_cpu:
        attempts.append(("standard_safe", True))

    free_mb = gpu_free_mb()
    if normalized_preset == "hq" and free_mb and free_mb < 12000:
        log(job_folder, f"GPU free {free_mb}MB < 12000MB: skip direct hq attempt and start with safer GPU preset.")
        attempts = [a for a in attempts if a != (normalized_preset, False)]

    attempts = list(dict.fromkeys(attempts))

    last_exc = None
    for preset_name, force_cpu in attempts:
        try:
            fused = _attempt(preset_name, force_cpu=force_cpu)
            latest = ensure_job_shape(load_json(job_path))
            latest["preset_used"] = preset_name
            latest["result_summary"] = (
                latest.get("result_summary")
                or f"COLMAP succeeded with preset={preset_name} force_cpu={force_cpu}."
            )
            save_json_atomic(job_path, latest)
            record_attempt(preset_name, force_cpu, success=True)
            return fused
        except Exception as exc:
            last_exc = exc
            record_attempt(preset_name, force_cpu, success=False, note=str(exc))
            if not _is_gpu_related_error(exc):
                log(job_folder, f"COLMAP attempt failed without GPU/memory signature: {exc}")
                raise
            log(job_folder, f"COLMAP GPU/memory-related failure on preset={preset_name} force_cpu={force_cpu}: {exc}")
            time.sleep(3)

    raise RuntimeError(f"COLMAP fallback exhausted. Last error: {last_exc}") from last_exc


def find_best_artifact(out_dir: Path) -> Path:
    for ext in [".ply", ".spz", ".sog", ".resume"]:
        cands = list(out_dir.rglob(f"*{ext}"))
        if cands:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0]
    raise RuntimeError("No LichtFeld output artifact found.")


def run_lichtfeld(job_folder: Path, job: dict, dense_dir: Path, preset: str) -> Path:
    requested_preset, lf_cfg, scaling = resolve_effective_preset(job)
    out_dir = lichtfeld_out_dir(job["job_id"])
    job.setdefault("artifacts", {})
    job["artifacts"]["preset_requested"] = requested_preset
    job["artifacts"]["preset_effective"] = scaling["effective"]
    job["artifacts"]["preset_scaling"] = scaling
    if scaling.get("max_width_capped_from"):
        log(job_folder, f"LichtFeld max-width capped: {scaling['max_width_capped_from']} -> {lf_cfg['max_width']}")
    cmd = [
        str(LICHTFELD_EXE),
        "--data-path", str(dense_dir),
        "--output-path", str(out_dir),
        "--images", "images",
        "--iter", str(lf_cfg["iter"]),
        "--strategy", str(lf_cfg["strategy"]),
        "--tile-mode", str(lf_cfg["tile_mode"]),
        "--resize_factor", str(lf_cfg["resize_factor"]),
        "--max-width", str(lf_cfg["max_width"]),
        "--max-cap", str(lf_cfg["max_cap"]),
        "--headless",
        "--log-level", "info",
        "--log-file", str(out_dir / "lichtfeld.log"),
    ] + list(lf_cfg["extra_flags"])
    rc, _, _ = run_cmd(job_folder, cmd)
    if rc != 0:
        raise RuntimeError("LichtFeld failed.")
    artifact = find_best_artifact(out_dir)
    if artifact.stat().st_size == 0:
        raise RuntimeError("LichtFeld produced an empty artifact.")
    return artifact


def write_summary_file(job: dict, dest: Path, success: bool):
    stage = job.get("failed_stage") or "none"
    lines = [
        "GS Pipeline summary",
        "",
        f"Job ID: {job.get('job_id', '')}",
        f"Owner: {owner_of(job)}",
        f"Tag: {job.get('input', {}).get('tag', '')}",
        f"State: {job.get('state', '')}",
        f"Result: {'SUCCESS' if success else 'FAILED'}",
        f"Preset: {job.get('preset_used') or job.get('preset') or ''}",
        f"Started: {job.get('started_at') or ''}",
        f"Finished: {job.get('finished_at') or ''}",
        f"Failed at: {job.get('failed_at') or ''}",
        f"Failed stage: {stage}",
        f"Registered images: {job.get('registered_images') or ''}",
        f"Artifact: {job.get('result_artifact') or ''}",
        "",
    ]
    if job.get("error"):
        lines.extend(["Error:", job["error"], ""])
    if job.get("result_summary"):
        lines.extend(["Summary:", job["result_summary"], ""])
    (dest / ("summary.txt" if success else "error.txt")).write_text("\n".join(lines), encoding="utf-8")


def copy_if_exists(src: Path, dest: Path):
    if src.exists() and src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))


def deliver_job_atomically(job_folder: Path, job: dict):
    tag = job.get("input", {}).get("tag", "project")
    owner = owner_of(job)
    job_id = job.get("job_id", job_folder.name)
    project_dir = outbox_project_dir(owner, tag)
    final_dest = project_dir / job_id
    temp_dest = project_dir / f".{job_id}.tmp"
    if temp_dest.exists():
        shutil.rmtree(temp_dest, ignore_errors=True)
    temp_dest.mkdir(parents=True, exist_ok=True)
    if final_dest.exists():
        archived = project_dir / f"{job_id}__previous_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        final_dest.replace(archived)
        log(job_folder, f"Bestaande outbox-output gearchiveerd: {final_dest} -> {archived}")
    success = job.get("state") == "done"
    artifact_path = Path(job["result_artifact"]) if job.get("result_artifact") else None
    if success and artifact_path and artifact_path.exists():
        delivered_artifact = final_dest / f"model{artifact_path.suffix.lower()}"
        copy_if_exists(artifact_path, temp_dest / delivered_artifact.name)
        job["result_artifact"] = str(delivered_artifact)
    job.setdefault("delivery", {})
    job["delivery"]["outbox_path"] = str(final_dest)
    job["delivery"]["delivered_at"] = iso_now()
    job["delivery"]["mode"] = "atomic_rename"
    copy_if_exists(job_folder / "worker.log", temp_dest / "worker.log")
    copy_if_exists(job_folder / "worker_subprocess.log", temp_dest / "worker_subprocess.log")
    persist_job(temp_dest / "job.json", job)
    lf_log = Path(job.get("artifacts", {}).get("lichtfeld_log", "")) if job.get("artifacts") else None
    if lf_log:
        copy_if_exists(lf_log, temp_dest / "lichtfeld.log")
    write_summary_file(job, temp_dest, success=success)
    temp_dest.replace(final_dest)
    return final_dest


def copy_final_artifacts_into_job(job_folder: Path, job: dict):
    final_dir = job_folder / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    artifact = Path(job.get("artifacts", {}).get("lichtfeld_artifact", ""))
    if artifact.exists():
        local_artifact = final_dir / f"model{artifact.suffix.lower()}"
        shutil.copy2(str(artifact), str(local_artifact))
        job["result_artifact"] = str(local_artifact)
    lf_log = Path(job.get("artifacts", {}).get("lichtfeld_log", ""))
    if lf_log.exists():
        copy_if_exists(lf_log, final_dir / "lichtfeld.log")


def _archive_copy_fallback(job_folder: Path, dest: Path):
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(
        job_folder,
        dest,
        ignore=shutil.ignore_patterns("worker_subprocess.log", "worker_started"),
        dirs_exist_ok=False,
    )


def archive_job_folder(job_folder: Path, job: dict) -> tuple[bool, str]:
    dest = archive_dir_for_state(job.get("state", "failed")) / job_folder.name
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    try:
        shutil.move(str(job_folder), str(dest))
        return True, str(dest)
    except PermissionError as exc:
        try:
            copy_dest = dest.parent / f"{dest.name}__copied"
            _archive_copy_fallback(job_folder, copy_dest)
            return False, f"archive move blocked by locked file; copied fallback to {copy_dest} ({exc})"
        except Exception as copy_exc:
            return False, f"archive move failed: {exc}; archive copy fallback also failed: {copy_exc}"
    except Exception as exc:
        return False, f"archive move failed: {exc}"


def cleanup_job_folder(job_folder: Path, job: dict):
    finished_at = parse_iso(job.get("finished_at"))
    if not finished_at:
        return
    if now_dt() - finished_at < timedelta(days=CLEANUP_AFTER_DAYS):
        return
    removed = []
    for target in [job_folder / "staging" / "frames", job_folder / "colmap" / "dense", job_folder / "checkpoints"]:
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
            removed.append(str(target))
    external_output = OUTPUT / job.get("job_id", "")
    if external_output.exists():
        shutil.rmtree(external_output, ignore_errors=True)
        removed.append(str(external_output))
    if removed:
        (CLEANUP_QUEUE / f"{job.get('job_id', job_folder.name)}-cleanup.txt").write_text(
            "Cleanup completed on " + iso_now() + "\n\n" + "\n".join(removed),
            encoding="utf-8",
        )


def run_retention_cleanup():
    cleanup_expired_status_files()
    for root in [ARCHIVE_DONE, ARCHIVE_FAILED]:
        if not root.exists():
            continue
        for job_folder in root.iterdir():
            job_json = job_folder / "job.json"
            if not job_folder.is_dir() or not job_json.exists():
                continue
            try:
                job = ensure_job_shape(load_json(job_json))
                cleanup_job_folder(job_folder, job)
            except Exception:
                continue


def _call_preprocess_photoset(job_folder: Path, src: Path, preset: str) -> int:
    try:
        return preprocess_photoset(job_folder, src, preset)
    except TypeError:
        return preprocess_photoset(job_folder, src)


def run_preprocessing(job_folder: Path, job_path: Path, job: dict):
    src = Path(job.get("input", {}).get("primary", ""))
    preset = map_colmap_preset(job.get("preset") or job.get("preset_used") or "standard")
    set_state(job_path, job, "preprocessing")
    unzip_if_needed(job_folder, src)
    counts_after_unzip = detect_counts(src)
    input_type = detect_input_type(src)
    if input_type == "archive":
        raise RuntimeError("Zip-bestanden zijn uitgepakt, maar er is nog geen bruikbare inhoud gevonden.")
    if input_type == "unknown":
        raise RuntimeError("Input is leeg of gemengd na uitpakken. Lever alleen video of alleen foto's aan per project.")
    job["input"]["type"] = input_type
    job["input"]["counts"].update(counts_after_unzip)
    job["input"]["counts"]["video_count"] = int(job["input"]["counts"].get("videos", 0))
    unzip_safety = _preprocess_safety_report(src, counts_after_unzip, preset, "after_unzip")
    _record_preprocess_safety(job, unzip_safety)
    persist_job(job_path, job)
    if unzip_safety["classification"] == "too_large_for_safe_preprocessing":
        raise RuntimeError(f"Input too large for safe preprocessing after unzip: {unzip_safety['reason']}")

    if input_type == "video":
        count = extract_frames_from_video(job_folder, src, preset)
    elif input_type == "photoset":
        count = _call_preprocess_photoset(job_folder, src, preset)
    else:
        raise RuntimeError(f"Unsupported input type: {input_type}")

    summary_path = job_folder / "preprocess" / "summary.json"
    gps_path = job_folder / "preprocess" / "gps_summary.json"
    if summary_path.exists():
        job["artifacts"]["preprocess_summary"] = str(summary_path)
        try:
            summary = load_json(summary_path)
            final_count = int(summary.get("final_count") or count)
            job["input"]["counts"]["photos"] = final_count
            job["input"]["counts"]["prepared_images"] = final_count
            job["meta"] = job.get("meta", {})
            job["meta"]["preprocess"] = {
                "input_type": summary.get("input_type"),
                "preset": summary.get("preset"),
                "final_count": final_count,
                "source_count": summary.get("source_count"),
                "staging_mode": summary.get("staging_mode"),
                "blur_removed": (summary.get("blur") or {}).get("removed"),
                "dedupe_removed": (summary.get("dedupe") or {}).get("purged_count"),
            }
        except Exception as exc:
            log(job_folder, f"Kon preprocess summary niet lezen: {exc}")
            job["input"]["counts"]["photos"] = count
            job["input"]["counts"]["prepared_images"] = count
    else:
        job["input"]["counts"]["photos"] = count
        job["input"]["counts"]["prepared_images"] = count

    if gps_path.exists():
        job["artifacts"]["preprocess_gps_summary"] = str(gps_path)

    final_count = int(job["input"]["counts"].get("prepared_images") or job["input"]["counts"].get("photos") or count)
    final_safety = _preprocess_safety_report(src, job["input"]["counts"], preset, "after_preprocessing", prepared_images=final_count)
    _record_preprocess_safety(job, final_safety)
    job.setdefault("meta", {}).setdefault("preprocess", {})["safety"] = final_safety
    if final_safety["classification"] == "too_large_for_safe_preprocessing":
        persist_job(job_path, job)
        raise RuntimeError(f"Input too large for safe preprocessing after preprocessing: {final_safety['reason']}")

    job["result_summary"] = f"Preprocessing complete: {job['input']['counts'].get('photos', count)} images ready for training."
    persist_job(job_path, job)
    set_state(job_path, job, "ready_for_training")


def run_colmap_stage(job_folder: Path, job_path: Path, job: dict):
    preset = map_colmap_preset(job.get("preset") or job.get("preset_used") or "standard")

    set_state(job_path, job, "colmap_running")

    frames = frames_dir(job_folder)
    total_frames = len([p for p in frames.glob("frame_*") if p.is_file()])
    colmap_workspace = job_folder / "colmap"
    colmap_workspace.mkdir(parents=True, exist_ok=True)
    fused_ply = run_colmap_with_fallback(job_path, frames, colmap_workspace, preset)

    dense_dir = colmap_workspace / "dense"
    if not dense_dir.exists():
        alt = colmap_workspace / "stereo"
        if alt.exists():
            log(job_folder, f"COLMAP dense dir ontbreekt; gebruik stereo dir als dense_dir: {alt}")
            dense_dir = alt
        else:
            raise RuntimeError("COLMAP dense output missing for LichtFeld stage.")

    registration = _read_registration_summary(colmap_workspace)
    reg = int(registration.get("registered_images") or 0)
    if registration.get("exists"):
        job["artifacts"]["colmap_registration_summary"] = registration["path"]
    if registration.get("text_model_dir"):
        job["artifacts"]["colmap_text_model_dir"] = registration["text_model_dir"]

    latest_job = ensure_job_shape(load_json(job_path))
    used_preset = latest_job.get("preset_used") or preset
    if used_preset != preset:
        log(job_folder, f"COLMAP fallback toegepast: gevraagd={preset}, gebruikt={used_preset}")

    ratio = round((reg / total_frames), 4) if total_frames else None
    job["artifacts"]["colmap_dense_dir"] = str(dense_dir)
    job["artifacts"]["colmap_fused_ply"] = str(fused_ply)
    job["artifacts"]["colmap_registered_images"] = reg
    job["artifacts"]["colmap_input_frames"] = total_frames
    if ratio is not None:
        job["artifacts"]["colmap_registration_ratio"] = ratio
    job.setdefault("meta", {})
    if isinstance(registration.get("raw"), dict):
        raw = registration["raw"]
        selected_matcher = raw.get("selected_matcher") or raw.get("chosen_matcher")
        selected_attempt = raw.get("selected_attempt") or raw.get("best_attempt")
        if selected_matcher:
            job["artifacts"]["colmap_selected_matcher"] = selected_matcher
            job["artifacts"]["colmap_chosen_matcher"] = selected_matcher
        if selected_attempt:
            job["meta"]["colmap_selected_attempt"] = selected_attempt
            job["meta"]["colmap_best_attempt"] = selected_attempt
        if raw.get("attempts"):
            job["meta"]["colmap_attempts"] = raw.get("attempts")

    job["registered_images"] = reg
    job["meta"]["colmap_preset"] = used_preset
    job["preset_used"] = used_preset

    if ratio is not None and ratio < LOW_REGISTRATION_WARNING_RATIO:
        log(job_folder, f"Waarschuwing: lage COLMAP registratie-ratio ({reg}/{total_frames} = {ratio:.2%}). Dit kan ghosting of dubbele gevels veroorzaken.")

    job["result_summary"] = f"COLMAP complete: {reg} registered images from {total_frames} prepared frames."
    persist_job(job_path, job)

    if reg < MIN_REGISTERED_IMAGES:
        raise RuntimeError(f"COLMAP registered only {reg} images (<{MIN_REGISTERED_IMAGES}).")

    set_state(job_path, job, "colmap_done")


def run_lichtfeld_stage(job_folder: Path, job_path: Path, job: dict):
    dense_dir = Path(job.get("artifacts", {}).get("colmap_dense_dir", ""))
    if not dense_dir.exists():
        raise RuntimeError("COLMAP dense output missing for LichtFeld stage.")
    set_state(job_path, job, "lichtfeld_running")
    artifact = run_lichtfeld(job_folder, job, dense_dir, job.get("preset") or "standard")
    job["artifacts"]["lichtfeld_artifact"] = str(artifact)
    job["artifacts"]["lichtfeld_log"] = str(lichtfeld_out_dir(job["job_id"]) / "lichtfeld.log")
    copy_final_artifacts_into_job(job_folder, job)
    job["result"] = "success"
    job["result_summary"] = "Training completed and final artifact prepared for delivery."
    persist_job(job_path, job)
    set_state(job_path, job, "done")


def finalize_terminal_job(job_folder: Path, job_path: Path, job: dict):
    delivery = job.setdefault("delivery", {})
    if delivery.get("finalized"):
        return
    outbox_path = deliver_job_atomically(job_folder, job)
    if job_path.exists():
        persist_job(job_path, job)
    archived, archive_note = archive_job_folder(job_folder, job)
    delivery["finalized"] = True
    delivery["archive_status"] = "archived" if archived else "not_archived"
    delivery["archive_note"] = archive_note
    job["delivery"] = delivery
    persist_job(outbox_path / "job.json", job)
    write_summary_file(job, outbox_path, success=job.get("state") == "done")
    # Persist only if the job folder still exists at original location.
    if job_path.exists():
        persist_job(job_path, job)
        if not archived:
            log(job_folder, f"Archive note: {archive_note}")
    elif archived:
        archived_job_path = Path(archive_note) / "job.json"
        if archived_job_path.parent.exists():
            persist_job(archived_job_path, job)


def normalize_resume_state(job_folder: Path, job_path: Path, job: dict):
    state = job.get("state", "queued")
    if state == "colmap_running":
        log(job_folder, "Resume detected: restarting COLMAP stage from beginning.")
        job["state"] = "ready_for_training"
        persist_job(job_path, job)
    elif state == "lichtfeld_running":
        log(job_folder, "Resume detected: restarting LichtFeld stage from beginning.")
        job["state"] = "colmap_done"
        persist_job(job_path, job)
    elif state == "preprocessing":
        log(job_folder, "Resume detected: restarting preprocessing stage from beginning.")
        job["state"] = "queued"
        persist_job(job_path, job)


def process_job(job_folder: Path):
    job_path = job_folder / "job.json"
    job = ensure_job_shape(load_json(job_path))
    persist_job(job_path, job)
    if job.get("state") in TERMINAL_STATES and job.get("delivery", {}).get("finalized"):
        return job
    if job.get("state") in TERMINAL_STATES:
        finalize_terminal_job(job_folder, job_path, job)
        return job
    normalize_resume_state(job_folder, job_path, job)
    current_state = job.get("state", "queued")
    try:
        if current_state == "queued":
            log(job_folder, f"Picked up job {job.get('job_id', job_folder.name)}")
            run_preprocessing(job_folder, job_path, job)
            current_state = job.get("state")
        if current_state == "ready_for_training":
            run_colmap_stage(job_folder, job_path, job)
            current_state = job.get("state")
        if current_state == "colmap_done":
            run_lichtfeld_stage(job_folder, job_path, job)
            current_state = job.get("state")
        if current_state == "done":
            finalize_terminal_job(job_folder, job_path, job)
        return job
    except Exception as exc:
        stage = {
            "queued": "preprocessing",
            "preprocessing": "preprocessing",
            "ready_for_training": "colmap",
            "colmap_running": "colmap",
            "colmap_done": "lichtfeld",
            "lichtfeld_running": "lichtfeld",
        }.get(job.get("state", current_state), current_state)
        fail_job(job_path, job, stage, exc)
        finalize_terminal_job(job_folder, job_path, job)
        return job


def main_loop():
    print("GS Worker running...")
    ensure_dirs()
    while True:
        try:
            run_retention_cleanup()
            for job_folder in list_jobs_sorted():
                process_job(job_folder)
        except Exception as e:
            print(f"[Worker] ERROR: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main_loop()
