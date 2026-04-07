import json
import os
import shutil
import subprocess
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
from utils.colmap_profiler import choose_preset
from utils.gpu_monitor import gpu_free_mb, monitor_peak
from utils.chunker import chunk_images, merge_colmap_models
from utils.config import get_config

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
MIN_REGISTERED_IMAGES = 30
CLEANUP_AFTER_DAYS = 7
MAX_LOG_TAIL = 6000
STATUS_RETENTION_HOURS = 24
CHUNK_THRESHOLD = int(cfg.get("CHUNK_THRESHOLD", 1500))

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
        "iter": 32000,
        "strategy": "mcmc",
        "tile_mode": 1,
        "resize_factor": "auto",
        "max_width": 3840,
        "max_cap": 1000000,
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
    if "fallback geprobeerd" in text:
        return (
            "De reconstructie faalde ook na veilige terugval-instellingen.",
            "We hebben automatisch een lichtere preset en daarna CPU-modus geprobeerd, maar zonder stabiel resultaat.",
            "Probeer minder beelden of lagere kwaliteit; neem contact op als je wilt dat we de logs analyseren.",
        )
    if "memory" in text or "cuda" in text or "out of memory" in text:
        return (
            "De verwerking vroeg meer geheugen dan nu beschikbaar is.",
            "Deze dataset of kwaliteitsinstelling is te zwaar voor de huidige machine-instelling.",
            "Gebruik minder input of kies een lichtere instelling. Neem contact op als je wilt dat we meekijken.",
        )
    if "registered only" in text or "colmap" in text:
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
    job["preset_used"] = job.get("preset") or job.get("preset_used") or "standard"
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


def colmap_paths(job_folder: Path):
    colmap_dir = job_folder / "colmap"
    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    dense = colmap_dir / "dense"
    return colmap_dir, db, sparse, dense


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
    return requested, cfg, scaling


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



# --- BEGIN: COLMAP fallback wrapper ---
def _is_gpu_related_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    markers = ("cuda", "out of memory", "memory", "out of mem", "cudaerror")
    return any(marker in text for marker in markers)


def run_colmap_with_fallback(job_path: Path, image_dir: Path, workspace: Path, requested_preset: str) -> Path:
    """
    Run COLMAP with fallback: requested preset -> standard_safe -> standard_safe + CPU-only flags.
    """
    job_folder = job_path.parent
    job = ensure_job_shape(load_json(job_path))
    input_type = job.get("input", {}).get("type", "photoset")
    normalized_preset = map_colmap_preset(requested_preset)
    safe_preset = f"{normalized_preset}_safe" if normalized_preset in {"hq", "standard"} else "standard_safe"
    if safe_preset not in {"hq_safe", "standard_safe"}:
        safe_preset = "standard_safe"

    def _record_event(preset_name: str, force_cpu: bool, success: bool, note: str, elapsed: float):
        meta = job.setdefault("meta", {})
        hist = meta.setdefault("fallback_history", [])
        dims = choose_preset(job_folder)
        event = {
            "preset": preset_name,
            "force_cpu": force_cpu,
            "ts": iso_now(),
            "success": success,
            "note": note,
            "num_images": dims.get("num_images", 0),
            "max_side": dims.get("max_side", 0),
            "gpu_free_at_start": gpu_free_mb(),
            "gpu_peak_used": monitor_peak(os.getpid(), interval=0.2, timeout=0.6),
            "time_taken": round(elapsed, 3),
        }
        hist.append(event)

    def _run_attempt(preset_name: str, force_cpu: bool = False) -> Path:
        started = time.time()
        log(job_folder, f"COLMAP attempt: preset={preset_name}, force_cpu={force_cpu}")
        try:
            fused = run_colmap_pipeline(
                image_dir=image_dir,
                workspace=workspace,
                preset=preset_name,
                input_type=input_type,
                colmap_bin=COLMAP_BIN or os.environ.get("COLMAP_BIN") or None,
                force_cpu=force_cpu,
            )
            _record_event(preset_name, force_cpu, True, "attempt-ok", time.time() - started)
            return fused
        except Exception as exc:
            _record_event(preset_name, force_cpu, False, str(exc), time.time() - started)
            raise

    try:
        fused = _run_attempt(normalized_preset, force_cpu=False)
        job["preset_used"] = normalized_preset
        persist_job(job_path, job)
        return fused
    except Exception as first_exc:
        if not _is_gpu_related_error(first_exc):
            raise
        log(job_folder, f"COLMAP GPU/memory fout gedetecteerd. Fallback naar {safe_preset}. Details: {first_exc}")
        write_user_status(
            job,
            "Camera locaties bepalen",
            note="GPU-limiet geraakt; we schakelen over op veilige instellingen (kan trager zijn).",
        )

    try:
        fused = _run_attempt(safe_preset, force_cpu=False)
        job["preset_used"] = safe_preset
        job["result_summary"] = f"{job.get('result_summary') or ''} COLMAP fallback gebruikt: {normalized_preset} -> {safe_preset}.".strip()
        persist_job(job_path, job)
        write_user_status(
            job,
            "Camera locaties bepalen",
            note=f"Fallback actief: preset verlaagd naar {safe_preset} (iets trager, mogelijk iets lagere kwaliteit).",
        )
        return fused
    except Exception as second_exc:
        log(job_folder, f"COLMAP veilige preset mislukte, probeer CPU-only fallback. Details: {second_exc}")

    try:
        fused = _run_attempt(safe_preset, force_cpu=True)
        job["preset_used"] = safe_preset
        job["result_summary"] = (
            f"{job.get('result_summary') or ''} COLMAP fallback gebruikt: {normalized_preset} -> {safe_preset} -> CPU-only."
        ).strip()
        persist_job(job_path, job)
        write_user_status(
            job,
            "Camera locaties bepalen",
            note="Fallback actief: CPU-modus ingeschakeld (duidelijk trager, mogelijk lagere detailkwaliteit).",
        )
        return fused
    except Exception as final_exc:
        log(job_folder, f"COLMAP fallback volledig mislukt (incl. CPU-only): {final_exc}")
        raise RuntimeError(f"COLMAP fallback geprobeerd (safe + CPU), maar mislukt: {final_exc}") from final_exc
# --- END: COLMAP fallback wrapper ---
def run_colmap(job_folder: Path, frames: Path, input_type: str, preset: str):
    colmap_workspace = job_folder / "colmap"
    colmap_workspace.mkdir(parents=True, exist_ok=True)
    colmap_bin = COLMAP_BIN or os.environ.get("COLMAP_BIN") or None
    effective_preset = map_colmap_preset(preset)
    log(job_folder, f"COLMAP start: preset={effective_preset}, input_type={input_type}, colmap_bin={colmap_bin or 'colmap'}")
    fused_ply = run_colmap_pipeline(
        image_dir=frames,
        workspace=colmap_workspace,
        preset=effective_preset,
        input_type=input_type,
        colmap_bin=colmap_bin,
    )
    dense_dir = colmap_workspace / "dense"
    reg = len(list(frames.glob("frame_*.jpg")))
    log(job_folder, f"COLMAP klaar: fused={fused_ply}, registered_images_estimate={reg}")
    return dense_dir, reg, fused_ply


def find_best_artifact(out_dir: Path) -> Path:
    for ext in [".ply", ".spz", ".sog", ".resume"]:
        cands = list(out_dir.rglob(f"*{ext}"))
        if cands:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0]
    raise RuntimeError("No LichtFeld output artifact found.")


def run_lichtfeld(job_folder: Path, job: dict, dense_dir: Path, preset: str) -> Path:
    requested_preset, cfg, scaling = resolve_effective_preset(job)
    out_dir = lichtfeld_out_dir(job["job_id"])
    job.setdefault("artifacts", {})
    job["artifacts"]["preset_requested"] = requested_preset
    job["artifacts"]["preset_effective"] = scaling["effective"]
    job["artifacts"]["preset_scaling"] = scaling
    cmd = [
        str(LICHTFELD_EXE),
        "--data-path", str(dense_dir),
        "--output-path", str(out_dir),
        "--images", "images",
        "--iter", str(cfg["iter"]),
        "--strategy", str(cfg["strategy"]),
        "--tile-mode", str(cfg["tile_mode"]),
        "--resize_factor", str(cfg["resize_factor"]),
        "--max-width", str(cfg["max_width"]),
        "--max-cap", str(cfg["max_cap"]),
        "--headless",
        "--log-level", "info",
        "--log-file", str(out_dir / "lichtfeld.log"),
    ] + list(cfg["extra_flags"])
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
        copy_if_exists(artifact_path, temp_dest / f"model{artifact_path.suffix.lower()}")
    copy_if_exists(job_folder / "worker.log", temp_dest / "worker.log")
    copy_if_exists(job_folder / "job.json", temp_dest / "job.json")
    lf_log = Path(job.get("artifacts", {}).get("lichtfeld_log", "")) if job.get("artifacts") else None
    if lf_log:
        copy_if_exists(lf_log, temp_dest / "lichtfeld.log")
    write_summary_file(job, temp_dest, success=success)
    temp_dest.replace(final_dest)
    job.setdefault("delivery", {})
    job["delivery"]["outbox_path"] = str(final_dest)
    job["delivery"]["delivered_at"] = iso_now()
    job["delivery"]["mode"] = "atomic_rename"


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


def archive_job_folder(job_folder: Path, job: dict):
    dest = archive_dir_for_state(job.get("state", "failed")) / job_folder.name
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(job_folder), str(dest))


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
    if input_type == "video":
        count = extract_frames_from_video(job_folder, src, preset)
        job["input"]["counts"]["photos"] = count
    elif input_type == "photoset":
        count = preprocess_photoset(job_folder, src)
        job["input"]["counts"]["photos"] = count
        bad_dir = src / "_bad"
        if bad_dir.exists():
            bad_files = sorted([p.name for p in bad_dir.glob("*") if p.is_file()])
            if bad_files:
                job.setdefault("meta", {})
                job["meta"]["preprocess_bad_files"] = bad_files
    else:
        raise RuntimeError(f"Unsupported input type: {input_type}")
    job["result_summary"] = f"Preprocessing complete: {count} images ready for training."
    persist_job(job_path, job)
    set_state(job_path, job, "ready_for_training")


def run_colmap_stage(job_folder: Path, job_path: Path, job: dict):
    prof = choose_preset(job_folder)
    preset = map_colmap_preset(prof.get("preset") or job.get("preset") or job.get("preset_used") or "standard")
    job.setdefault("meta", {})
    job["meta"]["preset_chosen"] = prof
    job["preset_used"] = preset
    persist_job(job_path, job)

    set_state(job_path, job, "colmap_running")

    frames = frames_dir(job_folder)
    colmap_workspace = job_folder / "colmap"
    colmap_workspace.mkdir(parents=True, exist_ok=True)
    frame_count = len(list(frames.glob("frame_*.jpg")))
    if frame_count > CHUNK_THRESHOLD:
        chunk_root = job_folder / "chunks"
        chunk_dirs = chunk_images(frames, chunk_root, chunk_size=500, overlap=50)
        model_dirs = []
        for idx, chunk in enumerate(chunk_dirs, start=1):
            chunk_workspace = chunk / "workspace"
            chunk_frames = chunk / "images"
            log(job_folder, f"Chunk COLMAP run {idx}/{len(chunk_dirs)}: {chunk_frames}")
            run_colmap_with_fallback(job_path, chunk_frames, chunk_workspace, preset)
            model_dirs.append(chunk)
        fused_ply = merge_colmap_models(model_dirs, colmap_workspace)
    else:
        fused_ply = run_colmap_with_fallback(job_path, frames, colmap_workspace, preset)
    dense_dir = colmap_workspace / "dense"
    reg = len(list(frames.glob("frame_*.jpg")))
    latest_job = ensure_job_shape(load_json(job_path))
    used_preset = latest_job.get("preset_used") or preset
    job["result_summary"] = latest_job.get("result_summary")
    if used_preset != preset:
        log(job_folder, f"COLMAP fallback toegepast: gevraagd={preset}, gebruikt={used_preset}")

    job["artifacts"]["colmap_dense_dir"] = str(dense_dir)
    job["artifacts"]["colmap_fused_ply"] = str(fused_ply)
    job["artifacts"]["colmap_registered_images"] = reg
    job["registered_images"] = reg

    job["meta"] = job.get("meta", {})
    job["meta"]["colmap_preset"] = used_preset
    job["preset_used"] = used_preset

    job["result_summary"] = f"COLMAP complete: {reg} images registered."
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
    deliver_job_atomically(job_folder, job)
    persist_job(job_path, job)
    archive_job_folder(job_folder, job)


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
    if job.get("state") in TERMINAL_STATES:
        finalize_terminal_job(job_folder, job_path, job)
        return
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
