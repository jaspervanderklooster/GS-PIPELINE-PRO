# watcher.py
# Aangepaste watcher voor GS_PIPELINE-PRO
# - repo-first runner-script resolutie
# - dispatch_job start runner met repo .venv python.exe indien aanwezig
# - intake detection, job creation en dispatch

import json
import logging
import os
import re
import shutil
import subprocess
import time
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path

from preprocessor import resolve_preset

# ---------- Base paths / runtime folders ----------
try:
    from utils.config import get_config
    cfg = get_config() or {}
    configured_root = cfg.get("GS_ROOT") or cfg.get("GS_ROOT_PATH") or cfg.get("BASE")
    if configured_root:
        BASE = Path(str(configured_root))
    else:
        BASE = Path(r"D:\GS_PIPELINE")
except Exception:
    BASE = Path(r"D:\GS_PIPELINE")

BASE = Path(BASE).expanduser().resolve()

# --- BEGIN: Runner script resolution (repo-first) ---
try:
    REPO_ROOT = Path(__file__).resolve().parents[0]
except Exception:
    REPO_ROOT = Path(r"D:\GS-PIPELINE-PRO")

_runner_from_config = None
try:
    _runner_from_config = cfg.get("RUNNER_SCRIPT") if isinstance(cfg, dict) else None
except Exception:
    _runner_from_config = None

if _runner_from_config:
    RUNNER_SCRIPT = Path(str(_runner_from_config))
else:
    RUNNER_SCRIPT = Path(REPO_ROOT) / "scripts" / "run_job.py"

COMPAT_RUNNER = Path(r"D:\scripts\run_job.py")
if not RUNNER_SCRIPT.exists() and COMPAT_RUNNER.exists():
    try:
        import logging
        logging.warning("Runner script not found at %s — using compatibility path %s", RUNNER_SCRIPT, COMPAT_RUNNER)
    except Exception:
        pass
    RUNNER_SCRIPT = COMPAT_RUNNER

try:
    import logging
    logging.debug("Using runner script: %s", RUNNER_SCRIPT)
except Exception:
    pass
# --- END: Runner script resolution ---

INBOX = BASE / "inbox"
PROCESSING = BASE / "processing"
OUTBOX = BASE / "outbox"
LOGS = BASE / "logs"
TEMP = BASE / "temp"
STATE_DIR = TEMP / "watcher_state"
STATUS_META_DIR = TEMP / "status_meta"
INTAKE_STATUS_DIR = STATE_DIR / "intake_status"
REJECTED = BASE / "rejected"
REJECTED_ROOT_FILES = REJECTED / "root_files"
REJECTED_LOCKED_TOO_LONG = REJECTED / "locked_too_long"
REJECTED_MANUAL_ATTENTION = REJECTED / "manual_attention"

for _p in (
    INBOX,
    PROCESSING,
    OUTBOX,
    LOGS,
    TEMP,
    STATE_DIR,
    STATUS_META_DIR,
    INTAKE_STATUS_DIR,
    REJECTED,
    REJECTED_ROOT_FILES,
    REJECTED_LOCKED_TOO_LONG,
    REJECTED_MANUAL_ATTENTION,
):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

POLL_SECONDS = 10
REQUIRED_STABLE_SCANS = 3
RECENT_TAG_COOLDOWN_MINUTES = 30
STALE_CLAIM_MINUTES = 30
STALE_SNAPSHOT_HOURS = 24
ROOT_REJECT_RETENTION_HOURS = 12
LOCK_WAIT_MINUTES = 15
LOCK_PROBE_BYTES = 1
DEFAULT_PRESET = "standard"
STATUS_RETENTION_HOURS = 24
FFPROBE_EXE = "ffprobe"

ALLOWED_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
ALLOWED_ARCHIVE_EXTENSIONS = {".zip"}
ALLOWED_EXTENSIONS = ALLOWED_PHOTO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_ARCHIVE_EXTENSIONS
CLAIM_FILENAME = ".intake_claimed"
STATE_FILE = STATE_DIR / "watcher_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "watcher.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

def now_dt() -> datetime:
    return datetime.now().astimezone()

def now_iso() -> str:
    return now_dt().isoformat(timespec="seconds")

def parse_iso(value: str | None):
    try:
        return datetime.fromisoformat(value or "")
    except Exception:
        return None

def safe_tag(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "project"
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._-")
    return name or "project"

def write_json_atomic(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def load_state():
    default = {
        "folder_snapshots": {},
        "recent_tags": {},
        "warned_loose_files": {},
        "cooldown_blocked": {},
        "folder_status": {},
        "lock_blocked": {},
        "manual_attention_alerts": {},
    }
    if not STATE_FILE.exists():
        return default
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return default
    for k, v in default.items():
        state.setdefault(k, v.copy() if isinstance(v, dict) else v)
    return state

def save_state(state):
    write_json_atomic(STATE_FILE, state)

def ensure_dirs():
    for p in [
        INBOX, PROCESSING, OUTBOX, LOGS, TEMP, STATE_DIR, STATUS_META_DIR,
        INTAKE_STATUS_DIR, REJECTED_ROOT_FILES, REJECTED_LOCKED_TOO_LONG, REJECTED_MANUAL_ATTENTION,
    ]:
        p.mkdir(parents=True, exist_ok=True)

def owner_outbox(owner: str) -> Path:
    d = OUTBOX / owner
    d.mkdir(parents=True, exist_ok=True)
    return d

def status_file(owner: str, project: str) -> Path:
    return owner_outbox(owner) / f"{safe_tag(project)}_status.txt"

def status_meta(owner: str, project: str) -> Path:
    return STATUS_META_DIR / owner / f"{safe_tag(project)}.json"

def write_status(owner: str, project: str, status: str, *, reason: str | None = None,
                 explanation: str | None = None, advice: str | None = None,
                 note: str | None = None, terminal: bool = False):
    project = project or "project"
    lines = [
        f"Project: {project}",
        f"Status: {status}",
    ]
    if reason:
        lines.append(f"Reden: {reason}")
    if explanation:
        lines.append(f"Uitleg: {explanation}")
    if advice:
        lines.append(f"Advies: {advice}")
    if note:
        lines.append(f"Opmerking: {note}")
    lines.append(f"Laatste update: {now_dt().strftime('%Y-%m-%d %H:%M')}")
    status_file(owner, project).write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "owner": owner,
        "project": project,
        "status": status,
        "updated_at": now_iso(),
        "terminal": terminal,
        "remove_after": (now_dt() + timedelta(hours=STATUS_RETENTION_HOURS)).isoformat(timespec="seconds") if terminal else None,
    }
    write_json_atomic(status_meta(owner, project), meta)

def cleanup_expired_status_files():
    if not STATUS_META_DIR.exists():
        return
    for meta_path in STATUS_META_DIR.rglob("*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta.get("terminal"):
                continue
            remove_after = parse_iso(meta.get("remove_after"))
            if not remove_after or now_dt() < remove_after:
                continue
            sf = status_file(str(meta.get("owner", "")), str(meta.get("project", "project")))
            sf.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        except Exception:
            continue

def claim_file_path(folder: Path) -> Path:
    return folder / CLAIM_FILENAME

def is_claimed(folder: Path) -> bool:
    return claim_file_path(folder).exists()

def read_claim(folder: Path) -> dict:
    p = claim_file_path(folder)
    if not p.exists():
        return {}
    data = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    except Exception:
        return {}
    return data

def write_claim(folder: Path, *, tag: str, preset: str, owner: str):
    claim_file_path(folder).write_text(
        f"claimed_at={now_iso()}\n"
        f"tag={tag}\n"
        f"preset={preset}\n"
        f"owner={owner}\n",
        encoding="utf-8",
    )

def list_supported_files(folder: Path):
    files = []
    for root, _, names in os.walk(folder):
        for fn in names:
            p = Path(root) / fn
            if p.name == CLAIM_FILENAME:
                continue
            if p.suffix.lower() in ALLOWED_EXTENSIONS:
                files.append(p)
    return sorted(files, key=lambda p: str(p).lower())

def folder_snapshot(folder: Path):
    files = list_supported_files(folder)
    total_size = 0
    newest_mtime = 0.0
    for f in files:
        try:
            st = f.stat()
        except FileNotFoundError:
            continue
        total_size += st.st_size
        newest_mtime = max(newest_mtime, st.st_mtime)
    return {"file_count": len(files), "total_size": total_size, "newest_mtime": newest_mtime}

def snapshot_equals(a, b):
    return a.get("file_count") == b.get("file_count") and a.get("total_size") == b.get("total_size") and float(a.get("newest_mtime", 0)) == float(b.get("newest_mtime", 0))

def snapshot_signature(snap: dict) -> dict:
    return {
        "file_count": int(snap.get("file_count", 0)),
        "total_size": int(snap.get("total_size", 0)),
        "newest_mtime": float(snap.get("newest_mtime", 0.0)),
    }

def folder_key(folder: Path) -> str:
    return str(folder.resolve())

def detect_counts(folder: Path) -> dict:
    counts = {"photos": 0, "videos": 0, "zips": 0}
    for p in list_supported_files(folder):
        ext = p.suffix.lower()
        if ext in ALLOWED_PHOTO_EXTENSIONS:
            counts["photos"] += 1
        elif ext in ALLOWED_VIDEO_EXTENSIONS:
            counts["videos"] += 1
        elif ext in ALLOWED_ARCHIVE_EXTENSIONS:
            counts["zips"] += 1
    return counts

def detect_input_type(counts: dict) -> str:
    photos = counts.get("photos", 0)
    videos = counts.get("videos", 0)
    zips = counts.get("zips", 0)
    if videos > 0 and photos == 0 and zips == 0:
        return "video"
    if photos > 0 and videos == 0 and zips == 0:
        return "photoset"
    if zips > 0 and photos == 0 and videos == 0:
        return "archive"
    return "unknown"

def get_video_duration_seconds(video_path: Path) -> float | None:
    try:
        r = subprocess.run(
            [FFPROBE_EXE, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode != 0:
            return None
        return float((r.stdout or "").strip())
    except Exception:
        return None

def assess_dataset(folder: Path, preset: str) -> dict:
    files = list_supported_files(folder)
    counts = detect_counts(folder)
    total_bytes = 0
    longest_video_seconds = 0.0
    total_video_seconds = 0.0
    video_count = counts.get("videos", 0)
    for p in files:
        try:
            total_bytes += p.stat().st_size
        except FileNotFoundError:
            continue
        if p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            duration = get_video_duration_seconds(p)
            if duration:
                total_video_seconds += duration
                longest_video_seconds = max(longest_video_seconds, duration)

    classification = "normal"
    reason = None
    message = None

    if counts["photos"] >= 1200 or total_bytes >= 50 * 1024**3 or total_video_seconds >= 1800:
        classification = "too_large_for_safe_auto_processing"
        reason = "De dataset is te groot voor stabiele automatische verwerking."
        message = "Gebruik minder foto's, splits de opname op of kies een korter videofragment."
    elif counts["photos"] >= 800 or total_bytes >= 25 * 1024**3 or total_video_seconds >= 900:
        classification = "heavy_but_allowed"
        reason = "Deze dataset is groter dan aanbevolen voor de huidige pipeline."
        message = "We proberen hem toch te verwerken, maar de kans op mislukken is verhoogd."

    if preset == "hq":
        if counts["photos"] >= 800 or total_bytes >= 30 * 1024**3 or total_video_seconds >= 1200:
            classification = "too_large_for_safe_auto_processing"
            reason = "HQ is voor deze dataset te zwaar voor stabiele automatische verwerking."
            message = "Gebruik minder input of laat de pipeline op een veiligere instelling werken."
        elif classification == "normal" and (counts["photos"] >= 500 or total_video_seconds >= 600):
            classification = "heavy_but_allowed"
            reason = "HQ is aangevraagd op een relatief zware dataset."
            message = "We proberen hem te verwerken, maar de pipeline kan intern veiliger afschalen."

    return {
        "classification": classification,
        "counts": counts,
        "total_bytes": total_bytes,
        "video_count": video_count,
        "total_video_seconds": round(total_video_seconds, 2),
        "longest_video_seconds": int(longest_video_seconds),
        "requested_preset": preset,
        "reason": reason,
        "message": message,
    }

def build_job_payload(job_id: str, tag: str, raw_dir: Path, preset: str, counts: dict, input_type: str, owner: str, assessment: dict, auto_bundled: bool = False) -> dict:
    input_counts = dict(counts or {})
    input_counts["video_count"] = int(assessment.get("video_count") or input_counts.get("videos") or 0)
    input_counts["total_video_seconds"] = float(assessment.get("total_video_seconds") or 0.0)
    input_counts["total_bytes"] = int(assessment.get("total_bytes") or 0)
    return {
        "job_id": job_id,
        "created_at": now_iso(),
        "state": "queued",
        "error": None,
        "preset": preset,
        "owner": owner,
        "submitted_by": owner,
        "assessment": assessment,
        "meta": {"colmap_preset": preset},
        "input": {
            "tag": tag,
            "primary": str(raw_dir),
            "type": input_type,
            "counts": input_counts,
            "owner": owner,
            "auto_bundled": auto_bundled,
        },
        "artifacts": {},
    }

def generate_job_id(tag: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"

def determine_preset(tag: str) -> str:
    resolved = resolve_preset(tag=tag, requested_preset="")
    if resolved == "high":
        return "hq"
    if resolved == "good":
        return "standard"
    return DEFAULT_PRESET

def active_tags_in_processing() -> set[str]:
    tags = set()
    if not PROCESSING.exists():
        return tags
    for job_dir in PROCESSING.iterdir():
        job_json = job_dir / "job.json"
        if not job_dir.is_dir() or not job_json.exists():
            continue
        try:
            job = json.loads(job_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        state = str(job.get("state", "")).lower().strip()
        if state in {"done", "failed"}:
            continue
        tag = safe_tag(str(job.get("input", {}).get("tag") or job.get("tag") or ""))
        if tag:
            tags.add(tag)
    return tags

def probe_file_access(file_path: Path) -> tuple[bool, str]:
    try:
        file_path.stat()
        with file_path.open("rb") as handle:
            handle.read(1)
        return True, ""
    except PermissionError as exc:
        return False, f"permission_denied: {exc}"
    except OSError as exc:
        return False, f"os_error: {exc}"
    except Exception as exc:
        return False, f"unexpected_error: {exc}"

def probe_folder_accessibility(folder: Path) -> tuple[bool, list[dict]]:
    problems = []
    for p in list_supported_files(folder):
        ok, reason = probe_file_access(p)
        if not ok:
            problems.append({"path": str(p), "name": p.name, "reason": reason})
    return len(problems) == 0, problems

def move_to_rejected_file(item: Path, rejected_dir: Path, payload: dict) -> Path:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    dest = rejected_dir / item.name
    if dest.exists():
        dest = rejected_dir / f"{dest.stem}__{datetime.now().strftime('%Y%m%d_%H%M%S')}{dest.suffix}"
    shutil.move(str(item), str(dest))
    write_json_atomic(dest.parent / f"{dest.name}.reject.json", payload)
    return dest

def move_folder_to_rejected(folder: Path, rejected_dir: Path, payload: dict) -> Path:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    dest = rejected_dir / folder.name
    if dest.exists():
        dest = rejected_dir / f"{folder.name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(folder), str(dest))
    write_json_atomic(dest.parent / f"{dest.name}.reject.json", payload)
    return dest

def prune_recent_tags(state):
    cutoff = now_dt() - timedelta(minutes=RECENT_TAG_COOLDOWN_MINUTES)
    state["recent_tags"] = {k: v for k, v in state["recent_tags"].items() if (parse_iso(v) and parse_iso(v) >= cutoff)}

def prune_warned_loose_files(state):
    cutoff = now_dt() - timedelta(hours=ROOT_REJECT_RETENTION_HOURS)
    state["warned_loose_files"] = {k: v for k, v in state["warned_loose_files"].items() if (parse_iso(v) and parse_iso(v) >= cutoff)}

def prune_stale_snapshots(state):
    cutoff = now_dt() - timedelta(hours=STALE_SNAPSHOT_HOURS)
    state["folder_snapshots"] = {k: v for k, v in state["folder_snapshots"].items() if (parse_iso(v.get("last_seen", "")) and parse_iso(v.get("last_seen", "")) >= cutoff)}
    state["lock_blocked"] = {k: v for k, v in state["lock_blocked"].items() if (parse_iso(v.get("last_seen", "")) and parse_iso(v.get("last_seen", "")) >= cutoff)}
    state["folder_status"] = {k: v for k, v in state["folder_status"].items() if (parse_iso(v.get("last_seen", "")) and parse_iso(v.get("last_seen", "")) >= cutoff)}

def tag_is_recent(state, tag: str) -> bool:
    ts = parse_iso(state["recent_tags"].get(tag))
    return bool(ts and ts >= (now_dt() - timedelta(minutes=RECENT_TAG_COOLDOWN_MINUTES)))

def mark_tag_recent(state, tag: str):
    state["recent_tags"][tag] = now_iso()

def owner_dirs():
    owners = []
    for item in INBOX.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            owners.append(item)
    return sorted(owners, key=lambda p: p.name.lower())

def folder_candidates():
    cands = []
    for owner_dir in owner_dirs():
        for item in owner_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                cands.append((owner_dir.name, item))
    cands.sort(key=lambda pair: str(pair[1]).lower())
    return cands

def cleanup_dead_snapshot_entries(state, current_folders: list[Path]):
    current_keys = {folder_key(p) for p in current_folders}
    state["folder_snapshots"] = {k: v for k, v in state["folder_snapshots"].items() if k in current_keys}
    state["cooldown_blocked"] = {k: v for k, v in state["cooldown_blocked"].items() if k in current_keys}
    state["lock_blocked"] = {k: v for k, v in state["lock_blocked"].items() if k in current_keys}
    state["folder_status"] = {k: v for k, v in state["folder_status"].items() if k in current_keys}

def update_status_cache(state, folder: Path, status: str, message: str = ""):
    state["folder_status"][folder_key(folder)] = {"status": status, "message": message, "updated_at": now_iso()}

# --- NEW: dispatch_job helper with venv detection ---
def dispatch_job(job_id: str, job_dir: Path):
    """
    Start a worker subprocess for the given job_dir.
    Creates job_dir/worker_started and job_dir/worker_subprocess.log.
    Returns True if dispatched, False if already started or failed to start.
    """
    started_file = job_dir / "worker_started"
    if started_file.exists():
        logging.info("Worker already started for job %s (marker exists)", job_id)
        return False

    log_fp = job_dir / "worker_subprocess.log"
    job_dir.mkdir(parents=True, exist_ok=True)

    # Prefer repo .venv python if present (Windows and Unix paths)
    venv_py_windows = Path(REPO_ROOT) / ".venv" / "Scripts" / "python.exe"
    venv_py_unix = Path(REPO_ROOT) / ".venv" / "bin" / "python"

    if venv_py_windows.exists():
        python_exe = str(venv_py_windows)
    elif venv_py_unix.exists():
        python_exe = str(venv_py_unix)
    else:
        python_exe = sys.executable

    # Prepare environment for subprocess: prepend venv bin to PATH so DLLs and scripts resolve
    env = os.environ.copy()
    try:
        venv_bin = str(Path(python_exe).parent)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    except Exception:
        pass

    runner = RUNNER_SCRIPT
    if not runner.exists():
        logging.error("Runner script not found: %s", runner)
        return False

    cmd = [python_exe, str(runner), str(job_dir)]

    # Start subprocess, direct stdout/stderr to log file
    try:
        f = open(log_fp, "a", encoding="utf-8")
        # write info header to log so we can see which python and PATH used
        f.write(f"[{now_iso()}] Starting worker subprocess with python: {python_exe}\n")
        f.write(f"[{now_iso()}] RUNNER_SCRIPT: {runner}\n")
        f.write(f"[{now_iso()}] PATH (head): {env.get('PATH','')[:200]}\n")
        f.flush()

        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
        started_payload = {"pid": proc.pid, "started_at": now_iso()}
        started_file.write_text(json.dumps(started_payload, indent=2), encoding="utf-8")
        logging.info("Dispatched worker for job %s (pid=%s) -> log=%s", job_id, proc.pid, log_fp)
        return True
    except Exception as exc:
        logging.exception("Failed to dispatch worker for job %s: %s", job_id, exc)
        try:
            f.write(f"[{now_iso()}] Failed to start worker subprocess: {exc}\n")
            f.close()
        except Exception:
            pass
        return False
# --- end dispatch_job ---

def evaluate_folder_stability(state, folder: Path):
    key = folder_key(folder)
    snap = folder_snapshot(folder)
    entry = state["folder_snapshots"].get(key)
    if not entry:
        state["folder_snapshots"][key] = {"last_snapshot": snap, "stable_count": 1, "last_seen": now_iso()}
        update_status_cache(state, folder, "seen", "Nieuwe intake-map gezien")
        return False, snap, 1
    stable_count = int(entry.get("stable_count", 0))
    if snapshot_equals(snap, entry.get("last_snapshot", {})):
        stable_count += 1
    else:
        stable_count = 1
    entry["last_snapshot"] = snap
    entry["stable_count"] = stable_count
    entry["last_seen"] = now_iso()
    state["folder_snapshots"][key] = entry
    if snap["file_count"] == 0:
        return False, snap, stable_count
    return stable_count >= REQUIRED_STABLE_SCANS, snap, stable_count

def try_recover_stale_claim(folder: Path, active_tags: set[str]):
    if not is_claimed(folder):
        return False
    claim = read_claim(folder)
    claimed_at = parse_iso(claim.get("claimed_at"))
    tag = safe_tag(claim.get("tag") or folder.name)
    if tag in active_tags:
        return False
    if claimed_at and claimed_at >= (now_dt() - timedelta(minutes=STALE_CLAIM_MINUTES)):
        return False
    claim_file_path(folder).unlink(missing_ok=True)
    return True

def create_job_from_folder(folder: Path, owner: str, tag: str, preset: str, assessment: dict, auto_bundled: bool = False):
    job_id = generate_job_id(tag)
    job_dir = PROCESSING / job_id
    raw_dir = job_dir / "input_raw" / tag
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(raw_dir))
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    counts = detect_counts(raw_dir)
    input_type = detect_input_type(counts)
    job = build_job_payload(job_id, tag, raw_dir, preset, counts, input_type, owner, assessment, auto_bundled)
    write_json_atomic(job_dir / "job.json", job)
    return job_id, input_type, counts

def bundle_owner_loose_files(state: dict, owner_dir: Path):
    files = [p for p in owner_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
    if not files:
        state.get("manual_attention_alerts", {}).pop(owner_dir.name, None)
        return []
    photos = [p for p in files if p.suffix.lower() in ALLOWED_PHOTO_EXTENSIONS]
    videos = [p for p in files if p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS]
    zips = [p for p in files if p.suffix.lower() in ALLOWED_ARCHIVE_EXTENSIONS]
    created = []

    def make_container(name: str, selected: list[Path], note: str):
        container = owner_dir / name
        idx = 1
        while container.exists():
            idx += 1
            container = owner_dir / f"{name}_{idx}"
        container.mkdir(parents=True, exist_ok=False)
        for src in selected:
            shutil.move(str(src), str(container / src.name))
        write_json_atomic(container / ".auto_bundle.json", {"created_at": now_iso(), "note": note})
        created.append(container)

    if videos and not photos and not zips:
        state.get("manual_attention_alerts", {}).pop(owner_dir.name, None)
        for video in videos:
            make_container(f"{safe_tag(video.stem)}_auto", [video], "Los videobestand automatisch als project gebundeld.")
        return created
    if zips and not photos and not videos:
        state.get("manual_attention_alerts", {}).pop(owner_dir.name, None)
        for archive in zips:
            make_container(f"{safe_tag(archive.stem)}_auto", [archive], "Los archief automatisch als project gebundeld.")
        return created
    if photos and not videos and not zips:
        state.get("manual_attention_alerts", {}).pop(owner_dir.name, None)
        make_container(f"foto_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}", photos, "Losse foto's automatisch als project gebundeld.")
        return created

    manifest_dir = REJECTED_MANUAL_ATTENTION / owner_dir.name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "loose_files_manual_attention.json"
    payload = {
        "owner": owner_dir.name,
        "detected_at": now_iso(),
        "reason": "Los bestand in inbox-root is geen geldige intake.",
        "advice": "Plaats bestanden in inbox\\<gebruiker>\\<project> of direct in inbox\\<gebruiker>.",
        "source_path": str(item),
        "status": "moved_to_rejected_root_files",
    }
    return created

def scan_loose_files_in_inbox_root(state):
    for item in INBOX.iterdir():
        if not item.is_file() or item.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        key = str(item.resolve())
        if key in state["warned_loose_files"] and not item.exists():
            continue
        payload = {
            "original_name": item.name,
            "detected_at": now_iso(),
            "reason": "Los bestand in inbox-root is geen geldige intake.",
            "advice": "Plaats bestanden in inbox\\<gebruiker>\\<project> of direct in inbox\\<gebruiker>.",
            "source_path": str(item),
            "status": "moved_to_rejected_root_files",
        }
        try:
            move_to_rejected_file(item, REJECTED_ROOT_FILES, payload)
            state["warned_loose_files"][key] = now_iso()
        except Exception:
            state["warned_loose_files"][key] = now_iso()

def process_candidates(state):
    for owner_dir in owner_dirs():
        bundle_owner_loose_files(state, owner_dir)

    active_tags = active_tags_in_processing()
    candidates = folder_candidates()
    cleanup_dead_snapshot_entries(state, [f for _, f in candidates])

    for owner, folder in candidates:
        try_recover_stale_claim(folder, active_tags)

        if is_claimed(folder):
            continue
        if not list_supported_files(folder):
            continue

        stable, snap, stable_count = evaluate_folder_stability(state, folder)
        if not stable:
            continue

        accessible, problems = probe_folder_accessibility(folder)
        if not accessible:
            first = problems[0] if problems else {"name": "onbekend"}
            write_status(owner, folder.name, "Upload ontvangen", note=f"Watcher wacht nog omdat een bestand in gebruik lijkt: {first['name']}")
            continue

        tag = safe_tag(folder.name)
        preset = determine_preset(tag)

        if tag in active_tags or tag_is_recent(state, tag):
            write_status(owner, folder.name, "Upload ontvangen", note="Er is al een actieve of recente intake met dezelfde projectnaam.")
            continue

        assessment = assess_dataset(folder, preset)
        auto_bundle_note = None
        auto_bundle_manifest = folder / ".auto_bundle.json"
        if auto_bundle_manifest.exists():
            try:
                auto_bundle_note = json.loads(auto_bundle_manifest.read_text(encoding="utf-8")).get("note")
            except Exception:
                auto_bundle_note = "Losse bestanden zijn automatisch als project behandeld."

        if assessment["classification"] == "too_large_for_safe_auto_processing":
            reason = assessment["reason"] or "De dataset is te groot voor stabiele automatische verwerking."
            explanation = assessment["message"] or "De kans op vastlopen of mislukken is te groot voor veilige automatische intake."
            advice = "Gebruik minder foto's, een korter videofragment of overleg als je wilt dat we meekijken."
            dest = move_folder_to_rejected(folder, REJECTED_MANUAL_ATTENTION, {
                "owner": owner,
                "detected_at": now_iso(),
                "assessment": assessment,
                "reason": reason,
                "advice": advice,
                "status": "rejected_too_large",
            })
            write_status(owner, folder.name, "Afgewezen", reason=reason, explanation=explanation, advice=advice, terminal=True)
            state["folder_snapshots"].pop(folder_key(folder), None)
            logging.warning("Intake afgewezen als te zwaar: %s -> %s", folder, dest)
            continue

        write_claim(folder, tag=tag, preset=preset, owner=owner)
        note_parts = []
        if auto_bundle_note:
            note_parts.append(auto_bundle_note)
        if assessment["classification"] == "heavy_but_allowed" and assessment.get("message"):
            note_parts.append(assessment["message"])
        write_status(owner, folder.name, "Upload ontvangen", note=" ".join(note_parts) if note_parts else None)

        try:
            job_id, input_type, counts = create_job_from_folder(folder, owner, tag, preset, assessment, auto_bundled=bool(auto_bundle_note))
            mark_tag_recent(state, tag)
            state["folder_snapshots"].pop(folder_key(folder), None)
            write_json_atomic(INTAKE_STATUS_DIR / f"queued__{job_id}.json", {
                "job_id": job_id,
                "owner": owner,
                "tag": tag,
                "status": "queued",
                "queued_at": now_iso(),
                "preset": preset,
                "input_type": input_type,
                "counts": counts,
                "assessment": assessment,
            })

            job_dir = PROCESSING / job_id
            dispatched = dispatch_job(job_id, job_dir)
            if dispatched:
                logging.info("Worker dispatched for job %s", job_id)
                write_status(owner, tag, "In verwerking", note="De job is automatisch gestart.")
            else:
                logging.warning("Worker NOT dispatched for job %s", job_id)
                write_status(owner, tag, "In wachtrij", note="Job klaargezet, maar worker is (nog) niet gestart.")
        except Exception as exc:
            claim_file_path(folder).unlink(missing_ok=True)
            write_status(owner, folder.name, "Mislukt", reason="De intake kon niet worden klaargezet.", explanation=str(exc), advice="Probeer het opnieuw of vraag iemand om mee te kijken.", terminal=True)

shutdown = False

def _handle_sig(signum, frame):
    global shutdown
    logging.info("Shutdown requested (signal=%s)", signum)
    shutdown = True

try:
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
except Exception:
    pass

def main():
    ensure_dirs()
    logging.info("Watcher gestart")
    state = load_state()
    while not shutdown:
        try:
            cleanup_expired_status_files()
            prune_recent_tags(state)
            prune_warned_loose_files(state)
            prune_stale_snapshots(state)
            scan_loose_files_in_inbox_root(state)
            process_candidates(state)
            save_state(state)
        except Exception as exc:
            logging.exception("Onverwachte fout in watcher-loop: %s", exc)

        slept = 0
        while slept < POLL_SECONDS and not shutdown:
            time.sleep(1)
            slept += 1

    logging.info("Watcher wordt netjes afgesloten (graceful shutdown)")

if __name__ == "__main__":
    main()