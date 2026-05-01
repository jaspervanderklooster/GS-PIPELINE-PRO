# preprocessor.py
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

# Pillow voor auto-orientatie / downscale / EXIF
try:
    from PIL import Image, ImageOps, ExifTags, UnidentifiedImageError
except Exception:
    Image = None
    ImageOps = None
    ExifTags = None
    UnidentifiedImageError = Exception

# OpenCV voor blur-score (optioneel, maar sterk aanbevolen)
try:
    import cv2
except Exception:
    cv2 = None

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

ALLOWED_PHOTOS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEOS = {".mp4", ".mov", ".avi", ".mkv"}
ALLOWED_ARCHIVES = {".zip"}

PRESETS = {
    "good": {
        "sample_interval": 2.0,
        "max_width": 2560,
        "max_frames": 1200,
        "jpeg_quality": 2,
        "blur_floor": 40.0,
        "dedupe_threshold": 0.965,
        "compare_width": 384,
        "min_keep_ratio": 0.55,
    },
    "high": {
        "sample_interval": 1.25,
        "max_width": 3200,
        "max_frames": 1600,
        "jpeg_quality": 2,
        "blur_floor": 55.0,
        "dedupe_threshold": 0.982,
        "compare_width": 448,
        "min_keep_ratio": 0.65,
    },
}

DOWNSCALE_THRESHOLD = 3500
DOWNSCALE_TO = 3000
DOWNSCALE_QUALITY = 90

_GPS_TAG_IDS = {}
if ExifTags is not None:
    _GPS_TAG_IDS = {v: k for k, v in ExifTags.TAGS.items()}


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(job_folder: Path, msg: str):
    lp = job_folder / "worker.log"
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("a", encoding="utf-8") as f:
        f.write(f"[{iso_now()}] {msg}\n")


def safe_name(name: str) -> str:
    name = (name or "").strip()
    return "".join(c for c in name if c.isalnum() or c in "_- ") or "upload"


def frames_dir(job_folder: Path) -> Path:
    d = job_folder / "staging" / "frames"
    d.mkdir(parents=True, exist_ok=True)
    return d


def preprocess_dir(job_folder: Path) -> Path:
    d = job_folder / "preprocess"
    d.mkdir(parents=True, exist_ok=True)
    return d


def collect_media_files(src_folder: Path) -> dict:
    photos = []
    videos = []
    archives = []
    sidecars = []

    for p in src_folder.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in ALLOWED_PHOTOS:
            photos.append(p)
        elif ext in ALLOWED_VIDEOS:
            videos.append(p)
        elif ext in ALLOWED_ARCHIVES:
            archives.append(p)
        elif ext in {".srt", ".gpx", ".nmea", ".csv"}:
            sidecars.append(p)

    photos.sort(key=lambda p: str(p).lower())
    videos.sort(key=lambda p: str(p).lower())
    archives.sort(key=lambda p: str(p).lower())
    sidecars.sort(key=lambda p: str(p).lower())

    return {
        "photos": photos,
        "videos": videos,
        "archives": archives,
        "sidecars": sidecars,
    }


def detect_counts(src_folder: Path) -> dict:
    media = collect_media_files(src_folder)
    return {
        "photos": len(media["photos"]),
        "videos": len(media["videos"]),
        "zips": len(media["archives"]),
        "sidecars": len(media["sidecars"]),
    }


def detect_input_type(src_folder: Path) -> str:
    counts = detect_counts(src_folder)
    photos = counts["photos"]
    videos = counts["videos"]
    zips = counts["zips"]

    if videos > 0 and photos == 0 and zips == 0:
        return "video"
    if photos > 0 and videos == 0 and zips == 0:
        return "photoset"
    if zips > 0 and photos == 0 and videos == 0:
        return "archive"
    return "unknown"


def resolve_preset(tag: str = "", requested_preset: str = "") -> str:
    rp = (requested_preset or "").strip().lower()
    if rp in {"high", "hq"}:
        return "high"
    if rp in {"good", "standard", "std"}:
        return "good"

    t = (tag or "").lower()
    if "hq" in t or "high" in t:
        return "high"
    if "good" in t or "standard" in t or "std" in t:
        return "good"

    return "good"


def safe_extract_zip(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)

            if member_path.is_absolute() or ".." in member_path.parts:
                continue

            target = (dest / member_path).resolve()
            if not str(target).startswith(str(dest.resolve())):
                continue

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def handle_zip_inputs(job_folder: Path, src_folder: Path) -> bool:
    media = collect_media_files(src_folder)
    zip_files = media["archives"]

    if not zip_files:
        return False

    for zip_file in zip_files:
        dest = src_folder / "_unzipped" / zip_file.stem
        log(job_folder, f"Unzipping {zip_file.name} -> {dest}")
        safe_extract_zip(zip_file, dest)
        try:
            zip_file.unlink()
        except Exception:
            log(job_folder, f"Kon zip niet verwijderen na uitpakken: {zip_file.name}")

    return True


def unzip_if_needed(path_a: Path, path_b: Path):
    """Compatibility helper.

    Supported call styles:
    - unzip_if_needed(job_folder, src_folder) -> bool (worker flow)
    - unzip_if_needed(zip_file, output_dir) -> Path (unit test/utility flow)
    """
    if path_a.is_file() and path_a.suffix.lower() in ALLOWED_ARCHIVES:
        safe_extract_zip(path_a, path_b)
        return path_b
    return handle_zip_inputs(path_a, path_b)


def _dms_to_decimal(value, ref) -> float | None:
    try:
        def _to_float(x):
            if isinstance(x, tuple):
                num, den = x
                return float(num) / float(den or 1)
            if hasattr(x, "numerator") and hasattr(x, "denominator"):
                return float(x.numerator) / float(x.denominator or 1)
            return float(x)

        d = _to_float(value[0])
        m = _to_float(value[1])
        s = _to_float(value[2])
        out = d + (m / 60.0) + (s / 3600.0)
        if str(ref).upper() in {"S", "W"}:
            out *= -1.0
        return out
    except Exception:
        return None


def _extract_photo_gps(path: Path) -> dict | None:
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            gps_tag = _GPS_TAG_IDS.get("GPSInfo")
            if gps_tag is None:
                return None
            gps = exif.get(gps_tag)
            if not gps:
                return None
            if isinstance(gps, dict):
                gps_dict = gps
            else:
                gps_dict = dict(gps)
            lat = _dms_to_decimal(gps_dict.get(2), gps_dict.get(1))
            lon = _dms_to_decimal(gps_dict.get(4), gps_dict.get(3))
            alt = gps_dict.get(6)
            alt_val = None
            if alt is not None:
                try:
                    if isinstance(alt, tuple):
                        alt_val = float(alt[0]) / float(alt[1] or 1)
                    elif hasattr(alt, "numerator") and hasattr(alt, "denominator"):
                        alt_val = float(alt.numerator) / float(alt.denominator or 1)
                    else:
                        alt_val = float(alt)
                except Exception:
                    alt_val = None
            if lat is None or lon is None:
                return None
            return {"lat": lat, "lon": lon, "alt": alt_val}
    except Exception:
        return None


def _extract_video_metadata(video: Path) -> dict:
    cmd = [
        FFPROBE,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0 or not r.stdout.strip():
            return {}
        payload = json.loads(r.stdout)
    except Exception:
        return {}

    tags = {}
    for key, value in (payload.get("format", {}).get("tags") or {}).items():
        tags[str(key)] = value
    for stream in payload.get("streams", []) or []:
        for key, value in (stream.get("tags") or {}).items():
            tags.setdefault(str(key), value)

    gps_keys = [
        "location",
        "com.apple.quicktime.location.ISO6709",
        "com.android.capture.fps",
        "com.apple.quicktime.make",
        "com.apple.quicktime.model",
        "creation_time",
    ]
    extracted = {k: tags.get(k) for k in gps_keys if tags.get(k) not in (None, "")}
    if payload.get("format", {}).get("duration"):
        extracted["duration"] = payload["format"]["duration"]
    return extracted


def write_gps_summary(job_folder: Path, src_folder: Path, staging: Path):
    media = collect_media_files(src_folder)
    photo_entries = []
    photos_with_gps = 0

    for path in sorted(staging.glob("frame_*"), key=lambda p: p.name.lower()):
        gps = _extract_photo_gps(path)
        if gps:
            photos_with_gps += 1
            photo_entries.append({"file": path.name, **gps})

    video_entries = []
    for video in media["videos"]:
        meta = _extract_video_metadata(video)
        if meta:
            video_entries.append({"file": video.name, "metadata": meta})

    sidecars = [str(p.name) for p in media.get("sidecars", [])]
    payload = {
        "created_at": iso_now(),
        "photos_with_gps": photos_with_gps,
        "photo_entries_preview": photo_entries[:50],
        "video_entries": video_entries,
        "sidecars": sidecars,
        "note": (
            "Foto-EXIF blijft behouden waar mogelijk. Video-GPS wordt gelogd en sidecars worden gezien, "
            "maar per-frame GPS voor video vraagt nog een aparte track-parser of EXIF-injectiestap."
        ),
    }
    out = preprocess_dir(job_folder) / "gps_summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(job_folder, f"GPS summary: photos_with_gps={photos_with_gps}, videos_with_metadata={len(video_entries)}, sidecars={len(sidecars)}")
    return out


def _auto_orient_and_downscale(
    path: Path,
    job_folder: Path,
    downscale_threshold: int = DOWNSCALE_THRESHOLD,
    downscale_to: int = DOWNSCALE_TO,
    quality: int = DOWNSCALE_QUALITY,
) -> None:
    if Image is None or ImageOps is None:
        log(job_folder, "Pillow niet beschikbaar: skip auto-orient/downscale.")
        return

    try:
        img = Image.open(path)
    except Exception as e:
        log(job_folder, f"Kon afbeelding niet openen voor auto-orient: {path.name} ({e})")
        return

    try:
        exif_bytes = img.info.get("exif")
        img = ImageOps.exif_transpose(img)
    except Exception:
        exif_bytes = img.info.get("exif")

    try:
        w, h = img.size
        if max(w, h) > downscale_threshold:
            if w >= h:
                new_w = downscale_to
                new_h = max(1, int(h * (new_w / float(w))))
            else:
                new_h = downscale_to
                new_w = max(1, int(w * (new_h / float(h))))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            log(job_folder, f"Downscaled {path.name}: {w}x{h} -> {new_w}x{new_h}")
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            save_kwargs = {"quality": quality, "optimize": True}
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
            img.save(path, "JPEG", **save_kwargs)
        elif ext == ".png":
            img.save(path, "PNG", optimize=True)
        else:
            img.save(path)
    except Exception as e:
        log(job_folder, f"Fout tijdens auto-orient/downscale voor {path.name}: {e}")
    finally:
        try:
            img.close()
        except Exception:
            pass


def verify_images(src_dir: Path) -> list[tuple[str, str]]:
    if Image is None:
        return []
    bad: list[tuple[str, str]] = []
    for f in sorted(src_dir.glob("frame_*"), key=lambda p: p.name.lower()):
        if f.suffix.lower() not in ALLOWED_PHOTOS:
            continue
        try:
            with Image.open(f) as im:
                im.verify()
        except Exception as exc:
            bad.append((f.name, str(exc)))
    return bad


def move_bad_files(job_folder: Path, src_dir: Path, bad: list[tuple[str, str]]):
    if not bad:
        return 0
    bad_dir = preprocess_dir(job_folder) / "bad_files"
    bad_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for filename, reason in bad:
        source = src_dir / filename
        if not source.exists():
            continue
        target = bad_dir / filename
        try:
            shutil.move(str(source), str(target))
            moved += 1
            log(job_folder, f"Bad file moved: {filename} ({reason})")
        except Exception as exc:
            log(job_folder, f"Failed moving bad file {filename}: {exc}")
    return moved


def _blur_score(path: Path) -> float | None:
    if cv2 is None:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    try:
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    finally:
        del img


def _remove_blurry_frames(job_folder: Path, staging: Path, preset_key: str) -> dict:
    files = sorted([p for p in staging.glob("frame_*") if p.suffix.lower() in ALLOWED_PHOTOS], key=lambda p: p.name.lower())
    out = {
        "evaluated": len(files),
        "removed": 0,
        "kept": len(files),
        "threshold": None,
        "scores": [],
        "mode": "skipped",
    }
    if not files:
        return out
    if cv2 is None:
        log(job_folder, "OpenCV niet beschikbaar: blur-filter wordt overgeslagen.")
        return out

    cfg = PRESETS[preset_key]
    blur_floor = float(cfg["blur_floor"])
    min_keep_ratio = float(cfg["min_keep_ratio"])

    scored = []
    for path in files:
        score = _blur_score(path)
        if score is None:
            score = 0.0
        scored.append({"file": path.name, "path": path, "blur_score": round(score, 3)})

    only_scores = [entry["blur_score"] for entry in scored]
    median_score = sorted(only_scores)[len(only_scores) // 2] if only_scores else 0.0
    dynamic_floor = min(blur_floor, max(12.0, median_score * 0.45))
    min_keep_count = max(24, int(math.ceil(len(scored) * min_keep_ratio)))

    scored_sorted = sorted(scored, key=lambda x: (x["blur_score"], x["file"]))
    keep_budget = len(scored)
    remove_candidates = []
    for entry in scored_sorted:
        if entry["blur_score"] >= dynamic_floor:
            continue
        if keep_budget - 1 < min_keep_count:
            continue
        remove_candidates.append(entry)
        keep_budget -= 1

    blur_dir = preprocess_dir(job_folder) / "blur_rejected"
    blur_dir.mkdir(parents=True, exist_ok=True)
    for entry in remove_candidates:
        target = blur_dir / entry["file"]
        try:
            shutil.move(str(entry["path"]), str(target))
        except Exception:
            shutil.copy2(str(entry["path"]), str(target))
            entry["path"].unlink(missing_ok=True)

    out.update({
        "removed": len(remove_candidates),
        "kept": keep_budget,
        "threshold": round(dynamic_floor, 3),
        "scores": [{"file": e["file"], "blur_score": e["blur_score"]} for e in scored],
        "mode": "laplacian_variance",
    })
    log(job_folder, f"Blur filter: evaluated={len(scored)} removed={len(remove_candidates)} kept={keep_budget} threshold={dynamic_floor:.2f}")
    return out


def _run_dedupe_headless(job_folder: Path, src_dir: Path, preset_key: str) -> dict:
    script = Path(__file__).resolve().parent / "dedupe_headless.py"
    out_dir = preprocess_dir(job_folder) / "dedupe"
    cfg = PRESETS[preset_key]
    result = {
        "status": "skipped",
        "kept_count": len(list(src_dir.glob("frame_*"))),
        "purged_count": 0,
    }
    if not script.exists():
        log(job_folder, "dedupe_headless.py niet gevonden: dedupe overgeslagen.")
        return result

    cmd = [
        os.environ.get("PYTHON", "python"),
        str(script),
        "--input-dir", str(src_dir),
        "--output-dir", str(out_dir),
        "--threshold", str(cfg["dedupe_threshold"]),
        "--compare-width", str(cfg["compare_width"]),
        "--job-id", job_folder.name,
        "--worker-log", str(job_folder / "worker.log"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(job_folder, f"dedupe_headless failed (non-fatal): {r.stderr[-2000:]}")
        return result

    analysis_path = out_dir / "analysis.json"
    if analysis_path.exists():
        try:
            result = json.loads(analysis_path.read_text(encoding="utf-8"))
        except Exception:
            result = {"status": "ok"}
    log(job_folder, f"Dedupe complete: kept={result.get('kept_count')} purged={result.get('purged_count')} threshold={cfg['dedupe_threshold']}")
    return result


def _normalize_frame_sequence(job_folder: Path, staging: Path):
    files = sorted([p for p in staging.glob("frame_*") if p.suffix.lower() in ALLOWED_PHOTOS], key=lambda p: p.name.lower())
    if not files:
        return 0
    temp_dir = staging / "_renumber"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(files, start=1):
        target = temp_dir / f"frame_{idx:06d}.jpg"
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.move(str(path), str(target))
        else:
            try:
                if Image is None:
                    shutil.move(str(path), str(target))
                else:
                    with Image.open(path) as img:
                        if img.mode not in ("RGB", "L"):
                            img = img.convert("RGB")
                        img.save(target, "JPEG", quality=92, optimize=True)
                    path.unlink(missing_ok=True)
            except Exception:
                shutil.move(str(path), str(target))
    for old in staging.glob("frame_*"):
        old.unlink(missing_ok=True)
    for tmp in temp_dir.glob("frame_*"):
        shutil.move(str(tmp), str(staging / tmp.name))
    shutil.rmtree(temp_dir, ignore_errors=True)
    log(job_folder, f"Frames opnieuw genummerd: {len(files)} stuks")
    return len(files)


def _write_preprocess_summary(job_folder: Path, payload: dict):
    out = preprocess_dir(job_folder) / "summary.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def log_pillow_status(job_folder: Path):
    if Image is None:
        log(job_folder, "Pillow not available — skipping reencode")
        return
    log(job_folder, f"PIL available: {getattr(Image, '__version__', 'unknown')}")


def _prepare_staging_from_photos(job_folder: Path, src_folder: Path) -> int:
    staging = frames_dir(job_folder)
    media = collect_media_files(src_folder)
    photos = media["photos"]
    moved = 0
    for idx, p in enumerate(photos, start=1):
        target = staging / f"frame_{idx:06d}{p.suffix.lower()}"
        if target.exists():
            continue
        try:
            shutil.move(str(p), str(target))
        except Exception:
            shutil.copy2(str(p), str(target))
            p.unlink(missing_ok=True)
        _auto_orient_and_downscale(target, job_folder)
        moved += 1
    log(job_folder, f"Photos staged: {moved} files -> {staging}")
    return moved


def preprocess_photoset(job_folder: Path, src_folder: Path, preset: str = "standard") -> int:
    preset_key = resolve_preset(requested_preset=preset)
    log_pillow_status(job_folder)
    staged_count = _prepare_staging_from_photos(job_folder, src_folder)
    staging = frames_dir(job_folder)
    bad_files = verify_images(staging)
    bad_moved = move_bad_files(job_folder, staging, bad_files)
    blur_info = _remove_blurry_frames(job_folder, staging, preset_key)
    dedupe_info = _run_dedupe_headless(job_folder, staging, preset_key)
    final_count = _normalize_frame_sequence(job_folder, staging)
    gps_path = write_gps_summary(job_folder, src_folder, staging)
    summary = {
        "created_at": iso_now(),
        "input_type": "photoset",
        "preset": preset_key,
        "staged_count": staged_count,
        "bad_files_moved": bad_moved,
        "blur": blur_info,
        "dedupe": {
            "status": dedupe_info.get("status"),
            "kept_count": dedupe_info.get("kept_count"),
            "purged_count": dedupe_info.get("purged_count"),
            "threshold": PRESETS[preset_key]["dedupe_threshold"],
        },
        "final_count": final_count,
        "gps_summary": str(gps_path),
    }
    _write_preprocess_summary(job_folder, summary)
    log(job_folder, f"Preprocessing photoset klaar: started={staged_count} final={final_count} blur_removed={blur_info.get('removed')} dedupe_removed={dedupe_info.get('purged_count')}")
    return final_count


def collect_video_files(src_folder: Path) -> list[Path]:
    media = collect_media_files(src_folder)
    vids = media["videos"]
    if not vids:
        raise RuntimeError("No video file found in input folder.")
    return vids


def ffprobe_duration_seconds(job_folder: Path, video: Path) -> float:
    cmd = [
        FFPROBE,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe timed out while reading video duration.")
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError("ffprobe failed to read video duration.")
    return float(r.stdout.strip())


def extract_frames_from_video(job_folder: Path, src_folder: Path, preset: str) -> int:
    preset_key = resolve_preset(requested_preset=preset)
    cfg = PRESETS[preset_key]

    interval = float(cfg["sample_interval"])
    max_width = int(cfg["max_width"])
    max_frames = int(cfg["max_frames"])
    jpeg_quality = int(cfg["jpeg_quality"])

    fps = 1.0 / interval
    staging = frames_dir(job_folder)
    videos = collect_video_files(src_folder)
    per_video_cap = max(1, max_frames // max(1, len(videos)))
    remainder = max_frames - (per_video_cap * len(videos))

    vf = f"fps={fps},scale='min({max_width},iw)':-2"
    total_count = 0
    global_frame_idx = 0

    for idx, video in enumerate(videos, start=1):
        duration = ffprobe_duration_seconds(job_folder, video)
        this_cap = per_video_cap + (1 if idx <= remainder else 0)
        out_pattern = str(staging / f"_tmp_v{idx:02d}_%06d.jpg")
        started = datetime.now()
        log(
            job_folder,
            f"Extracting frames from video {idx}/{len(videos)}: {video.name} "
            f"(duration={duration:.1f}s, preset={preset_key}, fps={fps:.4f}, max_frames={this_cap})"
        )
        cmd = [
            FFMPEG,
            "-hide_banner",
            "-y",
            "-i", str(video),
            "-vf", vf,
            "-q:v", str(jpeg_quality),
            "-frames:v", str(this_cap),
            out_pattern,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(job_folder, f"ffmpeg stderr ({video.name}):\n{r.stderr[-4000:]}")
            raise RuntimeError(f"ffmpeg frame extraction failed for {video.name}.")
        temp_frames = sorted(staging.glob(f"_tmp_v{idx:02d}_*.jpg"), key=lambda p: p.name.lower())
        extracted = 0
        for temp_frame in temp_frames:
            global_frame_idx += 1
            final_frame = staging / f"frame_{global_frame_idx:06d}.jpg"
            temp_frame.replace(final_frame)
            _auto_orient_and_downscale(final_frame, job_folder, downscale_threshold=DOWNSCALE_THRESHOLD, downscale_to=DOWNSCALE_TO)
            extracted += 1
        total_count += extracted
        elapsed = (datetime.now() - started).total_seconds()
        log(job_folder, f"Video klaar: {video.name} duur={duration:.1f}s frames={extracted} verwerkt_in={elapsed:.1f}s")

    bad_files = verify_images(staging)
    bad_moved = move_bad_files(job_folder, staging, bad_files)
    blur_info = _remove_blurry_frames(job_folder, staging, preset_key)
    dedupe_info = _run_dedupe_headless(job_folder, staging, preset_key)
    final_count = _normalize_frame_sequence(job_folder, staging)
    gps_path = write_gps_summary(job_folder, src_folder, staging)

    summary = {
        "created_at": iso_now(),
        "input_type": "video",
        "preset": preset_key,
        "extracted_count": total_count,
        "bad_files_moved": bad_moved,
        "blur": blur_info,
        "dedupe": {
            "status": dedupe_info.get("status"),
            "kept_count": dedupe_info.get("kept_count"),
            "purged_count": dedupe_info.get("purged_count"),
            "threshold": PRESETS[preset_key]["dedupe_threshold"],
        },
        "final_count": final_count,
        "gps_summary": str(gps_path),
    }
    _write_preprocess_summary(job_folder, summary)
    log(job_folder, f"Preprocessing video klaar: extracted={total_count} final={final_count} blur_removed={blur_info.get('removed')} dedupe_removed={dedupe_info.get('purged_count')}")
    return final_count


def extract_sharp_frames(job_folder: Path, src_folder: Path, preset: str) -> int:
    return extract_frames_from_video(job_folder, src_folder, preset)


def normalize_photos(job_folder: Path, src_folder: Path) -> int:
    return preprocess_photoset(job_folder, src_folder)


def parse_preset(tag: str = "", requested_preset: str = "", explicit_preset: str = "") -> str:
    chosen = explicit_preset or requested_preset
    return resolve_preset(tag=tag, requested_preset=chosen)
