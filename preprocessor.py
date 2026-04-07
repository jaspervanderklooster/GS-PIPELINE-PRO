import shutil
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime
import os

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:  # Pillow optioneel in sommige omgevingen
    Image = None
    ImageOps = None
    UnidentifiedImageError = Exception


FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

ALLOWED_PHOTOS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEOS = {".mp4", ".mov", ".avi", ".mkv"}
ALLOWED_ARCHIVES = {".zip"}

PRESETS = {
    "good": {
        "sample_interval": 3.0,
        "max_width": 2560,
        "max_frames": 1200,
        "jpeg_quality": 2,
    },
    "high": {
        "sample_interval": 2.0,
        "max_width": 3840,
        "max_frames": 1800,
        "jpeg_quality": 2,
    },
}


def iso_now():
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


def collect_media_files(src_folder: Path) -> dict:
    photos = []
    videos = []
    archives = []

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

    photos.sort(key=lambda p: str(p).lower())
    videos.sort(key=lambda p: str(p).lower())
    archives.sort(key=lambda p: str(p).lower())

    return {
        "photos": photos,
        "videos": videos,
        "archives": archives,
    }


def detect_counts(src_folder: Path) -> dict:
    media = collect_media_files(src_folder)
    return {
        "photos": len(media["photos"]),
        "videos": len(media["videos"]),
        "zips": len(media["archives"]),
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


def preprocess_photoset(job_folder: Path, src_folder: Path) -> int:
    log_pillow_status(job_folder)
    bad_files = reencode_and_downscale(job_folder, src_folder)
    bad_files.extend(verify_images(src_folder))
    move_bad_files(job_folder, src_folder, bad_files)
    run_dedupe_headless(job_folder, src_folder)

    staging = frames_dir(job_folder)
    media = collect_media_files(src_folder)
    photos = media["photos"]

    moved = 0
    for idx, p in enumerate(photos, start=1):
        ext = p.suffix.lower()
        target = staging / f"frame_{idx:06d}{ext}"

        if target.exists():
            continue

        try:
            shutil.move(str(p), str(target))
        except Exception:
            shutil.copy2(str(p), str(target))
            try:
                p.unlink()
            except Exception:
                pass

        moved += 1

    log(job_folder, f"Moved {moved} photos to {staging}")
    return moved


def log_pillow_status(job_folder: Path):
    if Image is None:
        log(job_folder, "Pillow not available — skipping reencode")
        return
    log(job_folder, f"PIL available: {getattr(Image, '__version__', 'unknown')}")


def reencode_and_downscale(job_folder: Path, src_dir: Path, max_side: int = 3000) -> list[tuple[str, str]]:
    if Image is None or ImageOps is None:
        return []
    bad: list[tuple[str, str]] = []
    for p in sorted(src_dir.glob("*.*")):
        if p.suffix.lower() not in ALLOWED_PHOTOS:
            continue
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                if max(im.size) > max_side:
                    scale = max_side / float(max(im.size))
                    new_size = (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)))
                    im = im.resize(new_size, Image.LANCZOS)
                im.save(p, format="JPEG", quality=95, optimize=True)
        except UnidentifiedImageError:
            bad.append((p.name, "UnidentifiedImageError"))
        except Exception as exc:
            bad.append((p.name, str(exc)))
    if bad:
        log(job_folder, f"reencode/downscale detected bad files: {bad}")
    return bad


def verify_images(src_dir: Path) -> list[tuple[str, str]]:
    if Image is None:
        return []
    bad: list[tuple[str, str]] = []
    for f in sorted(src_dir.glob("*.*")):
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
        return
    bad_dir = src_dir / "_bad"
    bad_dir.mkdir(parents=True, exist_ok=True)
    for filename, reason in bad:
        source = src_dir / filename
        if not source.exists():
            continue
        target = bad_dir / filename
        try:
            shutil.move(str(source), str(target))
            log(job_folder, f"Bad file moved: {filename} ({reason})")
        except Exception as exc:
            log(job_folder, f"Failed moving bad file {filename}: {exc}")


def run_dedupe_headless(job_folder: Path, src_dir: Path):
    script = Path(__file__).resolve().parent / "dedupe_headless.py"
    if not script.exists():
        return
    out_dir = job_folder / "preprocess" / "dedupe"
    cmd = [
        os.environ.get("PYTHON", "python"),
        str(script),
        "--input-dir", str(src_dir),
        "--output-dir", str(out_dir),
        "--job-id", job_folder.name,
        "--worker-log", str(job_folder / "worker.log"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(job_folder, f"dedupe_headless failed (non-fatal): {result.stderr[-2000:]}")
    else:
        log(job_folder, "dedupe_headless completed.")


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
    preset = resolve_preset(requested_preset=preset)
    cfg = PRESETS[preset]

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
            f"(duration={duration:.1f}s, preset={preset}, fps={fps:.4f}, max_frames={this_cap})"
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
            extracted += 1
        total_count += extracted
        elapsed = (datetime.now() - started).total_seconds()
        log(job_folder, f"Video klaar: {video.name} duur={duration:.1f}s frames={extracted} verwerkt_in={elapsed:.1f}s")

    log(job_folder, f"Extracted totaal {total_count} frames uit {len(videos)} video('s) naar {staging}")
    return total_count
def extract_sharp_frames(job_folder: Path, src_folder: Path, preset: str) -> int:
    return extract_frames_from_video(job_folder, src_folder, preset)

def normalize_photos(job_folder: Path, src_folder: Path) -> int:
    return preprocess_photoset(job_folder, src_folder)

def parse_preset(tag: str = "", requested_preset: str = "", explicit_preset: str = "") -> str:
    chosen = explicit_preset or requested_preset
    return resolve_preset(tag=tag, requested_preset=chosen)
