from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DedupeResult:
    status: str
    input_dir: str
    output_dir: str
    threshold: float
    compare_width: int
    total_files: int
    kept_count: int
    purged_count: int
    kept_files: list[str]
    purged_files: list[str]
    scores: list[dict[str, Any]]
    average_score: float | None
    min_score: float | None
    max_score: float | None
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless SSIM dedupe for GS_PIPELINE preprocessing")
    p.add_argument("--input-dir", required=True, help="Directory containing images")
    p.add_argument("--output-dir", required=True, help="Output directory for reports/artifacts")
    p.add_argument("--threshold", type=float, default=0.90, help="SSIM threshold")
    p.add_argument("--compare-width", type=int, default=300, help="Resize width for SSIM")
    p.add_argument("--job-id", default="job", help="Job id for purge folder naming")
    p.add_argument("--purge-mode", choices=["move", "delete"], default="move")
    p.add_argument("--fast", action="store_true", help="Reserved for future non-deterministic optimizations")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--worker-log", default=None, help="Path to worker.log (append mode)")
    return p


def configure_logging(level: str, worker_log: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if worker_log:
        worker_log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(worker_log, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def _load_gray_resized(path: Path, width: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Corrupt or unreadable image: {path}")
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    nh = max(1, int(h * scale))
    return cv2.resize(img, (width, nh), interpolation=cv2.INTER_AREA)


def _write_outputs_atomic(output_dir: Path, result: DedupeResult) -> None:
    tmp = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "analysis.json").write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    summary = (
        f"status={result.status}\n"
        f"total={result.total_files}\nkept={result.kept_count}\npurged={result.purged_count}\n"
        f"avg_ssim={result.average_score}\nmin_ssim={result.min_score}\nmax_ssim={result.max_score}\n"
    )
    (tmp / "summary.txt").write_text(summary, encoding="utf-8")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(tmp, output_dir)


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    worker_log = Path(args.worker_log) if args.worker_log else output_dir / "worker.log"
    configure_logging(args.log_level, worker_log)

    if args.fast:
        logging.warning("--fast selected; currently running stable single-threaded mode")

    if not input_dir.exists() or not input_dir.is_dir():
        logging.error("Invalid input directory: %s", input_dir)
        return 2

    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS], key=lambda p: p.name.lower())
    if len(files) < 1:
        logging.error("No input images found")
        return 2

    purge_final = output_dir / f"purged_duplicates_job_{args.job_id}"
    purge_temp = output_dir.parent / f".{purge_final.name}.tmp-{uuid.uuid4().hex}"
    purge_temp.mkdir(parents=True, exist_ok=True)

    kept: list[str] = []
    purged: list[str] = []
    scores: list[dict[str, Any]] = []

    try:
        prev = _load_gray_resized(files[0], args.compare_width)
        kept.append(files[0].name)

        for path in files[1:]:
            curr = _load_gray_resized(path, args.compare_width)
            score = _ssim_gray(prev, curr)
            scores.append({"file": path.name, "score": score})
            if score > args.threshold:
                purged.append(path.name)
                if not args.dry_run:
                    if args.purge_mode == "move":
                        shutil.move(str(path), str(purge_temp / path.name))
                    else:
                        path.unlink(missing_ok=True)
            else:
                kept.append(path.name)
                prev = curr
            # free current image memory and hint GC to reduce memory pressure
            try:
                del curr
            except Exception:
                pass
            gc.collect()

        if not args.dry_run and args.purge_mode == "move":
            if purge_final.exists():
                shutil.rmtree(purge_final)
            os.replace(purge_temp, purge_final)
        else:
            shutil.rmtree(purge_temp, ignore_errors=True)

        stat_scores = [x["score"] for x in scores]
        result = DedupeResult(
            status="ok",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            threshold=args.threshold,
            compare_width=args.compare_width,
            total_files=len(files),
            kept_count=len(kept),
            purged_count=len(purged),
            kept_files=kept,
            purged_files=purged,
            scores=scores,
            average_score=float(np.mean(stat_scores)) if stat_scores else None,
            min_score=float(min(stat_scores)) if stat_scores else None,
            max_score=float(max(stat_scores)) if stat_scores else None,
        )
        _write_outputs_atomic(output_dir, result)
        logging.info("Dedupe done: kept=%s purged=%s", len(kept), len(purged))
        return 0
    except Exception as exc:
        logging.exception("Fatal dedupe error")
        result = DedupeResult(
            status="failed",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            threshold=args.threshold,
            compare_width=args.compare_width,
            total_files=len(files),
            kept_count=len(kept),
            purged_count=len(purged),
            kept_files=kept,
            purged_files=purged,
            scores=scores,
            average_score=None,
            min_score=None,
            max_score=None,
            error=str(exc),
        )
        _write_outputs_atomic(output_dir, result)
        return 1


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))

