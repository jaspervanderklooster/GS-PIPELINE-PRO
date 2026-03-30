# apply_colmap_fallback_patch.ps1
# Run from repo root (D:\GS_PIPELINE-PRO)
$RepoRoot = "D:\GS_PIPELINE-PRO"
Set-Location $RepoRoot

# backup
Copy-Item -Path ".\worker.py" -Destination ".\worker.py.bak" -Force
Write-Output "Backup created: worker.py.bak"

# read file
$text = Get-Content .\worker.py -Raw

# define new block (replace between marker and def find_best_artifact)
$newblock = @'
# --- BEGIN: COLMAP fallback wrapper ---
def _is_gpu_related_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    markers = (
        "cuda", "out of memory", "memory", "out of mem", "cudaerror",
        "allocation", "failed to allocate", "memory exhausted"
    )
    return any(marker in text for marker in markers)


def run_colmap_with_fallback(job_path: Path, image_dir: Path, workspace: Path, requested_preset: str) -> Path:
    """
    Improved fallback:
     - Try requested preset (e.g. hq)
     - If GPU/memory error: try <requested>_safe (if present)
     - Then try standard_safe
     - CPU-only attempts only as very last resort
    Records attempts in job.meta.fallback_history and sets job.preset_used on success.
    """
    job_folder = job_path.parent
    job = ensure_job_shape(load_json(job_path))

    normalized_preset = map_colmap_preset(requested_preset)

    # --- robustly determine safe_variant (import colmap_runner if available) ---
    try:
        import colmap_runner  # type: ignore
        available_presets = set(colmap_runner.COLMAP_PRESET_ARGS.keys())
    except Exception:
        available_presets = set()
    safe_variant = f"{normalized_preset}_safe" if f"{normalized_preset}_safe" in available_presets else "standard_safe"

    # --- helper: free GPU memory in MB (best-effort) ---
    def gpu_free_mb() -> int:
        try:
            import subprocess
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

    # helper to record fallback attempts (with telemetry)
    def record_attempt(preset_name: str, force_cpu: bool, success: bool, note: str = ""):
        meta = job.setdefault("meta", {})
        hist = meta.setdefault("fallback_history", [])
        attempt_index = len(hist) + 1
        gpu_at = None
        try:
            gpu_at = gpu_free_mb()
        except Exception:
            gpu_at = None
        hist.append({
            "attempt_index": attempt_index,
            "preset": preset_name,
            "force_cpu": bool(force_cpu),
            "ts": iso_now(),
            "success": bool(success),
            "note": note,
            "gpu_free_mb": gpu_at,
        })
        # simple retries counter
        meta["colmap_retries"] = int(meta.get("colmap_retries", 0)) + (0 if success else 1)
        job["meta"] = meta
        persist_job(job_path, job)

    # Build ordered attempts
    if normalized_preset == "hq":
        attempts = [
            (normalized_preset, False),
            (safe_variant, False) if safe_variant != normalized_preset else None,
            ("standard_safe", False),
            (safe_variant, True),
            ("standard_safe", True),
        ]
    else:
        attempts = [
            (normalized_preset, False),
            ("standard_safe", False),
            ("standard_safe", True),
        ]

    attempts = [a for a in attempts if a]

    # --- optional GPU check: if low free memory skip full 'hq' attempt ---
    free_mb = gpu_free_mb()
    if normalized_preset == "hq" and free_mb and free_mb < 12000:
        log(job_folder, f"GPU free {free_mb}MB < 12000MB — skipping heavy 'hq' attempt and starting with safe variant.")
        attempts = [a for a in attempts if a[0] != normalized_preset]

    last_exc = None
    for idx, (preset_name, force_cpu) in enumerate(attempts, start=1):
        try:
            fused = _attempt(preset_name, force_cpu=force_cpu)
            job["preset_used"] = preset_name
            record_attempt(preset_name, force_cpu, success=True)
            persist_job(job_path, job)
            return fused
        except Exception as exc:
            last_exc = exc
            record_attempt(preset_name, force_cpu, success=False, note=str(exc))
            if not _is_gpu_related_error(exc):
                log(job_folder, f"COLMAP attempt {preset_name} failed (non-GPU error): {exc}")
            else:
                log(job_folder, f"COLMAP GPU/memory error on {preset_name}: {exc}")
            # small backoff before next attempt to avoid tight failure loops
            try:
                time.sleep(5)
            except Exception:
                pass

    log(job_folder, f"All COLMAP attempts failed (tried: {attempts}). Last error: {last_exc}")
    raise RuntimeError(f"COLMAP fallback exhausted. Last error: {last_exc}") from last_exc
# --- END: COLMAP fallback wrapper ---
'@

# find region to replace: from marker to before def find_best_artifact
$pattern = '(?s)# --- BEGIN: COLMAP fallback wrapper ---.*?(?=\r?\ndef find_best_artifact)'
$rx = [regex]::new($pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
$m = $rx.Match($text)
if (-not $m.Success) {
    Write-Error "Could not find the expected COLMAP fallback wrapper marker in worker.py. Aborting."
    exit 1
}

$newtext = $text.Substring(0, $m.Index) + $newblock + $text.Substring($m.Index + $m.Length)
# write back
$newtext | Set-Content -Path .\worker.py -Encoding UTF8 -Force
Write-Output "Patched worker.py successfully."

# syntax-check
& .\.venv\Scripts\Activate.ps1
python -m py_compile worker.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "py_compile failed. Restoring backup."
    Copy-Item -Path ".\worker.py.bak" -Destination ".\worker.py" -Force
    exit 1
}
Write-Output "worker.py compiled OK."

# diff (simple)
Write-Output "Showing diff (first 200 lines of new worker.py for context):"
Get-Content .\worker.py -TotalCount 200 | Select-Object -First 200