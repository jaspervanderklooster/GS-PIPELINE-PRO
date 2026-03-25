# OPERATIONS - GS_PIPELINE (short guide)

## Overview
This file contains operational notes for the GS pipeline: required versions, driver tips, and environment guidance.

### Required software & versions
- Python 3.12
- NVIDIA drivers compatible with CUDA 12.x (tuned for RTX 5000 Ada)
- nvidia-container-toolkit (for GPU in Docker)
- COLMAP (pin an exact version or commit)
- ffmpeg, ffprobe
- ImageMagick (magick)
- OpenCV (cv2), numpy, Pillow
- Optional: PyYAML (for presets.yaml)

### Driver / CUDA upgrade note (important)
When upgrading major NVIDIA drivers (e.g. 550 → 580 → 590), prefer a clean route:
1. Stop GPU workloads and CI runners.
2. Purge old nvidia packages (apt/dpkg/win uninstall as appropriate).
3. Reboot.
4. Install new driver + CUDA runtime.
5. Reboot again.
6. Reinstall toolchains (Docker nvidia-support, colmap if built from source).
This prevents dpkg/apt loops and leftover mismatched binaries. On Windows, reinstall drivers via NVIDIA installer and re-check `nvidia-smi`.

### Docker
- Base image recommendation: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- Install python 3.12 in Docker or use an Ubuntu image + apt install python3.12
- Use `--gpus all` when running heavy GPU jobs.
- Expose config via ENV variables (GS_CONFIG or explicit envs).

### Paths & Runtime workspace (example windows)
- `D:\GS_PIPELINE\inbox` — user uploads
- `D:\GS_PIPELINE\staging`
- `D:\GS_PIPELINE\processing\<job_id>`
- `D:\GS_PIPELINE\archive\done`
- `D:\GS_PIPELINE\archive\failed`
- `D:\GS_PIPELINE\outbox`
- `D:\GS_PIPELINE\temp\watcher_state\intake_status`
- `D:\GS_PIPELINE\logs`

### CI & Runners
- Unit tests + lint run on the standard CI.
- Integration (end-to-end baseline run) must run on a self-hosted runner with GPU for full verification or on a Docker image with `--gpus`.
- Mark GPU-heavy tests and run them only on runners that advertise `gpu:true`.

### Golden dataset & baseline
- Keep one golden dataset in `data/sample_dataset/` and expected artifacts in `expected/` with checksums.
- Tag a working baseline (e.g. `v0.1-working`) prior to any risky upgrade.

### Troubleshooting checklist
- If COLMAP fails: check `nvidia-smi`, `COLMAP_BIN`, and Docker GPU access.
- If heavy runs crash: try `*_safe` preset which reduces `max_width`/`max_cap`/iterations.
- If you see nondeterministic failures: try running single-threaded / fixed seeds or reduce parallelism in COLMAP.
