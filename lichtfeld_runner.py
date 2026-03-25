"""LichtFeld runner placeholder for preset mapping."""
from __future__ import annotations

import subprocess
from pathlib import Path

PRESET_ARGS = {
    "standard": ["--iterations", "22000", "--max-cap", "2800000"],
    "hq": ["--iterations", "32000", "--max-cap", "4600000"],
}


def run_lichtfeld(scene_dir: Path, out_dir: Path, preset: str, lichtfeld_bin: str = "lichtfeld") -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if preset == "good":
        preset = "standard"
    elif preset == "high":
        preset = "hq"
    if preset not in PRESET_ARGS:
        preset = "standard"
    cmd = [lichtfeld_bin, "train", "--input", str(scene_dir), "--output", str(out_dir), *PRESET_ARGS[preset]]
    return subprocess.run(cmd, check=False).returncode
