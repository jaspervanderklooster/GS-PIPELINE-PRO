"""GPU telemetry helpers backed by nvidia-smi."""
from __future__ import annotations

import subprocess
import time


def gpu_free_mb() -> int | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return None
        value = (r.stdout or "").strip().splitlines()[0].strip()
        return int(value)
    except Exception:
        return None


def monitor_peak(pid: int | None, interval: float = 1.0, timeout: float = 1.0) -> int | None:
    """Best-effort peak monitor; polls briefly when used post-run."""
    if not pid:
        return None
    peak = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            if r.returncode == 0:
                for line in (r.stdout or "").splitlines():
                    cols = [c.strip() for c in line.split(",")]
                    if len(cols) < 2:
                        continue
                    if int(cols[0]) == int(pid):
                        peak = max(peak, int(cols[1]))
        except Exception:
            pass
        time.sleep(interval)
    return peak or None
