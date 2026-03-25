# scripts/check_runtime.py
import sys
import shutil
import subprocess
import json
from pathlib import Path

def which_bin(name):
    return shutil.which(name)

def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return 1, "", str(e)

def check_python(min_ver=(3,12)):
    ok = sys.version_info >= min_ver
    return ok, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def check_packages(pkgs):
    missing = []
    for p in pkgs:
        try:
            __import__(p)
        except Exception:
            missing.append(p)
    return missing

def check_nvidia():
    n = which_bin("nvidia-smi")
    if not n:
        return False, "nvidia-smi not found in PATH"
    rc, out, err = run_cmd([n, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"])
    if rc != 0:
        return False, f"nvidia-smi returned error: {err or out}"
    return True, out.strip()

def check_binaries(bins):
    ok = {}
    for b in bins:
        path = which_bin(b)
        ok[b] = path or None
    return ok

if __name__ == "__main__":
    print("GS_PIPELINE runtime check")
    py_ok, py_v = check_python()
    print("Python:", py_v, "OK" if py_ok else "TOO OLD - require >=3.12")

    pkgs = ["cv2", "numpy", "PIL", "yaml"]
    missing = check_packages(pkgs)
    print("Python packages missing:", missing or 'none')

    bins = ["colmap","ffmpeg","ffprobe","magick","nvidia-smi"]
    bin_status = check_binaries(bins)
    print("Binaries (in PATH or accessible):")
    print(json.dumps(bin_status, indent=2))

    nv_ok, nv_info = check_nvidia()
    print("NVIDIA:", "OK" if nv_ok else "MISSING/ERROR")
    if nv_ok:
        print(nv_info)
    else:
        print(nv_info)
