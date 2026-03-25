# utils/config.py
from pathlib import Path
import os
import json

def load_json(path: Path):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def load_config():
    """
    Load configuration with the following priority:
    1. Environment variable GS_CONFIG -> points to a json file
    2. project config/config.json
    3. project config/config.example.json
    Environment variables with same names as keys override values.
    Adds PRESETS from config/presets.yaml if present.
    """
    cfg = {}
    gs_config_env = os.getenv("GS_CONFIG")
    if gs_config_env:
        p = Path(gs_config_env)
        if p.exists():
            cfg = load_json(p)

    if not cfg:
        p = Path("config/config.json")
        if p.exists():
            cfg = load_json(p)

    if not cfg:
        p = Path("config/config.example.json")
        if p.exists():
            cfg = load_json(p)

    # try to load presets.yaml (optional)
    presets_path = Path("config/presets.yaml")
    if presets_path.exists():
        try:
            import yaml
            with presets_path.open('r', encoding='utf-8') as f:
                cfg.setdefault("PRESETS", yaml.safe_load(f))
        except Exception:
            # yaml may not be installed in some dev setups; ignore if missing
            pass

    # override by environment variables (string-only)
    for k in list(cfg.keys()):
        if isinstance(cfg[k], (str, int, float, bool)):
            v = os.getenv(k)
            if v is not None:
                # best-effort convert ints
                if isinstance(cfg[k], int):
                    try:
                        cfg[k] = int(v)
                    except Exception:
                        cfg[k] = v
                else:
                    cfg[k] = v

    return cfg

# convenience getters
_config_cache = None

def get_config():
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache

def get(key, default=None):
    return os.getenv(key, get_config().get(key, default))
