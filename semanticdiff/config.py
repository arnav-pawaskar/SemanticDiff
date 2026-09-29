"""Configuration loading (YAML defaults + optional overrides)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(overrides: Optional[dict[str, Any]] = None,
                path: Optional[Path] = None) -> dict[str, Any]:
    """Load ``config/default.yaml`` and ``config/severity.yaml`` into one dict."""
    with open(path or CONFIG_DIR / "default.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    with open(CONFIG_DIR / "severity.yaml", encoding="utf-8") as fh:
        cfg["severity"] = yaml.safe_load(fh)
    calibrated = CONFIG_DIR / "calibrated.yaml"
    if calibrated.exists():
        with open(calibrated, encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
