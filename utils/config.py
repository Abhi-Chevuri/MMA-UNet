"""
utils/config.py
================
Loads configs/config.yaml and flattens it into the flat-dict "CFG" shape
that the rest of the codebase (datasets/, models/, losses/, utils/) expects
— i.e. cfg["IMG_SIZE"], cfg["BATCH_SIZE"], cfg["LR"], etc.

Usage
-----
    from utils.config import load_config

    cfg = load_config("configs/config.yaml")
    cfg = load_config("configs/config.yaml", overrides=["BATCH_SIZE=32", "LR=5e-5"])
"""
from __future__ import annotations

import os
import yaml


def _flatten(nested: dict) -> dict:
    """Flatten the top-level sections (paths/data/train/model/loss/eval/predict)
    into a single dict of UPPER_CASE keys, matching the original notebook's CFG.
    """
    flat = {}
    for _section, values in nested.items():
        if not isinstance(values, dict):
            flat[_section] = values
            continue
        for k, v in values.items():
            flat[k] = v
    return flat


def _cast(value: str):
    """Best-effort cast of a CLI override string to int/float/bool."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def load_config(path: str = "configs/config.yaml", overrides: list[str] | None = None) -> dict:
    """Load YAML config -> flat dict, apply optional KEY=VALUE overrides,
    and make sure every output directory referenced in `paths` exists.
    """
    with open(path, "r") as f:
        nested = yaml.safe_load(f)

    cfg = _flatten(nested)

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"--set override '{item}' must be KEY=VALUE")
            key, val = item.split("=", 1)
            cfg[key.strip()] = _cast(val.strip())

    # Ensure parent directories for every *_PATH / *_DIR entry exist.
    for key, val in cfg.items():
        if isinstance(val, str) and (key.endswith("_PATH") or key.endswith("_DIR")):
            target_dir = val if key.endswith("_DIR") else os.path.dirname(val)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)

    return cfg
