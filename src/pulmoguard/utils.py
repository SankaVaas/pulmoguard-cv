"""
Shared utilities: reproducibility, device selection, config loading, logging.

Kept dependency-free (stdlib + yaml + torch only) so this module can be
imported from training, evaluation, inference, and the serving app without
pulling in unnecessary weight.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout with a consistent format.

    Using a named logger (rather than root logger / print statements) means
    downstream consumers (e.g. the FastAPI app, batch scripts) can control
    verbosity independently and log output is properly attributable.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on repeated calls
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed all relevant RNGs for reproducible training runs.

    Full bitwise determinism is not guaranteed across GPU kernels, but this
    removes the dominant sources of run-to-run variance (data shuffling,
    weight init, dropout masks).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU. Single source of truth
    so every module picks the same device consistently."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Raises FileNotFoundError with a clear message rather than a raw yaml
    traceback, since a missing config is a common first-run mistake.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            f"Did you mean to run from the repo root?"
        )
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(base_dir: str | os.PathLike, relative_path: str) -> Path:
    """Resolve a config-relative path against a base directory (typically
    the repo root). Prevents path-resolution bugs when scripts are invoked
    from different working directories."""
    return Path(base_dir) / relative_path
