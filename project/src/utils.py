"""
utils.py - Shared utilities: config loading, seeding, logging, timing.
"""
from __future__ import annotations
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML configuration file and return as nested dict."""
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_shuffled_group_kfold(
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Shuffles unique group IDs using a specific random seed, then distributes groups
    into n_splits balanced folds so that:
      1. Zero unit/group overlap exists between train and validation within any fold.
      2. Group assignments to folds differ across different seeds.
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)

    # Group counts to balance fold sizes
    group_sizes = {g: int(np.sum(groups == g)) for g in unique_groups}

    # Assign groups to folds greedily by current fold size
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for g in shuffled_groups:
        min_fold = int(np.argmin(fold_sizes))
        folds[min_fold].append(int(g))
        fold_sizes[min_fold] += group_sizes[g]

    splits = []
    for f in range(n_splits):
        val_units = set(folds[f])
        val_mask = np.isin(groups, list(val_units))
        train_mask = ~val_mask
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))

    return splits



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to console (and optionally a file)."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that measures wall-clock time."""

    def __init__(self, name: str = "", logger: logging.Logger | None = None):
        self.name = name
        self.logger = logger

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start
        msg = f"{self.name} took {self.elapsed:.2f}s" if self.name else f"Elapsed: {self.elapsed:.2f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

def log_environment(logger: logging.Logger) -> None:
    """Log Python and key package versions for reproducibility."""
    logger.info("Python %s", sys.version)
    for pkg in ["numpy", "pandas", "sklearn", "xgboost", "torch", "shap", "optuna"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            logger.info("  %s==%s", pkg, ver)
        except ImportError:
            logger.warning("  %s not installed", pkg)
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA not available, using CPU")
