"""
trainer.py - Training loops for PyTorch models.

Key fixes vs previous version:
  - RUL targets are normalised to [0,1] (divide by rul_clip) before loss,
    inverted before metric computation. This dramatically stabilises training.
  - LR scheduler: removed deprecated `verbose` kwarg.
  - Gradient clipping norm 1.0 retained.
  - Returns epoch-by-epoch train/val RMSE for curve inspection.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Any

from models import RULDataset
from utils import get_device, get_logger


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
    groups: np.ndarray | None = None,
    model_key: str = "lstm",
    logger=None,
) -> tuple[nn.Module, list[float], list[float]]:
    """
    Train a PyTorch model with normalised RUL targets.

    Guarantees:
      - Targets are normalised to [0,1] before loss, inverted at evaluation.
      - If groups (engine unit IDs) is provided, train/val split is strictly partitioned
        by unit ID so that zero units overlap between train and validation.
      - Test engines are NEVER accessed or passed to this function.
    """
    if logger is None:
        logger = get_logger("trainer")

    rul_clip  = cfg["evaluation"]["rul_clip"]
    mcfg      = cfg["models"][model_key]
    device    = get_device()
    model     = model.to(device)

    # Normalise targets to [0,1]
    y_norm = y_train / rul_clip

    val_frac = cfg["models"]["validation_fraction"]
    seed     = cfg["models"]["random_seed"]

    if groups is not None:
        unique_units = np.unique(groups)
        rng = np.random.default_rng(seed)
        n_val_units = max(1, int(len(unique_units) * val_frac))
        val_units = set(rng.choice(unique_units, size=n_val_units, replace=False))
        val_mask = np.isin(groups, list(val_units))
        train_mask = ~val_mask

        train_ds = RULDataset(X_train[train_mask], y_norm[train_mask])
        val_ds   = RULDataset(X_train[val_mask],   y_norm[val_mask])
        n_train  = int(train_mask.sum())
        n_val    = int(val_mask.sum())
        logger.info("  [Unit Split] %d train units (%d windows) | %d val units (%d windows)",
                    len(unique_units) - n_val_units, n_train, n_val_units, n_val)
    else:
        dataset = RULDataset(X_train, y_norm)
        n_val   = max(1, int(len(dataset) * val_frac))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )

    train_loader = DataLoader(train_ds, batch_size=mcfg["batch_size"],
                              shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=mcfg["batch_size"],
                              shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg["learning_rate"])
    # ReduceLROnPlateau without deprecated verbose kwarg
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val      = float("inf")
    patience      = mcfg["patience"]
    wait          = 0
    best_state    = None
    train_hist: list[float] = []
    val_hist:   list[float] = []

    for epoch in range(1, mcfg["max_epochs"] + 1):
        # --- Train ---
        model.train()
        train_mse = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_mse += loss.item() * len(yb)
        train_mse /= n_train

        # --- Validate ---
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_mse += criterion(model(xb), yb).item() * len(yb)
        val_mse /= n_val

        scheduler.step(val_mse)

        # Convert back to original units for reporting
        train_rmse = float(train_mse ** 0.5) * rul_clip
        val_rmse   = float(val_mse   ** 0.5) * rul_clip
        train_hist.append(train_rmse)
        val_hist.append(val_rmse)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                "  Epoch %3d/%d | train_RMSE=%6.2f | val_RMSE=%6.2f | lr=%.2e",
                epoch, mcfg["max_epochs"], train_rmse, val_rmse, lr_now
            )

        if val_mse < best_val:
            best_val  = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("  Early stopping at epoch %d.", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), train_hist, val_hist


def predict_torch(model: nn.Module, X: np.ndarray,
                  rul_clip: float, batch_size: int = 512) -> np.ndarray:
    """
    Inference: output is in normalised [0,1] space inside model;
    we multiply by rul_clip to return predictions in original RUL units.
    """
    device = get_device()
    model  = model.to(device).eval()
    preds  = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size],
                               dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds) * rul_clip
