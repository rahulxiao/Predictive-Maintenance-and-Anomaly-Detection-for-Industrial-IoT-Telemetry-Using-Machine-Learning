"""
models.py - ML/DL models for RUL prediction on C-MAPSS.

Models:
  - RandomForestRUL   (sklearn)
  - XGBoostRUL        (xgboost)
  - LSTMModel         (PyTorch)
  - CNNModel          (PyTorch)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ---------------------------------------------------------------------------
# Sklearn / flat-feature wrappers
# ---------------------------------------------------------------------------

def flatten_windows(X: np.ndarray) -> np.ndarray:
    """Flatten (N, W, F) → (N, W*F) for sklearn models."""
    return X.reshape(X.shape[0], -1)


class RandomForestRUL:
    """Thin wrapper around sklearn RandomForestRegressor."""

    def __init__(self, cfg: dict[str, Any]):
        rf_cfg = cfg["models"]["random_forest"]
        self.model = RandomForestRegressor(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            random_state=cfg["models"]["random_seed"],
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(flatten_windows(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(flatten_windows(X))

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


class XGBoostRUL:
    """Thin wrapper around XGBoost regressor."""

    def __init__(self, cfg: dict[str, Any]):
        if not HAS_XGB:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        xgb_cfg = cfg["models"]["xgboost"]
        self.model = xgb.XGBRegressor(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            random_state=cfg["models"]["random_seed"],
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(flatten_windows(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(flatten_windows(X))


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class RULDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Stacked LSTM for sequence-to-scalar RUL regression."""

    def __init__(self, n_features: int, cfg: dict[str, Any]):
        super().__init__()
        lcfg = cfg["models"]["lstm"]
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lcfg["hidden_size"],
            num_layers=lcfg["num_layers"],
            batch_first=True,
            dropout=lcfg["dropout"] if lcfg["num_layers"] > 1 else 0.0,
        )
        self.dropout = nn.Dropout(lcfg["dropout"])
        self.fc = nn.Linear(lcfg["hidden_size"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # last time-step
        return self.fc(out).squeeze(1)


# ---------------------------------------------------------------------------
# 1-D CNN
# ---------------------------------------------------------------------------

class CNNModel(nn.Module):
    """1-D Convolutional network for time-series RUL regression."""

    def __init__(self, n_features: int, window_length: int, cfg: dict[str, Any]):
        super().__init__()
        ccfg = cfg["models"]["cnn"]
        channels = ccfg["channels"]
        ks = ccfg["kernel_size"]

        layers: list[nn.Module] = []
        in_ch = n_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch),
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(ccfg["dropout"])
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F) → (B, F, W) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)
