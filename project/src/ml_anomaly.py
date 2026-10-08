"""
ml_anomaly.py - Phase C & D: Machine Learning Anomaly Detection and Statistical Evaluation.

Models:
  1. Isolation Forest: Unsupervised, trained strictly on early healthy-life train data.
  2. Random Forest: Class-weighted, trained on window features (mean, std, slope, last).
  3. XGBoost: Class-weighted (scale_pos_weight), trained on window features.
  4. LSTM Classifier: PyTorch recurrent sequence classifier with pos_weight BCE loss.

Threshold Tuning (on validation engines only):
  - Threshold 1: Maximizes F1 score on validation engines.
  - Threshold 2: Matches the False Alarm Rate (FAR) of the best baseline.

Evaluation (Phase D):
  - 5-Fold GroupKFold by unit across full trajectories.
  - 5 Seeds (0, 1, 2, 3, 4) -> Reports Mean +- Std.
  - Metrics: Precision, Recall, F1, PR-AUC, ROC-AUC, FDR, Detection Rate, Lead Time, Premature Share.
  - Wilcoxon signed-rank test between best ML model and best baseline.
  - Window-level classification metrics on official test engines.
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from features_extractor import extract_window_features
from metrics_lead_time import compute_sustained_alarms, evaluate_engine_lead_time, evaluate_fleet_lead_time
from utils import get_device, get_logger, set_seed

logger = get_logger("ml_anomaly")


# ---------------------------------------------------------------------------
# 1. PyTorch LSTM Classifier
# ---------------------------------------------------------------------------

class PyTorchLSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :]).squeeze(-1)
        return logits


def train_lstm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    max_epochs: int = 25,
    batch_size: int = 256,
) -> tuple[nn.Module, float]:
    """Train small LSTM binary classifier with early stopping on val loss."""
    set_seed(seed)
    device = get_device()
    n_features = X_train.shape[2]
    model = PyTorchLSTMClassifier(n_features=n_features, hidden_size=128).to(device)

    # Class weighting
    pos_count = max(1.0, float(np.sum(y_train == 1)))
    neg_count = max(1.0, float(np.sum(y_train == 0)))
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    vl_ds = TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.float32))

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(yb)
        val_loss /= len(vl_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to(device).eval(), best_val_loss


def predict_lstm_probs(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    device = get_device()
    model = model.to(device).eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)


# ---------------------------------------------------------------------------
# 2. Anomaly Classifier Wrappers
# ---------------------------------------------------------------------------

class MLAnomalyDetector:
    def __init__(self, model_name: str, seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self.model: Any = None
        self.threshold_f1: float = 0.5
        self.threshold_far: float = 0.5
        self.s_min: float = 0.0
        self.s_max: float = 1.0

    def fit(
        self,
        X_train_raw: np.ndarray,
        X_train_feats: np.ndarray,
        y_train: np.ndarray,
        healthy_mask_train: np.ndarray,
    ) -> "MLAnomalyDetector":
        set_seed(self.seed)

        if self.model_name == "IsolationForest":
            # Trained on healthy early-life train data only
            X_healthy = X_train_feats[healthy_mask_train]
            self.model = IsolationForest(
                n_estimators=100,
                max_samples=min(1024, len(X_healthy)),
                contamination="auto",
                random_state=self.seed,
                n_jobs=-1,
            )
            self.model.fit(X_healthy)
            # Calibrate min/max score on training healthy data
            tr_scores = -self.model.decision_function(X_healthy)
            self.s_min = float(np.percentile(tr_scores, 1))
            self.s_max = float(np.percentile(tr_scores, 99))
            if self.s_max <= self.s_min:
                self.s_max = self.s_min + 1.0

        elif self.model_name == "RandomForest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                max_samples=0.8,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
            )
            self.model.fit(X_train_feats, y_train)

        elif self.model_name == "XGBoost":
            pos = float(np.sum(y_train == 1))
            neg = float(np.sum(y_train == 0))
            scale_pos = (neg / pos) if pos > 0 else 1.0
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                tree_method="hist",
                random_state=self.seed,
                n_jobs=-1,
            )
            self.model.fit(X_train_feats, y_train)

        return self

    def predict_score(self, X_raw: np.ndarray, X_feats: np.ndarray) -> np.ndarray:
        if self.model_name == "IsolationForest":
            # decision_function gives negative for anomalies -> invert
            raw_scores = -self.model.decision_function(X_feats)
            scores = (raw_scores - self.s_min) / (self.s_max - self.s_min + 1e-8)
            return np.clip(scores, 0.0, 1.0)

        elif self.model_name in ["RandomForest", "XGBoost"]:
            return self.model.predict_proba(X_feats)[:, 1]

        elif self.model_name == "LSTM":
            return predict_lstm_probs(self.model, X_raw)

        raise ValueError(f"Unknown model name {self.model_name}")

    def tune_thresholds(
        self,
        scores_val: np.ndarray,
        y_val: np.ndarray,
        target_baseline_far: float = 0.05,
    ) -> tuple[float, float]:
        """
        Choose thresholds on validation engines:
          (1) maximize F1
          (2) match target_baseline_far
        """
        candidate_threshs = np.linspace(0.02, 0.98, 97)
        best_f1 = -1.0
        best_th_f1 = 0.5
        best_th_far = 0.5
        min_far_diff = float("inf")

        healthy_mask = (y_val == 0)

        for th in candidate_threshs:
            preds = (scores_val >= th).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th_f1 = th

            far = (preds[healthy_mask] == 1).mean() if healthy_mask.sum() > 0 else 0.0
            far_diff = abs(far - target_baseline_far)
            if far_diff < min_far_diff:
                min_far_diff = far_diff
                best_th_far = th

        self.threshold_f1 = float(best_th_f1)
        self.threshold_far = float(best_th_far)
        return self.threshold_f1, self.threshold_far


# ---------------------------------------------------------------------------
# 3. Full Trajectory Lead-Time Evaluation for ML Models
# ---------------------------------------------------------------------------

def evaluate_ml_model_lead_time(
    scores: np.ndarray,
    threshold: float,
    val_units: np.ndarray,
    val_cycles: np.ndarray,
    unit_failure_cycles: dict[int, int],
    horizon: int = 100,
    persistence: int = 3,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Evaluate lead-time and alarms engine-by-engine over validation trajectories.
    """
    engine_results = []
    unique_uids = sorted(np.unique(val_units))

    for uid in unique_uids:
        mask = (val_units == uid)
        eng_scores = scores[mask]
        eng_cycles = val_cycles[mask]

        raw_alarms = (eng_scores >= threshold).astype(int)
        sustained = compute_sustained_alarms(raw_alarms, persistence=persistence)
        fail_c = unit_failure_cycles[uid]

        res = evaluate_engine_lead_time(eng_cycles, sustained, failure_cycle=fail_c, horizon=horizon)
        res["unit"] = uid
        engine_results.append(res)

    fleet = evaluate_fleet_lead_time(engine_results, horizon=horizon)
    return fleet, engine_results
