"""
baselines_extended.py - Part 2: Strengthened Traditional Baselines with 5-Fold GroupKFold.

Key Requirements addressed:
  a) Healthy baseline (mu, sigma) fit strictly on first 30% life of TRAIN engines.
  b) Sensor readings normalized per operating regime (KMeans k=6 on train) for FD002/FD004.
  c) 5-Fold GroupKFold over all train engines to tune:
       - Static: k in [1.5, 2.0, 2.5, 3.0, 3.5], vote_fraction, persistence
       - EWMA: lambda in {0.1, 0.2, 0.3}, L in [1.5, 2.0, 2.5, 3.0], vote_fraction, persistence
  d) Full-trajectory out-of-fold evaluation on train engines using Part 1 lead-time definition
     (H=100 and H=50). Test engines evaluated for window-level classification.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score

from data_loader import load_raw, select_features
from labels import add_anomaly_labels
from metrics_lead_time import compute_sustained_alarms, evaluate_engine_lead_time, evaluate_fleet_lead_time
from regime_norm import OperatingRegimeNormalizer, identify_healthy_early_cycles
from utils import get_logger, load_config, set_seed

logger = get_logger("baselines_ext")


class ParametricStaticDetector:
    def __init__(self, k: float = 2.5, vote_fraction: float = 0.3, persistence: int = 3):
        self.k = k
        self.vote_fraction = vote_fraction
        self.persistence = persistence
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None
        self.feature_cols: list[str] = []

    def fit(self, healthy_train_df: pd.DataFrame, feature_cols: list[str]) -> "ParametricStaticDetector":
        self.feature_cols = feature_cols
        self.means = healthy_train_df[feature_cols].mean()
        self.stds = healthy_train_df[feature_cols].std().replace(0, 1e-5)
        return self

    def predict_engine(self, grp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (raw_alarms, sustained_alarms)."""
        feats = grp[self.feature_cols]
        z_scores = np.abs((feats - self.means) / self.stds)
        raw_alarms = ((z_scores > self.k).mean(axis=1) >= self.vote_fraction).astype(int).values
        sustained_alarms = compute_sustained_alarms(raw_alarms, persistence=self.persistence)
        return raw_alarms, sustained_alarms


class ParametricEWMADetector:
    def __init__(self, ewma_lambda: float = 0.2, L: float = 2.5,
                 vote_fraction: float = 0.3, persistence: int = 3):
        self.ewma_lambda = ewma_lambda
        self.L = L
        self.vote_fraction = vote_fraction
        self.persistence = persistence
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None
        self.feature_cols: list[str] = []

    def fit(self, healthy_train_df: pd.DataFrame, feature_cols: list[str]) -> "ParametricEWMADetector":
        self.feature_cols = feature_cols
        self.means = healthy_train_df[feature_cols].mean()
        self.stds = healthy_train_df[feature_cols].std().replace(0, 1e-5)
        return self

    def predict_engine(self, grp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feats = grp[self.feature_cols]
        ewma_feats = feats.ewm(alpha=self.ewma_lambda, adjust=False).mean()
        factor = np.sqrt(self.ewma_lambda / (2.0 - self.ewma_lambda))
        ucl = self.means + self.L * self.stds * factor
        lcl = self.means - self.L * self.stds * factor
        out_of_control = ((ewma_feats > ucl) | (ewma_feats < lcl)).mean(axis=1).values
        raw_alarms = (out_of_control >= self.vote_fraction).astype(int)
        sustained_alarms = compute_sustained_alarms(raw_alarms, persistence=self.persistence)
        return raw_alarms, sustained_alarms


def evaluate_baseline_on_engines(
    detector: Any,
    df: pd.DataFrame,
    is_test: bool = False,
    test_rul_arr: np.ndarray | None = None,
    n_threshold: int = 30,
    horizon: int = 100,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    """
    Evaluates detector across engines.
    Returns:
      - classification_metrics (Precision, Recall, F1, PR_AUC, ROC_AUC, FAR)
      - lead_time_metrics (Detection_Rate_Pct, Mean_Lead_Time, Median_Lead_Time, Premature_Alarm_Share_Pct, FA_Per_100_Healthy)
      - per_engine_records
    """
    df_lab = add_anomaly_labels(df, n_threshold=n_threshold, is_test=is_test, test_rul_arr=test_rul_arr)
    all_raw_preds = []
    all_sustained_preds = []
    all_trues = []
    engine_results = []
    sorted_units = sorted(df["unit"].unique())

    for i, (unit_id, grp) in enumerate(df_lab.groupby("unit")):
        grp_sorted = grp.sort_values("cycle").reset_index(drop=True)
        raw_p, sust_p = detector.predict_engine(grp_sorted)
        trues = grp_sorted["anomaly_label"].values
        cycles = grp_sorted["cycle"].values

        all_raw_preds.extend(raw_p)
        all_sustained_preds.extend(sust_p)
        all_trues.extend(trues)

        # Failure cycle
        if not is_test:
            fail_c = int(cycles.max())
        else:
            uid_idx = sorted_units.index(unit_id)
            fail_c = int(cycles.max() + test_rul_arr[uid_idx])

        eng_eval = evaluate_engine_lead_time(cycles, sust_p, failure_cycle=fail_c, horizon=horizon)
        eng_eval["unit"] = unit_id
        engine_results.append(eng_eval)

    y_true = np.array(all_trues)
    y_pred = np.array(all_sustained_preds)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    pr_auc = float(average_precision_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0
    roc_auc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    fdr = (1.0 - p) if p > 0 else 0.0

    cls_metrics = {
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "FDR": float(fdr),
        "Prevalence": float(np.mean(y_true)),
    }

    fleet_metrics = evaluate_fleet_lead_time(engine_results, horizon=horizon)
    return cls_metrics, fleet_metrics, engine_results


def tune_and_evaluate_baselines_subset(
    subset_id: int,
    raw_dir: Path,
    n_splits: int = 5,
    horizon: int = 100,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Run 5-Fold GroupKFold baseline tuning and out-of-fold full trajectory evaluation.
    """
    set_seed(random_seed)
    train_raw, test_raw, test_rul = load_raw(raw_dir, subset_id)
    feature_cols = select_features(train_raw, 0.001)

    # Operating regime normalization
    normalizer = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=random_seed)
    normalizer.fit(train_raw, feature_cols, subset_id=subset_id)
    train_norm = normalizer.transform(train_raw)
    test_norm  = normalizer.transform(test_raw)

    units = train_norm["unit"].values
    gkf = GroupKFold(n_splits=n_splits)

    # 1. Parameter Grids per User Specification:
    # Static: k in 1.5-4.0, vote in {0.2, 0.3, 0.5}, persistence in {1, 3, 5}
    # EWMA: lambda in {0.1, 0.2, 0.3}, L in 1.5-4.0, vote in {0.2, 0.3, 0.5}, persistence in {1, 3, 5}
    k_candidates = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    vote_candidates = [0.2, 0.3, 0.5]
    pers_candidates = [1, 3, 5]

    ewma_lams = [0.1, 0.2, 0.3]
    ewma_Ls   = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    # Pre-label full train data with anomaly labels
    train_labeled = add_anomaly_labels(train_norm, n_threshold=30, is_test=False)

    # Dictionary to store out-of-fold predictions across folds for each param
    static_oof_trues: list[np.ndarray] = []
    static_oof_preds: dict[tuple, list[np.ndarray]] = {
        (k, vf, p): [] for k in k_candidates for vf in vote_candidates for p in pers_candidates
    }

    ewma_oof_preds: dict[tuple, list[np.ndarray]] = {
        (lam, L, vf, p): [] for lam in ewma_lams for L in ewma_Ls for vf in vote_candidates for p in pers_candidates
    }

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_norm, groups=units)):
        f_train = train_norm.iloc[tr_idx]
        f_val_lab = train_labeled.iloc[val_idx]
        f_healthy = identify_healthy_early_cycles(f_train, 0.30)

        means = f_healthy[feature_cols].mean().values
        stds = f_healthy[feature_cols].std().replace(0, 1e-5).values

        # Pre-extract validation engine data for fast vectorized iteration
        val_engs = []
        for uid, grp in f_val_lab.groupby("unit"):
            grp_s = grp.sort_values("cycle").reset_index(drop=True)
            feats = grp_s[feature_cols].values
            trues = grp_s["anomaly_label"].values
            val_engs.append((uid, feats, trues))

        # Static thresholding across val engines
        for k in k_candidates:
            for vf in vote_candidates:
                raw_list = []
                for _, feats, _ in val_engs:
                    z = np.abs((feats - means) / stds)
                    raw = ((z > k).mean(axis=1) >= vf).astype(int)
                    raw_list.append(raw)
                for p in pers_candidates:
                    p_preds = []
                    for raw in raw_list:
                        p_preds.append(compute_sustained_alarms(raw, persistence=p))
                    static_oof_preds[(k, vf, p)].append(np.concatenate(p_preds))

        # EWMA across val engines
        for lam in ewma_lams:
            factor = np.sqrt(lam / (2.0 - lam))
            ewma_feats_list = []
            for _, feats, _ in val_engs:
                ewma_f = pd.DataFrame(feats).ewm(alpha=lam, adjust=False).mean().values
                ewma_feats_list.append(ewma_f)

            for L in ewma_Ls:
                ucl = means + L * stds * factor
                lcl = means - L * stds * factor
                for vf in vote_candidates:
                    raw_list = []
                    for ewma_f in ewma_feats_list:
                        ooc = ((ewma_f > ucl) | (ewma_f < lcl)).mean(axis=1)
                        raw = (ooc >= vf).astype(int)
                        raw_list.append(raw)
                    for p in pers_candidates:
                        p_preds = []
                        for raw in raw_list:
                            p_preds.append(compute_sustained_alarms(raw, persistence=p))
                        ewma_oof_preds[(lam, L, vf, p)].append(np.concatenate(p_preds))

        # Record validation ground truths
        fold_trues = np.concatenate([t for _, _, t in val_engs])
        static_oof_trues.append(fold_trues)

    all_oof_trues = np.concatenate(static_oof_trues)

    # Compute out-of-fold F1 for each static param
    best_static_param = None
    best_static_f1 = -1.0
    for param, pred_list in static_oof_preds.items():
        all_p = np.concatenate(pred_list)
        _, _, f1, _ = precision_recall_fscore_support(all_oof_trues, all_p, average="binary", zero_division=0)
        if f1 > best_static_f1:
            best_static_f1 = float(f1)
            best_static_param = param

    # Compute out-of-fold F1 for each EWMA param
    best_ewma_param = None
    best_ewma_f1 = -1.0
    for param, pred_list in ewma_oof_preds.items():
        all_p = np.concatenate(pred_list)
        _, _, f1, _ = precision_recall_fscore_support(all_oof_trues, all_p, average="binary", zero_division=0)
        if f1 > best_ewma_f1:
            best_ewma_f1 = float(f1)
            best_ewma_param = param

    # --- Out-Of-Fold Evaluation on Full Trajectories of Held-Out Train Engines ---
    oof_static_engines = []
    oof_ewma_engines   = []
    oof_static_preds, oof_ewma_preds, oof_trues = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_norm, groups=units)):
        f_train = train_norm.iloc[tr_idx]
        f_val   = train_norm.iloc[val_idx]
        f_healthy = identify_healthy_early_cycles(f_train, 0.30)

        # Optimal static
        k, vf, p = best_static_param
        det_s = ParametricStaticDetector(k=k, vote_fraction=vf, persistence=p).fit(f_healthy, feature_cols)
        _, _, eng_s = evaluate_baseline_on_engines(det_s, f_val, is_test=False, horizon=horizon)
        oof_static_engines.extend(eng_s)

        # Optimal EWMA
        lam, L, vf, p = best_ewma_param
        det_e = ParametricEWMADetector(ewma_lambda=lam, L=L, vote_fraction=vf, persistence=p).fit(f_healthy, feature_cols)
        _, _, eng_e = evaluate_baseline_on_engines(det_e, f_val, is_test=False, horizon=horizon)
        oof_ewma_engines.extend(eng_e)

    oof_fleet_static = evaluate_fleet_lead_time(oof_static_engines, horizon=horizon)
    oof_fleet_ewma   = evaluate_fleet_lead_time(oof_ewma_engines, horizon=horizon)

    # Also at H=50
    oof_fleet_static_h50 = evaluate_fleet_lead_time(oof_static_engines, horizon=50)
    oof_fleet_ewma_h50   = evaluate_fleet_lead_time(oof_ewma_engines, horizon=50)

    # --- Test Window Classification Evaluation with Frozen Parameters ---
    train_healthy_all = identify_healthy_early_cycles(train_norm, 0.30)
    final_det_s = ParametricStaticDetector(*best_static_param).fit(train_healthy_all, feature_cols)
    final_det_e = ParametricEWMADetector(*best_ewma_param).fit(train_healthy_all, feature_cols)

    test_cls_s, _, _ = evaluate_baseline_on_engines(final_det_s, test_norm, is_test=True, test_rul_arr=test_rul, horizon=horizon)
    test_cls_e, _, _ = evaluate_baseline_on_engines(final_det_e, test_norm, is_test=True, test_rul_arr=test_rul, horizon=horizon)

    return {
        "subset": f"FD00{subset_id}",
        "best_static_param": best_static_param,
        "best_static_f1": best_static_f1,
        "best_ewma_param": best_ewma_param,
        "best_ewma_f1": best_ewma_f1,
        "oof_fleet_static": oof_fleet_static,
        "oof_fleet_ewma": oof_fleet_ewma,
        "oof_fleet_static_h50": oof_fleet_static_h50,
        "oof_fleet_ewma_h50": oof_fleet_ewma_h50,
        "test_cls_static": test_cls_s,
        "test_cls_ewma": test_cls_e,
        "oof_static_engines": oof_static_engines,
        "oof_ewma_engines": oof_ewma_engines,
    }
