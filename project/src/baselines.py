"""
baselines.py - Phase B: Traditional threshold and statistical control baselines.

Baselines implemented:
  1. Static k-sigma threshold:
     - Fit per-sensor mean (mu) and std (sigma) on healthy early cycles (cycles 1..30)
       of training engines.
     - Out-of-bounds if |x - mu| > k * sigma.
     - Multi-sensor voting (fraction >= 0.3) + persistence filter (C = 3 consecutive cycles).
     - Parameter k tuned strictly on training engines.

  2. EWMA Statistical Control Limits (Shewhart / EWMA style):
     - Exponentially weighted moving average per sensor: z_t = lambda*x_t + (1-lambda)*z_{t-1}.
     - Control limits: mu +- L * sigma * sqrt(lambda / (2 - lambda)).
     - Multi-sensor voting (fraction >= 0.3) + persistence filter (C = 3 consecutive cycles).
     - Parameter L tuned strictly on training engines.

CRITICAL GUARANTEE:
  - Test engines are NEVER used for parameter tuning, threshold selection, or early stopping.
  - Tuning evaluates candidate parameters strictly on training engines (split by unit ID).
  - Test engines are only evaluated once with frozen parameters.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from data_loader import load_raw, select_features
from labels import add_anomaly_labels
from utils import get_logger, load_config, set_seed

logger = get_logger("baselines")


# ---------------------------------------------------------------------------
# 1. Baseline Detectors
# ---------------------------------------------------------------------------

class StaticThresholdDetector:
    """
    Static k-sigma threshold detector fit on early healthy cycles of train engines.
    """

    def __init__(self, k: float = 3.0, healthy_cycles: int = 30,
                 vote_fraction: float = 0.3, consecutive_cycles: int = 3):
        self.k = k
        self.healthy_cycles = healthy_cycles
        self.vote_fraction = vote_fraction
        self.consecutive_cycles = consecutive_cycles
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None
        self.feature_cols: list[str] = []

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "StaticThresholdDetector":
        self.feature_cols = feature_cols
        healthy_df = train_df[train_df["cycle"] <= self.healthy_cycles]
        self.means = healthy_df[feature_cols].mean()
        self.stds = healthy_df[feature_cols].std().replace(0, 1e-6)
        return self

    def predict_engine(self, grp: pd.DataFrame) -> np.ndarray:
        """Return binary array of alarm flags per cycle for a single engine."""
        grp = grp.sort_values("cycle").reset_index(drop=True)
        feats = grp[self.feature_cols]
        z_scores = np.abs((feats - self.means) / self.stds)
        # Fraction of sensors out of bounds
        out_of_bounds = (z_scores > self.k).mean(axis=1).values
        raw_alarms = (out_of_bounds >= self.vote_fraction).astype(int)

        # Persistence filter (C consecutive cycles)
        alarms = np.zeros_like(raw_alarms)
        streak = 0
        alarm_latched = False
        for t in range(len(raw_alarms)):
            if raw_alarms[t] == 1:
                streak += 1
                if streak >= self.consecutive_cycles:
                    alarm_latched = True
            else:
                streak = 0
            if alarm_latched:
                alarms[t] = 1
        return alarms


class EWMABaselineDetector:
    """
    Exponentially Weighted Moving Average (EWMA) statistical control limits detector.
    """

    def __init__(self, L: float = 3.0, ewma_lambda: float = 0.2,
                 healthy_cycles: int = 30, vote_fraction: float = 0.3,
                 consecutive_cycles: int = 3):
        self.L = L
        self.ewma_lambda = ewma_lambda
        self.healthy_cycles = healthy_cycles
        self.vote_fraction = vote_fraction
        self.consecutive_cycles = consecutive_cycles
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None
        self.feature_cols: list[str] = []

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "EWMABaselineDetector":
        self.feature_cols = feature_cols
        healthy_df = train_df[train_df["cycle"] <= self.healthy_cycles]
        self.means = healthy_df[feature_cols].mean()
        self.stds = healthy_df[feature_cols].std().replace(0, 1e-6)
        return self

    def predict_engine(self, grp: pd.DataFrame) -> np.ndarray:
        """Return binary array of alarm flags per cycle for a single engine."""
        grp = grp.sort_values("cycle").reset_index(drop=True)
        feats = grp[self.feature_cols]

        # Compute causal EWMA per sensor
        ewma_feats = feats.ewm(alpha=self.ewma_lambda, adjust=False).mean()

        # Control limit half-width: L * sigma * sqrt(lambda / (2 - lambda))
        sigma_factor = np.sqrt(self.ewma_lambda / (2.0 - self.ewma_lambda))
        ucl = self.means + self.L * self.stds * sigma_factor
        lcl = self.means - self.L * self.stds * sigma_factor

        out_of_control = ((ewma_feats > ucl) | (ewma_feats < lcl)).mean(axis=1).values
        raw_alarms = (out_of_control >= self.vote_fraction).astype(int)

        # Persistence filter (C consecutive cycles)
        alarms = np.zeros_like(raw_alarms)
        streak = 0
        alarm_latched = False
        for t in range(len(raw_alarms)):
            if raw_alarms[t] == 1:
                streak += 1
                if streak >= self.consecutive_cycles:
                    alarm_latched = True
            else:
                streak = 0
            if alarm_latched:
                alarms[t] = 1
        return alarms


# ---------------------------------------------------------------------------
# 2. Evaluation Helper (Precision, Recall, F1, False Alarm Rate, Lead Time)
# ---------------------------------------------------------------------------

def evaluate_detector(
    detector: Any,
    df: pd.DataFrame,
    is_test: bool = False,
    test_rul_arr: np.ndarray | None = None,
    n_threshold: int = 30,
) -> dict[str, float]:
    """
    Evaluate detector on a dataset (train or test) engine-by-engine.
    """
    df_labeled = add_anomaly_labels(df, n_threshold=n_threshold,
                                    is_test=is_test, test_rul_arr=test_rul_arr)

    all_preds = []
    all_trues = []
    lead_times = []

    for unit_id, grp in df_labeled.groupby("unit"):
        preds = detector.predict_engine(grp)
        trues = grp["anomaly_label"].values
        cycles = grp["cycle"].values
        max_cycle = cycles.max()

        all_preds.extend(preds)
        all_trues.extend(trues)

        # Lead time: cycle of first confirmed alarm vs end of observation
        alarm_idx = np.where(preds == 1)[0]
        if len(alarm_idx) > 0:
            first_alarm_cycle = cycles[alarm_idx[0]]
            if not is_test:
                # In train, failure is at max_cycle
                lead_time = max_cycle - first_alarm_cycle
            else:
                # In test, true failure cycle is max_cycle + test_rul
                uid_sorted = sorted(df["unit"].unique())
                idx = uid_sorted.index(unit_id)
                actual_failure = max_cycle + test_rul_arr[idx]
                lead_time = actual_failure - first_alarm_cycle
            lead_times.append(lead_time)

    all_preds_arr = np.array(all_preds)
    all_trues_arr = np.array(all_trues)

    p, r, f1, _ = precision_recall_fscore_support(
        all_trues_arr, all_preds_arr, average="binary", zero_division=0
    )

    # False Alarm Rate (FAR): alarms during healthy periods / total healthy cycles
    healthy_mask = (all_trues_arr == 0)
    far = (all_preds_arr[healthy_mask] == 1).mean() if healthy_mask.sum() > 0 else 0.0

    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    detected_pct = (len(lead_times) / df["unit"].nunique()) * 100.0

    return {
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
        "FAR": float(far),
        "Mean_Lead_Time": mean_lead,
        "Detection_Rate_Pct": float(detected_pct),
    }


# ---------------------------------------------------------------------------
# 3. Parameter Tuning Strictly on Train Engines
# ---------------------------------------------------------------------------

def tune_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_rul_arr: np.ndarray,
    feature_cols: list[str],
    cfg: dict[str, Any],
    val_unit_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Tune baseline parameters strictly on training engines using a unit-level split.

    Split train units into:
      - 80% tuning-train (to fit healthy statistics mu, sigma)
      - 20% tuning-val   (to evaluate candidate parameters and select best k, L)
    Test engines are strictly held out and only evaluated once with the best parameters.
    """
    rng = np.random.default_rng(seed)
    units = sorted(train_df["unit"].unique())
    n_val_units = max(1, int(len(units) * val_unit_fraction))
    val_units = set(rng.choice(units, size=n_val_units, replace=False))
    train_units = [u for u in units if u not in val_units]

    tune_train_df = train_df[train_df["unit"].isin(train_units)].copy()
    tune_val_df   = train_df[train_df["unit"].isin(val_units)].copy()

    k_grid = cfg["baselines"]["static_k_grid"]
    l_grid = cfg["baselines"]["ewma_l_grid"]
    ewma_lambda = cfg["baselines"]["ewma_lambda"]
    consecutive = cfg["baselines"]["consecutive_cycles"]
    vote_frac   = cfg["baselines"]["sensor_vote_fraction"]
    n_thresh    = cfg["anomaly"]["threshold_n"]

    # --- Tune Static Threshold ---
    static_tuning_records = []
    best_k = k_grid[0]
    best_static_f1 = -1.0

    for k in k_grid:
        detector = StaticThresholdDetector(
            k=k, healthy_cycles=30, vote_fraction=vote_frac,
            consecutive_cycles=consecutive
        )
        detector.fit(tune_train_df, feature_cols)
        metrics = evaluate_detector(detector, tune_val_df, is_test=False, n_threshold=n_thresh)
        static_tuning_records.append({
            "Baseline": "Static_Threshold",
            "Param_k": k,
            "TuneVal_F1": metrics["F1"],
            "TuneVal_Precision": metrics["Precision"],
            "TuneVal_Recall": metrics["Recall"],
            "TuneVal_FAR": metrics["FAR"],
            "TuneVal_LeadTime": metrics["Mean_Lead_Time"],
        })
        if metrics["F1"] > best_static_f1:
            best_static_f1 = metrics["F1"]
            best_k = k

    # --- Tune EWMA Control Limits ---
    ewma_tuning_records = []
    best_L = l_grid[0]
    best_ewma_f1 = -1.0

    for L in l_grid:
        detector = EWMABaselineDetector(
            L=L, ewma_lambda=ewma_lambda, healthy_cycles=30,
            vote_fraction=vote_frac, consecutive_cycles=consecutive
        )
        detector.fit(tune_train_df, feature_cols)
        metrics = evaluate_detector(detector, tune_val_df, is_test=False, n_threshold=n_thresh)
        ewma_tuning_records.append({
            "Baseline": "EWMA_Control_Limits",
            "Param_L": L,
            "TuneVal_F1": metrics["F1"],
            "TuneVal_Precision": metrics["Precision"],
            "TuneVal_Recall": metrics["Recall"],
            "TuneVal_FAR": metrics["FAR"],
            "TuneVal_LeadTime": metrics["Mean_Lead_Time"],
        })
        if metrics["F1"] > best_ewma_f1:
            best_ewma_f1 = metrics["F1"]
            best_L = L

    tuning_df = pd.DataFrame(static_tuning_records + ewma_tuning_records)

    # --- Final Evaluation on TEST ENGINES using frozen parameters ---
    # Fit final models on ALL training engines
    final_static = StaticThresholdDetector(
        k=best_k, healthy_cycles=30, vote_fraction=vote_frac,
        consecutive_cycles=consecutive
    ).fit(train_df, feature_cols)

    final_ewma = EWMABaselineDetector(
        L=best_L, ewma_lambda=ewma_lambda, healthy_cycles=30,
        vote_fraction=vote_frac, consecutive_cycles=consecutive
    ).fit(train_df, feature_cols)

    test_metrics_static = evaluate_detector(
        final_static, test_df, is_test=True, test_rul_arr=test_rul_arr, n_threshold=n_thresh
    )
    test_metrics_ewma = evaluate_detector(
        final_ewma, test_df, is_test=True, test_rul_arr=test_rul_arr, n_threshold=n_thresh
    )

    test_eval_records = [
        {
            "Baseline": "Static Threshold",
            "Best_Param": f"k = {best_k:.1f}",
            "Test_Precision": test_metrics_static["Precision"],
            "Test_Recall": test_metrics_static["Recall"],
            "Test_F1": test_metrics_static["F1"],
            "Test_FAR": test_metrics_static["FAR"],
            "Test_Lead_Time": test_metrics_static["Mean_Lead_Time"],
            "Test_Detection_Rate": f"{test_metrics_static['Detection_Rate_Pct']:.1f}%",
        },
        {
            "Baseline": "EWMA Control Limits",
            "Best_Param": f"L = {best_L:.1f}",
            "Test_Precision": test_metrics_ewma["Precision"],
            "Test_Recall": test_metrics_ewma["Recall"],
            "Test_F1": test_metrics_ewma["F1"],
            "Test_FAR": test_metrics_ewma["FAR"],
            "Test_Lead_Time": test_metrics_ewma["Mean_Lead_Time"],
            "Test_Detection_Rate": f"{test_metrics_ewma['Detection_Rate_Pct']:.1f}%",
        },
    ]
    test_eval_df = pd.DataFrame(test_eval_records)

    results = {
        "best_k": best_k,
        "best_L": best_L,
        "test_static": test_metrics_static,
        "test_ewma": test_metrics_ewma,
    }
    return results, tuning_df, test_eval_df


# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------

def main() -> None:
    set_seed(42)
    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root / "configs" / "config.yaml")

    raw_dir = Path(cfg["data"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("  PHASE B: TRADITIONAL BASELINE TUNING & EVALUATION (FD001)")
    print("  (Tuning strictly on Train Engines; Test Engines evaluated only with frozen params)")
    print("=" * 90)

    train_raw, test_raw, test_rul = load_raw(raw_dir, 1)
    feature_cols = select_features(train_raw, 0.001)

    results, tuning_df, test_eval_df = tune_baselines(
        train_df=train_raw,
        test_df=test_raw,
        test_rul_arr=test_rul,
        feature_cols=feature_cols,
        cfg=cfg,
        val_unit_fraction=0.2,
        seed=42,
    )

    # Print Tuning Grid Search Results on Training Units
    print("\n--- BASELINE GRID SEARCH RESULTS (ON 20% HELD-OUT TRAINING UNITS) ---")
    print(f"  {'Baseline':<22} {'Param':<10} {'TuneVal F1':>12} {'TuneVal Prec':>13} "
          f"{'TuneVal Rec':>12} {'TuneVal FAR':>12} {'Mean Lead Time':>15}")
    print("-" * 92)
    for _, r in tuning_df.iterrows():
        param_str = f"k={r['Param_k']:.1f}" if pd.notna(r.get("Param_k")) else f"L={r['Param_L']:.1f}"
        print(f"  {r['Baseline']:<22} {param_str:<10} {r['TuneVal_F1']:>12.4f} "
              f"{r['TuneVal_Precision']:>13.4f} {r['TuneVal_Recall']:>12.4f} "
              f"{r['TuneVal_FAR']:>12.4f} {r['TuneVal_LeadTime']:>15.1f}")
    print("-" * 92)

    print(f"\nOptimal Selected Parameters (from Train Engines only):")
    print(f"  -> Static Threshold Best Parameter : k = {results['best_k']:.1f}")
    print(f"  -> EWMA Control Limit Best Parameter: L = {results['best_L']:.1f}")

    # Print Test Evaluation Table
    print("\n--- FINAL TEST EVALUATION (OFFICIAL TEST ENGINES WITH FROZEN PARAMETERS) ---")
    print(f"  {'Baseline':<22} {'Param':<10} {'Test Prec':>10} {'Test Rec':>10} "
          f"{'Test F1':>10} {'Test FAR':>10} {'Lead Time':>11} {'Detection %':>12}")
    print("-" * 90)
    for _, r in test_eval_df.iterrows():
        print(f"  {r['Baseline']:<22} {r['Best_Param']:<10} {r['Test_Precision']:>10.4f} "
              f"{r['Test_Recall']:>10.4f} {r['Test_F1']:>10.4f} {r['Test_FAR']:>10.4f} "
              f"{r['Test_Lead_Time']:>11.1f} {r['Test_Detection_Rate']:>12}")
    print("=" * 90 + "\n")

    # Save to CSV
    tuning_csv = results_dir / "baseline_tuning_results.csv"
    tuning_df.to_csv(tuning_csv, index=False)
    test_eval_csv = results_dir / "baseline_test_evaluation.csv"
    test_eval_df.to_csv(test_eval_csv, index=False)
    print(f"[Phase B] Baseline tuning results saved -> {tuning_csv}")
    print(f"[Phase B] Baseline test evaluation saved -> {test_eval_csv}")


if __name__ == "__main__":
    main()
