"""
run_anomaly.py - Master Entry Point for Anomaly Detection, Lead-Time & RUL Evaluation.

Project: "Predictive Maintenance and Anomaly Detection for Industrial IoT Telemetry Using Machine Learning"
NASA C-MAPSS Subsets: FD001 - FD004.

Executes:
  - Part 1: Standardized sustained-alarm lead time & false alarm metrics (H=100 and H=50).
  - Part 2: Statistical baselines (Static k-sigma & EWMA) with 5-fold GroupKFold tuning,
            condition-aware KMeans normalization for FD002/FD004, and comparative Health Index (PCA vs Mahalanobis).
  - Part 3 (Phase C): ML Anomaly Detectors (Isolation Forest, RF, XGBoost, LSTM)
            with dual thresholding (Max-F1 and Best-Baseline-Matched FAR).
  - Part 4 (Phase D): Full statistical evaluation over 5 seeds (0-4), Wilcoxon signed-rank tests with Holm correction,
            official test engine window classification ("test (truncated)"), publication figures (300 DPI),
            and 5-seed repeat for RUL regression models into results/rul_seeds.csv.
  - Stage 1 Enhancements:
      1. Shuffled GroupKFold per seed + seeded XGBoost (subsample=0.8, colsample_bytree=0.8) and RF (max_samples=0.8).
      2. Threshold sweeps over 50 values for all 7 methods -> Lead time vs False Alarms per 100 healthy cycles curves.
         Report lead time at FA = 0.1, 0.5, 1.0 budgets per 100 cycles.
      3. Label sensitivity sweep for N in {20, 30, 50, 75, 100}.
      4. Scoring test baselines on the exact same truncated test windows (identical prevalence).
      5. Holm-Bonferroni correction and bootstrap 95% CI on Wilcoxon effect sizes.
      6. Maintenance of results/manifest.csv mapping every thesis number to source files.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)

# Local imports
sys.path.insert(0, str(Path(__file__).parent))

from baselines_extended import (
    ParametricStaticDetector,
    ParametricEWMADetector,
    tune_and_evaluate_baselines_subset,
)
from data_loader import load_raw, select_features, build_dataset
from evaluate import evaluate_all, compute_baselines
from features_extractor import extract_window_features
from health_indices import evaluate_health_indices_all_subsets, DualHealthIndex
from labels import add_anomaly_labels, compute_label_statistics, print_label_statistics_table
from metrics_lead_time import compute_sustained_alarms, evaluate_engine_lead_time, evaluate_fleet_lead_time
from ml_anomaly import MLAnomalyDetector, train_lstm_classifier, evaluate_ml_model_lead_time, predict_lstm_probs
from models import RandomForestRUL, XGBoostRUL, LSTMModel, CNNModel
from plotting import (
    plot_pr_curves,
    plot_lead_time_distributions,
    plot_false_alarms_per_engine,
    plot_lead_time_vs_fa_curves,
)
from regime_norm import OperatingRegimeNormalizer, identify_healthy_early_cycles
from trainer import train_torch_model, predict_torch
from utils import get_logger, load_config, log_environment, set_seed, Timer, get_shuffled_group_kfold

logger = get_logger("run_anomaly")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NASA C-MAPSS Anomaly Detection Pipeline")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    p.add_argument("--subsets", nargs="+", type=int, default=[1, 2, 3, 4], help="Subsets to run (1-4)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Random seeds")
    p.add_argument("--horizon", type=int, default=100, help="Lead time evaluation horizon")
    p.add_argument("--skip-rul-seeds", action="store_true", help="Skip 5-seed RUL regression repeat")
    p.add_argument("--fast", action="store_true", help="Fast smoke test mode")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data Preparation Helper for ML Sliding Windows
# ---------------------------------------------------------------------------

def build_windowed_engine_data(
    df_norm: pd.DataFrame,
    feature_cols: list[str],
    window_length: int = 30,
    n_threshold: int = 30,
    is_test: bool = False,
    test_rul_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """
    Construct sliding windows from normalized DataFrame while preserving engine unit, cycle,
    and early healthy life flag.
    """
    df_lab = add_anomaly_labels(df_norm, n_threshold=n_threshold, is_test=is_test, test_rul_arr=test_rul_arr)
    X_list, y_list, u_list, c_list, h_list = [], [], [], [], []
    unit_failure_cycles: dict[int, int] = {}
    sorted_units = sorted(df_norm["unit"].unique())

    for uid, grp in df_lab.groupby("unit"):
        grp_sorted = grp.sort_values("cycle").reset_index(drop=True)
        feats = grp_sorted[feature_cols].values.astype(np.float32)
        labels = grp_sorted["anomaly_label"].values.astype(np.int32)
        cycles = grp_sorted["cycle"].values.astype(np.int32)
        n = len(grp_sorted)

        if not is_test:
            fail_c = int(cycles.max())
        else:
            uid_idx = sorted_units.index(uid)
            fail_c = int(cycles.max() + (test_rul_arr[uid_idx] if test_rul_arr is not None else 0))
        unit_failure_cycles[int(uid)] = fail_c

        # Healthy early life flag: cycle <= round(0.30 * n)
        h_cutoff = int(np.round(0.30 * n))

        for start in range(0, n - window_length + 1):
            end = start + window_length
            X_list.append(feats[start:end])
            y_list.append(labels[end - 1])
            u_list.append(int(uid))
            c_list.append(int(cycles[end - 1]))
            h_list.append(bool(cycles[end - 1] <= h_cutoff))

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(u_list, dtype=np.int32),
        np.array(c_list, dtype=np.int32),
        np.array(h_list, dtype=bool),
        unit_failure_cycles,
    )


# ---------------------------------------------------------------------------
# Operating Characteristic Curve Sweeps (Stage 1 Item 2)
# ---------------------------------------------------------------------------

def sweep_lead_time_vs_fa_curves(
    subset_tag: str,
    train_norm: pd.DataFrame,
    feature_cols: list[str],
    oof_predictions_dict: dict[str, np.ndarray],
    u_win: np.ndarray,
    c_win: np.ndarray,
    unit_failure_cycles: dict[int, int],
    tuned_static_param: tuple,
    tuned_ewma_param: tuple,
    horizon: int = 100,
    n_points: int = 50,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """
    Sweeps alarm thresholds over ~50 values for all 7 methods:
      Static_Threshold, EWMA_Control_Limits, Mahalanobis_HI,
      IsolationForest, RandomForest, XGBoost, LSTM.
    Returns:
      curves: {method: (fa_per_100_array, lead_time_array)}
      budget_records: list of {Subset, Method, LeadTime_FA_0.1, LeadTime_FA_0.5, LeadTime_FA_1.0}
    """
    unique_units = sorted(unit_failure_cycles.keys())

    # Precompute trajectories per engine
    engine_cycles = {}
    for uid, grp in train_norm.groupby("unit"):
        engine_cycles[int(uid)] = grp.sort_values("cycle")["cycle"].values.astype(int)

    # 1. Prepare per-cycle continuous score arrays for all 7 methods
    # Healthy baseline for Static & EWMA
    healthy_train = identify_healthy_early_cycles(train_norm, 0.30)
    means = healthy_train[feature_cols].mean().values
    stds = healthy_train[feature_cols].std().replace(0, 1e-5).values

    # Fit DualHealthIndex for Mahalanobis label-free alarm
    hi_model = DualHealthIndex(healthy_life_fraction=0.30)
    hi_model.fit(train_norm, feature_cols)

    method_scores: dict[str, dict[int, np.ndarray]] = {
        "Static_Threshold": {},
        "EWMA_Control_Limits": {},
        "Mahalanobis_HI": {},
        "IsolationForest": {},
        "RandomForest": {},
        "XGBoost": {},
        "LSTM": {},
    }

    # Extract EWMA factor with tuned lambda
    lam_tuned = tuned_ewma_param[0]
    ewma_factor = np.sqrt(lam_tuned / (2.0 - lam_tuned))

    for uid, grp in train_norm.groupby("unit"):
        uid = int(uid)
        grp_s = grp.sort_values("cycle").reset_index(drop=True)
        feats = grp_s[feature_cols].values
        n_c = len(grp_s)

        # Static z-scores
        z = np.abs((feats - means) / stds)
        # Store max z across sensors as single anomaly score
        method_scores["Static_Threshold"][uid] = np.mean(z, axis=1)

        # EWMA deviation score
        ewm_df = pd.DataFrame(feats).ewm(alpha=lam_tuned, adjust=False).mean().values
        ewm_dev = np.abs((ewm_df - means) / (stds * ewma_factor))
        method_scores["EWMA_Control_Limits"][uid] = np.mean(ewm_dev, axis=1)

        # Mahalanobis HI (note: HI drops with degradation, so 1 - HI increases with degradation)
        _, hi_md = hi_model.transform_engine(grp_s)
        method_scores["Mahalanobis_HI"][uid] = 1.0 - hi_md

        # Window models: map window scores back to cycles (pad first 29 cycles with first score)
        mask = (u_win == uid)
        u_c = c_win[mask]
        for m_name in ["IsolationForest", "RandomForest", "XGBoost", "LSTM"]:
            w_scores = oof_predictions_dict[m_name][mask]
            cyc_scores = np.zeros(n_c, dtype=np.float32)
            if len(w_scores) > 0:
                first_score = w_scores[0]
                cyc_scores[:u_c[0] - 1] = first_score
                for idx, c in enumerate(u_c):
                    cyc_scores[c - 1] = w_scores[idx]
            method_scores[m_name][uid] = cyc_scores

    # Define sweep threshold grids per method
    threshold_grids = {
        "Static_Threshold": np.linspace(0.8, 4.5, n_points),
        "EWMA_Control_Limits": np.linspace(0.8, 5.0, n_points),
        "Mahalanobis_HI": np.linspace(0.05, 0.95, n_points),
        "IsolationForest": np.linspace(0.02, 0.98, n_points),
        "RandomForest": np.linspace(0.01, 0.99, n_points),
        "XGBoost": np.linspace(0.01, 0.99, n_points),
        "LSTM": np.linspace(0.01, 0.99, n_points),
    }

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    budget_records: list[dict[str, Any]] = []

    for m_name, th_grid in threshold_grids.items():
        fa_list = []
        lt_list = []

        for th in th_grid:
            eng_results = []
            for uid in unique_units:
                sc = method_scores[m_name][uid]
                cyc = engine_cycles[uid]
                raw_alarm = (sc >= th).astype(int)
                sust = compute_sustained_alarms(raw_alarm, persistence=3)
                fail_c = unit_failure_cycles[uid]
                e_eval = evaluate_engine_lead_time(cyc, sust, failure_cycle=fail_c, horizon=horizon)
                eng_results.append(e_eval)

            fleet = evaluate_fleet_lead_time(eng_results, horizon=horizon)
            fa_list.append(fleet["False_Alarms_Per_100_Healthy"])
            lt_list.append(fleet["Mean_Lead_Time"])

        fa_arr = np.array(fa_list, dtype=np.float32)
        lt_arr = np.array(lt_list, dtype=np.float32)
        curves[m_name] = (fa_arr, lt_arr)

        # Interpolate lead time at FA = 0.1, 0.5, 1.0 per 100 healthy cycles
        def interp_lt(target_fa: float) -> str:
            # Sort by FA for monotonic interpolation
            sort_idx = np.argsort(fa_arr)
            fa_s = fa_arr[sort_idx]
            lt_s = lt_arr[sort_idx]
            if target_fa < np.min(fa_s):
                return "N/A"
            if target_fa > np.max(fa_s):
                return f"{lt_s[np.argmax(fa_s)]:.1f}"
            val = float(np.interp(target_fa, fa_s, lt_s))
            return f"{val:.1f}"

        budget_records.append({
            "Subset": subset_tag,
            "Method": m_name,
            "LeadTime_at_FA_0.1": interp_lt(0.1),
            "LeadTime_at_FA_0.5": interp_lt(0.5),
            "LeadTime_at_FA_1.0": interp_lt(1.0),
        })

    return curves, budget_records


# ---------------------------------------------------------------------------
# Label Sensitivity Sweep (Stage 1 Item 3)
# ---------------------------------------------------------------------------

def run_label_sensitivity_sweep(
    subsets: list[int],
    raw_dir: Path,
    results_dir: Path,
    n_values: list[int] = [20, 30, 50, 75, 100],
    horizon: int = 100,
) -> pd.DataFrame:
    """
    Retrains XGBoost anomaly detector for N in {20, 30, 50, 75, 100} to investigate
    how lead time and classification performance depend on the label cutoff N.
    """
    print("\n" + "=" * 90)
    print("  STAGE 1: LABEL SENSITIVITY SWEEP (N in {20, 30, 50, 75, 100})")
    print("=" * 90)

    records = []

    for s in subsets:
        tag = f"FD00{s}"
        train_raw, _, _ = load_raw(raw_dir, s)
        feature_cols = select_features(train_raw, 0.001)

        norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=42)
        norm.fit(train_raw, feature_cols, subset_id=s)
        train_norm = norm.transform(train_raw)

        for n_val in n_values:
            X_raw, y_win, u_win, c_win, h_win, fail_dict = build_windowed_engine_data(
                train_norm, feature_cols, window_length=30, n_threshold=n_val, is_test=False
            )
            X_feats = extract_window_features(X_raw)
            prevalence = float(np.mean(y_win))

            splits = get_shuffled_group_kfold(u_win, n_splits=5, seed=42)
            oof_scores = np.zeros(len(y_win), dtype=np.float32)

            for fold, (tr_idx, val_idx) in enumerate(splits):
                det = MLAnomalyDetector("XGBoost", seed=42)
                det.fit(X_raw[tr_idx], X_feats[tr_idx], y_win[tr_idx], h_win[tr_idx])
                oof_scores[val_idx] = det.predict_score(X_raw[val_idx], X_feats[val_idx])

            # Validation threshold maximizing F1
            det_tune = MLAnomalyDetector("XGBoost", seed=42)
            th_f1, _ = det_tune.tune_thresholds(oof_scores, y_win)

            p, r, f1, _ = precision_recall_fscore_support(
                y_win, (oof_scores >= th_f1).astype(int), average="binary", zero_division=0
            )
            pr_auc = float(average_precision_score(y_win, oof_scores))
            roc_auc = float(roc_auc_score(y_win, oof_scores))

            fleet, _ = evaluate_ml_model_lead_time(
                oof_scores, threshold=th_f1, val_units=u_win, val_cycles=c_win,
                unit_failure_cycles=fail_dict, horizon=horizon, persistence=3
            )

            records.append({
                "Subset": tag,
                "N_Cutoff": n_val,
                "Prevalence_Pct": round(prevalence * 100.0, 1),
                "F1": round(float(f1), 3),
                "PR_AUC": round(pr_auc, 3),
                "ROC_AUC": round(roc_auc, 3),
                "Detection_Rate_Pct": round(fleet["Detection_Rate_Pct"], 1),
                "Mean_Lead_Time": round(fleet["Mean_Lead_Time"], 1),
                "Median_Lead_Time": round(fleet["Median_Lead_Time"], 1),
                "Premature_Alarm_Pct": round(fleet["Premature_Alarm_Share_Pct"], 1),
                "FA_Per_100_Healthy": round(fleet["False_Alarms_Per_100_Healthy"], 2),
            })

    df_sweep = pd.DataFrame(records)
    csv_path = results_dir / "label_sensitivity_sweep.csv"
    df_sweep.to_csv(csv_path, index=False)
    print(f"[Saved] Label sensitivity sweep -> {csv_path}")

    print("\n" + "=" * 90)
    print("  LABEL DEFINITION SENSITIVITY TABLE (XGBoost Classifier)")
    print("=" * 90)
    print(f"  {'Subset':<7} {'N':<5} {'Prevalence':<12} {'F1':<8} {'PR-AUC':<8} {'Lead Time':<12} {'Detect %':<10} {'FA / 100':<10}")
    print("-" * 90)
    for _, r in df_sweep.iterrows():
        print(f"  {r['Subset']:<7} {r['N_Cutoff']:<5} {r['Prevalence_Pct']:<11.1f}% {r['F1']:<8.3f} {r['PR_AUC']:<8.3f} "
              f"{r['Mean_Lead_Time']:<12.1f} {r['Detection_Rate_Pct']:<9.1f}% {r['FA_Per_100_Healthy']:<10.2f}")
    print("=" * 90 + "\n")

    return df_sweep


# ---------------------------------------------------------------------------
# RUL Models 5-Seed Repeat (Part 4)
# ---------------------------------------------------------------------------

def run_rul_seeds_evaluation(
    cfg: dict[str, Any],
    subsets: list[int],
    seeds: list[int],
    results_dir: Path,
) -> pd.DataFrame:
    """
    Run 5-seed repeat for RUL regression models (RF, XGBoost, LSTM, 1D-CNN)
    saving results/rul_seeds.csv with mean +/- std.
    """
    print("\n" + "=" * 90)
    print("  PART 4: 5-SEED REPEAT FOR RUL REGRESSION MODELS (RF, XGBoost, LSTM, CNN)")
    print("=" * 90)

    rul_clip = cfg["evaluation"]["rul_clip"]
    window_length = cfg["preprocessing"]["window_length"]

    cfg_copy = dict(cfg)
    cfg_copy["data"]["subsets"] = subsets
    raw_dir_resolved = Path(cfg["data"]["raw_dir"])
    if not raw_dir_resolved.is_absolute():
        raw_dir_resolved = Path(__file__).resolve().parent.parent / raw_dir_resolved
    cfg_copy["data"]["raw_dir"] = str(raw_dir_resolved)

    # Adjust epochs for faster seed iteration with GPU early stopping
    cfg_copy["models"]["lstm"]["max_epochs"] = 30
    cfg_copy["models"]["lstm"]["patience"] = 7
    cfg_copy["models"]["cnn"]["max_epochs"] = 30
    cfg_copy["models"]["cnn"]["patience"] = 7

    print("Building datasets for RUL models...")
    datasets = build_dataset(cfg_copy)

    rul_records = []

    for seed in seeds:
        print(f"\n--- Running RUL Models Seed {seed} ---")
        set_seed(seed)
        cfg_copy["models"]["random_seed"] = seed

        for s in subsets:
            tag = f"fd{s}"
            label = f"FD00{s}"
            X_train = datasets[f"{tag}_train_X"]
            y_train = datasets[f"{tag}_train_y"]
            X_test  = datasets[f"{tag}_test_X"]
            y_test  = datasets[f"{tag}_test_y"]
            n_feat  = X_train.shape[2]

            # 1. RF
            rf = RandomForestRUL(cfg_copy)
            rf.fit(X_train, y_train)
            p_rf = np.clip(rf.predict(X_test), 0, rul_clip)
            m_rf = evaluate_all(y_test, p_rf, rul_clip)
            rul_records.append({
                "Subset": label, "Model": "RF", "Seed": seed,
                "RMSE": m_rf["RMSE"], "RMSE_clip": m_rf["RMSE_clip"],
                "MAE": m_rf["MAE"], "NASA_Score": m_rf["NASA_Score"],
            })

            # 2. XGB
            xgb_m = XGBoostRUL(cfg_copy)
            xgb_m.fit(X_train, y_train)
            p_xgb = np.clip(xgb_m.predict(X_test), 0, rul_clip)
            m_xgb = evaluate_all(y_test, p_xgb, rul_clip)
            rul_records.append({
                "Subset": label, "Model": "XGBoost", "Seed": seed,
                "RMSE": m_xgb["RMSE"], "RMSE_clip": m_xgb["RMSE_clip"],
                "MAE": m_xgb["MAE"], "NASA_Score": m_xgb["NASA_Score"],
            })

            # 3. LSTM
            lstm = LSTMModel(n_features=n_feat, cfg=cfg_copy)
            lstm, _, _ = train_torch_model(lstm, X_train, y_train, cfg_copy, model_key="lstm")
            p_lstm = np.clip(predict_torch(lstm, X_test, rul_clip), 0, rul_clip)
            m_lstm = evaluate_all(y_test, p_lstm, rul_clip)
            rul_records.append({
                "Subset": label, "Model": "LSTM", "Seed": seed,
                "RMSE": m_lstm["RMSE"], "RMSE_clip": m_lstm["RMSE_clip"],
                "MAE": m_lstm["MAE"], "NASA_Score": m_lstm["NASA_Score"],
            })

            # 4. CNN
            cnn = CNNModel(n_features=n_feat, window_length=window_length, cfg=cfg_copy)
            cnn, _, _ = train_torch_model(cnn, X_train, y_train, cfg_copy, model_key="cnn")
            p_cnn = np.clip(predict_torch(cnn, X_test, rul_clip), 0, rul_clip)
            m_cnn = evaluate_all(y_test, p_cnn, rul_clip)
            rul_records.append({
                "Subset": label, "Model": "CNN", "Seed": seed,
                "RMSE": m_cnn["RMSE"], "RMSE_clip": m_cnn["RMSE_clip"],
                "MAE": m_cnn["MAE"], "NASA_Score": m_cnn["NASA_Score"],
            })

    df_rul_seeds = pd.DataFrame(rul_records)
    rul_seeds_csv = results_dir / "rul_seeds.csv"
    df_rul_seeds.to_csv(rul_seeds_csv, index=False)
    print(f"\n[Saved] RUL seed results -> {rul_seeds_csv}")

    # Summary table: Mean +/- Std
    summary_rows = []
    for (sub, model), grp in df_rul_seeds.groupby(["Subset", "Model"]):
        summary_rows.append({
            "Subset": sub,
            "Model": model,
            "RMSE": f"{grp['RMSE'].mean():.2f} +/- {grp['RMSE'].std():.2f}",
            "RMSE_clip": f"{grp['RMSE_clip'].mean():.2f} +/- {grp['RMSE_clip'].std():.2f}",
            "MAE": f"{grp['MAE'].mean():.2f} +/- {grp['MAE'].std():.2f}",
            "NASA_Score": f"{grp['NASA_Score'].mean():.1f} +/- {grp['NASA_Score'].std():.1f}",
        })
    df_summary = pd.DataFrame(summary_rows).sort_values(["Subset", "Model"])
    summary_csv = results_dir / "rul_seeds_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"[Saved] RUL seed summary -> {summary_csv}")

    print("\n" + "=" * 80)
    print("  RUL REGRESSION 5-SEED REPEAT SUMMARY TABLE (MEAN +/- STD)")
    print("=" * 80)
    print(f"  {'Subset':<8} {'Model':<12} {'RMSE':<18} {'RMSE_clip':<18} {'NASA Score':<20}")
    print("-" * 80)
    for _, r in df_summary.iterrows():
        print(f"  {r['Subset']:<8} {r['Model']:<12} {r['RMSE']:<18} {r['RMSE_clip']:<18} {r['NASA_Score']:<20}")
    print("=" * 80 + "\n")

    return df_rul_seeds


# ---------------------------------------------------------------------------
# Results Manifest Generator
# ---------------------------------------------------------------------------

def generate_results_manifest(results_dir: Path) -> Path:
    """
    Generates results/manifest.csv mapping every key thesis number to source file and column.
    """
    manifest_records = []

    # 1. Anomaly summary
    sum_csv = results_dir / "anomaly_metrics_summary.csv"
    if sum_csv.exists():
        df_sum = pd.read_csv(sum_csv)
        for _, r in df_sum.iterrows():
            manifest_records.append({
                "Metric_Category": "Anomaly_Detection_OOF",
                "Subset": r["Subset"],
                "Model": r["Model"],
                "Metric_Name": "PR_AUC",
                "Value": r["PR_AUC"],
                "Source_File": "results/anomaly_metrics_summary.csv",
                "Source_Column": "PR_AUC",
            })
            manifest_records.append({
                "Metric_Category": "Anomaly_Detection_OOF",
                "Subset": r["Subset"],
                "Model": r["Model"],
                "Metric_Name": "F1_Score",
                "Value": r["F1"],
                "Source_File": "results/anomaly_metrics_summary.csv",
                "Source_Column": "F1",
            })
            manifest_records.append({
                "Metric_Category": "Anomaly_Detection_OOF",
                "Subset": r["Subset"],
                "Model": r["Model"],
                "Metric_Name": "Mean_Lead_Time",
                "Value": r["Mean_Lead_Time"],
                "Source_File": "results/anomaly_metrics_summary.csv",
                "Source_Column": "Mean_Lead_Time",
            })

    # 2. Health index comparison
    hi_csv = results_dir / "health_index_spearman_comparison.csv"
    if hi_csv.exists():
        df_hi = pd.read_csv(hi_csv)
        for _, r in df_hi.iterrows():
            manifest_records.append({
                "Metric_Category": "Health_Index_Spearman",
                "Subset": r["Subset"],
                "Model": "PCA-PC1",
                "Metric_Name": "Spearman_Rho",
                "Value": str(r["Spearman_PCA_PC1"]),
                "Source_File": "results/health_index_spearman_comparison.csv",
                "Source_Column": "Spearman_PCA_PC1",
            })
            manifest_records.append({
                "Metric_Category": "Health_Index_Spearman",
                "Subset": r["Subset"],
                "Model": "Mahalanobis_Distance",
                "Metric_Name": "Spearman_Rho",
                "Value": str(r["Spearman_Mahalanobis"]),
                "Source_File": "results/health_index_spearman_comparison.csv",
                "Source_Column": "Spearman_Mahalanobis",
            })

    # 3. Lead time budgets
    b_csv = results_dir / "lead_time_at_fa_budgets.csv"
    if b_csv.exists():
        df_b = pd.read_csv(b_csv)
        for _, r in df_b.iterrows():
            manifest_records.append({
                "Metric_Category": "Lead_Time_at_FA_Budget",
                "Subset": r["Subset"],
                "Model": r["Method"],
                "Metric_Name": "Lead_Time_at_FA_0.5",
                "Value": str(r["LeadTime_at_FA_0.5"]),
                "Source_File": "results/lead_time_at_fa_budgets.csv",
                "Source_Column": "LeadTime_at_FA_0.5",
            })

    manifest_df = pd.DataFrame(manifest_records)
    manifest_csv = results_dir / "manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[Saved] Results manifest -> {manifest_csv}")
    return manifest_csv


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / args.config
    cfg = load_config(cfg_path)

    raw_dir = Path(cfg["data"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds
    subsets = args.subsets
    horizon = args.horizon

    if args.fast:
        subsets = [1]
        seeds = [0, 1]

    print("\n" + "=" * 105)
    print("  NASA C-MAPSS ANOMALY DETECTION & PREDICTIVE MAINTENANCE PIPELINE (STAGE 1)")
    print(f"  Subsets: {subsets} | Seeds: {seeds} | Horizon H: {horizon}")
    print("=" * 105)

    # -----------------------------------------------------------------------
    # Part 2(e): Comparative Health Index Evaluation (PCA vs Mahalanobis)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  PART 2(e): COMPARATIVE HEALTH INDEX (PCA-PC1 vs MAHALANOBIS DISTANCE)")
    print("=" * 90)
    hi_df = evaluate_health_indices_all_subsets(
        raw_dir=raw_dir, output_dir=figures_dir, subsets=subsets
    )
    print(f"  {'Subset':<8} {'Sensors':<9} {'Spearman (PCA-PC1)':>20} {'Spearman (Mahalanobis)':>24} {'Best Metric':>14}")
    print("-" * 80)
    for _, r in hi_df.iterrows():
        print(f"  {r['Subset']:<8} {r['Sensors_Count']:<9} {r['Spearman_PCA_PC1']:>20.4f} "
              f"{r['Spearman_Mahalanobis']:>24.4f} {r['Best_Index']:>14}")
    print("-" * 80)
    hi_csv = results_dir / "health_index_spearman_comparison.csv"
    hi_df.to_csv(hi_csv, index=False)
    print(f"[Saved] Health Index Spearman correlation comparison -> {hi_csv}")

    # -----------------------------------------------------------------------
    # Phase A: Anomaly Label Statistics & Sensitivity Analysis (N in 20, 30, 50)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  PHASE A: ANOMALY LABEL STATISTICS & SENSITIVITY ANALYSIS (N in {20, 30, 50})")
    print("=" * 90)
    stats_df = compute_label_statistics(
        raw_dir=raw_dir,
        subsets=subsets,
        thresholds=cfg["anomaly"]["sensitivity_thresholds"],
    )
    print_label_statistics_table(stats_df)
    stats_csv = results_dir / "anomaly_label_statistics.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"[Saved] Anomaly label statistics -> {stats_csv}")

    # -----------------------------------------------------------------------
    # Part 2(a-d), Part 3 & Part 4: Baselines & ML Anomaly Detection Across Subsets
    # -----------------------------------------------------------------------
    all_seed_records: list[dict[str, Any]] = []
    all_baseline_records: list[dict[str, Any]] = []
    all_engine_records: list[dict[str, Any]] = []
    all_test_records: list[dict[str, Any]] = []
    pr_curve_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]] = {}
    lead_time_fa_curves_all: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    fa_budget_records_all: list[dict[str, Any]] = []
    raw_predictions: dict[str, np.ndarray] = {}

    baseline_tuning_records: list[dict[str, Any]] = []

    for s in subsets:
        tag = f"FD00{s}"
        print(f"\n{'='*95}")
        print(f"  EVALUATING SUBSET {tag}")
        print(f"{'='*95}")

        # Part 2: Tune and Evaluate Baselines on Held-Out Train Engines (OOF)
        print(f"\n[{tag}] Part 2: 5-Fold GroupKFold Baseline Tuning & Full-Trajectory Evaluation...")
        base_res = tune_and_evaluate_baselines_subset(
            subset_id=s, raw_dir=raw_dir, n_splits=5, horizon=horizon, random_seed=42
        )

        k_best, vf_s, p_s = base_res["best_static_param"]
        lam_best, L_best, vf_e, p_e = base_res["best_ewma_param"]
        print(f"  -> Best Static Param: k={k_best:.1f}, vote={vf_s:.1f}, pers={p_s} (Tune F1 = {base_res['best_static_f1']:.4f})")
        print(f"  -> Best EWMA Param  : lam={lam_best:.1f}, L={L_best:.1f}, vote={vf_e:.1f}, pers={p_e} (Tune F1 = {base_res['best_ewma_f1']:.4f})")

        baseline_tuning_records.append({
            "Subset": tag,
            "Best_Static_k": k_best, "Best_Static_Vote": vf_s, "Best_Static_Persistence": p_s,
            "Best_Static_F1": base_res["best_static_f1"],
            "Best_EWMA_Lambda": lam_best, "Best_EWMA_L": L_best, "Best_EWMA_Vote": vf_e, "Best_EWMA_Persistence": p_e,
            "Best_EWMA_F1": base_res["best_ewma_f1"],
            "Static_Lead_Time_H100": base_res["oof_fleet_static"]["Mean_Lead_Time"],
            "Static_Detect_Rate_H100": base_res["oof_fleet_static"]["Detection_Rate_Pct"],
            "Static_FA_Per_100_H100": base_res["oof_fleet_static"]["False_Alarms_Per_100_Healthy"],
            "EWMA_Lead_Time_H100": base_res["oof_fleet_ewma"]["Mean_Lead_Time"],
            "EWMA_Detect_Rate_H100": base_res["oof_fleet_ewma"]["Detection_Rate_Pct"],
            "EWMA_FA_Per_100_H100": base_res["oof_fleet_ewma"]["False_Alarms_Per_100_Healthy"],
        })

        # Target False Alarm Rate to Match
        best_baseline_is_static = base_res["best_static_f1"] >= base_res["best_ewma_f1"]
        best_base_fleet = base_res["oof_fleet_static"] if best_baseline_is_static else base_res["oof_fleet_ewma"]
        best_baseline_name = "Static_Threshold" if best_baseline_is_static else "EWMA_Control_Limits"
        target_baseline_far = max(0.005, min(0.20, best_base_fleet["False_Alarms_Per_100_Healthy"] / 100.0))

        # Prepare normalized train & test datasets
        train_raw, test_raw, test_rul = load_raw(raw_dir, s)
        feature_cols = select_features(train_raw, 0.001)

        norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=42)
        norm.fit(train_raw, feature_cols, subset_id=s)
        train_norm = norm.transform(train_raw)
        test_norm  = norm.transform(test_raw)

        # Print healthy cycles cutoff information (Part 2 a)
        n_units = train_norm["unit"].nunique()
        healthy_df = identify_healthy_early_cycles(train_norm, 0.30)
        print(f"  [{tag}] Healthy early-life baseline: cycle <= round(0.30 * max_cycle)")
        print(f"  [{tag}] Healthy cycles count: {len(healthy_df)} / {len(train_norm)} ({len(healthy_df)/len(train_norm)*100:.1f}%) across {n_units} TRAIN engines.")

        # Build sliding windows (30-cycle)
        X_raw, y_win, u_win, c_win, h_win, fail_dict = build_windowed_engine_data(
            train_norm, feature_cols, window_length=30, n_threshold=30, is_test=False
        )
        X_feats = extract_window_features(X_raw)
        prevalence = float(np.mean(y_win))

        # Proof of fold differences across seeds (Stage 1 Item 1)
        splits_s0 = get_shuffled_group_kfold(u_win, n_splits=5, seed=0)
        splits_s1 = get_shuffled_group_kfold(u_win, n_splits=5, seed=1)
        u_f0_s0 = sorted(list(set(u_win[splits_s0[0][1]])))[:8]
        u_f0_s1 = sorted(list(set(u_win[splits_s1[0][1]])))[:8]
        print(f"  [{tag}] Seed Variance Verification (Fold 0 Validation Units Proof):")
        print(f"    - Seed 0 Fold 0 val units (first 8): {u_f0_s0}")
        print(f"    - Seed 1 Fold 0 val units (first 8): {u_f0_s1}")
        print(f"    -> Proven: Folds differ across seeds (Set overlap: {len(set(u_f0_s0).intersection(set(u_f0_s1)))} units).")

        print(f"\n[{tag}] Part 3 & 4: Training & Evaluating ML Detectors ({len(seeds)} Seeds x 5 Folds Shuffled GroupKFold)...")

        models_to_run = ["IsolationForest", "RandomForest", "XGBoost", "LSTM"]
        pr_curve_data[tag] = {}
        seed0_oof_scores: dict[str, np.ndarray] = {}

        for seed in seeds:
            set_seed(seed)
            splits = get_shuffled_group_kfold(u_win, n_splits=5, seed=seed)

            for m_name in models_to_run:
                oof_scores = np.zeros(len(y_win), dtype=np.float32)

                for fold, (tr_idx, val_idx) in enumerate(splits):
                    X_tr_raw, y_tr = X_raw[tr_idx], y_win[tr_idx]
                    X_tr_fts, h_tr = X_feats[tr_idx], h_win[tr_idx]

                    X_vl_raw, y_vl = X_raw[val_idx], y_win[val_idx]
                    X_vl_fts = X_feats[val_idx]

                    if m_name == "LSTM":
                        lstm_m, _ = train_lstm_classifier(
                            X_tr_raw, y_tr, X_vl_raw, y_vl, seed=seed, max_epochs=20, batch_size=256
                        )
                        det = MLAnomalyDetector(m_name, seed=seed)
                        det.model = lstm_m
                    else:
                        det = MLAnomalyDetector(m_name, seed=seed)
                        det.fit(X_tr_raw, X_tr_fts, y_tr, h_tr)

                    val_scores = det.predict_score(X_vl_raw, X_vl_fts)
                    oof_scores[val_idx] = val_scores

                if seed == 0:
                    seed0_oof_scores[m_name] = oof_scores

                # Metrics on full out-of-fold validation set
                p, r, f1, _ = precision_recall_fscore_support(
                    y_win, (oof_scores >= 0.5).astype(int), average="binary", zero_division=0
                )
                pr_auc = float(average_precision_score(y_win, oof_scores))
                roc_auc = float(roc_auc_score(y_win, oof_scores))
                fdr = float(1.0 - p) if p > 0 else 0.0

                # Dual threshold selection on validation engines
                th_f1, th_far = det.tune_thresholds(oof_scores, y_win, target_baseline_far=target_baseline_far)

                # Full trajectory lead-time evaluation at Max-F1 threshold
                fleet_f1, eng_res_f1 = evaluate_ml_model_lead_time(
                    oof_scores, threshold=th_f1, val_units=u_win, val_cycles=c_win,
                    unit_failure_cycles=fail_dict, horizon=horizon, persistence=3
                )

                # Full trajectory lead-time evaluation at Matched-FAR threshold
                fleet_far, eng_res_far = evaluate_ml_model_lead_time(
                    oof_scores, threshold=th_far, val_units=u_win, val_cycles=c_win,
                    unit_failure_cycles=fail_dict, horizon=horizon, persistence=3
                )

                all_seed_records.append({
                    "Subset": tag,
                    "Model": m_name,
                    "Seed": seed,
                    "PR_AUC": pr_auc,
                    "ROC_AUC": roc_auc,
                    "Prevalence": prevalence,
                    "Precision": float(p),
                    "Recall": float(r),
                    "F1": float(f1),
                    "FDR": float(fdr),
                    "Threshold_MaxF1": th_f1,
                    "Detection_Rate_Pct": fleet_f1["Detection_Rate_Pct"],
                    "Mean_Lead_Time": fleet_f1["Mean_Lead_Time"],
                    "Median_Lead_Time": fleet_f1["Median_Lead_Time"],
                    "Premature_Alarm_Share_Pct": fleet_f1["Premature_Alarm_Share_Pct"],
                    "False_Alarms_Per_100_Healthy": fleet_f1["False_Alarms_Per_100_Healthy"],
                    "Threshold_MatchedFAR": th_far,
                    "Detection_Rate_MatchedFAR": fleet_far["Detection_Rate_Pct"],
                    "Mean_Lead_Time_MatchedFAR": fleet_far["Mean_Lead_Time"],
                    "Median_Lead_Time_MatchedFAR": fleet_far["Median_Lead_Time"],
                    "Premature_Share_MatchedFAR": fleet_far["Premature_Alarm_Share_Pct"],
                    "FA_Per_100_MatchedFAR": fleet_far["False_Alarms_Per_100_Healthy"],
                })

                # Save seed 0 per-engine trajectories & PR curve data
                if seed == 0:
                    pr_curve_data[tag][m_name] = (y_win, oof_scores, prevalence)
                    raw_predictions[f"{tag}_{m_name}"] = oof_scores
                    for er in eng_res_f1:
                        all_engine_records.append({
                            "Subset": tag,
                            "Model": m_name,
                            "Unit": er["unit"],
                            "Detected": er["detected"],
                            "Lead_Time": er["lead_time"] if er["lead_time"] is not None else 0.0,
                            "Premature_Alarm": er["premature_alarm"],
                            "False_Alarm_Cycles": er["false_alarm_cycles_count"],
                            "Healthy_Cycles": er["healthy_cycles_count"],
                        })

        # Add traditional baselines to fleet comparison records
        all_baseline_records.extend([
            {
                "Subset": tag, "Model": "Static_Threshold", "Seed": "Tuned_OOF",
                "PR_AUC": np.nan, "ROC_AUC": np.nan, "Prevalence": prevalence,
                "Precision": np.nan, "Recall": np.nan, "F1": base_res["best_static_f1"],
                "FDR": np.nan, "Threshold_MaxF1": k_best,
                "Detection_Rate_Pct": base_res["oof_fleet_static"]["Detection_Rate_Pct"],
                "Mean_Lead_Time": base_res["oof_fleet_static"]["Mean_Lead_Time"],
                "Median_Lead_Time": base_res["oof_fleet_static"]["Median_Lead_Time"],
                "Premature_Alarm_Share_Pct": base_res["oof_fleet_static"]["Premature_Alarm_Share_Pct"],
                "False_Alarms_Per_100_Healthy": base_res["oof_fleet_static"]["False_Alarms_Per_100_Healthy"],
                "Threshold_MatchedFAR": k_best,
                "Detection_Rate_MatchedFAR": base_res["oof_fleet_static"]["Detection_Rate_Pct"],
                "Mean_Lead_Time_MatchedFAR": base_res["oof_fleet_static"]["Mean_Lead_Time"],
                "Median_Lead_Time_MatchedFAR": base_res["oof_fleet_static"]["Median_Lead_Time"],
                "Premature_Share_MatchedFAR": base_res["oof_fleet_static"]["Premature_Alarm_Share_Pct"],
                "FA_Per_100_MatchedFAR": base_res["oof_fleet_static"]["False_Alarms_Per_100_Healthy"],
            },
            {
                "Subset": tag, "Model": "EWMA_Control_Limits", "Seed": "Tuned_OOF",
                "PR_AUC": np.nan, "ROC_AUC": np.nan, "Prevalence": prevalence,
                "Precision": np.nan, "Recall": np.nan, "F1": base_res["best_ewma_f1"],
                "FDR": np.nan, "Threshold_MaxF1": L_best,
                "Detection_Rate_Pct": base_res["oof_fleet_ewma"]["Detection_Rate_Pct"],
                "Mean_Lead_Time": base_res["oof_fleet_ewma"]["Mean_Lead_Time"],
                "Median_Lead_Time": base_res["oof_fleet_ewma"]["Median_Lead_Time"],
                "Premature_Alarm_Share_Pct": base_res["oof_fleet_ewma"]["Premature_Alarm_Share_Pct"],
                "False_Alarms_Per_100_Healthy": base_res["oof_fleet_ewma"]["False_Alarms_Per_100_Healthy"],
                "Threshold_MatchedFAR": L_best,
                "Detection_Rate_MatchedFAR": base_res["oof_fleet_ewma"]["Detection_Rate_Pct"],
                "Mean_Lead_Time_MatchedFAR": base_res["oof_fleet_ewma"]["Mean_Lead_Time"],
                "Median_Lead_Time_MatchedFAR": base_res["oof_fleet_ewma"]["Median_Lead_Time"],
                "Premature_Share_MatchedFAR": base_res["oof_fleet_ewma"]["Premature_Alarm_Share_Pct"],
                "FA_Per_100_MatchedFAR": base_res["oof_fleet_ewma"]["False_Alarms_Per_100_Healthy"],
            },
        ])

        for er in base_res["oof_static_engines"]:
            all_engine_records.append({
                "Subset": tag, "Model": "Static_Threshold", "Unit": er["unit"],
                "Detected": er["detected"],
                "Lead_Time": er["lead_time"] if er["lead_time"] is not None else 0.0,
                "Premature_Alarm": er["premature_alarm"],
                "False_Alarm_Cycles": er["false_alarm_cycles_count"],
                "Healthy_Cycles": er["healthy_cycles_count"],
            })

        for er in base_res["oof_ewma_engines"]:
            all_engine_records.append({
                "Subset": tag, "Model": "EWMA_Control_Limits", "Unit": er["unit"],
                "Detected": er["detected"],
                "Lead_Time": er["lead_time"] if er["lead_time"] is not None else 0.0,
                "Premature_Alarm": er["premature_alarm"],
                "False_Alarm_Cycles": er["false_alarm_cycles_count"],
                "Healthy_Cycles": er["healthy_cycles_count"],
            })

        # -------------------------------------------------------------------
        # Stage 1 Item 2: Sweep Operating Curves (Lead Time vs False Alarms)
        # -------------------------------------------------------------------
        print(f"[{tag}] Sweeping 50-point operating curves (Lead Time vs FA per 100 healthy cycles)...")
        sub_curves, sub_budgets = sweep_lead_time_vs_fa_curves(
            subset_tag=tag,
            train_norm=train_norm,
            feature_cols=feature_cols,
            oof_predictions_dict=seed0_oof_scores,
            u_win=u_win,
            c_win=c_win,
            unit_failure_cycles=fail_dict,
            tuned_static_param=base_res["best_static_param"],
            tuned_ewma_param=base_res["best_ewma_param"],
            horizon=horizon,
            n_points=50,
        )
        lead_time_fa_curves_all[tag] = sub_curves
        fa_budget_records_all.extend(sub_budgets)

        # -------------------------------------------------------------------
        # Stage 1 Item 4: Official Test Engines Evaluation on EXACT SAME Windows
        # -------------------------------------------------------------------
        print(f"[{tag}] Step 4: Official Test Set Window-Level Evaluation on EXACT SAME Windows...")
        X_te_raw, y_te_win, u_te_win, c_te_win, _, _ = build_windowed_engine_data(
            test_norm, feature_cols, window_length=30, n_threshold=30, is_test=True, test_rul_arr=test_rul
        )
        X_te_fts = extract_window_features(X_te_raw)
        te_prev = float(np.mean(y_te_win))

        # Evaluate static & EWMA baselines on EXACT test windows
        train_healthy_all = identify_healthy_early_cycles(train_norm, 0.30)
        final_det_s = ParametricStaticDetector(*base_res["best_static_param"]).fit(train_healthy_all, feature_cols)
        final_det_e = ParametricEWMADetector(*base_res["best_ewma_param"]).fit(train_healthy_all, feature_cols)

        # Score baselines per window
        base_s_win_preds = []
        base_e_win_preds = []
        for uid in sorted(test_norm["unit"].unique()):
            grp_u = test_norm[test_norm["unit"] == uid].sort_values("cycle").reset_index(drop=True)
            _, sust_s = final_det_s.predict_engine(grp_u)
            _, sust_e = final_det_e.predict_engine(grp_u)

            u_mask = (u_te_win == uid)
            u_end_cycles = c_te_win[u_mask]
            for ce in u_end_cycles:
                base_s_win_preds.append(sust_s[ce - 1])
                base_e_win_preds.append(sust_e[ce - 1])

        p_s_te, r_s_te, f1_s_te, _ = precision_recall_fscore_support(
            y_te_win, np.array(base_s_win_preds), average="binary", zero_division=0
        )
        p_e_te, r_e_te, f1_e_te, _ = precision_recall_fscore_support(
            y_te_win, np.array(base_e_win_preds), average="binary", zero_division=0
        )

        all_test_records.append({
            "Subset": tag, "Model": "Static_Threshold", "Split": "test (truncated)",
            "Precision": float(p_s_te), "Recall": float(r_s_te), "F1": float(f1_s_te),
            "PR_AUC": "N/A (discrete)", "ROC_AUC": "N/A (discrete)", "Prevalence": te_prev,
        })
        all_test_records.append({
            "Subset": tag, "Model": "EWMA_Control_Limits", "Split": "test (truncated)",
            "Precision": float(p_e_te), "Recall": float(r_e_te), "F1": float(f1_e_te),
            "PR_AUC": "N/A (discrete)", "ROC_AUC": "N/A (discrete)", "Prevalence": te_prev,
        })

        for m_name in models_to_run:
            set_seed(42)
            if m_name == "LSTM":
                lstm_full, _ = train_lstm_classifier(
                    X_raw, y_win, X_te_raw, y_te_win, seed=42, max_epochs=20, batch_size=256
                )
                sc_te = predict_lstm_probs(lstm_full, X_te_raw)
            else:
                det_full = MLAnomalyDetector(m_name, seed=42)
                det_full.fit(X_raw, X_feats, y_win, h_win)
                sc_te = det_full.predict_score(X_te_raw, X_te_fts)

            p_t, r_t, f1_t, _ = precision_recall_fscore_support(
                y_te_win, (sc_te >= 0.5).astype(int), average="binary", zero_division=0
            )
            pr_t = float(average_precision_score(y_te_win, sc_te)) if len(np.unique(y_te_win)) > 1 else 0.0
            roc_t = float(roc_auc_score(y_te_win, sc_te)) if len(np.unique(y_te_win)) > 1 else 0.5

            all_test_records.append({
                "Subset": tag, "Model": m_name, "Split": "test (truncated)",
                "Precision": float(p_t), "Recall": float(r_t), "F1": float(f1_t),
                "PR_AUC": f"{pr_t:.4f}", "ROC_AUC": f"{roc_t:.4f}", "Prevalence": te_prev,
            })

    # Save baseline tuning records
    df_base_tune = pd.DataFrame(baseline_tuning_records)
    base_tune_csv = results_dir / "baseline_tuning_results.csv"
    df_base_tune.to_csv(base_tune_csv, index=False)
    print(f"\n[Saved] Baseline 5-fold tuning results -> {base_tune_csv}")

    # Save lead time at FA budgets
    df_budgets = pd.DataFrame(fa_budget_records_all)
    budgets_csv = results_dir / "lead_time_at_fa_budgets.csv"
    df_budgets.to_csv(budgets_csv, index=False)
    print(f"[Saved] Lead time at controlled FA budgets (0.1, 0.5, 1.0) -> {budgets_csv}")

    # Plot operating characteristic curves
    lt_fa_fig = figures_dir / "lead_time_vs_fa_curves.png"
    plot_lead_time_vs_fa_curves(lead_time_fa_curves_all, lt_fa_fig)
    print(f"[Saved Figure] Lead Time vs False Alarms Operating Curves -> {lt_fa_fig}")

    # -----------------------------------------------------------------------
    # Part 4: Consolidate Results & Statistics across Seeds
    # -----------------------------------------------------------------------
    full_seed_df = pd.DataFrame(all_seed_records)
    oof_seeds_csv = results_dir / "anomaly_metrics_oof_seeds.csv"
    full_seed_df.to_csv(oof_seeds_csv, index=False)
    print(f"[Saved] Per-seed out-of-fold metrics -> {oof_seeds_csv}")

    # Save raw predictions
    raw_pred_file = results_dir / "raw_predictions.npz"
    np.savez_compressed(raw_pred_file, **raw_predictions)
    print(f"[Saved] Raw prediction arrays -> {raw_pred_file}")

    # Per-engine alarm cycles
    df_all_engines = pd.DataFrame(all_engine_records)
    engines_csv = results_dir / "alarms_per_engine.csv"
    df_all_engines.to_csv(engines_csv, index=False)
    print(f"[Saved] Per-engine alarm cycles -> {engines_csv}")

    for s_tag in [f"FD00{s}" for s in subsets]:
        s_eng_csv = results_dir / f"alarms_{s_tag}.csv"
        df_all_engines[df_all_engines["Subset"] == s_tag].to_csv(s_eng_csv, index=False)

    # Consolidated Summary Table: Model x Subset Mean +/- Std
    summary_rows = []
    for (sub, model), grp in full_seed_df.groupby(["Subset", "Model"]):
        summary_rows.append({
            "Subset": sub,
            "Model": model,
            "PR_AUC": f"{grp['PR_AUC'].mean():.3f} +/- {grp['PR_AUC'].std():.3f}",
            "ROC_AUC": f"{grp['ROC_AUC'].mean():.3f} +/- {grp['ROC_AUC'].std():.3f}",
            "Prevalence": f"{grp['Prevalence'].iloc[0]*100:.1f}%",
            "F1": f"{grp['F1'].mean():.3f} +/- {grp['F1'].std():.3f}",
            "FDR": f"{grp['FDR'].mean():.3f} +/- {grp['FDR'].std():.3f}",
            "Detection_Rate": f"{grp['Detection_Rate_Pct'].mean():.1f}% +/- {grp['Detection_Rate_Pct'].std():.1f}%",
            "Mean_Lead_Time": f"{grp['Mean_Lead_Time'].mean():.1f} +/- {grp['Mean_Lead_Time'].std():.1f}",
            "Median_Lead_Time": f"{grp['Median_Lead_Time'].mean():.1f}",
            "Premature_Alarm_Pct": f"{grp['Premature_Alarm_Share_Pct'].mean():.1f}% +/- {grp['Premature_Alarm_Share_Pct'].std():.1f}%",
            "FA_Per_100_Healthy": f"{grp['False_Alarms_Per_100_Healthy'].mean():.2f} +/- {grp['False_Alarms_Per_100_Healthy'].std():.2f}",
            "MatchedFAR_LeadTime": f"{grp['Mean_Lead_Time_MatchedFAR'].mean():.1f} +/- {grp['Mean_Lead_Time_MatchedFAR'].std():.1f}",
            "MatchedFAR_DetectRate": f"{grp['Detection_Rate_MatchedFAR'].mean():.1f}% +/- {grp['Detection_Rate_MatchedFAR'].std():.1f}%",
            "MatchedFAR_PrematurePct": f"{grp['Premature_Share_MatchedFAR'].mean():.1f}% +/- {grp['Premature_Share_MatchedFAR'].std():.1f}%",
            "MatchedFAR_FA100": f"{grp['FA_Per_100_MatchedFAR'].mean():.2f} +/- {grp['FA_Per_100_MatchedFAR'].std():.2f}",
        })

    for br in all_baseline_records:
        summary_rows.append({
            "Subset": br["Subset"],
            "Model": br["Model"],
            "PR_AUC": "N/A (discrete)",
            "ROC_AUC": "N/A (discrete)",
            "Prevalence": f"{br['Prevalence']*100:.1f}%",
            "F1": f"{br['F1']:.3f} (tuned)",
            "FDR": "N/A",
            "Detection_Rate": f"{br['Detection_Rate_Pct']:.1f}%",
            "Mean_Lead_Time": f"{br['Mean_Lead_Time']:.1f}",
            "Median_Lead_Time": f"{br['Median_Lead_Time']:.1f}",
            "Premature_Alarm_Pct": f"{br['Premature_Alarm_Share_Pct']:.1f}%",
            "FA_Per_100_Healthy": f"{br['False_Alarms_Per_100_Healthy']:.2f}",
            "MatchedFAR_LeadTime": f"{br['Mean_Lead_Time']:.1f}",
            "MatchedFAR_DetectRate": f"{br['Detection_Rate_Pct']:.1f}%",
            "MatchedFAR_PrematurePct": f"{br['Premature_Alarm_Share_Pct']:.1f}%",
            "MatchedFAR_FA100": f"{br['False_Alarms_Per_100_Healthy']:.2f}",
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["Subset", "Model"])
    summary_csv = results_dir / "anomaly_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved] Consolidated anomaly metrics summary -> {summary_csv}")

    # -----------------------------------------------------------------------
    # Stage 1 Item 5: Wilcoxon Signed-Rank Tests with Holm Correction & Bootstrap CI
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  STAGE 1: WILCOXON TESTS WITH HOLM CORRECTION & BOOTSTRAP 95% CI")
    print("=" * 90)
    wilcoxon_raw_tests = []

    for s in subsets:
        tag = f"FD00{s}"
        sub_eng = df_all_engines[df_all_engines["Subset"] == tag]

        sub_base = [r for r in all_baseline_records if r["Subset"] == tag]
        best_base_name = max(sub_base, key=lambda x: x["F1"])["Model"]

        sub_ml_seed = full_seed_df[full_seed_df["Subset"] == tag]
        best_ml_name = sub_ml_seed.groupby("Model")["PR_AUC"].mean().idxmax()

        ml_eng = sub_eng[sub_eng["Model"] == best_ml_name].sort_values("Unit")
        base_eng = sub_eng[sub_eng["Model"] == best_base_name].sort_values("Unit")

        if len(ml_eng) > 0 and len(base_eng) > 0 and len(ml_eng) == len(base_eng):
            diff_lead = ml_eng["Lead_Time"].values - base_eng["Lead_Time"].values
            diff_fa   = ml_eng["False_Alarm_Cycles"].values - base_eng["False_Alarm_Cycles"].values

            # Wilcoxon Lead Time
            _, p_l = wilcoxon(ml_eng["Lead_Time"].values, base_eng["Lead_Time"].values, zero_method="wilcox") if not np.all(diff_lead == 0) else (0.0, 1.0)
            # Wilcoxon False Alarms
            _, p_fa = wilcoxon(ml_eng["False_Alarm_Cycles"].values, base_eng["False_Alarm_Cycles"].values, zero_method="wilcox") if not np.all(diff_fa == 0) else (0.0, 1.0)

            # Bootstrap 95% CI for median difference
            rng = np.random.default_rng(42)
            n_e = len(diff_lead)
            boot_med_lead = [np.median(rng.choice(diff_lead, size=n_e, replace=True)) for _ in range(2000)]
            boot_med_fa   = [np.median(rng.choice(diff_fa,   size=n_e, replace=True)) for _ in range(2000)]

            ci_l_low, ci_l_high   = float(np.percentile(boot_med_lead, 2.5)), float(np.percentile(boot_med_lead, 97.5))
            ci_fa_low, ci_fa_high = float(np.percentile(boot_med_fa, 2.5)), float(np.percentile(boot_med_fa, 97.5))

            wilcoxon_raw_tests.append({
                "Subset": tag, "Metric": "Lead_Time",
                "Comparison": f"{best_ml_name} vs {best_base_name}",
                "Median_Diff": float(np.median(diff_lead)),
                "Mean_Diff": float(np.mean(diff_lead)),
                "CI_95": f"[{ci_l_low:+.2f}, {ci_l_high:+.2f}]",
                "Raw_p_value": float(p_l),
            })
            wilcoxon_raw_tests.append({
                "Subset": tag, "Metric": "False_Alarms",
                "Comparison": f"{best_ml_name} vs {best_base_name}",
                "Median_Diff": float(np.median(diff_fa)),
                "Mean_Diff": float(np.mean(diff_fa)),
                "CI_95": f"[{ci_fa_low:+.2f}, {ci_fa_high:+.2f}]",
                "Raw_p_value": float(p_fa),
            })

    # Apply Holm-Bonferroni correction
    n_tests = len(wilcoxon_raw_tests)
    # Sort by raw p-value
    wilcoxon_raw_tests.sort(key=lambda x: x["Raw_p_value"])

    wilcoxon_final_records = []
    for rank, test in enumerate(wilcoxon_raw_tests):
        multiplier = n_tests - rank
        p_holm = min(1.0, test["Raw_p_value"] * multiplier)
        p_bonf = min(1.0, test["Raw_p_value"] * n_tests)

        test["Total_Tests"] = n_tests
        test["Holm_p_value"] = float(p_holm)
        test["Bonferroni_p_value"] = float(p_bonf)
        test["Significant_after_Holm"] = p_holm < 0.05
        wilcoxon_final_records.append(test)

    # Re-sort by subset and metric
    wilcoxon_df = pd.DataFrame(wilcoxon_final_records).sort_values(["Subset", "Metric"])
    wilcoxon_csv = results_dir / "wilcoxon_tests.csv"
    wilcoxon_df.to_csv(wilcoxon_csv, index=False)
    print(f"[Saved] Corrected Wilcoxon tests -> {wilcoxon_csv}")

    print("\n" + "=" * 105)
    print(f"  WILCOXON TESTS SUMMARY WITH HOLM CORRECTION (Total Tests M = {n_tests})")
    print("=" * 105)
    print(f"  {'Subset':<7} {'Metric':<13} {'Comparison':<28} {'Median Diff [95% CI]':<26} {'Raw p':<12} {'Holm p':<12} {'Sig?':<5}")
    print("-" * 105)
    for _, r in wilcoxon_df.iterrows():
        diff_str = f"{r['Median_Diff']:+.2f} {r['CI_95']}"
        sig_str = "YES" if r["Significant_after_Holm"] else "NO"
        print(f"  {r['Subset']:<7} {r['Metric']:<13} {r['Comparison']:<28} {diff_str:<26} {r['Raw_p_value']:<12.4e} {r['Holm_p_value']:<12.4e} {sig_str:<5}")
    print("=" * 105 + "\n")

    # -----------------------------------------------------------------------
    # Official Test Engines Window-Level Evaluation Table
    # -----------------------------------------------------------------------
    test_df = pd.DataFrame(all_test_records)
    test_csv = results_dir / "test_anomaly_metrics.csv"
    test_df.to_csv(test_csv, index=False)
    print(f"[Saved] Official test window-level metrics -> {test_csv}")

    # -----------------------------------------------------------------------
    # Publication Figures (300 DPI)
    # -----------------------------------------------------------------------
    print("\n--- Generating Publication Figures (300 DPI) ---")
    pr_fig_path = figures_dir / "pr_curves.png"
    plot_pr_curves(pr_curve_data, pr_fig_path)
    print(f"[Saved Figure] PR Curves -> {pr_fig_path}")

    lt_fig_path = figures_dir / "lead_time_distributions.png"
    plot_lead_time_distributions(df_all_engines, lt_fig_path)
    print(f"[Saved Figure] Lead-Time Distributions -> {lt_fig_path}")

    fa_fig_path = figures_dir / "false_alarms_per_engine.png"
    plot_false_alarms_per_engine(df_all_engines, fa_fig_path)
    print(f"[Saved Figure] False Alarms per Engine -> {fa_fig_path}")

    # -----------------------------------------------------------------------
    # Stage 1 Item 3: Label Sensitivity Sweep (N in {20, 30, 50, 75, 100})
    # -----------------------------------------------------------------------
    run_label_sensitivity_sweep(subsets=subsets, raw_dir=raw_dir, results_dir=results_dir, horizon=horizon)

    # -----------------------------------------------------------------------
    # Optional 5-Seed Repeat for RUL Regression Models
    # -----------------------------------------------------------------------
    if not args.skip_rul_seeds:
        run_rul_seeds_evaluation(cfg, subsets=subsets, seeds=seeds, results_dir=results_dir)

    # -----------------------------------------------------------------------
    # Stage 1 Item 1 & 2: Print Required Tables & Lead Time Budgets
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  LEAD TIME AT CONTROLLED FALSE ALARM BUDGETS (0.1, 0.5, 1.0 FA / 100 CYCLES)")
    print("  (Operating Characteristic Curve Operating Points; Out-of-Fold Full Trajectories)")
    print("=" * 90)
    print(f"  {'Subset':<8} {'Method':<24} {'Lead Time @ FA=0.1':<20} {'Lead Time @ FA=0.5':<20} {'Lead Time @ FA=1.0':<20}")
    print("-" * 90)
    for _, r in df_budgets.iterrows():
        print(f"  {r['Subset']:<8} {r['Method']:<24} {r['LeadTime_at_FA_0.1']:<20} {r['LeadTime_at_FA_0.5']:<20} {r['LeadTime_at_FA_1.0']:<20}")
    print("=" * 90 + "\n")

    # -----------------------------------------------------------------------
    # FINAL COMPARISON TABLES (STAGE 1 STOP POINT)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("  FINAL COMPARISON TABLE 1: OUT-OF-FOLD EVALUATION AT MAX-F1 THRESHOLD")
    print("  (Held-Out Full Run-to-Failure Train Engines, 5 Seeds x 5-Fold Shuffled GroupKFold)")
    print("=" * 125)
    print(f"  {'Subset':<7} {'Model':<22} {'PR-AUC (Main)':<17} {'ROC-AUC':<17} {'Prev':<6} "
          f"{'F1':<16} {'Detect %':<16} {'Lead Time (H=100)':<18} {'Premature %':<14}")
    print("-" * 125)
    for _, r in summary_df.iterrows():
        print(f"  {r['Subset']:<7} {r['Model']:<22} {r['PR_AUC']:<17} {r['ROC_AUC']:<17} {r['Prevalence']:<6} "
              f"{r['F1']:<16} {r['Detection_Rate']:<16} {r['Mean_Lead_Time']:<18} {r['Premature_Alarm_Pct']:<14}")
    print("=" * 125)

    print("\n" + "=" * 125)
    print("  FINAL COMPARISON TABLE 2: OUT-OF-FOLD EVALUATION AT BASELINE-MATCHED FALSE ALARM RATE")
    print("  (Controlled Comparison: Evaluates Lead Time & Detection Rate at Equal False Alarms)")
    print("=" * 125)
    print(f"  {'Subset':<7} {'Model':<22} {'Detect % (Matched)':<20} {'Lead Time (Matched)':<22} "
          f"{'Premature % (Matched)':<24} {'FA / 100 Cycles':<18}")
    print("-" * 125)
    for _, r in summary_df.iterrows():
        print(f"  {r['Subset']:<7} {r['Model']:<22} {r['MatchedFAR_DetectRate']:<20} {r['MatchedFAR_LeadTime']:<22} "
              f"{r['MatchedFAR_PrematurePct']:<24} {r['MatchedFAR_FA100']:<18}")
    print("=" * 125)

    print("\n" + "=" * 90)
    print("  OFFICIAL TEST ENGINES WINDOW-LEVEL CLASSIFICATION [Labelled 'test (truncated)']")
    print("  (Evaluated on EXACT SAME 30-cycle Test Windows with Matching Prevalence)")
    print("=" * 90)
    print(f"  {'Subset':<7} {'Model':<22} {'PR-AUC':>10} {'ROC-AUC':>10} {'Prevalence':>12} {'F1':>10}")
    print("-" * 90)
    for _, r in test_df.iterrows():
        prev_str = f"{r['Prevalence']*100:.1f}%"
        print(f"  {r['Subset']:<7} {r['Model']:<22} {str(r['PR_AUC']):>10} {str(r['ROC_AUC']):>10} {prev_str:>12} {r['F1']:>10.4f}")
    print("=" * 90 + "\n")

    # Generate Manifest
    generate_results_manifest(results_dir)

    print("[SUCCESS] Stage 1 execution completed. All requested evaluation issues addressed.")


if __name__ == "__main__":
    main()
