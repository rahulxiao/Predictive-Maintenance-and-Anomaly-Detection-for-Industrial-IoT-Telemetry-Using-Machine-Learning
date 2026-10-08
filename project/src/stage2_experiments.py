"""
stage2_experiments.py - Implementation of Stage 2 Novelty Experiments.

Novelty Experiments:
  1. Cross-Condition Generalization (5 seeds):
     - Train FD001 -> Test FD002 and FD004; Train FD003 -> Test FD004;
     - Plus in-domain references (FD002->FD002, FD004->FD004, FD001->FD001, FD003->FD003).
     - Operating-regime normalizer fitted strictly on TRAINING domain only.
     - Reports RUL (RMSE, RMSE_clip, NASA) and Anomaly (F1, PR-AUC, Lead Time at FA=0.5) and drops.
  2. Uncertainty Quantification via Split Conformal Prediction:
     - Calibrated on whole held-out TRAIN engines (never test engines).
     - Reports empirical coverage at 90% target coverage and mean interval width on official test engines.
     - Plots predicted RUL with 90% confidence bands for 6 test engines.
  3. Explainability with SHAP:
     - TreeSHAP for best tree model per subset; top 10 sensors.
     - Sensor overlap table comparing top SHAP sensors with sensors driving the static baseline.
     - Publication plot at 300 DPI.
  4. Ablations (3 seeds: 0, 1, 2):
     - Smoothing: ON vs OFF.
     - Window length: 15, 30, 50.
     - Feature representation: Raw window, PCA 95% variance, Engineered statistics.
     - Answers Research Question 4 from verified numbers only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from data_loader import add_rul_train, load_raw, make_train_windows, make_test_windows, select_features
from evaluate import evaluate_all
from features_extractor import extract_window_features
from metrics_lead_time import compute_sustained_alarms, evaluate_engine_lead_time, evaluate_fleet_lead_time
from regime_norm import OperatingRegimeNormalizer, identify_healthy_early_cycles
from utils import get_logger, set_seed, Timer

logger = get_logger("stage2_exp")

# Common 14 sensors present with non-zero variance across all 4 subsets
COMMON_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]


# ---------------------------------------------------------------------------
# Helper: Build Sliding Windows for Anomaly Classification
# ---------------------------------------------------------------------------

def _build_windowed_anomaly_data(
    df_norm: pd.DataFrame,
    feature_cols: list[str],
    window_length: int = 30,
    n_threshold: int = 30,
    is_test: bool = False,
    test_rul_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct sliding windows from normalized DataFrame for anomaly detection.
    Returns: (X_raw, y_win, u_win, c_win, h_win)
    """
    X_list, y_list, u_list, c_list, h_list = [], [], [], [], []

    if is_test:
        assert test_rul_arr is not None
        sorted_units = sorted(df_norm["unit"].unique())
        unit_rul_map = {u: float(test_rul_arr[i]) for i, u in enumerate(sorted_units)}

        for uid in sorted_units:
            grp = df_norm[df_norm["unit"] == uid].sort_values("cycle").reset_index(drop=True)
            feats = grp[feature_cols].values.astype(np.float32)
            n_c = len(grp)
            final_rul = unit_rul_map[uid]
            fail_c = n_c + final_rul

            for start in range(0, n_c - window_length + 1):
                end = start + window_length
                end_cycle = int(grp.loc[end - 1, "cycle"])
                current_rul = fail_c - end_cycle
                label = 1 if current_rul <= n_threshold else 0
                is_healthy = 1 if current_rul > 100 else 0

                X_list.append(feats[start:end])
                y_list.append(label)
                u_list.append(uid)
                c_list.append(end_cycle)
                h_list.append(is_healthy)
    else:
        for uid, grp in df_norm.groupby("unit"):
            grp = grp.sort_values("cycle").reset_index(drop=True)
            feats = grp[feature_cols].values.astype(np.float32)
            max_c = grp["cycle"].max()
            n_c = len(grp)

            for start in range(0, n_c - window_length + 1):
                end = start + window_length
                end_cycle = int(grp.loc[end - 1, "cycle"])
                current_rul = max_c - end_cycle
                label = 1 if current_rul <= n_threshold else 0
                is_healthy = 1 if current_rul > 100 else 0

                X_list.append(feats[start:end])
                y_list.append(label)
                u_list.append(int(uid))
                c_list.append(end_cycle)
                h_list.append(is_healthy)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(u_list, dtype=np.int32),
        np.array(c_list, dtype=np.int32),
        np.array(h_list, dtype=np.int32),
    )


def _sweep_lead_time_at_fa(
    model: xgb.XGBClassifier,
    target_norm: pd.DataFrame,
    feature_cols: list[str],
    target_rul_arr: np.ndarray | None = None,
    is_test: bool = False,
    window_length: int = 30,
    target_fa_budget: float = 0.5,
    n_points: int = 50,
) -> float:
    """
    Sweeps 50 alarm thresholds across target engine trajectories to find lead time
    at controlled false alarm budget (default FA = 0.5 per 100 healthy cycles).
    """
    sorted_units = sorted(target_norm["unit"].unique())
    thresholds = np.linspace(0.01, 0.99, n_points)

    # Pre-extract engine features and scores
    engine_data = []
    for i, uid in enumerate(sorted_units):
        grp = target_norm[target_norm["unit"] == uid].sort_values("cycle").reset_index(drop=True)
        feats = grp[feature_cols].values.astype(np.float32)
        n_c = len(grp)
        cycles = grp["cycle"].values.astype(int)

        if is_test:
            assert target_rul_arr is not None
            fail_c = n_c + float(target_rul_arr[i])
        else:
            fail_c = float(grp["cycle"].max())

        # Build sliding windows for engine
        if n_c < window_length:
            continue

        w_feats = []
        w_cycles = []
        for start in range(0, n_c - window_length + 1):
            end = start + window_length
            w_feats.append(feats[start:end])
            w_cycles.append(cycles[end - 1])

        X_w = np.array(w_feats, dtype=np.float32)
        X_w_stats = extract_window_features(X_w)
        scores = model.predict_proba(X_w_stats)[:, 1]

        # Map back to continuous cycles (pad earliest cycles with first window score)
        full_scores = np.zeros(n_c, dtype=np.float32)
        first_c = w_cycles[0]
        full_scores[:first_c - 1] = scores[0]
        for idx, c in enumerate(w_cycles):
            full_scores[c - 1] = scores[idx]

        engine_data.append({
            "uid": uid,
            "cycles": cycles,
            "scores": full_scores,
            "fail_c": fail_c,
        })

    # Sweep thresholds
    curve_fa = []
    curve_lt = []

    for th in thresholds:
        eng_results = []
        for eng in engine_data:
            raw_alarms = (eng["scores"] >= th).astype(int)
            sust = compute_sustained_alarms(raw_alarms, persistence=3)
            res = evaluate_engine_lead_time(eng["cycles"], sust, failure_cycle=eng["fail_c"], horizon=100)
            res["unit"] = eng["uid"]
            eng_results.append(res)

        fleet = evaluate_fleet_lead_time(eng_results, horizon=100)
        curve_fa.append(fleet["False_Alarms_Per_100_Healthy"])
        curve_lt.append(fleet["Mean_Lead_Time"])

    arr_fa = np.array(curve_fa)
    arr_lt = np.array(curve_lt)

    # Sort by false alarm rate
    sort_idx = np.argsort(arr_fa)
    s_fa = arr_fa[sort_idx]
    s_lt = arr_lt[sort_idx]

    if target_fa_budget < s_fa.min() or target_fa_budget > s_fa.max():
        # If budget cannot be reached, return closest or nan
        if s_fa.min() > target_fa_budget:
            return float("nan")
        return float(s_lt[s_fa <= target_fa_budget][-1])

    val = float(np.interp(target_fa_budget, s_fa, s_lt))
    return val


# ---------------------------------------------------------------------------
# 1. Cross-Condition Generalization Experiment
# ---------------------------------------------------------------------------

def run_cross_condition_experiments(
    raw_dir: Path,
    results_dir: Path,
    seeds: list[int] = [0, 1, 2, 3, 4],
) -> pd.DataFrame:
    """
    Executes Cross-Condition Generalization across 5 seeds:
      Pairs:
        - FD001 -> FD002
        - FD001 -> FD004
        - FD003 -> FD004
      Plus in-domain references:
        - FD001 -> FD001
        - FD002 -> FD002
        - FD003 -> FD003
        - FD004 -> FD004

    Key constraint:
      OperatingRegimeNormalizer is fitted strictly on the TRAINING domain only!
    """
    logger.info("=" * 60)
    logger.info("STAGE 2 - EXPERIMENT 1: Cross-Condition Generalization")
    logger.info("=" * 60)

    pairs = [
        # In-domain references
        ("FD001", "FD001", 1, 1, "In_Domain"),
        ("FD002", "FD002", 2, 2, "In_Domain"),
        ("FD003", "FD003", 3, 3, "In_Domain"),
        ("FD004", "FD004", 4, 4, "In_Domain"),
        # Cross-condition transfers
        ("FD001", "FD002", 1, 2, "Cross_Domain"),
        ("FD001", "FD004", 1, 4, "Cross_Domain"),
        ("FD003", "FD004", 3, 4, "Cross_Domain"),
    ]

    all_seed_results = []

    for src_tag, tgt_tag, src_id, tgt_id, pair_type in pairs:
        logger.info(f"\nEvaluating Pair: {src_tag} -> {tgt_tag} ({pair_type})")

        # Load raw data
        train_src_raw, _, _ = load_raw(raw_dir, src_id)
        _, test_tgt_raw, test_tgt_rul = load_raw(raw_dir, tgt_id)
        train_tgt_raw, _, _ = load_raw(raw_dir, tgt_id)

        for seed in seeds:
            set_seed(seed)

            # CRITICAL: Fit OperatingRegimeNormalizer on TRAINING domain only!
            norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=seed)
            norm.fit(train_src_raw, COMMON_SENSORS, subset_id=src_id)

            # Transform source training and target test data
            train_src_norm = norm.transform(train_src_raw)
            test_tgt_norm  = norm.transform(test_tgt_raw)
            train_tgt_norm = norm.transform(train_tgt_raw)

            # --- A. RUL Regression Model ---
            # Prepare RUL training data
            train_src_rul_df = add_rul_train(train_src_norm, rul_clip=125)
            X_tr_rul, y_tr_rul, _ = make_train_windows(
                train_src_rul_df, COMMON_SENSORS, window_length=30, stride=1
            )
            X_te_rul, y_te_rul = make_test_windows(
                test_tgt_norm, COMMON_SENSORS, test_tgt_rul, window_length=30
            )

            # Extract engineered features for tree regressor
            X_tr_rul_feats = extract_window_features(X_tr_rul)
            X_te_rul_feats = extract_window_features(X_te_rul)

            xgb_rul = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                tree_method="hist",
                n_jobs=-1,
                verbosity=0,
            )
            xgb_rul.fit(X_tr_rul_feats, y_tr_rul)
            preds_rul = np.clip(xgb_rul.predict(X_te_rul_feats), 0.0, 125.0)
            metrics_rul = evaluate_all(y_te_rul, preds_rul, rul_clip=125)

            # --- B. Anomaly Detection Model ---
            X_tr_ano_raw, y_tr_ano, _, _, _ = _build_windowed_anomaly_data(
                train_src_norm, COMMON_SENSORS, window_length=30, n_threshold=30, is_test=False
            )
            X_te_ano_raw, y_te_ano, _, _, _ = _build_windowed_anomaly_data(
                test_tgt_norm, COMMON_SENSORS, window_length=30, n_threshold=30, is_test=True, test_rul_arr=test_tgt_rul
            )

            X_tr_ano_feats = extract_window_features(X_tr_ano_raw)
            X_te_ano_feats = extract_window_features(X_te_ano_raw)

            pos = float(np.sum(y_tr_ano == 1))
            neg = float(np.sum(y_tr_ano == 0))
            scale_pos = (neg / pos) if pos > 0 else 1.0

            xgb_ano = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )
            xgb_ano.fit(X_tr_ano_feats, y_tr_ano)
            ano_probs = xgb_ano.predict_proba(X_te_ano_feats)[:, 1]

            # Classification metrics on target test windows
            p_val, r_val, f1_val, _ = precision_recall_fscore_support(
                y_te_ano, (ano_probs >= 0.5).astype(int), average="binary", zero_division=0
            )
            pr_auc_val = float(average_precision_score(y_te_ano, ano_probs)) if len(np.unique(y_te_ano)) > 1 else 0.0

            # Lead time at FA = 0.5 per 100 healthy cycles evaluated on target full trajectories
            lt_fa05 = _sweep_lead_time_at_fa(
                xgb_ano,
                target_norm=train_tgt_norm,
                feature_cols=COMMON_SENSORS,
                target_rul_arr=None,
                is_test=False,
                target_fa_budget=0.5,
                n_points=50,
            )

            all_seed_results.append({
                "Source_Domain": src_tag,
                "Target_Domain": tgt_tag,
                "Pair": f"{src_tag}->{tgt_tag}",
                "Pair_Type": pair_type,
                "Seed": seed,
                "RMSE": metrics_rul["RMSE"],
                "RMSE_clip": metrics_rul["RMSE_clip"],
                "NASA_Score": metrics_rul["NASA_Score"],
                "Anomaly_F1": float(f1_val),
                "Anomaly_PRAUC": float(pr_auc_val),
                "Lead_Time_at_FA_0.5": lt_fa05,
            })

    df_seeds = pd.DataFrame(all_seed_results)

    # Aggregate Mean +/- Std across seeds
    summary_rows = []
    in_domain_benchmarks = {}

    for pair, grp in df_seeds.groupby("Pair"):
        src = grp["Source_Domain"].iloc[0]
        tgt = grp["Target_Domain"].iloc[0]
        ptype = grp["Pair_Type"].iloc[0]

        row = {
            "Source_Domain": src,
            "Target_Domain": tgt,
            "Pair": pair,
            "Pair_Type": ptype,
            "RUL_RMSE_Mean": grp["RMSE"].mean(),
            "RUL_RMSE_Std": grp["RMSE"].std(),
            "RUL_RMSE_clip_Mean": grp["RMSE_clip"].mean(),
            "RUL_RMSE_clip_Std": grp["RMSE_clip"].std(),
            "RUL_NASA_Mean": grp["NASA_Score"].mean(),
            "RUL_NASA_Std": grp["NASA_Score"].std(),
            "Anomaly_F1_Mean": grp["Anomaly_F1"].mean(),
            "Anomaly_F1_Std": grp["Anomaly_F1"].std(),
            "Anomaly_PRAUC_Mean": grp["Anomaly_PRAUC"].mean(),
            "Anomaly_PRAUC_Std": grp["Anomaly_PRAUC"].std(),
            "Lead_Time_FA05_Mean": grp["Lead_Time_at_FA_0.5"].mean(),
            "Lead_Time_FA05_Std": grp["Lead_Time_at_FA_0.5"].std(),
        }
        if ptype == "In_Domain":
            in_domain_benchmarks[tgt] = row
        summary_rows.append(row)

    # Compute drop vs in-domain
    for r in summary_rows:
        tgt = r["Target_Domain"]
        bench = in_domain_benchmarks[tgt]
        r["Delta_RMSE_vs_InDomain"] = r["RUL_RMSE_Mean"] - bench["RUL_RMSE_Mean"]
        r["Delta_RMSE_clip_vs_InDomain"] = r["RUL_RMSE_clip_Mean"] - bench["RUL_RMSE_clip_Mean"]
        r["Delta_NASA_vs_InDomain"] = r["RUL_NASA_Mean"] - bench["RUL_NASA_Mean"]
        r["Delta_F1_vs_InDomain"] = bench["Anomaly_F1_Mean"] - r["Anomaly_F1_Mean"]
        r["Delta_PRAUC_vs_InDomain"] = bench["Anomaly_PRAUC_Mean"] - r["Anomaly_PRAUC_Mean"]
        r["Delta_LeadTime_vs_InDomain"] = bench["Lead_Time_FA05_Mean"] - r["Lead_Time_FA05_Mean"]

    summary_df = pd.DataFrame(summary_rows)
    out_csv = results_dir / "cross_condition_transfer.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"[Saved] Cross-condition generalization results -> {out_csv}")
    return summary_df


# ---------------------------------------------------------------------------
# 2. Split Conformal Prediction Intervals for RUL
# ---------------------------------------------------------------------------

def run_conformal_prediction_experiments(
    raw_dir: Path,
    results_dir: Path,
    figures_dir: Path,
    target_coverage: float = 0.90,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split Conformal Prediction for RUL:
      - Calibrated on whole held-out TRAIN engines (never test engines).
      - Calibration fraction = 20% of train units.
      - Evaluates empirical coverage and interval width at 90% target coverage on test engines.
      - Plots predicted RUL with 90% confidence bands for 6 test engines.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2 - EXPERIMENT 2: Split Conformal Prediction Intervals")
    logger.info("=" * 60)

    subsets = [1, 2, 3, 4]
    conformal_results = []
    models_dict = {}
    quantiles_dict = {}
    test_data_dict = {}

    for s in subsets:
        tag = f"FD00{s}"
        train_raw, test_raw, test_rul = load_raw(raw_dir, s)
        features = select_features(train_raw, 0.001)

        # Condition-aware normalizer fit on train
        norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=seed)
        norm.fit(train_raw, features, subset_id=s)
        train_norm = norm.transform(train_raw)
        test_norm  = norm.transform(test_raw)

        train_norm = add_rul_train(train_norm, rul_clip=125)

        # Split TRAIN engines into Proper Train (80%) and Calibration (20%)
        train_units = sorted(train_norm["unit"].unique())
        units_tr, units_cal = train_test_split(train_units, test_size=0.20, random_state=seed)

        df_proper_tr = train_norm[train_norm["unit"].isin(units_tr)]
        df_cal = train_norm[train_norm["unit"].isin(units_cal)]

        # Windows
        X_tr, y_tr, _ = make_train_windows(df_proper_tr, features, window_length=30, stride=1)
        X_cal, y_cal, _ = make_train_windows(df_cal, features, window_length=30, stride=1)
        X_te, y_te = make_test_windows(test_norm, features, test_rul, window_length=30)

        X_tr_f = extract_window_features(X_tr)
        X_cal_f = extract_window_features(X_cal)
        X_te_f = extract_window_features(X_te)

        # Train model
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr_f, y_tr)

        # Compute calibration non-conformity scores: |y - y_hat|
        cal_preds = np.clip(model.predict(X_cal_f), 0.0, 125.0)
        cal_scores = np.abs(y_cal - cal_preds)

        # Compute conformal quantile at 1 - alpha = target_coverage
        n_cal = len(cal_scores)
        q_level = min(1.0, np.ceil((n_cal + 1) * target_coverage) / n_cal)
        q = float(np.quantile(cal_scores, q_level, method="higher"))

        # Evaluate on test engines
        test_preds = np.clip(model.predict(X_te_f), 0.0, 125.0)
        lower_bounds = np.maximum(0.0, test_preds - q)
        upper_bounds = np.minimum(125.0, test_preds + q)

        # Check coverage
        in_interval = (y_te >= test_preds - q) & (y_te <= test_preds + q)
        emp_coverage = float(np.mean(in_interval))
        mean_width_nominal = float(2.0 * q)
        mean_width_clipped = float(np.mean(upper_bounds - lower_bounds))

        logger.info(
            f"[{tag}] Conformal q={q:.2f} | Nominal Width={mean_width_nominal:.2f} | "
            f"Empirical Coverage={emp_coverage*100:.1f}% (Target: {target_coverage*100:.1f}%) "
            f"over {len(y_te)} test engines."
        )

        conformal_results.append({
            "Subset": tag,
            "Model": "XGBoost",
            "Target_Coverage": target_coverage,
            "Calibration_Units_Count": len(units_cal),
            "Calibration_Windows_Count": n_cal,
            "Test_Engines_Count": len(y_te),
            "Conformal_Quantile_q": q,
            "Empirical_Coverage_Test": emp_coverage,
            "Mean_Interval_Width_Nominal": mean_width_nominal,
            "Mean_Interval_Width_Clipped": mean_width_clipped,
        })

        models_dict[s] = model
        quantiles_dict[s] = q
        test_data_dict[s] = (test_norm, features, test_rul)

    df_conformal = pd.DataFrame(conformal_results)
    out_csv = results_dir / "conformal_prediction_results.csv"
    df_conformal.to_csv(out_csv, index=False)
    logger.info(f"[Saved] Conformal prediction metrics -> {out_csv}")

    # Plot 6 representative test engines
    _plot_conformal_six_engines(
        subsets_dict=test_data_dict,
        models_dict=models_dict,
        quantiles_dict=quantiles_dict,
        out_fig=figures_dir / "conformal_prediction_intervals.png",
    )

    return df_conformal


def _plot_conformal_six_engines(
    subsets_dict: dict[int, tuple[pd.DataFrame, list[str], np.ndarray]],
    models_dict: dict[int, Any],
    quantiles_dict: dict[int, float],
    out_fig: Path,
) -> None:
    """
    Plots predicted RUL with 90% confidence bands for 6 test engines.
    Engines: FD001 unit 31 & 68, FD002 unit 45, FD003 unit 15, FD004 unit 10 & 55.
    """
    fig_engines = [
        (1, 31),
        (1, 68),
        (2, 45),
        (3, 15),
        (4, 11),
        (4, 55),
    ]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    axes = axes.flatten()

    for idx, (sub_id, unit_id) in enumerate(fig_engines):
        ax = axes[idx]
        test_norm, features, test_rul = subsets_dict[sub_id]
        model = models_dict[sub_id]
        q = quantiles_dict[sub_id]

        sorted_units = sorted(test_norm["unit"].unique())
        unit_idx = sorted_units.index(unit_id)
        true_final_rul = float(test_rul[unit_idx])

        grp = test_norm[test_norm["unit"] == unit_id].sort_values("cycle").reset_index(drop=True)
        feats = grp[features].values.astype(np.float32)
        n_c = len(grp)
        fail_c = n_c + true_final_rul

        # Trajectory windows
        cycles_list = []
        preds_list = []
        true_rul_list = []

        window_len = 30
        w_feats = []

        if n_c < window_len:
            pad = np.repeat(feats[:1], window_len - n_c, axis=0)
            feats_padded = np.vstack([pad, feats])
            w_feats.append(feats_padded)
            cycles_list.append(int(grp.loc[n_c - 1, "cycle"]))
            true_rul_list.append(fail_c - int(grp.loc[n_c - 1, "cycle"]))
        else:
            for start in range(0, n_c - window_len + 1):
                end = start + window_len
                w_feats.append(feats[start:end])
                c_val = int(grp.loc[end - 1, "cycle"])
                cycles_list.append(c_val)
                true_rul_list.append(fail_c - c_val)

        X_w = np.array(w_feats, dtype=np.float32)
        X_wf = extract_window_features(X_w)
        preds_list = np.clip(model.predict(X_wf), 0.0, 125.0)

        c_arr = np.array(cycles_list)
        p_arr = np.array(preds_list)
        t_arr = np.array(true_rul_list)

        # Plot
        ax.plot(c_arr, t_arr, label="True RUL", color="#1f77b4", linestyle="--", linewidth=2.2)
        ax.plot(c_arr, p_arr, label="Predicted RUL (XGBoost)", color="#d62728", linewidth=2.0)
        ax.fill_between(
            c_arr,
            np.maximum(0.0, p_arr - q),
            np.minimum(125.0, p_arr + q),
            color="#d62728",
            alpha=0.20,
            label=f"90% Conformal Band (±{q:.1f})",
        )

        ax.scatter([c_arr[-1]], [true_final_rul], color="#1f77b4", s=60, zorder=5, label=f"True End RUL={true_final_rul:.0f}")

        ax.set_title(f"FD00{sub_id} — Unit {unit_id} Trajectory", fontsize=12, fontweight="bold")
        ax.set_xlabel("Operational Cycle", fontsize=10)
        ax.set_ylabel("Remaining Useful Life (Cycles)", fontsize=10)
        ax.set_ylim(-5, 140)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Split Conformal Prediction Intervals (90% Target Coverage) on Unseen Test Engines", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    logger.info(f"[Saved Figure] Conformal intervals plot -> {out_fig}")


# ---------------------------------------------------------------------------
# 3. Explainability with SHAP & Sensor Overlap
# ---------------------------------------------------------------------------

def run_shap_explainability_experiments(
    raw_dir: Path,
    results_dir: Path,
    figures_dir: Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    TreeSHAP Explainability on Best Tree Model per Subset:
      - Computes feature importances across all 4 subsets.
      - Aggregates importance to sensor level.
      - Compares top 10 sensors with sensors driving the static baseline detector.
      - Produces overlap table and publication plot.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2 - EXPERIMENT 3: SHAP Explainability & Sensor Overlap")
    logger.info("=" * 60)

    subsets = [1, 2, 3, 4]
    feature_imp_records = []
    overlap_records = []
    shap_summary_dict = {}

    for s in subsets:
        tag = f"FD00{s}"
        train_raw, test_raw, _ = load_raw(raw_dir, s)
        features = select_features(train_raw, 0.001)

        norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=seed)
        norm.fit(train_raw, features, subset_id=s)
        train_norm = norm.transform(train_raw)

        train_norm = add_rul_train(train_norm, rul_clip=125)
        X_tr, y_tr, _ = make_train_windows(train_norm, features, window_length=30, stride=2)
        X_tr_f = extract_window_features(X_tr)

        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr_f, y_tr)

        # Background sample for TreeSHAP
        rng = np.random.RandomState(seed)
        sample_indices = rng.choice(len(X_tr_f), size=min(600, len(X_tr_f)), replace=False)
        X_sample = X_tr_f[sample_indices]

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        # Aggregate 4 feature types (mean, std, slope, last) per sensor
        # X_tr_f has columns: [means (len(features)), stds (len), slopes (len), lasts (len)]
        n_sensors = len(features)
        sensor_shap = {}
        for idx, s_name in enumerate(features):
            val = (
                mean_abs_shap[idx]
                + mean_abs_shap[idx + n_sensors]
                + mean_abs_shap[idx + 2 * n_sensors]
                + mean_abs_shap[idx + 3 * n_sensors]
            )
            sensor_shap[s_name] = float(val)

        # Rank sensors by SHAP
        sorted_sensors = sorted(sensor_shap.items(), key=lambda x: x[1], reverse=True)
        top10_shap = [s_name for s_name, _ in sorted_sensors[:10]]

        for rank, (s_name, val) in enumerate(sorted_sensors, start=1):
            feature_imp_records.append({
                "Subset": tag,
                "Sensor": s_name,
                "SHAP_Mean_Abs": val,
                "SHAP_Rank": rank,
            })

        shap_summary_dict[tag] = sorted_sensors[:10]

        # Sensor activation frequency in Static Threshold baseline
        healthy_early = identify_healthy_early_cycles(train_norm, 0.30)
        means = healthy_early[features].mean()
        stds = healthy_early[features].std().replace(0, 1e-5)

        # Look at degraded cycles (RUL <= 30)
        degraded = train_norm[train_norm["RUL"] <= 30]
        z_scores = np.abs((degraded[features] - means) / stds)
        sensor_violations = (z_scores > 2.5).mean(axis=0)
        sorted_baseline = sensor_violations.sort_values(ascending=False).index.tolist()
        top10_base = sorted_baseline[:10]

        # Compute overlaps
        top3_shap = set(top10_shap[:3])
        top3_base = set(top10_base[:3])
        top5_shap = set(top10_shap[:5])
        top5_base = set(top10_base[:5])
        top10_shap_set = set(top10_shap)
        top10_base_set = set(top10_base)

        overlap_records.append({
            "Subset": tag,
            "Top10_SHAP_Sensors": ", ".join(top10_shap),
            "Top10_StaticBaseline_Sensors": ", ".join(top10_base),
            "Top3_Overlap_Count": len(top3_shap.intersection(top3_base)),
            "Top3_Overlap_Sensors": ", ".join(sorted(list(top3_shap.intersection(top3_base)))),
            "Top5_Overlap_Count": len(top5_shap.intersection(top5_base)),
            "Top5_Overlap_Sensors": ", ".join(sorted(list(top5_shap.intersection(top5_base)))),
            "Top10_Overlap_Count": len(top10_shap_set.intersection(top10_base_set)),
            "Top10_Overlap_Sensors": ", ".join(sorted(list(top10_shap_set.intersection(top10_base_set)))),
        })

    df_imp = pd.DataFrame(feature_imp_records)
    df_overlap = pd.DataFrame(overlap_records)

    out_imp = results_dir / "shap_feature_importance.csv"
    out_overlap = results_dir / "shap_sensor_overlap.csv"

    df_imp.to_csv(out_imp, index=False)
    df_overlap.to_csv(out_overlap, index=False)

    logger.info(f"[Saved] SHAP feature importance -> {out_imp}")
    logger.info(f"[Saved] SHAP vs Baseline overlap -> {out_overlap}")

    # Plot Top 10 sensors across 4 subsets
    _plot_shap_importances(shap_summary_dict, figures_dir / "shap_feature_importance.png")

    return df_imp, df_overlap


def _plot_shap_importances(
    shap_summary_dict: dict[str, list[tuple[str, float]]],
    out_fig: Path,
) -> None:
    """Plots horizontal bar charts of Top 10 sensors by SHAP value across all subsets."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (sub_tag, sensor_list) in enumerate(shap_summary_dict.items()):
        ax = axes[idx]
        sensors = [x[0] for x in sensor_list][::-1]
        values = [x[1] for x in sensor_list][::-1]

        y_pos = np.arange(len(sensors))
        ax.barh(y_pos, values, color=colors[idx], alpha=0.85, edgecolor="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sensors, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on RUL)", fontsize=10)
        ax.set_title(f"{sub_tag} — Top 10 Most Influential Sensors", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Explainability: TreeSHAP Feature Attribution across C-MAPSS Subsets", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    logger.info(f"[Saved Figure] SHAP feature importance plot -> {out_fig}")


# ---------------------------------------------------------------------------
# 4. Ablations Study (3 Seeds)
# ---------------------------------------------------------------------------

def run_ablation_experiments(
    raw_dir: Path,
    results_dir: Path,
    seeds: list[int] = [0, 1, 2],
) -> pd.DataFrame:
    """
    Ablations on FD001 over 3 seeds (0, 1, 2):
      1. Temporal Smoothing: ON vs OFF
      2. Window Length: 15, 30, 50
      3. Feature Representation: Raw Window, PCA 95% Variance, Engineered Statistics
    """
    logger.info("=" * 60)
    logger.info("STAGE 2 - EXPERIMENT 4: Ablation Studies (3 Seeds)")
    logger.info("=" * 60)

    train_raw, test_raw, test_rul = load_raw(raw_dir, 1)
    features = select_features(train_raw, 0.001)

    ablation_runs = []

    configs = [
        # Baseline Reference
        ("Baseline", "Smoothing_OFF_W30_EngineeredStats", False, 30, "Engineered_Stats"),
        # Smoothing Ablation
        ("Smoothing", "Smoothing_ON_W30_EngineeredStats", True, 30, "Engineered_Stats"),
        # Window Length Ablation
        ("Window_Length", "Smoothing_OFF_W15_EngineeredStats", False, 15, "Engineered_Stats"),
        ("Window_Length", "Smoothing_OFF_W50_EngineeredStats", False, 50, "Engineered_Stats"),
        # Feature Representation Ablation
        ("Representation", "Smoothing_OFF_W30_RawWindow", False, 30, "Raw_Window"),
        ("Representation", "Smoothing_OFF_W30_PCA95", False, 30, "PCA_95"),
    ]

    for cat, name, use_smoothing, win_len, rep in configs:
        logger.info(f"Running Configuration: {name} (Category: {cat})")

        for seed in seeds:
            set_seed(seed)

            # 1. Preprocessing
            df_tr = train_raw.copy()
            df_te = test_raw.copy()

            if use_smoothing:
                # EWM smoothing per engine
                smoothed_tr = []
                for uid, grp in df_tr.groupby("unit"):
                    grp_s = grp.sort_values("cycle").copy()
                    grp_s[features] = grp_s[features].ewm(alpha=0.2, adjust=False).mean()
                    smoothed_tr.append(grp_s)
                df_tr = pd.concat(smoothed_tr, ignore_index=True)

                smoothed_te = []
                for uid, grp in df_te.groupby("unit"):
                    grp_s = grp.sort_values("cycle").copy()
                    grp_s[features] = grp_s[features].ewm(alpha=0.2, adjust=False).mean()
                    smoothed_te.append(grp_s)
                df_te = pd.concat(smoothed_te, ignore_index=True)

            norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=seed)
            norm.fit(df_tr, features, subset_id=1)
            df_tr_norm = norm.transform(df_tr)
            df_te_norm = norm.transform(df_te)

            df_tr_rul = add_rul_train(df_tr_norm, rul_clip=125)

            # 2. Windowing
            X_tr_rul, y_tr_rul, _ = make_train_windows(df_tr_rul, features, window_length=win_len, stride=1)
            X_te_rul, y_te_rul = make_test_windows(df_te_norm, features, test_rul, window_length=win_len)

            # 3. Representation
            if rep == "Engineered_Stats":
                X_tr_f = extract_window_features(X_tr_rul)
                X_te_f = extract_window_features(X_te_rul)
            elif rep == "Raw_Window":
                X_tr_f = X_tr_rul.reshape(X_tr_rul.shape[0], -1)
                X_te_f = X_te_rul.reshape(X_te_rul.shape[0], -1)
            elif rep == "PCA_95":
                X_tr_flat = X_tr_rul.reshape(X_tr_rul.shape[0], -1)
                X_te_flat = X_te_rul.reshape(X_te_rul.shape[0], -1)
                pca = PCA(n_components=0.95, svd_solver="full", random_state=seed)
                X_tr_f = pca.fit_transform(X_tr_flat)
                X_te_f = pca.transform(X_te_flat)
            else:
                raise ValueError(f"Unknown representation {rep}")

            # 4. Train RUL Regressor
            xgb_r = xgb.XGBRegressor(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                tree_method="hist",
                n_jobs=-1,
                verbosity=0,
            )
            xgb_r.fit(X_tr_f, y_tr_rul)
            preds_r = np.clip(xgb_r.predict(X_te_f), 0.0, 125.0)
            m_rul = evaluate_all(y_te_rul, preds_r, rul_clip=125)

            # 5. Train Anomaly Classifier
            X_tr_ano_raw, y_tr_ano, _, _, _ = _build_windowed_anomaly_data(
                df_tr_norm, features, window_length=win_len, n_threshold=30, is_test=False
            )
            X_te_ano_raw, y_te_ano, _, _, _ = _build_windowed_anomaly_data(
                df_te_norm, features, window_length=win_len, n_threshold=30, is_test=True, test_rul_arr=test_rul
            )

            if rep == "Engineered_Stats":
                X_tr_ano_f = extract_window_features(X_tr_ano_raw)
                X_te_ano_f = extract_window_features(X_te_ano_raw)
            elif rep == "Raw_Window":
                X_tr_ano_f = X_tr_ano_raw.reshape(X_tr_ano_raw.shape[0], -1)
                X_te_ano_f = X_te_ano_raw.reshape(X_te_ano_raw.shape[0], -1)
            elif rep == "PCA_95":
                X_tr_ano_flat = X_tr_ano_raw.reshape(X_tr_ano_raw.shape[0], -1)
                X_te_ano_flat = X_te_ano_raw.reshape(X_te_ano_raw.shape[0], -1)
                pca_ano = PCA(n_components=0.95, svd_solver="full", random_state=seed)
                X_tr_ano_f = pca_ano.fit_transform(X_tr_ano_flat)
                X_te_ano_f = pca_ano.transform(X_te_ano_flat)

            pos = float(np.sum(y_tr_ano == 1))
            neg = float(np.sum(y_tr_ano == 0))
            scale_pos = (neg / pos) if pos > 0 else 1.0

            xgb_a = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )
            xgb_a.fit(X_tr_ano_f, y_tr_ano)
            ano_probs = xgb_a.predict_proba(X_te_ano_f)[:, 1]

            p_val, r_val, f1_val, _ = precision_recall_fscore_support(
                y_te_ano, (ano_probs >= 0.5).astype(int), average="binary", zero_division=0
            )
            pr_auc_val = float(average_precision_score(y_te_ano, ano_probs)) if len(np.unique(y_te_ano)) > 1 else 0.0

            ablation_runs.append({
                "Category": cat,
                "Config_Name": name,
                "Smoothing": "ON" if use_smoothing else "OFF",
                "Window_Length": win_len,
                "Representation": rep,
                "Seed": seed,
                "RMSE": m_rul["RMSE"],
                "RMSE_clip": m_rul["RMSE_clip"],
                "MAE": m_rul["MAE"],
                "Anomaly_F1": float(f1_val),
                "Anomaly_PRAUC": float(pr_auc_val),
            })

    df_runs = pd.DataFrame(ablation_runs)

    # Compute summary table (Mean +/- Std)
    summary_rows = []
    for (cat, name), grp in df_runs.groupby(["Category", "Config_Name"]):
        summary_rows.append({
            "Category": cat,
            "Configuration": name,
            "Smoothing": grp["Smoothing"].iloc[0],
            "Window_Length": grp["Window_Length"].iloc[0],
            "Representation": grp["Representation"].iloc[0],
            "RUL_RMSE_Mean": grp["RMSE"].mean(),
            "RUL_RMSE_Std": grp["RMSE"].std(),
            "RUL_RMSE_clip_Mean": grp["RMSE_clip"].mean(),
            "RUL_RMSE_clip_Std": grp["RMSE_clip"].std(),
            "RUL_MAE_Mean": grp["MAE"].mean(),
            "RUL_MAE_Std": grp["MAE"].std(),
            "Anomaly_F1_Mean": grp["Anomaly_F1"].mean(),
            "Anomaly_F1_Std": grp["Anomaly_F1"].std(),
            "Anomaly_PRAUC_Mean": grp["Anomaly_PRAUC"].mean(),
            "Anomaly_PRAUC_Std": grp["Anomaly_PRAUC"].std(),
        })

    df_summary = pd.DataFrame(summary_rows)
    out_csv = results_dir / "ablations_summary.csv"
    df_summary.to_csv(out_csv, index=False)
    logger.info(f"[Saved] Ablation experiments summary -> {out_csv}")
    return df_summary


# ---------------------------------------------------------------------------
# 5. Manifest Synchronizer
# ---------------------------------------------------------------------------

def update_stage2_manifest(results_dir: Path) -> Path:
    """
    Appends all Stage 2 metrics to results/manifest.csv.
    """
    manifest_csv = results_dir / "manifest.csv"
    if manifest_csv.exists():
        records = pd.read_csv(manifest_csv).to_dict("records")
    else:
        records = []

    # 1. Cross-condition transfer metrics
    cc_csv = results_dir / "cross_condition_transfer.csv"
    if cc_csv.exists():
        df_cc = pd.read_csv(cc_csv)
        for _, r in df_cc.iterrows():
            pair = r["Pair"]
            records.append({
                "Metric_Category": "Cross_Condition_Generalization",
                "Subset": pair,
                "Model": "XGBoost",
                "Metric_Name": "RUL_RMSE_Mean",
                "Value": f"{r['RUL_RMSE_Mean']:.2f} +/- {r['RUL_RMSE_Std']:.2f}",
                "Source_File": "results/cross_condition_transfer.csv",
                "Source_Column": "RUL_RMSE_Mean",
            })
            records.append({
                "Metric_Category": "Cross_Condition_Generalization",
                "Subset": pair,
                "Model": "XGBoost",
                "Metric_Name": "Anomaly_F1_Mean",
                "Value": f"{r['Anomaly_F1_Mean']:.3f} +/- {r['Anomaly_F1_Std']:.3f}",
                "Source_File": "results/cross_condition_transfer.csv",
                "Source_Column": "Anomaly_F1_Mean",
            })
            records.append({
                "Metric_Category": "Cross_Condition_Generalization",
                "Subset": pair,
                "Model": "XGBoost",
                "Metric_Name": "Delta_RMSE_vs_InDomain",
                "Value": f"{r['Delta_RMSE_vs_InDomain']:.2f}",
                "Source_File": "results/cross_condition_transfer.csv",
                "Source_Column": "Delta_RMSE_vs_InDomain",
            })

    # 2. Conformal prediction metrics
    cp_csv = results_dir / "conformal_prediction_results.csv"
    if cp_csv.exists():
        df_cp = pd.read_csv(cp_csv)
        for _, r in df_cp.iterrows():
            sub = r["Subset"]
            records.append({
                "Metric_Category": "Conformal_Prediction_90",
                "Subset": sub,
                "Model": r["Model"],
                "Metric_Name": "Empirical_Coverage_Test",
                "Value": f"{r['Empirical_Coverage_Test']*100:.1f}%",
                "Source_File": "results/conformal_prediction_results.csv",
                "Source_Column": "Empirical_Coverage_Test",
            })
            records.append({
                "Metric_Category": "Conformal_Prediction_90",
                "Subset": sub,
                "Model": r["Model"],
                "Metric_Name": "Mean_Interval_Width_Nominal",
                "Value": f"{r['Mean_Interval_Width_Nominal']:.2f}",
                "Source_File": "results/conformal_prediction_results.csv",
                "Source_Column": "Mean_Interval_Width_Nominal",
            })

    # 3. Ablations metrics
    ab_csv = results_dir / "ablations_summary.csv"
    if ab_csv.exists():
        df_ab = pd.read_csv(ab_csv)
        for _, r in df_ab.iterrows():
            cfg_name = r["Configuration"]
            records.append({
                "Metric_Category": "Ablation_Study",
                "Subset": "FD001",
                "Model": cfg_name,
                "Metric_Name": "RUL_RMSE_Mean",
                "Value": f"{r['RUL_RMSE_Mean']:.2f} +/- {r['RUL_RMSE_Std']:.2f}",
                "Source_File": "results/ablations_summary.csv",
                "Source_Column": "RUL_RMSE_Mean",
            })
            records.append({
                "Metric_Category": "Ablation_Study",
                "Subset": "FD001",
                "Model": cfg_name,
                "Metric_Name": "Anomaly_F1_Mean",
                "Value": f"{r['Anomaly_F1_Mean']:.3f} +/- {r['Anomaly_F1_Std']:.3f}",
                "Source_File": "results/ablations_summary.csv",
                "Source_Column": "Anomaly_F1_Mean",
            })

    new_manifest_df = pd.DataFrame(records).drop_duplicates(subset=["Metric_Category", "Subset", "Model", "Metric_Name"])
    new_manifest_df.to_csv(manifest_csv, index=False)
    logger.info(f"[Saved] Results manifest updated -> {manifest_csv} (Total Entries: {len(new_manifest_df)})")
    return manifest_csv
