"""
run_phase_cd.py - Master Pipeline for Part 1, Part 2, Phase C, and Phase D.

Executes:
  1. Part 2(e): Comparative Health Index (PCA-PC1 vs Mahalanobis Distance) across FD001-FD004.
  2. Part 2(a-d): 5-Fold GroupKFold baseline tuning, condition-aware normalization,
     and out-of-fold trajectory lead-time evaluation (H=100 & H=50).
  3. Phase C & D: ML anomaly detectors (Isolation Forest, RF, XGBoost, LSTM)
     evaluated with 5 seeds across 5-fold GroupKFold.
  4. Dual thresholding (max F1 and baseline-matched FAR).
  5. Wilcoxon signed-rank tests between best ML model and best baseline.
  6. Official test engine window-level evaluation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)

from baselines_extended import (
    ParametricStaticDetector,
    ParametricEWMADetector,
    tune_and_evaluate_baselines_subset,
)
from data_loader import load_raw, select_features
from features_extractor import extract_window_features
from health_indices import evaluate_health_indices_all_subsets
from labels import add_anomaly_labels
from metrics_lead_time import compute_sustained_alarms, evaluate_engine_lead_time, evaluate_fleet_lead_time
from ml_anomaly import MLAnomalyDetector, train_lstm_classifier, evaluate_ml_model_lead_time, predict_lstm_probs
from regime_norm import OperatingRegimeNormalizer, identify_healthy_early_cycles
from utils import get_logger, load_config, set_seed

logger = get_logger("run_cd")


def build_windowed_engine_data(
    df_norm: pd.DataFrame,
    feature_cols: list[str],
    window_length: int = 30,
    n_threshold: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """
    Construct sliding windows from normalized DataFrame while preserving engine unit and cycle info.
    """
    df_lab = add_anomaly_labels(df_norm, n_threshold=n_threshold, is_test=False)
    X_list, y_list, u_list, c_list, h_list = [], [], [], [], []
    unit_failure_cycles = {}

    for uid, grp in df_lab.groupby("unit"):
        grp_sorted = grp.sort_values("cycle").reset_index(drop=True)
        feats = grp_sorted[feature_cols].values.astype(np.float32)
        labels = grp_sorted["anomaly_label"].values.astype(np.int32)
        cycles = grp_sorted["cycle"].values.astype(np.int32)
        n = len(grp_sorted)
        unit_failure_cycles[int(uid)] = int(cycles.max())

        # Healthy early life flag: cycle <= round(0.30 * max_cycle)
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


def run_experiment_subset(
    subset_id: int,
    raw_dir: Path,
    results_dir: Path,
    seeds: list[int] = [0, 1, 2, 3, 4],
    n_splits: int = 5,
    horizon: int = 100,
) -> dict[str, Any]:
    """
    Run 5-Fold GroupKFold baseline tuning + 5-seed ML anomaly detection on one subset.
    """
    tag = f"FD00{subset_id}"
    print(f"\n{'='*95}")
    print(f"  RUNNING COMPLETE EVALUATION: SUBSET {tag}")
    print(f"{'='*95}")

    # 1. Run & Tune Baselines via 5-Fold GroupKFold
    print(f"\n[{tag}] Step 1: 5-Fold GroupKFold Baseline Tuning & Full-Trajectory Evaluation...")
    base_res = tune_and_evaluate_baselines_subset(
        subset_id=subset_id, raw_dir=raw_dir, n_splits=n_splits, horizon=horizon, random_seed=42
    )

    k_best, vf_s, p_s = base_res["best_static_param"]
    lam_best, L_best, vf_e, p_e = base_res["best_ewma_param"]
    print(f"  -> Best Static Param: k={k_best:.1f}, vote={vf_s:.1f}, pers={p_s} (Tune F1 = {base_res['best_static_f1']:.4f})")
    print(f"  -> Best EWMA Param  : lam={lam_best:.1f}, L={L_best:.1f}, vote={vf_e:.1f}, pers={p_e} (Tune F1 = {base_res['best_ewma_f1']:.4f})")

    # Best baseline FAR to match
    best_baseline_far = min(
        base_res["oof_fleet_static"]["False_Alarms_Per_100_Healthy"] / 100.0,
        base_res["oof_fleet_ewma"]["False_Alarms_Per_100_Healthy"] / 100.0,
    )
    best_baseline_far = max(0.01, min(0.15, best_baseline_far))

    # 2. Prepare Window Data for ML Models
    train_raw, test_raw, test_rul = load_raw(raw_dir, subset_id)
    feature_cols = select_features(train_raw, 0.001)

    norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=42)
    norm.fit(train_raw, feature_cols, subset_id=subset_id)
    train_norm = norm.transform(train_raw)
    test_norm  = norm.transform(test_raw)

    X_raw, y_win, u_win, c_win, h_win, fail_dict = build_windowed_engine_data(
        train_norm, feature_cols, window_length=30, n_threshold=30
    )
    X_feats = extract_window_features(X_raw)

    print(f"[{tag}] Step 2: Training & Evaluating ML Detectors (5 Seeds x 5 Folds GroupKFold)...")

    models_to_run = ["IsolationForest", "RandomForest", "XGBoost", "LSTM"]
    per_seed_records = []
    engine_alarm_records = []

    for seed in seeds:
        set_seed(seed)
        gkf = GroupKFold(n_splits=n_splits)

        for m_name in models_to_run:
            oof_scores = np.zeros(len(y_win), dtype=np.float32)

            for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_raw, groups=u_win)):
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

            # Classification metrics over entire out-of-fold predictions
            p, r, f1, _ = precision_recall_fscore_support(y_win, (oof_scores >= 0.5).astype(int), average="binary", zero_division=0)
            pr_auc = float(average_precision_score(y_win, oof_scores))
            roc_auc = float(roc_auc_score(y_win, oof_scores))
            fdr = 1.0 - p if p > 0 else 0.0

            # Dual thresholding on full OOF
            th_f1, th_far = det.tune_thresholds(oof_scores, y_win, target_baseline_far=best_baseline_far)

            # Full trajectory lead-time evaluation at max-F1 threshold
            fleet_f1, eng_res_f1 = evaluate_ml_model_lead_time(
                oof_scores, threshold=th_f1, val_units=u_win, val_cycles=c_win,
                unit_failure_cycles=fail_dict, horizon=horizon, persistence=3
            )
            # Full trajectory lead-time evaluation at matched-FAR threshold
            fleet_far, _ = evaluate_ml_model_lead_time(
                oof_scores, threshold=th_far, val_units=u_win, val_cycles=c_win,
                unit_failure_cycles=fail_dict, horizon=horizon, persistence=3
            )

            per_seed_records.append({
                "Subset": tag,
                "Model": m_name,
                "Seed": seed,
                "PR_AUC": pr_auc,
                "ROC_AUC": roc_auc,
                "Prevalence": float(np.mean(y_win)),
                "Precision": p,
                "Recall": r,
                "F1": f1,
                "FDR": fdr,
                "Threshold_MaxF1": th_f1,
                "Detection_Rate_Pct": fleet_f1["Detection_Rate_Pct"],
                "Mean_Lead_Time": fleet_f1["Mean_Lead_Time"],
                "Median_Lead_Time": fleet_f1["Median_Lead_Time"],
                "Premature_Alarm_Share_Pct": fleet_f1["Premature_Alarm_Share_Pct"],
                "False_Alarms_Per_100_Healthy": fleet_f1["False_Alarms_Per_100_Healthy"],
                "Threshold_MatchedFAR": th_far,
                "Detection_Rate_MatchedFAR": fleet_far["Detection_Rate_Pct"],
                "Mean_Lead_Time_MatchedFAR": fleet_far["Mean_Lead_Time"],
                "FA_Per_100_MatchedFAR": fleet_far["False_Alarms_Per_100_Healthy"],
            })

            # Record per-engine predictions for Wilcoxon test
            if seed == 0:
                for er in eng_res_f1:
                    engine_alarm_records.append({
                        "Subset": tag,
                        "Model": m_name,
                        "Unit": er["unit"],
                        "Detected": er["detected"],
                        "Lead_Time": er["lead_time"] if er["lead_time"] is not None else 0.0,
                        "Premature_Alarm": er["premature_alarm"],
                        "False_Alarm_Cycles": er["false_alarm_cycles_count"],
                    })

    df_seed = pd.DataFrame(per_seed_records)

    # 3. Add baseline rows to summary for direct comparison
    baseline_records = [
        {
            "Subset": tag,
            "Model": "Static_Threshold",
            "Seed": "Tuned_OOF",
            "PR_AUC": 0.0,
            "ROC_AUC": 0.0,
            "Prevalence": float(np.mean(y_win)),
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": base_res["best_static_f1"],
            "FDR": 0.0,
            "Threshold_MaxF1": k_best,
            "Detection_Rate_Pct": base_res["oof_fleet_static"]["Detection_Rate_Pct"],
            "Mean_Lead_Time": base_res["oof_fleet_static"]["Mean_Lead_Time"],
            "Median_Lead_Time": base_res["oof_fleet_static"]["Median_Lead_Time"],
            "Premature_Alarm_Share_Pct": base_res["oof_fleet_static"]["Premature_Alarm_Share_Pct"],
            "False_Alarms_Per_100_Healthy": base_res["oof_fleet_static"]["False_Alarms_Per_100_Healthy"],
            "Threshold_MatchedFAR": k_best,
            "Detection_Rate_MatchedFAR": base_res["oof_fleet_static"]["Detection_Rate_Pct"],
            "Mean_Lead_Time_MatchedFAR": base_res["oof_fleet_static"]["Mean_Lead_Time"],
            "FA_Per_100_MatchedFAR": base_res["oof_fleet_static"]["False_Alarms_Per_100_Healthy"],
        },
        {
            "Subset": tag,
            "Model": "EWMA_Control_Limits",
            "Seed": "Tuned_OOF",
            "PR_AUC": 0.0,
            "ROC_AUC": 0.0,
            "Prevalence": float(np.mean(y_win)),
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": base_res["best_ewma_f1"],
            "FDR": 0.0,
            "Threshold_MaxF1": L_best,
            "Detection_Rate_Pct": base_res["oof_fleet_ewma"]["Detection_Rate_Pct"],
            "Mean_Lead_Time": base_res["oof_fleet_ewma"]["Mean_Lead_Time"],
            "Median_Lead_Time": base_res["oof_fleet_ewma"]["Median_Lead_Time"],
            "Premature_Alarm_Share_Pct": base_res["oof_fleet_ewma"]["Premature_Alarm_Share_Pct"],
            "False_Alarms_Per_100_Healthy": base_res["oof_fleet_ewma"]["False_Alarms_Per_100_Healthy"],
            "Threshold_MatchedFAR": L_best,
            "Detection_Rate_MatchedFAR": base_res["oof_fleet_ewma"]["Detection_Rate_Pct"],
            "Mean_Lead_Time_MatchedFAR": base_res["oof_fleet_ewma"]["Mean_Lead_Time"],
            "FA_Per_100_MatchedFAR": base_res["oof_fleet_ewma"]["False_Alarms_Per_100_Healthy"],
        },
    ]

    # Save baseline engine records for seed 0 Wilcoxon
    for er in base_res["oof_static_engines"]:
        engine_alarm_records.append({
            "Subset": tag,
            "Model": "Static_Threshold",
            "Unit": er["unit"],
            "Detected": er["detected"],
            "Lead_Time": er["lead_time"] if er["lead_time"] is not None else 0.0,
            "Premature_Alarm": er["premature_alarm"],
            "False_Alarm_Cycles": er["false_alarm_cycles_count"],
        })

    # 4. Official Test Window Classification Metrics (Clearly Labelled)
    print(f"[{tag}] Step 3: Official Test Set Window-Level Evaluation...")
    # Train full models on all training data with seed 42 to evaluate on official test windows
    test_records = []
    # Test baselines
    test_records.append({
        "Subset": tag, "Model": "Static_Threshold", "Split": "Official_Test",
        "Precision": base_res["test_cls_static"]["Precision"],
        "Recall": base_res["test_cls_static"]["Recall"],
        "F1": base_res["test_cls_static"]["F1"],
        "PR_AUC": base_res["test_cls_static"]["PR_AUC"],
        "ROC_AUC": base_res["test_cls_static"]["ROC_AUC"],
        "Prevalence": base_res["test_cls_static"]["Prevalence"],
    })
    test_records.append({
        "Subset": tag, "Model": "EWMA_Control_Limits", "Split": "Official_Test",
        "Precision": base_res["test_cls_ewma"]["Precision"],
        "Recall": base_res["test_cls_ewma"]["Recall"],
        "F1": base_res["test_cls_ewma"]["F1"],
        "PR_AUC": base_res["test_cls_ewma"]["PR_AUC"],
        "ROC_AUC": base_res["test_cls_ewma"]["ROC_AUC"],
        "Prevalence": base_res["test_cls_ewma"]["Prevalence"],
    })

    # Construct test windows
    X_te_raw, y_te_win, u_te_win, c_te_win, h_te_win, _ = build_windowed_engine_data(
        test_norm, feature_cols, window_length=30, n_threshold=30
    )
    X_te_fts = extract_window_features(X_te_raw)

    for m_name in models_to_run:
        set_seed(42)
        if m_name == "LSTM":
            # Quick full train
            lstm_m, _ = train_lstm_classifier(
                X_raw, y_win, X_te_raw, y_te_win, seed=42, max_epochs=20, batch_size=256
            )
            scores_te = predict_lstm_probs(lstm_m, X_te_raw)
        else:
            det = MLAnomalyDetector(m_name, seed=42)
            det.fit(X_raw, X_feats, y_win, h_win)
            scores_te = det.predict_score(X_te_raw, X_te_fts)

        p_te, r_te, f1_te, _ = precision_recall_fscore_support(
            y_te_win, (scores_te >= 0.5).astype(int), average="binary", zero_division=0
        )
        pr_te = float(average_precision_score(y_te_win, scores_te)) if len(np.unique(y_te_win)) > 1 else 0.0
        roc_te = float(roc_auc_score(y_te_win, scores_te)) if len(np.unique(y_te_win)) > 1 else 0.5

        test_records.append({
            "Subset": tag, "Model": m_name, "Split": "Official_Test",
            "Precision": float(p_te),
            "Recall": float(r_te),
            "F1": float(f1_te),
            "PR_AUC": pr_te,
            "ROC_AUC": roc_te,
            "Prevalence": float(np.mean(y_te_win)),
        })

    return {
        "subset": tag,
        "seed_df": df_seed,
        "baseline_records": baseline_records,
        "engine_records": engine_alarm_records,
        "test_records": test_records,
    }


def main() -> None:
    set_seed(42)
    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root / "configs" / "config.yaml")

    raw_dir = Path(cfg["data"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 95)
    print("  NASA C-MAPSS ANOMALY DETECTION: PHASES A, B, C, & D FULL PIPELINE")
    print("=" * 95)

    # -----------------------------------------------------------------------
    # Part 2(e): Comparative Health Indices (PCA-PC1 vs Mahalanobis Distance)
    # -----------------------------------------------------------------------
    print("\n--- PART 2(e): HEALTH INDEX EVALUATION (PCA-PC1 vs MAHALANOBIS DISTANCE) ---")
    hi_df = evaluate_health_indices_all_subsets(
        raw_dir=raw_dir, output_dir=figures_dir, subsets=cfg["data"]["subsets"]
    )
    print(f"  {'Subset':<7} {'Sensors':<9} {'Spearman (PCA-PC1)':>20} {'Spearman (Mahalanobis)':>24} {'Best Metric':>14}")
    print("-" * 78)
    for _, r in hi_df.iterrows():
        print(f"  {r['Subset']:<7} {r['Sensors_Count']:<9} {r['Spearman_PCA_PC1']:>20.4f} "
              f"{r['Spearman_Mahalanobis']:>24.4f} {r['Best_Index']:>14}")
    print("-" * 78)
    hi_csv = results_dir / "health_index_spearman_comparison.csv"
    hi_df.to_csv(hi_csv, index=False)
    print(f"Health index comparative table saved -> {hi_csv}")

    # -----------------------------------------------------------------------
    # Part 1, 2, C, D: Multi-Seed GroupKFold Evaluation across all Subsets
    # -----------------------------------------------------------------------
    all_seed_dfs = []
    all_baseline_recs = []
    all_engine_recs = []
    all_test_recs = []

    for s in cfg["data"]["subsets"]:
        sub_res = run_experiment_subset(
            subset_id=s,
            raw_dir=raw_dir,
            results_dir=results_dir,
            seeds=[0, 1, 2, 3, 4],
            n_splits=5,
            horizon=100,
        )
        all_seed_dfs.append(sub_res["seed_df"])
        all_baseline_recs.extend(sub_res["baseline_records"])
        all_engine_recs.extend(sub_res["engine_records"])
        all_test_recs.extend(sub_res["test_records"])

    # Aggregate and Compute Mean +/- Std across Seeds
    full_seed_df = pd.concat(all_seed_dfs, ignore_index=True)
    full_seed_csv = results_dir / "anomaly_metrics_oof_seeds.csv"
    full_seed_df.to_csv(full_seed_csv, index=False)
    print(f"\n[Saved] Per-seed out-of-fold results -> {full_seed_csv}")

    # Groupby Model & Subset
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
        })

    # Add baseline rows
    for br in all_baseline_recs:
        summary_rows.append({
            "Subset": br["Subset"],
            "Model": br["Model"],
            "PR_AUC": "N/A (binary)",
            "ROC_AUC": "N/A (binary)",
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
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["Subset", "Model"])
    summary_csv = results_dir / "anomaly_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved] Consolidated summary table -> {summary_csv}")

    # Print Summary Comparison Table Out-of-Fold
    print("\n" + "=" * 110)
    print("  PHASE D: OUT-OF-FOLD COMPARISON TABLE (5 SEEDS x 5-FOLD GROUPKFOLD FULL TRAJECTORIES)")
    print("=" * 110)
    print(f"  {'Subset':<7} {'Model':<20} {'PR-AUC (Main)':<17} {'ROC-AUC':<17} {'Prev':<6} "
          f"{'F1':<16} {'Detect %':<16} {'Lead Time (H=100)':<18} {'Premature %':<14}")
    print("-" * 110)
    for _, r in summary_df.iterrows():
        print(f"  {r['Subset']:<7} {r['Model']:<20} {r['PR_AUC']:<17} {r['ROC_AUC']:<17} {r['Prevalence']:<6} "
              f"{r['F1']:<16} {r['Detection_Rate']:<16} {r['Mean_Lead_Time']:<18} {r['Premature_Alarm_Pct']:<14}")
    print("=" * 110)

    # -----------------------------------------------------------------------
    # Wilcoxon Signed-Rank Tests (Best ML Model vs Best Baseline)
    # -----------------------------------------------------------------------
    print("\n--- WILCOXON SIGNED-RANK TEST (BEST ML MODEL vs BEST BASELINE) ---")
    df_eng = pd.DataFrame(all_engine_recs)
    wilcoxon_records = []

    for sub in cfg["data"]["subsets"]:
        tag = f"FD00{sub}"
        sub_eng = df_eng[df_eng["Subset"] == tag]
        # Compare XGBoost vs Static_Threshold
        xgb_eng = sub_eng[sub_eng["Model"] == "XGBoost"].sort_values("Unit")
        sta_eng = sub_eng[sub_eng["Model"] == "Static_Threshold"].sort_values("Unit")

        if len(xgb_eng) > 0 and len(sta_eng) > 0 and len(xgb_eng) == len(sta_eng):
            diff_lead = xgb_eng["Lead_Time"].values - sta_eng["Lead_Time"].values
            diff_fa   = xgb_eng["False_Alarm_Cycles"].values - sta_eng["False_Alarm_Cycles"].values

            # Wilcoxon requires non-zero differences
            stat_l, p_l = wilcoxon(xgb_eng["Lead_Time"].values, sta_eng["Lead_Time"].values, zero_method="wilcox") if not np.all(diff_lead == 0) else (0.0, 1.0)
            stat_fa, p_fa = wilcoxon(xgb_eng["False_Alarm_Cycles"].values, sta_eng["False_Alarm_Cycles"].values, zero_method="wilcox") if not np.all(diff_fa == 0) else (0.0, 1.0)

            wilcoxon_records.append({
                "Subset": tag,
                "Comparison": "XGBoost vs Static_Threshold",
                "Lead_Time_Diff_Mean": float(np.mean(diff_lead)),
                "Lead_Time_p_value": float(p_l),
                "Lead_Time_Sig": p_l < 0.05,
                "False_Alarm_Diff_Mean": float(np.mean(diff_fa)),
                "False_Alarm_p_value": float(p_fa),
                "False_Alarm_Sig": p_fa < 0.05,
            })
            print(f"  [{tag}] XGBoost vs Static Baseline:")
            print(f"    - Lead Time Diff (mean)   : {np.mean(diff_lead):+.2f} cycles (Wilcoxon p = {p_l:.4e}) {'[STAT SIGNIFICANT]' if p_l < 0.05 else ''}")
            print(f"    - False Alarms Diff (mean): {np.mean(diff_fa):+.2f} cycles (Wilcoxon p = {p_fa:.4e}) {'[STAT SIGNIFICANT]' if p_fa < 0.05 else ''}")

    wilcoxon_csv = results_dir / "wilcoxon_tests.csv"
    pd.DataFrame(wilcoxon_records).to_csv(wilcoxon_csv, index=False)
    print(f"[Saved] Wilcoxon signed-rank tests -> {wilcoxon_csv}")

    # -----------------------------------------------------------------------
    # Official Test Engines Window-Level Evaluation Table
    # -----------------------------------------------------------------------
    test_df = pd.DataFrame(all_test_recs)
    test_csv = results_dir / "test_anomaly_metrics.csv"
    test_df.to_csv(test_csv, index=False)
    print(f"\n[Saved] Official test window-level metrics -> {test_csv}")

    print("\n" + "=" * 85)
    print("  OFFICIAL TEST ENGINES WINDOW-LEVEL CLASSIFICATION (CLEARLY LABELLED)")
    print("  (Note: Test engines are truncated prior to failure; evaluated at snapshot)")
    print("=" * 85)
    print(f"  {'Subset':<7} {'Model':<22} {'PR-AUC':>10} {'ROC-AUC':>10} {'Prevalence':>12} {'F1':>10}")
    print("-" * 85)
    for _, r in test_df.iterrows():
        pr_str = f"{r['PR_AUC']:.4f}" if isinstance(r['PR_AUC'], (float, int)) and r['PR_AUC'] > 0 else "N/A"
        roc_str = f"{r['ROC_AUC']:.4f}" if isinstance(r['ROC_AUC'], (float, int)) and r['ROC_AUC'] > 0 else "N/A"
        prev_str = f"{r['Prevalence']*100:.1f}%"
        print(f"  {r['Subset']:<7} {r['Model']:<22} {pr_str:>10} {roc_str:>10} {prev_str:>12} {r['F1']:>10.4f}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
