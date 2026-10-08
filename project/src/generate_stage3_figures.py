"""
generate_stage3_figures.py - Publication-Quality Figure Generation (300 DPI, Colour-Blind Safe).

Generates:
  1. sensor_degradation.png: Multi-sensor degradation trajectories over cycles (healthy, degradation, failure).
  2. n_sweep_lead_time_circularity.png: Lead time vs N sweep proving lead time circularity.
  3. predicted_vs_true_rul.png: Predicted vs True RUL scatter / parity plots across all 4 subsets.
  4. cross_condition_bars.png: In-domain vs cross-condition RUL RMSE, F1, and PR-AUC drops.
"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_raw, select_features, make_train_windows, make_test_windows, add_rul_train
from features_extractor import extract_window_features
from regime_norm import OperatingRegimeNormalizer


def generate_sensor_degradation_plot(raw_dir: Path, output_path: Path) -> None:
    """Plot multi-sensor degradation trajectory over full life for FD001 Unit 1 and FD002 Unit 1."""
    train_1, _, _ = load_raw(raw_dir, 1)
    train_2, _, _ = load_raw(raw_dir, 2)

    eng1 = train_1[train_1["unit"] == 1].sort_values("cycle").reset_index(drop=True)
    eng2 = train_2[train_2["unit"] == 1].sort_values("cycle").reset_index(drop=True)

    # Key degrading sensors
    sensors = ["s2", "s3", "s4", "s11", "s17"]
    sensor_names = {
        "s2": "LPC Outlet Temp (s2)",
        "s3": "HPC Outlet Temp (s3)",
        "s4": "LPT Outlet Temp (s4)",
        "s11": "HPC Static Press (s11)",
        "s17": "HPT Flow (s17)",
    }

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # FD001 Unit 1 (Single Condition)
    ax1 = axes[0]
    for s in sensors:
        s_vals = eng1[s].values
        # Standardize for comparison
        s_norm = (s_vals - s_vals[:30].mean()) / (s_vals[:30].std() + 1e-5)
        ax1.plot(eng1["cycle"], s_norm, label=sensor_names[s], linewidth=2.0)

    max_c1 = eng1["cycle"].max()
    ax1.axvline(max_c1 - 30, color="crimson", linestyle="--", linewidth=1.8, label="Degradation Window (RUL <= 30)")
    ax1.axvline(round(0.30 * max_c1), color="green", linestyle=":", linewidth=1.8, label="Healthy Baseline (Early 30%)")
    ax1.set_title(f"FD001 Unit 1 — Multi-Sensor Degradation Trajectory (Life: {max_c1} Cycles)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Operational Flight Cycle", fontsize=10)
    ax1.set_ylabel("Standardized Sensor Deviation (z-score)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9, frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # FD002 Unit 1 (Regime-Normalized)
    ax2 = axes[1]
    norm2 = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=42)
    norm2.fit(train_2, select_features(train_2, 0.001), subset_id=2)
    eng2_norm = norm2.transform(eng2)

    for s in sensors:
        s_vals = eng2_norm[s].values
        ax2.plot(eng2_norm["cycle"], s_vals, label=sensor_names[s], linewidth=2.0)

    max_c2 = eng2["cycle"].max()
    ax2.axvline(max_c2 - 30, color="crimson", linestyle="--", linewidth=1.8, label="Degradation Window (RUL <= 30)")
    ax2.axvline(round(0.30 * max_c2), color="green", linestyle=":", linewidth=1.8, label="Healthy Baseline (Early 30%)")
    ax2.set_title(f"FD002 Unit 1 — Regime-Normalized Degradation Trajectory (Life: {max_c2} Cycles)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Operational Flight Cycle", fontsize=10)
    ax2.set_ylabel("Regime-Standardized Deviation (z-score)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=9, frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Sensor Telemetry Degradation Dynamics (NASA C-MAPSS)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] Sensor degradation trajectories -> {output_path}")


def generate_n_sweep_plot(csv_path: Path, output_path: Path) -> None:
    """Plot Lead Time vs N sweep demonstrating circularity across all subsets."""
    df = pd.read_csv(csv_path)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    colors = {"FD001": "#1f77b4", "FD002": "#ff7f0e", "FD003": "#2ca02c", "FD004": "#d62728"}
    markers = {"FD001": "o", "FD002": "s", "FD003": "D", "FD004": "^"}

    # Subplot 1: Lead Time vs N Cutoff
    ax1 = axes[0]
    n_vals = [20, 30, 50, 75, 100]
    ax1.plot(n_vals, n_vals, color="black", linestyle="--", linewidth=2.0, label="Theoretical Bound (Lead Time = N)")

    for sub in subsets:
        s_df = df[df["Subset"] == sub].sort_values("N_Cutoff")
        ax1.plot(
            s_df["N_Cutoff"], s_df["Mean_Lead_Time"],
            label=f"{sub} (Empirical Lead Time)",
            color=colors[sub], marker=markers[sub], markersize=8, linewidth=2.2
        )

    ax1.set_title("Circularity Proof: Achieved Lead Time vs Ground-Truth Label Cutoff N", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Degraded Label Threshold N (Cycles before Failure)", fontsize=10)
    ax1.set_ylabel("Empirical Mean Lead Time (Cycles)", fontsize=10)
    ax1.set_xticks(n_vals)
    ax1.set_yticks(np.arange(0, 110, 10))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9.5, frameon=True)

    # Subplot 2: False Alarm Rate vs N Cutoff
    ax2 = axes[1]
    for sub in subsets:
        s_df = df[df["Subset"] == sub].sort_values("N_Cutoff")
        ax2.plot(
            s_df["N_Cutoff"], s_df["FA_Per_100_Healthy"],
            label=f"{sub} False Alarms",
            color=colors[sub], marker=markers[sub], markersize=8, linewidth=2.2
        )

    ax2.set_title("False Alarm Penalty: Alarms per 100 Healthy Cycles vs N Cutoff", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Degraded Label Threshold N (Cycles before Failure)", fontsize=10)
    ax2.set_ylabel("False Alarms per 100 Healthy Cycles", fontsize=10)
    ax2.set_xticks(n_vals)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9.5, frameon=True)

    plt.suptitle("Label Sensitivity Analysis: Supervised Alarms Land Near Cutoff N", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] N-sweep circularity proof -> {output_path}")


def generate_predicted_vs_true_rul_plot(raw_dir: Path, output_path: Path) -> None:
    """Generate 2x2 scatter/parity plot of Predicted RUL vs True RUL for official test engines."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()

    subsets = [1, 2, 3, 4]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, s in enumerate(subsets):
        ax = axes[idx]
        sub_tag = f"FD00{s}"

        train_raw, test_raw, test_rul = load_raw(raw_dir, s)
        features = select_features(train_raw, 0.001)

        norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=42)
        norm.fit(train_raw, features, subset_id=s)
        train_norm = norm.transform(train_raw)
        test_norm = norm.transform(test_raw)

        train_rul_df = add_rul_train(train_norm, rul_clip=125)
        X_tr, y_tr, _ = make_train_windows(train_rul_df, features, window_length=30, stride=1)
        X_te, y_te = make_test_windows(test_norm, features, test_rul, window_length=30)

        X_tr_f = extract_window_features(X_tr)
        X_te_f = extract_window_features(X_te)

        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr_f, y_tr)
        preds = np.clip(model.predict(X_te_f), 0.0, 125.0)

        rmse = np.sqrt(np.mean((preds - np.clip(y_te, 0, 125)) ** 2))
        mae = np.mean(np.abs(preds - np.clip(y_te, 0, 125)))

        # Scatter
        ax.scatter(y_te, preds, color=colors[idx], alpha=0.75, edgecolors="k", linewidth=0.5, s=45, label=f"Test Engines (N={len(y_te)})")
        # Parity line y = x
        max_val = max(140, int(y_te.max()) + 10)
        ax.plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1.8, label="Ideal Parity (y = x)")
        ax.axvline(125, color="gray", linestyle=":", alpha=0.7, label="RUL Clip Cap (125)")
        ax.axhline(125, color="gray", linestyle=":", alpha=0.7)

        ax.set_title(f"{sub_tag} RUL Regression (RMSE={rmse:.2f}, MAE={mae:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel("True Remaining Useful Life (Cycles)", fontsize=10)
        ax.set_ylabel("Predicted Remaining Useful Life (Cycles)", fontsize=10)
        ax.set_xlim([-5, max_val])
        ax.set_ylim([-5, 135])
        ax.legend(loc="upper left", fontsize=9, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Prognostic Fidelity: Predicted vs True RUL on NASA C-MAPSS Test Fleet", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] Predicted vs True RUL scatter -> {output_path}")


def generate_cross_condition_bars(csv_path: Path, output_path: Path) -> None:
    """Generate publication grouped bar chart comparing in-domain vs cross-domain metrics."""
    df = pd.read_csv(csv_path)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Pairs to plot: target FD002 and target FD004
    pairs_plot = [
        ("FD002 In-Domain", df[df["Pair"] == "FD002->FD002"].iloc[0]),
        ("FD001->FD002 Transfer", df[df["Pair"] == "FD001->FD002"].iloc[0]),
        ("FD004 In-Domain", df[df["Pair"] == "FD004->FD004"].iloc[0]),
        ("FD001->FD004 Transfer", df[df["Pair"] == "FD001->FD004"].iloc[0]),
        ("FD003->FD004 Transfer", df[df["Pair"] == "FD003->FD004"].iloc[0]),
    ]

    labels = [p[0] for p in pairs_plot]
    rmse_means = [p[1]["RUL_RMSE_Mean"] for p in pairs_plot]
    rmse_stds = [p[1]["RUL_RMSE_Std"] for p in pairs_plot]
    f1_means = [p[1]["Anomaly_F1_Mean"] for p in pairs_plot]
    f1_stds = [p[1]["Anomaly_F1_Std"] for p in pairs_plot]
    prauc_means = [p[1]["Anomaly_PRAUC_Mean"] for p in pairs_plot]
    prauc_stds = [p[1]["Anomaly_PRAUC_Std"] for p in pairs_plot]

    bar_colors = ["#2ca02c", "#d62728", "#2ca02c", "#d62728", "#d62728"]

    # Subplot 1: RUL RMSE
    ax1 = axes[0]
    y_pos = np.arange(len(labels))
    bars1 = ax1.barh(y_pos, rmse_means, xerr=rmse_stds, color=bar_colors, alpha=0.85, edgecolor="k", linewidth=0.8, capsize=4)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax1.set_xlabel("RUL RMSE (Lower is Better)", fontsize=10)
    ax1.set_title("RUL Prediction Error: In-Domain vs Cross-Condition Transfer", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    for bar, val in zip(bars1, rmse_means):
        ax1.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center", fontsize=9.5, fontweight="bold")

    # Subplot 2: Anomaly F1 & PR-AUC
    ax2 = axes[1]
    width = 0.38
    b1 = ax2.bar(y_pos - width / 2, f1_means, width, yerr=f1_stds, label="Anomaly F1 Score", color="#1f77b4", alpha=0.85, edgecolor="k", capsize=4)
    b2 = ax2.bar(y_pos + width / 2, prauc_means, width, yerr=prauc_stds, label="PR-AUC", color="#ff7f0e", alpha=0.85, edgecolor="k", capsize=4)
    ax2.set_xticks(y_pos)
    ax2.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=9, fontweight="bold")
    ax2.set_ylabel("Classification Score (Higher is Better)", fontsize=10)
    ax2.set_ylim([0, 1.05])
    ax2.set_title("Anomaly Detection Fidelity: Catastrophic Drop Under Domain Shift", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9.5, frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Cross-Condition Transferability Degradation (Normalized on Training Domain Only)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved Figure] Cross-condition transfer bars -> {output_path}")


def main() -> None:
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data/raw"
    if not raw_dir.exists():
        raw_dir = project_root.parent / "CMAPSSData"
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Stage 3 Publication Figures at 300 DPI...")
    generate_sensor_degradation_plot(raw_dir, figures_dir / "sensor_degradation.png")
    generate_n_sweep_plot(results_dir / "label_sensitivity_sweep.csv", figures_dir / "n_sweep_lead_time_circularity.png")
    generate_predicted_vs_true_rul_plot(raw_dir, figures_dir / "predicted_vs_true_rul.png")
    generate_cross_condition_bars(results_dir / "cross_condition_transfer.csv", figures_dir / "cross_condition_bars.png")
    print("All Stage 3 publication figures generated successfully.")


if __name__ == "__main__":
    main()
