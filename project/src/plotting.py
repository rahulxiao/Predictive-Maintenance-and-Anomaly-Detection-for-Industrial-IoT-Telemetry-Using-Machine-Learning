"""
plotting.py - Publication-Quality Figure Generation (300 DPI).

Figures generated:
  1. Precision-Recall curves (PR-AUC) across FD001-FD004 with class prevalence baselines.
  2. Lead-Time distributions comparing traditional baselines vs ML detectors.
  3. False alarms per engine in healthy operational periods (baseline vs ML).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_pr_curves(
    pr_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, float]]],
    output_path: Path,
) -> None:
    """
    Plot Precision-Recall curves for all subsets in a 2x2 grid at 300 DPI.
    pr_data: {subset_tag: {model_name: (y_true, scores, prevalence)}}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    subsets = ["FD001", "FD002", "FD003", "FD004"]

    colors = {
        "IsolationForest": "#2ca02c",
        "RandomForest": "#1f77b4",
        "XGBoost": "#ff7f0e",
        "LSTM": "#9467bd",
    }

    for idx, sub in enumerate(subsets):
        ax = axes[idx // 2, idx % 2]
        if sub not in pr_data:
            continue

        prev = 0.0
        for m_name, (y_true, scores, p_val) in pr_data[sub].items():
            prev = p_val
            precision, recall, _ = precision_recall_curve(y_true, scores)
            pr_auc = average_precision_score(y_true, scores)
            color = colors.get(m_name, "#333333")
            ax.plot(recall, precision, label=f"{m_name} (PR-AUC = {pr_auc:.3f})",
                    color=color, linewidth=2.0)

        # Plot prevalence horizontal baseline
        ax.axhline(prev, color="gray", linestyle="--", alpha=0.8,
                   label=f"Prevalence ({prev*100:.1f}%)")

        ax.set_title(f"{sub} Precision-Recall Curves (Class Prevalence: {prev*100:.1f}%)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_xlim([0.0, 1.02])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_lead_time_distributions(
    engine_alarm_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot Lead-Time distributions across engines comparing baselines and ML detectors.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    models = ["Static_Threshold", "EWMA_Control_Limits", "IsolationForest", "RandomForest", "XGBoost", "LSTM"]

    palette = ["#7f7f7f", "#bcbd22", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    for idx, sub in enumerate(subsets):
        ax = axes[idx // 2, idx % 2]
        sub_df = engine_alarm_df[engine_alarm_df["Subset"] == sub]
        data_to_plot = []
        labels_to_plot = []
        plot_colors = []

        for m, c in zip(models, palette):
            m_df = sub_df[sub_df["Model"] == m]
            # Detected engines only (lead time > 0)
            det_lt = m_df[m_df["Detected"]]["Lead_Time"].values
            if len(det_lt) > 0:
                data_to_plot.append(det_lt)
                labels_to_plot.append(m.replace("_", "\n"))
                plot_colors.append(c)

        if data_to_plot:
            bplot = ax.boxplot(
                data_to_plot,
                tick_labels=labels_to_plot,
                patch_artist=True,
                showmeans=True,
                meanline=True,
            )
            for patch, col in zip(bplot["boxes"], plot_colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.65)

            ax.axhline(100, color="crimson", linestyle="--", alpha=0.7, label="Horizon H=100")
            ax.axhline(50, color="darkorange", linestyle=":", alpha=0.7, label="Horizon H=50")

        ax.set_title(f"{sub} Lead-Time Distribution (Detected Engines Only)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Lead Time (Cycles before Failure)", fontsize=10)
        ax.set_ylim([-5, 110])
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_false_alarms_per_engine(
    engine_alarm_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot False Alarms per Engine during Healthy Operational Phase across subsets.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    models = ["Static_Threshold", "EWMA_Control_Limits", "IsolationForest", "RandomForest", "XGBoost", "LSTM"]
    palette = ["#7f7f7f", "#bcbd22", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    for idx, sub in enumerate(subsets):
        ax = axes[idx // 2, idx % 2]
        sub_df = engine_alarm_df[engine_alarm_df["Subset"] == sub]
        data_to_plot = []
        labels_to_plot = []
        plot_colors = []

        for m, c in zip(models, palette):
            m_df = sub_df[sub_df["Model"] == m]
            fa = m_df["False_Alarm_Cycles"].values
            if len(fa) > 0:
                data_to_plot.append(fa)
                labels_to_plot.append(m.replace("_", "\n"))
                plot_colors.append(c)

        if data_to_plot:
            bplot = ax.boxplot(
                data_to_plot,
                tick_labels=labels_to_plot,
                patch_artist=True,
                showmeans=True,
                meanline=True,
            )
            for patch, col in zip(bplot["boxes"], plot_colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.65)

        ax.set_title(f"{sub} False Alarm Cycles in Healthy Phase (RUL > 100)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("False Alarm Cycles", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_lead_time_vs_fa_curves(
    curves_dict: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    output_path: Path,
) -> None:
    """
    Plot Mean Lead Time vs False Alarms per 100 Healthy Cycles operating characteristic
    curves across all subsets at 300 DPI.
    curves_dict: {subset_tag: {method_name: (fa_per_100_array, mean_lead_time_array)}}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    subsets = ["FD001", "FD002", "FD003", "FD004"]

    colors = {
        "Static_Threshold": "#7f7f7f",
        "EWMA_Control_Limits": "#bcbd22",
        "Mahalanobis_HI": "#17becf",
        "IsolationForest": "#2ca02c",
        "RandomForest": "#1f77b4",
        "XGBoost": "#ff7f0e",
        "LSTM": "#9467bd",
    }
    markers = {
        "Static_Threshold": "o",
        "EWMA_Control_Limits": "s",
        "Mahalanobis_HI": "D",
        "IsolationForest": "^",
        "RandomForest": "v",
        "XGBoost": "P",
        "LSTM": "X",
    }

    for idx, sub in enumerate(subsets):
        ax = axes[idx // 2, idx % 2]
        if sub not in curves_dict:
            continue

        for m_name, (fa_arr, lt_arr) in curves_dict[sub].items():
            if len(fa_arr) == 0:
                continue
            # Sort by FA for a clean curve
            sort_idx = np.argsort(fa_arr)
            fa_s = fa_arr[sort_idx]
            lt_s = lt_arr[sort_idx]
            color = colors.get(m_name, "#333333")
            marker = markers.get(m_name, "o")

            ax.plot(fa_s, lt_s, label=m_name.replace("_", " "),
                    color=color, marker=marker, markersize=4, linewidth=1.8, alpha=0.85)

        # Reference operating points FA = 0.1, 0.5, 1.0
        ax.axvline(0.1, color="gray", linestyle=":", alpha=0.7, label="FA = 0.1")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="FA = 0.5")
        ax.axvline(1.0, color="black", linestyle="-.", alpha=0.7, label="FA = 1.0")

        ax.set_title(f"{sub} Operating Curves: Lead Time vs False Alarms",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("False Alarms per 100 Healthy Cycles (Budget)", fontsize=10)
        ax.set_ylabel("Mean Lead Time (Cycles before Failure)", fontsize=10)
        ax.set_xlim([-0.05, 5.0])
        ax.set_ylim([0, 95])
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

