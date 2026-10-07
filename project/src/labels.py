"""
labels.py - Phase A: Anomaly label definition, sensitivity analysis, and Health Index.

Anomaly Label Definition:
  - An engine is considered degraded / anomalous when its Remaining Useful Life (RUL)
    falls below or equal to N cycles before failure:
        Label = 1 (Degraded) if RUL <= N else 0 (Healthy)
  - Default N = 30.
  - Sensitivity analysis evaluates N in {20, 30, 50}.
  - Test engines: since test runs end prior to failure, true RUL at cycle c is
    RUL(c) = true_test_RUL + (last_cycle - c).

Health Index (HI):
  - Unsupervised health indicator constructed using PCA (first principal component)
    fit on the early healthy operational phase (cycles 1..30) of training engines.
  - Scaled to [0, 1] where 1.0 represents brand new / healthy and 0.0 represents failure.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

from data_loader import load_raw, select_features, fit_scaler, apply_scaler
from utils import get_logger, load_config, set_seed


logger = get_logger("labels")


# ---------------------------------------------------------------------------
# 1. Anomaly Label Construction
# ---------------------------------------------------------------------------

def add_anomaly_labels(
    df: pd.DataFrame,
    n_threshold: int = 30,
    is_test: bool = False,
    test_rul_arr: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Assign binary anomaly label (1 = degraded, 0 = healthy) based on RUL <= n_threshold.

    For training:
        RUL = max_cycle - cycle
    For test:
        RUL(c) = test_rul_arr[unit - 1] + (last_cycle - c)
    """
    df = df.copy()
    if not is_test:
        max_cycles = df.groupby("unit")["cycle"].max().rename("max_cycle")
        df = df.join(max_cycles, on="unit")
        df["true_RUL"] = (df["max_cycle"] - df["cycle"]).astype(np.float32)
        df.drop(columns=["max_cycle"], inplace=True)
    else:
        if test_rul_arr is None:
            raise ValueError("test_rul_arr is required to label test data.")
        sorted_units = sorted(df["unit"].unique())
        unit_to_rul = {uid: float(test_rul_arr[i]) for i, uid in enumerate(sorted_units)}
        last_cycles = df.groupby("unit")["cycle"].max().rename("last_cycle")
        df = df.join(last_cycles, on="unit")
        df["final_test_RUL"] = df["unit"].map(unit_to_rul)
        df["true_RUL"] = (df["final_test_RUL"] + (df["last_cycle"] - df["cycle"])).astype(np.float32)
        df.drop(columns=["last_cycle", "final_test_RUL"], inplace=True)

    df["anomaly_label"] = (df["true_RUL"] <= n_threshold).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Label Statistics & Sensitivity Analysis
# ---------------------------------------------------------------------------

def compute_label_statistics(
    raw_dir: Path,
    subsets: list[int] = [1, 2, 3, 4],
    thresholds: list[int] = [20, 30, 50],
) -> pd.DataFrame:
    """
    Compute distribution of healthy vs degraded cycles across subsets and thresholds.
    """
    records = []
    for s in subsets:
        train_raw, test_raw, test_rul = load_raw(raw_dir, s)
        n_train_engines = train_raw["unit"].nunique()
        n_test_engines  = test_raw["unit"].nunique()

        for n in thresholds:
            tr_labeled = add_anomaly_labels(train_raw, n_threshold=n, is_test=False)
            te_labeled = add_anomaly_labels(test_raw,  n_threshold=n, is_test=True, test_rul_arr=test_rul)

            for split_name, df_lab, n_eng in [("Train", tr_labeled, n_train_engines),
                                               ("Test",  te_labeled, n_test_engines)]:
                tot = len(df_lab)
                deg = int(df_lab["anomaly_label"].sum())
                hea = tot - deg
                pct_deg = (deg / tot) * 100.0 if tot > 0 else 0.0
                ratio = hea / deg if deg > 0 else float("inf")

                records.append({
                    "Subset": f"FD00{s}",
                    "Threshold_N": n,
                    "Split": split_name,
                    "Engines": n_eng,
                    "Total_Cycles": tot,
                    "Healthy_Cycles": hea,
                    "Degraded_Cycles": deg,
                    "Degraded_Pct": round(pct_deg, 2),
                    "Healthy_to_Degraded_Ratio": round(ratio, 2),
                })

    return pd.DataFrame(records)


def print_label_statistics_table(stats_df: pd.DataFrame) -> None:
    """Print beautifully formatted sensitivity and label statistics tables."""
    print("\n" + "=" * 90)
    print("  PHASE A: ANOMALY LABEL STATISTICS & SENSITIVITY ANALYSIS (N in {20, 30, 50})")
    print("=" * 90)
    print(f"  {'Subset':<7} {'N':<4} {'Split':<6} {'Engines':>8} {'Total':>8} "
          f"{'Healthy':>8} {'Degraded':>9} {'% Degraded':>11} {'Ratio (H:D)':>12}")
    print("-" * 90)
    for _, r in stats_df.iterrows():
        print(f"  {r['Subset']:<7} {r['Threshold_N']:<4} {r['Split']:<6} {r['Engines']:>8} "
              f"{r['Total_Cycles']:>8} {r['Healthy_Cycles']:>8} {r['Degraded_Cycles']:>9} "
              f"{r['Degraded_Pct']:>10.2f}% {r['Healthy_to_Degraded_Ratio']:>11.1f}:1")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# 3. Health Index Construction
# ---------------------------------------------------------------------------

class HealthIndexModel:
    """
    Unsupervised Health Index (HI) using PCA fit on healthy early operating cycles.
    """

    def __init__(self, healthy_cycles: int = 30):
        self.healthy_cycles = healthy_cycles
        self.pca = PCA(n_components=1)
        self.feature_cols: list[str] = []
        self.healthy_mean: float = 1.0
        self.failure_mean: float = 0.0
        self.is_inverted: bool = False

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "HealthIndexModel":
        """
        Fit PCA on healthy cycles (cycle <= healthy_cycles) across all training units.
        """
        self.feature_cols = feature_cols
        # Healthy early cycles
        healthy_df = train_df[train_df["cycle"] <= self.healthy_cycles]
        X_healthy = healthy_df[feature_cols].values

        # Fit PCA on healthy baseline
        self.pca.fit(X_healthy)

        # Determine orientation: healthy early life should have HI near 1.0,
        # and end of life (last 10 cycles before failure) should have HI near 0.0
        train_proj = self.pca.transform(train_df[feature_cols].values).ravel()
        train_df_tmp = train_df.copy()
        train_df_tmp["proj"] = train_proj

        early_proj = train_df_tmp[train_df_tmp["cycle"] <= self.healthy_cycles]["proj"].mean()
        # Find late cycles (last 10 cycles of each unit)
        max_cyc = train_df_tmp.groupby("unit")["cycle"].transform("max")
        late_proj = train_df_tmp[train_df_tmp["cycle"] >= (max_cyc - 10)]["proj"].mean()

        # We want HI to decrease with age
        if early_proj < late_proj:
            self.is_inverted = True

        # Transform and compute min/max for normalization to [0, 1]
        projs = -train_proj if self.is_inverted else train_proj
        self.proj_min = float(np.percentile(projs, 1))
        self.proj_max = float(np.percentile(projs, 99))
        return self

    def transform_engine(self, grp: pd.DataFrame) -> np.ndarray:
        """
        Transform a single engine trajectory into a smooth Health Index in [0, 1].
        """
        X = grp[self.feature_cols].values
        proj = self.pca.transform(X).ravel()
        if self.is_inverted:
            proj = -proj
        # Min-max scale so healthy ~ 1.0, failure ~ 0.0
        hi = (proj - self.proj_min) / (self.proj_max - self.proj_min + 1e-8)
        hi = np.clip(hi, 0.0, 1.0)
        # Apply gentle causal smoothing (EWM) to reduce sensor noise
        hi_smooth = pd.Series(hi).ewm(span=5, adjust=False).mean().values
        return hi_smooth


# ---------------------------------------------------------------------------
# 4. Plot Health Index
# ---------------------------------------------------------------------------

def plot_health_index(
    raw_dir: Path,
    output_dir: Path,
    subset_id: int = 1,
    sample_units: list[int] = [1, 2, 3, 4, 5],
) -> tuple[Path, dict[str, float]]:
    """
    Generate and save high-resolution Health Index degradation plots for sample engines.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_raw, _, _ = load_raw(raw_dir, subset_id)
    feature_cols = select_features(train_raw, 0.001)

    # Scaler
    scaler = fit_scaler(train_raw, feature_cols)
    train_scaled = apply_scaler(train_raw, scaler, feature_cols)

    # Fit HI model
    hi_model = HealthIndexModel(healthy_cycles=30)
    hi_model.fit(train_scaled, feature_cols)

    # Compute correlation with true RUL
    train_labeled = add_anomaly_labels(train_scaled, n_threshold=30, is_test=False)
    all_hi = []
    all_rul = []

    for _, grp in train_labeled.groupby("unit"):
        hi = hi_model.transform_engine(grp)
        all_hi.extend(hi)
        all_rul.extend(grp["true_RUL"].values)

    corr, _ = spearmanr(all_hi, all_rul)

    # Plot 1: Cycles vs HI for FD001
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot A: Absolute Cycles
    ax = axes[0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sample_units)))
    for uid, color in zip(sample_units, colors):
        grp = train_scaled[train_scaled["unit"] == uid].sort_values("cycle")
        hi = hi_model.transform_engine(grp)
        ax.plot(grp["cycle"].values, hi, label=f"Engine {uid}", color=color, linewidth=1.8)

    ax.axhline(0.2, color="crimson", linestyle="--", linewidth=1.5, label="Degradation Threshold (HI=0.2)")
    ax.set_title(f"FD00{subset_id} Health Index vs Operational Cycles", fontsize=12, fontweight="bold")
    ax.set_xlabel("Flight Cycles", fontsize=11)
    ax.set_ylabel("Health Index (Unsupervised PCA)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)

    # Subplot B: Normalized % Lifetime
    ax = axes[1]
    for uid, color in zip(sample_units, colors):
        grp = train_scaled[train_scaled["unit"] == uid].sort_values("cycle")
        hi = hi_model.transform_engine(grp)
        norm_life = (grp["cycle"].values / grp["cycle"].max()) * 100.0
        ax.plot(norm_life, hi, label=f"Engine {uid}", color=color, linewidth=1.8)

    ax.axhline(0.2, color="crimson", linestyle="--", linewidth=1.5, label="Degradation Threshold (HI=0.2)")
    ax.set_title(f"FD00{subset_id} Health Index vs % Normalized Lifetime", fontsize=12, fontweight="bold")
    ax.set_xlabel("% Engine Lifetime Expended", fontsize=11)
    ax.set_ylabel("Health Index", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plot_path = output_dir / f"health_index_fd00{subset_id}.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info("Health Index plot saved -> %s", plot_path)

    stats = {
        "Spearman_RUL_Correlation": float(corr),
        "Initial_Health_Index_Mean": float(np.mean([all_hi[i] for i, r in enumerate(all_rul) if r > 125])),
        "Failure_Health_Index_Mean": float(np.mean([all_hi[i] for i, r in enumerate(all_rul) if r <= 5])),
    }
    return plot_path, stats


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

    figures_dir = Path(cfg["results"]["figures_dir"])
    if not figures_dir.is_absolute():
        figures_dir = project_root / figures_dir

    # 1. Label Statistics & Sensitivity Analysis
    stats_df = compute_label_statistics(
        raw_dir=raw_dir,
        subsets=cfg["data"]["subsets"],
        thresholds=cfg["anomaly"]["sensitivity_thresholds"],
    )
    print_label_statistics_table(stats_df)

    # Save CSV
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stats_csv_path = results_dir / "anomaly_label_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"[Phase A] Anomaly label statistics saved -> {stats_csv_path}")

    # 2. Health Index Plot & Statistics
    plot_path, hi_stats = plot_health_index(
        raw_dir=raw_dir,
        output_dir=figures_dir,
        subset_id=1,
        sample_units=[1, 2, 3, 4, 5],
    )

    print("\n" + "=" * 60)
    print("  PHASE A: HEALTH INDEX EVALUATION (FD001)")
    print("=" * 60)
    print(f"  Figure generated: {plot_path.name}")
  
    print(f"  Spearman correlation with ground-truth RUL: {hi_stats['Spearman_RUL_Correlation']:+.4f}")
    print(f"  Mean HI during healthy phase (RUL > 125) : {hi_stats['Initial_Health_Index_Mean']:.4f}")
    print(f"  Mean HI at failure (RUL <= 5)            : {hi_stats['Failure_Health_Index_Mean']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
