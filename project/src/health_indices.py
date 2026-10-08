"""
health_indices.py - Comparative Health Indices: PCA-PC1 vs. Mahalanobis Distance.

Implements and evaluates:
  1. PCA-PC1 Health Index: First principal component fit on healthy early life.
  2. Mahalanobis Distance Health Index (MD-HI): Distance from the healthy cluster
     mean with regularized covariance matrix, converted to an index in [0, 1].

Both indices are fit strictly on the early healthy phase (first 30% cycles) of TRAIN engines.
Reports Spearman rank correlation with true RUL across all four subsets.
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
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from data_loader import load_raw, select_features
from regime_norm import OperatingRegimeNormalizer, identify_healthy_early_cycles
from utils import get_logger, load_config, set_seed

logger = get_logger("health_indices")


class DualHealthIndex:
    """
    Fits both PCA-PC1 and Mahalanobis Distance Health Index on the healthy cluster.
    """

    def __init__(self, healthy_life_fraction: float = 0.30):
        self.healthy_life_fraction = healthy_life_fraction
        self.pca = PCA(n_components=1)
        self.mu_healthy: np.ndarray | None = None
        self.inv_cov_healthy: np.ndarray | None = None
        self.is_pca_inverted: bool = False
        self.feature_cols: list[str] = []
        self.pca_p1: float = 0.0
        self.pca_p99: float = 1.0
        self.md_p99: float = 1.0

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "DualHealthIndex":
        self.feature_cols = feature_cols
        healthy_df = identify_healthy_early_cycles(train_df, self.healthy_life_fraction)
        X_h = healthy_df[feature_cols].values

        # 1. Fit PCA
        self.pca.fit(X_h)
        all_X = train_df[feature_cols].values
        pca_proj = self.pca.transform(all_X).ravel()

        # Orientation check: early vs late
        max_cyc = train_df.groupby("unit")["cycle"].transform("max")
        early_mask = (train_df["cycle"] <= np.round(self.healthy_life_fraction * max_cyc)).values
        late_mask  = (train_df["cycle"] >= (max_cyc - 10)).values

        if pca_proj[early_mask].mean() < pca_proj[late_mask].mean():
            self.is_pca_inverted = True

        projs = -pca_proj if self.is_pca_inverted else pca_proj
        self.pca_p1  = float(np.percentile(projs, 1))
        self.pca_p99 = float(np.percentile(projs, 99))

        # 2. Fit Mahalanobis parameters
        self.mu_healthy = np.mean(X_h, axis=0)
        cov = np.cov(X_h, rowvar=False)
        # Regularize covariance to ensure well-conditioned inverse
        cov_reg = cov + np.eye(len(feature_cols)) * 1e-4
        self.inv_cov_healthy = np.linalg.pinv(cov_reg)

        # Precompute 99th percentile of MD on training set for scaling
        diffs = all_X - self.mu_healthy
        mds = np.sqrt(np.sum(diffs @ self.inv_cov_healthy * diffs, axis=1))
        self.md_p99 = float(np.percentile(mds, 99))
        if self.md_p99 < 1e-4:
            self.md_p99 = 1.0

        return self

    def transform_engine(self, grp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (hi_pca, hi_md) arrays for an engine trajectory, both in [0, 1].
        """
        X = grp[self.feature_cols].values
        # PCA-HI
        p = self.pca.transform(X).ravel()
        if self.is_pca_inverted:
            p = -p
        hi_pca = (p - self.pca_p1) / (self.pca_p99 - self.pca_p1 + 1e-8)
        hi_pca = np.clip(hi_pca, 0.0, 1.0)
        hi_pca_smooth = pd.Series(hi_pca).ewm(span=5, adjust=False).mean().values

        # MD-HI: Distance increases as engine degrades -> convert to descending HI
        diffs = X - self.mu_healthy
        md = np.sqrt(np.maximum(0, np.sum(diffs @ self.inv_cov_healthy * diffs, axis=1)))
        hi_md = 1.0 - (md / self.md_p99)
        hi_md = np.clip(hi_md, 0.0, 1.0)
        hi_md_smooth = pd.Series(hi_md).ewm(span=5, adjust=False).mean().values

        return hi_pca_smooth, hi_md_smooth


def evaluate_health_indices_all_subsets(
    raw_dir: Path,
    output_dir: Path,
    subsets: list[int] = [1, 2, 3, 4],
) -> pd.DataFrame:
    """
    Fit and evaluate both health indices across all subsets. Save figures and report Spearman r.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    fig, axes = plt.subplots(len(subsets), 2, figsize=(14, 4 * len(subsets)), dpi=300)
    if len(subsets) == 1:
        axes = np.expand_dims(axes, 0)

    for idx, s in enumerate(subsets):
        train_raw, _, _ = load_raw(raw_dir, s)
        feature_cols = select_features(train_raw, 0.001)

        # Apply regime-aware normalizer
        norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=42)
        norm.fit(train_raw, feature_cols, subset_id=s)
        train_norm = norm.transform(train_raw)

        # Ground truth RUL
        max_c = train_norm.groupby("unit")["cycle"].max().rename("max_cycle")
        train_norm = train_norm.join(max_c, on="unit")
        train_norm["true_RUL"] = train_norm["max_cycle"] - train_norm["cycle"]

        # Fit Dual HI
        hi_model = DualHealthIndex(healthy_life_fraction=0.30)
        hi_model.fit(train_norm, feature_cols)

        all_hi_pca = []
        all_hi_md  = []
        all_ruls   = []

        # Evaluate over all train units
        for uid, grp in train_norm.groupby("unit"):
            grp_sorted = grp.sort_values("cycle")
            hi_pca, hi_md = hi_model.transform_engine(grp_sorted)
            all_hi_pca.extend(hi_pca)
            all_hi_md.extend(hi_md)
            all_ruls.extend(grp_sorted["true_RUL"].values)

        r_pca, _ = spearmanr(all_hi_pca, all_ruls)
        r_md, _  = spearmanr(all_hi_md, all_ruls)

        records.append({
            "Subset": f"FD00{s}",
            "Sensors_Count": len(feature_cols),
            "Spearman_PCA_PC1": round(float(r_pca), 4),
            "Spearman_Mahalanobis": round(float(r_md), 4),
            "Best_Index": "Mahalanobis" if r_md > r_pca else "PCA-PC1",
        })

        # Plot sample engine 1
        u1 = train_norm[train_norm["unit"] == 1].sort_values("cycle")
        u1_pca, u1_md = hi_model.transform_engine(u1)
        cyc = u1["cycle"].values

        ax_pca = axes[idx, 0]
        ax_pca.plot(cyc, u1_pca, color="royalblue", linewidth=2.0, label="PCA-PC1 HI")
        ax_pca.axhline(0.2, color="crimson", linestyle="--", alpha=0.8, label="Alarm Threshold (0.2)")
        ax_pca.set_title(f"FD00{s} Engine 1: PCA-PC1 Health Index (Spearman r = {r_pca:+.4f})", fontweight="bold")
        ax_pca.set_xlabel("Flight Cycles")
        ax_pca.set_ylabel("Health Index")
        ax_pca.set_ylim(-0.05, 1.05)
        ax_pca.grid(True, linestyle=":", alpha=0.6)
        ax_pca.legend(loc="lower left")

        ax_md = axes[idx, 1]
        ax_md.plot(cyc, u1_md, color="darkorange", linewidth=2.0, label="Mahalanobis Distance HI")
        ax_md.axhline(0.2, color="crimson", linestyle="--", alpha=0.8, label="Alarm Threshold (0.2)")
        ax_md.set_title(f"FD00{s} Engine 1: Mahalanobis Distance HI (Spearman r = {r_md:+.4f})", fontweight="bold")
        ax_md.set_xlabel("Flight Cycles")
        ax_md.set_ylabel("Health Index")
        ax_md.set_ylim(-0.05, 1.05)
        ax_md.grid(True, linestyle=":", alpha=0.6)
        ax_md.legend(loc="lower left")

    plt.tight_layout()
    plot_file = output_dir / "health_index_comparison_all_subsets.png"
    plt.savefig(plot_file)
    plt.close()
    logger.info("Health Index comparative plot saved -> %s", plot_file)

    df_res = pd.DataFrame(records)
    return df_res
