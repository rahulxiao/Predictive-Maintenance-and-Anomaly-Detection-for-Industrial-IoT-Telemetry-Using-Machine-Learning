"""
regime_norm.py - Operating regime clustering and condition-aware sensor normalization.

Features:
  - For multi-condition datasets (FD002, FD004):
      1. Cluster operational settings ('op1', 'op2', 'op3') into 6 regimes using KMeans.
         KMeans is fit strictly on TRAIN data.
      2. For each regime and sensor, compute healthy baseline statistics (mu, sigma)
         from the early healthy operational phase (first 30% of life of TRAIN engines).
      3. Normalize sensor readings per regime: z = (x - mu_{regime}) / sigma_{regime}.
  - For single-condition datasets (FD001, FD003):
      Regime is single (regime 0), and baseline (mu, sigma) is computed on the first 30%
      of life of TRAIN engines.
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from utils import get_logger

logger = get_logger("regime_norm")


def identify_healthy_early_cycles(train_df: pd.DataFrame, life_fraction: float = 0.30) -> pd.DataFrame:
    """
    Filter to early healthy life of TRAIN engines only:
    cycle <= round(life_fraction * max_cycle_of_engine).
    """
    max_c = train_df.groupby("unit")["cycle"].transform("max")
    cutoff = np.maximum(1, np.round(life_fraction * max_c)).astype(int)
    healthy_mask = train_df["cycle"] <= cutoff
    return train_df[healthy_mask].copy()


class OperatingRegimeNormalizer:
    """
    Cluster operating conditions (if multi-condition) and normalize sensors
    relative to healthy early-life baseline within each regime.
    """

    def __init__(self, n_regimes: int = 6, life_fraction: float = 0.30, random_seed: int = 42):
        self.n_regimes = n_regimes
        self.life_fraction = life_fraction
        self.random_seed = random_seed
        self.is_multi_condition = False
        self.kmeans: KMeans | None = None
        self.regime_stats: dict[int, dict[str, tuple[float, float]]] = {}
        self.feature_cols: list[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        subset_id: int,
    ) -> "OperatingRegimeNormalizer":
        self.feature_cols = feature_cols
        self.is_multi_condition = (subset_id in [2, 4])

        # Confirm and log healthy early life definition
        n_units = train_df["unit"].nunique()
        tot_cycles = len(train_df)
        healthy_df = identify_healthy_early_cycles(train_df, self.life_fraction)
        n_healthy_cycles = len(healthy_df)

        logger.info(
            "[FD00%d Normalizer] Healthy baseline fitted on first %.0f%% of TRAIN engines: "
            "%d / %d cycles across %d engines.",
            subset_id, self.life_fraction * 100.0, n_healthy_cycles, tot_cycles, n_units
        )

        if self.is_multi_condition:
            # Fit KMeans(k=6) on operational settings (op1, op2, op3) on TRAIN only
            op_cols = ["op1", "op2", "op3"]
            self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_seed, n_init=10)
            self.kmeans.fit(train_df[op_cols].values)

            # Assign regimes to healthy cycles
            healthy_df = healthy_df.copy()
            healthy_regimes = self.kmeans.predict(healthy_df[op_cols].values)
            healthy_df["regime"] = healthy_regimes

            # Compute per-regime mean and std for each sensor
            for r in range(self.n_regimes):
                r_df = healthy_df[healthy_df["regime"] == r]
                self.regime_stats[r] = {}
                for col in feature_cols:
                    if len(r_df) > 1:
                        m = float(r_df[col].mean())
                        s = float(r_df[col].std())
                        s = s if s > 1e-5 else 1.0
                    else:
                        m = float(healthy_df[col].mean())
                        s = float(healthy_df[col].std())
                        s = s if s > 1e-5 else 1.0
                    self.regime_stats[r][col] = (m, s)
        else:
            # Single condition: global mean and std from healthy early life
            self.regime_stats[0] = {}
            for col in feature_cols:
                m = float(healthy_df[col].mean())
                s = float(healthy_df[col].std())
                s = s if s > 1e-5 else 1.0
                self.regime_stats[0][col] = (m, s)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features in df using regime-specific healthy baseline.
        """
        out = df.copy()
        for col in self.feature_cols:
            out[col] = out[col].astype(float)
        if self.is_multi_condition and self.kmeans is not None:
            op_cols = ["op1", "op2", "op3"]
            regimes = self.kmeans.predict(out[op_cols].values)
            out["regime"] = regimes

            for r in range(self.n_regimes):
                idx = (out["regime"] == r)
                if not idx.any():
                    continue
                for col in self.feature_cols:
                    m, s = self.regime_stats[r][col]
                    out.loc[idx, col] = (out.loc[idx, col] - m) / s
        else:
            for col in self.feature_cols:
                m, s = self.regime_stats[0][col]
                out[col] = (out[col] - m) / s

        return out
