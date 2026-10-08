"""
features_extractor.py - Phase C Window-Level Feature Extraction.

Extracts rich statistical features for tree-based models over sliding windows of length W:
  1. Mean per sensor: average value over the 30-cycle window
  2. Standard Deviation per sensor: volatility over the window
  3. Linear Trend Slope per sensor: rate of degradation change over the window
  4. Last Value per sensor: most recent cycle telemetry snapshot

Total feature dimensionality: 4 * n_sensors.
All operations are vectorized for maximum numerical efficiency.
"""
from __future__ import annotations

import numpy as np


def extract_window_features(X: np.ndarray) -> np.ndarray:
    """
    Extract window-level engineered features from 3D array X.

    Parameters:
        X: (N, W, D) float32 array
           N: number of windows
           W: window length (e.g. 30)
           D: number of sensor features

    Returns:
        feats: (N, 4 * D) float32 array
    """
    N, W, D = X.shape

    # 1. Mean
    f_mean = np.mean(X, axis=1)

    # 2. Standard deviation
    f_std = np.std(X, axis=1)

    # 3. Last value
    f_last = X[:, -1, :]

    # 4. Vectorized linear regression slope over window length W
    # slope = sum((t - t_bar) * (x_t - x_bar)) / sum((t - t_bar)^2)
    t = np.arange(W, dtype=np.float32)
    t_bar = (W - 1) / 2.0
    t_diff = t - t_bar
    denom = np.sum(t_diff ** 2)

    # Centered X along time axis
    X_centered = X - f_mean[:, np.newaxis, :]
    # Vectorized dot product: shape (N, D)
    f_slope = np.tensordot(X_centered, t_diff, axes=([1], [0])) / denom

    return np.hstack([f_mean, f_std, f_slope, f_last]).astype(np.float32)
