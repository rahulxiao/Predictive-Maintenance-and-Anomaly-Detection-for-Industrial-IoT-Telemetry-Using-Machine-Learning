"""
tests/test_features_extractor.py - Unit test for vectorized window features.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features_extractor import extract_window_features


def test_window_features_extraction():
    # 2 windows, length 10, 3 features
    N, W, D = 2, 10, 3
    # Synthetic linear trend for feature 0: x(t) = 2.0 * t + 5.0
    X = np.zeros((N, W, D), dtype=np.float32)
    t = np.arange(W)
    for i in range(N):
        X[i, :, 0] = 2.0 * t + 5.0  # slope should be exactly 2.0
        X[i, :, 1] = 10.0          # constant: slope should be 0.0, std = 0.0
        X[i, :, 2] = t**2          # non-linear

    feats = extract_window_features(X)
    assert feats.shape == (N, 4 * D)

    # Check feature 0 slope
    slope_idx = 2 * D + 0
    np.testing.assert_allclose(feats[:, slope_idx], 2.0, rtol=1e-5)

    # Check feature 1 std and slope
    std_idx = D + 1
    slope1_idx = 2 * D + 1
    np.testing.assert_allclose(feats[:, std_idx], 0.0, atol=1e-5)
    np.testing.assert_allclose(feats[:, slope1_idx], 0.0, atol=1e-5)

    # Check feature 0 last value
    last_idx = 3 * D + 0
    np.testing.assert_allclose(feats[:, last_idx], 2.0 * 9 + 5.0, rtol=1e-5)
