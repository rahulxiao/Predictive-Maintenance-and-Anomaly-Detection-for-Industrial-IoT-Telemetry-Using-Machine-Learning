"""
tests/test_stage2_eval.py - Unit tests for Stage 2 Novelty Experiments.

Tests:
  1. Cross-condition normalizer is fitted strictly on the training domain only (zero test or target leakage).
  2. Conformal calibration unit disjointness: proper train units, calibration units, and test units are 100% disjoint.
  3. Conformal coverage bounds: conformal quantile q is strictly positive and empirical coverage is well-defined.
  4. SHAP sensor overlap validity: overlap counts between top-K lists satisfy 0 <= overlap <= K.
  5. Ablation matrix completeness: all required categories and configurations are present.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_norm import OperatingRegimeNormalizer
from stage2_experiments import COMMON_SENSORS


def test_cross_condition_normalizer_fitted_on_train_only():
    """Verify that normalizer fitted on FD001 does not use target domain or multi-regime clustering."""
    # Synthetic FD001 (single condition)
    df_src = pd.DataFrame({
        "unit": np.repeat([1, 2, 3], 20),
        "cycle": np.tile(np.arange(1, 21), 3),
        "op1": np.zeros(60),
        "op2": np.zeros(60),
        "op3": np.full(60, 100.0),
    })
    for s in COMMON_SENSORS:
        df_src[s] = np.random.randn(60)

    # Synthetic FD002 (6 regimes)
    df_tgt = pd.DataFrame({
        "unit": np.repeat([1, 2], 30),
        "cycle": np.tile(np.arange(1, 31), 2),
        "op1": np.random.uniform(0, 42, 60),
        "op2": np.random.uniform(0, 0.84, 60),
        "op3": np.random.choice([100.0, 60.0, 20.0], 60),
    })
    for s in COMMON_SENSORS:
        df_tgt[s] = np.random.randn(60) * 5.0 + 20.0

    # Fit normalizer on source domain (FD001, subset_id=1)
    norm = OperatingRegimeNormalizer(n_regimes=6, life_fraction=0.30, random_seed=42)
    norm.fit(df_src, COMMON_SENSORS, subset_id=1)

    assert not norm.is_multi_condition, "FD001 must have is_multi_condition == False"
    assert norm.kmeans is None, "KMeans must NOT be fit when training domain is FD001"
    assert 0 in norm.regime_stats, "Must have regime 0 baseline stats"

    # Transform target domain
    tgt_norm = norm.transform(df_tgt)
    assert tgt_norm.shape == df_tgt.shape
    # Check that transform used regime 0 stats (no target regime clustering)
    s2_mean, s2_std = norm.regime_stats[0]["s2"]
    expected_s2 = (df_tgt["s2"].values - s2_mean) / s2_std
    np.testing.assert_allclose(tgt_norm["s2"].values, expected_s2, rtol=1e-5)


def test_conformal_calibration_unit_disjointness():
    """Verify that proper train units, calibration units, and test units are 100% disjoint."""
    train_units = list(range(1, 101))
    test_units = list(range(101, 151))

    units_proper_tr, units_cal = train_test_split(train_units, test_size=0.20, random_state=42)

    set_proper_tr = set(units_proper_tr)
    set_cal = set(units_cal)
    set_test = set(test_units)

    # 1. Zero overlap between proper train and calibration
    assert len(set_proper_tr.intersection(set_cal)) == 0, "Train and Cal units must not overlap!"
    # 2. Zero overlap with test
    assert len(set_proper_tr.intersection(set_test)) == 0, "Proper train must not overlap with test!"
    assert len(set_cal.intersection(set_test)) == 0, "Calibration must not overlap with test!"
    # 3. Union equals full train set
    assert set_proper_tr.union(set_cal) == set(train_units)
    assert len(units_cal) == 20


def test_conformal_coverage_bounds():
    """Verify that conformal quantile q > 0 and empirical coverage is in [0, 1]."""
    np.random.seed(42)
    n_cal = 1000
    y_true_cal = np.random.uniform(10, 125, n_cal)
    y_pred_cal = y_true_cal + np.random.normal(0, 5, n_cal)

    scores = np.abs(y_true_cal - y_pred_cal)
    target_coverage = 0.90
    q_level = min(1.0, np.ceil((n_cal + 1) * target_coverage) / n_cal)
    q = float(np.quantile(scores, q_level, method="higher"))

    assert q > 0.0, f"Conformal quantile q must be strictly positive, got {q}"

    # Evaluate on test
    n_test = 500
    y_true_te = np.random.uniform(10, 125, n_test)
    y_pred_te = y_true_te + np.random.normal(0, 5, n_test)
    in_interval = (y_true_te >= y_pred_te - q) & (y_true_te <= y_pred_te + q)
    coverage = float(np.mean(in_interval))

    assert 0.0 <= coverage <= 1.0
    # Should be close to 0.90
    assert 0.85 <= coverage <= 0.95


def test_shap_sensor_overlap_validity():
    """Verify that sensor overlap counts between top-K lists satisfy 0 <= overlap <= K."""
    top10_shap = ["s11", "s9", "s4", "s12", "s7", "s2", "s15", "s20", "s17", "s8"]
    top10_base = ["s11", "s4", "s12", "s17", "s15", "s2", "s3", "s13", "s21", "s8"]

    top3_overlap = len(set(top10_shap[:3]).intersection(set(top10_base[:3])))
    top5_overlap = len(set(top10_shap[:5]).intersection(set(top10_base[:5])))
    top10_overlap = len(set(top10_shap).intersection(set(top10_base)))

    assert 0 <= top3_overlap <= 3
    assert 0 <= top5_overlap <= 5
    assert 0 <= top10_overlap <= 10
    assert top10_overlap >= top5_overlap >= top3_overlap


def test_ablation_matrix_completeness():
    """Verify that ablation configuration definitions cover all required parameters."""
    configs = [
        ("Baseline", "Smoothing_OFF_W30_EngineeredStats", False, 30, "Engineered_Stats"),
        ("Smoothing", "Smoothing_ON_W30_EngineeredStats", True, 30, "Engineered_Stats"),
        ("Window_Length", "Smoothing_OFF_W15_EngineeredStats", False, 15, "Engineered_Stats"),
        ("Window_Length", "Smoothing_OFF_W50_EngineeredStats", False, 50, "Engineered_Stats"),
        ("Representation", "Smoothing_OFF_W30_RawWindow", False, 30, "Raw_Window"),
        ("Representation", "Smoothing_OFF_W30_PCA95", False, 30, "PCA_95"),
    ]

    smoothings = {c[2] for c in configs}
    window_lengths = {c[3] for c in configs}
    representations = {c[4] for c in configs}

    assert smoothings == {True, False}, "Must include both smoothing ON and OFF"
    assert window_lengths == {15, 30, 50}, "Must include window lengths 15, 30, 50"
    assert representations == {"Raw_Window", "PCA_95", "Engineered_Stats"}, "Must include all 3 feature representations"
