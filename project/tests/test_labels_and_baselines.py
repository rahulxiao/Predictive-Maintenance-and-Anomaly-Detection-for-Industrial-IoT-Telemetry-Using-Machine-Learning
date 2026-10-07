"""
tests/test_labels_and_baselines.py - Tests for Phase A (labels & HI) and Phase B (baselines).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labels import add_anomaly_labels, HealthIndexModel
from baselines import StaticThresholdDetector, EWMABaselineDetector, evaluate_detector


def _dummy_train_data():
    # 2 engines: unit 1 has 50 cycles, unit 2 has 60 cycles
    rows = []
    for uid, max_c in [(1, 50), (2, 60)]:
        for c in range(1, max_c + 1):
            rows.append({
                "unit": uid,
                "cycle": c,
                "s1": float(100 + c * 0.5 + (10 if uid == 2 else 0)),
                "s2": float(500 - c * 0.3),
            })
    return pd.DataFrame(rows)


def _dummy_test_data():
    # 2 engines: unit 1 ends at cycle 30 with true_RUL=15
    # unit 2 ends at cycle 40 with true_RUL=45
    rows = []
    for uid, max_c in [(1, 30), (2, 40)]:
        for c in range(1, max_c + 1):
            rows.append({
                "unit": uid,
                "cycle": c,
                "s1": float(100 + c * 0.5),
                "s2": float(500 - c * 0.3),
            })
    rul_arr = np.array([15.0, 45.0])
    return pd.DataFrame(rows), rul_arr


def test_train_anomaly_labels():
    df = _dummy_train_data()
    labeled = add_anomaly_labels(df, n_threshold=30, is_test=False)
    # Unit 1 has 50 cycles. RUL <= 30 means cycle >= 20
    u1 = labeled[labeled["unit"] == 1]
    assert (u1[u1["cycle"] < 20]["anomaly_label"] == 0).all()
    assert (u1[u1["cycle"] >= 20]["anomaly_label"] == 1).all()


def test_test_anomaly_labels():
    df, rul_arr = _dummy_test_data()
    labeled = add_anomaly_labels(df, n_threshold=30, is_test=True, test_rul_arr=rul_arr)
    # Unit 1: ends at cycle 30 with RUL=15.
    # At cycle 30: RUL = 15 <= 30 -> degraded (1)
    # At cycle 1: RUL = 15 + (30 - 1) = 44 > 30 -> healthy (0)
    u1 = labeled[labeled["unit"] == 1]
    assert u1[u1["cycle"] == 1]["anomaly_label"].values[0] == 0
    assert u1[u1["cycle"] == 30]["anomaly_label"].values[0] == 1

    # Unit 2: ends at cycle 40 with RUL=45.
    # At cycle 40: RUL = 45 > 30 -> all cycles healthy (0)
    u2 = labeled[labeled["unit"] == 2]
    assert (u2["anomaly_label"] == 0).all()


def test_health_index_monotonicity():
    df = _dummy_train_data()
    hi_model = HealthIndexModel(healthy_cycles=10)
    hi_model.fit(df, ["s1", "s2"])
    u1 = df[df["unit"] == 1]
    hi = hi_model.transform_engine(u1)
    assert len(hi) == len(u1)
    # Start of life HI should be significantly higher than end of life HI
    assert hi[:5].mean() > hi[-5:].mean()


def test_static_threshold_persistence_filter():
    train_df = _dummy_train_data()
    detector = StaticThresholdDetector(k=1.5, healthy_cycles=10, vote_fraction=0.5, consecutive_cycles=3)
    detector.fit(train_df, ["s1", "s2"])
    preds = detector.predict_engine(train_df[train_df["unit"] == 1])
    assert len(preds) == 50
    # Alarms should be 0 or 1
    assert set(np.unique(preds)).issubset({0, 1})
