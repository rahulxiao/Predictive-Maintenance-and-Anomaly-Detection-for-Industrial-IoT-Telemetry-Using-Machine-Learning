"""
tests/test_pipeline.py - Unit tests for correctness of data pipeline.

Tests:
  1. No unit overlap between train windows and validation windows (GroupKFold).
  2. Test window count == number of test engines.
  3. Scaler fitted on train only; test set values can exceed [0,1] range.
  4. Label alignment: hand-made example, check RUL values are correct.
  5. Padding: engine shorter than window_length gets padded correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    add_rul_train,
    fit_scaler,
    apply_scaler,
    make_train_windows,
    make_test_windows,
    select_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
FEAT_COLS = ["s1", "s2", "s3"]


def _make_train_df(n_units: int = 10, max_life: int = 50,
                   seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        life = rng.integers(35, max_life)
        for cycle in range(1, life + 1):
            row = {"unit": unit, "cycle": cycle,
                   "op1": 0.0, "op2": 0.0, "op3": 0.0}
            for fi, f in enumerate(FEAT_COLS):
                row[f] = float(unit * 10 + cycle * fi)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def _make_test_df(n_units: int = 5, max_life: int = 40,
                  seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    rul_arr = []
    for unit in range(1, n_units + 1):
        life = rng.integers(20, max_life)
        for cycle in range(1, life + 1):
            row = {"unit": unit, "cycle": cycle,
                   "op1": 0.0, "op2": 0.0, "op3": 0.0}
            for fi, f in enumerate(FEAT_COLS):
                row[f] = float(unit * 10 + cycle * fi)
            rows.append(row)
        rul_arr.append(float(rng.integers(5, 100)))
    return pd.DataFrame(rows), np.array(rul_arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test 1: No unit overlap between train and val windows (GroupKFold)
# ---------------------------------------------------------------------------

def test_no_unit_overlap_in_groupkfold():
    df = _make_train_df(n_units=20, max_life=40)
    df = add_rul_train(df, rul_clip=125)
    scaler = MinMaxScaler().fit(df[FEAT_COLS])
    df_s = apply_scaler(df, scaler, FEAT_COLS)
    X, y, groups = make_train_windows(df_s, FEAT_COLS, window_length=10, stride=1)

    gkf = GroupKFold(n_splits=5)
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        train_units = set(groups[tr_idx].tolist())
        val_units   = set(groups[va_idx].tolist())
        overlap = train_units & val_units
        assert len(overlap) == 0, (
            f"Fold {fold_i}: unit overlap between train and val: {overlap}"
        )


# ---------------------------------------------------------------------------
# Test 2: Test window count == number of test engines
# ---------------------------------------------------------------------------

def test_test_window_count_equals_engines():
    _, test_df, rul_arr = _make_train_df(), *_make_test_df(n_units=7)
    scaler = MinMaxScaler().fit(test_df[FEAT_COLS])
    test_scaled = apply_scaler(test_df, scaler, FEAT_COLS)

    X_test, y_test = make_test_windows(test_scaled, FEAT_COLS, rul_arr, window_length=10)

    n_engines = test_df["unit"].nunique()
    assert X_test.shape[0] == n_engines, (
        f"Expected {n_engines} test windows, got {X_test.shape[0]}"
    )
    assert y_test.shape[0] == n_engines


# ---------------------------------------------------------------------------
# Test 3: Scaler fitted on train only
# ---------------------------------------------------------------------------

def test_scaler_fit_on_train_only():
    train_df = _make_train_df(n_units=5, max_life=50)
    test_df, _  = _make_test_df(n_units=3)

    # Give test data values well outside train range
    test_df[FEAT_COLS[0]] += 10_000.0

    scaler = fit_scaler(train_df, FEAT_COLS)
    train_scaled = apply_scaler(train_df, scaler, FEAT_COLS)
    test_scaled  = apply_scaler(test_df,  scaler, FEAT_COLS)

    # Train should be in [0,1] (roughly)
    assert train_scaled[FEAT_COLS].max().max() <= 1.01
    assert train_scaled[FEAT_COLS].min().min() >= -0.01

    # Test (with boosted values) should exceed 1.0
    assert test_scaled[FEAT_COLS[0]].max() > 1.0, (
        "Test set should be able to exceed [0,1] if scaler was fit on train only"
    )


# ---------------------------------------------------------------------------
# Test 4: Label alignment - hand-made example
# ---------------------------------------------------------------------------

def test_rul_label_alignment():
    """
    2 units: unit 1 with 5 cycles, unit 2 with 3 cycles.
    rul_clip = 10 (large, so no clipping in this tiny example).
    Expected RUL at each cycle for unit 1: [4, 3, 2, 1, 0].
    """
    rows = []
    for cycle in range(1, 6):
        rows.append({"unit": 1, "cycle": cycle, "s1": float(cycle)})
    for cycle in range(1, 4):
        rows.append({"unit": 2, "cycle": cycle, "s1": float(cycle)})

    df = pd.DataFrame(rows)
    df["op1"] = df["op2"] = df["op3"] = 0.0
    for fi in range(2, 22):
        df[f"s{fi}"] = 0.0

    df = add_rul_train(df, rul_clip=1000)

    unit1 = df[df["unit"] == 1].sort_values("cycle")["RUL"].values
    unit2 = df[df["unit"] == 2].sort_values("cycle")["RUL"].values

    np.testing.assert_array_equal(unit1, [4, 3, 2, 1, 0])
    np.testing.assert_array_equal(unit2, [2, 1, 0])


# ---------------------------------------------------------------------------
# Test 5: Padding for short engines
# ---------------------------------------------------------------------------

def test_short_engine_padding():
    """Engine with only 5 cycles padded to window_length=10."""
    short_test = pd.DataFrame({
        "unit":  [1] * 5,
        "cycle": list(range(1, 6)),
        "s1": [float(i) for i in range(5)],
    })
    for fi in range(2, 22):
        short_test[f"s{fi}"] = 0.0
    short_test["op1"] = short_test["op2"] = short_test["op3"] = 0.0

    rul_arr = np.array([50.0], dtype=np.float32)
    X_test, y_test = make_test_windows(short_test, ["s1"], rul_arr, window_length=10)

    assert X_test.shape == (1, 10, 1), f"Expected (1, 10, 1), got {X_test.shape}"
    # First 5 rows should be the pad (repeat of first cycle value = 0.0)
    np.testing.assert_array_equal(X_test[0, :5, 0], [0.0] * 5)
    # Last 5 rows should be the actual data [0,1,2,3,4]
    np.testing.assert_array_equal(X_test[0, 5:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Test 6: Test windows matched 1:1 with RUL array in unit-ID order
# ---------------------------------------------------------------------------

def test_test_rul_ordering():
    """
    3 units (IDs 3,1,2 in DataFrame order).
    rul_arr = [10, 20, 30] -> should match sorted unit IDs [1,2,3].
    So y_test should be [10, 20, 30] for units [1, 2, 3].
    """
    rows = []
    for uid, cycles in [(3, 8), (1, 15), (2, 5)]:
        for c in range(1, cycles + 1):
            rows.append({"unit": uid, "cycle": c, "s1": float(c)})
    test_df = pd.DataFrame(rows)
    for fi in range(2, 22):
        test_df[f"s{fi}"] = 0.0
    test_df["op1"] = test_df["op2"] = test_df["op3"] = 0.0

    rul_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)  # for units 1,2,3
    X_test, y_test = make_test_windows(test_df, ["s1"], rul_arr, window_length=4)

    # sorted units: [1, 2, 3] -> RUL [10, 20, 30]
    np.testing.assert_array_equal(y_test, [10.0, 20.0, 30.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
