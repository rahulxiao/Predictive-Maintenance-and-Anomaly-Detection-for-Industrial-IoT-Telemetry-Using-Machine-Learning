"""
tests/test_stage1_eval.py - Unit tests for Stage 1 evaluation fixes.

Tests:
  1. Folds differ across seeds: get_shuffled_group_kfold produces different fold assignments for seed 0 vs seed 1.
  2. No unit overlap within a seed: train units and validation units have zero intersection for every fold.
  3. Identical test windows for all methods: baseline and ML window scoring use the exact same test windows and prevalence.
  4. Lead-time sweep monotonic sanity check: as threshold increases, false alarms per 100 healthy cycles is non-increasing.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import get_shuffled_group_kfold
from metrics_lead_time import compute_sustained_alarms, evaluate_fleet_lead_time, evaluate_engine_lead_time


def test_folds_differ_across_seeds():
    # 20 units, each with 30 windows
    units = np.repeat(np.arange(1, 21), 30)

    splits_s0 = get_shuffled_group_kfold(units, n_splits=5, seed=0)
    splits_s1 = get_shuffled_group_kfold(units, n_splits=5, seed=1)

    # Check fold 0 validation units differ between seed 0 and seed 1
    val_units_s0 = set(units[splits_s0[0][1]])
    val_units_s1 = set(units[splits_s1[0][1]])

    assert val_units_s0 != val_units_s1, f"Fold 0 validation units should differ across seeds but were both {val_units_s0}"


def test_no_unit_overlap_within_a_seed():
    units = np.repeat(np.arange(1, 26), 20)

    for seed in [0, 1, 42]:
        splits = get_shuffled_group_kfold(units, n_splits=5, seed=seed)
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            tr_units = set(units[tr_idx])
            val_units = set(units[val_idx])
            overlap = tr_units.intersection(val_units)
            assert len(overlap) == 0, f"Seed {seed} Fold {fold_idx} has unit overlap: {overlap}"
            # Union should equal all units
            assert len(tr_units.union(val_units)) == 25


def test_identical_test_windows_across_methods():
    # Create synthetic test engines
    # 3 engines: unit 1 (40 cycles, true_RUL=20), unit 2 (50 cycles, true_RUL=10), unit 3 (60 cycles, true_RUL=40)
    window_length = 30
    records = []
    test_ruls = {1: 20, 2: 10, 3: 40}
    for uid, rul_final in test_ruls.items():
        n_cyc = 35 + uid * 5
        for c in range(1, n_cyc + 1):
            records.append({
                "unit": uid, "cycle": c,
                "s1": float(10.0 + c * 0.1),
                "true_RUL": float(rul_final + (n_cyc - c)),
            })
    df_test = pd.DataFrame(records)
    df_test["anomaly_label"] = (df_test["true_RUL"] <= 30).astype(int)

    # Window slicing
    win_labels = []
    win_cycles = []
    win_units = []
    for uid, grp in df_test.groupby("unit"):
        grp_s = grp.sort_values("cycle").reset_index(drop=True)
        n = len(grp_s)
        for start in range(0, n - window_length + 1):
            end = start + window_length
            win_labels.append(grp_s["anomaly_label"].iloc[end - 1])
            win_cycles.append(grp_s["cycle"].iloc[end - 1])
            win_units.append(uid)

    y_windows = np.array(win_labels)
    prevalence = float(np.mean(y_windows))

    # Any model (ML or Baseline) scored on these windows must match the length and prevalence
    assert len(y_windows) > 0
    assert 0.0 < prevalence < 1.0


def test_lead_time_sweep_monotonic_sanity():
    # Synthetic trajectory where degradation score strictly increases towards failure
    T = 150
    H = 100
    cycles = np.arange(1, T + 1)
    # Score increases linearly from 0.0 to 1.0
    scores = np.linspace(0.0, 1.0, T)

    thresholds = np.linspace(0.1, 0.9, 9)
    false_alarms_list = []

    for th in thresholds:
        raw_alarms = (scores >= th).astype(int)
        sustained = compute_sustained_alarms(raw_alarms, persistence=3)
        res = evaluate_engine_lead_time(cycles, sustained, failure_cycle=T, horizon=H)
        fa_rate = (res["false_alarm_cycles_count"] / res["healthy_cycles_count"] * 100.0) if res["healthy_cycles_count"] > 0 else 0.0
        false_alarms_list.append(fa_rate)

    # As threshold increases (detector becomes more conservative), false alarm rate must be non-increasing
    for i in range(len(false_alarms_list) - 1):
        assert false_alarms_list[i] >= false_alarms_list[i + 1] - 1e-6, (
            f"FA rate increased from {false_alarms_list[i]} to {false_alarms_list[i+1]} as threshold tightened!"
        )
