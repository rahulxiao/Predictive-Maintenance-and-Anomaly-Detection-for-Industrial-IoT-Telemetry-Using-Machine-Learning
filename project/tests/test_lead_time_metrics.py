"""
tests/test_lead_time_metrics.py - Unit test for Part 1 lead-time definition using hand-made alarm sequences.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics_lead_time import (
    compute_sustained_alarms,
    evaluate_engine_lead_time,
    evaluate_fleet_lead_time,
)


def test_sustained_alarm_latching():
    # raw alarms: 2 consecutive (not sustained), then 3 consecutive (sustained and latched)
    raw = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    sustained = compute_sustained_alarms(raw, persistence=3)
    # At index 6 (third 1), sustained latches to 1 and stays 1
    expected = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(sustained, expected)


def test_valid_early_warning_lead_time():
    # Engine life T = 200 cycles, Horizon H = 100
    # Alarm fires at cycle 140, 141, 142 (sustained at 142)
    # RUL at 142 = 200 - 142 = 58 <= 100 -> Valid detection with lead time = 58
    cycles = np.arange(1, 201)
    raw = np.zeros(200, dtype=int)
    raw[139:] = 1  # cycles 140..200
    sustained = compute_sustained_alarms(raw, persistence=3)

    res = evaluate_engine_lead_time(cycles, sustained, failure_cycle=200, horizon=100)
    assert res["detected"] is True
    assert res["lead_time"] == 58.0
    assert res["premature_alarm"] is False
    assert res["first_alarm_cycle"] == 142
    assert res["false_alarm_cycles_count"] == 0


def test_premature_alarm_counted_as_false_alarm():
    # Engine life T = 200 cycles, Horizon H = 100
    # Alarm fires at cycle 80, 81, 82 (sustained at 82)
    # RUL at 82 = 200 - 82 = 118 > 100 -> Premature alarm (False Alarm), NOT lead time
    cycles = np.arange(1, 201)
    raw = np.zeros(200, dtype=int)
    raw[79:] = 1  # cycles 80..200
    sustained = compute_sustained_alarms(raw, persistence=3)

    res = evaluate_engine_lead_time(cycles, sustained, failure_cycle=200, horizon=100)
    assert res["detected"] is False
    assert res["lead_time"] is None
    assert res["premature_alarm"] is True
    assert res["first_alarm_cycle"] == 82
    # Healthy cycles are cycles with RUL > 100 (cycles 1..99 -> 99 cycles)
    # Alarms active during cycles 82..99 = 18 false alarm cycles
    assert res["healthy_cycles_count"] == 99
    assert res["false_alarm_cycles_count"] == 18


def test_missed_engine():
    # Engine life T = 200 cycles, Horizon H = 100
    # Alarm never fires
    cycles = np.arange(1, 201)
    sustained = np.zeros(200, dtype=int)

    res = evaluate_engine_lead_time(cycles, sustained, failure_cycle=200, horizon=100)
    assert res["detected"] is False
    assert res["lead_time"] is None
    assert res["premature_alarm"] is False
    assert res["first_alarm_cycle"] is None


def test_fleet_evaluation_summary():
    # Fleet with 3 engines:
    # Engine 1: Detected with lead time 60
    # Engine 2: Detected with lead time 40
    # Engine 3: Premature alarm with 10 false alarm cycles over 100 healthy cycles
    e1 = {
        "detected": True, "lead_time": 60.0, "premature_alarm": False,
        "healthy_cycles_count": 100, "false_alarm_cycles_count": 0
    }
    e2 = {
        "detected": True, "lead_time": 40.0, "premature_alarm": False,
        "healthy_cycles_count": 100, "false_alarm_cycles_count": 0
    }
    e3 = {
        "detected": False, "lead_time": None, "premature_alarm": True,
        "healthy_cycles_count": 100, "false_alarm_cycles_count": 10
    }

    fleet = evaluate_fleet_lead_time([e1, e2, e3], horizon=100)
    assert pytest.approx(fleet["Detection_Rate_Pct"], rel=1e-3) == 66.66666
    assert fleet["Mean_Lead_Time"] == 50.0
    assert fleet["Median_Lead_Time"] == 50.0
    assert pytest.approx(fleet["Premature_Alarm_Share_Pct"], rel=1e-3) == 33.33333
    # 10 false alarm cycles over 300 healthy cycles * 100 = 3.3333...
    assert pytest.approx(fleet["False_Alarms_Per_100_Healthy"], rel=1e-3) == 3.33333


def test_hand_made_alarm_sequence_covering_all_three_cases():
    """
    Part 1 requirement: Unit test with a hand-made alarm sequence covering all three cases:
      Case 1: Valid early warning (sustained alarm at RUL <= H) -> computes lead time = T - c.
      Case 2: Premature alarm (sustained alarm at RUL > H) -> FALSE ALARM, not lead time.
      Case 3: Missed engine (no sustained alarm with RUL <= H).
    """
    T = 200
    H = 100
    cycles = np.arange(1, T + 1)

    # 1. Engine 1: Valid early warning
    # Fires from cycle 140 onwards. 3 consecutive cycles reached at cycle 142.
    # RUL at 142 is 200 - 142 = 58 <= 100 -> Valid detection, lead time = 58
    raw_1 = np.zeros(T, dtype=int)
    raw_1[139:] = 1
    sust_1 = compute_sustained_alarms(raw_1, persistence=3)
    res_1 = evaluate_engine_lead_time(cycles, sust_1, failure_cycle=T, horizon=H)

    assert res_1["detected"] is True
    assert res_1["lead_time"] == 58.0
    assert res_1["premature_alarm"] is False
    assert res_1["first_alarm_cycle"] == 142
    assert res_1["false_alarm_cycles_count"] == 0

    # 2. Engine 2: Premature alarm
    # Fires from cycle 80 onwards. 3 consecutive cycles reached at cycle 82.
    # RUL at 82 is 200 - 82 = 118 > 100 -> Premature alarm (False alarm, never lead time)
    raw_2 = np.zeros(T, dtype=int)
    raw_2[79:] = 1
    sust_2 = compute_sustained_alarms(raw_2, persistence=3)
    res_2 = evaluate_engine_lead_time(cycles, sust_2, failure_cycle=T, horizon=H)

    assert res_2["detected"] is False
    assert res_2["lead_time"] is None
    assert res_2["premature_alarm"] is True
    assert res_2["first_alarm_cycle"] == 82
    assert res_2["false_alarm_cycles_count"] > 0

    # 3. Engine 3: Missed engine
    # Isolated alarms that never reach persistence = 3 (e.g., 2 cycles then 0)
    raw_3 = np.zeros(T, dtype=int)
    raw_3[150:152] = 1  # only 2 consecutive cycles
    raw_3[170:171] = 1  # only 1 cycle
    sust_3 = compute_sustained_alarms(raw_3, persistence=3)
    res_3 = evaluate_engine_lead_time(cycles, sust_3, failure_cycle=T, horizon=H)

    assert res_3["detected"] is False
    assert res_3["lead_time"] is None
    assert res_3["premature_alarm"] is False
    assert res_3["first_alarm_cycle"] is None

    # Fleet aggregation across the 3 engines
    fleet = evaluate_fleet_lead_time([res_1, res_2, res_3], horizon=H)
    # Exactly 1 out of 3 engines detected -> 33.33%
    assert pytest.approx(fleet["Detection_Rate_Pct"], rel=1e-3) == 33.33333
    # Detected engine lead time is 58.0
    assert fleet["Mean_Lead_Time"] == 58.0
    assert fleet["Median_Lead_Time"] == 58.0
    # Exactly 1 out of 3 engines suffered premature alarm -> 33.33%
    assert pytest.approx(fleet["Premature_Alarm_Share_Pct"], rel=1e-3) == 33.33333

