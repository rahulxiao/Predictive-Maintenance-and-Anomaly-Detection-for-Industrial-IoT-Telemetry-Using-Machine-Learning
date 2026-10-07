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
    assert pytest.approx(fleet["False_Ala