"""
metrics_lead_time.py - Part 1: Standardized Lead-Time & Alarm Evaluation Metrics.

Definitions:
  - An alarm at cycle c is 'sustained' when raw alarms fire for `persistence`
    consecutive cycles (default persistence = 3).
  - Horizon H (default H = 100, also evaluated at H = 50):
    * Valid early warning: first sustained alarm at cycle c where RUL(c) <= H.
      Lead time = T - c = RUL(c) (where T is failure cycle).
    * Premature alarm: any sustained alarm at cycle c where RUL(c) > H.
      Counts as a FALSE ALARM / premature alarm, NEVER as valid lead time.
    * Missed engine: no sustained alarm occurs with RUL(c) <= H.
  - Metrics per method:
    * Detection Rate (%): share of engines with a valid early warning (RUL <= H).
    * Mean & Median Lead Time: computed over detected engines only.
    * Premature Alarm Share (%): share of engines that suffered >= 1 premature alarm.
    * False Alarms per 100 Healthy Cycles: (false alarm cycles in healthy period / healthy cycles) * 100.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def compute_sustained_alarms(raw_alarms: np.ndarray, persistence: int = 3) -> np.ndarray:
    """
    Given binary raw_alarms array, return binary sustained_alarms array.
    A sustained alarm latches / activates once raw alarms fire for `persistence`
    consecutive cycles.
    """
    n = len(raw_alarms)
    sustained = np.zeros(n, dtype=int)
    streak = 0
    latched = False
    for t in range(n):
        if raw_alarms[t] == 1:
            streak += 1
            if streak >= persistence:
                latched = True
        else:
            streak = 0
        if latched:
            sustained[t] = 1
    return sustained


def evaluate_engine_lead_time(
    cycles: np.ndarray,
    sustained_alarms: np.ndarray,
    failure_cycle: int,
    horizon: int = 100,
) -> dict[str, Any]:
    """
    Evaluate alarm sequence for a single engine trajectory.

    Returns:
      - detected: bool (valid early warning occurred with RUL <= horizon)
      - lead_time: float or None (lead time in cycles if detected, else None)
      - premature_alarm: bool (sustained alarm occurred when RUL > horizon)
      - first_alarm_cycle: int or None
      - healthy_cycles_count: int (cycles where RUL > horizon)
      - false_alarm_cycles_count: int (cycles where RUL > horizon and alarm is active)
    """
    ruls = failure_cycle - cycles
    healthy_mask = ruls > horizon
    healthy_cycles_count = int(np.sum(healthy_mask))
    false_alarm_cycles_count = int(np.sum(sustained_alarms[healthy_mask] == 1))

    # Find first sustained alarm cycle
    alarm_indices = np.where(sustained_alarms == 1)[0]
    if len(alarm_indices) == 0:
        return {
            "detected": False,
            "lead_time": None,
            "premature_alarm": False,
            "first_alarm_cycle": None,
            "healthy_cycles_count": healthy_cycles_count,
            "false_alarm_cycles_count": false_alarm_cycles_count,
        }

    first_idx = alarm_indices[0]
    first_cycle = int(cycles[first_idx])
    rul_at_first = failure_cycle - first_cycle

    if rul_at_first > horizon:
        # Premature alarm! Counts as false alarm, NOT as valid lead time
        return {
            "detected": False,
            "lead_time": None,
            "premature_alarm": True,
            "first_alarm_cycle": first_cycle,
            "healthy_cycles_count": healthy_cycles_count,
            "false_alarm_cycles_count": false_alarm_cycles_count,
        }
    else:
        # Valid early warning within horizon
        return {
            "detected": True,
            "lead_time": float(rul_at_first),
            "premature_alarm": False,
            "first_alarm_cycle": first_cycle,
            "healthy_cycles_count": healthy_cycles_count,
            "false_alarm_cycles_count": false_alarm_cycles_count,
        }


def evaluate_fleet_lead_time(
    engine_results: list[dict[str, Any]],
    horizon: int = 100,
) -> dict[str, float]:
    """
    Aggregate per-engine lead-time evaluations across an entire fleet.
    """
    n_engines = len(engine_results)
    if n_engines == 0:
        return {
            "Detection_Rate_Pct": 0.0,
            "Mean_Lead_Time": 0.0,
            "Median_Lead_Time": 0.0,
            "Premature_Alarm_Share_Pct": 0.0,
            "False_Alarms_Per_100_Healthy": 0.0,
            "Horizon": float(horizon),
        }

    detected_lead_times = [r["lead_time"] for r in engine_results if r["detected"] and r["lead_time"] is not None]
    n_detected = len(detected_lead_times)
    n_premature = sum(1 for r in engine_results if r["premature_alarm"])

    tot_healthy_cycles = sum(r["healthy_cycles_count"] for r in engine_results)
    tot_false_alarm_cycles = sum(r["false_alarm_cycles_count"] for r in engine_results)

    fa_per_100 = (tot_false_alarm_cycles / tot_healthy_cycles * 100.0) if tot_healthy_cycles > 0 else 0.0

    return {
        "Detection_Rate_Pct": (n_detected / n_engines) * 100.0,
        "Mean_Lead_Time": float(np.mean(detected_lead_times)) if n_detected > 0 else 0.0,
        "Median_Lead_Time": float(np.median(detected_lead_times)) if n_detected > 0 else 0.0,
        "Premature_Alarm_Share_Pct": (n_premature / n_engines) * 100.0,
        "False_Alarms_Per_100_Healthy": float(fa_per_100),
        "Horizon": float(horizon),
    }
