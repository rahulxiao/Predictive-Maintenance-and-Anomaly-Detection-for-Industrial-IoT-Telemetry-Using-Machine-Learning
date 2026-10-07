"""
evaluate.py - Evaluation metrics, baselines, and result saving.

Key additions vs previous version:
  - Reports both unclipped RMSE and clipped RMSE (clipped to rul_clip).
  - Sanity baselines: constant=rul_clip, constant=train_mean.
  - Prints 15 predicted vs true RUL rows per model.
  - print_metrics_table prints all entries sorted by subset then RMSE.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA scoring function (lower is better, positive d = late prediction):
    S = sum( exp(-d/13) - 1  if d < 0,
             exp( d/10) - 1  if d >= 0 )
    where d = y_pred - y_true.
    """
    d = y_pred.astype(float) - y_true.astype(float)
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rul_clip: int = 125,
) -> dict[str, float]:
    """
    Compute RMSE, MAE, NASA score on raw (unclipped) values,
    plus RMSE_clip where both y_true and y_pred are clipped to rul_clip.
    """
    y_pred_clipped  = np.clip(y_pred,  0, rul_clip)
    y_true_clipped  = np.clip(y_true,  0, rul_clip)
    return {
        "RMSE":       rmse(y_true, y_pred_clipped),
        "RMSE_clip":  rmse(y_true_clipped, y_pred_clipped),
        "MAE":        mae(y_true,  y_pred_clipped),
        "NASA_Score": nasa_score(y_true, y_pred_clipped),
    }


# ---------------------------------------------------------------------------
# Sanity baselines
# ---------------------------------------------------------------------------

def compute_baselines(
    y_train: np.ndarray,
    y_test: np.ndarray,
    rul_clip: int = 125,
) -> dict[str, dict[str, float]]:
    """
    Two naive baselines every real model must beat:
      - constant_clip  : always predict rul_clip (125)
      - constant_mean  : always predict train mean RUL
    """
    train_mean = float(y_train.mean())
    baselines: dict[str, dict[str, float]] = {}
    for name, val in [("constant_clip", rul_clip), ("constant_mean", train_mean)]:
        pred = np.full_like(y_test, val, dtype=float)
        baselines[name] = evaluate_all(y_test, pred, rul_clip)
        baselines[name]["constant_value"] = val
    return baselines


# ---------------------------------------------------------------------------
# Print predictions vs truth
# ---------------------------------------------------------------------------

def print_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    subset_label: str,
    n: int = 15,
) -> None:
    print(f"\n  [{subset_label} {model_name}] Predicted vs True RUL (first {n} test engines):")
    print(f"  {'#':>4}  {'True':>8}  {'Pred':>8}  {'Err':>8}")
    for i in range(min(n, len(y_true))):
        err = y_pred[i] - y_true[i]
        print(f"  {i+1:>4}  {y_true[i]:>8.1f}  {y_pred[i]:>8.1f}  {err:>+8.1f}")


# ---------------------------------------------------------------------------
# Saving and display
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict[str, Any], output_dir: Path,
                 filename: str = "metrics.json") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Evaluate] Metrics saved -> {out_path}")


def print_metrics_table(
    metrics: dict[str, dict[str, float]],
    baselines: dict[str, dict[str, dict[str, float]]] | None = None,
) -> None:
    """Print comparison table of all models x subsets, with baselines."""
    rows = []
    for key, vals in metrics.items():
        parts = key.split("_", 1)
        rows.append({
            "Subset":     parts[0].upper(),
            "Model":      parts[1] if len(parts) > 1 else key,
            "RMSE":       f"{vals['RMSE']:.2f}",
            "RMSE_clip":  f"{vals.get('RMSE_clip', float('nan')):.2f}",
            "MAE":        f"{vals['MAE']:.2f}",
            "NASA_Score": f"{vals['NASA_Score']:.1f}",
        })

    if baselines:
        for subset_tag, blines in baselines.items():
            for bname, bvals in blines.items():
                rows.append({
                    "Subset":     subset_tag.upper(),
                    "Model":      f"[baseline] {bname} ({bvals.get('constant_value',0):.1f})",
                    "RMSE":       f"{bvals['RMSE']:.2f}",
                    "RMSE_clip":  f"{bvals.get('RMSE_clip', float('nan')):.2f}",
                    "MAE":        f"{bvals['MAE']:.2f}",
                    "NASA_Score": f"{bvals['NASA_Score']:.1f}",
                })

    df = pd.DataFrame(rows).sort_values(["Subset", "RMSE"])
    col_w = max(df["Model"].str.len().max() + 2, 20)
    print("\n" + "=" * (col_w + 55))
    print(f"  {'Subset':<7} {'Model':<{col_w}} {'RMSE':>8} {'RMSE_clip':>10} "
          f"{'MAE':>8} {'NASA_Score':>11}")
    print("=" * (col_w + 55))
    for _, row in df.iterrows():
        print(f"  {row['Subset']:<7} {row['Model']:<{col_w}} {row['RMSE']:>8} "
              f"{row['RMSE_clip']:>10} {row['MAE']:>8} {row['NASA_Score']:>11}")
    print("=" * (col_w + 55) + "\n")
