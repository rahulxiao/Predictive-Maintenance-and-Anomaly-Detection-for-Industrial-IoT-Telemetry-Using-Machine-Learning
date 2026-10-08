"""
run_stage2.py - Master Runner for Stage 2 Novelty Experiments.

Executes:
  1. Cross-Condition Generalization (FD001->FD002/FD004, FD003->FD004, plus in-domain, 5 seeds).
  2. Split Conformal Prediction Intervals for RUL (90% target coverage, calibrated on train engines only).
  3. Explainability with TreeSHAP & Sensor Overlap vs Static Threshold Baseline.
  4. Ablation Studies (Smoothing on/off, Window 15/30/50, Representation raw/PCA/engineered stats, 3 seeds).
  5. Results Manifest update.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from stage2_experiments import (
    run_cross_condition_experiments,
    run_conformal_prediction_experiments,
    run_shap_explainability_experiments,
    run_ablation_experiments,
    update_stage2_manifest,
)
from utils import get_logger, load_config, log_environment, Timer

logger = get_logger("run_stage2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 Novelty Experiments Runner")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    p.add_argument("--quick", action="store_true", help="Fast smoke test (1 seed, subset 1, key pairs only)")
    p.add_argument("--skip-cross", action="store_true", help="Skip cross-condition experiments")
    p.add_argument("--skip-conformal", action="store_true", help="Skip conformal prediction experiments")
    p.add_argument("--skip-shap", action="store_true", help="Skip SHAP explainability experiments")
    p.add_argument("--skip-ablations", action="store_true", help="Skip ablation experiments")
    return p.parse_args()


def print_table_formatted(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(f" {title} ")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).parent.parent

    cfg_path = project_root / args.config
    cfg = load_config(cfg_path)

    raw_dir = project_root / cfg["data"]["raw_dir"]
    results_dir = project_root / cfg["results"]["metrics_dir"]
    figures_dir = project_root / cfg["results"]["figures_dir"]

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 90)
    print(f" STAGE 2: NOVELTY EXPERIMENTS {'(QUICK SMOKE TEST)' if args.quick else '(NASA C-MAPSS FD001-FD004)'}")
    print("#" * 90)
    log_environment(logger)

    # -----------------------------------------------------------------------
    # Part 1: Cross-Condition Generalization
    # -----------------------------------------------------------------------
    if not args.skip_cross:
        with Timer("Cross-Condition Generalization", logger):
            if args.quick:
                df_cross = run_cross_condition_experiments(
                    raw_dir=raw_dir,
                    results_dir=results_dir,
                    seeds=[0],
                    pairs=[
                        ("FD001", "FD001", 1, 1, "In_Domain"),
                        ("FD001", "FD002", 1, 2, "Cross_Domain"),
                    ],
                )
            else:
                df_cross = run_cross_condition_experiments(
                    raw_dir=raw_dir,
                    results_dir=results_dir,
                    seeds=[0, 1, 2, 3, 4],
                )
        print_table_formatted("TABLE 1: Cross-Condition Generalization vs In-Domain", df_cross)
    else:
        df_cross = pd.read_csv(results_dir / "cross_condition_transfer.csv") if (results_dir / "cross_condition_transfer.csv").exists() else pd.DataFrame()

    # -----------------------------------------------------------------------
    # Part 2: Split Conformal Prediction Intervals
    # -----------------------------------------------------------------------
    if not args.skip_conformal:
        with Timer("Split Conformal Prediction", logger):
            df_conformal = run_conformal_prediction_experiments(
                raw_dir=raw_dir,
                results_dir=results_dir,
                figures_dir=figures_dir,
                subsets=[1] if args.quick else [1, 2, 3, 4],
                target_coverage=0.90,
                seed=42,
            )
        print_table_formatted("TABLE 2: Split Conformal Prediction Intervals (90% Target Coverage)", df_conformal)
    else:
        df_conformal = pd.read_csv(results_dir / "conformal_prediction_results.csv") if (results_dir / "conformal_prediction_results.csv").exists() else pd.DataFrame()

    # -----------------------------------------------------------------------
    # Part 3: Explainability with TreeSHAP & Sensor Overlap
    # -----------------------------------------------------------------------
    if not args.skip_shap:
        with Timer("SHAP Explainability & Sensor Overlap", logger):
            df_imp, df_overlap = run_shap_explainability_experiments(
                raw_dir=raw_dir,
                results_dir=results_dir,
                figures_dir=figures_dir,
                seed=42,
            )
        print_table_formatted("TABLE 3A: Sensor Overlap: TreeSHAP vs Static Threshold Baseline", df_overlap)
        print_table_formatted("TABLE 3B: Top Sensor Importances by TreeSHAP (Sample)", df_imp.head(20))
    else:
        df_overlap = pd.read_csv(results_dir / "shap_sensor_overlap.csv") if (results_dir / "shap_sensor_overlap.csv").exists() else pd.DataFrame()

    # -----------------------------------------------------------------------
    # Part 4: Ablation Studies
    # -----------------------------------------------------------------------
    if not args.skip_ablations:
        with Timer("Ablation Studies", logger):
            if args.quick:
                df_ablations = run_ablation_experiments(
                    raw_dir=raw_dir,
                    results_dir=results_dir,
                    seeds=[0],
                    configs=[
                        ("Baseline", "Smoothing_OFF_W30_EngineeredStats", False, 30, "Engineered_Stats"),
                        ("Smoothing", "Smoothing_ON_W30_EngineeredStats", True, 30, "Engineered_Stats"),
                    ],
                )
            else:
                df_ablations = run_ablation_experiments(
                    raw_dir=raw_dir,
                    results_dir=results_dir,
                    seeds=[0, 1, 2],
                )
        print_table_formatted("TABLE 4: Ablation Studies on Preprocessing & Representation", df_ablations)
    else:
        df_ablations = pd.read_csv(results_dir / "ablations_summary.csv") if (results_dir / "ablations_summary.csv").exists() else pd.DataFrame()

    # -----------------------------------------------------------------------
    # Part 5: Update Results Manifest
    # -----------------------------------------------------------------------
    manifest_path = update_stage2_manifest(results_dir)
    print(f"\n[Manifest Synchronized] Results manifest -> {manifest_path}")

    print("\nStage 2 execution completed successfully.\n")


if __name__ == "__main__":
    main()
