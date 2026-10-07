"""
main.py - Orchestrates the C-MAPSS predictive maintenance pipeline.

Usage:
    python src/main.py                         # all subsets, all models
    python src/main.py --subset 1              # FD001 only
    python src/main.py --models rf xgb        # skip DL models
    python src/main.py --fast                  # fewer epochs, FD001 only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from data_loader import build_dataset
from evaluate import (
    evaluate_all, save_metrics, print_metrics_table,
    compute_baselines, print_pred_vs_true,
)
from models import RandomForestRUL, XGBoostRUL, LSTMModel, CNNModel
from trainer import train_torch_model, predict_torch
from utils import get_logger, load_config, log_environment, set_seed, Timer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C-MAPSS Predictive Maintenance")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--subset", type=int, default=None,
                   help="Run only this FD subset (1-4).")
    p.add_argument("--models", nargs="+",
                   choices=["rf", "xgb", "lstm", "cnn"],
                   default=["rf", "xgb", "lstm", "cnn"])
    p.add_argument("--fast", action="store_true",
                   help="Quick smoke-test: FD001 only, few epochs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    project_root = Path(__file__).parent.parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / args.config
    cfg = load_config(cfg_path)

    # Fast-mode overrides
    if args.fast:
        cfg["data"]["subsets"] = [1]
        cfg["models"]["lstm"]["max_epochs"]  = 10
        cfg["models"]["cnn"]["max_epochs"]   = 10
        cfg["models"]["random_forest"]["n_estimators"] = 50
        cfg["models"]["xgboost"]["n_estimators"]       = 50

    if args.subset is not None:
        cfg["data"]["subsets"] = [args.subset]

    # Resolve paths relative to project root
    for key in ("raw_dir", "processed_dir"):
        p = Path(cfg["data"][key])
        if not p.is_absolute():
            cfg["data"][key] = str(project_root / p)
    results_dir = Path(cfg["results"]["metrics_dir"])
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    log_file = results_dir / "run.log"
    logger = get_logger("main", log_file=log_file)
    logger.info("=" * 60)
    logger.info("NASA C-MAPSS Predictive Maintenance Pipeline")
    logger.info("=" * 60)
    log_environment(logger)

    set_seed(cfg["models"]["random_seed"])

    # Data
    with Timer("Data loading & preprocessing", logger):
        datasets = build_dataset(cfg)

    rul_clip      = cfg["evaluation"]["rul_clip"]
    window_length = cfg["preprocessing"]["window_length"]

    all_metrics:   dict[str, dict] = {}
    all_baselines: dict[str, dict] = {}

    for s in cfg["data"]["subsets"]:
        tag   = f"fd{s}"
        label = f"FD00{s}"

        X_train = datasets[f"{tag}_train_X"]
        y_train = datasets[f"{tag}_train_y"]
        X_test  = datasets[f"{tag}_test_X"]
        y_test  = datasets[f"{tag}_test_y"]
        n_feat  = X_train.shape[2]

        logger.info("\n--- Subset %s ---", label)
        logger.info("Train: %s | Test: %s | Features: %d",
                    X_train.shape, X_test.shape, n_feat)

        # Sanity baselines
        blines = compute_baselines(y_train, y_test, rul_clip)
        all_baselines[tag] = blines
        logger.info("[BASELINE] constant_clip(125): RMSE=%.2f | MAE=%.2f | NASA=%.1f",
                    blines["constant_clip"]["RMSE"],
                    blines["constant_clip"]["MAE"],
                    blines["constant_clip"]["NASA_Score"])
        logger.info("[BASELINE] constant_mean(%.1f): RMSE=%.2f | MAE=%.2f | NASA=%.1f",
                    blines["constant_mean"]["constant_value"],
                    blines["constant_mean"]["RMSE"],
                    blines["constant_mean"]["MAE"],
                    blines["constant_mean"]["NASA_Score"])

        # ── Random Forest ─────────────────────────────────────────────────
        if "rf" in args.models:
            logger.info("\n[RF] Training Random Forest...")
            with Timer("RF training", logger):
                rf = RandomForestRUL(cfg)
                rf.fit(X_train, y_train)
            preds = np.clip(rf.predict(X_test), 0, rul_clip)
            m = evaluate_all(y_test, preds, rul_clip)
            all_metrics[f"{tag}_RF"] = m
            logger.info("[RF]  RMSE=%.2f | RMSE_clip=%.2f | MAE=%.2f | NASA=%.1f",
                        m["RMSE"], m["RMSE_clip"], m["MAE"], m["NASA_Score"])
            print_pred_vs_true(y_test, preds, "RF", label)

        # ── XGBoost ──────────────────────────────────────────────────────
        if "xgb" in args.models:
            logger.info("\n[XGB] Training XGBoost...")
            try:
                with Timer("XGB training", logger):
                    xgb_model = XGBoostRUL(cfg)
                    xgb_model.fit(X_train, y_train)
                preds = np.clip(xgb_model.predict(X_test), 0, rul_clip)
                m = evaluate_all(y_test, preds, rul_clip)
                all_metrics[f"{tag}_XGB"] = m
                logger.info("[XGB] RMSE=%.2f | RMSE_clip=%.2f | MAE=%.2f | NASA=%.1f",
                            m["RMSE"], m["RMSE_clip"], m["MAE"], m["NASA_Score"])
                print_pred_vs_true(y_test, preds, "XGB", label)
            except ImportError as e:
                logger.warning("[XGB] Skipped: %s", e)

        # ── LSTM ─────────────────────────────────────────────────────────
        if "lstm" in args.models:
            logger.info("\n[LSTM] Training LSTM...")
            with Timer("LSTM training", logger):
                lstm = LSTMModel(n_features=n_feat, cfg=cfg)
                lstm, tr_hist, vl_hist = train_torch_model(
                    lstm, X_train, y_train, cfg,
                    model_key="lstm", logger=logger
                )
            preds = np.clip(predict_torch(lstm, X_test, rul_clip), 0, rul_clip)
            m = evaluate_all(y_test, preds, rul_clip)
            all_metrics[f"{tag}_LSTM"] = m
            logger.info("[LSTM] RMSE=%.2f | RMSE_clip=%.2f | MAE=%.2f | NASA=%.1f",
                        m["RMSE"], m["RMSE_clip"], m["MAE"], m["NASA_Score"])
            logger.info("[LSTM] Final train_RMSE=%.2f | val_RMSE=%.2f",
                        tr_hist[-1], vl_hist[-1])
            print_pred_vs_true(y_test, preds, "LSTM", label)

        # ── CNN ──────────────────────────────────────────────────────────
        if "cnn" in args.models:
            logger.info("\n[CNN] Training 1D-CNN...")
            with Timer("CNN training", logger):
                cnn = CNNModel(n_features=n_feat, window_length=window_length, cfg=cfg)
                cnn, tr_hist, vl_hist = train_torch_model(
                    cnn, X_train, y_train, cfg,
                    model_key="cnn", logger=logger
                )
            preds = np.clip(predict_torch(cnn, X_test, rul_clip), 0, rul_clip)
            m = evaluate_all(y_test, preds, rul_clip)
            all_metrics[f"{tag}_CNN"] = m
            logger.info("[CNN]  RMSE=%.2f | RMSE_clip=%.2f | MAE=%.2f | NASA=%.1f",
                        m["RMSE"], m["RMSE_clip"], m["MAE"], m["NASA_Score"])
            logger.info("[CNN]  Final train_RMSE=%.2f | val_RMSE=%.2f",
                        tr_hist[-1], vl_hist[-1])
            print_pred_vs_true(y_test, preds, "CNN", label)

    # Summary table with baselines
    print_metrics_table(all_metrics, all_baselines)

    # Save full result dict
    full_results = {"models": all_metrics, "baselines": {}}
    for stag, blines in all_baselines.items():
        for bname, bvals in blines.items():
            full_results["baselines"][f"{stag}_{bname}"] = bvals
    save_metrics(full_results, results_dir, "metrics.json")
    logger.info("All done. Results in %s", results_dir)


if __name__ == "__main__":
    main()
