"""
data_loader.py - NASA C-MAPSS data loading, preprocessing, and windowing.

Correctness guarantees:
  - No synthetic data. Missing files raise a clear error.
  - Scaler is fit on TRAIN only and applied to both train and test.
  - Test windows: one per test engine, using the LAST window_length cycles.
    Engines shorter than window_length are padded by repeating the first row.
  - Test engines are sorted by unit ID and matched 1:1 with RUL_FD00X.txt.
  - RUL clip (rul_clip) is applied consistently to train targets only.
    True test RUL values from the file are used as-is for metric computation
    (unclipped), with a clipped version also reported.
  - Feature selection uses variance on TRAIN only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)


# ---------------------------------------------------------------------------
# 1. Raw file loading
# ---------------------------------------------------------------------------

def _check_files(raw_dir: Path, subsets: list[int]) -> None:
    """Raise FileNotFoundError listing every missing file."""
    missing = []
    for s in subsets:
        tag = f"FD00{s}"
        for prefix in ("train_", "test_", "RUL_"):
            ext = ".txt"
            p = raw_dir / f"{prefix}{tag}{ext}"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Missing C-MAPSS raw files:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\n\nDownload from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/"
        )


def load_raw(raw_dir: Path, subset_id: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load train/test DataFrames and true RUL array for one subset.

    Returns:
        train   : DataFrame with COLS columns
        test    : DataFrame with COLS columns (all cycles of each test engine)
        rul_arr : 1-D array of true final RUL for each test engine (from RUL file)
    """
    tag = f"FD00{subset_id}"
    train = pd.read_csv(
        raw_dir / f"train_{tag}.txt",
        sep=r"\s+", header=None, names=COLS, engine="python"
    )
    test = pd.read_csv(
        raw_dir / f"test_{tag}.txt",
        sep=r"\s+", header=None, names=COLS, engine="python"
    )
    rul = pd.read_csv(
        raw_dir / f"RUL_{tag}.txt",
        sep=r"\s+", header=None, names=["RUL"], engine="python"
    )
    return train, test, rul["RUL"].values.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Feature selection (fit on train only)
# ---------------------------------------------------------------------------

def select_features(train: pd.DataFrame, var_threshold: float = 0.001) -> list[str]:
    """Return sensor columns whose variance (on train) exceeds var_threshold."""
    sensor_cols = [c for c in train.columns if c.startswith("s")]
    variances = train[sensor_cols].var()
    selected = variances[variances > var_threshold].index.tolist()
    return selected


# ---------------------------------------------------------------------------
# 3. RUL labelling (train only)
# ---------------------------------------------------------------------------

def add_rul_train(df: pd.DataFrame, rul_clip: int = 125) -> pd.DataFrame:
    """
    Add clipped RUL column to training DataFrame.
    RUL = max_cycle_for_unit - current_cycle, clipped to rul_clip.
    """
    df = df.copy()
    max_cycles = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="unit")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_clip).astype(np.float32)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 4. Scaler (fit on train, transform both)
# ---------------------------------------------------------------------------

def fit_scaler(train: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler,
                 feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# ---------------------------------------------------------------------------
# 5. Sliding-window creation (train)
# ---------------------------------------------------------------------------

def make_train_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_length: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding windows over the training set.

    Returns:
        X      : (N, window_length, n_features)  float32
        y      : (N,)  float32  RUL at last step of window
        groups : (N,)  int      unit ID (for GroupKFold)
    """
    X_list, y_list, g_list = [], [], []
    for unit_id, grp in df.groupby("unit"):
        grp = grp.sort_values("cycle").reset_index(drop=True)
        feats = grp[feature_cols].values.astype(np.float32)
        rul   = grp["RUL"].values.astype(np.float32)
        n = len(grp)
        for start in range(0, n - window_length + 1, stride):
            end = start + window_length
            X_list.append(feats[start:end])
            y_list.append(rul[end - 1])
            g_list.append(int(unit_id))
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(g_list, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# 6. Test window creation: one per engine, last window_length cycles
# ---------------------------------------------------------------------------

def make_test_windows(
    test: pd.DataFrame,
    feature_cols: list[str],
    rul_arr: np.ndarray,
    window_length: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one window per test engine (last window_length cycles).
    Engines with fewer cycles than window_length are padded by repeating the
    FIRST row on the left (matches convention of padding with earliest reading).

    Test engines are sorted by unit ID.
    rul_arr must be 1-D with one entry per engine (same order as sorted unit IDs).

    Returns:
        X_test : (n_engines, window_length, n_features)  float32
        y_test : (n_engines,)  float32  true final RUL (from RUL file, unclipped)
    """
    sorted_units = sorted(test["unit"].unique())
    assert len(sorted_units) == len(rul_arr), (
        f"Unit count ({len(sorted_units)}) != RUL file lines ({len(rul_arr)})"
    )
    X_list, y_list = [], []
    for i, unit_id in enumerate(sorted_units):
        grp = test[test["unit"] == unit_id].sort_values("cycle")
        feats = grp[feature_cols].values.astype(np.float32)
        n = len(feats)
        if n < window_length:
            pad = np.repeat(feats[:1], window_length - n, axis=0)
            feats = np.vstack([pad, feats])
        X_list.append(feats[-window_length:])
        y_list.append(rul_arr[i])
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# 7. Diagnostic: print first 10 test engines (unit, last_cycle, true_RUL)
# ---------------------------------------------------------------------------

def print_test_alignment(test: pd.DataFrame, rul_arr: np.ndarray,
                          subset_id: int, n: int = 10) -> None:
    sorted_units = sorted(test["unit"].unique())
    last_cycles  = test.groupby("unit")["cycle"].max()
    print(f"\n[FD00{subset_id}] First {n} test engines (unit / last_cycle / true_RUL from file):")
    print(f"  {'unit':>6}  {'last_cycle':>10}  {'true_RUL':>9}")
    for i, uid in enumerate(sorted_units[:n]):
        print(f"  {uid:>6}  {last_cycles[uid]:>10}  {rul_arr[i]:>9.0f}")


# ---------------------------------------------------------------------------
# 8. High-level pipeline
# ---------------------------------------------------------------------------

def build_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Full preprocessing pipeline for all configured subsets.

    Returns a dict with keys like 'fd1_train_X', 'fd1_train_y', 'fd1_groups',
    'fd1_test_X', 'fd1_test_y', 'fd1_features', 'fd1_scaler'.
    """
    raw_dir   = Path(cfg["data"]["raw_dir"])
    proc_dir  = Path(cfg["data"]["processed_dir"])
    subsets   = cfg["data"]["subsets"]

    rul_clip      = cfg["preprocessing"]["rul_clip"]
    var_thresh    = cfg["preprocessing"]["near_constant_variance_threshold"]
    window_length = cfg["preprocessing"]["window_length"]
    stride        = cfg["preprocessing"]["window_stride"]

    # Fail fast on missing files
    _check_files(raw_dir, subsets)

    proc_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Any] = {}

    for s in subsets:
        tag = f"fd{s}"
        print(f"\n[DataLoader] Subset FD00{s}")

        train_raw, test_raw, rul_test = load_raw(raw_dir, s)
        print(f"  Raw  train: {train_raw.shape}  test: {test_raw.shape}  "
              f"rul_lines: {len(rul_test)}")

        # Feature selection on TRAIN only
        feature_cols = select_features(train_raw, var_thresh)
        print(f"  Features ({len(feature_cols)}): {feature_cols}")

        # RUL labelling on train
        train_raw = add_rul_train(train_raw, rul_clip)

        # Scaler fit on TRAIN only, applied to both
        scaler = fit_scaler(train_raw, feature_cols)
        train_scaled = apply_scaler(train_raw, scaler, feature_cols)
        test_scaled  = apply_scaler(test_raw,  scaler, feature_cols)

        # Print test alignment before windowing
        print_test_alignment(test_raw, rul_test, s)

        # Windows
        X_train, y_train, groups = make_train_windows(
            train_scaled, feature_cols, window_length, stride
        )
        X_test, y_test = make_test_windows(
            test_scaled, feature_cols, rul_test, window_length
        )

        print(f"  Windows  train: {X_train.shape}  test: {X_test.shape}")
        print(f"  y_train  min={y_train.min():.1f}  max={y_train.max():.1f}  "
              f"mean={y_train.mean():.1f}")
        print(f"  y_test   min={y_test.min():.1f}   max={y_test.max():.1f}   "
              f"mean={y_test.mean():.1f}")

        datasets.update({
            f"{tag}_train_X": X_train,
            f"{tag}_train_y": y_train,
            f"{tag}_groups":  groups,
            f"{tag}_test_X":  X_test,
            f"{tag}_test_y":  y_test,
            f"{tag}_features": feature_cols,
            f"{tag}_scaler":   scaler,
        })

        # Persist
        np.save(proc_dir / f"{tag}_train_X.npy", X_train)
        np.save(proc_dir / f"{tag}_train_y.npy", y_train)
        np.save(proc_dir / f"{tag}_groups.npy",  groups)
        np.save(proc_dir / f"{tag}_test_X.npy",  X_test)
        np.save(proc_dir / f"{tag}_test_y.npy",  y_test)

    print("\n[DataLoader] Done.")
    return datasets
