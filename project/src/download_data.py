"""
download_data.py - NASA C-MAPSS Dataset Verification and Downloader.

Verifies the integrity of the 12 canonical C-MAPSS telemetry files in data/raw/:
  - train_FD001.txt, test_FD001.txt, RUL_FD001.txt
  - train_FD002.txt, test_FD002.txt, RUL_FD002.txt
  - train_FD003.txt, test_FD003.txt, RUL_FD003.txt
  - train_FD004.txt, test_FD004.txt, RUL_FD004.txt

If files are missing, provides direct download instructions and links:
  - Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
  - NASA Prognostics Center of Excellence (PCoE)
"""
from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = [
    f"{split}_{subset}.txt"
    for subset in ["FD001", "FD002", "FD003", "FD004"]
    for split in ["train", "test", "RUL"]
]

KAGGLE_URL = "https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"
GITHUB_MIRROR_URL = "https://raw.githubusercontent.com/rahulxiao/Predictive-Maintenance-and-Anomaly-Detection-for-Industrial-IoT-Telemetry-Using-Machine-Learning/main/project/data/raw/"


def verify_dataset(raw_dir: Path | None = None) -> tuple[bool, list[str]]:
    """Checks if all 12 raw C-MAPSS files exist and are non-empty."""
    target_dir = raw_dir or RAW_DIR
    if not target_dir.exists():
        return False, EXPECTED_FILES

    missing = []
    for fname in EXPECTED_FILES:
        fpath = target_dir / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            missing.append(fname)

    return len(missing) == 0, missing


def download_missing_files(raw_dir: Path | None = None) -> bool:
    """Attempts to download missing files from repository mirror or prints manual instructions."""
    target_dir = raw_dir or RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    is_complete, missing = verify_dataset(target_dir)
    if is_complete:
        print("[OK] All 12 NASA C-MAPSS raw data files are present and verified.")
        return True

    print(f"\n[INFO] Missing {len(missing)} dataset files in {target_dir}:")
    for f in missing:
        print(f"  - {f}")

    print("\nAttempting automatic retrieval from repository mirror...")
    downloaded_all = True
    for fname in missing:
        file_url = GITHUB_MIRROR_URL + fname
        dest_path = target_dir / fname
        try:
            print(f"--> Downloading {fname} ...", end="", flush=True)
            urllib.request.urlretrieve(file_url, dest_path)
            if dest_path.exists() and dest_path.stat().st_size > 0:
                print(" [OK]")
            else:
                print(" [FAIL]")
                downloaded_all = False
        except Exception as e:
            print(f" [FAIL: {e}]")
            downloaded_all = False

    is_complete, missing = verify_dataset(target_dir)
    if is_complete:
        print("\n[SUCCESS] All dataset files successfully retrieved and verified!")
        return True

    print("\n" + "=" * 78)
    print(" [MANUAL DATASET DOWNLOAD REQUIRED]")
    print("=" * 78)
    print(" The NASA C-MAPSS dataset could not be fully downloaded automatically.")
    print(" Please download the files from Kaggle or NASA and place them into:")
    print(f"   {target_dir.resolve()}\n")
    print(" Download Links:")
    print(f"   * Kaggle Dataset : {KAGGLE_URL}")
    print("   * NASA Repository: https://data.nasa.gov/\n")
    print(f" Expected files in {target_dir.name}/:")
    for f in EXPECTED_FILES:
        print(f"   - {f}")
    print("=" * 78 + "\n")
    return False


if __name__ == "__main__":
    success = download_missing_files()
    sys.exit(0 if success else 1)
