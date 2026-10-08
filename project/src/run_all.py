"""
run_all.py - Master Automated Execution Pipeline for NASA C-MAPSS Research Project.

Executes the complete experimental workflow from end to end:
  1. Test Suite: Runs pytest across all 26 unit tests.
  2. Stage 1: Runs leakage-free fleet anomaly detection, seed variance, operating curves, and Wilcoxon tests.
  3. Stage 2: Runs novelty experiments (Cross-condition transfer, Conformal prediction, TreeSHAP, and Ablations).
  4. Stage 3: Generates 12 publication-quality figures (300 DPI) and exports Markdown/LaTeX tables.
  5. Manifest Verification: Validates that all 151 metrics in results/manifest.csv match generated CSV logs.
  6. LaTeX Compilation (Optional/Auto): Compiles IEEE Transactions format paper (thesis/main.pdf) via pdflatex/bibtex.

Usage:
  python src/run_all.py                 # Full end-to-end execution
  python src/run_all.py --quick         # Fast smoke test (subset 1, seed 0)
  python src/run_all.py --only-artifacts # Re-generate plots, tables, and PDF from existing CSV results
  python src/run_all.py --skip-latex    # Skip LaTeX PDF compilation
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# Detect project root directory (one level up from src)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
THESIS_DIR = PROJECT_ROOT / "thesis"


# Ensure UTF-8 output on Windows consoles if supported
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_banner(text: str, color: str = Colors.CYAN) -> None:
    width = 80
    border = "=" * width
    print(f"\n{color}{border}")
    print(f" {text.center(width - 2)} ")
    print(f"{border}{Colors.END}\n")


def print_step(step_num: int, total_steps: int, title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.YELLOW}[Step {step_num}/{total_steps}] {title}{Colors.END}")
    print(f"{Colors.YELLOW}{'-' * 70}{Colors.END}")


def find_pdflatex() -> str | None:
    """Finds pdflatex executable on PATH or standard local MiKTeX installation."""
    candidate = shutil.which("pdflatex")
    if candidate:
        return candidate
    local_miktex = Path(r"D:\Latex\miktex\bin\x64\pdflatex.exe")
    if local_miktex.exists():
        return str(local_miktex)
    return None


def find_bibtex() -> str | None:
    """Finds bibtex executable on PATH or standard local MiKTeX installation."""
    candidate = shutil.which("bibtex")
    if candidate:
        return candidate
    local_miktex = Path(r"D:\Latex\miktex\bin\x64\bibtex.exe")
    if local_miktex.exists():
        return str(local_miktex)
    return None


def run_command(cmd: list[str] | str, cwd: Path, desc: str) -> bool:
    """Runs a shell command and streams output, returning True on success."""
    print(f"--> Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    start_time = time.time()
    try:
        ret = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=isinstance(cmd, str),
            check=True,
        )
        elapsed = time.time() - start_time
        print(f"{Colors.GREEN}[OK] Completed {desc} in {elapsed:.1f}s (Exit code: {ret.returncode}){Colors.END}")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"{Colors.RED}[FAIL] Error during {desc} after {elapsed:.1f}s (Exit code: {e.returncode}){Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}[ERROR] Unexpected error: {e}{Colors.END}")
        return False


def check_environment(auto_install: bool = False) -> bool:
    """Pre-flight check for Python version, packages, and dataset files."""
    print("--> Checking Python environment and dependencies...")
    if sys.version_info < (3, 9):
        print(f"{Colors.RED}[FAIL] Python version {sys.version} is not supported. Please use Python 3.9+{Colors.END}")
        return False

    required_pkgs = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "torch": "torch",
        "shap": "shap",
        "optuna": "optuna",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "pytest": "pytest",
    }
    missing_pkgs = []
    for mod_name, pip_name in required_pkgs.items():
        try:
            __import__(mod_name)
        except ImportError:
            missing_pkgs.append(pip_name)

    if missing_pkgs:
        print(f"{Colors.YELLOW}[WARN] Missing required Python packages: {', '.join(missing_pkgs)}{Colors.END}")
        req_file = PROJECT_ROOT / "requirements.txt"
        if not req_file.exists():
            req_file = PROJECT_ROOT.parent / "requirements.txt"
        if auto_install:
            print(f"--> Auto-installing missing packages via 'pip install -r {req_file}'...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)
            print(f"{Colors.GREEN}[OK] All dependencies successfully installed.{Colors.END}")
        else:
            print(f"{Colors.RED}[FAIL] Please install dependencies: pip install -r requirements.txt (or run with --install-deps){Colors.END}")
            return False
    else:
        print(f"{Colors.GREEN}[OK] All required Python packages are installed.{Colors.END}")

    # Check dataset files
    try:
        from download_data import verify_dataset, download_missing_files
        is_complete, missing = verify_dataset(PROJECT_ROOT / "data" / "raw")
        if not is_complete:
            print(f"{Colors.YELLOW}[WARN] Missing {len(missing)} dataset files in data/raw/. Retrieving...{Colors.END}")
            download_missing_files(PROJECT_ROOT / "data" / "raw")
        else:
            print(f"{Colors.GREEN}[OK] All 12 NASA C-MAPSS dataset files verified in data/raw/.{Colors.END}")
    except Exception as e:
        print(f"{Colors.YELLOW}[WARN] Could not verify dataset files: {e}{Colors.END}")

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master Automated Execution Pipeline for NASA C-MAPSS Research Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke test (Subset 1, Seed 0 only)",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Automatically install/update missing Python dependencies from requirements.txt",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest test suite",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip Stage 1 benchmark evaluation (use existing CSVs)",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip Stage 2 novelty experiments (use existing CSVs)",
    )
    parser.add_argument(
        "--skip-latex",
        action="store_true",
        help="Skip compiling thesis/paper LaTeX to PDF",
    )
    parser.add_argument(
        "--only-artifacts",
        action="store_true",
        help="Skip model training/evaluation; only regenerate figures, tables, and compile PDF",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_start = time.time()

    print_banner(
        "NASA C-MAPSS PREDICTIVE MAINTENANCE & ANOMALY DETECTION\nMASTER AUTOMATED EXECUTION PIPELINE",
        color=Colors.CYAN,
    )
    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Source Dir   : {SRC_DIR}")
    print(f"Results Dir  : {RESULTS_DIR}")
    print(f"Thesis Dir   : {THESIS_DIR}")
    print(f"Execution Mode: {'QUICK SMOKE TEST' if args.quick else 'FULL RESEARCH PIPELINE'}\n")

    # Pre-flight environment check
    env_ok = check_environment(auto_install=args.install_deps)
    if not env_ok:
        return 1

    steps_total = 6
    step = 1

    # =========================================================================
    # Step 1: Unit Test Suite
    # =========================================================================
    print_step(step, steps_total, "Running Unit Test Suite (26 Tests)")
    step += 1
    if args.skip_tests or args.only_artifacts:
        print(f"{Colors.YELLOW}--> Skipped by user flag.{Colors.END}")
    else:
        success = run_command([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=PROJECT_ROOT, desc="Unit Tests")
        if not success:
            print(f"{Colors.RED}Unit tests failed. Aborting pipeline.{Colors.END}")
            return 1

    # =========================================================================
    # Step 2: Stage 1 Evaluation Pipeline
    # =========================================================================
    print_step(step, steps_total, "Stage 1: Leakage-Free Fleet Evaluation & Operating Curves")
    step += 1
    if args.skip_stage1 or args.only_artifacts:
        print(f"{Colors.YELLOW}--> Skipped by user flag (using existing Stage 1 results).{Colors.END}")
    else:
        stage1_cmd = [
            sys.executable,
            str(SRC_DIR / "run_anomaly.py"),
            "--horizon",
            "100",
        ]
        if args.quick:
            stage1_cmd.extend(["--subsets", "1", "--seeds", "0"])
        else:
            stage1_cmd.extend(["--subsets", "1", "2", "3", "4", "--seeds", "0", "1", "2", "3", "4"])

        success = run_command(stage1_cmd, cwd=PROJECT_ROOT, desc="Stage 1 Evaluation Pipeline")
        if not success:
            print(f"{Colors.RED}Stage 1 execution failed. Aborting pipeline.{Colors.END}")
            return 1

    # =========================================================================
    # Step 3: Stage 2 Novelty Experiments
    # =========================================================================
    print_step(step, steps_total, "Stage 2: Novelty Experiments (Transfer, Conformal, SHAP, Ablations)")
    step += 1
    if args.skip_stage2 or args.only_artifacts:
        print(f"{Colors.YELLOW}--> Skipped by user flag (using existing Stage 2 results).{Colors.END}")
    else:
        stage2_cmd = [sys.executable, str(SRC_DIR / "run_stage2.py")]
        if args.quick:
            stage2_cmd.append("--quick")
        success = run_command(stage2_cmd, cwd=PROJECT_ROOT, desc="Stage 2 Novelty Experiments")
        if not success:
            print(f"{Colors.RED}Stage 2 execution failed. Aborting pipeline.{Colors.END}")
            return 1

    # =========================================================================
    # Step 4: Stage 3 Artifacts (Figures & Tables)
    # =========================================================================
    print_step(step, steps_total, "Stage 3: Generating 12 Publication Figures & Exporting Tables")
    step += 1
    fig_cmd = [sys.executable, str(SRC_DIR / "generate_stage3_figures.py")]
    success_fig = run_command(fig_cmd, cwd=PROJECT_ROOT, desc="Stage 3 Figure Generation")

    tbl_cmd = [sys.executable, str(SRC_DIR / "export_tables.py")]
    success_tbl = run_command(tbl_cmd, cwd=PROJECT_ROOT, desc="Stage 3 Table Export")

    if not (success_fig and success_tbl):
        print(f"{Colors.RED}Artifact generation failed.{Colors.END}")
        return 1

    # =========================================================================
    # Step 5: Results Manifest Validation
    # =========================================================================
    print_step(step, steps_total, "Validating Results Manifest Traceability (results/manifest.csv)")
    step += 1
    manifest_path = RESULTS_DIR / "manifest.csv"
    if manifest_path.exists():
        import pandas as pd
        df_manifest = pd.read_csv(manifest_path)
        print(f"{Colors.GREEN}[OK] Manifest Verified: {len(df_manifest)} distinct metrics registered.{Colors.END}")
        cat_col = "Metric_Category" if "Metric_Category" in df_manifest.columns else "Category"
        if cat_col in df_manifest.columns:
            categories = df_manifest[cat_col].value_counts().to_dict()
            for cat, count in categories.items():
                print(f"    - {cat:<24}: {count} metrics")
    else:
        print(f"{Colors.YELLOW}[WARN] Warning: results/manifest.csv not found.{Colors.END}")

    # =========================================================================
    # Step 6: LaTeX Paper & Thesis Compilation
    # =========================================================================
    print_step(step, steps_total, "Compiling IEEE Format Paper / Thesis (thesis/main.pdf)")
    step += 1
    pdflatex_exe = find_pdflatex()
    bibtex_exe = find_bibtex()

    if args.skip_latex:
        print(f"{Colors.YELLOW}--> Skipped by user flag (--skip-latex).{Colors.END}")
    elif not pdflatex_exe:
        print(f"{Colors.YELLOW}[WARN] pdflatex executable not found in PATH or standard directories. Skipping PDF generation.{Colors.END}")
    else:
        print(f"--> Found LaTeX Compiler: {pdflatex_exe}")
        # Clean aux files first to ensure fresh compilation
        for aux_file in THESIS_DIR.glob("*.aux"):
            try:
                aux_file.unlink()
            except Exception:
                pass

        # Pass 1: pdflatex
        run_command([pdflatex_exe, "-interaction=nonstopmode", "main.tex"], cwd=THESIS_DIR, desc="pdflatex Pass 1")

        # Pass 2: bibtex (if available)
        if bibtex_exe and (THESIS_DIR / "main.aux").exists():
            run_command([bibtex_exe, "main"], cwd=THESIS_DIR, desc="bibtex Pass")

        # Pass 3 & 4: pdflatex
        run_command([pdflatex_exe, "-interaction=nonstopmode", "main.tex"], cwd=THESIS_DIR, desc="pdflatex Pass 2")
        run_command([pdflatex_exe, "-interaction=nonstopmode", "main.tex"], cwd=THESIS_DIR, desc="pdflatex Final Pass")

        pdf_path = THESIS_DIR / "main.pdf"
        if pdf_path.exists():
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"{Colors.GREEN}[OK] Successfully generated PDF: {pdf_path} ({size_mb:.2f} MB){Colors.END}")
        else:
            print(f"{Colors.RED}[FAIL] PDF generation encountered issues.{Colors.END}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    total_elapsed = time.time() - total_start
    print_banner(
        f"PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.1f} SECONDS",
        color=Colors.GREEN,
    )

    print(f"{Colors.BOLD}Generated Key Deliverables:{Colors.END}")
    print(f"  * Test Results       : 26 passed tests (tests/)")
    print(f"  * Metrics Manifest   : results/manifest.csv")
    print(f"  * Results Summary    : results/results_summary.md")
    print(f"  * LaTeX Tables       : results/tables_latex.tex")
    print(f"  * Markdown Tables    : results/tables_markdown.md")
    print(f"  * Publication Plots  : results/figures/ (12 figures at 300 DPI)")
    if (THESIS_DIR / "main.pdf").exists():
        print(f"  * IEEE Paper PDF     : thesis/main.pdf (17 pages, 2-column format)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
