#!/usr/bin/env bash
# =========================================================================
# NASA C-MAPSS Research Project - Quick Run Launcher (Linux / macOS)
# Usage: ./run.sh [options]
# =========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect Python 3
PYTHON_CMD="python3"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    PYTHON_CMD="python"
fi

if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not found in system PATH."
    echo "Please install Python 3.9+ from https://python.org or via your system package manager."
    exit 1
fi

# If command-line arguments are provided, pass directly to run_all.py
if [ $# -gt 0 ]; then
    "$PYTHON_CMD" "${SCRIPT_DIR}/run_all.py" "$@"
    exit $?
fi

# Interactive menu if run without arguments
echo "========================================================================"
echo " NASA C-MAPSS PREDICTIVE MAINTENANCE AND ANOMALY DETECTION"
echo " Autonomous Benchmark & Execution Launcher"
echo "========================================================================"
echo "Select an Execution Mode:"
echo "  [1] Fast Smoke Test (~30s, runs all pipeline stages on FD001) [RECOMMENDED]"
echo "  [2] Full Research Benchmark (All 4 datasets, all models, 5 seeds)"
echo "  [3] Instant Demonstration (Regenerates 12 plots & tables in 10s from results)"
echo "  [4] Run Unit Test Suite Only (26 tests via pytest)"
echo "  [5] Install / Upgrade Dependencies (pip install -r requirements.txt)"
echo "  [6] Exit"
echo ""
read -p "Enter option [1-6, default=1]: " CHOICE
CHOICE=${CHOICE:-1}

case "$CHOICE" in
    1)
        echo -e "\n--> Launching Fast Smoke Test..."
        "$PYTHON_CMD" "${SCRIPT_DIR}/run_all.py" --quick --skip-latex
        ;;
    2)
        echo -e "\n--> Launching Full Research Benchmark..."
        "$PYTHON_CMD" "${SCRIPT_DIR}/run_all.py"
        ;;
    3)
        echo -e "\n--> Generating Demonstration Artifacts..."
        "$PYTHON_CMD" "${SCRIPT_DIR}/run_all.py" --only-artifacts --skip-latex
        ;;
    4)
        echo -e "\n--> Running Unit Test Suite..."
        "$PYTHON_CMD" -m pytest "${SCRIPT_DIR}/project/tests" -v
        ;;
    5)
        echo -e "\n--> Installing Dependencies from requirements.txt..."
        "$PYTHON_CMD" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
        ;;
    6)
        exit 0
        ;;
    *)
        echo -e "\n[WARN] Unrecognized option '$CHOICE'. Launching Fast Smoke Test as default..."
        "$PYTHON_CMD" "${SCRIPT_DIR}/run_all.py" --quick --skip-latex
        ;;
esac
