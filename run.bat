@echo off
setlocal enabledelayedexpansion
REM =========================================================================
REM NASA C-MAPSS Research Project - Quick Run Launcher (Windows)
REM Double-click or run from command line: run.bat [options]
REM =========================================================================

echo ========================================================================
echo  NASA C-MAPSS PREDICTIVE MAINTENANCE AND ANOMALY DETECTION
echo  Autonomous Benchmark & Execution Launcher
echo ========================================================================
echo.

REM 1. Detect Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    py --version >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Python is not installed or not in your system PATH.
        echo Please download and install Python 3.9+ from https://python.org
        echo Make sure to check "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
    set PY_CMD=py
) else (
    set PY_CMD=python
)

REM 2. Check if arguments passed on command line
if not "%~1"=="" (
    echo Executing: %PY_CMD% "%~dp0run_all.py" %*
    %PY_CMD% "%~dp0run_all.py" %*
    goto :POST_RUN
)

REM 3. Interactive Menu for Double-Click Users
echo Select an Execution Mode:
echo   [1] Fast Smoke Test (~30s, runs all pipeline stages on FD001) [RECOMMENDED]
echo   [2] Full Research Benchmark (All 4 datasets, all models, 5 seeds)
echo   [3] Instant Demonstration (Regenerates 12 plots & tables in 10s from results)
echo   [4] Run Unit Test Suite Only (26 tests via pytest)
echo   [5] Install / Upgrade Dependencies (pip install -r requirements.txt)
echo   [6] Exit
echo.
set /p CHOICE="Enter option [1-6, default=1]: "

if "%CHOICE%"=="" set CHOICE=1
if "%CHOICE%"=="1" (
    echo.
    echo --> Launching Fast Smoke Test...
    %PY_CMD% "%~dp0run_all.py" --quick --skip-latex
    goto :POST_RUN
)
if "%CHOICE%"=="2" (
    echo.
    echo --> Launching Full Research Benchmark...
    %PY_CMD% "%~dp0run_all.py"
    goto :POST_RUN
)
if "%CHOICE%"=="3" (
    echo.
    echo --> Generating Demonstration Artifacts...
    %PY_CMD% "%~dp0run_all.py" --only-artifacts --skip-latex
    goto :POST_RUN
)
if "%CHOICE%"=="4" (
    echo.
    echo --> Running Unit Test Suite...
    %PY_CMD% -m pytest "%~dp0project\tests" -v
    goto :POST_RUN
)
if "%CHOICE%"=="5" (
    echo.
    echo --> Installing Dependencies from requirements.txt...
    %PY_CMD% -m pip install -r "%~dp0requirements.txt"
    goto :POST_RUN
)
if "%CHOICE%"=="6" (
    exit /b 0
)

echo.
echo [WARN] Unrecognized option "%CHOICE%". Launching Fast Smoke Test as default...
%PY_CMD% "%~dp0run_all.py" --quick --skip-latex

:POST_RUN
set EXIT_CODE=%ERRORLEVEL%
echo.
if %EXIT_CODE% NEQ 0 (
    echo ========================================================================
    echo  [ERROR] Execution encountered an issue (Exit Code: %EXIT_CODE%).
    echo ========================================================================
) else (
    echo ========================================================================
    echo  [SUCCESS] Pipeline execution finished cleanly!
    echo  Key deliverables are available in:
    echo    - Results & Metrics : project\results\
    echo    - Figures (300 DPI) : project\results\figures\
    echo    - Paper / Thesis    : project\thesis\main.pdf
    echo ========================================================================
)
echo.
pause
exit /b %EXIT_CODE%
