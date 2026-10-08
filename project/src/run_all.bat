@echo off
REM =========================================================================
REM NASA C-MAPSS Research Project - Automated One-Click Pipeline Runner
REM =========================================================================

echo Starting automated pipeline...
python "%~dp0run_all.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Pipeline encountered an error (Code: %ERRORLEVEL%).
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Pipeline execution finished cleanly!
pause
