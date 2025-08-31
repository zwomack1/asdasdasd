@echo off
REM HRM-Gemini AI System Launcher
REM This batch file launches the HRM-Gemini AI system

echo ========================================
echo    HRM-Gemini AI System Launcher
echo ========================================
echo.
echo Starting HRM-Gemini AI System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.7+ from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Change to the script directory
cd /d "%~dp0"
set "OPENROUTER_API_KEY=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"

REM Check if the main script exists
if not exist "demo\cli_interface.py" (
    echo ERROR: demo\cli_interface.py not found!
    echo Please make sure you're running this from the project root directory.
    echo.
    pause
    exit /b 1
)

REM Display system info
echo Python version:
python --version
echo.
echo Working directory: %CD%
echo.

REM Verify environment variable
echo OPENROUTER_API_KEY is set to: %OPENROUTER_API_KEY%

REM Run the HRM-Gemini system
echo Launching HRM-Gemini AI System...
echo Type 'help' for available commands or 'exit' to quit
echo.
python demo/cli_interface.py

REM Keep the window open if there was an error
if errorlevel 1 (
    echo.
    echo ========================================
    echo         SYSTEM ENCOUNTERED AN ERROR
    echo ========================================
    echo.
    echo This might be due to:
    echo - Missing dependencies (run: pip install -r requirements.txt)
    echo - Google Cloud credentials not configured
    echo - Network connectivity issues
    echo - Insufficient permissions
    echo.
    echo Please check the error messages above for more details.
    echo.
    pause
)

echo.
echo HRM-Gemini AI System has exited.
pause
