@echo off
echo Setting up environment...
set PYTHONPATH=%CD%

:: Check Python installation
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not in your system PATH
    exit /b 1
)

:: Check Python version
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Failed to get Python version
    exit /b 1
)

echo.
echo Running test with Python...
python -c "import sys; print('Python path:', sys.path)"

echo.
echo Running OpenRouter test...
python test_openrouter_integration.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Test failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

pause
