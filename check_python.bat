@echo off
echo Testing Python environment...
where python
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH
    exit /b 1
)

echo.
echo Python version:
python --version

if %ERRORLEVEL% NEQ 0 (
    echo Failed to get Python version
    exit /b 1
)

echo.
echo Python path:
python -c "import sys; print('\n'.join(sys.path))"

if %ERRORLEVEL% NEQ 0 (
    echo Failed to get Python path
    exit /b 1
)

echo.
echo Testing simple Python command...
python -c "print('Python is working! 1+1 =', 1+1)"

if %ERRORLEVEL% NEQ 0 (
    echo Python command failed
    exit /b 1
)

echo.
echo Environment test completed successfully!
pause
