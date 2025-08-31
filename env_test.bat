@echo off
echo Testing environment...
echo.
echo === Python Test ===
python --version
echo.
echo === PIP Test ===
pip --version
echo.
echo === PATH Test ===
echo %PATH%
echo.
echo === Python Modules Test ===
python -c "import sys; print(sys.path)"
pause
