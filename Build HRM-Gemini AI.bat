@echo off
REM HRM-Gemini AI Desktop Application - Automated EXE Builder
REM This script automatically builds the HRM-Gemini AI desktop application into an executable

echo ============================================
echo    HRM-Gemini AI Desktop EXE Builder
echo ============================================
echo.
echo This script will automatically build HRM-Gemini AI
echo into a standalone executable (.exe) file.
echo.
echo Requirements:
echo - Python 3.8+
echo - PyInstaller
echo - All project dependencies
echo.
pause

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.8+ from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    goto :error
)

echo Python version:
python --version
echo.

REM Check if main app exists
if not exist "hrm_desktop_app.py" (
    echo ERROR: hrm_desktop_app.py not found!
    echo Please make sure you're running this from the project root directory.
    goto :error
)

echo Found main application file: hrm_desktop_app.py
echo.

REM Create build directory if it doesn't exist
if not exist "build" mkdir build
if not exist "dist" mkdir dist

echo 📦 Installing/Updating required packages...
echo.

REM Install PyInstaller if not available
python -m pip install --upgrade pip
python -m pip install pyinstaller --quiet

REM Install other required packages
python -m pip install tk tkinter ttkthemes google-generativeai pillow psutil --quiet

echo ✅ Dependencies installed
echo.

echo 🔨 Building executable...
echo This may take several minutes...
echo.

REM Build the executable with optimized settings
python -m pyinstaller ^
    --onefile ^
    --windowed ^
    --name "HRM-Gemini AI" ^
    --icon "icon.ico" ^
    --add-data "config;config" ^
    --add-data "models;models" ^
    --add-data "brain_memory;brain_memory" ^
    --hidden-import tkinter ^
    --hidden-import tkinter.ttk ^
    --hidden-import sqlite3 ^
    --hidden-import pathlib ^
    --hidden-import PIL ^
    --hidden-import PIL.Image ^
    --hidden-import google.generativeai ^
    --hidden-import hrm_memory_system ^
    --hidden-import file_upload_system ^
    --hidden-import rpg_chatbot ^
    --hidden-import performance_monitor ^
    --hidden-import config ^
    --clean ^
    --noconfirm ^
    hrm_desktop_app.py

REM Check if build was successful
if exist "dist\HRM-Gemini AI.exe" (
    echo.
    echo ============================================
    echo         BUILD SUCCESSFUL!
    echo ============================================
    echo.
    echo ✅ Executable created: dist\HRM-Gemini AI.exe
    echo.

    REM Get file size
    for %%A in ("dist\HRM-Gemini AI.exe") do set "size=%%~zA"
    set /a "size_mb=%size%/1024/1024"
    echo 📏 File size: %size_mb% MB
    echo.

    echo 🚀 To run the application:
    echo    Double-click: dist\HRM-Gemini AI.exe
    echo.

    REM Create desktop shortcut option
    echo 💡 Would you like to create a desktop shortcut?
    echo    Run: create_shortcut.bat (as Administrator)
    echo.

    goto :success
) else (
    echo.
    echo ============================================
    echo           BUILD FAILED!
    echo ============================================
    echo.
    echo ❌ Executable was not created.
    echo.
    echo Possible causes:
    echo - Missing dependencies
    echo - PyInstaller issues
    echo - Antivirus interference
    echo - Insufficient disk space
    echo.
    echo Check the error messages above for details.
    echo.
    goto :error
)

:success
echo ============================================
echo      HRM-Gemini AI Desktop App Ready!
echo ============================================
echo.
echo Your AI brain application has been successfully
echo packaged into a standalone executable.
echo.
echo Features included:
echo ✅ Complete GUI with multiple modes
echo ✅ Google Gemini API integration
echo ✅ Brain-like memory system
echo ✅ File upload and processing
echo ✅ RPG gaming functionality
echo ✅ Performance monitoring
echo ✅ Standalone executable (no installation needed)
echo.
echo Enjoy your intelligent AI companion! 🤖✨
echo.
pause
exit /b 0

:error
echo.
echo ============================================
echo            BUILD ERROR
echo ============================================
echo.
echo The build process encountered an error.
echo Please check the messages above and try again.
echo.
pause
exit /b 1
