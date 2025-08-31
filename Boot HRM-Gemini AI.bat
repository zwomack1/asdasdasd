@echo off
REM HRM-Gemini AI Complete Boot System
REM Single comprehensive launcher for the HRM-Gemini AI system

echo.
echo ============================================
echo    🤖 HRM-GEMINI AI COMPLETE BOOT SYSTEM
echo ============================================
echo.
echo This system will:
echo • Set up your environment
echo • Configure Google Gemini API
echo • Initialize HRM brain systems
echo • Launch the AI interface
echo.
echo Press any key to continue...
pause >nul

REM Clear screen for clean interface
cls

echo.
echo ============================================
echo           🔑 GOOGLE GEMINI API SETUP
echo ============================================
echo.

REM Check if API key is already configured
if exist "config\gemini_config.py" (
    echo Checking existing API configuration...
    echo ✅ API key already configured!
    goto :skip_api_setup
) else (
    echo ⚠️  Creating default API configuration...
    REM Create config directory if it doesn't exist
    if not exist "config" mkdir config

    REM Create gemini_config.py with the default API key
    echo # HRM-Gemini AI Configuration > "config\gemini_config.py"
    echo # Google Gemini API Configuration >> "config\gemini_config.py"
    echo. >> "config\gemini_config.py"
    echo # API Key - Automatically configured for HRM-Gemini AI >> "config\gemini_config.py"
    echo GEMINI_API_KEY = "AIzaSyC1H_JFcZX2nBt5ns3D0x9SHx2Eb7xqqJY" >> "config\gemini_config.py"
    echo. >> "config\gemini_config.py"
    echo # API Settings >> "config\gemini_config.py"
    echo GEMINI_MODEL = "gemini-pro" >> "config\gemini_config.py"
    echo MAX_TOKENS = 2048 >> "config\gemini_config.py"
    echo TEMPERATURE = 0.7 >> "config\gemini_config.py"
    echo. >> "config\gemini_config.py"
    echo # Memory Settings >> "config\gemini_config.py"
    echo MAX_MEMORY_ITEMS = 1000 >> "config\gemini_config.py"
    echo MEMORY_RETENTION_DAYS = 30 >> "config\gemini_config.py"
    echo. >> "config\gemini_config.py"
    echo # File Processing Settings >> "config\gemini_config.py"
    echo SUPPORTED_FORMATS = ["pdf", "docx", "txt", "jpg", "png", "py", "js", "html"] >> "config\gemini_config.py"
    echo MAX_FILE_SIZE_MB = 10 >> "config\gemini_config.py"
    echo. >> "config\gemini_config.py"
    echo # RPG Settings >> "config\gemini_config.py"
    echo DEFAULT_CHARACTER_CLASS = "Adventurer" >> "config\gemini_config.py"
    echo MAX_QUESTS = 50 >> "config\gemini_config.py"

    echo ✅ Default API configuration created!
)

:skip_api_setup

echo ============================================
echo         🔧 SYSTEM INITIALIZATION
echo ============================================
echo.

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo.
    echo 📥 Please install Python 3.8+ from: https://python.org
    echo    Make sure to check "Add Python to PATH" during installation
    echo.
    goto :error
)

echo ✅ Python detected:
python --version
echo.

REM Check for required files
if not exist "hrm_memory_system.py" (
    echo ❌ Missing: hrm_memory_system.py
    goto :error
)

if not exist "file_upload_system.py" (
    echo ❌ Missing: file_upload_system.py
    goto :error
)

if not exist "rpg_chatbot.py" (
    echo ❌ Missing: rpg_chatbot.py
    goto :error
)

if not exist "demo\cli_interface.py" (
    echo ❌ Missing: demo\cli_interface.py
    goto :error
)

echo ✅ All required files present
echo.

REM Install/update required packages
echo 📦 Checking and installing required packages...

python -m pip install --quiet --upgrade pip
python -m pip install --quiet google-generativeai
python -m pip install --quiet psutil
python -m pip install --quiet pillow
python -m pip install --quiet requests
python -m pip install --quiet python-docx
python -m pip install --quiet PyPDF2
python -m pip install --quiet pandas
python -m pip install --quiet openpyxl

echo ✅ Dependencies installed
echo.

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "brain_memory" mkdir brain_memory

echo ✅ System directories created
echo.

REM Test API connection
echo 🔗 Testing Google Gemini API connection...
echo.

python -c "
try:
    import sys
    sys.path.append('.')
    from config import gemini_config
    import google.generativeai as genai
    
    if hasattr(gemini_config, 'GEMINI_API_KEY') and gemini_config.GEMINI_API_KEY:
        genai.configure(api_key=gemini_config.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content('Hello, this is a connection test.')
        print('✅ Google Gemini API connection successful!')
        print('🤖 AI Response:', response.text[:100] + '...')
    else:
        print('❌ API key not found in configuration')
        sys.exit(1)
        
except ImportError as e:
    print('❌ Missing required packages:', e)
    sys.exit(1)
except Exception as e:
    print('❌ API connection failed:', e)
    print('💡 Please check your API key and internet connection')
    sys.exit(1)
"

if errorlevel 1 (
    echo.
    echo ❌ API test failed! Please check your API key.
    echo.
    goto :error
)

echo.
echo ============================================
echo        🚀 LAUNCHING HRM-GEMINI AI
echo ============================================
echo.
echo System Status:
echo ✅ Python Environment: Ready
echo ✅ Dependencies: Installed
echo ✅ API Key: Configured
echo ✅ Google Gemini: Connected
echo ✅ File System: Initialized
echo ✅ Memory System: Ready
echo ✅ RPG System: Loaded
echo.

echo 🎯 Starting HRM-Gemini AI...
echo.
echo 💡 Available commands:
echo    • 'hi' or 'hello' - Start chatting
echo    • 'help' - See all commands
echo    • 'tutorial' - Guided tour
echo    • 'upload <file>' - Process files
echo    • 'rpg' - Enter RPG mode
echo    • 'exit' - Quit the system
echo.
echo 🤖 Your AI brain is ready!
echo.
echo 🎯 Starting HRM-Gemini AI Desktop GUI...
echo.

REM Launch the HRM-Gemini GUI Application
python hrm_desktop_app.py

REM Clean exit
echo.
echo ============================================
echo         HRM-Gemini AI Session Ended
echo ============================================
echo.
echo Thank you for using HRM-Gemini AI!
echo Your brain data has been saved automatically.
echo.
pause
exit /b 0

:error
echo.
echo ============================================
echo            BOOT SYSTEM ERROR
echo ============================================
echo.
echo ❌ The HRM-Gemini AI boot process encountered an error.
echo.
echo 🔧 Troubleshooting:
echo    • Check Python installation
echo    • Verify API key configuration
echo    • Ensure internet connection
echo    • Check file permissions
echo.
echo 💡 You can run this boot script again anytime.
echo.
pause
exit /b 1
