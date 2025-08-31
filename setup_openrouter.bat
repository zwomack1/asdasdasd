@echo off
setlocal

:: Initialize logging
set "LOG_FILE=setup_openrouter.log"
echo [%date% %time%] Starting setup > %LOG_FILE%

:: OpenRouter Integration Setup Script
echo Setting up OpenRouter integration...
echo [%date% %time%] Setting up OpenRouter integration >> %LOG_FILE%

:: 1. Set API Key
set /p api_key="Enter your OpenRouter API key: "
setx OPENROUTER_API_KEY "%api_key%" /M
echo API key set in system environment variables.
echo [%date% %time%] OPENROUTER_API_KEY set >> %LOG_FILE%

:: 2. Set Default Provider
setx HRM_PROVIDER "openrouter" /M
echo Default provider set to OpenRouter.
echo [%date% %time%] HRM_PROVIDER set to openrouter >> %LOG_FILE%

:: 3. Verify Setup
echo [%date% %time%] Verifying environment settings >> %LOG_FILE%
python -c "import os; print(f'OPENROUTER_API_KEY: {os.getenv(\'OPENROUTER_API_KEY\')}'); print(f'HRM_PROVIDER: {os.getenv(\'HRM_PROVIDER\')}')"

echo.
echo Setup complete! Run your application normally.
echo [%date% %time%] Setup completed successfully >> %LOG_FILE%
echo Setup complete. Please restart your command prompt.
echo Note: Please close and reopen this command prompt for the changes to take effect.

endlocal
