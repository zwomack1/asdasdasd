@echo off
REM Create Desktop Shortcut for HRM-Gemini AI System

echo Creating desktop shortcut for HRM-Gemini AI System...

REM Get the current directory and desktop path
set "CURRENT_DIR=%~dp0"
set "DESKTOP_DIR=%USERPROFILE%\Desktop"

REM Create the shortcut using PowerShell
powershell "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%DESKTOP_DIR%\HRM-Gemini AI.lnk'); $s.TargetPath = '%CURRENT_DIR%Run HRM-Gemini AI.bat'; $s.WorkingDirectory = '%CURRENT_DIR%'; $s.Description = 'HRM-Gemini AI System - Intelligent Code Generation and Management'; $s.Save()"

echo Desktop shortcut created successfully!
echo You can now double-click "HRM-Gemini AI.lnk" on your desktop to launch the system.
pause
