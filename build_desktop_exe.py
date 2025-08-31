# HRM-Gemini Desktop Application Build Script
# Creates a standalone executable with all dependencies

import os
import sys
import shutil
from pathlib import Path

def create_desktop_shortcut():
    """Create desktop shortcut for the application"""
    desktop_path = Path.home() / "Desktop"
    shortcut_name = "HRM-Gemini AI.lnk"

    # PowerShell script to create shortcut
    ps_script = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{desktop_path / shortcut_name}")
$Shortcut.TargetPath = "{Path.cwd() / "dist" / "HRM-Gemini AI.exe"}"
$Shortcut.WorkingDirectory = "{Path.cwd() / "dist"}"
$Shortcut.Description = "HRM-Gemini AI Desktop Application - Advanced AI with Memory and RPG"
$Shortcut.Save()
'''

    ps_file = Path.cwd() / "create_shortcut.ps1"
    with open(ps_file, 'w') as f:
        f.write(ps_script)

    print(f"Created shortcut creation script: {ps_file}")

def create_installer_config():
    """Create installer configuration"""
    installer_config = f'''
[Application]
Name=HRM-Gemini AI
Version=1.0.0
Publisher=HRM Systems
Description=Advanced AI desktop application with memory management and RPG capabilities

[Files]
Source: "dist\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{{commondesktop}}\\HRM-Gemini AI"; Filename: "{{app}}\\HRM-Gemini AI.exe"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\HRM-Gemini AI.exe"; Description: "Launch HRM-Gemini AI"; Flags: nowait postinstall

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if WizardIsTaskSelected('desktopicon') then
    begin
      // Desktop icon creation is handled automatically
    end;
  end;
end;
'''

    iss_file = Path.cwd() / "installer.iss"
    with open(iss_file, 'w') as f:
        f.write(installer_config)

    print(f"Created installer config: {iss_file}")

def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building HRM-Gemini AI Desktop Executable...")
    print("="*60)

    # Ensure required packages are available
    required_packages = [
        'tkinter', 'sqlite3', 'json', 'pathlib', 'time', 'threading',
        'subprocess', 'webbrowser', 'tempfile', 'shutil', 'uuid',
        'datetime', 'random', 'hashlib', 'base64', 'mimetypes'
    ]

    print("📦 Checking required packages...")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        for package in missing_packages:
            os.system(f"pip install {package}")

    # Create dist directory if it doesn't exist
    dist_dir = Path.cwd() / "dist"
    dist_dir.mkdir(exist_ok=True)

    # Build command
    build_cmd = [
        sys.executable, "-m", "pyinstaller",
        "--onefile",  # Single executable file
        "--windowed",  # No console window
        "--name=HRM-Gemini AI",
        "--icon=icon.ico" if Path("icon.ico").exists() else None,
        "--add-data", f"{Path.cwd() / 'config'};config",
        "--add-data", f"{Path.cwd() / 'models'};models",
        "--add-data", f"{Path.cwd() / 'brain_memory'};brain_memory",
        "--hidden-import", "tkinter",
        "--hidden-import", "sqlite3",
        "--hidden-import", "pathlib",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "google.generativeai",
        "--hidden-import", "hrm_memory_system",
        "--hidden-import", "file_upload_system",
        "--hidden-import", "rpg_chatbot",
        "--hidden-import", "performance_monitor",
        "--hidden-import", "config",
        "hrm_desktop_app.py"
    ]

    # Remove None values
    build_cmd = [cmd for cmd in build_cmd if cmd is not None]

    print(f"\n🚀 Executing build command:")
    print(" ".join(build_cmd))
    print("\n⏳ Building executable... This may take a few minutes...")

    # Execute build
    result = os.system(" ".join(build_cmd))

    if result == 0:
        exe_path = dist_dir / "HRM-Gemini AI.exe"
        if exe_path.exists():
            file_size = exe_path.stat().st_size / (1024 * 1024)  # Size in MB

            print("
✅ Build successful!"            print(f"📁 Executable created: {exe_path}")
            print(".2f"
            # Create desktop shortcut
            create_desktop_shortcut()

            # Create installer config
            create_installer_config()

            print("
📋 Next steps:"            print(f"   1. Run the executable: {exe_path}")
            print("   2. Or run create_shortcut.ps1 as Administrator to create desktop icon"
            print("   3. Optional: Use installer.iss with Inno Setup for professional installer"

            return True
        else:
            print("❌ Build completed but executable not found")
            return False
    else:
        print(f"❌ Build failed with exit code: {result}")
        return False

def create_run_script():
    """Create a simple run script for testing"""
    run_script = '''@echo off
REM HRM-Gemini AI Desktop Application Launcher

echo ========================================
echo    HRM-Gemini AI Desktop Application
echo ========================================
echo.

REM Check if executable exists
if exist "dist\\HRM-Gemini AI.exe" (
    echo Starting HRM-Gemini AI...
    start "" "dist\\HRM-Gemini AI.exe"
) else (
    echo Executable not found. Please run build script first.
    echo.
    pause
    exit /b 1
)

echo Application launched successfully!
pause
'''

    with open("run_desktop_app.bat", 'w') as f:
        f.write(run_script)

    print("✅ Created run script: run_desktop_app.bat")

def main():
    """Main build function"""
    print("🏗️  HRM-GEMINI AI DESKTOP APPLICATION BUILDER")
    print("="*60)

    # Check if main app exists
    if not Path("hrm_desktop_app.py").exists():
        print("❌ Error: hrm_desktop_app.py not found!")
        return False

    # Create run script
    create_run_script()

    # Build executable
    success = build_executable()

    if success:
        print("\n🎉 BUILD COMPLETE!")
        print("="*60)
        print("Your HRM-Gemini AI Desktop Application is ready!")
        print("\n🚀 To run the application:")
        print("   1. Double-click: run_desktop_app.bat")
        print("   2. Or directly run: dist/HRM-Gemini AI.exe")
        print("   3. Desktop shortcut: Run create_shortcut.ps1")
        print("\n📱 Features included:")
        print("   • Complete GUI with multiple modes")
        print("   • Google Gemini API integration")
        print("   • File upload and processing")
        print("   • Memory management system")
        print("   • RPG chatbot functionality")
        print("   • Performance monitoring")
        print("   • System tray integration")

        return True
    else:
        print("\n❌ BUILD FAILED")
        print("="*60)
        print("Troubleshooting tips:")
        print("• Ensure all dependencies are installed")
        print("• Check Python version (3.8+)")
        print("• Verify Google Gemini API key is configured")
        print("• Try running as Administrator")
        print("• Check antivirus/firewall settings")

        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
