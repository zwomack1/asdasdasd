#!/usr/bin/env python3
"""
Build script for HRM-Gemini AI System executable
"""

import os
import sys
import subprocess
from pathlib import Path

def create_spec_file():
    """Create PyInstaller spec file for HRM-Gemini"""
    spec_content = '''
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['demo/cli_interface.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('models', 'models'),
        ('services', 'services'),
        ('controllers', 'controllers'),
        ('README.md', '.'),
    ],
    hiddenimports=[
        'sys',
        'pathlib',
        'threading',
        'subprocess',
        'importlib',
        'pkg_resources',
        'google',
        'google.generativeai',
        'torch',
        'torch.nn',
        'numpy',
        'transformers',
        'sklearn',
        'scipy',
        'matplotlib',
        'seaborn',
        'pandas',
        'requests',
        'beautifulsoup4',
        'lxml',
        'openpyxl',
        'xlrd',
        'xlwt',
        'xlsxwriter',
        'pydantic',
        'typing_extensions',
        'colorama',
        'tqdm',
        'click',
        'rich',
        'prompt_toolkit',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'seaborn',
        'plotly',
        'bokeh',
        'altair',
        'holoviews',
        'panel',
        'streamlit',
        'dash',
        'gradio',
        'flask',
        'django',
        'fastapi',
        'uvicorn',
        'gunicorn',
        'celery',
        'redis',
        'pymongo',
        'sqlalchemy',
        'psycopg2',
        'pymysql',
        'sqlite3',
        'pymssql',
        'cx_Oracle',
        'pyodbc',
        'pypyodbc',
        'mysqlclient',
        'mariadb',
        'pymssql',
        'pyodbc',
        'teradata',
        'ibm_db',
        'pyhdb',
        'pysap',
        'pymqi',
        'pymqi',
        'pyodbc',
        'teradata',
        'ibm_db',
        'pyhdb',
        'pysap',
        'pymqi',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HRM-Gemini-AI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''

    with open('hrm_gemini.spec', 'w') as f:
        f.write(spec_content.strip())
    print("✅ Created PyInstaller spec file: hrm_gemini.spec")

def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building HRM-Gemini executable...")

    # Method 1: Use the spec file
    try:
        print("📦 Building with spec file...")
        result = subprocess.run([
            sys.executable, '-m', 'pyinstaller',
            '--clean',
            'hrm_gemini.spec'
        ], capture_output=True, text=True, cwd='.')

        if result.returncode == 0:
            print("✅ Executable built successfully!")
            print("📁 Executable location: dist/HRM-Gemini-AI.exe")
            return True
        else:
            print(f"❌ Spec file build failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Spec file build error: {e}")

        # Method 2: Direct PyInstaller command
        try:
            print("🔄 Trying direct PyInstaller command...")
            result = subprocess.run([
                sys.executable, '-m', 'pyinstaller',
                '--onefile',
                '--windowed',
                '--name=HRM-Gemini-AI',
                '--add-data=config;config',
                '--add-data=models;models',
                '--add-data=services;services',
                '--add-data=controllers;controllers',
                '--add-data=README.md;.',
                '--hidden-import=sys',
                '--hidden-import=pathlib',
                '--hidden-import=threading',
                '--hidden-import=subprocess',
                '--hidden-import=importlib',
                '--hidden-import=pkg_resources',
                'demo/cli_interface.py'
            ], capture_output=True, text=True, cwd='.')

            if result.returncode == 0:
                print("✅ Executable built successfully with direct command!")
                print("📁 Executable location: dist/HRM-Gemini-AI.exe")
                return True
            else:
                print(f"❌ Direct build failed: {result.stderr}")
                return False

        except Exception as e2:
            print(f"❌ Direct build error: {e2}")
            return False

def create_batch_launcher():
    """Create a batch file to launch the executable"""
    batch_content = '''@echo off
echo Starting HRM-Gemini AI System...
echo.
echo If you see any errors, make sure you have:
echo 1. Google Cloud credentials configured
echo 2. All required dependencies installed
echo 3. Python environment properly set up
echo.
pause
'''

    with open('run_hrm_gemini.bat', 'w') as f:
        f.write(batch_content)
    print("✅ Created batch launcher: run_hrm_gemini.bat")

def main():
    """Main build function"""
    print("🚀 HRM-Gemini AI System - Executable Builder")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path('demo/cli_interface.py').exists():
        print("❌ Error: demo/cli_interface.py not found!")
        print("Please run this script from the project root directory.")
        return False

    # Create spec file
    create_spec_file()

    # Build executable
    success = build_executable()

    if success:
        # Create launcher batch file
        create_batch_launcher()

        print("\n🎉 Build completed successfully!")
        print("📁 Files created:")
        print("   - dist/HRM-Gemini-AI.exe (Main executable)")
        print("   - hrm_gemini.spec (PyInstaller spec file)")
        print("   - run_hrm_gemini.bat (Launcher batch file)")
        print("\n💡 To run the application:")
        print("   - Double-click 'run_hrm_gemini.bat'")
        print("   - Or run 'dist/HRM-Gemini-AI.exe' directly")
        return True
    else:
        print("\n❌ Build failed!")
        print("💡 Troubleshooting tips:")
        print("   - Make sure PyInstaller is installed: pip install pyinstaller")
        print("   - Check that all dependencies are installed")
        print("   - Try running as administrator")
        print("   - Check the PyInstaller documentation for Windows-specific issues")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
