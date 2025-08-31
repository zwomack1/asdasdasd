# HRM-Gemini AI System Launcher (PowerShell)
# This script launches the HRM-Gemini AI system

param(
    [switch]$NoPause
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    HRM-Gemini AI System Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting HRM-Gemini AI System..." -ForegroundColor Green
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.7+ from: https://python.org" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Write-Host ""
    if (-not $NoPause) {
        Read-Host "Press Enter to exit"
    }
    exit 1
}

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Working directory: $scriptDir" -ForegroundColor Green
Write-Host ""

# Change to the script directory
Set-Location $scriptDir

# Check if virtual environment exists, create if not
if (-not (Test-Path .\venv\Scripts\python.exe)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created." -ForegroundColor Green
    Write-Host ""
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "Virtual environment activated." -ForegroundColor Green
Write-Host ""

# Set default LLM provider if not specified
if (-not $env:HRM_PROVIDER) {
    $env:HRM_PROVIDER = "llama"
    Write-Host "Using Llama as default LLM provider. Set HRM_PROVIDER=gemini for Gemini." -ForegroundColor Cyan
}

# Check for required dependencies
Write-Host "Checking dependencies..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -c "import bs4, numpy" >$null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing missing dependencies..." -ForegroundColor Yellow
    .\venv\Scripts\pip.exe install -r requirements.txt
    Write-Host "Dependencies installed." -ForegroundColor Green
} else {
    Write-Host "Dependencies OK." -ForegroundColor Green
}
Write-Host ""

# Check if the main script exists
$mainScript = Join-Path $scriptDir "demo\cli_interface.py"
if (-not (Test-Path $mainScript)) {
    Write-Host "ERROR: demo\cli_interface.py not found!" -ForegroundColor Red
    Write-Host "Please make sure you're running this from the project root directory." -ForegroundColor Yellow
    Write-Host ""
    if (-not $NoPause) {
        Read-Host "Press Enter to exit"
    }
    exit 1
}

# Check for requirements.txt and suggest installation
$requirementsFile = Join-Path $scriptDir "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "Checking dependencies..." -ForegroundColor Yellow
    try {
        python -c "import sys; print('Python path:'); [print(p) for p in sys.path]" | Out-Null
    } catch {
        Write-Host "Warning: Could not verify Python imports" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Launching HRM-Gemini AI System..." -ForegroundColor Green
Write-Host "Type 'help' for available commands or 'exit' to quit" -ForegroundColor Cyan
Write-Host ""

# Run the HRM-Gemini system
try {
    .\venv\Scripts\python.exe demo/cli_interface.py
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "         SYSTEM ENCOUNTERED AN ERROR" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "This might be due to:" -ForegroundColor Yellow
    Write-Host "- Missing dependencies (run: pip install -r requirements.txt)" -ForegroundColor Yellow
    Write-Host "- Google Cloud credentials not configured" -ForegroundColor Yellow
    Write-Host "- Network connectivity issues" -ForegroundColor Yellow
    Write-Host "- Insufficient permissions" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please check the error messages above for more details." -ForegroundColor Yellow
    Write-Host ""
    $exitCode = 1
}

Write-Host ""
Write-Host "HRM-Gemini AI System has exited." -ForegroundColor Cyan

if (-not $NoPause) {
    Read-Host "Press Enter to exit"
}

exit $exitCode
