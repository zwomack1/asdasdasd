# Test-OpenRouter.ps1
# PowerShell script to test OpenRouter API integration

# Set up error handling
$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

# Create log file
$logFile = "$PSScriptRoot\openrouter_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Start-Transcript -Path $logFile -Force

Write-Host "=== OpenRouter Integration Test ===" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Current directory: $PSScriptRoot"
Write-Host ""

# Function to write status messages
function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    $timestamp = Get-Date -Format "HH:mm:ss"
    $statusColor = switch ($Status) {
        "SUCCESS" { "Green" }
        "ERROR"   { "Red" }
        "WARNING" { "Yellow" }
        default    { "Cyan" }
    }
    Write-Host "[$timestamp] [$Status]" -NoNewline -ForegroundColor $statusColor
    Write-Host " $Message"
}

try {
    # Test Python installation
    Write-Status "Checking Python installation..."
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python is not installed or not in PATH"
    }
    Write-Status "Python version: $pythonVersion" "SUCCESS"

    # Test Python modules
    $modules = @("requests", "json")
    foreach ($module in $modules) {
        Write-Status "Checking Python module: $module"
        $testCmd = "import $module; print('OK')"
        $result = python -c $testCmd 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Python module $module is not installed: $result"
        }
        Write-Status "Module $module is installed" "SUCCESS"
    }

    # Test internet connectivity
    Write-Status "Testing internet connectivity..."
    try {
        $testUrl = "https://www.google.com"
        $response = Invoke-WebRequest -Uri $testUrl -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
        Write-Status "Internet connection is working (Status: $($response.StatusCode))" "SUCCESS"
    } catch {
        Write-Status "Internet connection test failed: $_" "ERROR"
        throw "No internet connection or proxy issues detected"
    }

    # Test OpenRouter API
    Write-Status "Testing OpenRouter API..."
    # Get API key from environment
    $apiKey = $env:OPENROUTER_API_KEY
    if (-not $apiKey) {
        throw "OPENROUTER_API_KEY environment variable is not set"
    }
    $headers = @{
        "Authorization" = "Bearer $apiKey"
        "Content-Type" = "application/json"
    }

    # Test authentication
    Write-Status "Testing authentication..."
    $authUrl = "https://openrouter.ai/api/v1/auth/key"
    try {
        $authResponse = Invoke-RestMethod -Uri $authUrl -Headers $headers -Method Get -TimeoutSec 30 -ErrorAction Stop
        Write-Status "✅ OpenRouter authentication successful" "SUCCESS"
        Write-Host (ConvertTo-Json $authResponse -Depth 5) -ForegroundColor DarkGray
    } catch {
        $errorDetails = $_.Exception.Response
        $statusCode = $errorDetails.StatusCode.value__
        $statusDescription = $errorDetails.StatusDescription
        Write-Status "❌ OpenRouter authentication failed (Status: $statusCode $statusDescription)" "ERROR"
        if ($errorDetails) {
            $reader = New-Object System.IO.StreamReader($errorDetails.GetResponseStream())
            $reader.BaseStream.Position = 0
            $reader.DiscardBufferedData()
            $responseBody = $reader.ReadToEnd()
            Write-Host "Response: $responseBody" -ForegroundColor Red
        }
        throw "Authentication failed"
    }

    # Test chat completion
    Write-Status "Testing chat completion..."
    $chatUrl = "https://openrouter.ai/api/v1/chat/completions"
    $chatData = @{
        model = "kimi"
        messages = @(
            @{role = "user"; content = "Hello, how are you?"}
        )
    } | ConvertTo-Json

    try {
        $chatResponse = Invoke-RestMethod -Uri $chatUrl -Headers $headers -Method Post -Body $chatData -ContentType "application/json" -TimeoutSec 60 -ErrorAction Stop
        Write-Status "✅ Chat test successful" "SUCCESS"
        $reply = $chatResponse.choices[0].message.content
        Write-Host "🤖 Response: $reply" -ForegroundColor Green
    } catch {
        $errorDetails = $_.Exception.Response
        $statusCode = $errorDetails.StatusCode.value__
        $statusDescription = $errorDetails.StatusDescription
        Write-Status "❌ Chat test failed (Status: $statusCode $statusDescription)" "ERROR"
        if ($errorDetails) {
            $reader = New-Object System.IO.StreamReader($errorDetails.GetResponseStream())
            $reader.BaseStream.Position = 0
            $reader.DiscardBufferedData()
            $responseBody = $reader.ReadToEnd()
            Write-Host "Response: $responseBody" -ForegroundColor Red
        }
        throw "Chat test failed"
    }

    Write-Host "`n=== All tests completed successfully! ===" -ForegroundColor Green
} catch {
    Write-Status "TEST FAILED: $_" "ERROR"
    Write-Host "Error details: $($_.ScriptStackTrace)" -ForegroundColor Red
    exit 1
} finally {
    Stop-Transcript
    Write-Host "`nLog file created: $logFile" -ForegroundColor Cyan
}
