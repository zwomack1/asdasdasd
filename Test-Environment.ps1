# Test-Environment.ps1
# Comprehensive environment test script

Write-Host "=== Environment Test ===" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/4] Checking Python..." -ForegroundColor Yellow
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($pythonPath) {
    Write-Host "✅ Python found at: $pythonPath" -ForegroundColor Green
    
    # Get Python version
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "   Version: $pythonVersion"
        
        # Check Python modules
        $modules = @("requests", "os", "sys", "json")
        Write-Host "`n   Checking Python modules..."
        foreach ($module in $modules) {
            $result = python -c "import $module; print('✅ OK')" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   - $module: $result" -ForegroundColor Green
            } else {
                Write-Host "   - $module: ❌ Not available" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "   ❌ Error checking Python: $_" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Python not found in PATH" -ForegroundColor Red
}

# 2. Check Network Connectivity
Write-Host "`n[2/4] Checking Network Connectivity..." -ForegroundColor Yellow
try {
    $testUrls = @(
        "https://www.google.com",
        "https://openrouter.ai",
        "https://api.openrouter.ai"
    )
    
    foreach ($url in $testUrls) {
        try {
            $request = Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
            Write-Host "   ✅ $url - Status: $($request.StatusCode)" -ForegroundColor Green
        } catch {
            Write-Host "   ❌ $url - Error: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
} catch {
    Write-Host "   ❌ Network test failed: $_" -ForegroundColor Red
}

# 3. Test OpenRouter API
Write-Host "`n[3/4] Testing OpenRouter API..." -ForegroundColor Yellow
$apiKey = "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"
$headers = @{
    "Authorization" = "Bearer $apiKey"
    "Content-Type" = "application/json"
}

try {
    $response = Invoke-RestMethod -Uri "https://openrouter.ai/api/v1/auth/key" -Headers $headers -Method Get -TimeoutSec 10 -ErrorAction Stop
    Write-Host "   ✅ OpenRouter API connection successful!" -ForegroundColor Green
    Write-Host "      Plan: $($response.data.plan)" -ForegroundColor Cyan
    Write-Host "      Rate Limit: $($response.data.rate_limit.remaining)/$($response.data.rate_limit.limit) requests remaining" -ForegroundColor Cyan
} catch {
    Write-Host "   ❌ OpenRouter API connection failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response) {
        $statusCode = $_.Exception.Response.StatusCode.value__
        $statusDescription = $_.Exception.Response.StatusDescription
        Write-Host "      Status: $statusCode $statusDescription" -ForegroundColor Red
        
        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $reader.BaseStream.Position = 0
            $reader.DiscardBufferedData()
            $responseBody = $reader.ReadToEnd()
            if ($responseBody) {
                Write-Host "      Response: $responseBody" -ForegroundColor Red
            }
        } catch {}
    }
}

# 4. Check Environment Variables
Write-Host "`n[4/4] Checking Environment Variables..." -ForegroundColor Yellow
$envVars = @("OPENROUTER_API_KEY", "HRM_PROVIDER", "PYTHONPATH")

foreach ($var in $envVars) {
    $value = [System.Environment]::GetEnvironmentVariable($var)
    if ($value) {
        if ($var -eq "OPENROUTER_API_KEY") {
            $displayValue = if ($value.Length -gt 8) { "********" + $value.Substring($value.Length - 4) } else { "********" }
        } else {
            $displayValue = $value
        }
        Write-Host "   ✅ $var = $displayValue" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  $var is not set" -ForegroundColor Yellow
    }
}

Write-Host "`n=== Test Complete ===`n" -ForegroundColor Cyan

# Wait for user input before closing
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
