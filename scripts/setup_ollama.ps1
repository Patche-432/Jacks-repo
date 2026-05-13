# ============================================================================
# setup_ollama.ps1 -- One-shot Ollama setup for the Fortis multi-agent bot
# ============================================================================
#
# Run from the repo root:
#     .\scripts\setup_ollama.ps1
#
# What it does:
#   1. Checks Ollama is installed (and tells you exactly how to install it
#      if it isn't).
#   2. Verifies the Ollama service is running on the configured URL
#      (defaults to http://localhost:11434, override via $env:OLLAMA_URL).
#   3. Pulls the configured model (default qwen2.5:3b-instruct, override
#      via $env:OLLAMA_MODEL).
#   4. Sends a tiny test chat to confirm the agent stack will work.
#
# Exits non-zero on any failure so you can chain it into deploy steps.
# ============================================================================

$ErrorActionPreference = "Stop"

# -- Config (env-var overridable) --------------------------------------------
$OllamaUrl   = if ($env:OLLAMA_URL)   { $env:OLLAMA_URL }   else { "http://localhost:11434" }
$OllamaModel = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "qwen2.5:3b-instruct" }

# Make sure the default Ollama install dir is on PATH for this session.
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    if (Test-Path "C:\Program Files\Ollama\ollama.exe") {
        $env:PATH += ";C:\Program Files\Ollama"
    } elseif (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") {
        $env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"
    }
}

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  Fortis Agent Stack -- Ollama Setup" -ForegroundColor Cyan
Write-Host "===============================================================" -ForegroundColor Cyan
Write-Host "  URL:   $OllamaUrl"
Write-Host "  Model: $OllamaModel"
Write-Host ""

# -- Step 1 -- Ollama installed? ---------------------------------------------
Write-Host "[1/4] Checking Ollama is installed..." -ForegroundColor Yellow
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    Write-Host ""
    Write-Host "  [X] Ollama is not installed (or not on PATH)." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Install it with one of:" -ForegroundColor Yellow
    Write-Host "    - Download:  https://ollama.com/download/windows"
    Write-Host "    - Winget:    winget install Ollama.Ollama"
    Write-Host ""
    Write-Host "  After install, open a NEW terminal and run this script again."
    exit 1
}
Write-Host "  [OK] Ollama found: $($ollamaCmd.Source)" -ForegroundColor Green

# -- Step 2 -- Ollama service reachable? -------------------------------------
Write-Host ""
Write-Host "[2/4] Checking Ollama service at $OllamaUrl..." -ForegroundColor Yellow
$tagsUrl = "$OllamaUrl/api/tags"
try {
    $tagsResp = Invoke-RestMethod -Uri $tagsUrl -Method GET -TimeoutSec 5
    Write-Host "  [OK] Ollama service is running." -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "  [X] Cannot reach Ollama at $OllamaUrl." -ForegroundColor Red
    Write-Host "    Error: $($_.Exception.Message)"
    Write-Host ""
    Write-Host "  On Windows, the Ollama installer registers a background service" -ForegroundColor Yellow
    Write-Host "  that auto-starts on boot. If it isn't running, try:"
    Write-Host "    - Open the Ollama desktop app once (it will start the service)"
    Write-Host "    - Or run:  ollama serve     (in a separate terminal)"
    Write-Host ""
    exit 1
}

# -- Step 3 -- Model pulled? -------------------------------------------------
Write-Host ""
Write-Host "[3/4] Checking model '$OllamaModel' is pulled..." -ForegroundColor Yellow
$installedNames = @()
if ($tagsResp.models) {
    $installedNames = $tagsResp.models | ForEach-Object { $_.name }
}

# Match either exact tag or base-name (e.g. "qwen2.5:3b-instruct" or "qwen2.5")
$modelBaseName = $OllamaModel.Split(":")[0]
$alreadyHave = $installedNames | Where-Object {
    $_ -eq $OllamaModel -or $_ -like "$modelBaseName*"
}

if ($alreadyHave) {
    Write-Host "  [OK] Model already pulled: $($alreadyHave -join ', ')" -ForegroundColor Green
} else {
    Write-Host "  [..] Pulling $OllamaModel (this can take a few minutes)..." -ForegroundColor Yellow
    & ollama pull $OllamaModel
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "  [X] 'ollama pull $OllamaModel' failed (exit $LASTEXITCODE)." -ForegroundColor Red
        Write-Host "    Check the model name and your internet connection."
        exit 1
    }
    Write-Host "  [OK] Model pulled." -ForegroundColor Green
}

# -- Step 4 -- Smoke test: tiny chat call ------------------------------------
Write-Host ""
Write-Host "[4/4] Smoke-testing Ollama with a tiny chat request..." -ForegroundColor Yellow

$chatBody = @{
    model    = $OllamaModel
    messages = @(
        @{ role = "system"; content = "Reply with the single word OK and nothing else." }
        @{ role = "user";   content = "ping" }
    )
    stream  = $false
    options = @{ temperature = 0.0; num_predict = 8 }
} | ConvertTo-Json -Depth 6

try {
    $chatResp = Invoke-RestMethod -Uri "$OllamaUrl/api/chat" `
                                  -Method POST `
                                  -ContentType "application/json" `
                                  -Body $chatBody `
                                  -TimeoutSec 60
    $reply = $chatResp.message.content
    if ([string]::IsNullOrWhiteSpace($reply)) {
        Write-Host "  [X] Got an empty reply from Ollama." -ForegroundColor Red
        exit 1
    }
    $preview = $reply.Trim()
    if ($preview.Length -gt 60) { $preview = $preview.Substring(0, 60) + "..." }
    Write-Host "  [OK] Smoke test passed. Reply: $preview" -ForegroundColor Green
} catch {
    Write-Host "  [X] Smoke test failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===============================================================" -ForegroundColor Green
Write-Host "  [OK] Ollama is ready for the Fortis agent stack." -ForegroundColor Green
Write-Host "===============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. python scripts\preflight.py        # full pre-flight check"
Write-Host "  2. python core\server.py              # start the dashboard"
Write-Host ""

