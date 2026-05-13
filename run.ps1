# run.ps1 — Start the Fortis dashboard from the repo root
# Run from VS Code terminal or PowerShell:
#   .\run.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

Write-Host ""
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Fortis Multi-Agent Dashboard" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════" -ForegroundColor Cyan

# Always use the .venv interpreter — never the system Python
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host ""
    Write-Host "  ERROR: .venv not found at $Python" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Create it first:" -ForegroundColor Yellow
    Write-Host "    python -m venv .venv"
    Write-Host "    .\.venv\Scripts\Activate.ps1"
    Write-Host "    pip install -r requirements.txt"
    exit 1
}

# Run from the repo root so all relative imports work
Set-Location $RepoRoot
Write-Host "  Python: $Python" -ForegroundColor Green
Write-Host "  Starting on http://localhost:5000" -ForegroundColor Green
Write-Host ""

& $Python scripts\serve.py @args
