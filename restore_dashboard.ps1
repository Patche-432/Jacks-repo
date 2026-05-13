# restore_dashboard.ps1
# Run once from the repo root to restore dashboard.js from git and apply
# the two memory-panel hook lines.
#
# Usage: powershell -ExecutionPolicy Bypass -File restore_dashboard.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Restoring dashboard.js from git..." -ForegroundColor Cyan
Set-Location $repo
git checkout HEAD -- core/js/dashboard.js
if ($LASTEXITCODE -ne 0) {
    Write-Error "git checkout failed. Make sure you are in a git repo and the file is tracked."
    exit 1
}
Write-Host "  Restored." -ForegroundColor Green

# Patch 1: hook _btLoadMemory into showTab for the backtest tab
$path = Join-Path $repo "core\js\dashboard.js"
$content = Get-Content $path -Raw -Encoding UTF8

$old1 = "        // 'backtest' is static HTML`u{2014} no fetch required"
$new1 = "        if (tab === 'backtest') _btLoadMemory();"
if ($content.Contains($old1)) {
    $content = $content.Replace($old1, $new1)
    Write-Host "  Applied hook 1: showTab backtest branch." -ForegroundColor Green
} else {
    Write-Warning "Hook 1 anchor not found (check dashboard.js version). Skipping."
}

# Patch 2: reload memory panel after every completed backtest run
$old2 = "            _btRenderAssumptions(data.per_pair);`n            const dur"
$new2 = "            _btRenderAssumptions(data.per_pair);`n            _btLoadMemory();`n            const dur"
if ($content.Contains($old2)) {
    $content = $content.Replace($old2, $new2)
    Write-Host "  Applied hook 2: done-event handler." -ForegroundColor Green
} else {
    Write-Warning "Hook 2 anchor not found. Skipping."
}

[System.IO.File]::WriteAllText($path, $content, [System.Text.Encoding]::UTF8)
Write-Host "  dashboard.js written." -ForegroundColor Green

Write-Host ""
Write-Host "Done. Restart the Flask server and hard-refresh the browser." -ForegroundColor Cyan
Write-Host "The Trade Memory panel will appear below ML Insights on the Backtest tab." -ForegroundColor Cyan
