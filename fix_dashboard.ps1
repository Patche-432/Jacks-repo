# fix_dashboard.ps1
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File fix_dashboard.ps1
#
# Restores dashboard.js from the last good git commit, then applies the
# two memory-panel hook lines.

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
Set-Location $repo

Write-Host ""
Write-Host "[1/3] Restoring core/js/dashboard.js from git..." -ForegroundColor Cyan
git checkout HEAD -- core/js/dashboard.js
if ($LASTEXITCODE -ne 0) {
    Write-Error "git checkout failed. Is this a git repo with dashboard.js tracked?"
    exit 1
}
$path = Join-Path $repo "core\js\dashboard.js"
$size = (Get-Item $path).Length
Write-Host "      Restored: $size bytes" -ForegroundColor Green

Write-Host ""
Write-Host "[2/3] Applying showTab hook..." -ForegroundColor Cyan
$content = Get-Content $path -Raw -Encoding UTF8

# Original anchor uses an em-dash (U+2014).
$emdash = [char]0x2014
$old1 = "        // 'backtest' is static HTML $emdash no fetch required"
$new1 = "        if (tab === 'backtest') _btLoadMemory();"

if ($content.Contains($old1)) {
    $content = $content.Replace($old1, $new1)
    Write-Host "      Hook 1 applied (exact match)" -ForegroundColor Green
} else {
    # Fuzzy fallback: comment without em-dash matched by regex to end-of-line.
    $old1b = "        // 'backtest' is static HTML"
    if ($content.Contains($old1b)) {
        $content = [regex]::Replace($content, [regex]::Escape($old1b) + "[^`n]*", $new1)
        Write-Host "      Hook 1 applied (fuzzy match)" -ForegroundColor Yellow
    } else {
        Write-Warning "      Hook 1 anchor not found. SKIPPED."
    }
}

Write-Host ""
Write-Host "[3/3] Applying done-handler hook..." -ForegroundColor Cyan
$old2 = "            _btRenderAssumptions(data.per_pair);`n            const dur"
$new2 = "            _btRenderAssumptions(data.per_pair);`n            _btLoadMemory();`n            const dur"

if ($content.Contains($old2)) {
    $content = $content.Replace($old2, $new2)
    Write-Host "      Hook 2 applied" -ForegroundColor Green
} else {
    Write-Warning "      Hook 2 anchor not found. SKIPPED."
}

[System.IO.File]::WriteAllText($path, $content, [System.Text.UTF8Encoding]::new($false))

Write-Host ""
Write-Host "[done] dashboard.js fixed." -ForegroundColor Green
$lines = (Get-Content $path).Length
Write-Host "      $lines lines written to $path"
Write-Host ""
Write-Host "      Restart Flask, then hard-refresh the browser (Ctrl+Shift+R)." -ForegroundColor Cyan
