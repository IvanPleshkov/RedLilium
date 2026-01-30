# ============================================================================
# setup-hooks.ps1 - Install git hooks for RedLilium Engine (Windows)
# ============================================================================
# This script installs the project's git hooks by copying them to the
# .git/hooks directory.
#
# Usage: .\scripts\setup-hooks.ps1
# ============================================================================

$ErrorActionPreference = "Stop"

# Get script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$hooksSource = Join-Path $scriptDir "hooks"
$hooksTarget = Join-Path $projectRoot ".git\hooks"

Write-Host "Setting up git hooks for RedLilium Engine" -ForegroundColor Blue
Write-Host ""

# Check if we're in a git repository
$gitDir = Join-Path $projectRoot ".git"
if (-not (Test-Path $gitDir)) {
    Write-Host "Error: Not a git repository. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

# Check if hooks source directory exists
if (-not (Test-Path $hooksSource)) {
    Write-Host "Error: Hooks source directory not found: $hooksSource" -ForegroundColor Red
    exit 1
}

# Install each hook
Get-ChildItem -Path $hooksSource -File | ForEach-Object {
    $hookName = $_.Name
    $source = $_.FullName
    $target = Join-Path $hooksTarget $hookName

    # Remove existing hook if it exists
    if (Test-Path $target) {
        Remove-Item $target -Force
        Write-Host "Replaced existing hook: " -ForegroundColor Yellow -NoNewline
        Write-Host $hookName
    }

    # Copy the hook file (symlinks can be problematic on Windows)
    Copy-Item $source $target -Force
    Write-Host "Installed hook: " -ForegroundColor Green -NoNewline
    Write-Host $hookName
}

Write-Host ""
Write-Host "Git hooks installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "The following hooks are now active:"
Write-Host "  - pre-commit: Runs 'cargo fmt --check' before each commit"
Write-Host ""
Write-Host "Note: " -ForegroundColor Yellow -NoNewline
Write-Host "To bypass hooks temporarily, use: git commit --no-verify"
