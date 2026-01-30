# ============================================================================
# test-all.ps1 - Cross-platform test script for RedLilium Engine (PowerShell)
# ============================================================================
# This script runs all tests for the project:
#   1. Native target build
#   2. Web target build (wasm)
#   3. Unit tests for all crates
#   4. Clippy linter
#
# Usage: .\scripts\test-all.ps1 [OPTIONS]
#
# Options:
#   -SkipNative    Skip native build test
#   -SkipWeb       Skip web build test
#   -SkipTests     Skip unit tests
#   -SkipClippy    Skip clippy linter
#   -Verbose       Show verbose output
#   -Help          Show this help message
# ============================================================================

param(
    [switch]$SkipNative,
    [switch]$SkipWeb,
    [switch]$SkipTests,
    [switch]$SkipClippy,
    [switch]$Verbose,
    [switch]$Help
)

# Stop on first error
$ErrorActionPreference = "Stop"

# Configuration
$script:Passed = 0
$script:Failed = 0
$script:Skipped = 0

function Show-Help {
    @"
test-all.ps1 - Cross-platform test script for RedLilium Engine

Usage: .\scripts\test-all.ps1 [OPTIONS]

Options:
    -SkipNative    Skip native build test
    -SkipWeb       Skip web build test
    -SkipTests     Skip unit tests
    -SkipClippy    Skip clippy linter
    -Verbose       Show verbose output
    -Help          Show this help message

Examples:
    .\scripts\test-all.ps1                    # Run all tests
    .\scripts\test-all.ps1 -SkipWeb           # Skip web build
    .\scripts\test-all.ps1 -SkipNative -SkipWeb   # Only run tests and clippy
"@
}

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "============================================================" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[PASS] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Error2 {
    param([string]$Message)
    Write-Host "[FAIL] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Warning2 {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Skip {
    param([string]$Message)
    Write-Host "[SKIP] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Check-Tools {
    Write-Header "Checking Required Tools"

    # Check cargo
    if (Test-Command "cargo") {
        $cargoVersion = cargo --version
        Write-Success "cargo found: $cargoVersion"
    } else {
        Write-Error2 "cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    }

    # Check clippy
    $clippyCheck = cargo clippy --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "clippy found: $clippyCheck"
    } else {
        Write-Warning2 "clippy not found. Installing..."
        rustup component add clippy
    }

    # Check wasm-pack (only if web build is not skipped)
    if (-not $SkipWeb) {
        if (Test-Command "wasm-pack") {
            $wasmPackVersion = wasm-pack --version
            Write-Success "wasm-pack found: $wasmPackVersion"
        } else {
            Write-Error2 "wasm-pack not found. Install: https://rustwasm.github.io/wasm-pack/installer/"
            Write-Warning2 "Skipping web build tests"
            $script:SkipWeb = $true
        }
    }
}

function Test-NativeBuild {
    if ($SkipNative) {
        Write-Skip "Native build (-SkipNative)"
        $script:Skipped++
        return $true
    }

    Write-Header "Testing Native Build"

    try {
        if ($Verbose) {
            cargo build --workspace
        } else {
            $null = cargo build --workspace 2>&1
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Native build succeeded"
            $script:Passed++
            return $true
        } else {
            Write-Error2 "Native build failed"
            $script:Failed++
            return $false
        }
    } catch {
        Write-Error2 "Native build failed: $_"
        $script:Failed++
        return $false
    }
}

function Test-WebBuild {
    if ($SkipWeb) {
        Write-Skip "Web build (-SkipWeb)"
        $script:Skipped++
        return $true
    }

    Write-Header "Testing Web Build (WASM)"

    # Get project root
    $scriptDir = Split-Path -Parent $MyInvocation.ScriptName
    $projectRoot = Split-Path -Parent $scriptDir

    try {
        $demosPath = Join-Path $projectRoot "demos"

        if ($Verbose) {
            wasm-pack build $demosPath --target web --out-dir web/pkg
        } else {
            $null = wasm-pack build $demosPath --target web --out-dir web/pkg 2>&1
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Web build succeeded"
            $script:Passed++
            return $true
        } else {
            Write-Error2 "Web build failed"
            $script:Failed++
            return $false
        }
    } catch {
        Write-Error2 "Web build failed: $_"
        $script:Failed++
        return $false
    }
}

function Test-UnitTests {
    if ($SkipTests) {
        Write-Skip "Unit tests (-SkipTests)"
        $script:Skipped++
        return $true
    }

    Write-Header "Running Unit Tests"

    try {
        if ($Verbose) {
            cargo test --workspace
        } else {
            cargo test --workspace 2>&1 | Out-Host
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Success "All unit tests passed"
            $script:Passed++
            return $true
        } else {
            Write-Error2 "Unit tests failed"
            $script:Failed++
            return $false
        }
    } catch {
        Write-Error2 "Unit tests failed: $_"
        $script:Failed++
        return $false
    }
}

function Test-Clippy {
    if ($SkipClippy) {
        Write-Skip "Clippy linter (-SkipClippy)"
        $script:Skipped++
        return $true
    }

    Write-Header "Running Clippy Linter"

    try {
        if ($Verbose) {
            cargo clippy --workspace --all-targets -- -D warnings
        } else {
            cargo clippy --workspace --all-targets -- -D warnings 2>&1 | Out-Host
        }

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Clippy passed (no warnings)"
            $script:Passed++
            return $true
        } else {
            Write-Error2 "Clippy found issues"
            $script:Failed++
            return $false
        }
    } catch {
        Write-Error2 "Clippy failed: $_"
        $script:Failed++
        return $false
    }
}

function Show-Summary {
    Write-Header "Test Summary"

    Write-Host "Passed:  " -ForegroundColor Green -NoNewline
    Write-Host $script:Passed
    Write-Host "Failed:  " -ForegroundColor Red -NoNewline
    Write-Host $script:Failed
    Write-Host "Skipped: " -ForegroundColor Yellow -NoNewline
    Write-Host $script:Skipped
    Write-Host ""

    if ($script:Failed -eq 0) {
        Write-Host "All tests passed!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "Some tests failed." -ForegroundColor Red
        return $false
    }
}

# Main execution
function Main {
    if ($Help) {
        Show-Help
        exit 0
    }

    Write-Host "RedLilium Engine - Test Suite" -ForegroundColor Blue
    Write-Host "Running comprehensive tests..."

    Check-Tools

    # Disable error stopping temporarily to collect all results
    $ErrorActionPreference = "Continue"

    Test-NativeBuild | Out-Null
    Test-WebBuild | Out-Null
    Test-UnitTests | Out-Null
    Test-Clippy | Out-Null

    $ErrorActionPreference = "Stop"

    $success = Show-Summary

    if (-not $success) {
        exit 1
    }
}

Main
