# ============================================================================
# build-book.ps1 - Build mdbook documentation for RedLilium Engine (PowerShell)
# ============================================================================
# This script builds the mdbook and optionally generates a single-page
# portable HTML file with all assets inlined.
#
# Usage: .\scripts\build-book.ps1 [OPTIONS]
#
# Options:
#   -SinglePage    Generate a single portable HTML file
#   -Open          Open the book in the default browser after building
#   -Clean         Remove previous build output before building
#   -Help          Show this help message
# ============================================================================

param(
    [switch]$SinglePage,
    [switch]$Open,
    [switch]$Clean,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Resolve paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$bookDir = Join-Path $projectRoot "book"
$bookOutput = Join-Path $bookDir "book"
$singlePageOutput = Join-Path $bookDir "redlilium-book.html"

function Show-Help {
    @"
build-book.ps1 - Build mdbook documentation for RedLilium Engine

Usage: .\scripts\build-book.ps1 [OPTIONS]

Options:
    -SinglePage    Generate a single portable HTML file
    -Open          Open the book in the default browser after building
    -Clean         Remove previous build output before building
    -Help          Show this help message

Examples:
    .\scripts\build-book.ps1                      # Build the book
    .\scripts\build-book.ps1 -SinglePage          # Build + single-page HTML
    .\scripts\build-book.ps1 -SinglePage -Open    # Build, pack, and open
    .\scripts\build-book.ps1 -Clean -SinglePage   # Clean rebuild + single-page
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

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

function Check-Tools {
    Write-Header "Checking Required Tools"

    if (Test-Command "mdbook") {
        $version = mdbook --version
        Write-Success "mdbook found: $version"
    } else {
        Write-Error2 "mdbook not found. Installing..."
        cargo install mdbook
        Write-Success "mdbook installed"
    }

    if ($SinglePage) {
        if (Test-Command "monolith") {
            $version = monolith --version
            Write-Success "monolith found: $version"
        } else {
            Write-Error2 "monolith not found. Installing..."
            cargo install monolith
            Write-Success "monolith installed"
        }
    }
}

function Clean-Build {
    if ($Clean) {
        Write-Info "Cleaning previous build..."
        if (Test-Path $bookOutput) {
            Remove-Item -Recurse -Force $bookOutput
        }
        if (Test-Path $singlePageOutput) {
            Remove-Item -Force $singlePageOutput
        }
    }
}

function Build-Book {
    Write-Header "Building mdbook"

    mdbook build $bookDir
    if ($LASTEXITCODE -eq 0) {
        Write-Success "mdbook built successfully"
        Write-Info "Output: $bookOutput"
    } else {
        Write-Error2 "mdbook build failed"
        exit 1
    }
}

function Build-SinglePage {
    if (-not $SinglePage) { return }

    Write-Header "Generating Single-Page HTML"

    $printHtml = Join-Path $bookOutput "print.html"
    if (-not (Test-Path $printHtml)) {
        Write-Error2 "print.html not found at $printHtml"
        exit 1
    }

    monolith $printHtml -o $singlePageOutput 2>$null
    if ($LASTEXITCODE -eq 0) {
        # Remove print button and print CSS so the page behaves as a normal reader
        $content = Get-Content -Raw $singlePageOutput
        $content = [regex]::Replace($content, '<a [^>]*title="Print this book"[^>]*>.*?</a>', '', [System.Text.RegularExpressions.RegexOptions]::Singleline)
        $content = [regex]::Replace($content, '<link[^>]*media="print"[^>]*>', '')
        Set-Content -Path $singlePageOutput -Value $content -NoNewline

        $fileSize = (Get-Item $singlePageOutput).Length / 1MB
        $fileSizeStr = "{0:N1} MB" -f $fileSize
        Write-Success "Single-page HTML generated ($fileSizeStr)"
        Write-Info "Output: $singlePageOutput"
    } else {
        Write-Error2 "Failed to generate single-page HTML"
        exit 1
    }
}

function Open-Book {
    if (-not $Open) { return }

    if ($SinglePage -and (Test-Path $singlePageOutput)) {
        $target = $singlePageOutput
    } else {
        $target = Join-Path $bookOutput "index.html"
    }

    Write-Info "Opening $target"
    Start-Process $target
}

# Main execution
function Main {
    if ($Help) {
        Show-Help
        exit 0
    }

    Write-Host "RedLilium Engine - Book Builder" -ForegroundColor Blue

    Check-Tools
    Clean-Build
    Build-Book
    Build-SinglePage
    Open-Book

    Write-Header "Done"
    Write-Success "Book build complete"
}

Main
