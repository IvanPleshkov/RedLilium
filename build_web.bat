@echo off
REM Build script for Web (Windows)
REM Usage: build_web.bat [--release]

setlocal enabledelayedexpansion

echo ==========================================
echo  Graphics Engine - Web Build (Windows)
echo ==========================================
echo.

REM Check for wasm-pack
where wasm-pack >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] wasm-pack is not installed.
    echo.
    echo Install it with:
    echo   cargo install wasm-pack
    echo.
    echo Or visit: https://rustwasm.github.io/wasm-pack/installer/
    exit /b 1
)

REM Check for wasm32 target
rustup target list --installed | findstr /C:"wasm32-unknown-unknown" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Installing wasm32-unknown-unknown target...
    rustup target add wasm32-unknown-unknown
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install wasm32 target
        exit /b 1
    )
)

REM Parse arguments
set BUILD_MODE=--dev
set OUTPUT_DIR=web\pkg
if "%1"=="--release" (
    set BUILD_MODE=--release
    echo [INFO] Building in RELEASE mode
) else (
    echo [INFO] Building in DEBUG mode
)

echo.
echo [STEP 1/3] Building WASM module...
echo.

REM Build the library with wasm-pack
wasm-pack build --target web --out-dir %OUTPUT_DIR% --no-typescript %BUILD_MODE% --no-default-features --features web

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] wasm-pack build failed!
    exit /b 1
)

echo.
echo [STEP 2/3] Copying HTML files...
echo.

REM The index.html is already in web/ folder, wasm-pack outputs to web/pkg/

echo [STEP 3/3] Build complete!
echo.
echo ==========================================
echo  Build successful!
echo ==========================================
echo.
echo Output files:
echo   - web\index.html
echo   - web\pkg\graphics_engine.js
echo   - web\pkg\graphics_engine_bg.wasm
echo.
echo To run the demo:
echo   1. Start a local web server in the 'web' directory
echo   2. Open http://localhost:8000 in a WebGPU-capable browser
echo.
echo Quick server options:
echo   Python 3:  cd web ^&^& python -m http.server 8000
echo   Node.js:   npx serve web
echo   Rust:      cargo install miniserve ^&^& miniserve web --port 8000
echo.
echo Or run: run_web.bat
echo.

exit /b 0
