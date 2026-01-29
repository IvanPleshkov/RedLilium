@echo off
REM Run script for Web demo (Windows)
REM This script builds and runs the web demo with a local server

setlocal enabledelayedexpansion

echo ==========================================
echo  Graphics Engine - Web Demo Runner
echo ==========================================
echo.

REM Build first
call build_web.bat %*
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo [INFO] Starting local web server...
echo.
echo ==========================================
echo  Server running at: http://localhost:8000
echo  Press Ctrl+C to stop
echo ==========================================
echo.
echo Opening browser...

REM Try to open browser
start "" "http://localhost:8000"

REM Start server (try Python first, then Node)
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    cd web
    python -m http.server 8000
    cd ..
    exit /b 0
)

where python3 >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    cd web
    python3 -m http.server 8000
    cd ..
    exit /b 0
)

where npx >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    npx serve web -l 8000
    exit /b 0
)

where miniserve >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    miniserve web --port 8000
    exit /b 0
)

echo [ERROR] No suitable web server found!
echo Please install one of:
echo   - Python 3
echo   - Node.js (for npx serve)
echo   - miniserve (cargo install miniserve)
exit /b 1
