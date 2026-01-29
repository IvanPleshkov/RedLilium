#!/bin/bash
# Run script for Web demo (Linux/macOS)
# This script builds and runs the web demo with a local server

set -e

echo "=========================================="
echo " Graphics Engine - Web Demo Runner"
echo "=========================================="
echo

# Build first
./build_web.sh "$@"

echo
echo "[INFO] Starting local web server..."
echo
echo "=========================================="
echo " Server running at: http://localhost:8000"
echo " Press Ctrl+C to stop"
echo "=========================================="
echo

# Try to open browser
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8000" &
elif command -v open &> /dev/null; then
    open "http://localhost:8000" &
fi

# Start server (try Python first, then others)
cd web

if command -v python3 &> /dev/null; then
    python3 -m http.server 8000
elif command -v python &> /dev/null; then
    python -m http.server 8000
elif command -v npx &> /dev/null; then
    npx serve . -l 8000
elif command -v miniserve &> /dev/null; then
    miniserve . --port 8000
else
    echo "[ERROR] No suitable web server found!"
    echo "Please install one of:"
    echo "  - Python 3"
    echo "  - Node.js (for npx serve)"
    echo "  - miniserve (cargo install miniserve)"
    exit 1
fi
