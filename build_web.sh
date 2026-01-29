#!/bin/bash
# Build script for Web (Linux/macOS)
# Usage: ./build_web.sh [--release]

set -e

echo "=========================================="
echo " Graphics Engine - Web Build (Unix)"
echo "=========================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for wasm-pack
if ! command -v wasm-pack &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} wasm-pack is not installed."
    echo
    echo "Install it with:"
    echo "  cargo install wasm-pack"
    echo
    echo "Or visit: https://rustwasm.github.io/wasm-pack/installer/"
    exit 1
fi

# Check for wasm32 target
if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
    echo -e "${YELLOW}[INFO]${NC} Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

# Parse arguments
BUILD_MODE="--dev"
if [ "$1" == "--release" ]; then
    BUILD_MODE="--release"
    echo -e "${GREEN}[INFO]${NC} Building in RELEASE mode"
else
    echo -e "${GREEN}[INFO]${NC} Building in DEBUG mode"
fi

OUTPUT_DIR="web/pkg"

echo
echo -e "${GREEN}[STEP 1/3]${NC} Building WASM module..."
echo

# Build the library with wasm-pack
wasm-pack build --target web --out-dir "$OUTPUT_DIR" --no-typescript $BUILD_MODE --no-default-features --features web

echo
echo -e "${GREEN}[STEP 2/3]${NC} Copying HTML files..."
echo

# The index.html is already in web/ folder, wasm-pack outputs to web/pkg/

echo -e "${GREEN}[STEP 3/3]${NC} Build complete!"
echo
echo "=========================================="
echo -e "${GREEN} Build successful!${NC}"
echo "=========================================="
echo
echo "Output files:"
echo "  - web/index.html"
echo "  - web/pkg/graphics_engine.js"
echo "  - web/pkg/graphics_engine_bg.wasm"
echo
echo "To run the demo:"
echo "  1. Start a local web server in the 'web' directory"
echo "  2. Open http://localhost:8000 in a WebGPU-capable browser"
echo
echo "Quick server options:"
echo "  Python 3:  cd web && python -m http.server 8000"
echo "  Node.js:   npx serve web"
echo "  Rust:      cargo install miniserve && miniserve web --port 8000"
echo
echo "Or run: ./run_web.sh"
echo
