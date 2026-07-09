#!/usr/bin/env bash
# ============================================================================
# test-all.sh - Cross-platform test script for RedLilium Engine
# ============================================================================
# This script runs all tests for the project:
#   1. Native target build
#   2. Web target build (wasm)
#   3. Unit tests for all crates
#   4. Clippy linter
#
# Usage: ./scripts/test-all.sh [OPTIONS]
#
# Options:
#   --skip-native    Skip native build test
#   --skip-web       Skip web build test
#   --skip-tests     Skip unit tests
#   --skip-clippy    Skip clippy linter
#   --verbose        Show verbose output
#   --help           Show this help message
# ============================================================================

set -e  # Exit on first error

# Colors for output (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Configuration
SKIP_NATIVE=false
SKIP_WEB=false
SKIP_TESTS=false
SKIP_CLIPPY=false
VERBOSE=false
WASM_PACK_AVAILABLE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-native)
            SKIP_NATIVE=true
            shift
            ;;
        --skip-web)
            SKIP_WEB=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-clippy)
            SKIP_CLIPPY=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            head -24 "$0" | tail -20
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Check for required tools
check_tools() {
    print_header "Checking Required Tools"

    # Check cargo
    if command -v cargo &> /dev/null; then
        CARGO_VERSION=$(cargo --version)
        print_success "cargo found: $CARGO_VERSION"
    else
        print_error "cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    fi

    # Check clippy
    if cargo clippy --version &> /dev/null; then
        CLIPPY_VERSION=$(cargo clippy --version)
        print_success "clippy found: $CLIPPY_VERSION"
    else
        print_warning "clippy not found. Installing..."
        rustup component add clippy
    fi

    # Check wasm-pack (only if web build is not skipped). Its absence is NOT a
    # failure and does not skip the web check — test_web_build falls back to a
    # plain `cargo build --target wasm32-unknown-unknown`, which needs only the
    # rustup target. wasm-pack additionally exercises wasm-bindgen + wasm-opt
    # packaging, so we prefer it when present.
    if [ "$SKIP_WEB" = false ]; then
        if command -v wasm-pack &> /dev/null; then
            WASM_PACK_VERSION=$(wasm-pack --version)
            print_success "wasm-pack found: $WASM_PACK_VERSION"
        else
            WASM_PACK_AVAILABLE=false
            print_warning "wasm-pack not found — web check will compile-only via cargo."
            print_warning "For full packaging: https://rustwasm.github.io/wasm-pack/installer/"
        fi
    fi
}

# Test native build
test_native_build() {
    if [ "$SKIP_NATIVE" = true ]; then
        print_skip "Native build (--skip-native)"
        ((SKIPPED++))
        return
    fi

    print_header "Testing Native Build"

    if [ "$VERBOSE" = true ]; then
        if cargo build --workspace; then
            print_success "Native build succeeded"
            ((PASSED++))
        else
            print_error "Native build failed"
            ((FAILED++))
            return 1
        fi
    else
        if cargo build --workspace 2>&1; then
            print_success "Native build succeeded"
            ((PASSED++))
        else
            print_error "Native build failed"
            ((FAILED++))
            return 1
        fi
    fi
}

# Test web build.
#
# Prefers `wasm-pack` (full pipeline: cargo wasm build + wasm-bindgen + wasm-opt
# packaging into demos/web/pkg, which is gitignored). When wasm-pack is absent,
# falls back to a plain `cargo build --target wasm32-unknown-unknown` of the
# demos lib — this still catches the common regression (something that does not
# compile for wasm) using only the rustup target, so `--skip-web` is a speed
# flag, not a requirement. Only truly skips if the wasm32 target is also missing.
test_web_build() {
    if [ "$SKIP_WEB" = true ]; then
        print_skip "Web build (--skip-web)"
        ((SKIPPED++))
        return
    fi

    print_header "Testing Web Build (WASM)"

    local SCRIPT_DIR PROJECT_ROOT
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

    if [ "$WASM_PACK_AVAILABLE" = true ]; then
        if wasm-pack build "$PROJECT_ROOT/demos" --target web --out-dir web/pkg 2>&1; then
            print_success "Web build succeeded (wasm-pack: full packaging)"
            ((PASSED++))
        else
            print_error "Web build failed"
            ((FAILED++))
            return 1
        fi
        return
    fi

    # Fallback: no wasm-pack. Verify the demos lib at least compiles for wasm.
    if ! rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
        print_skip "Web build (no wasm-pack and wasm32 target missing — rustup target add wasm32-unknown-unknown)"
        ((SKIPPED++))
        return
    fi

    if cargo build --target wasm32-unknown-unknown -p redlilium-demos --lib 2>&1; then
        print_success "Web build succeeded (cargo compile-only — install wasm-pack for packaging)"
        ((PASSED++))
    else
        print_error "Web build failed (wasm32 compile)"
        ((FAILED++))
        return 1
    fi
}

# Run unit tests
test_unit_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        print_skip "Unit tests (--skip-tests)"
        ((SKIPPED++))
        return
    fi

    print_header "Running Unit Tests"

    if [ "$VERBOSE" = true ]; then
        if cargo test --workspace; then
            print_success "All unit tests passed"
            ((PASSED++))
        else
            print_error "Unit tests failed"
            ((FAILED++))
            return 1
        fi
    else
        if cargo test --workspace 2>&1; then
            print_success "All unit tests passed"
            ((PASSED++))
        else
            print_error "Unit tests failed"
            ((FAILED++))
            return 1
        fi
    fi
}

# Run clippy
test_clippy() {
    if [ "$SKIP_CLIPPY" = true ]; then
        print_skip "Clippy linter (--skip-clippy)"
        ((SKIPPED++))
        return
    fi

    print_header "Running Clippy Linter"

    if [ "$VERBOSE" = true ]; then
        if cargo clippy --workspace --all-targets -- -D warnings; then
            print_success "Clippy passed (no warnings)"
            ((PASSED++))
        else
            print_error "Clippy found issues"
            ((FAILED++))
            return 1
        fi
    else
        if cargo clippy --workspace --all-targets -- -D warnings 2>&1; then
            print_success "Clippy passed (no warnings)"
            ((PASSED++))
        else
            print_error "Clippy found issues"
            ((FAILED++))
            return 1
        fi
    fi
}

# Check baked shaders are fresh (#33). The offline slang->WGSL + reflection bake
# (graphics/src/shader/baked_generated.rs) can only be produced where the native
# Slang compiler is available, which needs the pinned SDK and the xtask `slang`
# feature. When the SDK is present, re-bake into memory and byte-compare against
# the committed file (never writing it), so a forgotten rebake fails preflight.
# When it is absent (a dev without the SDK, or a wasm-only checkout) skip with a
# warning, mirroring the wasm-pack skip above. The SDK is the repo-local `.slang`
# (scripts/fetch-slang.sh) unless $SLANG_DIR overrides it.
test_shader_bake_check() {
    local slang_dir="${SLANG_DIR:-.slang}"
    if [ ! -e "$slang_dir/lib" ]; then
        print_skip "Baked shader staleness check (Slang SDK not found — run scripts/fetch-slang.sh)"
        ((SKIPPED++))
        return
    fi

    print_header "Checking Baked Shaders Are Fresh (WASM)"

    if cargo run -q -p xtask --features slang -- bake-shaders --check 2>&1; then
        print_success "Baked shaders match their sources"
        ((PASSED++))
    else
        print_error "Baked shaders are stale — run: cargo run -p xtask --features slang -- bake-shaders"
        ((FAILED++))
    fi
}

# Print summary
print_summary() {
    print_header "Test Summary"

    echo -e "${GREEN}Passed:${NC}  $PASSED"
    echo -e "${RED}Failed:${NC}  $FAILED"
    echo -e "${YELLOW}Skipped:${NC} $SKIPPED"
    echo ""

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed.${NC}"
        return 1
    fi
}

# Main execution
main() {
    echo -e "${BLUE}RedLilium Engine - Test Suite${NC}"
    echo "Running comprehensive tests..."

    check_tools

    # Run all tests, collecting results
    test_native_build || true
    test_web_build || true
    test_shader_bake_check || true
    test_unit_tests || true
    test_clippy || true

    print_summary

    # Exit with appropriate code
    if [ $FAILED -gt 0 ]; then
        exit 1
    fi
}

main
