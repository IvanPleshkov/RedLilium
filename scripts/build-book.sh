#!/usr/bin/env bash
# ============================================================================
# build-book.sh - Build mdbook documentation for RedLilium Engine
# ============================================================================
# This script builds the mdbook and optionally generates a single-page
# portable HTML file with all assets inlined.
#
# Usage: ./scripts/build-book.sh [OPTIONS]
#
# Options:
#   --single-page    Generate a single portable HTML file
#   --open           Open the book in the default browser after building
#   --clean          Remove previous build output before building
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
SINGLE_PAGE=false
OPEN_BOOK=false
CLEAN=false

# Resolve project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BOOK_DIR="$PROJECT_ROOT/book"
BOOK_OUTPUT="$BOOK_DIR/book"
SINGLE_PAGE_OUTPUT="$BOOK_DIR/redlilium-book.html"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --single-page)
            SINGLE_PAGE=true
            shift
            ;;
        --open)
            OPEN_BOOK=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            head -16 "$0" | tail -12
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

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check required tools
check_tools() {
    print_header "Checking Required Tools"

    if command -v mdbook &> /dev/null; then
        MDBOOK_VERSION=$(mdbook --version)
        print_success "mdbook found: $MDBOOK_VERSION"
    else
        print_error "mdbook not found. Installing..."
        cargo install mdbook
        print_success "mdbook installed"
    fi

    if [ "$SINGLE_PAGE" = true ]; then
        if command -v monolith &> /dev/null; then
            MONOLITH_VERSION=$(monolith --version)
            print_success "monolith found: $MONOLITH_VERSION"
        else
            print_error "monolith not found. Installing..."
            cargo install monolith
            print_success "monolith installed"
        fi
    fi
}

# Clean previous build
clean_build() {
    if [ "$CLEAN" = true ]; then
        print_info "Cleaning previous build..."
        rm -rf "$BOOK_OUTPUT"
        rm -f "$SINGLE_PAGE_OUTPUT"
    fi
}

# Build the book
build_book() {
    print_header "Building mdbook"

    if mdbook build "$BOOK_DIR"; then
        print_success "mdbook built successfully"
        print_info "Output: $BOOK_OUTPUT"
    else
        print_error "mdbook build failed"
        exit 1
    fi
}

# Generate single-page HTML
build_single_page() {
    if [ "$SINGLE_PAGE" = false ]; then
        return
    fi

    print_header "Generating Single-Page HTML"

    PRINT_HTML="$BOOK_OUTPUT/print.html"
    if [ ! -f "$PRINT_HTML" ]; then
        print_error "print.html not found at $PRINT_HTML"
        exit 1
    fi

    if monolith "$PRINT_HTML" -o "$SINGLE_PAGE_OUTPUT" 2>/dev/null; then
        # Patch the single-page HTML:
        # 1. Remove print button and print CSS
        # 2. Rewrite internal .html links to in-page #anchors
        python3 -c "
import re, sys

with open(sys.argv[1], 'r') as f:
    html = f.read()

# Remove print button, print CSS, and auto-print JS
html = re.sub(r'<a [^>]*title=\"Print this book\"[^>]*>.*?</a>', '', html, flags=re.DOTALL)
html = re.sub(r'<link[^>]*media=\"print\"[^>]*>', '', html)
html = re.sub(r'window\.setTimeout\(window\.print.*?\);?', '', html)

# Build link map by matching TOC hrefs (in order) to h1 ids (in order)
toc_hrefs = re.findall(r'href=\"([^\"]*?\.html)\"', html)
# Filter to only internal chapter links (not external urls or file:// paths)
toc_hrefs = [h for h in toc_hrefs if not h.startswith('http') and not h.startswith('file:')]
# Deduplicate preserving order
seen = set()
unique_hrefs = []
for h in toc_hrefs:
    if h not in seen:
        seen.add(h)
        unique_hrefs.append(h)

h1_ids = re.findall(r'<h1 id=\"([^\"]+)\">', html)

link_map = {}
for href, anchor in zip(unique_hrefs, h1_ids):
    link_map[href] = '#' + anchor

def replace_link(m):
    href = m.group(1)
    if href in link_map:
        return 'href=\"' + link_map[href] + '\"'
    return m.group(0)

html = re.sub(r'href=\"([^\"]*?\.html)\"', replace_link, html)

with open(sys.argv[1], 'w') as f:
    f.write(html)
" "$SINGLE_PAGE_OUTPUT"

        FILE_SIZE=$(du -h "$SINGLE_PAGE_OUTPUT" | cut -f1 | xargs)
        print_success "Single-page HTML generated ($FILE_SIZE)"
        print_info "Output: $SINGLE_PAGE_OUTPUT"
    else
        print_error "Failed to generate single-page HTML"
        exit 1
    fi
}

# Open the book
open_book() {
    if [ "$OPEN_BOOK" = false ]; then
        return
    fi

    if [ "$SINGLE_PAGE" = true ] && [ -f "$SINGLE_PAGE_OUTPUT" ]; then
        TARGET="$SINGLE_PAGE_OUTPUT"
    else
        TARGET="$BOOK_OUTPUT/index.html"
    fi

    print_info "Opening $TARGET"

    case "$(uname)" in
        Darwin)  open "$TARGET" ;;
        Linux)   xdg-open "$TARGET" ;;
        MINGW*|MSYS*|CYGWIN*)  start "$TARGET" ;;
        *)       print_error "Don't know how to open browser on $(uname)" ;;
    esac
}

# Main execution
main() {
    echo -e "${BLUE}RedLilium Engine - Book Builder${NC}"

    check_tools
    clean_build
    build_book
    build_single_page
    open_book

    print_header "Done"
    print_success "Book build complete"
}

main
