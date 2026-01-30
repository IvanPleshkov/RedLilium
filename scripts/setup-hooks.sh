#!/usr/bin/env bash
# ============================================================================
# setup-hooks.sh - Install git hooks for RedLilium Engine
# ============================================================================
# This script installs the project's git hooks by creating symlinks from
# .git/hooks to the scripts/hooks directory.
#
# Usage: ./scripts/setup-hooks.sh
# ============================================================================

set -e

# Colors for output
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_SOURCE="$SCRIPT_DIR/hooks"
HOOKS_TARGET="$PROJECT_ROOT/.git/hooks"

echo -e "${BLUE}Setting up git hooks for RedLilium Engine${NC}"
echo ""

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo -e "${RED}Error: Not a git repository. Please run this script from the project root.${NC}"
    exit 1
fi

# Check if hooks source directory exists
if [ ! -d "$HOOKS_SOURCE" ]; then
    echo -e "${RED}Error: Hooks source directory not found: $HOOKS_SOURCE${NC}"
    exit 1
fi

# Install each hook
for hook in "$HOOKS_SOURCE"/*; do
    if [ -f "$hook" ]; then
        hook_name=$(basename "$hook")
        target="$HOOKS_TARGET/$hook_name"

        # Remove existing hook if it exists
        if [ -e "$target" ] || [ -L "$target" ]; then
            rm "$target"
            echo -e "${YELLOW}Replaced existing hook:${NC} $hook_name"
        fi

        # Create symlink
        ln -s "$hook" "$target"
        chmod +x "$hook"
        echo -e "${GREEN}Installed hook:${NC} $hook_name"
    fi
done

echo ""
echo -e "${GREEN}Git hooks installed successfully!${NC}"
echo ""
echo "The following hooks are now active:"
echo "  - pre-commit: Runs 'cargo fmt --check' before each commit"
echo ""
echo -e "${YELLOW}Note:${NC} To bypass hooks temporarily, use: git commit --no-verify"
