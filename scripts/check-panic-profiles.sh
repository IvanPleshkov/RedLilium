#!/bin/bash
#
# CI guard: verify no crate uses panic=abort in any profile.
#
# Panic=abort breaks panic catching for the Play/Pause/Resume/Stop flow.
# All profiles must use panic=unwind (the default) to support panic recovery.
#
# Usage:
#   bash scripts/check-panic-profiles.sh
#
# Exit codes:
#   0 = all profiles safe (panic != abort)
#   1 = panic=abort found in workspace or any crate
#

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

# Check workspace Cargo.toml for any profile setting panic=abort
echo "Checking workspace Cargo.toml profiles..."
if grep -E '^\[profile\.' Cargo.toml > /dev/null 2>&1; then
    if grep -A 10 '^\[profile\.' Cargo.toml | grep -E 'panic\s*=\s*"abort"' > /dev/null 2>&1; then
        echo "FAIL: workspace Cargo.toml has panic=abort in profile"
        exit 1
    fi
fi

# Check each crate's Cargo.toml for panic=abort
CRATES=(core ecs ecs-macro graphics app runtime demos editor debug_drawer vfs assets)
for crate in "${CRATES[@]}"; do
    CRATE_TOML="$crate/Cargo.toml"
    if [ -f "$CRATE_TOML" ]; then
        echo "Checking $CRATE_TOML..."
        if grep -E '^\[profile\.' "$CRATE_TOML" > /dev/null 2>&1; then
            if grep -A 10 '^\[profile\.' "$CRATE_TOML" | grep -E 'panic\s*=\s*"abort"' > /dev/null 2>&1; then
                echo "FAIL: $CRATE_TOML has panic=abort in profile"
                exit 1
            fi
        fi
    fi
done

echo "PASS: All profiles use panic=unwind (panic catching enabled)"
exit 0
