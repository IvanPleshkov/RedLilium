#!/usr/bin/env bash
# Provision the pinned Slang SDK into the repo-local `.slang/` directory (which is
# gitignored). This is the ONLY thing a shader developer needs to run before
# building with the Slang compiler:
#
#   scripts/fetch-slang.sh
#   cargo run -p xtask --features slang -- bake-shaders          # re-bake shaders
#   cargo run -p redlilium-editor --features slang-shaders       # author live
#
# A default build needs none of this — it uses the committed baked WGSL table.
#
# The SDK is a large PREBUILT release from github.com/shader-slang/slang (we do
# NOT build Slang from source). The version is PINNED: Slang's WGSL emission is
# not stable across versions, so the baked table is tied to this exact release
# (see graphics/src/shader/baked_generated.rs `BAKED_SLANG_TAG`). Bump both here
# and by re-baking, in one reviewed change.
set -euo pipefail

SLANG_VERSION="2026.8"

# Optional integrity check. Fill these with the real sha256 of each release
# archive (the script prints the digest it downloaded) and they will be enforced;
# left empty, the script proceeds with a loud warning instead of failing.
SHA256_macos_arm64="e182d44b403f3e5d78d66e7608cf6db1b81119883426fa9cc30f6af70c7ed554"
SHA256_macos_x64="b2ae6d712134c7d1c841261f1606f51fa3474895598672887a1a951a58759245"
SHA256_linux_x64="b23af8f2569e7961ca143c6ecd9abf6d7ac14a9bda739a1c6281694f774fa3eb"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/.slang"

# Already provisioned at the pinned version? Nothing to do (idempotent).
if [ -f "$DEST/lib/libslang.dylib" ] || [ -f "$DEST/lib/libslang.so" ]; then
    if ls "$DEST"/lib/*"$SLANG_VERSION"* >/dev/null 2>&1; then
        echo "Slang $SLANG_VERSION already present in $DEST — nothing to do."
        exit 0
    fi
    echo "A different Slang build is in $DEST; replacing it with $SLANG_VERSION."
fi

# Map uname -> Slang release asset name + archive type + expected checksum.
os="$(uname -s)"
arch="$(uname -m)"
case "$os/$arch" in
    Darwin/arm64)         asset="slang-${SLANG_VERSION}-macos-aarch64.zip";     kind="zip"; want="$SHA256_macos_arm64" ;;
    Darwin/x86_64)        asset="slang-${SLANG_VERSION}-macos-x86_64.zip";      kind="zip"; want="$SHA256_macos_x64" ;;
    Linux/x86_64)         asset="slang-${SLANG_VERSION}-linux-x86_64.tar.gz";   kind="tgz"; want="$SHA256_linux_x64" ;;
    *)
        echo "error: no Slang fetch mapping for $os/$arch." >&2
        echo "Download slang-${SLANG_VERSION} for your platform from" >&2
        echo "  https://github.com/shader-slang/slang/releases/tag/v${SLANG_VERSION}" >&2
        echo "and extract it so that \$SLANG_DIR ($DEST) has bin/ include/ lib/." >&2
        exit 1
        ;;
esac

URL="https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/${asset}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $URL"
# curl (not a browser) so macOS does not stamp the com.apple.quarantine xattr,
# which would make Gatekeeper kill the unsigned dylibs on arm64.
curl --fail --location --progress-bar --output "$tmp/$asset" "$URL"

digest="$(shasum -a 256 "$tmp/$asset" | awk '{print $1}')"
echo "sha256: $digest"
if [ -n "$want" ]; then
    if [ "$digest" != "$want" ]; then
        echo "error: checksum mismatch for $asset" >&2
        echo "  expected $want" >&2
        echo "  got      $digest" >&2
        exit 1
    fi
    echo "checksum OK"
else
    echo "warning: no pinned checksum for this platform — skipping integrity check." >&2
    echo "         (paste the sha256 above into fetch-slang.sh to enforce it.)" >&2
fi

echo "Extracting into $DEST"
mkdir -p "$tmp/x"
case "$kind" in
    zip) unzip -q "$tmp/$asset" -d "$tmp/x" ;;
    tgz) tar -xzf "$tmp/$asset" -C "$tmp/x" ;;
esac

# Releases may unpack either directly (bin/ include/ lib/ at top) or nested in a
# single versioned directory — normalize to the layout $SLANG_DIR expects.
src="$tmp/x"
if [ ! -d "$src/lib" ]; then
    inner="$(find "$tmp/x" -maxdepth 2 -type d -name lib | head -1)"
    [ -n "$inner" ] || { echo "error: no lib/ dir in $asset" >&2; exit 1; }
    src="$(dirname "$inner")"
fi

rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$src"/. "$DEST"/

# Belt and suspenders on macOS: strip any quarantine and ad-hoc re-sign so the
# unsigned dylibs load on arm64 regardless of how they arrived.
if [ "$os" = "Darwin" ]; then
    xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
    codesign --force --sign - "$DEST"/lib/*.dylib 2>/dev/null || true
fi

echo "Slang $SLANG_VERSION provisioned in $DEST"
