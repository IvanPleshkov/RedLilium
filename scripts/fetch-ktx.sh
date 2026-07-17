#!/usr/bin/env bash
# Provision the pinned KTX-Software CLI (`ktx`) into the repo-local `.ktx/`
# directory (gitignored). Needed only to (re)generate compressed-texture
# fixtures and, later, to run the #122 bake step — a default build needs none
# of this (the derived KTX2 artifacts are checked in).
#
#   scripts/fetch-ktx.sh
#   .ktx/bin/ktx --version
#
# The version is PINNED: encoder output is not stable across releases, so
# regenerated fixtures/bakes are tied to this exact release. Bump it together
# with a regeneration, in one reviewed change.
set -euo pipefail

KTX_VERSION="4.4.2"

# Optional integrity check, same policy as fetch-slang.sh: fill with the real
# sha256 (the script prints the digest it downloaded) to enforce; left empty,
# the script proceeds with a loud warning.
SHA256_macos_arm64=""
SHA256_macos_x64=""
SHA256_linux_x64=""
SHA256_linux_arm64=""

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/.ktx"

# Already provisioned at the pinned version? Nothing to do (idempotent).
if [ -x "$DEST/bin/ktx" ] && "$DEST/bin/ktx" --version 2>/dev/null | grep -q "$KTX_VERSION"; then
    echo "ktx $KTX_VERSION already present in $DEST — nothing to do."
    exit 0
fi

os="$(uname -s)"
arch="$(uname -m)"
case "$os/$arch" in
    Darwin/arm64)  asset="KTX-Software-${KTX_VERSION}-Darwin-arm64.pkg";      kind="pkg"; want="$SHA256_macos_arm64" ;;
    Darwin/x86_64) asset="KTX-Software-${KTX_VERSION}-Darwin-x86_64.pkg";     kind="pkg"; want="$SHA256_macos_x64" ;;
    Linux/x86_64)  asset="KTX-Software-${KTX_VERSION}-Linux-x86_64.tar.bz2";  kind="tbz"; want="$SHA256_linux_x64" ;;
    Linux/aarch64) asset="KTX-Software-${KTX_VERSION}-Linux-arm64.tar.bz2";   kind="tbz"; want="$SHA256_linux_arm64" ;;
    *)
        echo "error: no KTX-Software fetch mapping for $os/$arch." >&2
        echo "Download KTX-Software ${KTX_VERSION} for your platform from" >&2
        echo "  https://github.com/KhronosGroup/KTX-Software/releases/tag/v${KTX_VERSION}" >&2
        echo "and install it so that $DEST/bin/ktx exists." >&2
        exit 1
        ;;
esac

URL="https://github.com/KhronosGroup/KTX-Software/releases/download/v${KTX_VERSION}/${asset}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $URL"
# curl (not a browser) so macOS does not stamp the com.apple.quarantine xattr.
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
    echo "         (paste the sha256 above into fetch-ktx.sh to enforce it.)" >&2
fi

echo "Extracting into $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"
case "$kind" in
    pkg)
        # Expand the installer without installing (no sudo). The distribution
        # holds sub-packages (tools, library, dev, jni), each with a
        # Payload/usr/local tree — merge every payload's usr/local into DEST
        # so the CLI finds libktx next to itself.
        pkgutil --expand-full "$tmp/$asset" "$tmp/x"
        found=0
        for payload in "$tmp"/x/*.pkg/Payload/usr/local; do
            [ -d "$payload" ] || continue
            cp -R "$payload"/. "$DEST"/
            found=1
        done
        [ "$found" = 1 ] || { echo "error: no Payload/usr/local in $asset" >&2; exit 1; }
        ;;
    tbz)
        mkdir -p "$tmp/x"
        tar -xjf "$tmp/$asset" -C "$tmp/x"
        inner="$(find "$tmp/x" -maxdepth 3 -type d -name bin | head -1)"
        [ -n "$inner" ] || { echo "error: no bin/ dir in $asset" >&2; exit 1; }
        cp -R "$(dirname "$inner")"/. "$DEST"/
        ;;
esac

# Strip quarantine and ad-hoc re-sign so the unsigned binaries run on arm64
# regardless of how they arrived.
if [ "$os" = "Darwin" ]; then
    xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
    codesign --force --sign - "$DEST"/bin/* "$DEST"/lib/*.dylib 2>/dev/null || true
fi

"$DEST/bin/ktx" --version
echo "ktx $KTX_VERSION provisioned in $DEST"
