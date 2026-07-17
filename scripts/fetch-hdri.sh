#!/usr/bin/env bash
# Fetch the pinned source HDRI for the IBL bake (#137) into the repo-local
# `.hdri/` directory (gitignored). Only the derived KTX2 artifacts in
# std-assets are checked in — this source is needed solely to (re)run
# `cargo run -p xtask -- bake-ibl`, which self-skips when it is absent.
#
# The environment is "Spruit Sunrise" from Poly Haven (CC0). The file is
# PINNED: the bake output is tied to these exact bytes — swap the environment
# and re-bake in one reviewed change.
set -euo pipefail

HDRI_NAME="spruit_sunrise_2k.hdr"
HDRI_URL="https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/${HDRI_NAME}"
# Optional integrity check, same policy as fetch-slang.sh / fetch-ktx.sh.
SHA256="be35af1e825dc7506df48bf86568f9c05f8a8552cf57edf36beb16e54534ec57"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/.hdri"

if [ -f "$DEST/$HDRI_NAME" ]; then
    echo "$HDRI_NAME already present in $DEST — nothing to do."
    exit 0
fi

mkdir -p "$DEST"
echo "Downloading $HDRI_URL"
curl --fail --location --progress-bar --output "$DEST/$HDRI_NAME" "$HDRI_URL"

digest="$(shasum -a 256 "$DEST/$HDRI_NAME" | awk '{print $1}')"
echo "sha256: $digest"
if [ -n "$SHA256" ]; then
    if [ "$digest" != "$SHA256" ]; then
        echo "error: checksum mismatch for $HDRI_NAME" >&2
        rm -f "$DEST/$HDRI_NAME"
        exit 1
    fi
    echo "checksum OK"
else
    echo "warning: no pinned checksum — paste the sha256 above into fetch-hdri.sh to enforce it." >&2
fi

echo "$HDRI_NAME provisioned in $DEST"
