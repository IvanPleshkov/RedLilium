#!/usr/bin/env bash
# Regenerate the compressed-texture demo fixtures (#120) in
# demos/assets/compressed/: procedural source PNGs plus their BC7/BC5/BC4
# KTX2 encodings (full mip chains, Zstd-supercompressed).
#
# Needs the pinned KTX-Software CLI:  scripts/fetch-ktx.sh
#
# The BC files are produced through the UASTC route (`ktx create --encode
# uastc` → `ktx transcode --target bcN`) because KTX-Software ships no direct
# BC encoder — the same route #121 implements at load time. Encoder output is
# tied to the pinned KTX version; regenerate and bump fetch-ktx.sh together.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KTX="$ROOT/.ktx/bin/ktx"
OUT="$ROOT/demos/assets/compressed"

[ -x "$KTX" ] || { echo "error: $KTX not found — run scripts/fetch-ktx.sh first" >&2; exit 1; }
mkdir -p "$OUT"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Procedural 64×64 sources: an albedo with hue banding that shows BC7 decode
# errors, a hemisphere-bump normal map, and a radial grayscale ramp that
# shows BC4 banding. Pure-stdlib python (zlib/struct) — no PIL dependency.
python3 - "$OUT" << 'EOF'
import zlib, struct, math, sys

out = sys.argv[1]

def write_png(path, w, h, pixels, gray=False):
    color_type = 0 if gray else 2
    raw = b''
    for row in pixels:
        raw += b'\x00'
        for px in row:
            raw += bytes([px] if gray else px)
    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, color_type, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, 9))
    png += chunk(b'IEND', b'')
    open(f'{out}/{path}', 'wb').write(png)

W = H = 64
albedo = [[(
    (x*4) % 256 if x < W//2 else 255 - (y*3) % 200,
    (y*4) % 256,
    255 - (x*2+y*2) % 256,
) for x in range(W)] for y in range(H)]
write_png('albedo.png', W, H, albedo)

normal = []
for y in range(H):
    row = []
    for x in range(W):
        dx, dy = (x - W/2 + .5) / (W/2), (y - H/2 + .5) / (H/2)
        r2 = dx*dx + dy*dy
        if r2 < 0.9:
            nz = math.sqrt(max(0.0, 1.0 - r2))
            n = (dx, dy, nz)
        else:
            n = (0.0, 0.0, 1.0)
        row.append(tuple(int(round((c * 0.5 + 0.5) * 255)) for c in n))
    normal.append(row)
write_png('normal.png', W, H, normal)

gray = [[int(255 * max(0.0, 1.0 - math.hypot(x-W/2+.5, y-H/2+.5)/(W/2)))
         for x in range(W)] for y in range(H)]
write_png('gray.png', W, H, gray, gray=True)
print('source PNGs written')
EOF

# PNG → UASTC KTX2 (full mip chain) → BC-transcoded KTX2 with Zstd.
"$KTX" create --format R8G8B8A8_SRGB --generate-mipmap --encode uastc --uastc-quality 2 \
    "$OUT/albedo.png" "$tmp/albedo_uastc.ktx2"
"$KTX" transcode --target bc7 --zstd 18 "$tmp/albedo_uastc.ktx2" "$OUT/albedo_bc7.ktx2"

"$KTX" create --format R8G8B8A8_UNORM --assign-tf linear --generate-mipmap --encode uastc \
    --uastc-quality 2 "$OUT/normal.png" "$tmp/normal_uastc.ktx2"
"$KTX" transcode --target bc5 --zstd 18 "$tmp/normal_uastc.ktx2" "$OUT/normal_bc5.ktx2"

"$KTX" create --format R8_UNORM --assign-tf linear --generate-mipmap --encode uastc \
    --uastc-quality 2 "$OUT/gray.png" "$tmp/gray_uastc.ktx2"
"$KTX" transcode --target bc4 --zstd 18 "$tmp/gray_uastc.ktx2" "$OUT/gray_bc4.ktx2"

for f in albedo_bc7 normal_bc5 gray_bc4; do
    "$KTX" info "$OUT/$f.ktx2" | grep -E "vkFormat|levelCount|supercompressionScheme"
done
echo "fixtures regenerated in $OUT"
