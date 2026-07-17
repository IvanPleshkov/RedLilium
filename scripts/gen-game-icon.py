#!/usr/bin/env python3
"""Generate the car-game icon set (#133): game/icon/{icon.png,icon.icns,icon.ico}.

Pure stdlib (zlib + struct) — no PIL/numpy — so the icon is reproducible on
any machine with python3. The .icns step shells out to `sips`/`iconutil` and
therefore only runs on macOS; the master PNG and the .ico are cross-platform.

Design: flat red arcade car on a dark rounded square (Big Sur-style margin).
"""

import math
import os
import struct
import subprocess
import sys
import tempfile
import zlib

SIZE = 1024
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "game", "icon")


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def sdf_rounded_rect(px, py, cx, cy, hw, hh, r):
    """Signed distance to a rounded rect centered at (cx,cy)."""
    qx = abs(px - cx) - (hw - r)
    qy = abs(py - cy) - (hh - r)
    ox = max(qx, 0.0)
    oy = max(qy, 0.0)
    return math.hypot(ox, oy) + min(max(qx, qy), 0.0) - r


def sdf_circle(px, py, cx, cy, r):
    return math.hypot(px - cx, py - cy) - r


def coverage(d):
    return clamp(0.5 - d, 0.0, 1.0)


def over(dst, src, cov):
    """Blend src RGBA over dst RGBA with extra coverage factor."""
    a = src[3] / 255.0 * cov
    if a <= 0.0:
        return dst
    out_a = a + dst[3] / 255.0 * (1.0 - a)
    if out_a <= 0.0:
        return (0, 0, 0, 0)
    return tuple(
        int(round((src[i] * a + dst[i] * (dst[3] / 255.0) * (1.0 - a)) / out_a))
        for i in range(3)
    ) + (int(round(out_a * 255.0)),)


def render():
    rows = []
    for y in range(SIZE):
        row = bytearray()
        # Background gradient endpoints (dark slate, slightly lighter on top).
        t = y / (SIZE - 1)
        bg = tuple(
            int(round(a + (b - a) * t)) for a, b in ((0x24, 0x12), (0x28, 0x14), (0x31, 0x19))
        )
        for x in range(SIZE):
            px, py = x + 0.5, y + 0.5
            p = (0, 0, 0, 0)
            # Rounded-square plate: 856x856 centered, radius 200.
            d = sdf_rounded_rect(px, py, 512, 512, 428, 428, 200)
            p = over(p, (bg[0], bg[1], bg[2], 255), coverage(d))
            # Cabin (darker red).
            d = sdf_rounded_rect(px, py, 512, 515, 140, 85, 60)
            p = over(p, (0xC6, 0x2F, 0x28, 255), coverage(d))
            # Window (background-colored inset).
            d = sdf_rounded_rect(px, py, 512, 511, 110, 49, 36)
            p = over(p, (0x17, 0x1A, 0x20, 255), coverage(d))
            # Body (brighter red).
            d = sdf_rounded_rect(px, py, 512, 618, 244, 58, 58)
            p = over(p, (0xE8, 0x43, 0x3C, 255), coverage(d))
            # Headlight.
            d = sdf_circle(px, py, 728, 598, 13)
            p = over(p, (0xFF, 0xD9, 0x8A, 255), coverage(d))
            # Wheels + hubs.
            for wx in (392, 632):
                d = sdf_circle(px, py, wx, 676, 66)
                p = over(p, (0x10, 0x13, 0x18, 255), coverage(d))
                d = sdf_circle(px, py, wx, 676, 26)
                p = over(p, (0x45, 0x4C, 0x58, 255), coverage(d))
            row += bytes(p)
        rows.append(bytes(row))
    return rows


def write_png(path, rows, width, height):
    raw = b"".join(b"\x00" + r for r in rows)

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def resize_png(src, dst, size):
    subprocess.run(
        ["sips", "-z", str(size), str(size), src, "--out", dst],
        check=True,
        capture_output=True,
    )


def write_icns(master, out_path):
    """Build an .iconset with sips and compile it with iconutil (macOS only)."""
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "icon.iconset")
        os.mkdir(iconset)
        for size in (16, 32, 128, 256, 512):
            resize_png(master, os.path.join(iconset, f"icon_{size}x{size}.png"), size)
            resize_png(
                master, os.path.join(iconset, f"icon_{size}x{size}@2x.png"), size * 2
            )
        subprocess.run(
            ["iconutil", "-c", "icns", iconset, "-o", out_path],
            check=True,
            capture_output=True,
        )


def write_ico(master, out_path):
    """ICO with PNG-compressed entries (supported since Windows Vista)."""
    sizes = (16, 24, 32, 48, 64, 128, 256)
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        for size in sizes:
            p = os.path.join(tmp, f"{size}.png")
            resize_png(master, p, size)
            with open(p, "rb") as f:
                images.append((size, f.read()))
    header = struct.pack("<HHH", 0, 1, len(images))
    entries = b""
    offset = len(header) + 16 * len(images)
    blobs = b""
    for size, data in images:
        dim = 0 if size == 256 else size  # 0 means 256 in ICO directories
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    with open(out_path, "wb") as f:
        f.write(header + entries + blobs)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    master = os.path.join(OUT_DIR, "icon.png")
    print("rendering icon.png (1024x1024)...")
    write_png(master, render(), SIZE, SIZE)
    if sys.platform == "darwin":
        print("building icon.icns...")
        write_icns(master, os.path.join(OUT_DIR, "icon.icns"))
        print("building icon.ico...")
        write_ico(master, os.path.join(OUT_DIR, "icon.ico"))
    else:
        print("skipping .icns/.ico (needs macOS sips/iconutil)")
    print("done:", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
