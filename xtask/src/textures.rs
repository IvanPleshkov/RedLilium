//! Offline bake of the std PBR test textures (ADR-039).
//!
//! ```bash
//! cargo run -p xtask -- bake-textures
//! ```
//!
//! Generates the deterministic procedural set under
//! `std-assets/textures/pbr/` as Zstd-supercompressed KTX2 with baked mips:
//!
//! - `checker_basecolor.ktx2` — sRGB two-tone checker (UV + color-space
//!   check);
//! - `uv_grid_basecolor.ktx2` — sRGB UV gradient with grid lines
//!   (orientation/tiling check);
//! - `checker_orm.ktx2` — linear ORM (R = AO, G = roughness, B = metallic):
//!   tiles alternate polished metal / matte dielectric, so a wrong channel
//!   packing is visible at a glance.
//!
//! Mips are box-filtered in **linear** space (sRGB payloads decode →
//! average → re-encode). Patterns are pure functions of the texel
//! coordinate and zstd is pinned, so a re-run is byte-identical.

use std::path::Path;

const SIZE: u32 = 256;
const ZSTD_LEVEL: i32 = 19;

/// VkFormat codes understood by `redlilium_core::texture::ktx2::parse_ktx2`.
const VK_FORMAT_R8G8B8A8_UNORM: u32 = 37;
const VK_FORMAT_R8G8B8A8_SRGB: u32 = 43;

pub fn run() {
    let out_dir = Path::new("std-assets/textures/pbr");
    std::fs::create_dir_all(out_dir).expect("create textures/pbr dir");

    // Two-tone checker: warm light gray vs deep red, 8x8 tiles.
    write_texture(
        &out_dir.join("checker_basecolor.ktx2"),
        VK_FORMAT_R8G8B8A8_SRGB,
        true,
        |u, v| {
            let tile = (u * 8.0) as u32 + (v * 8.0) as u32;
            if tile.is_multiple_of(2) {
                [0.72, 0.70, 0.66, 1.0]
            } else {
                [0.42, 0.045, 0.04, 1.0]
            }
        },
    );

    // UV gradient (R = u, G = v) with dark grid lines every 1/8th.
    write_texture(
        &out_dir.join("uv_grid_basecolor.ktx2"),
        VK_FORMAT_R8G8B8A8_SRGB,
        true,
        |u, v| {
            let on_line = |x: f32| (x * 8.0).fract() < 0.06;
            if on_line(u) || on_line(v) {
                [0.05, 0.05, 0.06, 1.0]
            } else {
                [u, v, 0.25, 1.0]
            }
        },
    );

    // ORM checker (linear): R = AO (vignette per tile), G = roughness,
    // B = metallic. Alternating tiles: polished metal / matte dielectric.
    write_texture(
        &out_dir.join("checker_orm.ktx2"),
        VK_FORMAT_R8G8B8A8_UNORM,
        false,
        |u, v| {
            let tu = (u * 8.0).fract();
            let tv = (v * 8.0).fract();
            let tile = (u * 8.0) as u32 + (v * 8.0) as u32;
            // Soft AO darkening toward tile edges — reads as grout lines.
            let edge = (tu.min(1.0 - tu).min(tv).min(1.0 - tv) * 8.0).min(1.0);
            let ao = 0.55 + 0.45 * edge;
            if tile.is_multiple_of(2) {
                [ao, 0.15, 1.0, 1.0] // polished metal
            } else {
                [ao, 0.85, 0.0, 1.0] // matte dielectric
            }
        },
    );
}

/// Evaluate `texel` over a SIZE×SIZE grid, build the linear-space mip chain,
/// encode (sRGB or linear) to RGBA8, and write the KTX2 file.
fn write_texture(path: &Path, vk_format: u32, srgb: bool, texel: impl Fn(f32, f32) -> [f32; 4]) {
    // Mip 0 in linear f32.
    let mut linear: Vec<[f32; 4]> = Vec::with_capacity((SIZE * SIZE) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let u = (x as f32 + 0.5) / SIZE as f32;
            let v = (y as f32 + 0.5) / SIZE as f32;
            linear.push(texel(u, v));
        }
    }

    // Box-filtered chain down to 1×1, averaging in linear space.
    let mut levels_linear = vec![linear];
    let mut size = SIZE;
    while size > 1 {
        let half = size / 2;
        let prev = levels_linear.last().unwrap();
        let mut next = Vec::with_capacity((half * half) as usize);
        for y in 0..half {
            for x in 0..half {
                let mut acc = [0.0f32; 4];
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    let p = prev[((y * 2 + dy) * size + x * 2 + dx) as usize];
                    for c in 0..4 {
                        acc[c] += p[c];
                    }
                }
                next.push([acc[0] / 4.0, acc[1] / 4.0, acc[2] / 4.0, acc[3] / 4.0]);
            }
        }
        levels_linear.push(next);
        size = half;
    }

    let levels: Vec<Vec<u8>> = levels_linear
        .iter()
        .map(|level| {
            let mut bytes = Vec::with_capacity(level.len() * 4);
            for px in level {
                for (c, &value) in px.iter().enumerate() {
                    // Alpha stays linear even in sRGB payloads.
                    let encoded = if srgb && c < 3 {
                        srgb_encode(value)
                    } else {
                        value
                    };
                    bytes.push((encoded.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            bytes
        })
        .collect();

    let file = to_ktx2_rgba8(vk_format, srgb, SIZE, &levels);
    std::fs::write(path, &file).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!(
        "wrote {} ({} mips, {} bytes)",
        path.display(),
        levels.len(),
        file.len()
    );
}

fn srgb_encode(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// Serialize an RGBA8 2D mip chain as KTX2 with Zstandard supercompression —
/// the 8-bit sibling of the IBL bake's writer (`ibl::Ktx2Output`), kept
/// parseable by `redlilium_core::texture::ktx2::parse_ktx2` (round-trip
/// tested below).
fn to_ktx2_rgba8(vk_format: u32, srgb: bool, size: u32, levels: &[Vec<u8>]) -> Vec<u8> {
    for (mip, level) in levels.iter().enumerate() {
        let dim = (size >> mip).max(1) as usize;
        assert_eq!(level.len(), dim * dim * 4, "level {mip} payload size");
    }
    let compressed: Vec<Vec<u8>> = levels
        .iter()
        .map(|level| zstd::encode_all(&level[..], ZSTD_LEVEL).expect("zstd encode"))
        .collect();

    const MAGIC: [u8; 12] = [
        0xAB, b'K', b'T', b'X', b' ', b'2', b'0', 0xBB, b'\r', b'\n', 0x1A, b'\n',
    ];
    let channels = 4u32;
    let dfd_len: u32 = 4 + 8 + 16 + 16 * channels;
    let level_count = levels.len();
    let index_off = 80usize;
    let dfd_off = index_off + level_count * 24;
    let data_off = dfd_off as u64 + dfd_len as u64;

    // Physical placement: smallest mip first (KTX2 streaming order).
    let mut offsets = vec![0u64; level_count];
    let mut cursor = data_off;
    for mip in (0..level_count).rev() {
        offsets[mip] = cursor;
        cursor += compressed[mip].len() as u64;
    }

    let mut out = Vec::with_capacity(cursor as usize);
    out.extend_from_slice(&MAGIC);
    for v in [
        vk_format,
        1, // typeSize: 8-bit components
        size,
        size,
        0, // pixelDepth
        0, // layerCount
        1, // faceCount
        level_count as u32,
        2, // supercompressionScheme: Zstandard
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(dfd_off as u32).to_le_bytes());
    out.extend_from_slice(&dfd_len.to_le_bytes());
    out.extend_from_slice(&[0u8; 8]); // kvd offset + length
    out.extend_from_slice(&[0u8; 16]); // sgd offset + length
    assert_eq!(out.len(), index_off);

    for (mip, comp) in compressed.iter().enumerate() {
        out.extend_from_slice(&offsets[mip].to_le_bytes());
        out.extend_from_slice(&(comp.len() as u64).to_le_bytes());
        out.extend_from_slice(&(levels[mip].len() as u64).to_le_bytes());
    }

    // Basic DFD, version 2: colorModel RGBSDA, primaries BT709, transfer
    // sRGB or LINEAR, 1×1 texel block of 4 bytes; one 8-bit UNORM sample per
    // channel. For sRGB payloads the alpha sample carries the LINEAR
    // qualifier (alpha is never sRGB-encoded).
    assert_eq!(out.len(), dfd_off);
    out.extend_from_slice(&dfd_len.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&((24 + 16 * channels) as u16).to_le_bytes());
    let mut body = [0u8; 16];
    body[0] = 1; // KHR_DF_MODEL_RGBSDA
    body[1] = 1; // KHR_DF_PRIMARIES_BT709
    body[2] = if srgb { 2 } else { 1 }; // KHR_DF_TRANSFER_SRGB / _LINEAR
    body[8] = 4; // bytesPlane0
    out.extend_from_slice(&body);
    for channel in 0..channels {
        // RGBSDA channel ids: R=0, G=1, B=2, A=15; alpha in an sRGB DFD is
        // additionally marked LINEAR (0x40).
        let channel_id: u32 = if channel == 3 { 15 | 0x40 } else { channel };
        // bitOffset(8*c) | (bitLength-1)(8) | channelType+qualifiers(8).
        let word0 = (8 * channel) | (7 << 16) | (channel_id << 24);
        out.extend_from_slice(&word0.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // samplePositions
        out.extend_from_slice(&0u32.to_le_bytes()); // sampleLower: 0
        out.extend_from_slice(&255u32.to_le_bytes()); // sampleUpper: 255
    }

    for mip in (0..level_count).rev() {
        assert_eq!(out.len() as u64, offsets[mip]);
        out.extend_from_slice(&compressed[mip]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::texture::{TextureDimension, TextureFormat, ktx2::parse_ktx2};

    /// The writer and the engine parser agree end to end for both transfer
    /// functions, including the full mip chain byte-for-byte.
    #[test]
    fn rgba8_ktx2_round_trip() {
        let levels = vec![
            (0..4 * 4 * 4).map(|i| i as u8).collect::<Vec<u8>>(),
            (0..2 * 2 * 4).map(|i| (i * 3) as u8).collect(),
            vec![1, 2, 3, 4],
        ];
        for (vk_format, format) in [
            (VK_FORMAT_R8G8B8A8_SRGB, TextureFormat::Rgba8UnormSrgb),
            (VK_FORMAT_R8G8B8A8_UNORM, TextureFormat::Rgba8Unorm),
        ] {
            let file = to_ktx2_rgba8(vk_format, vk_format == VK_FORMAT_R8G8B8A8_SRGB, 4, &levels);
            let cpu = parse_ktx2(&file).expect("parse own output");
            assert_eq!(cpu.format, format);
            assert_eq!(cpu.dimension, TextureDimension::D2);
            assert_eq!((cpu.width, cpu.height), (4, 4));
            assert_eq!(cpu.mip_level_count, 3);
            for (mip, level) in levels.iter().enumerate() {
                assert_eq!(
                    &cpu.data[cpu.byte_range(mip as u32, 0)],
                    &level[..],
                    "mip {mip}"
                );
            }
        }
    }
}
