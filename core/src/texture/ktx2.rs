//! KTX2 container parsing (#120).
//!
//! Parses a KTX2 file into a [`CpuTexture`]: mip chain, array layers, cube
//! faces, optional Zstandard supercompression. The output blob follows the
//! `CpuTexture` layout contract (mip-major, layers within a mip, no row
//! padding) — which matches KTX2's own per-level `layer → face → z` order,
//! so level payloads are appended verbatim after decompression.
//!
//! Universal formats are rejected at the seam #121 plugs into: UASTC and
//! ETC1S/BasisLZ payloads need a transcoder, not a decompressor, and come
//! back as [`Ktx2Error::RequiresTranscoder`].

use super::{CpuTexture, TextureDimension, TextureFormat};
use ktx2::{ColorModel, Format, SupercompressionScheme};

/// The 12-byte KTX2 file identifier.
pub const MAGIC: [u8; 12] = [
    0xAB, b'K', b'T', b'X', b' ', b'2', b'0', 0xBB, b'\r', b'\n', 0x1A, b'\n',
];

/// Whether `bytes` starts with the KTX2 magic — the loader's cheap sniff
/// before committing to a full parse.
pub fn is_ktx2(bytes: &[u8]) -> bool {
    bytes.len() >= MAGIC.len() && bytes[..MAGIC.len()] == MAGIC
}

/// Why a KTX2 file could not be turned into a [`CpuTexture`].
#[derive(Debug)]
pub enum Ktx2Error {
    /// The container is malformed or uses a spec feature with no engine
    /// consumer (e.g. ZLIB supercompression).
    Parse(String),
    /// The payload is a universal format (UASTC / ETC1S+BasisLZ) that must be
    /// transcoded to a device-native format first — the #121 seam.
    RequiresTranscoder(String),
    /// A well-formed vkFormat the engine has no [`TextureFormat`] for.
    UnsupportedVkFormat(u32),
}

impl std::fmt::Display for Ktx2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(msg) => write!(f, "KTX2 parse error: {msg}"),
            Self::RequiresTranscoder(what) => {
                write!(
                    f,
                    "KTX2 payload is {what}, which requires a transcoder (#121)"
                )
            }
            Self::UnsupportedVkFormat(raw) => {
                write!(f, "KTX2 vkFormat {raw} has no engine TextureFormat mapping")
            }
        }
    }
}

impl std::error::Error for Ktx2Error {}

/// Parse a KTX2 file into a [`CpuTexture`].
///
/// Supported: every vkFormat with a [`TextureFormat`] mapping, full mip
/// chains, 1D/2D/3D, array layers, cubemaps and cube arrays, Zstandard
/// supercompression. Every level's byte count is validated against the
/// format arithmetic before the data is accepted.
pub fn parse_ktx2(bytes: &[u8]) -> Result<CpuTexture, Ktx2Error> {
    let reader =
        ktx2::Reader::new(bytes).map_err(|e| Ktx2Error::Parse(format!("bad container: {e:?}")))?;
    let header = reader.header();

    // Universal-format rejection first: BasisLZ is a supercompression scheme,
    // UASTC/ETC1S hide behind VK_FORMAT_UNDEFINED with a DFD color model.
    if header.supercompression_scheme == Some(SupercompressionScheme::BasisLZ) {
        return Err(Ktx2Error::RequiresTranscoder("ETC1S/BasisLZ".to_string()));
    }
    let Some(vk_format) = header.format else {
        return Err(match reader.color_model() {
            Some(ColorModel::UASTC) => Ktx2Error::RequiresTranscoder("UASTC".to_string()),
            Some(ColorModel::ETC1S) => Ktx2Error::RequiresTranscoder("ETC1S".to_string()),
            other => Ktx2Error::Parse(format!(
                "VK_FORMAT_UNDEFINED with unrecognized DFD color model {other:?}"
            )),
        });
    };
    let format = map_vk_format(vk_format)?;

    // Shape. KTX2 encodes "not that kind of texture" as a 0 in the height/
    // depth/layer fields; face_count is 6 for cubemaps and 1 otherwise.
    if header.pixel_width == 0 {
        return Err(Ktx2Error::Parse("pixel_width is 0".to_string()));
    }
    let is_cube = match header.face_count {
        1 => false,
        6 => true,
        n => {
            return Err(Ktx2Error::Parse(format!(
                "face_count {n} (expected 1 or 6)"
            )));
        }
    };
    let is_array = header.layer_count > 0;
    let dimension = match (header.pixel_height, header.pixel_depth, is_cube, is_array) {
        (_, d, true, _) if d > 0 => {
            return Err(Ktx2Error::Parse("cubemap with pixel_depth > 0".to_string()));
        }
        (0, _, true, _) => {
            return Err(Ktx2Error::Parse("cubemap with pixel_height 0".to_string()));
        }
        (_, _, true, false) => TextureDimension::Cube,
        (_, _, true, true) => TextureDimension::CubeArray,
        (_, d, false, arr) if d > 0 => {
            if arr {
                return Err(Ktx2Error::Parse(
                    "3D texture with layer_count > 0".to_string(),
                ));
            }
            TextureDimension::D3
        }
        (0, _, false, false) => TextureDimension::D1,
        (0, _, false, true) => TextureDimension::D1Array,
        (_, _, false, false) => TextureDimension::D2,
        (_, _, false, true) => TextureDimension::D2Array,
    };
    let depth_or_array_layers = match dimension {
        TextureDimension::D3 => header.pixel_depth,
        TextureDimension::D1Array | TextureDimension::D2Array | TextureDimension::CubeArray => {
            header.layer_count
        }
        _ => 1,
    };

    // `level_count == 0` means "generate the chain yourself" — one stored mip.
    let mut cpu = CpuTexture {
        name: None,
        data: Vec::new(),
        width: header.pixel_width,
        height: header.pixel_height.max(1),
        format,
        dimension,
        mip_level_count: header.level_count.max(1),
        depth_or_array_layers,
    };

    let layers = cpu.layer_count() as usize;
    let mut blob = Vec::with_capacity(cpu.expected_data_len());
    let mut zstd = (header.supercompression_scheme == Some(SupercompressionScheme::Zstandard))
        .then(ruzstd::decoding::FrameDecoder::new);
    if let Some(scheme) = header.supercompression_scheme
        && scheme != SupercompressionScheme::Zstandard
    {
        return Err(Ktx2Error::Parse(format!(
            "supercompression scheme {scheme:?} has no engine decoder (only Zstandard)"
        )));
    }

    for (mip, level) in reader.levels().enumerate() {
        let expected = cpu.image_size_bytes(mip as u32) * layers;
        let start = blob.len();
        match &mut zstd {
            Some(decoder) => {
                blob.reserve(expected);
                decoder
                    .decode_all_to_vec(level.data, &mut blob)
                    .map_err(|e| Ktx2Error::Parse(format!("zstd (mip {mip}): {e}")))?;
            }
            None => blob.extend_from_slice(level.data),
        }
        let got = blob.len() - start;
        if got != expected {
            return Err(Ktx2Error::Parse(format!(
                "mip {mip}: {got} bytes, expected {expected} \
                 ({layers} layer(s) of {} bytes)",
                cpu.image_size_bytes(mip as u32)
            )));
        }
    }
    if blob.len() != cpu.expected_data_len() {
        return Err(Ktx2Error::Parse(format!(
            "level count mismatch: {} bytes total, expected {} for {} mips",
            blob.len(),
            cpu.expected_data_len(),
            cpu.mip_level_count
        )));
    }

    cpu.data = blob;
    Ok(cpu)
}

/// vkFormat → engine [`TextureFormat`]. The inverse of the Vulkan backend's
/// format table; a format outside it is a named error, never a silent
/// RGBA8 fallback.
fn map_vk_format(format: Format) -> Result<TextureFormat, Ktx2Error> {
    Ok(match format {
        Format::R8_UNORM => TextureFormat::R8Unorm,
        Format::R8_SNORM => TextureFormat::R8Snorm,
        Format::R8_UINT => TextureFormat::R8Uint,
        Format::R8_SINT => TextureFormat::R8Sint,
        Format::R8G8_UNORM => TextureFormat::Rg8Unorm,
        Format::R16_UNORM => TextureFormat::R16Unorm,
        Format::R16_SFLOAT => TextureFormat::R16Float,
        Format::R32_UINT => TextureFormat::R32Uint,
        Format::R32_SFLOAT => TextureFormat::R32Float,
        Format::R16G16_SFLOAT => TextureFormat::Rg16Float,
        Format::R32G32_SFLOAT => TextureFormat::Rg32Float,
        Format::R8G8B8A8_UNORM => TextureFormat::Rgba8Unorm,
        Format::R8G8B8A8_SRGB => TextureFormat::Rgba8UnormSrgb,
        Format::B8G8R8A8_UNORM => TextureFormat::Bgra8Unorm,
        Format::B8G8R8A8_SRGB => TextureFormat::Bgra8UnormSrgb,
        Format::A2B10G10R10_UNORM_PACK32 => TextureFormat::Rgba10a2Unorm,
        Format::A2R10G10B10_UNORM_PACK32 => TextureFormat::Bgra10a2Unorm,
        Format::R16G16B16A16_SFLOAT => TextureFormat::Rgba16Float,
        Format::R32G32B32A32_SFLOAT => TextureFormat::Rgba32Float,
        Format::D16_UNORM => TextureFormat::Depth16Unorm,
        Format::D32_SFLOAT => TextureFormat::Depth32Float,
        Format::D24_UNORM_S8_UINT => TextureFormat::Depth24PlusStencil8,
        Format::D32_SFLOAT_S8_UINT => TextureFormat::Depth32FloatStencil8,
        Format::BC1_RGBA_UNORM_BLOCK => TextureFormat::Bc1RgbaUnorm,
        Format::BC1_RGBA_SRGB_BLOCK => TextureFormat::Bc1RgbaUnormSrgb,
        Format::BC2_UNORM_BLOCK => TextureFormat::Bc2RgbaUnorm,
        Format::BC2_SRGB_BLOCK => TextureFormat::Bc2RgbaUnormSrgb,
        Format::BC3_UNORM_BLOCK => TextureFormat::Bc3RgbaUnorm,
        Format::BC3_SRGB_BLOCK => TextureFormat::Bc3RgbaUnormSrgb,
        Format::BC4_UNORM_BLOCK => TextureFormat::Bc4RUnorm,
        Format::BC4_SNORM_BLOCK => TextureFormat::Bc4RSnorm,
        Format::BC5_UNORM_BLOCK => TextureFormat::Bc5RgUnorm,
        Format::BC5_SNORM_BLOCK => TextureFormat::Bc5RgSnorm,
        Format::BC6H_UFLOAT_BLOCK => TextureFormat::Bc6hRgbUfloat,
        Format::BC6H_SFLOAT_BLOCK => TextureFormat::Bc6hRgbFloat,
        Format::BC7_UNORM_BLOCK => TextureFormat::Bc7RgbaUnorm,
        Format::BC7_SRGB_BLOCK => TextureFormat::Bc7RgbaUnormSrgb,
        Format::ETC2_R8G8B8_UNORM_BLOCK => TextureFormat::Etc2Rgb8Unorm,
        Format::ETC2_R8G8B8_SRGB_BLOCK => TextureFormat::Etc2Rgb8UnormSrgb,
        Format::ETC2_R8G8B8A1_UNORM_BLOCK => TextureFormat::Etc2Rgb8A1Unorm,
        Format::ETC2_R8G8B8A1_SRGB_BLOCK => TextureFormat::Etc2Rgb8A1UnormSrgb,
        Format::ETC2_R8G8B8A8_UNORM_BLOCK => TextureFormat::Etc2Rgba8Unorm,
        Format::ETC2_R8G8B8A8_SRGB_BLOCK => TextureFormat::Etc2Rgba8UnormSrgb,
        Format::EAC_R11_UNORM_BLOCK => TextureFormat::EacR11Unorm,
        Format::EAC_R11_SNORM_BLOCK => TextureFormat::EacR11Snorm,
        Format::EAC_R11G11_UNORM_BLOCK => TextureFormat::EacRg11Unorm,
        Format::EAC_R11G11_SNORM_BLOCK => TextureFormat::EacRg11Snorm,
        Format::ASTC_4x4_UNORM_BLOCK => TextureFormat::Astc4x4Unorm,
        Format::ASTC_4x4_SRGB_BLOCK => TextureFormat::Astc4x4UnormSrgb,
        Format::ASTC_5x4_UNORM_BLOCK => TextureFormat::Astc5x4Unorm,
        Format::ASTC_5x4_SRGB_BLOCK => TextureFormat::Astc5x4UnormSrgb,
        Format::ASTC_5x5_UNORM_BLOCK => TextureFormat::Astc5x5Unorm,
        Format::ASTC_5x5_SRGB_BLOCK => TextureFormat::Astc5x5UnormSrgb,
        Format::ASTC_6x5_UNORM_BLOCK => TextureFormat::Astc6x5Unorm,
        Format::ASTC_6x5_SRGB_BLOCK => TextureFormat::Astc6x5UnormSrgb,
        Format::ASTC_6x6_UNORM_BLOCK => TextureFormat::Astc6x6Unorm,
        Format::ASTC_6x6_SRGB_BLOCK => TextureFormat::Astc6x6UnormSrgb,
        Format::ASTC_8x5_UNORM_BLOCK => TextureFormat::Astc8x5Unorm,
        Format::ASTC_8x5_SRGB_BLOCK => TextureFormat::Astc8x5UnormSrgb,
        Format::ASTC_8x6_UNORM_BLOCK => TextureFormat::Astc8x6Unorm,
        Format::ASTC_8x6_SRGB_BLOCK => TextureFormat::Astc8x6UnormSrgb,
        Format::ASTC_8x8_UNORM_BLOCK => TextureFormat::Astc8x8Unorm,
        Format::ASTC_8x8_SRGB_BLOCK => TextureFormat::Astc8x8UnormSrgb,
        Format::ASTC_10x5_UNORM_BLOCK => TextureFormat::Astc10x5Unorm,
        Format::ASTC_10x5_SRGB_BLOCK => TextureFormat::Astc10x5UnormSrgb,
        Format::ASTC_10x6_UNORM_BLOCK => TextureFormat::Astc10x6Unorm,
        Format::ASTC_10x6_SRGB_BLOCK => TextureFormat::Astc10x6UnormSrgb,
        Format::ASTC_10x8_UNORM_BLOCK => TextureFormat::Astc10x8Unorm,
        Format::ASTC_10x8_SRGB_BLOCK => TextureFormat::Astc10x8UnormSrgb,
        Format::ASTC_10x10_UNORM_BLOCK => TextureFormat::Astc10x10Unorm,
        Format::ASTC_10x10_SRGB_BLOCK => TextureFormat::Astc10x10UnormSrgb,
        Format::ASTC_12x10_UNORM_BLOCK => TextureFormat::Astc12x10Unorm,
        Format::ASTC_12x10_SRGB_BLOCK => TextureFormat::Astc12x10UnormSrgb,
        Format::ASTC_12x12_UNORM_BLOCK => TextureFormat::Astc12x12Unorm,
        Format::ASTC_12x12_SRGB_BLOCK => TextureFormat::Astc12x12UnormSrgb,
        other => return Err(Ktx2Error::UnsupportedVkFormat(other.value())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal KTX2 writer for tests: header + level index + a minimal basic
    /// DFD block (the crate validates its presence) + level payloads.
    /// `levels` are the per-mip (possibly supercompressed) payloads with
    /// their uncompressed sizes, mip 0 first. `color_model` fills the DFD's
    /// colorModel byte (0 = unspecified; 163 = ETC1S, 166 = UASTC).
    #[allow(clippy::too_many_arguments)]
    fn build_ktx2(
        vk_format: u32,
        width: u32,
        height: u32,
        depth: u32,
        layer_count: u32,
        face_count: u32,
        scheme: u32,
        color_model: u8,
        levels: &[(Vec<u8>, u64)],
    ) -> Vec<u8> {
        const DFD_LEN: usize = 4 + 8 + 16; // totalSize + block header + basic block body
        let dfd_offset = 80 + levels.len() * 24;

        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        for v in [
            vk_format,
            1, /* typeSize */
            width,
            height,
            depth,
            layer_count,
            face_count,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(levels.len() as u32).to_le_bytes());
        out.extend_from_slice(&scheme.to_le_bytes());
        out.extend_from_slice(&(dfd_offset as u32).to_le_bytes());
        out.extend_from_slice(&(DFD_LEN as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; 8]); // kvd offset + length
        out.extend_from_slice(&[0u8; 16]); // sgd offset + length
        assert_eq!(out.len(), 80);

        // Level index (24 bytes per level), data laid out after the DFD.
        let mut offset = (dfd_offset + DFD_LEN) as u64;
        for (data, uncompressed) in levels {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&(data.len() as u64).to_le_bytes());
            out.extend_from_slice(&uncompressed.to_le_bytes());
            offset += data.len() as u64;
        }

        // DFD: totalSize, then one basic block — vendor 0 / type 0 (u32),
        // version 2 (u16), descriptorBlockSize 24 (u16), then the 16-byte
        // body starting with colorModel.
        out.extend_from_slice(&(DFD_LEN as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&24u16.to_le_bytes());
        let mut body = [0u8; 16];
        body[0] = color_model;
        out.extend_from_slice(&body);

        for (data, _) in levels {
            out.extend_from_slice(data);
        }
        out
    }

    #[test]
    fn magic_sniff() {
        assert!(!is_ktx2(b"not a ktx2 file"));
        assert!(!is_ktx2(&MAGIC[..11]));
        let file = build_ktx2(37, 1, 1, 0, 0, 1, 0, 0, &[(vec![0u8; 4], 4)]);
        assert!(is_ktx2(&file));
    }

    /// 4×4 RGBA8 with a full 3-mip chain: shape, per-level sizes, and the
    /// mip-major blob order all round-trip.
    #[test]
    fn parse_2d_mip_chain() {
        let levels = [
            (vec![0xAAu8; 4 * 4 * 4], (4 * 4 * 4) as u64),
            (vec![0xBBu8; 2 * 2 * 4], (2 * 2 * 4) as u64),
            (vec![0xCCu8; 4], 4),
        ];
        let file = build_ktx2(37 /* R8G8B8A8_UNORM */, 4, 4, 0, 0, 1, 0, 0, &levels);
        let cpu = parse_ktx2(&file).expect("parse");
        assert_eq!(cpu.format, TextureFormat::Rgba8Unorm);
        assert_eq!(cpu.dimension, TextureDimension::D2);
        assert_eq!((cpu.width, cpu.height), (4, 4));
        assert_eq!(cpu.mip_level_count, 3);
        assert_eq!(cpu.depth_or_array_layers, 1);
        assert_eq!(cpu.data.len(), cpu.expected_data_len());
        assert!(cpu.data[cpu.byte_range(0, 0)].iter().all(|&b| b == 0xAA));
        assert!(cpu.data[cpu.byte_range(1, 0)].iter().all(|&b| b == 0xBB));
        assert!(cpu.data[cpu.byte_range(2, 0)].iter().all(|&b| b == 0xCC));
    }

    /// NPOT compressed: 5×3 BC1 is 2×1 blocks (16 bytes) at mip 0; the edge
    /// mips still occupy whole blocks (8 bytes each).
    #[test]
    fn parse_npot_bc1_edge_mips() {
        let levels = [
            (vec![1u8; 16], 16),
            (vec![2u8; 8], 8), // 2×1
            (vec![3u8; 8], 8), // 1×1
        ];
        let file = build_ktx2(
            133, /* BC1_RGBA_UNORM_BLOCK */
            5, 3, 0, 0, 1, 0, 0, &levels,
        );
        let cpu = parse_ktx2(&file).expect("parse");
        assert_eq!(cpu.format, TextureFormat::Bc1RgbaUnorm);
        assert_eq!(cpu.mip_level_count, 3);
        assert_eq!(cpu.byte_range(0, 0), 0..16);
        assert_eq!(cpu.byte_range(1, 0), 16..24);
        assert_eq!(cpu.byte_range(2, 0), 24..32);
    }

    /// face_count 6 → Cube; faces are the layers, in KTX2 face order, and
    /// each mip stores all six faces contiguously.
    #[test]
    fn parse_cubemap_faces() {
        let face_bytes = 2 * 2 * 4;
        let mip0: Vec<u8> = (0..6u8).flat_map(|f| vec![f; face_bytes]).collect();
        let mip1: Vec<u8> = (0..6u8).flat_map(|f| vec![0x10 + f; 4]).collect();
        let levels = [(mip0, (face_bytes * 6) as u64), (mip1, 24)];
        let file = build_ktx2(43 /* R8G8B8A8_SRGB */, 2, 2, 0, 0, 6, 0, 0, &levels);
        let cpu = parse_ktx2(&file).expect("parse");
        assert_eq!(cpu.dimension, TextureDimension::Cube);
        assert_eq!(cpu.layer_count(), 6);
        assert_eq!(cpu.mip_level_count, 2);
        for face in 0..6u8 {
            assert!(
                cpu.data[cpu.byte_range(0, face as u32)]
                    .iter()
                    .all(|&b| b == face),
                "face {face} mip 0"
            );
            assert!(
                cpu.data[cpu.byte_range(1, face as u32)]
                    .iter()
                    .all(|&b| b == 0x10 + face),
                "face {face} mip 1"
            );
        }
    }

    /// Zstandard supercompression decodes per level and validates sizes.
    #[test]
    fn parse_zstd_supercompressed() {
        let raw = vec![0x5Au8; 4 * 4 * 4];
        let compressed = ruzstd::encoding::compress_to_vec(
            &raw[..],
            ruzstd::encoding::CompressionLevel::Fastest,
        );
        let levels = [(compressed, raw.len() as u64)];
        let file = build_ktx2(37, 4, 4, 0, 0, 1, 2 /* Zstandard */, 0, &levels);
        let cpu = parse_ktx2(&file).expect("parse");
        assert_eq!(cpu.data, raw);
    }

    /// BasisLZ payloads need the #121 transcoder, not a decompressor.
    #[test]
    fn basis_lz_requires_transcoder() {
        // uncompressed_byte_length is 0 for BasisLZ per spec; the payload is
        // opaque transcoder input.
        let file = build_ktx2(
            0, /* UNDEFINED */
            4,
            4,
            0,
            0,
            1,
            1, /* BasisLZ */
            163,
            &[(vec![0u8; 16], 0)],
        );
        let result = parse_ktx2(&file);
        assert!(
            matches!(result, Err(Ktx2Error::RequiresTranscoder(_))),
            "{result:?}"
        );
    }

    /// A legal vkFormat outside the engine's format table is a named error,
    /// not a silent RGBA8 fallback.
    #[test]
    fn unmapped_vk_format_is_named_error() {
        let file = build_ktx2(
            30, /* B8G8R8_UNORM */
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            &[(vec![0u8; 3], 3)],
        );
        assert!(matches!(
            parse_ktx2(&file),
            Err(Ktx2Error::UnsupportedVkFormat(30))
        ));
    }

    /// A level whose decompressed size disagrees with the format arithmetic
    /// is rejected.
    #[test]
    fn level_size_mismatch_rejected() {
        let levels = [(vec![0u8; 63], 63)]; // 4×4 RGBA8 needs 64
        let file = build_ktx2(37, 4, 4, 0, 0, 1, 0, 0, &levels);
        assert!(matches!(parse_ktx2(&file), Err(Ktx2Error::Parse(_))));
    }
}
