//! CPU-side texture types.
//!
//! Provides [`CpuTexture`] for holding raw pixel data, along with
//! [`TextureFormat`] and [`TextureDimension`] enums shared between
//! CPU and GPU code.

/// Texture dimension enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureDimension {
    /// 1D texture.
    D1,
    /// 1D array texture (multiple layers).
    D1Array,
    /// 2D texture (default).
    #[default]
    D2,
    /// 2D array texture (multiple layers).
    D2Array,
    /// 3D texture (volume).
    D3,
    /// Cubemap texture (6 faces).
    Cube,
    /// Cubemap array texture (N × 6 faces).
    CubeArray,
}

impl TextureDimension {
    /// Returns the number of array layers for this dimension.
    ///
    /// For cubemaps, this is 6. For cube arrays, multiply by 6.
    /// For other dimensions, it's the depth/array_layers from the extent.
    pub fn layer_count(&self, depth_or_array_layers: u32) -> u32 {
        match self {
            Self::D1 | Self::D1Array | Self::D2 | Self::D2Array | Self::D3 => depth_or_array_layers,
            Self::Cube => 6,
            Self::CubeArray => depth_or_array_layers * 6,
        }
    }

    /// Returns true if this is a cubemap or cubemap array dimension.
    pub fn is_cubemap(&self) -> bool {
        matches!(self, Self::Cube | Self::CubeArray)
    }

    /// Returns true if this is an array texture (2D array or cube array).
    pub fn is_array(&self) -> bool {
        matches!(self, Self::D1Array | Self::D2Array | Self::CubeArray)
    }
}

/// Texture format enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum TextureFormat {
    // 8-bit formats
    /// 8-bit red channel, unsigned normalized.
    R8Unorm,
    /// 8-bit red channel, signed normalized.
    R8Snorm,
    /// 8-bit red channel, unsigned integer.
    R8Uint,
    /// 8-bit red channel, signed integer.
    R8Sint,

    // 16-bit formats
    /// 16-bit red channel, unsigned normalized.
    R16Unorm,
    /// 16-bit red channel, float.
    R16Float,
    /// 8-bit RG channels, unsigned normalized.
    Rg8Unorm,

    // 32-bit formats
    /// 32-bit red channel, float.
    R32Float,
    /// 32-bit red channel, unsigned integer.
    R32Uint,
    /// 16-bit RG channels, float.
    Rg16Float,
    /// 8-bit RGBA channels, unsigned normalized.
    #[default]
    Rgba8Unorm,
    /// 8-bit RGBA channels, sRGB.
    Rgba8UnormSrgb,
    /// 8-bit BGRA channels, unsigned normalized.
    Bgra8Unorm,
    /// 8-bit BGRA channels, sRGB.
    Bgra8UnormSrgb,
    /// 10-bit RGB with 2-bit alpha, unsigned normalized (HDR10 compatible).
    Rgba10a2Unorm,
    /// 10-bit BGR with 2-bit alpha, unsigned normalized (HDR10 compatible).
    Bgra10a2Unorm,

    // 64-bit formats
    /// 16-bit RGBA channels, float.
    Rgba16Float,
    /// 32-bit RG channels, float.
    Rg32Float,

    // 128-bit formats
    /// 32-bit RGBA channels, float.
    Rgba32Float,

    // Depth/stencil formats
    /// 16-bit depth.
    Depth16Unorm,
    /// 24-bit depth.
    Depth24Plus,
    /// 24-bit depth with 8-bit stencil.
    Depth24PlusStencil8,
    /// 32-bit depth, float.
    Depth32Float,
    /// 32-bit depth float with 8-bit stencil.
    Depth32FloatStencil8,

    // BC compressed formats (desktop standard, 4x4 blocks)
    /// BC1 (DXT1) RGBA, unsigned normalized. 8 bytes per 4×4 block.
    Bc1RgbaUnorm,
    /// BC1 (DXT1) RGBA, sRGB. 8 bytes per 4×4 block.
    Bc1RgbaUnormSrgb,
    /// BC2 (DXT3) RGBA, unsigned normalized. 16 bytes per 4×4 block.
    Bc2RgbaUnorm,
    /// BC2 (DXT3) RGBA, sRGB. 16 bytes per 4×4 block.
    Bc2RgbaUnormSrgb,
    /// BC3 (DXT5) RGBA, unsigned normalized. 16 bytes per 4×4 block.
    Bc3RgbaUnorm,
    /// BC3 (DXT5) RGBA, sRGB. 16 bytes per 4×4 block.
    Bc3RgbaUnormSrgb,
    /// BC4 single-channel, unsigned normalized. 8 bytes per 4×4 block.
    Bc4RUnorm,
    /// BC4 single-channel, signed normalized. 8 bytes per 4×4 block.
    Bc4RSnorm,
    /// BC5 two-channel, unsigned normalized. 16 bytes per 4×4 block.
    Bc5RgUnorm,
    /// BC5 two-channel, signed normalized. 16 bytes per 4×4 block.
    Bc5RgSnorm,
    /// BC6H RGB HDR, unsigned float. 16 bytes per 4×4 block.
    Bc6hRgbUfloat,
    /// BC6H RGB HDR, signed float. 16 bytes per 4×4 block.
    Bc6hRgbFloat,
    /// BC7 RGBA, unsigned normalized. 16 bytes per 4×4 block.
    Bc7RgbaUnorm,
    /// BC7 RGBA, sRGB. 16 bytes per 4×4 block.
    Bc7RgbaUnormSrgb,

    // ETC2 compressed formats (mobile standard, 4x4 blocks)
    /// ETC2 RGB, unsigned normalized. 8 bytes per 4×4 block.
    Etc2Rgb8Unorm,
    /// ETC2 RGB, sRGB. 8 bytes per 4×4 block.
    Etc2Rgb8UnormSrgb,
    /// ETC2 RGB with 1-bit alpha, unsigned normalized. 8 bytes per 4×4 block.
    Etc2Rgb8A1Unorm,
    /// ETC2 RGB with 1-bit alpha, sRGB. 8 bytes per 4×4 block.
    Etc2Rgb8A1UnormSrgb,
    /// ETC2 RGBA, unsigned normalized. 16 bytes per 4×4 block.
    Etc2Rgba8Unorm,
    /// ETC2 RGBA, sRGB. 16 bytes per 4×4 block.
    Etc2Rgba8UnormSrgb,
    /// EAC single-channel, unsigned normalized. 8 bytes per 4×4 block.
    EacR11Unorm,
    /// EAC single-channel, signed normalized. 8 bytes per 4×4 block.
    EacR11Snorm,
    /// EAC two-channel, unsigned normalized. 16 bytes per 4×4 block.
    EacRg11Unorm,
    /// EAC two-channel, signed normalized. 16 bytes per 4×4 block.
    EacRg11Snorm,

    // ASTC compressed formats (128 bits = 16 bytes per block, variable block sizes)
    /// ASTC 4×4, unsigned normalized.
    Astc4x4Unorm,
    /// ASTC 4×4, sRGB.
    Astc4x4UnormSrgb,
    /// ASTC 5×4, unsigned normalized.
    Astc5x4Unorm,
    /// ASTC 5×4, sRGB.
    Astc5x4UnormSrgb,
    /// ASTC 5×5, unsigned normalized.
    Astc5x5Unorm,
    /// ASTC 5×5, sRGB.
    Astc5x5UnormSrgb,
    /// ASTC 6×5, unsigned normalized.
    Astc6x5Unorm,
    /// ASTC 6×5, sRGB.
    Astc6x5UnormSrgb,
    /// ASTC 6×6, unsigned normalized.
    Astc6x6Unorm,
    /// ASTC 6×6, sRGB.
    Astc6x6UnormSrgb,
    /// ASTC 8×5, unsigned normalized.
    Astc8x5Unorm,
    /// ASTC 8×5, sRGB.
    Astc8x5UnormSrgb,
    /// ASTC 8×6, unsigned normalized.
    Astc8x6Unorm,
    /// ASTC 8×6, sRGB.
    Astc8x6UnormSrgb,
    /// ASTC 8×8, unsigned normalized.
    Astc8x8Unorm,
    /// ASTC 8×8, sRGB.
    Astc8x8UnormSrgb,
    /// ASTC 10×5, unsigned normalized.
    Astc10x5Unorm,
    /// ASTC 10×5, sRGB.
    Astc10x5UnormSrgb,
    /// ASTC 10×6, unsigned normalized.
    Astc10x6Unorm,
    /// ASTC 10×6, sRGB.
    Astc10x6UnormSrgb,
    /// ASTC 10×8, unsigned normalized.
    Astc10x8Unorm,
    /// ASTC 10×8, sRGB.
    Astc10x8UnormSrgb,
    /// ASTC 10×10, unsigned normalized.
    Astc10x10Unorm,
    /// ASTC 10×10, sRGB.
    Astc10x10UnormSrgb,
    /// ASTC 12×10, unsigned normalized.
    Astc12x10Unorm,
    /// ASTC 12×10, sRGB.
    Astc12x10UnormSrgb,
    /// ASTC 12×12, unsigned normalized.
    Astc12x12Unorm,
    /// ASTC 12×12, sRGB.
    Astc12x12UnormSrgb,
}

impl TextureFormat {
    /// Returns true if this is a depth or stencil format.
    pub fn is_depth_stencil(&self) -> bool {
        matches!(
            self,
            Self::Depth16Unorm
                | Self::Depth24Plus
                | Self::Depth24PlusStencil8
                | Self::Depth32Float
                | Self::Depth32FloatStencil8
        )
    }

    /// Returns true if this format has a stencil component.
    pub fn has_stencil(&self) -> bool {
        matches!(self, Self::Depth24PlusStencil8 | Self::Depth32FloatStencil8)
    }

    /// Returns true if this is an sRGB format (hardware applies linear→sRGB on write).
    ///
    /// sRGB formats automatically convert linear color values to sRGB gamma space
    /// when writing to the framebuffer. Shaders should output linear values when
    /// rendering to an sRGB surface.
    pub fn is_srgb(&self) -> bool {
        matches!(
            self,
            Self::Rgba8UnormSrgb
                | Self::Bgra8UnormSrgb
                | Self::Bc1RgbaUnormSrgb
                | Self::Bc2RgbaUnormSrgb
                | Self::Bc3RgbaUnormSrgb
                | Self::Bc7RgbaUnormSrgb
                | Self::Etc2Rgb8UnormSrgb
                | Self::Etc2Rgb8A1UnormSrgb
                | Self::Etc2Rgba8UnormSrgb
                | Self::Astc4x4UnormSrgb
                | Self::Astc5x4UnormSrgb
                | Self::Astc5x5UnormSrgb
                | Self::Astc6x5UnormSrgb
                | Self::Astc6x6UnormSrgb
                | Self::Astc8x5UnormSrgb
                | Self::Astc8x6UnormSrgb
                | Self::Astc8x8UnormSrgb
                | Self::Astc10x5UnormSrgb
                | Self::Astc10x6UnormSrgb
                | Self::Astc10x8UnormSrgb
                | Self::Astc10x10UnormSrgb
                | Self::Astc12x10UnormSrgb
                | Self::Astc12x12UnormSrgb
        )
    }

    /// Returns true if this is an HDR (High Dynamic Range) format.
    ///
    /// HDR formats have higher precision or wider color gamut than standard 8-bit formats.
    /// This includes 10-bit formats (HDR10) and floating-point formats.
    pub fn is_hdr(&self) -> bool {
        matches!(
            self,
            Self::Rgba10a2Unorm
                | Self::Bgra10a2Unorm
                | Self::Rgba16Float
                | Self::Rgba32Float
                | Self::R16Float
                | Self::Rg16Float
                | Self::R32Float
                | Self::Rg32Float
                | Self::Bc6hRgbUfloat
                | Self::Bc6hRgbFloat
        )
    }

    /// Returns true if this is a compressed (block-based) texture format.
    pub fn is_compressed(&self) -> bool {
        matches!(
            self,
            Self::Bc1RgbaUnorm
                | Self::Bc1RgbaUnormSrgb
                | Self::Bc2RgbaUnorm
                | Self::Bc2RgbaUnormSrgb
                | Self::Bc3RgbaUnorm
                | Self::Bc3RgbaUnormSrgb
                | Self::Bc4RUnorm
                | Self::Bc4RSnorm
                | Self::Bc5RgUnorm
                | Self::Bc5RgSnorm
                | Self::Bc6hRgbUfloat
                | Self::Bc6hRgbFloat
                | Self::Bc7RgbaUnorm
                | Self::Bc7RgbaUnormSrgb
                | Self::Etc2Rgb8Unorm
                | Self::Etc2Rgb8UnormSrgb
                | Self::Etc2Rgb8A1Unorm
                | Self::Etc2Rgb8A1UnormSrgb
                | Self::Etc2Rgba8Unorm
                | Self::Etc2Rgba8UnormSrgb
                | Self::EacR11Unorm
                | Self::EacR11Snorm
                | Self::EacRg11Unorm
                | Self::EacRg11Snorm
                | Self::Astc4x4Unorm
                | Self::Astc4x4UnormSrgb
                | Self::Astc5x4Unorm
                | Self::Astc5x4UnormSrgb
                | Self::Astc5x5Unorm
                | Self::Astc5x5UnormSrgb
                | Self::Astc6x5Unorm
                | Self::Astc6x5UnormSrgb
                | Self::Astc6x6Unorm
                | Self::Astc6x6UnormSrgb
                | Self::Astc8x5Unorm
                | Self::Astc8x5UnormSrgb
                | Self::Astc8x6Unorm
                | Self::Astc8x6UnormSrgb
                | Self::Astc8x8Unorm
                | Self::Astc8x8UnormSrgb
                | Self::Astc10x5Unorm
                | Self::Astc10x5UnormSrgb
                | Self::Astc10x6Unorm
                | Self::Astc10x6UnormSrgb
                | Self::Astc10x8Unorm
                | Self::Astc10x8UnormSrgb
                | Self::Astc10x10Unorm
                | Self::Astc10x10UnormSrgb
                | Self::Astc12x10Unorm
                | Self::Astc12x10UnormSrgb
                | Self::Astc12x12Unorm
                | Self::Astc12x12UnormSrgb
        )
    }

    /// Returns the block dimensions (width, height) for this format.
    ///
    /// Uncompressed formats return (1, 1). Compressed formats return the
    /// block size in texels (e.g., (4, 4) for BC/ETC2 formats).
    pub fn block_dimensions(&self) -> (u32, u32) {
        match self {
            // All BC formats use 4×4 blocks
            Self::Bc1RgbaUnorm
            | Self::Bc1RgbaUnormSrgb
            | Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbFloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb => (4, 4),

            // All ETC2/EAC formats use 4×4 blocks
            Self::Etc2Rgb8Unorm
            | Self::Etc2Rgb8UnormSrgb
            | Self::Etc2Rgb8A1Unorm
            | Self::Etc2Rgb8A1UnormSrgb
            | Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb
            | Self::EacR11Unorm
            | Self::EacR11Snorm
            | Self::EacRg11Unorm
            | Self::EacRg11Snorm => (4, 4),

            // ASTC formats — variable block sizes
            Self::Astc4x4Unorm | Self::Astc4x4UnormSrgb => (4, 4),
            Self::Astc5x4Unorm | Self::Astc5x4UnormSrgb => (5, 4),
            Self::Astc5x5Unorm | Self::Astc5x5UnormSrgb => (5, 5),
            Self::Astc6x5Unorm | Self::Astc6x5UnormSrgb => (6, 5),
            Self::Astc6x6Unorm | Self::Astc6x6UnormSrgb => (6, 6),
            Self::Astc8x5Unorm | Self::Astc8x5UnormSrgb => (8, 5),
            Self::Astc8x6Unorm | Self::Astc8x6UnormSrgb => (8, 6),
            Self::Astc8x8Unorm | Self::Astc8x8UnormSrgb => (8, 8),
            Self::Astc10x5Unorm | Self::Astc10x5UnormSrgb => (10, 5),
            Self::Astc10x6Unorm | Self::Astc10x6UnormSrgb => (10, 6),
            Self::Astc10x8Unorm | Self::Astc10x8UnormSrgb => (10, 8),
            Self::Astc10x10Unorm | Self::Astc10x10UnormSrgb => (10, 10),
            Self::Astc12x10Unorm | Self::Astc12x10UnormSrgb => (12, 10),
            Self::Astc12x12Unorm | Self::Astc12x12UnormSrgb => (12, 12),

            // Uncompressed formats
            _ => (1, 1),
        }
    }

    /// Returns the size in bytes per pixel (uncompressed) or per block (compressed).
    pub fn block_size(&self) -> u32 {
        match self {
            Self::R8Unorm | Self::R8Snorm | Self::R8Uint | Self::R8Sint => 1,
            Self::R16Unorm | Self::R16Float | Self::Rg8Unorm | Self::Depth16Unorm => 2,
            Self::R32Float
            | Self::R32Uint
            | Self::Rg16Float
            | Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Rgba10a2Unorm
            | Self::Bgra10a2Unorm
            | Self::Depth24Plus
            | Self::Depth24PlusStencil8
            | Self::Depth32Float => 4,
            Self::Rgba16Float | Self::Rg32Float | Self::Depth32FloatStencil8 => 8,
            Self::Rgba32Float => 16,

            // BC formats: 8 or 16 bytes per 4×4 block
            Self::Bc1RgbaUnorm | Self::Bc1RgbaUnormSrgb | Self::Bc4RUnorm | Self::Bc4RSnorm => 8,
            Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbFloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb => 16,

            // ETC2/EAC formats: 8 or 16 bytes per 4×4 block
            Self::Etc2Rgb8Unorm
            | Self::Etc2Rgb8UnormSrgb
            | Self::Etc2Rgb8A1Unorm
            | Self::Etc2Rgb8A1UnormSrgb
            | Self::EacR11Unorm
            | Self::EacR11Snorm => 8,
            Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb
            | Self::EacRg11Unorm
            | Self::EacRg11Snorm => 16,

            // ASTC formats: always 16 bytes (128 bits) per block
            Self::Astc4x4Unorm
            | Self::Astc4x4UnormSrgb
            | Self::Astc5x4Unorm
            | Self::Astc5x4UnormSrgb
            | Self::Astc5x5Unorm
            | Self::Astc5x5UnormSrgb
            | Self::Astc6x5Unorm
            | Self::Astc6x5UnormSrgb
            | Self::Astc6x6Unorm
            | Self::Astc6x6UnormSrgb
            | Self::Astc8x5Unorm
            | Self::Astc8x5UnormSrgb
            | Self::Astc8x6Unorm
            | Self::Astc8x6UnormSrgb
            | Self::Astc8x8Unorm
            | Self::Astc8x8UnormSrgb
            | Self::Astc10x5Unorm
            | Self::Astc10x5UnormSrgb
            | Self::Astc10x6Unorm
            | Self::Astc10x6UnormSrgb
            | Self::Astc10x8Unorm
            | Self::Astc10x8UnormSrgb
            | Self::Astc10x10Unorm
            | Self::Astc10x10UnormSrgb
            | Self::Astc12x10Unorm
            | Self::Astc12x10UnormSrgb
            | Self::Astc12x12Unorm
            | Self::Astc12x12UnormSrgb => 16,
        }
    }
}

/// CPU-side texture data.
///
/// Holds raw pixel data along with dimensions and format metadata.
/// This is the CPU-side counterpart to a GPU texture resource.
#[derive(Debug, Clone)]
pub struct CpuTexture {
    /// Texture name.
    pub name: Option<String>,
    /// Raw pixel data.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: TextureFormat,
    /// Texture dimension.
    pub dimension: TextureDimension,
}

impl CpuTexture {
    /// Create a new 2D CPU texture.
    pub fn new(width: u32, height: u32, format: TextureFormat, data: Vec<u8>) -> Self {
        Self {
            name: None,
            data,
            width,
            height,
            format,
            dimension: TextureDimension::D2,
        }
    }

    /// Set the texture name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the texture dimension.
    pub fn with_dimension(mut self, dimension: TextureDimension) -> Self {
        self.dimension = dimension;
        self
    }
}
