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
    /// 2D texture (default).
    #[default]
    D2,
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
            Self::D1 | Self::D2 | Self::D3 => depth_or_array_layers,
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
        matches!(self, Self::CubeArray)
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
        )
    }

    /// Returns the size in bytes per pixel/block.
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
