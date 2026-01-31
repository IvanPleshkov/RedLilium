//! Texture types and descriptors.

use super::Extent3d;
use bitflags::bitflags;

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
            | Self::Depth24Plus
            | Self::Depth24PlusStencil8
            | Self::Depth32Float => 4,
            Self::Rgba16Float | Self::Rg32Float | Self::Depth32FloatStencil8 => 8,
            Self::Rgba32Float => 16,
        }
    }
}

bitflags! {
    /// Usage flags for textures.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TextureUsage: u32 {
        /// Texture can be copied from.
        const COPY_SRC = 1 << 0;
        /// Texture can be copied to.
        const COPY_DST = 1 << 1;
        /// Texture can be sampled in a shader.
        const TEXTURE_BINDING = 1 << 2;
        /// Texture can be used as a storage texture.
        const STORAGE_BINDING = 1 << 3;
        /// Texture can be used as a render attachment.
        const RENDER_ATTACHMENT = 1 << 4;
    }
}

impl Default for TextureUsage {
    fn default() -> Self {
        Self::empty()
    }
}

/// Descriptor for creating a texture.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureDescriptor {
    /// Debug label for the texture.
    pub label: Option<String>,
    /// Size of the texture.
    pub size: Extent3d,
    /// Mip level count.
    pub mip_level_count: u32,
    /// Sample count for multisampling.
    pub sample_count: u32,
    /// Texture format.
    pub format: TextureFormat,
    /// Usage flags.
    pub usage: TextureUsage,
}

impl TextureDescriptor {
    /// Create a new 2D texture descriptor.
    pub fn new_2d(width: u32, height: u32, format: TextureFormat, usage: TextureUsage) -> Self {
        Self {
            label: None,
            size: Extent3d::new_2d(width, height),
            mip_level_count: 1,
            sample_count: 1,
            format,
            usage,
        }
    }

    /// Set the debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the mip level count.
    pub fn with_mip_levels(mut self, count: u32) -> Self {
        self.mip_level_count = count;
        self
    }

    /// Set the sample count for multisampling.
    pub fn with_sample_count(mut self, count: u32) -> Self {
        self.sample_count = count;
        self
    }
}

impl Default for TextureDescriptor {
    fn default() -> Self {
        Self {
            label: None,
            size: Extent3d::default(),
            mip_level_count: 1,
            sample_count: 1,
            format: TextureFormat::default(),
            usage: TextureUsage::empty(),
        }
    }
}
