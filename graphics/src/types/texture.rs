//! Texture types and descriptors.

use super::Extent3d;
use bitflags::bitflags;

// Re-export CPU-side types from core.
pub use redlilium_core::texture::{CpuTexture, TextureDimension, TextureFormat};

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
    /// Texture dimension (1D, 2D, 3D, Cube, CubeArray).
    pub dimension: TextureDimension,
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
            dimension: TextureDimension::D2,
            format,
            usage,
        }
    }

    /// Create a new cubemap texture descriptor.
    ///
    /// Cubemaps have 6 faces (layers): +X, -X, +Y, -Y, +Z, -Z.
    /// The size is the width/height of each face (must be square).
    pub fn new_cube(size: u32, format: TextureFormat, usage: TextureUsage) -> Self {
        Self {
            label: None,
            size: Extent3d::new_3d(size, size, 6),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::Cube,
            format,
            usage,
        }
    }

    /// Create a new 2D array texture descriptor.
    ///
    /// A 2D array is a stack of `layer_count` 2D textures sharing the same dimensions
    /// and format. Layers are indexed in shaders via the array index.
    pub fn new_2d_array(
        width: u32,
        height: u32,
        layer_count: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self {
        Self {
            label: None,
            size: Extent3d::new_3d(width, height, layer_count),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2Array,
            format,
            usage,
        }
    }

    /// Create a new 3D (volume) texture descriptor.
    pub fn new_3d(
        width: u32,
        height: u32,
        depth: u32,
        format: TextureFormat,
        usage: TextureUsage,
    ) -> Self {
        Self {
            label: None,
            size: Extent3d::new_3d(width, height, depth),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
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

    /// Set the texture dimension.
    pub fn with_dimension(mut self, dimension: TextureDimension) -> Self {
        self.dimension = dimension;
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
            dimension: TextureDimension::default(),
            format: TextureFormat::default(),
            usage: TextureUsage::empty(),
        }
    }
}
