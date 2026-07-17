//! Texture types and descriptors.

use super::Extent3d;
use bitflags::bitflags;

// Re-export CPU-side types from core.
pub use redlilium_core::texture::{CompressionFamily, CpuTexture, TextureDimension, TextureFormat};

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
    /// Whether this texture may be accessed by BOTH the graphics queue and
    /// the async compute queue (#88).
    ///
    /// Declared textures are created `VK_SHARING_MODE_CONCURRENT` on devices
    /// with a distinct compute queue family, which can cost framebuffer
    /// compression (DCC) on some hardware; undeclared textures stay
    /// `EXCLUSIVE` and keep compression, but a graph that touches one is
    /// never routed to the async compute queue (the async hint falls back to
    /// the graphics queue). Buffers need no declaration — compression does
    /// not apply to them and they are always cross-queue-capable.
    pub cross_queue: bool,
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
            cross_queue: false,
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
            cross_queue: false,
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
            cross_queue: false,
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
            cross_queue: false,
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

    /// Declare that this texture may be accessed by both the graphics and
    /// the async compute queue (see the `cross_queue` field).
    pub fn with_cross_queue(mut self, cross_queue: bool) -> Self {
        self.cross_queue = cross_queue;
        self
    }

    /// Array layer count and real depth implied by `dimension` and `size`.
    ///
    /// Matches how `create_texture` interprets the descriptor on both
    /// backends: `size.depth` holds the layer count for array textures, the
    /// cube count for `CubeArray` (× 6 faces), and the actual depth only for
    /// `D3`. `Cube` is always 6 layers regardless of `size.depth`.
    pub fn layers_and_depth(&self) -> (u32, u32) {
        match self.dimension {
            TextureDimension::D1 | TextureDimension::D2 => (1, 1),
            TextureDimension::D1Array | TextureDimension::D2Array => (self.size.depth.max(1), 1),
            TextureDimension::D3 => (1, self.size.depth.max(1)),
            TextureDimension::Cube => (6, 1),
            TextureDimension::CubeArray => (self.size.depth.max(1) * 6, 1),
        }
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
            cross_queue: false,
        }
    }
}
