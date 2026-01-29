//! Virtual resources for the render graph

use crate::backend::types::*;

/// Unique identifier for a render graph resource
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub(crate) u32);

/// Virtual texture resource in the render graph
#[derive(Debug, Clone)]
pub struct VirtualTexture {
    pub id: ResourceId,
    pub desc: TextureDescriptor,
    pub name: String,
}

/// Virtual buffer resource in the render graph
#[derive(Debug, Clone)]
pub struct VirtualBuffer {
    pub id: ResourceId,
    pub desc: BufferDescriptor,
    pub name: String,
}

/// Resource type enumeration
#[derive(Debug, Clone)]
pub enum VirtualResource {
    Texture(VirtualTexture),
    Buffer(VirtualBuffer),
    /// External resource (like swapchain image)
    External(ResourceId),
}

impl VirtualResource {
    pub fn id(&self) -> ResourceId {
        match self {
            VirtualResource::Texture(t) => t.id,
            VirtualResource::Buffer(b) => b.id,
            VirtualResource::External(id) => *id,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            VirtualResource::Texture(t) => &t.name,
            VirtualResource::Buffer(b) => &b.name,
            VirtualResource::External(_) => "external",
        }
    }
}

/// How a pass uses a resource
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceUsage {
    /// Read as a texture (sampled)
    TextureRead,
    /// Write as a render target
    RenderTarget,
    /// Read/write as storage texture
    StorageRead,
    StorageWrite,
    StorageReadWrite,
    /// Depth/stencil attachment
    DepthStencilRead,
    DepthStencilWrite,
    /// Read as uniform buffer
    UniformBuffer,
    /// Read/write as storage buffer
    StorageBufferRead,
    StorageBufferWrite,
}

/// Resource access declaration for a pass
#[derive(Debug, Clone)]
pub struct ResourceAccess {
    pub resource: ResourceId,
    pub usage: ResourceUsage,
}

impl ResourceAccess {
    pub fn is_read(&self) -> bool {
        matches!(
            self.usage,
            ResourceUsage::TextureRead
                | ResourceUsage::StorageRead
                | ResourceUsage::DepthStencilRead
                | ResourceUsage::UniformBuffer
                | ResourceUsage::StorageBufferRead
        )
    }

    pub fn is_write(&self) -> bool {
        matches!(
            self.usage,
            ResourceUsage::RenderTarget
                | ResourceUsage::StorageWrite
                | ResourceUsage::StorageReadWrite
                | ResourceUsage::DepthStencilWrite
                | ResourceUsage::StorageBufferWrite
        )
    }
}

/// Describes texture dimensions that can be relative to screen size
#[derive(Debug, Clone, Copy)]
pub enum TextureSize {
    /// Absolute size in pixels
    Absolute { width: u32, height: u32 },
    /// Relative to screen size (1.0 = full screen)
    Relative { width_scale: f32, height_scale: f32 },
}

impl Default for TextureSize {
    fn default() -> Self {
        TextureSize::Relative {
            width_scale: 1.0,
            height_scale: 1.0,
        }
    }
}

impl TextureSize {
    pub fn resolve(&self, screen_width: u32, screen_height: u32) -> (u32, u32) {
        match self {
            TextureSize::Absolute { width, height } => (*width, *height),
            TextureSize::Relative {
                width_scale,
                height_scale,
            } => (
                ((screen_width as f32) * width_scale) as u32,
                ((screen_height as f32) * height_scale) as u32,
            ),
        }
    }
}
