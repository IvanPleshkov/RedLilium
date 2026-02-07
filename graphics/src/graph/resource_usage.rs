//! Resource usage declarations for automatic barrier generation.
//!
//! This module defines how textures and buffers are used within passes, enabling
//! automatic layout tracking and barrier placement by the Vulkan backend.

use std::sync::Arc;

use crate::resources::{Buffer, Texture};

#[cfg(feature = "vulkan-backend")]
use crate::backend::vulkan::layout::TextureLayout;

/// How a texture is used within a pass.
///
/// Each access mode corresponds to a specific Vulkan image layout that
/// the texture must be in for the operation to be valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureAccessMode {
    /// Written as color render target (color attachment).
    RenderTargetWrite,
    /// Written as depth/stencil render target (depth attachment).
    DepthStencilWrite,
    /// Read-only depth/stencil (sampling + depth test).
    DepthStencilReadOnly,
    /// Sampled in a shader (texture read).
    ShaderRead,
    /// Read/write as storage texture.
    StorageReadWrite,
    /// Source of a copy/transfer operation.
    TransferRead,
    /// Destination of a copy/transfer operation.
    TransferWrite,
}

impl TextureAccessMode {
    /// Convert to the required Vulkan image layout.
    #[cfg(feature = "vulkan-backend")]
    pub fn to_layout(self) -> TextureLayout {
        match self {
            Self::RenderTargetWrite => TextureLayout::ColorAttachment,
            Self::DepthStencilWrite => TextureLayout::DepthStencilAttachment,
            Self::DepthStencilReadOnly => TextureLayout::DepthStencilReadOnly,
            Self::ShaderRead => TextureLayout::ShaderReadOnly,
            Self::StorageReadWrite => TextureLayout::General,
            Self::TransferRead => TextureLayout::TransferSrc,
            Self::TransferWrite => TextureLayout::TransferDst,
        }
    }

    /// Check if this access mode is a write operation.
    pub fn is_write(self) -> bool {
        matches!(
            self,
            Self::RenderTargetWrite
                | Self::DepthStencilWrite
                | Self::StorageReadWrite
                | Self::TransferWrite
        )
    }

    /// Check if this access mode is a read operation.
    pub fn is_read(self) -> bool {
        matches!(
            self,
            Self::DepthStencilReadOnly
                | Self::ShaderRead
                | Self::StorageReadWrite
                | Self::TransferRead
        )
    }
}

/// How a buffer is used within a pass.
///
/// Each access mode determines the required memory barriers for proper
/// synchronization between passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferAccessMode {
    /// Read as vertex buffer data.
    VertexBuffer,
    /// Read as index buffer data.
    IndexBuffer,
    /// Read as uniform buffer (constant data).
    UniformRead,
    /// Read as storage buffer.
    StorageRead,
    /// Write as storage buffer.
    StorageWrite,
    /// Read and write as storage buffer.
    StorageReadWrite,
    /// Read as indirect draw arguments.
    IndirectRead,
    /// Source of a transfer/copy operation.
    TransferRead,
    /// Destination of a transfer/copy operation.
    TransferWrite,
}

impl BufferAccessMode {
    /// Check if this access mode is a write operation.
    pub fn is_write(self) -> bool {
        matches!(
            self,
            Self::StorageWrite | Self::StorageReadWrite | Self::TransferWrite
        )
    }

    /// Check if this access mode is a read operation.
    pub fn is_read(self) -> bool {
        matches!(
            self,
            Self::VertexBuffer
                | Self::IndexBuffer
                | Self::UniformRead
                | Self::StorageRead
                | Self::StorageReadWrite
                | Self::IndirectRead
                | Self::TransferRead
        )
    }

    /// Get the Vulkan access flags for this buffer access mode (as source).
    #[cfg(feature = "vulkan-backend")]
    pub fn src_access_mask(self) -> ash::vk::AccessFlags {
        use ash::vk::AccessFlags;
        match self {
            Self::VertexBuffer => AccessFlags::VERTEX_ATTRIBUTE_READ,
            Self::IndexBuffer => AccessFlags::INDEX_READ,
            Self::UniformRead => AccessFlags::UNIFORM_READ,
            Self::StorageRead => AccessFlags::SHADER_READ,
            Self::StorageWrite => AccessFlags::SHADER_WRITE,
            Self::StorageReadWrite => AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
            Self::IndirectRead => AccessFlags::INDIRECT_COMMAND_READ,
            Self::TransferRead => AccessFlags::TRANSFER_READ,
            Self::TransferWrite => AccessFlags::TRANSFER_WRITE,
        }
    }

    /// Get the Vulkan access flags for this buffer access mode (as destination).
    #[cfg(feature = "vulkan-backend")]
    pub fn dst_access_mask(self) -> ash::vk::AccessFlags {
        use ash::vk::AccessFlags;
        match self {
            Self::VertexBuffer => AccessFlags::VERTEX_ATTRIBUTE_READ,
            Self::IndexBuffer => AccessFlags::INDEX_READ,
            Self::UniformRead => AccessFlags::UNIFORM_READ,
            Self::StorageRead => AccessFlags::SHADER_READ,
            Self::StorageWrite => AccessFlags::SHADER_WRITE,
            Self::StorageReadWrite => AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
            Self::IndirectRead => AccessFlags::INDIRECT_COMMAND_READ,
            Self::TransferRead => AccessFlags::TRANSFER_READ,
            Self::TransferWrite => AccessFlags::TRANSFER_WRITE,
        }
    }

    /// Get the Vulkan pipeline stage for this buffer access mode (as source).
    #[cfg(feature = "vulkan-backend")]
    pub fn src_stage(self) -> ash::vk::PipelineStageFlags {
        use ash::vk::PipelineStageFlags;
        match self {
            Self::VertexBuffer => PipelineStageFlags::VERTEX_INPUT,
            Self::IndexBuffer => PipelineStageFlags::VERTEX_INPUT,
            Self::UniformRead => {
                PipelineStageFlags::VERTEX_SHADER | PipelineStageFlags::FRAGMENT_SHADER
            }
            Self::StorageRead | Self::StorageWrite | Self::StorageReadWrite => {
                PipelineStageFlags::VERTEX_SHADER
                    | PipelineStageFlags::FRAGMENT_SHADER
                    | PipelineStageFlags::COMPUTE_SHADER
            }
            Self::IndirectRead => PipelineStageFlags::DRAW_INDIRECT,
            Self::TransferRead | Self::TransferWrite => PipelineStageFlags::TRANSFER,
        }
    }

    /// Get the Vulkan pipeline stage for this buffer access mode (as destination).
    #[cfg(feature = "vulkan-backend")]
    pub fn dst_stage(self) -> ash::vk::PipelineStageFlags {
        use ash::vk::PipelineStageFlags;
        match self {
            Self::VertexBuffer => PipelineStageFlags::VERTEX_INPUT,
            Self::IndexBuffer => PipelineStageFlags::VERTEX_INPUT,
            Self::UniformRead => {
                PipelineStageFlags::VERTEX_SHADER | PipelineStageFlags::FRAGMENT_SHADER
            }
            Self::StorageRead | Self::StorageWrite | Self::StorageReadWrite => {
                PipelineStageFlags::VERTEX_SHADER
                    | PipelineStageFlags::FRAGMENT_SHADER
                    | PipelineStageFlags::COMPUTE_SHADER
            }
            Self::IndirectRead => PipelineStageFlags::DRAW_INDIRECT,
            Self::TransferRead | Self::TransferWrite => PipelineStageFlags::TRANSFER,
        }
    }
}

/// A texture usage declaration for barrier analysis.
///
/// This describes how a single texture is used within a pass,
/// including the access mode and subresource range.
#[derive(Debug, Clone)]
pub struct TextureUsageDecl {
    /// The texture being used.
    pub texture: Arc<Texture>,
    /// How the texture is accessed.
    pub access: TextureAccessMode,
    /// Starting mip level (default: 0).
    pub mip_level: u32,
    /// Number of mip levels (default: 1).
    pub mip_count: u32,
    /// Starting array layer (default: 0).
    pub array_layer: u32,
    /// Number of array layers (default: 1).
    pub layer_count: u32,
}

impl TextureUsageDecl {
    /// Create a new texture usage declaration with default subresource range.
    pub fn new(texture: Arc<Texture>, access: TextureAccessMode) -> Self {
        Self {
            texture,
            access,
            mip_level: 0,
            mip_count: 1,
            array_layer: 0,
            layer_count: 1,
        }
    }

    /// Set the mip level range.
    pub fn with_mip_levels(mut self, base: u32, count: u32) -> Self {
        self.mip_level = base;
        self.mip_count = count;
        self
    }

    /// Set the array layer range.
    pub fn with_array_layers(mut self, base: u32, count: u32) -> Self {
        self.array_layer = base;
        self.layer_count = count;
        self
    }
}

/// A buffer usage declaration for barrier analysis.
///
/// This describes how a single buffer is used within a pass,
/// including the access mode and byte range.
#[derive(Debug, Clone)]
pub struct BufferUsageDecl {
    /// The buffer being used.
    pub buffer: Arc<Buffer>,
    /// How the buffer is accessed.
    pub access: BufferAccessMode,
    /// Byte offset into the buffer (default: 0).
    pub offset: u64,
    /// Size in bytes (default: entire buffer).
    pub size: u64,
}

impl BufferUsageDecl {
    /// Create a new buffer usage declaration for the entire buffer.
    pub fn new(buffer: Arc<Buffer>, access: BufferAccessMode) -> Self {
        let size = buffer.size();
        Self {
            buffer,
            access,
            offset: 0,
            size,
        }
    }

    /// Create a new buffer usage declaration with a specific range.
    pub fn with_range(
        buffer: Arc<Buffer>,
        access: BufferAccessMode,
        offset: u64,
        size: u64,
    ) -> Self {
        Self {
            buffer,
            access,
            offset,
            size,
        }
    }

    /// Set the byte offset.
    pub fn at_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// Set the size in bytes.
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = size;
        self
    }
}

/// How the swapchain surface is accessed by a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceAccess {
    /// Write only (Clear or DontCare load op).
    Write,
    /// Read existing contents then write (Load op).
    ReadWrite,
}

impl SurfaceAccess {
    /// Check if this access reads the surface.
    pub fn is_read(self) -> bool {
        matches!(self, Self::ReadWrite)
    }

    /// Check if this access writes the surface.
    pub fn is_write(self) -> bool {
        true // Both variants write
    }
}

/// Resource usage declarations for a pass.
///
/// This collects all texture and buffer usages for a pass, enabling the barrier
/// generation system to determine required layout transitions and memory barriers.
#[derive(Debug, Default, Clone)]
pub struct PassResourceUsage {
    /// All texture usages declared for this pass.
    pub texture_usages: Vec<TextureUsageDecl>,
    /// All buffer usages declared for this pass.
    pub buffer_usages: Vec<BufferUsageDecl>,
    /// Surface (swapchain) access mode, if the pass renders to the surface.
    pub surface_access: Option<SurfaceAccess>,
}

impl PassResourceUsage {
    /// Create a new empty resource usage.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a texture usage declaration using builder pattern.
    pub fn with_texture(mut self, texture: Arc<Texture>, access: TextureAccessMode) -> Self {
        self.texture_usages
            .push(TextureUsageDecl::new(texture, access));
        self
    }

    /// Add a texture usage declaration.
    pub fn add_texture(&mut self, texture: Arc<Texture>, access: TextureAccessMode) {
        self.texture_usages
            .push(TextureUsageDecl::new(texture, access));
    }

    /// Add a pre-built texture usage declaration.
    pub fn add_texture_decl(&mut self, decl: TextureUsageDecl) {
        self.texture_usages.push(decl);
    }

    /// Check if there are any texture usages.
    pub fn has_textures(&self) -> bool {
        !self.texture_usages.is_empty()
    }

    /// Get the number of texture usages.
    pub fn texture_count(&self) -> usize {
        self.texture_usages.len()
    }

    // ========================================================================
    // Buffer Usage Methods
    // ========================================================================

    /// Add a buffer usage declaration using builder pattern.
    pub fn with_buffer(mut self, buffer: Arc<Buffer>, access: BufferAccessMode) -> Self {
        self.buffer_usages
            .push(BufferUsageDecl::new(buffer, access));
        self
    }

    /// Add a buffer usage declaration.
    pub fn add_buffer(&mut self, buffer: Arc<Buffer>, access: BufferAccessMode) {
        self.buffer_usages
            .push(BufferUsageDecl::new(buffer, access));
    }

    /// Add a pre-built buffer usage declaration.
    pub fn add_buffer_decl(&mut self, decl: BufferUsageDecl) {
        self.buffer_usages.push(decl);
    }

    /// Check if there are any buffer usages.
    pub fn has_buffers(&self) -> bool {
        !self.buffer_usages.is_empty()
    }

    /// Get the number of buffer usages.
    pub fn buffer_count(&self) -> usize {
        self.buffer_usages.len()
    }

    // ========================================================================
    // Combined Methods
    // ========================================================================

    /// Set the surface access mode.
    pub fn set_surface_access(&mut self, access: SurfaceAccess) {
        self.surface_access = Some(access);
    }

    /// Check if this pass accesses the surface.
    pub fn has_surface_access(&self) -> bool {
        self.surface_access.is_some()
    }

    /// Merge another resource usage into this one.
    pub fn merge(&mut self, other: PassResourceUsage) {
        self.texture_usages.extend(other.texture_usages);
        self.buffer_usages.extend(other.buffer_usages);
        if other.surface_access.is_some() {
            self.surface_access = other.surface_access;
        }
    }

    /// Check if any texture usage is a write operation.
    pub fn has_texture_writes(&self) -> bool {
        self.texture_usages.iter().any(|u| u.access.is_write())
    }

    /// Check if any texture usage is a read operation.
    pub fn has_texture_reads(&self) -> bool {
        self.texture_usages.iter().any(|u| u.access.is_read())
    }

    /// Check if any buffer usage is a write operation.
    pub fn has_buffer_writes(&self) -> bool {
        self.buffer_usages.iter().any(|u| u.access.is_write())
    }

    /// Check if any buffer usage is a read operation.
    pub fn has_buffer_reads(&self) -> bool {
        self.buffer_usages.iter().any(|u| u.access.is_read())
    }

    /// Check if any usage (texture, buffer, or surface) is a write operation.
    pub fn has_writes(&self) -> bool {
        self.has_texture_writes()
            || self.has_buffer_writes()
            || self.surface_access.is_some_and(|a| a.is_write())
    }

    /// Check if any usage (texture, buffer, or surface) is a read operation.
    pub fn has_reads(&self) -> bool {
        self.has_texture_reads()
            || self.has_buffer_reads()
            || self.surface_access.is_some_and(|a| a.is_read())
    }

    /// Check if there are any resource usages.
    pub fn is_empty(&self) -> bool {
        self.texture_usages.is_empty()
            && self.buffer_usages.is_empty()
            && self.surface_access.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_access_mode_is_write() {
        assert!(TextureAccessMode::RenderTargetWrite.is_write());
        assert!(TextureAccessMode::DepthStencilWrite.is_write());
        assert!(TextureAccessMode::StorageReadWrite.is_write());
        assert!(TextureAccessMode::TransferWrite.is_write());

        assert!(!TextureAccessMode::ShaderRead.is_write());
        assert!(!TextureAccessMode::TransferRead.is_write());
    }

    #[test]
    fn test_texture_access_mode_is_read() {
        assert!(TextureAccessMode::ShaderRead.is_read());
        assert!(TextureAccessMode::DepthStencilReadOnly.is_read());
        assert!(TextureAccessMode::StorageReadWrite.is_read());
        assert!(TextureAccessMode::TransferRead.is_read());

        assert!(!TextureAccessMode::RenderTargetWrite.is_read());
        assert!(!TextureAccessMode::TransferWrite.is_read());
    }

    #[test]
    fn test_buffer_access_mode_is_write() {
        assert!(BufferAccessMode::StorageWrite.is_write());
        assert!(BufferAccessMode::StorageReadWrite.is_write());
        assert!(BufferAccessMode::TransferWrite.is_write());

        assert!(!BufferAccessMode::VertexBuffer.is_write());
        assert!(!BufferAccessMode::IndexBuffer.is_write());
        assert!(!BufferAccessMode::UniformRead.is_write());
        assert!(!BufferAccessMode::StorageRead.is_write());
        assert!(!BufferAccessMode::IndirectRead.is_write());
        assert!(!BufferAccessMode::TransferRead.is_write());
    }

    #[test]
    fn test_buffer_access_mode_is_read() {
        assert!(BufferAccessMode::VertexBuffer.is_read());
        assert!(BufferAccessMode::IndexBuffer.is_read());
        assert!(BufferAccessMode::UniformRead.is_read());
        assert!(BufferAccessMode::StorageRead.is_read());
        assert!(BufferAccessMode::StorageReadWrite.is_read());
        assert!(BufferAccessMode::IndirectRead.is_read());
        assert!(BufferAccessMode::TransferRead.is_read());

        assert!(!BufferAccessMode::StorageWrite.is_read());
        assert!(!BufferAccessMode::TransferWrite.is_read());
    }

    #[test]
    fn test_pass_resource_usage_empty() {
        let usage = PassResourceUsage::new();
        assert!(!usage.has_textures());
        assert!(!usage.has_buffers());
        assert!(usage.is_empty());
        assert_eq!(usage.texture_count(), 0);
        assert_eq!(usage.buffer_count(), 0);
    }

    #[test]
    fn test_pass_resource_usage_merge() {
        let mut usage1 = PassResourceUsage::new();
        let usage2 = PassResourceUsage::new();

        usage1.merge(usage2);
        assert!(!usage1.has_textures());
        assert!(!usage1.has_buffers());
        assert!(usage1.is_empty());
    }
}
