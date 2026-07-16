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
    /// Read as acceleration-structure build **input** (#110): vertex/index
    /// buffers consumed by a BLAS build, or the instance buffer consumed by a
    /// TLAS build. `SHADER_READ` at the `ACCELERATION_STRUCTURE_BUILD` stage.
    AccelerationStructureBuildInput,
    /// A BLAS backing buffer read **by a TLAS build** referencing it (#110).
    /// `ACCELERATION_STRUCTURE_READ` at the build stage — distinct from
    /// [`Self::AccelerationStructureBuildInput`] (plain shader-read access)
    /// and from [`Self::AccelerationStructureShaderRead`] (ray-query stages).
    AccelerationStructureBuildRead,
    /// Written by an acceleration-structure build (#110): the destination AS
    /// backing buffer and the build scratch buffer. Scratch is read *and*
    /// written during a build, so the access mask carries both.
    AccelerationStructureWrite,
    /// Acceleration-structure memory (TLAS + the BLASes it references)
    /// traversed by ray queries in shaders (#110).
    /// `ACCELERATION_STRUCTURE_READ` at the shader stages that may query.
    AccelerationStructureShaderRead,
}

impl BufferAccessMode {
    /// Check if this access mode is a write operation.
    pub fn is_write(self) -> bool {
        matches!(
            self,
            Self::StorageWrite
                | Self::StorageReadWrite
                | Self::TransferWrite
                | Self::AccelerationStructureWrite
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
                | Self::AccelerationStructureBuildInput
                | Self::AccelerationStructureBuildRead
                | Self::AccelerationStructureWrite
                | Self::AccelerationStructureShaderRead
        )
    }

    /// Get the Vulkan access flags for this buffer access mode (as source).
    #[cfg(feature = "vulkan-backend")]
    pub fn src_access_mask(self) -> ash::vk::AccessFlags2 {
        use ash::vk::AccessFlags2;
        match self {
            Self::VertexBuffer => AccessFlags2::VERTEX_ATTRIBUTE_READ,
            Self::IndexBuffer => AccessFlags2::INDEX_READ,
            Self::UniformRead => AccessFlags2::UNIFORM_READ,
            Self::StorageRead => AccessFlags2::SHADER_READ,
            Self::StorageWrite => AccessFlags2::SHADER_WRITE,
            Self::StorageReadWrite => AccessFlags2::SHADER_READ | AccessFlags2::SHADER_WRITE,
            Self::IndirectRead => AccessFlags2::INDIRECT_COMMAND_READ,
            Self::TransferRead => AccessFlags2::TRANSFER_READ,
            Self::TransferWrite => AccessFlags2::TRANSFER_WRITE,
            // Build inputs are read with plain SHADER_READ at the build stage
            // (Vulkan sync chapter: input buffers of
            // vkCmdBuildAccelerationStructuresKHR use SHADER_READ).
            Self::AccelerationStructureBuildInput => AccessFlags2::SHADER_READ,
            Self::AccelerationStructureBuildRead => AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            // Destination AS is written; scratch is read AND written within
            // the build, so the write mode carries both masks.
            Self::AccelerationStructureWrite => {
                AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
                    | AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
            }
            Self::AccelerationStructureShaderRead => AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        }
    }

    /// Get the Vulkan access flags for this buffer access mode (as destination).
    #[cfg(feature = "vulkan-backend")]
    pub fn dst_access_mask(self) -> ash::vk::AccessFlags2 {
        // Source and destination masks coincide for every mode (the
        // distinction exists for future asymmetric modes).
        self.src_access_mask()
    }

    /// Get the Vulkan pipeline stage for this buffer access mode (as source).
    ///
    /// `BufferAccessMode` does not record which shader stage performs the
    /// access, so shader modes return the union of all stages that could —
    /// including `COMPUTE_SHADER` for uniform reads (a UBO can feed a
    /// dispatch just as well as a draw).
    #[cfg(feature = "vulkan-backend")]
    pub fn src_stage(self) -> ash::vk::PipelineStageFlags2 {
        use ash::vk::PipelineStageFlags2;
        match self {
            Self::VertexBuffer => PipelineStageFlags2::VERTEX_INPUT,
            Self::IndexBuffer => PipelineStageFlags2::VERTEX_INPUT,
            Self::UniformRead | Self::StorageRead | Self::StorageWrite | Self::StorageReadWrite => {
                PipelineStageFlags2::VERTEX_SHADER
                    | PipelineStageFlags2::FRAGMENT_SHADER
                    | PipelineStageFlags2::COMPUTE_SHADER
            }
            Self::IndirectRead => PipelineStageFlags2::DRAW_INDIRECT,
            Self::TransferRead | Self::TransferWrite => PipelineStageFlags2::ALL_TRANSFER,
            Self::AccelerationStructureBuildInput
            | Self::AccelerationStructureBuildRead
            | Self::AccelerationStructureWrite => {
                PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
            }
            // Ray queries are legal in any shader stage the engine exposes;
            // like the storage modes, the union covers all of them.
            Self::AccelerationStructureShaderRead => {
                PipelineStageFlags2::VERTEX_SHADER
                    | PipelineStageFlags2::FRAGMENT_SHADER
                    | PipelineStageFlags2::COMPUTE_SHADER
            }
        }
    }

    /// Get the Vulkan pipeline stage for this buffer access mode (as destination).
    ///
    /// See [`src_stage`](Self::src_stage) for why shader modes return the
    /// union of vertex/fragment/compute stages.
    #[cfg(feature = "vulkan-backend")]
    pub fn dst_stage(self) -> ash::vk::PipelineStageFlags2 {
        // Source and destination stages coincide for every mode.
        self.src_stage()
    }
}

/// A texture usage declaration for barrier analysis.
///
/// This describes how a single texture is used within a pass,
/// including the access mode and subresource range.
///
/// # Subresource granularity
///
/// The mip/layer range fields describe intent, but the Vulkan barrier system
/// currently tracks one layout per image and emits **whole-image**
/// transitions — a pass cannot yet hold different mips/layers of one image
/// in different layouts (e.g. write mip N while sampling mip N−1 during
/// mipmap generation). Nothing in the engine declares partial ranges today;
/// per-subresource tracking is the prerequisite for such workflows.
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
///
/// # `Load` scope
///
/// `LoadOp::Load` on the surface preserves contents **within a frame only**
/// (e.g. a UI overlay pass loading the scene pass's output). The first
/// surface-writing pass of each frame starts from discarded contents — the
/// previously *presented* image is stale (swapchain images rotate) and the
/// Vulkan backend transitions it from `UNDEFINED`. For cross-frame
/// accumulation, render to an offscreen texture.
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

    /// #110: AS access modes classify correctly — builds write, everything
    /// else reads (scratch is modeled as part of the write mode).
    #[test]
    fn acceleration_structure_modes_read_write() {
        assert!(BufferAccessMode::AccelerationStructureWrite.is_write());
        assert!(!BufferAccessMode::AccelerationStructureBuildInput.is_write());
        assert!(!BufferAccessMode::AccelerationStructureBuildRead.is_write());
        assert!(!BufferAccessMode::AccelerationStructureShaderRead.is_write());
        assert!(BufferAccessMode::AccelerationStructureBuildInput.is_read());
        assert!(BufferAccessMode::AccelerationStructureBuildRead.is_read());
        assert!(BufferAccessMode::AccelerationStructureShaderRead.is_read());
    }

    /// #110: AS modes lower to the acceleration-structure build stage /
    /// access masks, and traversal reads land on the shader stages.
    #[cfg(feature = "vulkan-backend")]
    #[test]
    fn acceleration_structure_modes_vulkan_scopes() {
        use ash::vk::{AccessFlags2, PipelineStageFlags2};

        for mode in [
            BufferAccessMode::AccelerationStructureBuildInput,
            BufferAccessMode::AccelerationStructureBuildRead,
            BufferAccessMode::AccelerationStructureWrite,
        ] {
            assert_eq!(
                mode.dst_stage(),
                PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
            );
        }
        assert_eq!(
            BufferAccessMode::AccelerationStructureBuildInput.dst_access_mask(),
            AccessFlags2::SHADER_READ
        );
        assert_eq!(
            BufferAccessMode::AccelerationStructureBuildRead.dst_access_mask(),
            AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
        );
        assert!(
            BufferAccessMode::AccelerationStructureWrite
                .dst_access_mask()
                .contains(AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        );
        assert!(
            BufferAccessMode::AccelerationStructureShaderRead
                .dst_stage()
                .contains(
                    PipelineStageFlags2::FRAGMENT_SHADER | PipelineStageFlags2::COMPUTE_SHADER
                )
        );
        assert_eq!(
            BufferAccessMode::AccelerationStructureShaderRead.dst_access_mask(),
            AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
        );
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
