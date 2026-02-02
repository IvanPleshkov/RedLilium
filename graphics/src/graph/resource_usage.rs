//! Resource usage declarations for automatic barrier generation.
//!
//! This module defines how textures are used within passes, enabling
//! automatic layout tracking and barrier placement by the Vulkan backend.

use std::sync::Arc;

use crate::resources::Texture;

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

/// Resource usage declarations for a pass.
///
/// This collects all texture usages for a pass, enabling the barrier
/// generation system to determine required layout transitions.
#[derive(Debug, Default, Clone)]
pub struct PassResourceUsage {
    /// All texture usages declared for this pass.
    pub texture_usages: Vec<TextureUsageDecl>,
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

    /// Merge another resource usage into this one.
    pub fn merge(&mut self, other: PassResourceUsage) {
        self.texture_usages.extend(other.texture_usages);
    }

    /// Check if any usage is a write operation.
    pub fn has_writes(&self) -> bool {
        self.texture_usages.iter().any(|u| u.access.is_write())
    }

    /// Check if any usage is a read operation.
    pub fn has_reads(&self) -> bool {
        self.texture_usages.iter().any(|u| u.access.is_read())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_mode_is_write() {
        assert!(TextureAccessMode::RenderTargetWrite.is_write());
        assert!(TextureAccessMode::DepthStencilWrite.is_write());
        assert!(TextureAccessMode::StorageReadWrite.is_write());
        assert!(TextureAccessMode::TransferWrite.is_write());

        assert!(!TextureAccessMode::ShaderRead.is_write());
        assert!(!TextureAccessMode::TransferRead.is_write());
    }

    #[test]
    fn test_access_mode_is_read() {
        assert!(TextureAccessMode::ShaderRead.is_read());
        assert!(TextureAccessMode::DepthStencilReadOnly.is_read());
        assert!(TextureAccessMode::StorageReadWrite.is_read());
        assert!(TextureAccessMode::TransferRead.is_read());

        assert!(!TextureAccessMode::RenderTargetWrite.is_read());
        assert!(!TextureAccessMode::TransferWrite.is_read());
    }

    #[test]
    fn test_pass_resource_usage() {
        let usage = PassResourceUsage::new();
        assert!(!usage.has_textures());
        assert_eq!(usage.texture_count(), 0);
    }

    #[test]
    fn test_pass_resource_usage_merge() {
        let mut usage1 = PassResourceUsage::new();
        let usage2 = PassResourceUsage::new();

        usage1.merge(usage2);
        assert!(!usage1.has_textures());
    }
}
