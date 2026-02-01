//! Render target types for render passes.
//!
//! This module defines the types used to configure render targets (color and depth/stencil
//! attachments) for graphics passes.

use std::sync::Arc;

use crate::resources::Texture;
use crate::swapchain::SurfaceTexture;
use crate::types::{ClearValue, TextureFormat};

#[cfg(feature = "wgpu-backend")]
use crate::backend::wgpu_impl::SurfaceTextureView;

/// Operation to perform when loading an attachment at the start of a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LoadOp {
    /// Clear the attachment with a specified value.
    Clear(ClearValue),
    /// Load the existing contents of the attachment.
    #[default]
    Load,
    /// Don't care about the existing contents (may be undefined).
    DontCare,
}

impl LoadOp {
    /// Create a clear operation with a color value.
    pub fn clear_color(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Clear(ClearValue::color(r, g, b, a))
    }

    /// Create a clear operation with a depth value.
    pub fn clear_depth(depth: f32) -> Self {
        Self::Clear(ClearValue::depth(depth))
    }
}

/// Operation to perform when storing an attachment at the end of a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StoreOp {
    /// Store the attachment contents for later use.
    #[default]
    Store,
    /// Don't care about the contents after the pass (may be discarded).
    DontCare,
}

/// A render target that can be rendered to.
///
/// This can be either a texture or a surface texture from the swapchain.
#[derive(Debug, Clone)]
pub enum RenderTarget {
    /// Render to a texture.
    Texture {
        /// The texture to render to.
        texture: Arc<Texture>,
        /// Mip level to render to (default: 0).
        mip_level: u32,
        /// Array layer to render to (default: 0).
        array_layer: u32,
    },
    /// Render to a surface texture (swapchain).
    Surface {
        /// The format of the surface texture.
        format: TextureFormat,
        /// Width of the surface.
        width: u32,
        /// Height of the surface.
        height: u32,
        /// The actual wgpu texture view for rendering (only set when wgpu backend is used).
        #[cfg(feature = "wgpu-backend")]
        view: Option<SurfaceTextureView>,
    },
}

impl RenderTarget {
    /// Create a render target from a texture.
    pub fn from_texture(texture: Arc<Texture>) -> Self {
        Self::Texture {
            texture,
            mip_level: 0,
            array_layer: 0,
        }
    }

    /// Create a render target from a texture with specific mip level.
    pub fn from_texture_mip(texture: Arc<Texture>, mip_level: u32) -> Self {
        Self::Texture {
            texture,
            mip_level,
            array_layer: 0,
        }
    }

    /// Create a render target from a surface texture.
    pub fn from_surface(surface_texture: &SurfaceTexture) -> Self {
        Self::Surface {
            format: surface_texture.format(),
            width: surface_texture.width(),
            height: surface_texture.height(),
            #[cfg(feature = "wgpu-backend")]
            view: surface_texture.wgpu_view(),
        }
    }

    /// Get the format of the render target.
    pub fn format(&self) -> TextureFormat {
        match self {
            Self::Texture { texture, .. } => texture.format(),
            Self::Surface { format, .. } => *format,
        }
    }

    /// Get the width of the render target.
    pub fn width(&self) -> u32 {
        match self {
            Self::Texture { texture, .. } => texture.width(),
            Self::Surface { width, .. } => *width,
        }
    }

    /// Get the height of the render target.
    pub fn height(&self) -> u32 {
        match self {
            Self::Texture { texture, .. } => texture.height(),
            Self::Surface { height, .. } => *height,
        }
    }
}

/// A color attachment for a render pass.
#[derive(Debug, Clone)]
pub struct ColorAttachment {
    /// The render target.
    pub target: RenderTarget,
    /// Optional resolve target for MSAA.
    pub resolve_target: Option<RenderTarget>,
    /// Operation when loading the attachment.
    pub load_op: LoadOp,
    /// Operation when storing the attachment.
    pub store_op: StoreOp,
}

impl ColorAttachment {
    /// Create a new color attachment.
    pub fn new(target: RenderTarget) -> Self {
        Self {
            target,
            resolve_target: None,
            load_op: LoadOp::default(),
            store_op: StoreOp::default(),
        }
    }

    /// Create a color attachment from a texture.
    pub fn from_texture(texture: Arc<Texture>) -> Self {
        Self::new(RenderTarget::from_texture(texture))
    }

    /// Create a color attachment from a surface texture.
    pub fn from_surface(surface_texture: &SurfaceTexture) -> Self {
        Self::new(RenderTarget::from_surface(surface_texture))
    }

    /// Set the load operation.
    pub fn with_load_op(mut self, load_op: LoadOp) -> Self {
        self.load_op = load_op;
        self
    }

    /// Set the store operation.
    pub fn with_store_op(mut self, store_op: StoreOp) -> Self {
        self.store_op = store_op;
        self
    }

    /// Set a clear color.
    pub fn with_clear_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.load_op = LoadOp::clear_color(r, g, b, a);
        self
    }

    /// Set the resolve target for MSAA.
    pub fn with_resolve_target(mut self, target: RenderTarget) -> Self {
        self.resolve_target = Some(target);
        self
    }

    /// Get the texture for this attachment.
    ///
    /// Panics if this is a surface attachment (not a texture).
    pub fn texture(&self) -> &Arc<Texture> {
        match &self.target {
            RenderTarget::Texture { texture, .. } => texture,
            RenderTarget::Surface { .. } => panic!("Cannot get texture from surface attachment"),
        }
    }

    /// Get the load operation.
    pub fn load_op(&self) -> LoadOp {
        self.load_op
    }

    /// Get the store operation.
    pub fn store_op(&self) -> StoreOp {
        self.store_op
    }
}

/// A depth/stencil attachment for a render pass.
#[derive(Debug, Clone)]
pub struct DepthStencilAttachment {
    /// The render target (must be a depth/stencil format).
    pub target: RenderTarget,
    /// Operation when loading the depth component.
    pub depth_load_op: LoadOp,
    /// Operation when storing the depth component.
    pub depth_store_op: StoreOp,
    /// Whether depth is read-only.
    pub depth_read_only: bool,
    /// Operation when loading the stencil component.
    pub stencil_load_op: LoadOp,
    /// Operation when storing the stencil component.
    pub stencil_store_op: StoreOp,
    /// Whether stencil is read-only.
    pub stencil_read_only: bool,
}

impl DepthStencilAttachment {
    /// Create a new depth/stencil attachment.
    pub fn new(target: RenderTarget) -> Self {
        Self {
            target,
            depth_load_op: LoadOp::default(),
            depth_store_op: StoreOp::default(),
            depth_read_only: false,
            stencil_load_op: LoadOp::default(),
            stencil_store_op: StoreOp::default(),
            stencil_read_only: false,
        }
    }

    /// Create a depth/stencil attachment from a texture.
    pub fn from_texture(texture: Arc<Texture>) -> Self {
        Self::new(RenderTarget::from_texture(texture))
    }

    /// Set the depth load operation.
    pub fn with_depth_load_op(mut self, load_op: LoadOp) -> Self {
        self.depth_load_op = load_op;
        self
    }

    /// Set the depth store operation.
    pub fn with_depth_store_op(mut self, store_op: StoreOp) -> Self {
        self.depth_store_op = store_op;
        self
    }

    /// Set depth to read-only mode.
    pub fn with_depth_read_only(mut self, read_only: bool) -> Self {
        self.depth_read_only = read_only;
        self
    }

    /// Clear depth to a specific value.
    pub fn with_clear_depth(mut self, depth: f32) -> Self {
        self.depth_load_op = LoadOp::clear_depth(depth);
        self
    }

    /// Set the stencil load operation.
    pub fn with_stencil_load_op(mut self, load_op: LoadOp) -> Self {
        self.stencil_load_op = load_op;
        self
    }

    /// Set the stencil store operation.
    pub fn with_stencil_store_op(mut self, store_op: StoreOp) -> Self {
        self.stencil_store_op = store_op;
        self
    }

    /// Set stencil to read-only mode.
    pub fn with_stencil_read_only(mut self, read_only: bool) -> Self {
        self.stencil_read_only = read_only;
        self
    }

    /// Get the texture for this attachment.
    ///
    /// Panics if this is a surface attachment (not a texture).
    pub fn texture(&self) -> &Arc<Texture> {
        match &self.target {
            RenderTarget::Texture { texture, .. } => texture,
            RenderTarget::Surface { .. } => panic!("Cannot get texture from surface attachment"),
        }
    }

    /// Get the depth load operation.
    pub fn depth_load_op(&self) -> LoadOp {
        self.depth_load_op
    }

    /// Get the depth store operation.
    pub fn depth_store_op(&self) -> StoreOp {
        self.depth_store_op
    }

    /// Get the stencil load operation.
    pub fn stencil_load_op(&self) -> LoadOp {
        self.stencil_load_op
    }

    /// Get the stencil store operation.
    pub fn stencil_store_op(&self) -> StoreOp {
        self.stencil_store_op
    }
}

/// Configuration for render pass targets.
///
/// This describes what the render pass will render to.
#[derive(Debug, Clone, Default)]
pub struct RenderTargetConfig {
    /// Color attachments for the render pass.
    pub color_attachments: Vec<ColorAttachment>,
    /// Optional depth/stencil attachment.
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
}

impl RenderTargetConfig {
    /// Create a new empty render target configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a color attachment.
    pub fn with_color(mut self, attachment: ColorAttachment) -> Self {
        self.color_attachments.push(attachment);
        self
    }

    /// Set the depth/stencil attachment.
    pub fn with_depth_stencil(mut self, attachment: DepthStencilAttachment) -> Self {
        self.depth_stencil_attachment = Some(attachment);
        self
    }

    /// Get the render area dimensions.
    ///
    /// Returns the dimensions of the first color attachment, or the depth attachment
    /// if no color attachments are present.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        if let Some(color) = self.color_attachments.first() {
            return Some((color.target.width(), color.target.height()));
        }
        if let Some(depth) = &self.depth_stencil_attachment {
            return Some((depth.target.width(), depth.target.height()));
        }
        None
    }

    /// Check if this config has any attachments.
    pub fn has_attachments(&self) -> bool {
        !self.color_attachments.is_empty() || self.depth_stencil_attachment.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;
    use crate::types::{TextureDescriptor, TextureUsage};

    fn create_test_texture() -> Arc<Texture> {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        device
            .create_texture(&TextureDescriptor::new_2d(
                1920,
                1080,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap()
    }

    fn create_depth_texture() -> Arc<Texture> {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        device
            .create_texture(&TextureDescriptor::new_2d(
                1920,
                1080,
                TextureFormat::Depth32Float,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap()
    }

    #[test]
    fn test_load_op_default() {
        assert_eq!(LoadOp::default(), LoadOp::Load);
    }

    #[test]
    fn test_store_op_default() {
        assert_eq!(StoreOp::default(), StoreOp::Store);
    }

    #[test]
    fn test_color_attachment_from_texture() {
        let texture = create_test_texture();
        let attachment =
            ColorAttachment::from_texture(texture).with_clear_color(0.0, 0.0, 0.0, 1.0);

        assert!(matches!(attachment.load_op, LoadOp::Clear(_)));
        assert_eq!(attachment.store_op, StoreOp::Store);
    }

    #[test]
    fn test_depth_stencil_attachment() {
        let texture = create_depth_texture();
        let attachment = DepthStencilAttachment::from_texture(texture)
            .with_clear_depth(1.0)
            .with_depth_store_op(StoreOp::DontCare);

        assert!(matches!(attachment.depth_load_op, LoadOp::Clear(_)));
        assert_eq!(attachment.depth_store_op, StoreOp::DontCare);
    }

    #[test]
    fn test_render_target_config() {
        let color = create_test_texture();
        let depth = create_depth_texture();

        let config = RenderTargetConfig::new()
            .with_color(ColorAttachment::from_texture(color).with_clear_color(0.1, 0.2, 0.3, 1.0))
            .with_depth_stencil(DepthStencilAttachment::from_texture(depth).with_clear_depth(1.0));

        assert_eq!(config.color_attachments.len(), 1);
        assert!(config.depth_stencil_attachment.is_some());
        assert_eq!(config.dimensions(), Some((1920, 1080)));
        assert!(config.has_attachments());
    }

    #[test]
    fn test_render_target_dimensions() {
        let texture = create_test_texture();
        let target = RenderTarget::from_texture(texture);

        assert_eq!(target.width(), 1920);
        assert_eq!(target.height(), 1080);
        assert_eq!(target.format(), TextureFormat::Rgba8Unorm);
    }
}
