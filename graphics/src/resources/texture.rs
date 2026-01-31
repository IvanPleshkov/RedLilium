//! GPU texture resource.

use std::sync::Arc;

use crate::device::GraphicsDevice;
use crate::types::{Extent3d, TextureDescriptor, TextureFormat};

/// A GPU texture resource.
///
/// Textures are created by [`GraphicsDevice::create_texture`] and are reference-counted.
/// They hold a strong reference to their parent device, keeping it alive.
///
/// # Example
///
/// ```ignore
/// let texture = device.create_texture(&TextureDescriptor::new_2d(
///     1920, 1080,
///     TextureFormat::Rgba8Unorm,
///     TextureUsage::RENDER_ATTACHMENT,
/// ))?;
/// println!("Texture size: {}x{}", texture.width(), texture.height());
/// ```
pub struct Texture {
    device: Arc<GraphicsDevice>,
    descriptor: TextureDescriptor,
}

impl Texture {
    /// Create a new texture (called by GraphicsDevice).
    pub(crate) fn new(device: Arc<GraphicsDevice>, descriptor: TextureDescriptor) -> Self {
        Self { device, descriptor }
    }

    /// Get the parent device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Get the texture descriptor.
    pub fn descriptor(&self) -> &TextureDescriptor {
        &self.descriptor
    }

    /// Get the texture size.
    pub fn size(&self) -> Extent3d {
        self.descriptor.size
    }

    /// Get the texture width.
    pub fn width(&self) -> u32 {
        self.descriptor.size.width
    }

    /// Get the texture height.
    pub fn height(&self) -> u32 {
        self.descriptor.size.height
    }

    /// Get the texture depth.
    pub fn depth(&self) -> u32 {
        self.descriptor.size.depth
    }

    /// Get the texture format.
    pub fn format(&self) -> TextureFormat {
        self.descriptor.format
    }

    /// Get the mip level count.
    pub fn mip_level_count(&self) -> u32 {
        self.descriptor.mip_level_count
    }

    /// Get the sample count.
    pub fn sample_count(&self) -> u32 {
        self.descriptor.sample_count
    }

    /// Get the texture label, if set.
    pub fn label(&self) -> Option<&str> {
        self.descriptor.label.as_deref()
    }
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Texture")
            .field("size", &self.descriptor.size)
            .field("format", &self.descriptor.format)
            .field("usage", &self.descriptor.usage)
            .field("label", &self.descriptor.label)
            .finish()
    }
}

// Ensure Texture is Send + Sync
static_assertions::assert_impl_all!(Texture: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;
    use crate::types::TextureUsage;

    fn create_test_device() -> Arc<GraphicsDevice> {
        let instance = GraphicsInstance::new().unwrap();
        instance.create_device().unwrap()
    }

    #[test]
    fn test_texture_debug() {
        let desc = TextureDescriptor::new_2d(
            1920,
            1080,
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT,
        );
        let texture = Texture::new(create_test_device(), desc);
        let debug = format!("{:?}", texture);
        assert!(debug.contains("Texture"));
        assert!(debug.contains("1920"));
    }

    #[test]
    fn test_texture_dimensions() {
        let desc = TextureDescriptor::new_2d(
            800,
            600,
            TextureFormat::Rgba8Unorm,
            TextureUsage::TEXTURE_BINDING,
        );
        let texture = Texture::new(create_test_device(), desc);
        assert_eq!(texture.width(), 800);
        assert_eq!(texture.height(), 600);
        assert_eq!(texture.depth(), 1);
    }
}
