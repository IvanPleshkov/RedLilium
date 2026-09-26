//! GPU texture resource.

use std::sync::Arc;

use crate::backend::GpuTexture;
use crate::device::GraphicsDevice;
use crate::types::{Extent3d, TextureDescriptor, TextureDimension, TextureFormat};

/// Attachment views select exactly one mip and array layer. Cached by the
/// owning texture, so their lifetime covers every submitted graph using it.
pub(crate) enum AttachmentView {
    Dummy,
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu::TextureView),
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        view: ash::vk::ImageView,
    },
}

#[cfg(feature = "vulkan-backend")]
impl Drop for AttachmentView {
    fn drop(&mut self) {
        if let Self::Vulkan { device, view } = self {
            unsafe { device.destroy_image_view(*view, None) };
        }
    }
}

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
    descriptor: TextureDescriptor,
    attachment_views:
        parking_lot::Mutex<std::collections::HashMap<(u32, u32), Arc<AttachmentView>>>,
    gpu_handle: GpuTexture,
    /// Declared after `gpu_handle` deliberately: fields drop in declaration
    /// order, and this keep-alive must outlive the handle's `Drop`, which
    /// needs the backend (owned by the device→instance chain) alive. See #50.
    device: Arc<GraphicsDevice>,
}

impl Texture {
    /// Create a new texture (called by GraphicsDevice).
    pub(crate) fn new(
        device: Arc<GraphicsDevice>,
        descriptor: TextureDescriptor,
        gpu_handle: GpuTexture,
    ) -> Self {
        Self {
            device,
            descriptor,
            attachment_views: Default::default(),
            gpu_handle,
        }
    }

    pub(crate) fn validate_attachment(
        &self,
        mip: u32,
        layer: u32,
    ) -> Result<(), crate::GraphicsError> {
        use crate::{GraphicsError, TextureUsage};
        let layers = match self.dimension() {
            TextureDimension::D2 => 1,
            TextureDimension::D2Array => self.depth(),
            TextureDimension::Cube => 6,
            TextureDimension::CubeArray => self.depth() * 6,
            _ => {
                return Err(GraphicsError::FeatureNotSupported(
                    "render attachments require a 2D texture or array/cube face".into(),
                ));
            }
        };
        if mip >= self.mip_level_count() || layer >= layers {
            return Err(GraphicsError::InvalidParameter(
                "attachment mip/layer is out of bounds".into(),
            ));
        }
        if !self.usage().contains(TextureUsage::RENDER_ATTACHMENT) {
            return Err(GraphicsError::InvalidParameter(
                "texture lacks RENDER_ATTACHMENT usage".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn attachment_view(
        &self,
        mip: u32,
        layer: u32,
    ) -> Result<Arc<AttachmentView>, crate::GraphicsError> {
        self.validate_attachment(mip, layer)?;
        let mut views = self.attachment_views.lock();
        if let Some(view) = views.get(&(mip, layer)) {
            return Ok(Arc::clone(view));
        }
        let view = match &self.gpu_handle {
            GpuTexture::Dummy => AttachmentView::Dummy,
            #[cfg(feature = "wgpu-backend")]
            GpuTexture::Wgpu { texture, .. } => {
                AttachmentView::Wgpu(texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                }))
            }
            #[cfg(feature = "vulkan-backend")]
            GpuTexture::Vulkan {
                device,
                image,
                format,
                ..
            } => {
                use ash::vk;
                let aspect = if self.format().is_depth_stencil() {
                    if self.format().has_stencil() {
                        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                    } else {
                        vk::ImageAspectFlags::DEPTH
                    }
                } else {
                    vk::ImageAspectFlags::COLOR
                };
                let info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(*format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: mip,
                        level_count: 1,
                        base_array_layer: layer,
                        layer_count: 1,
                    });
                let view = unsafe { device.create_image_view(&info, None) }.map_err(|e| {
                    crate::GraphicsError::ResourceCreationFailed(format!("attachment view: {e:?}"))
                })?;
                AttachmentView::Vulkan {
                    device: device.clone(),
                    view,
                }
            }
        };
        let view = Arc::new(view);
        views.insert((mip, layer), Arc::clone(&view));
        Ok(view)
    }

    /// Get the GPU handle for this texture.
    pub fn gpu_handle(&self) -> &GpuTexture {
        &self.gpu_handle
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

    /// Get the texture usage flags.
    pub fn usage(&self) -> crate::types::TextureUsage {
        self.descriptor.usage
    }

    /// Get the mip level count.
    pub fn mip_level_count(&self) -> u32 {
        self.descriptor.mip_level_count
    }

    /// Get the sample count.
    pub fn sample_count(&self) -> u32 {
        self.descriptor.sample_count
    }

    /// Get the texture dimension.
    pub fn dimension(&self) -> TextureDimension {
        self.descriptor.dimension
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
        let device = create_test_device();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                1920,
                1080,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap();
        let debug = format!("{:?}", texture);
        assert!(debug.contains("Texture"));
        assert!(debug.contains("1920"));
    }

    #[test]
    fn test_texture_dimensions() {
        let device = create_test_device();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                800,
                600,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();
        assert_eq!(texture.width(), 800);
        assert_eq!(texture.height(), 600);
        assert_eq!(texture.depth(), 1);
    }
}
