//! Private native lowering for validated texture view descriptors.
/// Backend-owned view handle. The parent texture keeps cached handles alive.
pub(crate) enum GpuTextureView {
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
impl Drop for GpuTextureView {
    fn drop(&mut self) {
        if let Self::Vulkan { device, view } = self {
            unsafe { device.destroy_image_view(*view, None) };
        }
    }
}

use super::{GpuBackend, GpuTexture};
use crate::{GraphicsError, TextureFormat, TextureViewDescriptor};
pub(crate) fn create_view(
    backend: &GpuBackend,
    texture: &GpuTexture,
    texture_format: TextureFormat,
    desc: &TextureViewDescriptor,
) -> Result<GpuTextureView, GraphicsError> {
    let _ = (backend, texture_format, desc); // Some arguments are backend-feature-specific.
    Ok(match texture {
        GpuTexture::Dummy => GpuTextureView::Dummy,
        #[cfg(feature = "wgpu-backend")]
        GpuTexture::Wgpu { texture, .. } => {
            use crate::{TextureAspect as A, TextureViewDimension as D};
            let dimension = match desc.dimension {
                D::D1 => wgpu::TextureViewDimension::D1,
                D::D2 => wgpu::TextureViewDimension::D2,
                D::D2Array => wgpu::TextureViewDimension::D2Array,
                D::Cube => wgpu::TextureViewDimension::Cube,
                D::CubeArray => wgpu::TextureViewDimension::CubeArray,
                D::D3 => wgpu::TextureViewDimension::D3,
                D::D1Array => {
                    return Err(crate::GraphicsError::FeatureNotSupported(
                        "wgpu has no 1D array views".into(),
                    ));
                }
            };
            let native_desc = wgpu::TextureViewDescriptor {
                dimension: Some(dimension),
                aspect: match desc.aspect {
                    A::All => wgpu::TextureAspect::All,
                    A::DepthOnly => wgpu::TextureAspect::DepthOnly,
                    A::StencilOnly => wgpu::TextureAspect::StencilOnly,
                },
                base_mip_level: desc.base_mip_level,
                mip_level_count: desc.mip_level_count,
                base_array_layer: desc.base_array_layer,
                array_layer_count: if desc.dimension == D::D3 {
                    None
                } else {
                    desc.array_layer_count
                },
                ..Default::default()
            };
            let crate::backend::GpuBackend::Wgpu(backend) = backend else {
                unreachable!()
            };
            GpuTextureView::Wgpu(backend.create_texture_view(texture, &native_desc)?)
        }
        #[cfg(feature = "vulkan-backend")]
        GpuTexture::Vulkan {
            device,
            image,
            format,
            ..
        } => {
            use crate::{TextureAspect as A, TextureViewDimension as D};
            use ash::vk;
            let view_type = match desc.dimension {
                D::D1 => vk::ImageViewType::TYPE_1D,
                D::D1Array => vk::ImageViewType::TYPE_1D_ARRAY,
                D::D2 => vk::ImageViewType::TYPE_2D,
                D::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
                D::D3 => vk::ImageViewType::TYPE_3D,
                D::Cube => vk::ImageViewType::CUBE,
                D::CubeArray => vk::ImageViewType::CUBE_ARRAY,
            };
            let aspect = match desc.aspect {
                A::DepthOnly => vk::ImageAspectFlags::DEPTH,
                A::StencilOnly => vk::ImageAspectFlags::STENCIL,
                A::All => {
                    if texture_format.is_depth_stencil() {
                        vk::ImageAspectFlags::DEPTH
                            | if texture_format.has_stencil() {
                                vk::ImageAspectFlags::STENCIL
                            } else {
                                vk::ImageAspectFlags::empty()
                            }
                    } else {
                        vk::ImageAspectFlags::COLOR
                    }
                }
            };
            let info = vk::ImageViewCreateInfo::default()
                .image(*image)
                .format(*format)
                .view_type(view_type)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: desc.base_mip_level,
                    level_count: desc.mip_level_count.unwrap(),
                    base_array_layer: desc.base_array_layer,
                    layer_count: desc.array_layer_count.unwrap(),
                });
            let view = unsafe { device.create_image_view(&info, None) }.map_err(|e| {
                crate::GraphicsError::ResourceCreationFailed(format!("texture view: {e:?}"))
            })?;
            GpuTextureView::Vulkan {
                device: device.clone(),
                view,
            }
        }
    })
}
