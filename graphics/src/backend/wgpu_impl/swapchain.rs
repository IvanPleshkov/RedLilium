//! wgpu surface implementation.
//!
//! This module contains the wgpu-specific surface and surface texture handling.

use std::sync::Arc;

use super::conversion::{convert_present_mode, convert_texture_format};
use super::{SurfaceTextureView, WgpuBackend};
use crate::error::GraphicsError;
use crate::swapchain::SurfaceConfiguration;

/// Configure a wgpu surface.
pub fn configure_surface(
    surface: &wgpu::Surface<'static>,
    backend: &WgpuBackend,
    config: &SurfaceConfiguration,
) {
    let wgpu_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: convert_texture_format(config.format),
        width: config.width,
        height: config.height,
        present_mode: convert_present_mode(config.present_mode),
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(backend.device(), &wgpu_config);
    log::info!("Configured wgpu surface");
}

/// Acquire a surface texture from a wgpu surface.
pub fn acquire_surface_texture(
    surface: &wgpu::Surface<'static>,
) -> Result<WgpuSurfaceAcquireResult, GraphicsError> {
    let surface_texture = surface.get_current_texture().map_err(|e| {
        GraphicsError::ResourceCreationFailed(format!("Failed to acquire surface texture: {e}"))
    })?;

    let view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let surface_view = SurfaceTextureView {
        view: Arc::new(view),
    };

    Ok(WgpuSurfaceAcquireResult {
        texture: surface_texture,
        view: surface_view,
    })
}

/// Result of acquiring a wgpu surface texture.
pub struct WgpuSurfaceAcquireResult {
    /// The raw surface texture (needed for presentation).
    pub texture: wgpu::SurfaceTexture,
    /// The texture view for rendering.
    pub view: SurfaceTextureView,
}

/// Present a wgpu surface texture.
pub fn present_surface_texture(texture: wgpu::SurfaceTexture) {
    texture.present();
}
