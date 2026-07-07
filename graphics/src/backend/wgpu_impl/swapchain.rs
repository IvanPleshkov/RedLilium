//! wgpu surface implementation.
//!
//! This module contains the wgpu-specific surface and surface texture handling.

use std::sync::Arc;

use super::conversion::{convert_present_mode, convert_texture_format};
use super::{SurfaceTextureView, WgpuBackend};
use crate::error::GraphicsError;
use crate::swapchain::SurfaceConfiguration;

/// Configure a wgpu surface.
///
/// Everything is validated against `Surface::get_capabilities` first: wgpu
/// panics on an invalid configuration (unsupported format/present mode, zero
/// size), so unchecked values must never reach `surface.configure`.
pub fn configure_surface(
    surface: &wgpu::Surface<'static>,
    backend: &WgpuBackend,
    config: &SurfaceConfiguration,
) -> Result<(), GraphicsError> {
    if config.width == 0 || config.height == 0 {
        return Err(GraphicsError::InvalidParameter(format!(
            "surface extent is {}x{} (window minimized?); configuration skipped",
            config.width, config.height
        )));
    }

    let caps = surface.get_capabilities(backend.adapter());

    // The requested format must be supported exactly — all pipelines are
    // compiled against it, so substituting another format would mismatch
    // every render pass.
    let format = convert_texture_format(config.format);
    if !caps.formats.contains(&format) {
        return Err(GraphicsError::InvalidParameter(format!(
            "surface does not support format {:?}; supported: {:?}",
            config.format, caps.formats
        )));
    }

    // Unsupported present mode falls back to FIFO (always available) with a
    // warning — matching the Vulkan backend, where the same fallback applies.
    let mut present_mode = convert_present_mode(config.present_mode);
    if !caps.present_modes.contains(&present_mode) {
        log::warn!(
            "present mode {:?} not supported by surface (supported: {:?}); falling back to Fifo",
            config.present_mode,
            caps.present_modes
        );
        present_mode = wgpu::PresentMode::Fifo;
    }

    // Prefer opaque composition; otherwise take whatever the surface offers
    // (Wayland/Android surfaces may only report Inherit/PreMultiplied).
    let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
        wgpu::CompositeAlphaMode::Opaque
    } else {
        caps.alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto)
    };

    // RENDER_ATTACHMENT is mandatory; COPY_SRC is added opportunistically so
    // the final frame can be read back (screenshots, tests).
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    if caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }

    let wgpu_config = wgpu::SurfaceConfiguration {
        usage,
        format,
        width: config.width,
        height: config.height,
        present_mode,
        alpha_mode,
        view_formats: vec![],
        desired_maximum_frame_latency: config.frames_in_flight as u32,
    };
    surface.configure(backend.device(), &wgpu_config);
    log::info!(
        "Configured wgpu surface: {}x{} {:?} {:?}",
        config.width,
        config.height,
        format,
        present_mode
    );
    Ok(())
}

/// Acquire a surface texture from a wgpu surface.
///
/// Surface errors are mapped to their recovery strategy: `SurfaceOutdated` /
/// `SurfaceLost` mean "reconfigure and retry", `Timeout` means "skip this
/// frame", `OutOfMemory` is fatal.
pub fn acquire_surface_texture(
    surface: &wgpu::Surface<'static>,
) -> Result<WgpuSurfaceAcquireResult, GraphicsError> {
    let surface_texture = surface.get_current_texture().map_err(|e| match e {
        wgpu::SurfaceError::Outdated => GraphicsError::SurfaceOutdated,
        wgpu::SurfaceError::Lost => GraphicsError::SurfaceLost,
        wgpu::SurfaceError::Timeout => {
            GraphicsError::Timeout("surface texture acquire timed out".into())
        }
        wgpu::SurfaceError::OutOfMemory => GraphicsError::OutOfMemory,
        wgpu::SurfaceError::Other => {
            GraphicsError::Internal("Failed to acquire surface texture".into())
        }
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
///
/// Returns `Err(SurfaceOutdated)` if the surface reported the texture as
/// suboptimal — the frame is still shown, but the surface should be
/// reconfigured before the next acquire.
pub fn present_surface_texture(texture: wgpu::SurfaceTexture) -> Result<(), GraphicsError> {
    let suboptimal = texture.suboptimal;
    texture.present();
    if suboptimal {
        Err(GraphicsError::SurfaceOutdated)
    } else {
        Ok(())
    }
}
