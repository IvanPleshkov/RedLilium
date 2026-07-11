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

    // `config.format` is the engine's *render/view* format (sRGB where possible;
    // pipelines and the present blit target it). Surfaces that offer it directly
    // (native Vulkan/Metal/DX) configure the canvas with it and need no extra
    // view formats. WebGPU only exposes the non-sRGB base canvas format, so the
    // canvas is configured with the base and the sRGB form is added as a texture
    // *view* format — the acquired view is then created as sRGB so the hardware
    // encode still runs (#33).
    let requested = convert_texture_format(config.format);
    let (canvas_format, view_formats) = if caps.formats.contains(&requested) {
        (requested, vec![])
    } else {
        let base = requested.remove_srgb_suffix();
        if requested != base && caps.formats.contains(&base) {
            (base, vec![requested])
        } else {
            return Err(GraphicsError::InvalidParameter(format!(
                "surface does not support format {:?} (nor its non-sRGB base); supported: {:?}",
                config.format, caps.formats
            )));
        }
    };

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
        format: canvas_format,
        width: config.width,
        height: config.height,
        present_mode,
        alpha_mode,
        view_formats,
        desired_maximum_frame_latency: config.frames_in_flight as u32,
    };
    surface.configure(backend.device(), &wgpu_config);
    log::info!(
        "Configured wgpu surface: {}x{} canvas={:?} view={:?} {:?}",
        config.width,
        config.height,
        canvas_format,
        requested,
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
    view_format: crate::types::TextureFormat,
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

    // Create the view with the engine's render format. On native this equals the
    // canvas texture format; on WebGPU the canvas is a non-sRGB base format and
    // this sRGB view (declared in the surface's `view_formats`) restores the
    // hardware linear→sRGB encode (#33).
    let view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor {
            format: Some(convert_texture_format(view_format)),
            ..Default::default()
        });

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
