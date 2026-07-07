//! Swapchain and surface management.
//!
//! This module provides abstractions for presenting rendered frames to a window.
//!
//! # Overview
//!
//! - [`Surface`] - Represents a window surface that can display rendered content
//! - [`SurfaceConfiguration`] - Configuration for the surface (format, size, present mode)
//! - [`SurfaceTexture`] - A texture from the swapchain that will be presented
//! - [`PresentMode`] - Controls vsync behavior
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::{GraphicsInstance, SurfaceConfiguration, PresentMode};
//!
//! let instance = GraphicsInstance::new()?;
//! let surface = instance.create_surface(&window)?;
//!
//! let config = SurfaceConfiguration::new(1920, 1080)
//!     .with_format(surface.preferred_format())
//!     .with_present_mode(PresentMode::Fifo);
//! surface.configure(&device, &config);
//!
//! // In render loop:
//! let frame = surface.acquire_texture()?;
//! // ... render to frame.texture() ...
//! frame.present()?; // Err(SurfaceOutdated) => reconfigure before next acquire
//! ```

use std::sync::{Arc, RwLock};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::backend::{GpuSurface, GpuSurfaceTexture};
use crate::device::GraphicsDevice;
use crate::error::GraphicsError;
use crate::instance::GraphicsInstance;
use crate::types::TextureFormat;

/// Presentation mode for the swapchain.
///
/// Controls how frames are synchronized with the display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PresentMode {
    /// No synchronization. May cause tearing but has lowest latency.
    Immediate,
    /// Triple buffering. Low latency without tearing.
    Mailbox,
    /// VSync enabled. No tearing, but may have higher latency.
    #[default]
    Fifo,
    /// VSync with relaxed timing. May tear if a frame is late.
    FifoRelaxed,
}

/// Configuration for a surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceConfiguration {
    /// The texture format for the swapchain.
    pub format: TextureFormat,
    /// Width of the surface in pixels.
    pub width: u32,
    /// Height of the surface in pixels.
    pub height: u32,
    /// Presentation mode (vsync behavior).
    pub present_mode: PresentMode,
    /// Number of frames that can be in flight simultaneously.
    ///
    /// Controls how many sets of synchronization primitives (semaphores, fences)
    /// the swapchain creates. Higher values allow more CPU-GPU overlap but
    /// increase input latency and memory usage.
    ///
    /// | Count | Behavior |
    /// |-------|----------|
    /// | 1 | CPU waits for GPU every frame. Simple but slow. |
    /// | 2 | Good balance. CPU works on N+1 while GPU renders N. (Default) |
    /// | 3 | More overlap, higher latency. Useful for VR or heavy CPU work. |
    pub frames_in_flight: usize,
}

impl SurfaceConfiguration {
    /// Create a new surface configuration.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            format: TextureFormat::Bgra8Unorm,
            width,
            height,
            present_mode: PresentMode::default(),
            frames_in_flight: 2,
        }
    }

    /// Set the texture format.
    pub fn with_format(mut self, format: TextureFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the present mode.
    pub fn with_present_mode(mut self, present_mode: PresentMode) -> Self {
        self.present_mode = present_mode;
        self
    }

    /// Set the number of frames in flight.
    ///
    /// Must be at least 1. Defaults to 2.
    pub fn with_frames_in_flight(mut self, frames_in_flight: usize) -> Self {
        assert!(frames_in_flight > 0, "frames_in_flight must be at least 1");
        self.frames_in_flight = frames_in_flight;
        self
    }
}

/// A surface for presenting rendered frames to a window.
///
/// The surface is created from a window using [`GraphicsInstance::create_surface`].
/// It must be configured with [`Surface::configure`] before use.
pub struct Surface {
    config: RwLock<Option<SurfaceConfiguration>>,
    /// Current frame index (cycles through swapchain images).
    frame_index: RwLock<u64>,
    /// The backend-specific GPU surface.
    gpu_surface: GpuSurface,
    /// Keep-alives, declared after `gpu_surface` deliberately: fields drop in
    /// declaration order, and surface/swapchain teardown needs the backend
    /// (owned by the device→instance chain) to still be alive. See #50.
    device: RwLock<Option<Arc<GraphicsDevice>>>,
    instance: Arc<GraphicsInstance>,
}

impl Surface {
    /// Create a new surface from an owned window.
    ///
    /// The `Arc<W>` is retained by the backend surface, so the window is
    /// guaranteed to outlive it (no lifetime erasure).
    pub(crate) fn new<W>(
        instance: Arc<GraphicsInstance>,
        window: Arc<W>,
    ) -> Result<Self, GraphicsError>
    where
        W: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static,
    {
        log::info!("Creating surface from window");

        let gpu_surface = instance.backend().create_surface(window)?;

        Ok(Self {
            instance,
            config: RwLock::new(None),
            device: RwLock::new(None),
            frame_index: RwLock::new(0),
            gpu_surface,
        })
    }

    /// Get the parent graphics instance.
    pub fn instance(&self) -> &Arc<GraphicsInstance> {
        &self.instance
    }

    /// Get the underlying GPU surface (internal use only).
    pub(crate) fn gpu_surface(&self) -> &GpuSurface {
        &self.gpu_surface
    }

    /// Get the preferred texture format for this surface.
    ///
    /// Prefers sRGB formats (e.g. `Bgra8UnormSrgb`) when available so that
    /// shaders can output linear values and the hardware applies the
    /// linear-to-sRGB conversion automatically. Falls back to the first
    /// supported format if no sRGB variant is available.
    pub fn preferred_format(&self) -> TextureFormat {
        let formats = self.supported_formats();
        // Prefer sRGB so both Vulkan and wgpu get consistent gamma handling
        if let Some(&srgb) = formats.iter().find(|f| f.is_srgb()) {
            return srgb;
        }
        formats
            .first()
            .copied()
            .unwrap_or(TextureFormat::Bgra8Unorm)
    }

    /// Get the supported texture formats for this surface.
    ///
    /// Queries actual GPU capabilities for supported surface formats.
    /// This includes both standard SDR formats and HDR formats when
    /// the display supports them. An empty result means the query failed —
    /// no format is guaranteed to work, so no fallback is invented here.
    pub fn supported_formats(&self) -> Vec<TextureFormat> {
        self.gpu_surface
            .get_supported_formats(&self.instance.backend())
    }

    /// Get the supported HDR texture formats for this surface.
    ///
    /// Returns only HDR-capable formats from the actual GPU capabilities.
    /// An empty result means HDR is not supported by this display.
    pub fn supported_hdr_formats(&self) -> Vec<TextureFormat> {
        self.supported_formats()
            .into_iter()
            .filter(|f| f.is_hdr())
            .collect()
    }

    /// Get the preferred HDR format for this surface.
    ///
    /// Returns the first available HDR format from the GPU capabilities,
    /// preferring `Rgba16Float` for maximum precision, then `Rgba10a2Unorm` (HDR10).
    /// Falls back to `Rgba16Float` if no HDR formats are available.
    pub fn preferred_hdr_format(&self) -> TextureFormat {
        let hdr_formats = self.supported_hdr_formats();
        // Prefer Rgba16Float for highest precision, then Rgba10a2Unorm
        if hdr_formats.contains(&TextureFormat::Rgba16Float) {
            TextureFormat::Rgba16Float
        } else if hdr_formats.contains(&TextureFormat::Rgba10a2Unorm) {
            TextureFormat::Rgba10a2Unorm
        } else {
            hdr_formats
                .first()
                .copied()
                .unwrap_or(TextureFormat::Rgba16Float)
        }
    }

    /// Get the present modes actually supported by this surface.
    ///
    /// Queries GPU capabilities; [`PresentMode::Fifo`] is always included
    /// (both Vulkan and wgpu guarantee it).
    pub fn supported_present_modes(&self) -> Vec<PresentMode> {
        self.gpu_surface
            .get_supported_present_modes(&self.instance.backend())
    }

    /// Configure the surface for rendering.
    ///
    /// This must be called before acquiring textures. It should also be called
    /// when the window is resized.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn configure(
        &self,
        device: &Arc<GraphicsDevice>,
        config: &SurfaceConfiguration,
    ) -> Result<(), GraphicsError> {
        // Validate configuration
        if config.width == 0 || config.height == 0 {
            return Err(GraphicsError::InvalidParameter(
                "surface dimensions cannot be zero".to_string(),
            ));
        }

        if !self.supported_formats().contains(&config.format) {
            return Err(GraphicsError::InvalidParameter(format!(
                "unsupported surface format: {:?}",
                config.format
            )));
        }

        // Clamp an unsupported present mode to FIFO (guaranteed available)
        // instead of passing it through: Vulkan would silently fall back
        // while wgpu would panic — this makes both backends behave the same,
        // with a warning.
        let mut config = config.clone();
        if !self
            .supported_present_modes()
            .contains(&config.present_mode)
        {
            log::warn!(
                "present mode {:?} not supported by surface; falling back to Fifo",
                config.present_mode
            );
            config.present_mode = PresentMode::Fifo;
        }
        let config = &config;

        log::info!(
            "Configuring surface: {}x{} {:?} {:?}",
            config.width,
            config.height,
            config.format,
            config.present_mode
        );

        // Configure the backend-specific surface
        self.gpu_surface
            .configure(&self.instance.backend(), config)?;

        // Store the configuration
        if let Ok(mut current_config) = self.config.write() {
            *current_config = Some(config.clone());
        }
        if let Ok(mut current_device) = self.device.write() {
            *current_device = Some(Arc::clone(device));
        }

        Ok(())
    }

    /// Get the current configuration, if set.
    pub fn config(&self) -> Option<SurfaceConfiguration> {
        self.config.read().ok().and_then(|c| c.clone())
    }

    /// Get the current width, if configured.
    pub fn width(&self) -> Option<u32> {
        self.config().map(|c| c.width)
    }

    /// Get the current height, if configured.
    pub fn height(&self) -> Option<u32> {
        self.config().map(|c| c.height)
    }

    /// Acquire the next texture from the swapchain.
    ///
    /// The returned [`SurfaceTexture`] must be presented or dropped before
    /// the next frame can be acquired.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The surface is not configured
    /// - The surface is outdated (window was resized)
    /// - The surface was lost
    pub fn acquire_texture(&self) -> Result<SurfaceTexture, GraphicsError> {
        let config = self
            .config
            .read()
            .ok()
            .and_then(|c| c.clone())
            .ok_or_else(|| GraphicsError::InvalidParameter("surface not configured".to_string()))?;

        let device = self
            .device
            .read()
            .ok()
            .and_then(|d| d.clone())
            .ok_or_else(|| GraphicsError::InvalidParameter("surface not configured".to_string()))?;

        // Increment frame index (cycles through frames in flight)
        let frame_index = {
            let mut idx = self.frame_index.write().map_err(|_| {
                GraphicsError::Internal("failed to acquire frame index lock".to_string())
            })?;
            let current = *idx;
            *idx = (*idx + 1) % config.frames_in_flight as u64;
            current
        };

        // Acquire the backend-specific surface texture
        let gpu_texture = self.acquire_gpu_texture()?;

        Ok(SurfaceTexture {
            device,
            instance: Arc::clone(&self.instance),
            format: config.format,
            width: config.width,
            height: config.height,
            frame_index,
            presented: RwLock::new(false),
            gpu_texture: Some(gpu_texture),
        })
    }

    /// Acquire the backend-specific surface texture.
    fn acquire_gpu_texture(&self) -> Result<GpuSurfaceTexture, GraphicsError> {
        self.gpu_surface.acquire_texture(&self.instance.backend())
    }
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("config", &self.config())
            .finish()
    }
}

// GpuSurface is explicitly marked as Send + Sync in backend/mod.rs

/// A texture acquired from the swapchain for rendering.
///
/// This represents the current frame's render target. After rendering,
/// call [`SurfaceTexture::present`] to display the frame.
///
/// If dropped without presenting, the frame is discarded.
pub struct SurfaceTexture {
    format: TextureFormat,
    width: u32,
    height: u32,
    frame_index: u64,
    presented: RwLock<bool>,
    /// The backend-specific surface texture.
    gpu_texture: Option<GpuSurfaceTexture>,
    /// Keep-alives, declared after `gpu_texture` deliberately: fields drop in
    /// declaration order, and the texture's teardown needs the backend
    /// (owned by the device→instance chain) to still be alive. See #50.
    device: Arc<GraphicsDevice>,
    instance: Arc<GraphicsInstance>,
}

impl SurfaceTexture {
    /// Get the device associated with this texture.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Get the texture format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Get the texture width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the texture height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the frame index (for debugging/profiling).
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Get the underlying GPU surface texture.
    ///
    /// This provides access to backend-specific texture views for rendering.
    /// Use `GpuSurfaceTexture::wgpu_view()` or `GpuSurfaceTexture::vulkan_view()`
    /// to get the backend-specific view.
    pub fn gpu_texture(&self) -> Option<&GpuSurfaceTexture> {
        self.gpu_texture.as_ref()
    }

    /// Present the texture to the screen.
    ///
    /// This displays the rendered content in the window. After calling this,
    /// the texture is no longer valid for rendering.
    ///
    /// # Errors
    ///
    /// [`GraphicsError::SurfaceOutdated`] / [`GraphicsError::SurfaceLost`]
    /// mean the surface no longer matches the window (resize, monitor
    /// change) and must be reconfigured with [`Surface::configure`] before
    /// the next acquire; the frame itself may still have been shown. Other
    /// errors indicate the frame was not presented.
    pub fn present(mut self) -> Result<(), GraphicsError> {
        if let Ok(mut presented) = self.presented.write() {
            *presented = true;
        }

        match self.gpu_texture.take() {
            Some(gpu_texture) => gpu_texture.present(&self.instance.backend(), self.frame_index),
            None => Ok(()),
        }
    }
}

impl std::fmt::Debug for SurfaceTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceTexture")
            .field("format", &self.format)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("frame_index", &self.frame_index)
            .finish()
    }
}

impl Drop for SurfaceTexture {
    fn drop(&mut self) {
        // Dropped without presenting (error path, panic unwind). The acquired
        // image must still be released: on Vulkan its pending semaphore
        // signals can only be drained by presenting, so `abandon` presents
        // the undrawn image — one undefined frame beats permanently poisoned
        // sync objects. On wgpu this is a plain drop.
        if let Some(gpu_texture) = self.gpu_texture.take() {
            gpu_texture.abandon(&self.instance.backend(), self.frame_index);
        }
    }
}

// Ensure SurfaceTexture is Send + Sync
static_assertions::assert_impl_all!(SurfaceTexture: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_configuration() {
        let config = SurfaceConfiguration::new(1920, 1080)
            .with_format(TextureFormat::Rgba8Unorm)
            .with_present_mode(PresentMode::Mailbox);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.format, TextureFormat::Rgba8Unorm);
        assert_eq!(config.present_mode, PresentMode::Mailbox);
    }

    #[test]
    fn test_present_mode_default() {
        assert_eq!(PresentMode::default(), PresentMode::Fifo);
    }

    #[test]
    fn test_surface_configuration_default_format() {
        let config = SurfaceConfiguration::new(800, 600);
        assert_eq!(config.format, TextureFormat::Bgra8Unorm);
        assert_eq!(config.present_mode, PresentMode::Fifo);
        assert_eq!(config.frames_in_flight, 2);
    }
}
