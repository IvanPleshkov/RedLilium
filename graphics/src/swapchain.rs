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
//! let config = SurfaceConfiguration {
//!     format: surface.preferred_format(),
//!     width: 1920,
//!     height: 1080,
//!     present_mode: PresentMode::Fifo,
//! };
//! surface.configure(&device, &config);
//!
//! // In render loop:
//! let frame = surface.acquire_texture()?;
//! // ... render to frame.texture() ...
//! frame.present();
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
}

impl SurfaceConfiguration {
    /// Create a new surface configuration.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            format: TextureFormat::Bgra8Unorm,
            width,
            height,
            present_mode: PresentMode::default(),
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
}

/// A surface for presenting rendered frames to a window.
///
/// The surface is created from a window using [`GraphicsInstance::create_surface`].
/// It must be configured with [`Surface::configure`] before use.
pub struct Surface {
    instance: Arc<GraphicsInstance>,
    config: RwLock<Option<SurfaceConfiguration>>,
    device: RwLock<Option<Arc<GraphicsDevice>>>,
    /// Current frame index (cycles through swapchain images).
    frame_index: RwLock<u64>,
    /// The backend-specific GPU surface.
    gpu_surface: GpuSurface,
}

impl Surface {
    /// Create a new surface from a window.
    ///
    /// # Safety
    ///
    /// The window handle must remain valid for the lifetime of the surface.
    pub(crate) fn new<W>(instance: Arc<GraphicsInstance>, window: &W) -> Result<Self, GraphicsError>
    where
        W: HasWindowHandle + HasDisplayHandle + Sync,
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

    /// Get the preferred texture format for this surface.
    ///
    /// This format is guaranteed to be supported and is typically the most efficient.
    pub fn preferred_format(&self) -> TextureFormat {
        // Most platforms prefer BGRA8 for swapchain
        TextureFormat::Bgra8Unorm
    }

    /// Get the supported texture formats for this surface.
    pub fn supported_formats(&self) -> Vec<TextureFormat> {
        vec![
            TextureFormat::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
        ]
    }

    /// Get the supported present modes for this surface.
    pub fn supported_present_modes(&self) -> Vec<PresentMode> {
        vec![
            PresentMode::Fifo, // Always supported
            PresentMode::Immediate,
            PresentMode::Mailbox,
            PresentMode::FifoRelaxed,
        ]
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

        log::info!(
            "Configuring surface: {}x{} {:?} {:?}",
            config.width,
            config.height,
            config.format,
            config.present_mode
        );

        // Configure the backend-specific surface
        self.gpu_surface
            .configure(self.instance.backend(), config)?;

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

        // Increment frame index
        let frame_index = {
            let mut idx = self.frame_index.write().map_err(|_| {
                GraphicsError::Internal("failed to acquire frame index lock".to_string())
            })?;
            let current = *idx;
            *idx = (*idx + 1) % 3; // Triple buffering
            current
        };

        // Acquire the backend-specific surface texture
        let gpu_texture = self.acquire_gpu_texture()?;

        log::trace!("Acquired surface texture, frame index: {}", frame_index);

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
        self.gpu_surface.acquire_texture(self.instance.backend())
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
    device: Arc<GraphicsDevice>,
    instance: Arc<GraphicsInstance>,
    format: TextureFormat,
    width: u32,
    height: u32,
    frame_index: u64,
    presented: RwLock<bool>,
    /// The backend-specific surface texture.
    gpu_texture: Option<GpuSurfaceTexture>,
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
    pub fn present(mut self) {
        if let Ok(mut presented) = self.presented.write() {
            *presented = true;
        }
        log::trace!("Presenting frame {}", self.frame_index);

        if let Some(gpu_texture) = self.gpu_texture.take() {
            gpu_texture.present(self.instance.backend(), self.frame_index);
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
        let presented = self.presented.read().map(|p| *p).unwrap_or(false);
        if !presented {
            log::trace!(
                "SurfaceTexture dropped without presenting (frame {})",
                self.frame_index
            );
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
    }
}
