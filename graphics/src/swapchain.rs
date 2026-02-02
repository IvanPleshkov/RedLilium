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

use crate::device::GraphicsDevice;
use crate::error::GraphicsError;
use crate::instance::GraphicsInstance;
use crate::types::TextureFormat;

#[cfg(feature = "wgpu-backend")]
use crate::backend::wgpu_impl::SurfaceTextureView;

#[cfg(feature = "vulkan-backend")]
use crate::backend::vulkan::VulkanSurfaceTextureView;

#[cfg(feature = "vulkan-backend")]
use ash::vk;

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
    /// The underlying wgpu surface (only when using wgpu backend).
    #[cfg(feature = "wgpu-backend")]
    wgpu_surface: Option<wgpu::Surface<'static>>,
    /// The underlying Vulkan surface (only when using vulkan backend).
    #[cfg(feature = "vulkan-backend")]
    vulkan_surface: Option<vk::SurfaceKHR>,
    /// The Vulkan swapchain (only when using vulkan backend).
    #[cfg(feature = "vulkan-backend")]
    vulkan_swapchain: RwLock<Option<crate::backend::vulkan::swapchain::VulkanSwapchain>>,
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

        #[cfg(feature = "wgpu-backend")]
        let wgpu_surface = {
            use crate::backend::GpuBackend;
            match instance.backend() {
                GpuBackend::Wgpu(wgpu_backend) => {
                    // Create wgpu surface from window
                    // SAFETY: The caller guarantees the window handle remains valid for the
                    // lifetime of the surface. We transmute to 'static to satisfy wgpu's
                    // Surface<'static> requirement, but the Surface is dropped before the
                    // window in practice.
                    let surface: wgpu::Surface<'static> = unsafe {
                        std::mem::transmute(
                            wgpu_backend
                                .instance()
                                .create_surface(window)
                                .map_err(|e| {
                                    GraphicsError::ResourceCreationFailed(format!(
                                        "Failed to create wgpu surface: {e}"
                                    ))
                                })?,
                        )
                    };
                    Some(surface)
                }
                _ => None,
            }
        };

        #[cfg(feature = "vulkan-backend")]
        let vulkan_surface = {
            use crate::backend::GpuBackend;
            match instance.backend() {
                GpuBackend::Vulkan(vulkan_backend) => {
                    // Create Vulkan surface from window
                    let display_handle = window.display_handle().map_err(|e| {
                        GraphicsError::ResourceCreationFailed(format!(
                            "Failed to get display handle: {e}"
                        ))
                    })?;
                    let window_handle = window.window_handle().map_err(|e| {
                        GraphicsError::ResourceCreationFailed(format!(
                            "Failed to get window handle: {e}"
                        ))
                    })?;

                    let surface = unsafe {
                        ash_window::create_surface(
                            vulkan_backend.entry(),
                            vulkan_backend.instance(),
                            display_handle.as_raw(),
                            window_handle.as_raw(),
                            None,
                        )
                    }
                    .map_err(|e| {
                        GraphicsError::ResourceCreationFailed(format!(
                            "Failed to create Vulkan surface: {e}"
                        ))
                    })?;
                    Some(surface)
                }
                _ => None,
            }
        };

        Ok(Self {
            instance,
            config: RwLock::new(None),
            device: RwLock::new(None),
            frame_index: RwLock::new(0),
            #[cfg(feature = "wgpu-backend")]
            wgpu_surface,
            #[cfg(feature = "vulkan-backend")]
            vulkan_surface,
            #[cfg(feature = "vulkan-backend")]
            vulkan_swapchain: RwLock::new(None),
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

        // Configure the wgpu surface if available
        #[cfg(feature = "wgpu-backend")]
        if let Some(wgpu_surface) = &self.wgpu_surface {
            use crate::backend::GpuBackend;
            if let GpuBackend::Wgpu(wgpu_backend) = self.instance.backend() {
                crate::backend::wgpu_impl::swapchain::configure_surface(
                    wgpu_surface,
                    wgpu_backend,
                    config,
                );
            }
        }

        // Configure the Vulkan swapchain if available
        #[cfg(feature = "vulkan-backend")]
        if let Some(vulkan_surface) = self.vulkan_surface {
            use crate::backend::GpuBackend;
            use crate::backend::vulkan::swapchain::VulkanSwapchain;

            if let GpuBackend::Vulkan(vulkan_backend) = self.instance.backend() {
                // Destroy old swapchain if it exists
                if let Ok(mut swapchain_guard) = self.vulkan_swapchain.write()
                    && let Some(ref mut old_swapchain) = *swapchain_guard
                {
                    old_swapchain.destroy(vulkan_backend);
                    *swapchain_guard = None;
                }

                // Create new swapchain
                let new_swapchain = VulkanSwapchain::new(vulkan_backend, vulkan_surface, config)?;

                if let Ok(mut swapchain_guard) = self.vulkan_swapchain.write() {
                    *swapchain_guard = Some(new_swapchain);
                }
                log::info!("Configured Vulkan swapchain");
            }
        }

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

        // Acquire the actual wgpu surface texture if using wgpu backend
        #[cfg(feature = "wgpu-backend")]
        let (wgpu_texture, wgpu_view) = if let Some(wgpu_surface) = &self.wgpu_surface {
            let result =
                crate::backend::wgpu_impl::swapchain::acquire_surface_texture(wgpu_surface)?;
            (Some(result.texture), Some(result.view))
        } else {
            (None, None)
        };

        // Acquire the Vulkan swapchain image if using Vulkan backend
        #[cfg(feature = "vulkan-backend")]
        let (
            vulkan_view,
            vulkan_image_index,
            vulkan_frame_index,
            vulkan_swapchain_handle,
            vulkan_image_available_semaphore,
            vulkan_render_finished_semaphore,
            vulkan_in_flight_fence,
            vulkan_present_command_buffer,
        ) = {
            use crate::backend::GpuBackend;

            if let GpuBackend::Vulkan(vulkan_backend) = self.instance.backend() {
                if let Ok(mut swapchain_guard) = self.vulkan_swapchain.write() {
                    if let Some(ref mut swapchain) = *swapchain_guard {
                        let result = swapchain.acquire_next_image(vulkan_backend)?;
                        (
                            Some(result.view),
                            result.image_index,
                            result.frame_index,
                            Some(result.swapchain),
                            result.image_available_semaphore,
                            result.render_finished_semaphore,
                            result.in_flight_fence,
                            result.present_command_buffer,
                        )
                    } else {
                        (
                            None,
                            0,
                            0,
                            None,
                            vk::Semaphore::null(),
                            vk::Semaphore::null(),
                            vk::Fence::null(),
                            vk::CommandBuffer::null(),
                        )
                    }
                } else {
                    (
                        None,
                        0,
                        0,
                        None,
                        vk::Semaphore::null(),
                        vk::Semaphore::null(),
                        vk::Fence::null(),
                        vk::CommandBuffer::null(),
                    )
                }
            } else {
                (
                    None,
                    0,
                    0,
                    None,
                    vk::Semaphore::null(),
                    vk::Semaphore::null(),
                    vk::Fence::null(),
                    vk::CommandBuffer::null(),
                )
            }
        };

        log::trace!("Acquired surface texture, frame index: {}", frame_index);

        Ok(SurfaceTexture {
            device,
            instance: Arc::clone(&self.instance),
            format: config.format,
            width: config.width,
            height: config.height,
            frame_index,
            presented: RwLock::new(false),
            #[cfg(feature = "wgpu-backend")]
            wgpu_texture,
            #[cfg(feature = "wgpu-backend")]
            wgpu_view,
            #[cfg(feature = "vulkan-backend")]
            vulkan_view,
            #[cfg(feature = "vulkan-backend")]
            vulkan_image_index,
            #[cfg(feature = "vulkan-backend")]
            vulkan_frame_index,
            #[cfg(feature = "vulkan-backend")]
            vulkan_swapchain: vulkan_swapchain_handle,
            #[cfg(feature = "vulkan-backend")]
            vulkan_image_available_semaphore,
            #[cfg(feature = "vulkan-backend")]
            vulkan_render_finished_semaphore,
            #[cfg(feature = "vulkan-backend")]
            vulkan_in_flight_fence,
            #[cfg(feature = "vulkan-backend")]
            vulkan_present_command_buffer,
        })
    }
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("config", &self.config())
            .finish()
    }
}

// Surface is Send + Sync when not using wgpu backend.
// With wgpu backend, the Surface contains wgpu::Surface which may not be Send + Sync
// on all platforms, so we skip the assertion.
#[cfg(not(feature = "wgpu-backend"))]
static_assertions::assert_impl_all!(Surface: Send, Sync);

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
    /// The underlying wgpu surface texture (only when using wgpu backend).
    #[cfg(feature = "wgpu-backend")]
    wgpu_texture: Option<wgpu::SurfaceTexture>,
    /// The texture view for rendering (only when using wgpu backend).
    #[cfg(feature = "wgpu-backend")]
    wgpu_view: Option<SurfaceTextureView>,
    /// The Vulkan texture view for rendering (only when using vulkan backend).
    #[cfg(feature = "vulkan-backend")]
    vulkan_view: Option<VulkanSurfaceTextureView>,
    /// The swapchain image index (only when using vulkan backend).
    #[cfg(feature = "vulkan-backend")]
    vulkan_image_index: u32,
    /// The Vulkan frame-in-flight index (for sync primitive lookup).
    #[cfg(feature = "vulkan-backend")]
    #[allow(dead_code)] // Reserved for future use in advanced synchronization
    vulkan_frame_index: usize,
    /// The Vulkan swapchain handle (for presentation).
    #[cfg(feature = "vulkan-backend")]
    vulkan_swapchain: Option<vk::SwapchainKHR>,
    /// The image available semaphore for this frame (for presentation wait).
    #[cfg(feature = "vulkan-backend")]
    vulkan_image_available_semaphore: vk::Semaphore,
    /// The render finished semaphore for this frame (for presentation signal).
    #[cfg(feature = "vulkan-backend")]
    vulkan_render_finished_semaphore: vk::Semaphore,
    /// The in-flight fence for this frame (to signal after presentation).
    #[cfg(feature = "vulkan-backend")]
    vulkan_in_flight_fence: vk::Fence,
    /// The command buffer for this frame's presentation.
    #[cfg(feature = "vulkan-backend")]
    vulkan_present_command_buffer: vk::CommandBuffer,
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

    /// Get the wgpu texture view for rendering (only available with wgpu backend).
    #[cfg(feature = "wgpu-backend")]
    pub fn wgpu_view(&self) -> Option<SurfaceTextureView> {
        self.wgpu_view.clone()
    }

    /// Get the wgpu texture view for rendering (stub for non-wgpu builds).
    #[cfg(not(feature = "wgpu-backend"))]
    pub fn wgpu_view(&self) -> Option<()> {
        None
    }

    /// Get the Vulkan texture view for rendering (only available with vulkan backend).
    #[cfg(feature = "vulkan-backend")]
    pub fn vulkan_view(&self) -> Option<VulkanSurfaceTextureView> {
        self.vulkan_view.clone()
    }

    /// Get the Vulkan texture view for rendering (stub for non-vulkan builds).
    #[cfg(not(feature = "vulkan-backend"))]
    pub fn vulkan_view(&self) -> Option<()> {
        None
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

        // Present the wgpu surface texture
        #[cfg(feature = "wgpu-backend")]
        if let Some(wgpu_texture) = self.wgpu_texture.take() {
            crate::backend::wgpu_impl::swapchain::present_surface_texture(wgpu_texture);
        }

        // Present the Vulkan swapchain image
        #[cfg(feature = "vulkan-backend")]
        if let (Some(vulkan_view), Some(swapchain)) = (&self.vulkan_view, self.vulkan_swapchain)
            && let crate::backend::GpuBackend::Vulkan(vulkan_backend) = self.instance.backend()
            && let Err(e) = crate::backend::vulkan::swapchain::present_vulkan_frame(
                vulkan_backend,
                vulkan_view,
                swapchain,
                self.vulkan_image_index,
                self.vulkan_image_available_semaphore,
                self.vulkan_render_finished_semaphore,
                self.vulkan_in_flight_fence,
                self.vulkan_present_command_buffer,
                self.frame_index,
            )
        {
            log::error!("Failed to present Vulkan frame: {}", e);
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
