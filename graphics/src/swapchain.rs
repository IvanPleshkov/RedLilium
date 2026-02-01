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
    vulkan_surface: Option<ash::vk::SurfaceKHR>,
    /// The Vulkan swapchain (only when using vulkan backend).
    #[cfg(feature = "vulkan-backend")]
    vulkan_swapchain: RwLock<Option<VulkanSwapchain>>,
}

/// Vulkan swapchain resources.
#[cfg(feature = "vulkan-backend")]
struct VulkanSwapchain {
    swapchain: ash::vk::SwapchainKHR,
    images: Vec<ash::vk::Image>,
    image_views: Vec<ash::vk::ImageView>,
    #[allow(dead_code)] // Reserved for future use
    format: ash::vk::Format,
    #[allow(dead_code)] // Reserved for future use
    extent: ash::vk::Extent2D,
    current_image_index: u32,
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
                let wgpu_config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: convert_texture_format_to_wgpu(config.format),
                    width: config.width,
                    height: config.height,
                    present_mode: convert_present_mode_to_wgpu(config.present_mode),
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                };
                wgpu_surface.configure(wgpu_backend.device(), &wgpu_config);
                log::info!("Configured wgpu surface");
            }
        }

        // Configure the Vulkan swapchain if available
        #[cfg(feature = "vulkan-backend")]
        if let Some(vulkan_surface) = self.vulkan_surface {
            use crate::backend::GpuBackend;
            if let GpuBackend::Vulkan(vulkan_backend) = self.instance.backend() {
                // Destroy old swapchain if it exists
                if let Ok(mut swapchain_guard) = self.vulkan_swapchain.write()
                    && let Some(old_swapchain) = swapchain_guard.take()
                {
                    // Wait for device to be idle before destroying
                    unsafe {
                        let _ = vulkan_backend.device().device_wait_idle();
                        for view in old_swapchain.image_views {
                            vulkan_backend.device().destroy_image_view(view, None);
                        }
                        vulkan_backend
                            .swapchain_loader()
                            .destroy_swapchain(old_swapchain.swapchain, None);
                    }
                }

                // Create new swapchain
                let new_swapchain =
                    create_vulkan_swapchain(vulkan_backend, vulkan_surface, config)?;

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
            let surface_texture = wgpu_surface.get_current_texture().map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to acquire surface texture: {e}"
                ))
            })?;
            let view = surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let surface_view = SurfaceTextureView {
                view: Arc::new(view),
            };
            (Some(surface_texture), Some(surface_view))
        } else {
            (None, None)
        };

        // Acquire the Vulkan swapchain image if using Vulkan backend
        #[cfg(feature = "vulkan-backend")]
        let (vulkan_view, vulkan_image_index) = {
            use crate::backend::GpuBackend;
            use crate::backend::vulkan::{VulkanImageView, VulkanSurfaceTextureView};

            if let GpuBackend::Vulkan(vulkan_backend) = self.instance.backend() {
                if let Ok(mut swapchain_guard) = self.vulkan_swapchain.write() {
                    if let Some(ref mut swapchain) = *swapchain_guard {
                        // Acquire next image
                        let (image_index, _suboptimal) = unsafe {
                            vulkan_backend.swapchain_loader().acquire_next_image(
                                swapchain.swapchain,
                                u64::MAX,
                                vk::Semaphore::null(),
                                vk::Fence::null(),
                            )
                        }
                        .map_err(|e| {
                            GraphicsError::ResourceCreationFailed(format!(
                                "Failed to acquire swapchain image: {:?}",
                                e
                            ))
                        })?;

                        swapchain.current_image_index = image_index;
                        let image = swapchain.images[image_index as usize];
                        let view = swapchain.image_views[image_index as usize];

                        let vulkan_view = VulkanSurfaceTextureView {
                            image,
                            view: Arc::new(VulkanImageView::new(
                                vulkan_backend.device().clone(),
                                view,
                            )),
                        };
                        (Some(vulkan_view), image_index)
                    } else {
                        (None, 0)
                    }
                } else {
                    (None, 0)
                }
            } else {
                (None, 0)
            }
        };

        log::trace!("Acquired surface texture, frame index: {}", frame_index);

        Ok(SurfaceTexture {
            device,
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
            wgpu_texture.present();
        }

        // Present the Vulkan swapchain image
        // Note: Vulkan presentation is typically done via queue_present after rendering
        // For now, we just log that we're ready to present
        #[cfg(feature = "vulkan-backend")]
        if self.vulkan_view.is_some() {
            log::trace!(
                "Vulkan surface texture ready for presentation, image index: {}",
                self.vulkan_image_index
            );
            // Actual presentation happens through the graphics queue
            // This would need the swapchain reference to call vkQueuePresentKHR
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

// ============================================================================
// wgpu conversion helpers
// ============================================================================

#[cfg(feature = "wgpu-backend")]
fn convert_texture_format_to_wgpu(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        TextureFormat::R8Snorm => wgpu::TextureFormat::R8Snorm,
        TextureFormat::R8Uint => wgpu::TextureFormat::R8Uint,
        TextureFormat::R8Sint => wgpu::TextureFormat::R8Sint,
        TextureFormat::R16Unorm => wgpu::TextureFormat::R16Unorm,
        TextureFormat::R16Float => wgpu::TextureFormat::R16Float,
        TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
        TextureFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        TextureFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
        TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        TextureFormat::Depth16Unorm => wgpu::TextureFormat::Depth16Unorm,
        TextureFormat::Depth24Plus => wgpu::TextureFormat::Depth24Plus,
        TextureFormat::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        TextureFormat::Depth32FloatStencil8 => wgpu::TextureFormat::Depth32FloatStencil8,
    }
}

#[cfg(feature = "wgpu-backend")]
fn convert_present_mode_to_wgpu(mode: PresentMode) -> wgpu::PresentMode {
    match mode {
        PresentMode::Immediate => wgpu::PresentMode::Immediate,
        PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        PresentMode::Fifo => wgpu::PresentMode::Fifo,
        PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
    }
}

// ============================================================================
// Vulkan swapchain helpers
// ============================================================================

#[cfg(feature = "vulkan-backend")]
fn create_vulkan_swapchain(
    vulkan_backend: &crate::backend::vulkan::VulkanBackend,
    surface: vk::SurfaceKHR,
    config: &SurfaceConfiguration,
) -> Result<VulkanSwapchain, GraphicsError> {
    // Get surface capabilities
    let capabilities = vulkan_backend.get_surface_capabilities(surface)?;

    // Choose format
    let formats = vulkan_backend.get_surface_formats(surface)?;
    let surface_format = formats
        .iter()
        .find(|f| f.format == convert_texture_format_to_vk(config.format))
        .cloned()
        .unwrap_or(formats[0]);

    // Choose present mode
    let present_modes = vulkan_backend.get_surface_present_modes(surface)?;
    let present_mode = convert_present_mode_to_vk(config.present_mode);
    let present_mode = if present_modes.contains(&present_mode) {
        present_mode
    } else {
        vk::PresentModeKHR::FIFO // Always available
    };

    // Choose extent
    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: config.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: config.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };

    // Choose image count (prefer triple buffering)
    let image_count = (capabilities.min_image_count + 1).min(if capabilities.max_image_count > 0 {
        capabilities.max_image_count
    } else {
        u32::MAX
    });

    // Create swapchain
    let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain = unsafe {
        vulkan_backend
            .swapchain_loader()
            .create_swapchain(&swapchain_create_info, None)
    }
    .map_err(|e| {
        GraphicsError::ResourceCreationFailed(format!("Failed to create swapchain: {:?}", e))
    })?;

    // Get swapchain images
    let images = unsafe {
        vulkan_backend
            .swapchain_loader()
            .get_swapchain_images(swapchain)
    }
    .map_err(|e| {
        GraphicsError::ResourceCreationFailed(format!("Failed to get swapchain images: {:?}", e))
    })?;

    // Create image views
    let image_views: Vec<vk::ImageView> = images
        .iter()
        .map(|&image| vulkan_backend.create_swapchain_image_view(image, surface_format.format))
        .collect::<Result<Vec<_>, _>>()?;

    log::info!(
        "Created Vulkan swapchain: {}x{} with {} images",
        extent.width,
        extent.height,
        images.len()
    );

    Ok(VulkanSwapchain {
        swapchain,
        images,
        image_views,
        format: surface_format.format,
        extent,
        current_image_index: 0,
    })
}

#[cfg(feature = "vulkan-backend")]
fn convert_texture_format_to_vk(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::R8Unorm => vk::Format::R8_UNORM,
        TextureFormat::R8Snorm => vk::Format::R8_SNORM,
        TextureFormat::R8Uint => vk::Format::R8_UINT,
        TextureFormat::R8Sint => vk::Format::R8_SINT,
        TextureFormat::R16Unorm => vk::Format::R16_UNORM,
        TextureFormat::R16Float => vk::Format::R16_SFLOAT,
        TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        TextureFormat::R32Float => vk::Format::R32_SFLOAT,
        TextureFormat::R32Uint => vk::Format::R32_UINT,
        TextureFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
        TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
        TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
        TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
        TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
        TextureFormat::Depth16Unorm => vk::Format::D16_UNORM,
        TextureFormat::Depth24Plus => vk::Format::D24_UNORM_S8_UINT,
        TextureFormat::Depth24PlusStencil8 => vk::Format::D24_UNORM_S8_UINT,
        TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
        TextureFormat::Depth32FloatStencil8 => vk::Format::D32_SFLOAT_S8_UINT,
    }
}

#[cfg(feature = "vulkan-backend")]
fn convert_present_mode_to_vk(mode: PresentMode) -> vk::PresentModeKHR {
    match mode {
        PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
    }
}

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
