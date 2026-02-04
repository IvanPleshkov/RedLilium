//! GPU backend abstraction layer.
//!
//! This module provides an enum-based abstraction for GPU backends,
//! allowing the graphics crate to work with different GPU APIs without
//! dynamic dispatch.
//!
//! # Available Backends
//!
//! - `Dummy` (default): No-op backend for testing and development
//! - `Wgpu`: Cross-platform backend using wgpu (requires `wgpu-backend` feature)
//! - `Vulkan`: Native Vulkan backend using ash (requires `vulkan-backend` feature)
//!
//! # Architecture
//!
//! The [`GpuBackend`] enum provides:
//! - Instance and device creation
//! - Resource creation (buffers, textures, samplers)
//! - Command buffer recording and submission
//! - Synchronization primitives

#[cfg(feature = "wgpu-backend")]
pub mod wgpu_impl;

#[cfg(feature = "vulkan-backend")]
pub mod vulkan;

pub mod dummy;

use std::sync::Arc;

#[cfg(feature = "vulkan-backend")]
use ash::vk;
#[cfg(feature = "vulkan-backend")]
use gpu_allocator::vulkan::Allocation;
#[cfg(feature = "vulkan-backend")]
use parking_lot::Mutex;
#[cfg(feature = "vulkan-backend")]
use parking_lot::RwLock;
#[cfg(feature = "vulkan-backend")]
use vulkan::DeferredDestructor;

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, RenderGraph};
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

/// Handle to a GPU buffer resource.
#[allow(clippy::large_enum_variant)]
pub enum GpuBuffer {
    /// Dummy backend (no GPU allocation)
    Dummy,
    /// wgpu backend buffer
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu::Buffer),
    /// Vulkan backend buffer
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        buffer: vk::Buffer,
        allocation: Mutex<Option<Allocation>>,
        size: u64,
        /// Deferred destructor for safe cleanup.
        deferred: Arc<DeferredDestructor>,
    },
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuBuffer::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(buffer) => f.debug_tuple("GpuBuffer::Wgpu").field(buffer).finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { buffer, size, .. } => f
                .debug_struct("GpuBuffer::Vulkan")
                .field("buffer", buffer)
                .field("size", size)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to a GPU texture resource.
#[allow(clippy::large_enum_variant)]
pub enum GpuTexture {
    /// Dummy backend (no GPU allocation)
    Dummy,
    /// wgpu backend texture
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
    /// Vulkan backend texture
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        image: vk::Image,
        view: vk::ImageView,
        allocation: Mutex<Option<Allocation>>,
        format: vk::Format,
        extent: vk::Extent3D,
        /// Deferred destructor for safe cleanup.
        deferred: Arc<DeferredDestructor>,
    },
}

impl std::fmt::Debug for GpuTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuTexture::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { texture, view } => f
                .debug_struct("GpuTexture::Wgpu")
                .field("texture", texture)
                .field("view", view)
                .finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan {
                image,
                view,
                format,
                extent,
                ..
            } => f
                .debug_struct("GpuTexture::Vulkan")
                .field("image", image)
                .field("view", view)
                .field("format", format)
                .field("extent", extent)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to a GPU sampler resource.
#[allow(clippy::large_enum_variant)]
pub enum GpuSampler {
    /// Dummy backend (no GPU allocation)
    Dummy,
    /// wgpu backend sampler
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu::Sampler),
    /// Vulkan backend sampler
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        sampler: vk::Sampler,
        /// Deferred destructor for safe cleanup.
        deferred: Arc<DeferredDestructor>,
    },
}

impl std::fmt::Debug for GpuSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuSampler::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(sampler) => f.debug_tuple("GpuSampler::Wgpu").field(sampler).finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { sampler, .. } => f
                .debug_struct("GpuSampler::Vulkan")
                .field("sampler", sampler)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to a GPU fence for CPU-GPU synchronization.
///
/// # Backend Differences
///
/// Fences are implemented differently across backends due to API constraints:
///
/// ## Vulkan (`vk::Fence`)
/// True GPU fence with binary signaled/unsignaled state. The GPU signals the fence
/// when command buffer execution completes. CPU can wait or poll the fence status.
/// This provides precise synchronization and supports multiple frames in flight.
///
/// ## wgpu (`SubmissionIndex`)
/// wgpu abstracts over multiple backends (Vulkan, Metal, DX12, WebGPU) and doesn't
/// expose native fence handles to maintain portability. Instead, it uses submission
/// indices that can be polled via `device.poll()`. Key differences:
/// - No true "unsignaled" state - fence tracks submissions, not binary state
/// - Polling checks if work is complete, not a specific fence state
/// - `execute_graph` with fence provided returns immediately (async)
/// - `execute_graph` without fence blocks until completion (sync, backwards compatible)
///
/// **Why not expose GPU fences in wgpu?** Each backend (Metal, DX12, WebGPU) has
/// different synchronization primitives. wgpu's submission index abstraction works
/// across all backends at the cost of less precise control.
///
/// ## Dummy (`AtomicBool`)
/// Simple CPU-side flag for testing without GPU hardware.
#[allow(clippy::large_enum_variant)]
pub enum GpuFence {
    /// Dummy backend - CPU-side atomic boolean for testing.
    Dummy {
        signaled: std::sync::atomic::AtomicBool,
    },
    /// wgpu backend - tracks submission index for polling.
    /// Note: wgpu fences track submissions rather than binary state.
    /// A fence with `submission_index: None` is considered "signaled" (no pending work).
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        device: Arc<wgpu::Device>,
        submission_index: std::sync::Mutex<Option<wgpu::SubmissionIndex>>,
    },
    /// Vulkan backend - true GPU fence via `vkFence`.
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        fence: vk::Fence,
        /// Deferred destructor for safe cleanup.
        deferred: Arc<DeferredDestructor>,
    },
}

impl std::fmt::Debug for GpuFence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy { signaled } => f
                .debug_struct("GpuFence::Dummy")
                .field("signaled", signaled)
                .finish(),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu {
                submission_index, ..
            } => f
                .debug_struct("GpuFence::Wgpu")
                .field("submission_index", submission_index)
                .finish_non_exhaustive(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { fence, .. } => f
                .debug_struct("GpuFence::Vulkan")
                .field("fence", fence)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to a GPU semaphore for GPU-GPU synchronization.
#[allow(clippy::large_enum_variant)]
pub enum GpuSemaphore {
    /// Dummy backend (no GPU semaphore)
    Dummy,
    /// wgpu backend (semaphores are implicit in wgpu)
    #[cfg(feature = "wgpu-backend")]
    Wgpu,
    /// Vulkan backend semaphore
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        semaphore: vk::Semaphore,
        /// Deferred destructor for safe cleanup.
        deferred: Arc<DeferredDestructor>,
    },
}

impl std::fmt::Debug for GpuSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuSemaphore::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu => write!(f, "GpuSemaphore::Wgpu"),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { semaphore, .. } => f
                .debug_struct("GpuSemaphore::Vulkan")
                .field("semaphore", semaphore)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to an acquired surface texture for presentation.
///
/// This encapsulates all backend-specific state needed to render to a surface
/// and present the result.
#[allow(clippy::large_enum_variant)]
pub enum GpuSurfaceTexture {
    /// Dummy backend (no GPU texture)
    Dummy,
    /// wgpu backend surface texture
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        /// The raw surface texture (needed for presentation).
        texture: wgpu::SurfaceTexture,
        /// The texture view for rendering.
        view: wgpu_impl::SurfaceTextureView,
    },
    /// Vulkan backend surface texture
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        /// The texture view for rendering.
        view: vulkan::VulkanSurfaceTextureView,
        /// The swapchain image index.
        image_index: u32,
        /// The frame-in-flight index.
        #[allow(dead_code)] // Reserved for future use
        frame_index: usize,
        /// The swapchain handle.
        swapchain: vk::SwapchainKHR,
        /// The image available semaphore.
        image_available_semaphore: vk::Semaphore,
        /// The render finished semaphore.
        render_finished_semaphore: vk::Semaphore,
        /// The in-flight fence.
        in_flight_fence: vk::Fence,
        /// The command buffer for presentation.
        present_command_buffer: vk::CommandBuffer,
    },
}

impl std::fmt::Debug for GpuSurfaceTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuSurfaceTexture::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { .. } => f.debug_struct("GpuSurfaceTexture::Wgpu").finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { image_index, .. } => f
                .debug_struct("GpuSurfaceTexture::Vulkan")
                .field("image_index", image_index)
                .finish_non_exhaustive(),
        }
    }
}

impl GpuSurfaceTexture {
    /// Get the wgpu texture view for rendering (only available with wgpu backend).
    #[cfg(feature = "wgpu-backend")]
    pub fn wgpu_view(&self) -> Option<wgpu_impl::SurfaceTextureView> {
        match self {
            Self::Wgpu { view, .. } => Some(view.clone()),
            _ => None,
        }
    }

    /// Get the wgpu texture view for rendering (stub for non-wgpu builds).
    #[cfg(not(feature = "wgpu-backend"))]
    pub fn wgpu_view(&self) -> Option<()> {
        None
    }

    /// Get the Vulkan texture view for rendering (only available with vulkan backend).
    #[cfg(feature = "vulkan-backend")]
    pub fn vulkan_view(&self) -> Option<vulkan::VulkanSurfaceTextureView> {
        match self {
            Self::Vulkan { view, .. } => Some(view.clone()),
            _ => None,
        }
    }

    /// Get the Vulkan texture view for rendering (stub for non-vulkan builds).
    #[cfg(not(feature = "vulkan-backend"))]
    pub fn vulkan_view(&self) -> Option<()> {
        None
    }

    /// Present the surface texture.
    ///
    /// Takes the backend for Vulkan presentation.
    pub fn present(self, backend: &GpuBackend, frame_index: u64) {
        match self {
            Self::Dummy => {
                log::trace!("Presenting dummy frame {}", frame_index);
            }
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { texture, .. } => {
                wgpu_impl::swapchain::present_surface_texture(texture);
                log::trace!("Presented wgpu frame {}", frame_index);
            }
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan {
                view,
                image_index,
                swapchain,
                image_available_semaphore,
                render_finished_semaphore,
                in_flight_fence,
                present_command_buffer,
                ..
            } => {
                if let GpuBackend::Vulkan(vulkan_backend) = backend {
                    if let Err(e) = vulkan::swapchain::present_vulkan_frame(
                        vulkan_backend,
                        &view,
                        swapchain,
                        image_index,
                        image_available_semaphore,
                        render_finished_semaphore,
                        in_flight_fence,
                        present_command_buffer,
                        frame_index,
                    ) {
                        log::error!("Failed to present Vulkan frame: {}", e);
                    }
                } else {
                    log::error!("Vulkan surface texture requires Vulkan backend for presentation");
                }
            }
        }
    }
}

/// Represents a GPU surface for presentation.
///
/// This encapsulates all backend-specific state needed to create and manage
/// a presentation surface (swapchain).
#[allow(clippy::large_enum_variant)]
pub enum GpuSurface {
    /// Dummy backend (no GPU surface)
    Dummy,
    /// wgpu backend surface
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        /// The wgpu surface.
        surface: wgpu::Surface<'static>,
    },
    /// Vulkan backend surface
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        /// The Vulkan surface handle.
        surface: vk::SurfaceKHR,
        /// The Vulkan swapchain (created on configure).
        swapchain: RwLock<Option<vulkan::swapchain::VulkanSwapchain>>,
    },
}

impl std::fmt::Debug for GpuSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuSurface::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { .. } => f.debug_struct("GpuSurface::Wgpu").finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { surface, .. } => f
                .debug_struct("GpuSurface::Vulkan")
                .field("surface", surface)
                .finish_non_exhaustive(),
        }
    }
}

// GpuSurface is Send + Sync because all variant backends are Send + Sync
unsafe impl Send for GpuSurface {}
unsafe impl Sync for GpuSurface {}

impl GpuSurface {
    /// Configure the surface for rendering.
    ///
    /// This must be called before acquiring textures. It should also be called
    /// when the window is resized.
    pub fn configure(
        &self,
        backend: &GpuBackend,
        config: &crate::swapchain::SurfaceConfiguration,
    ) -> Result<(), GraphicsError> {
        match (self, backend) {
            (Self::Dummy, _) => {
                log::info!("Configured dummy surface");
                Ok(())
            }

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu { surface }, GpuBackend::Wgpu(wgpu_backend)) => {
                wgpu_impl::swapchain::configure_surface(surface, wgpu_backend, config);
                Ok(())
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan { surface, swapchain }, GpuBackend::Vulkan(vulkan_backend)) => {
                use vulkan::swapchain::VulkanSwapchain;

                // Destroy old swapchain if it exists
                if let Some(ref mut old_swapchain) = *swapchain.write() {
                    old_swapchain.destroy();
                }

                // Create new swapchain
                let new_swapchain = VulkanSwapchain::new(vulkan_backend, *surface, config)?;
                *swapchain.write() = Some(new_swapchain);
                log::info!("Configured Vulkan swapchain");
                Ok(())
            }

            #[allow(unreachable_patterns)]
            _ => Err(GraphicsError::Internal(
                "Surface and backend type mismatch".to_string(),
            )),
        }
    }

    /// Acquire the next texture from the swapchain.
    ///
    /// Returns a backend-specific surface texture for rendering.
    pub fn acquire_texture(
        &self,
        backend: &GpuBackend,
    ) -> Result<GpuSurfaceTexture, GraphicsError> {
        match (self, backend) {
            (Self::Dummy, _) => Ok(GpuSurfaceTexture::Dummy),

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu { surface }, GpuBackend::Wgpu(_)) => {
                let result = wgpu_impl::swapchain::acquire_surface_texture(surface)?;
                Ok(GpuSurfaceTexture::Wgpu {
                    texture: result.texture,
                    view: result.view,
                })
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan { swapchain, .. }, GpuBackend::Vulkan(vulkan_backend)) => {
                if let Some(ref mut swapchain) = *swapchain.write() {
                    let result = swapchain.acquire_next_image(vulkan_backend)?;
                    Ok(GpuSurfaceTexture::Vulkan {
                        view: result.view,
                        image_index: result.image_index,
                        frame_index: result.frame_index,
                        swapchain: result.swapchain,
                        image_available_semaphore: result.image_available_semaphore,
                        render_finished_semaphore: result.render_finished_semaphore,
                        in_flight_fence: result.in_flight_fence,
                        present_command_buffer: result.present_command_buffer,
                    })
                } else {
                    Err(GraphicsError::Internal(
                        "Vulkan swapchain not configured".to_string(),
                    ))
                }
            }

            #[allow(unreachable_patterns)]
            _ => Err(GraphicsError::Internal(
                "Surface and backend type mismatch".to_string(),
            )),
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuSurface {
    fn drop(&mut self) {
        if let GpuSurface::Vulkan { swapchain, .. } = self {
            // Destroy the swapchain before the surface is dropped.
            // The VulkanSwapchain stores its own device handles for cleanup.
            if let Some(ref mut sc) = *swapchain.write() {
                sc.destroy();
            }
        }
    }
}

// ============================================================================
// Vulkan Resource Cleanup (Drop implementations)
// ============================================================================
//
// These Drop implementations use deferred destruction to ensure GPU resources
// are not destroyed while the GPU may still be using them. Resources are queued
// for destruction and only actually destroyed after enough frames have passed.

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let GpuBuffer::Vulkan {
            device,
            buffer,
            allocation,
            deferred,
            ..
        } = self
        {
            // Take the allocation out and queue for deferred destruction
            let alloc = allocation.lock().take();
            deferred.queue(vulkan::DeferredResource::Buffer {
                device: device.clone(),
                buffer: *buffer,
                allocation: alloc,
            });
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuTexture {
    fn drop(&mut self) {
        if let GpuTexture::Vulkan {
            device,
            image,
            view,
            allocation,
            deferred,
            ..
        } = self
        {
            // Take the allocation out and queue for deferred destruction
            let alloc = allocation.lock().take();
            deferred.queue(vulkan::DeferredResource::Texture {
                device: device.clone(),
                image: *image,
                view: *view,
                allocation: alloc,
            });
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuSampler {
    fn drop(&mut self) {
        if let GpuSampler::Vulkan {
            device,
            sampler,
            deferred,
        } = self
        {
            deferred.queue(vulkan::DeferredResource::Sampler {
                device: device.clone(),
                sampler: *sampler,
            });
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuFence {
    fn drop(&mut self) {
        if let GpuFence::Vulkan {
            device,
            fence,
            deferred,
        } = self
        {
            deferred.queue(vulkan::DeferredResource::Fence {
                device: device.clone(),
                fence: *fence,
            });
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuSemaphore {
    fn drop(&mut self) {
        if let GpuSemaphore::Vulkan {
            device,
            semaphore,
            deferred,
        } = self
        {
            deferred.queue(vulkan::DeferredResource::Semaphore {
                device: device.clone(),
                semaphore: *semaphore,
            });
        }
    }
}

// ============================================================================
// GPU Backend Enum
// ============================================================================

/// GPU backend enum for abstracting different GPU APIs.
///
/// Unlike a trait-based approach, this enum allows for static dispatch
/// and avoids the overhead of dynamic dispatch.
#[allow(clippy::large_enum_variant)]
pub enum GpuBackend {
    /// Dummy backend for testing and development.
    Dummy(dummy::DummyBackend),
    /// wgpu backend for cross-platform GPU access.
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu_impl::WgpuBackend),
    /// Native Vulkan backend using ash.
    #[cfg(feature = "vulkan-backend")]
    Vulkan(vulkan::VulkanBackend),
}

impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy(backend) => f.debug_tuple("GpuBackend::Dummy").field(backend).finish(),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => f.debug_tuple("GpuBackend::Wgpu").field(backend).finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => f.debug_tuple("GpuBackend::Vulkan").field(backend).finish(),
        }
    }
}

// GpuBackend is Send + Sync because all variant backends are Send + Sync
unsafe impl Send for GpuBackend {}
unsafe impl Sync for GpuBackend {}

impl GpuBackend {
    /// Get the backend name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dummy(backend) => backend.name(),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.name(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.name(),
        }
    }

    /// Create a buffer resource.
    pub fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<GpuBuffer, GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.create_buffer(descriptor),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_buffer(descriptor),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_buffer(descriptor),
        }
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.create_texture(descriptor),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_texture(descriptor),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_texture(descriptor),
        }
    }

    /// Create a sampler resource.
    pub fn create_sampler(
        &self,
        descriptor: &SamplerDescriptor,
    ) -> Result<GpuSampler, GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.create_sampler(descriptor),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_sampler(descriptor),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_sampler(descriptor),
        }
    }

    /// Create a fence for CPU-GPU synchronization.
    pub fn create_fence(&self, signaled: bool) -> GpuFence {
        match self {
            Self::Dummy(backend) => backend.create_fence(signaled),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_fence(signaled),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_fence(signaled),
        }
    }

    /// Wait for a fence to be signaled.
    pub fn wait_fence(&self, fence: &GpuFence) {
        match self {
            Self::Dummy(backend) => backend.wait_fence(fence),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.wait_fence(fence),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.wait_fence(fence),
        }
    }

    /// Wait for a fence to be signaled with a timeout.
    ///
    /// Returns `true` if the fence was signaled, `false` if the timeout elapsed.
    pub fn wait_fence_timeout(&self, fence: &GpuFence, timeout: std::time::Duration) -> bool {
        match self {
            Self::Dummy(backend) => backend.wait_fence_timeout(fence, timeout),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.wait_fence_timeout(fence, timeout),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.wait_fence_timeout(fence, timeout),
        }
    }

    /// Check if a fence is signaled (non-blocking).
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        match self {
            Self::Dummy(backend) => backend.is_fence_signaled(fence),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.is_fence_signaled(fence),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.is_fence_signaled(fence),
        }
    }

    /// Signal a fence (for testing/dummy backend).
    pub fn signal_fence(&self, fence: &GpuFence) {
        match self {
            Self::Dummy(backend) => backend.signal_fence(fence),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.signal_fence(fence),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.signal_fence(fence),
        }
    }

    /// Execute a compiled render graph.
    ///
    /// This records commands from the graph into a command buffer and submits it.
    pub fn execute_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.execute_graph(graph, compiled, signal_fence),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.execute_graph(graph, compiled, signal_fence),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.execute_graph(graph, compiled, signal_fence),
        }
    }

    /// Write data to a buffer.
    pub fn write_buffer(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.write_buffer(buffer, offset, data),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.write_buffer(buffer, offset, data),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.write_buffer(buffer, offset, data),
        }
    }

    /// Read data from a buffer.
    ///
    /// This is a blocking operation that waits for the GPU to finish.
    pub fn read_buffer(&self, buffer: &GpuBuffer, offset: u64, size: u64) -> Vec<u8> {
        match self {
            Self::Dummy(backend) => backend.read_buffer(buffer, offset, size),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.read_buffer(buffer, offset, size),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.read_buffer(buffer, offset, size),
        }
    }

    /// Write data to a texture.
    pub fn write_texture(
        &self,
        texture: &GpuTexture,
        data: &[u8],
        descriptor: &TextureDescriptor,
    ) -> Result<(), GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.write_texture(texture, data, descriptor),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.write_texture(texture, data, descriptor),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.write_texture(texture, data, descriptor),
        }
    }

    /// Create a surface from a window.
    ///
    /// # Safety
    ///
    /// The window handle must remain valid for the lifetime of the surface.
    pub fn create_surface<W>(&self, window: &W) -> Result<GpuSurface, GraphicsError>
    where
        W: raw_window_handle::HasWindowHandle + raw_window_handle::HasDisplayHandle + Sync,
    {
        match self {
            Self::Dummy(_) => {
                log::info!("Created dummy surface");
                Ok(GpuSurface::Dummy)
            }

            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(wgpu_backend) => {
                // Create wgpu surface from window
                // SAFETY: The caller guarantees the window handle remains valid for the
                // lifetime of the surface. We transmute to 'static to satisfy wgpu's
                // Surface<'static> requirement, but the Surface is dropped before the
                // window in practice.
                let surface: wgpu::Surface<'static> = unsafe {
                    std::mem::transmute(wgpu_backend.instance().create_surface(window).map_err(
                        |e| {
                            GraphicsError::ResourceCreationFailed(format!(
                                "Failed to create wgpu surface: {e}"
                            ))
                        },
                    )?)
                };
                log::info!("Created wgpu surface");
                Ok(GpuSurface::Wgpu { surface })
            }

            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(vulkan_backend) => {
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
                log::info!("Created Vulkan surface");
                Ok(GpuSurface::Vulkan {
                    surface,
                    swapchain: RwLock::new(None),
                })
            }
        }
    }

    /// Ensure the backend is compatible with the given surface.
    ///
    /// For wgpu backend, this may re-request an adapter if the current one is not
    /// compatible with the surface. For Vulkan backend, this verifies the physical
    /// device supports the surface.
    ///
    /// Returns `Ok(true)` if compatible, `Err` if no compatible adapter could be found.
    pub fn ensure_compatible_with_surface(
        &mut self,
        surface: &GpuSurface,
    ) -> Result<bool, GraphicsError> {
        match (self, surface) {
            (Self::Dummy(_), _) => {
                // Dummy backend is always "compatible"
                Ok(true)
            }

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu(wgpu_backend), GpuSurface::Wgpu { surface }) => {
                wgpu_backend.ensure_compatible_with_surface(surface)
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan(vulkan_backend), GpuSurface::Vulkan { surface, .. }) => {
                if vulkan_backend.is_surface_supported(*surface) {
                    Ok(true)
                } else {
                    Err(GraphicsError::ResourceCreationFailed(
                        "Vulkan physical device does not support presentation to this surface"
                            .to_string(),
                    ))
                }
            }

            #[allow(unreachable_patterns)]
            _ => Err(GraphicsError::Internal(
                "Surface and backend type mismatch".to_string(),
            )),
        }
    }

    /// Check if the backend is compatible with the given surface without modifying it.
    pub fn is_compatible_with_surface(&self, surface: &GpuSurface) -> bool {
        match (self, surface) {
            (Self::Dummy(_), _) => true,

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu(wgpu_backend), GpuSurface::Wgpu { surface }) => {
                wgpu_backend.is_adapter_compatible_with_surface(surface)
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan(vulkan_backend), GpuSurface::Vulkan { surface, .. }) => {
                vulkan_backend.is_surface_supported(*surface)
            }

            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

/// Selects and creates the appropriate backend based on available features.
///
/// This uses default parameters (auto-select best backend).
pub fn create_backend() -> Result<GpuBackend, GraphicsError> {
    create_backend_with_params(&crate::instance::InstanceParameters::default())
}

/// Selects and creates the appropriate backend based on parameters.
pub fn create_backend_with_params(
    params: &crate::instance::InstanceParameters,
) -> Result<GpuBackend, GraphicsError> {
    use crate::instance::BackendType;

    match params.backend {
        BackendType::Auto => create_backend_auto(params),
        BackendType::Dummy => {
            log::info!("Using dummy backend (requested)");
            Ok(GpuBackend::Dummy(dummy::DummyBackend::new()))
        }
        BackendType::Wgpu => {
            #[cfg(feature = "wgpu-backend")]
            {
                let backend = wgpu_impl::WgpuBackend::with_params(params)?;
                log::info!("Using wgpu backend (requested)");
                Ok(GpuBackend::Wgpu(backend))
            }
            #[cfg(not(feature = "wgpu-backend"))]
            {
                Err(GraphicsError::ResourceCreationFailed(
                    "wgpu backend requested but wgpu-backend feature is not enabled".to_string(),
                ))
            }
        }
        BackendType::Vulkan => {
            #[cfg(feature = "vulkan-backend")]
            {
                let backend = vulkan::VulkanBackend::new()?;
                log::info!("Using Vulkan backend (requested)");
                Ok(GpuBackend::Vulkan(backend))
            }
            #[cfg(not(feature = "vulkan-backend"))]
            {
                Err(GraphicsError::ResourceCreationFailed(
                    "Vulkan backend requested but vulkan-backend feature is not enabled"
                        .to_string(),
                ))
            }
        }
    }
}

/// Auto-select the best available backend.
fn create_backend_auto(
    params: &crate::instance::InstanceParameters,
) -> Result<GpuBackend, GraphicsError> {
    // Try wgpu backend first if available (supports WGSL shaders and full draw commands)
    #[cfg(feature = "wgpu-backend")]
    {
        match wgpu_impl::WgpuBackend::with_params(params) {
            Ok(backend) => {
                log::info!("Using wgpu backend");
                return Ok(GpuBackend::Wgpu(backend));
            }
            Err(e) => {
                log::warn!("Failed to create wgpu backend: {}", e);
            }
        }
    }

    // Try Vulkan backend if wgpu unavailable (native Vulkan via ash)
    #[cfg(feature = "vulkan-backend")]
    {
        match vulkan::VulkanBackend::new() {
            Ok(backend) => {
                log::info!("Using Vulkan backend (ash)");
                return Ok(GpuBackend::Vulkan(backend));
            }
            Err(e) => {
                log::warn!("Failed to create Vulkan backend: {}", e);
            }
        }
    }

    // Fall back to dummy backend
    log::info!("Using dummy backend");
    Ok(GpuBackend::Dummy(dummy::DummyBackend::new()))
}

/// Check if a real GPU backend is available.
pub fn has_gpu_backend() -> bool {
    cfg!(any(feature = "vulkan-backend", feature = "wgpu-backend"))
}
