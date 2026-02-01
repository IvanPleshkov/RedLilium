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

#[cfg(feature = "wgpu-backend")]
use std::sync::Arc;

#[cfg(feature = "vulkan-backend")]
use ash::vk;
#[cfg(feature = "vulkan-backend")]
use gpu_allocator::vulkan::Allocation;
#[cfg(feature = "vulkan-backend")]
use parking_lot::Mutex;

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
    Wgpu(Arc<wgpu::Buffer>),
    /// Vulkan backend buffer
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        buffer: vk::Buffer,
        allocation: Mutex<Option<Allocation>>,
        size: u64,
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

impl Clone for GpuBuffer {
    fn clone(&self) -> Self {
        match self {
            Self::Dummy => Self::Dummy,
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(buffer) => Self::Wgpu(buffer.clone()),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { .. } => {
                panic!("Vulkan buffers cannot be cloned - use Arc<Buffer> instead")
            }
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
        texture: Arc<wgpu::Texture>,
        view: Arc<wgpu::TextureView>,
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

impl Clone for GpuTexture {
    fn clone(&self) -> Self {
        match self {
            Self::Dummy => Self::Dummy,
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { texture, view } => Self::Wgpu {
                texture: texture.clone(),
                view: view.clone(),
            },
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { .. } => {
                panic!("Vulkan textures cannot be cloned - use Arc<Texture> instead")
            }
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
    Wgpu(Arc<wgpu::Sampler>),
    /// Vulkan backend sampler
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        sampler: vk::Sampler,
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

impl Clone for GpuSampler {
    fn clone(&self) -> Self {
        match self {
            Self::Dummy => Self::Dummy,
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(sampler) => Self::Wgpu(sampler.clone()),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { .. } => {
                panic!("Vulkan samplers cannot be cloned - use Arc<Sampler> instead")
            }
        }
    }
}

/// Handle to a GPU fence for CPU-GPU synchronization.
#[allow(clippy::large_enum_variant)]
pub enum GpuFence {
    /// Dummy backend (no GPU fence)
    Dummy {
        signaled: std::sync::atomic::AtomicBool,
    },
    /// wgpu backend fence (submission index for polling)
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        device: Arc<wgpu::Device>,
        submission_index: std::sync::Mutex<Option<wgpu::SubmissionIndex>>,
    },
    /// Vulkan backend fence
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        fence: vk::Fence,
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

// ============================================================================
// Vulkan Resource Cleanup (Drop implementations)
// ============================================================================

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let GpuBuffer::Vulkan {
            device,
            buffer,
            allocation,
            ..
        } = self
        {
            // Take the allocation out - it will be freed when the VulkanBackend drops the allocator
            let _ = allocation.lock().take();
            unsafe {
                device.destroy_buffer(*buffer, None);
            }
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
            ..
        } = self
        {
            // Take the allocation out
            let _ = allocation.lock().take();
            unsafe {
                device.destroy_image_view(*view, None);
                device.destroy_image(*image, None);
            }
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuSampler {
    fn drop(&mut self) {
        if let GpuSampler::Vulkan { device, sampler } = self {
            unsafe {
                device.destroy_sampler(*sampler, None);
            }
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuFence {
    fn drop(&mut self) {
        if let GpuFence::Vulkan { device, fence } = self {
            unsafe {
                device.destroy_fence(*fence, None);
            }
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuSemaphore {
    fn drop(&mut self) {
        if let GpuSemaphore::Vulkan { device, semaphore } = self {
            unsafe {
                device.destroy_semaphore(*semaphore, None);
            }
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
    pub fn write_buffer(&self, buffer: &GpuBuffer, offset: u64, data: &[u8]) {
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
}

/// Selects and creates the appropriate backend based on available features.
pub fn create_backend() -> Result<GpuBackend, GraphicsError> {
    // Try wgpu backend first if available (supports WGSL shaders and full draw commands)
    #[cfg(feature = "wgpu-backend")]
    {
        match wgpu_impl::WgpuBackend::new() {
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
    // Note: Vulkan backend currently doesn't support draw commands - only transfer operations
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
