//! GPU backend abstraction layer.
//!
//! This module provides a trait-based abstraction for GPU backends,
//! allowing the graphics crate to work with different GPU APIs.
//!
//! # Available Backends
//!
//! - `dummy` (default): No-op backend for testing and development
//! - `wgpu-backend`: Cross-platform backend using wgpu
//! - `vulkan-backend`: Native Vulkan backend using ash
//!
//! # Architecture
//!
//! Each backend implements the [`GpuBackend`] trait, which provides:
//! - Instance and device creation
//! - Resource creation (buffers, textures, samplers)
//! - Command buffer recording and submission
//! - Synchronization primitives

#[cfg(feature = "wgpu-backend")]
pub mod wgpu_backend;

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

/// GPU backend trait for abstracting different GPU APIs.
pub trait GpuBackend: Send + Sync + 'static {
    /// Get the backend name.
    fn name(&self) -> &'static str;

    /// Create a buffer resource.
    fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<GpuBuffer, GraphicsError>;

    /// Create a texture resource.
    fn create_texture(&self, descriptor: &TextureDescriptor) -> Result<GpuTexture, GraphicsError>;

    /// Create a sampler resource.
    fn create_sampler(&self, descriptor: &SamplerDescriptor) -> Result<GpuSampler, GraphicsError>;

    /// Create a fence for CPU-GPU synchronization.
    fn create_fence(&self, signaled: bool) -> GpuFence;

    /// Wait for a fence to be signaled.
    fn wait_fence(&self, fence: &GpuFence);

    /// Check if a fence is signaled (non-blocking).
    fn is_fence_signaled(&self, fence: &GpuFence) -> bool;

    /// Signal a fence (for testing/dummy backend).
    fn signal_fence(&self, fence: &GpuFence);

    /// Execute a compiled render graph.
    ///
    /// This records commands from the graph into a command buffer and submits it.
    fn execute_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError>;

    /// Write data to a buffer.
    fn write_buffer(&self, buffer: &GpuBuffer, offset: u64, data: &[u8]);

    /// Read data from a buffer.
    ///
    /// This is a blocking operation that waits for the GPU to finish.
    fn read_buffer(&self, buffer: &GpuBuffer, offset: u64, size: u64) -> Vec<u8>;
}

/// Selects and creates the appropriate backend based on available features.
pub fn create_backend() -> Result<Arc<dyn GpuBackend>, GraphicsError> {
    // Try Vulkan backend first if available (native Vulkan via ash)
    #[cfg(feature = "vulkan-backend")]
    {
        match vulkan::VulkanBackend::new() {
            Ok(backend) => {
                log::info!("Using Vulkan backend (ash)");
                return Ok(Arc::new(backend));
            }
            Err(e) => {
                log::warn!("Failed to create Vulkan backend: {}", e);
            }
        }
    }

    // Try wgpu backend if available
    #[cfg(feature = "wgpu-backend")]
    {
        match wgpu_backend::WgpuBackend::new() {
            Ok(backend) => {
                log::info!("Using wgpu backend");
                return Ok(Arc::new(backend));
            }
            Err(e) => {
                log::warn!("Failed to create wgpu backend: {}", e);
            }
        }
    }

    // Fall back to dummy backend
    log::info!("Using dummy backend");
    Ok(Arc::new(dummy::DummyBackend::new()))
}

/// Check if a real GPU backend is available.
pub fn has_gpu_backend() -> bool {
    cfg!(any(feature = "vulkan-backend", feature = "wgpu-backend"))
}
