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

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, RenderGraph};
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

/// Handle to a GPU buffer resource.
#[derive(Debug, Clone)]
pub enum GpuBuffer {
    /// Dummy backend (no GPU allocation)
    Dummy,
    /// wgpu backend buffer
    #[cfg(feature = "wgpu-backend")]
    Wgpu(Arc<wgpu::Buffer>),
}

/// Handle to a GPU texture resource.
#[derive(Debug, Clone)]
pub enum GpuTexture {
    /// Dummy backend (no GPU allocation)
    Dummy,
    /// wgpu backend texture
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        texture: Arc<wgpu::Texture>,
        view: Arc<wgpu::TextureView>,
    },
}

/// Handle to a GPU sampler resource.
#[derive(Debug, Clone)]
pub enum GpuSampler {
    /// Dummy backend (no GPU allocation)
    Dummy,
    /// wgpu backend sampler
    #[cfg(feature = "wgpu-backend")]
    Wgpu(Arc<wgpu::Sampler>),
}

/// Handle to a GPU fence for CPU-GPU synchronization.
#[derive(Debug)]
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
}

/// Handle to a GPU semaphore for GPU-GPU synchronization.
#[derive(Debug)]
pub enum GpuSemaphore {
    /// Dummy backend (no GPU semaphore)
    Dummy,
    /// wgpu backend (semaphores are implicit in wgpu)
    #[cfg(feature = "wgpu-backend")]
    Wgpu,
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
    // Try wgpu backend first if available
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
    #[cfg(feature = "wgpu-backend")]
    {
        return true;
    }
    #[cfg(not(feature = "wgpu-backend"))]
    {
        false
    }
}
