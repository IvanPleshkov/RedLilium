//! Dummy GPU backend for testing and development.
//!
//! This backend doesn't perform actual GPU operations but provides
//! a valid implementation for testing the graphics API without
//! requiring GPU hardware.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, RenderGraph};
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

use super::{GpuBuffer, GpuFence, GpuSampler, GpuTexture};

/// Dummy GPU backend.
#[derive(Debug)]
pub struct DummyBackend;

impl DummyBackend {
    /// Create a new dummy backend.
    pub fn new() -> Self {
        Self
    }

    /// Get the backend name.
    pub fn name(&self) -> &'static str {
        "Dummy Backend"
    }

    /// Create a buffer resource.
    pub fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<GpuBuffer, GraphicsError> {
        log::trace!(
            "DummyBackend: creating buffer {:?} (size: {})",
            descriptor.label,
            descriptor.size
        );
        Ok(GpuBuffer::Dummy)
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        log::trace!(
            "DummyBackend: creating texture {:?} ({}x{}x{})",
            descriptor.label,
            descriptor.size.width,
            descriptor.size.height,
            descriptor.size.depth
        );
        Ok(GpuTexture::Dummy)
    }

    /// Create a sampler resource.
    pub fn create_sampler(
        &self,
        descriptor: &SamplerDescriptor,
    ) -> Result<GpuSampler, GraphicsError> {
        log::trace!("DummyBackend: creating sampler {:?}", descriptor.label);
        Ok(GpuSampler::Dummy)
    }

    /// Create a fence for CPU-GPU synchronization.
    pub fn create_fence(&self, signaled: bool) -> GpuFence {
        GpuFence::Dummy {
            signaled: AtomicBool::new(signaled),
        }
    }

    /// Wait for a fence to be signaled.
    pub fn wait_fence(&self, fence: &GpuFence) {
        match fence {
            GpuFence::Dummy { signaled } => {
                // In dummy mode, just spin until signaled (or assume signaled)
                while !signaled.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }
            #[cfg(feature = "wgpu-backend")]
            GpuFence::Wgpu { .. } => {}
            #[cfg(feature = "vulkan-backend")]
            GpuFence::Vulkan { .. } => {}
        }
    }

    /// Wait for a fence to be signaled with a timeout.
    ///
    /// Returns `true` if the fence was signaled, `false` if the timeout elapsed.
    pub fn wait_fence_timeout(&self, fence: &GpuFence, timeout: std::time::Duration) -> bool {
        match fence {
            GpuFence::Dummy { signaled } => {
                let start = std::time::Instant::now();
                while !signaled.load(Ordering::Acquire) {
                    if start.elapsed() >= timeout {
                        return false;
                    }
                    std::thread::yield_now();
                }
                true
            }
            #[cfg(feature = "wgpu-backend")]
            GpuFence::Wgpu { .. } => false,
            #[cfg(feature = "vulkan-backend")]
            GpuFence::Vulkan { .. } => false,
        }
    }

    /// Check if a fence is signaled (non-blocking).
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        match fence {
            GpuFence::Dummy { signaled } => signaled.load(Ordering::Acquire),
            #[cfg(feature = "wgpu-backend")]
            GpuFence::Wgpu { .. } => false,
            #[cfg(feature = "vulkan-backend")]
            GpuFence::Vulkan { .. } => false,
        }
    }

    /// Signal a fence (for testing/simulation).
    pub fn signal_fence(&self, fence: &GpuFence) {
        match fence {
            GpuFence::Dummy { signaled } => {
                signaled.store(true, Ordering::Release);
            }
            #[cfg(feature = "wgpu-backend")]
            GpuFence::Wgpu { .. } => {}
            #[cfg(feature = "vulkan-backend")]
            GpuFence::Vulkan { .. } => {}
        }
    }

    /// Execute a compiled render graph.
    pub fn execute_graph(
        &self,
        _graph: &RenderGraph,
        compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError> {
        log::trace!(
            "DummyBackend: executing graph with {} passes",
            compiled.pass_order().len()
        );

        // Log each pass for debugging
        for (i, _handle) in compiled.pass_order().iter().enumerate() {
            log::trace!("DummyBackend: executing pass {}", i);
        }

        // Signal the fence immediately since we don't do real GPU work
        if let Some(fence) = signal_fence {
            self.signal_fence(fence);
        }

        Ok(())
    }

    /// Write data to a buffer.
    pub fn write_buffer(
        &self,
        _buffer: &GpuBuffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), crate::error::GraphicsError> {
        log::trace!(
            "DummyBackend: write_buffer offset={} len={}",
            offset,
            data.len()
        );
        Ok(())
    }

    /// Read data from a buffer.
    pub fn read_buffer(&self, _buffer: &GpuBuffer, offset: u64, size: u64) -> Vec<u8> {
        log::trace!("DummyBackend: read_buffer offset={} size={}", offset, size);
        // Return zeroed data
        vec![0u8; size as usize]
    }

    /// Write data to a texture.
    pub fn write_texture(
        &self,
        _texture: &GpuTexture,
        data: &[u8],
        descriptor: &TextureDescriptor,
    ) -> Result<(), crate::error::GraphicsError> {
        log::trace!(
            "DummyBackend: write_texture {:?} ({}x{}) len={}",
            descriptor.label,
            descriptor.size.width,
            descriptor.size.height,
            data.len()
        );
        Ok(())
    }
}

impl Default for DummyBackend {
    fn default() -> Self {
        Self::new()
    }
}
