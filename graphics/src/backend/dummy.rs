//! Dummy GPU backend for testing and development.
//!
//! This backend doesn't perform actual GPU operations but provides
//! a valid implementation of the [`GpuBackend`] trait for testing
//! the graphics API without requiring GPU hardware.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, RenderGraph};
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

use super::{GpuBackend, GpuBuffer, GpuFence, GpuSampler, GpuTexture};

/// Dummy GPU backend.
#[derive(Debug)]
pub struct DummyBackend;

impl DummyBackend {
    /// Create a new dummy backend.
    pub fn new() -> Self {
        Self
    }
}

impl Default for DummyBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for DummyBackend {
    fn name(&self) -> &'static str {
        "Dummy Backend"
    }

    fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<GpuBuffer, GraphicsError> {
        log::trace!(
            "DummyBackend: creating buffer {:?} (size: {})",
            descriptor.label,
            descriptor.size
        );
        Ok(GpuBuffer::Dummy)
    }

    fn create_texture(&self, descriptor: &TextureDescriptor) -> Result<GpuTexture, GraphicsError> {
        log::trace!(
            "DummyBackend: creating texture {:?} ({}x{}x{})",
            descriptor.label,
            descriptor.size.width,
            descriptor.size.height,
            descriptor.size.depth
        );
        Ok(GpuTexture::Dummy)
    }

    fn create_sampler(&self, descriptor: &SamplerDescriptor) -> Result<GpuSampler, GraphicsError> {
        log::trace!("DummyBackend: creating sampler {:?}", descriptor.label);
        Ok(GpuSampler::Dummy)
    }

    fn create_fence(&self, signaled: bool) -> GpuFence {
        GpuFence::Dummy {
            signaled: AtomicBool::new(signaled),
        }
    }

    fn wait_fence(&self, fence: &GpuFence) {
        match fence {
            GpuFence::Dummy { signaled } => {
                // In dummy mode, just spin until signaled (or assume signaled)
                while !signaled.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }
            #[cfg(feature = "wgpu-backend")]
            _ => {}
        }
    }

    fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        match fence {
            GpuFence::Dummy { signaled } => signaled.load(Ordering::Acquire),
            #[cfg(feature = "wgpu-backend")]
            _ => false,
        }
    }

    fn signal_fence(&self, fence: &GpuFence) {
        match fence {
            GpuFence::Dummy { signaled } => {
                signaled.store(true, Ordering::Release);
            }
            #[cfg(feature = "wgpu-backend")]
            _ => {}
        }
    }

    fn execute_graph(
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

    fn write_buffer(&self, _buffer: &GpuBuffer, offset: u64, data: &[u8]) {
        log::trace!(
            "DummyBackend: write_buffer offset={} len={}",
            offset,
            data.len()
        );
    }

    fn read_buffer(&self, _buffer: &GpuBuffer, offset: u64, size: u64) -> Vec<u8> {
        log::trace!("DummyBackend: read_buffer offset={} size={}", offset, size);
        // Return zeroed data
        vec![0u8; size as usize]
    }
}
