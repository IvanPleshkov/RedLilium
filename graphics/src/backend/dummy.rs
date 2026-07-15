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

    /// What the dummy backend *pretends* to support (ADR-027). It enforces
    /// none of it — these values exist so engine-level validation behaves
    /// like a typical desktop device in tests.
    pub fn capabilities(&self) -> crate::device::DeviceCapabilities {
        crate::device::DeviceCapabilities {
            tier: crate::device::DeviceTier::Baseline,
            max_texture_dimension: 16384,
            max_buffer_size: 1 << 30, // 1 GB
            max_sampler_anisotropy: 16,
            sample_count_mask: 1 | 2 | 4 | 8,
            wireframe: true,
            async_compute: false,
            transfer_queue: false,
            compute_shaders: true,
            gpu_timestamps: false,
            mip_generation: false,
            memory_budget: false,
            // The dummy backend accepts BLAS/TLAS creation (returns dummy
            // handles) so graph-level tests can exercise AS-build passes, but
            // it advertises no ray-query support — engine-level gating (#110)
            // must behave as on a device without the feature.
            ray_query: false,
        }
    }

    /// Latest retired-slot GPU timings — always empty; the dummy backend
    /// records no timestamps (#95).
    pub fn latest_gpu_timings(&self) -> crate::device::FrameGpuTimings {
        crate::device::FrameGpuTimings::default()
    }

    /// GPU-memory statistics — always empty; the dummy backend has no GPU
    /// heaps or allocator (#98). Resource counts are filled at the device layer.
    pub fn latest_memory_stats(&self) -> crate::device::GpuMemoryStats {
        crate::device::GpuMemoryStats::default()
    }

    /// Adapter info for the dummy backend.
    pub fn adapter_info(&self) -> crate::instance::AdapterInfo {
        crate::instance::AdapterInfo {
            name: "Dummy Adapter".to_string(),
            vendor: "RedLilium".to_string(),
            device_type: crate::instance::AdapterType::Software,
            vendor_id: 0,
            device_id: 0,
        }
    }

    /// Create a buffer resource.
    pub fn create_buffer(
        &self,
        _descriptor: &BufferDescriptor,
    ) -> Result<GpuBuffer, GraphicsError> {
        Ok(GpuBuffer::Dummy)
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        _descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        Ok(GpuTexture::Dummy)
    }

    /// Create a sampler resource.
    pub fn create_sampler(
        &self,
        _descriptor: &SamplerDescriptor,
    ) -> Result<GpuSampler, GraphicsError> {
        Ok(GpuSampler::Dummy)
    }

    /// Create a fence for CPU-GPU synchronization.
    pub fn create_fence(&self, signaled: bool) -> Result<GpuFence, GraphicsError> {
        Ok(GpuFence::Dummy {
            signaled: AtomicBool::new(signaled),
        })
    }

    /// Wait for a fence to be signaled.
    pub fn wait_fence(&self, fence: &GpuFence) -> Result<(), GraphicsError> {
        match fence {
            GpuFence::Dummy { signaled } => {
                // In dummy mode, just spin until signaled (or assume signaled)
                while !signaled.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                Ok(())
            }
            #[cfg(feature = "wgpu-backend")]
            GpuFence::Wgpu { .. } => Ok(()),
            #[cfg(feature = "vulkan-backend")]
            GpuFence::Vulkan { .. } => Ok(()),
        }
    }

    /// Wait for a fence to be signaled with a timeout.
    ///
    /// Returns `Ok(true)` if the fence was signaled, `Ok(false)` on timeout.
    pub fn wait_fence_timeout(
        &self,
        fence: &GpuFence,
        timeout: std::time::Duration,
    ) -> Result<bool, GraphicsError> {
        match fence {
            GpuFence::Dummy { signaled } => {
                let start = web_time::Instant::now();
                while !signaled.load(Ordering::Acquire) {
                    if start.elapsed() >= timeout {
                        return Ok(false);
                    }
                    std::thread::yield_now();
                }
                Ok(true)
            }
            #[cfg(feature = "wgpu-backend")]
            GpuFence::Wgpu { .. } => Ok(false),
            #[cfg(feature = "vulkan-backend")]
            GpuFence::Vulkan { .. } => Ok(false),
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
        _compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError> {
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
        _offset: u64,
        _data: &[u8],
    ) -> Result<(), crate::error::GraphicsError> {
        Ok(())
    }

    /// Read data from a buffer (dummy: zeroed data of the requested size).
    pub fn read_buffer(
        &self,
        _buffer: &GpuBuffer,
        _offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, GraphicsError> {
        Ok(vec![0u8; size as usize])
    }

    /// Non-blocking readback (dummy: fills `dst` synchronously and clears the
    /// pending flag — there is no real GPU to wait on).
    pub fn read_buffer_async(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        size: u64,
        dst: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
        map_pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        if let Ok(data) = self.read_buffer(buffer, offset, size) {
            *dst.lock().unwrap_or_else(|e| e.into_inner()) = data;
        }
        map_pending.store(false, std::sync::atomic::Ordering::Release);
    }
}

impl Default for DummyBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-backend fence contract (XB-M2): a fence created unsignaled must
    /// poll as unsignaled until something signals it, on every backend.
    #[test]
    fn fresh_unsignaled_fence_polls_unsignaled() {
        let backend = DummyBackend::new();
        let fence = backend.create_fence(false).unwrap();
        assert!(!backend.is_fence_signaled(&fence));
        // Nothing will signal it — a bounded wait must time out, not succeed.
        assert!(
            !backend
                .wait_fence_timeout(&fence, std::time::Duration::from_millis(5))
                .unwrap()
        );

        backend.signal_fence(&fence);
        assert!(backend.is_fence_signaled(&fence));
        backend.wait_fence(&fence).unwrap();
    }

    #[test]
    fn fresh_signaled_fence_polls_signaled() {
        let backend = DummyBackend::new();
        let fence = backend.create_fence(true).unwrap();
        assert!(backend.is_fence_signaled(&fence));
        assert!(
            backend
                .wait_fence_timeout(&fence, std::time::Duration::from_millis(5))
                .unwrap()
        );
    }
}
