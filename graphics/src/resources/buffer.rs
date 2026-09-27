//! GPU buffer resource.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::backend::GpuBuffer;
use crate::device::GraphicsDevice;
use crate::error::GraphicsError;
use crate::types::BufferDescriptor;

/// A GPU buffer resource.
///
/// Buffers are created by [`GraphicsDevice::create_buffer`] and are reference-counted.
/// They hold a strong reference to their parent device, keeping it alive.
///
/// # Example
///
/// ```ignore
/// let buffer = device.create_buffer(&BufferDescriptor::new(1024, BufferUsage::VERTEX))?;
/// println!("Buffer size: {}", buffer.size());
/// ```
pub struct Buffer {
    descriptor: BufferDescriptor,
    gpu_handle: GpuBuffer,
    /// True while an async `map_async` readback of this buffer is in flight
    /// (#33). Guards against a second overlapping map (a wgpu validation error)
    /// and against writing the buffer while it is still mapped. Cleared by the
    /// readback completion callback after the last queued batch.
    map_pending: Arc<AtomicBool>,
    readbacks: parking_lot::Mutex<crate::readback::ReadbackBatch>,
    /// Declared after `gpu_handle` deliberately: fields drop in declaration
    /// order, and this keep-alive must outlive the handle's `Drop` (which
    /// calls `vkDestroyBuffer` and needs the backend, transitively owned by
    /// the device→instance chain, to still be alive). See #50.
    device: Arc<GraphicsDevice>,
}

impl Buffer {
    /// Create a new buffer (called by GraphicsDevice).
    pub(crate) fn new(
        device: Arc<GraphicsDevice>,
        descriptor: BufferDescriptor,
        gpu_handle: GpuBuffer,
    ) -> Self {
        Self {
            device,
            descriptor,
            gpu_handle,
            map_pending: Arc::new(AtomicBool::new(false)),
            readbacks: Default::default(),
        }
    }

    /// Get the GPU handle for this buffer.
    pub(crate) fn gpu_handle(&self) -> &GpuBuffer {
        &self.gpu_handle
    }

    /// Get the parent device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Get the buffer descriptor.
    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }

    /// Get the buffer size in bytes.
    pub fn size(&self) -> u64 {
        self.descriptor.size
    }

    /// True while an async `map_async` readback of this buffer is still in
    /// flight (#33). Used by the debug write-while-mapped guard
    /// ([`scheduler`](crate::scheduler)) and by consumers that must avoid
    /// starting a new readback / reallocating the buffer until the prior map
    /// resolves (e.g. the editor pick path). Completion clears it after all
    /// queued batches finish; wgpu completion requires device polling.
    pub fn is_map_pending(&self) -> bool {
        self.map_pending.load(Ordering::Acquire)
    }

    /// Get the buffer label, if set.
    pub fn label(&self) -> Option<&str> {
        self.descriptor.label.as_deref()
    }

    /// Crate-internal direct write into this buffer's host-visible mapped memory.
    ///
    /// This is the low-level primitive behind [`RingBuffer`](crate::RingBuffer);
    /// it is **not** public so that external code cannot do unsynchronized GPU
    /// writes. The caller must guarantee the GPU is not currently reading the
    /// written region (the ring buffer guarantees this via per-frame slots).
    pub(crate) fn write_mapped(&self, offset: u64, data: &[u8]) -> Result<(), GraphicsError> {
        self.device
            .instance()
            .backend()
            .write_buffer(&self.gpu_handle, offset, data)
    }

    /// Queue a batch and map its union once. Later batches wait until unmap.
    pub(crate) fn read_mapped_batch(self: &Arc<Self>, requests: crate::readback::ReadbackBatch) {
        let mut waiting = self.readbacks.lock();
        waiting.extend(requests);
        if self.map_pending.swap(true, Ordering::AcqRel) {
            return;
        }
        let batch = std::mem::take(&mut *waiting);
        drop(waiting);
        self.start_readback(batch);
    }
    fn start_readback(self: &Arc<Self>, mut batch: crate::readback::ReadbackBatch) {
        batch.retain(|(range, result)| {
            if range.is_empty() {
                result.complete(Ok(Vec::new()));
                false
            } else {
                true
            }
        });
        if batch.is_empty() {
            self.finish_readback();
            return;
        }
        let start = batch.iter().map(|(r, _)| r.start).min().unwrap();
        let end = batch.iter().map(|(r, _)| r.end).max().unwrap();
        let buffer = self.clone();
        self.device.instance().backend().read_buffer_async(
            &self.gpu_handle,
            start as u64,
            (end - start) as u64,
            Box::new(move |result| {
                for (range, dst) in batch {
                    dst.complete(match &result {
                        Ok(bytes) => Ok(bytes[range.start - start..range.end - start].to_vec()),
                        Err(e) => Err(e.clone()),
                    });
                }
                buffer.finish_readback();
            }),
        );
    }
    fn finish_readback(self: &Arc<Self>) {
        let mut waiting = self.readbacks.lock();
        if waiting.is_empty() {
            self.map_pending.store(false, Ordering::Release);
            return;
        }
        let next = std::mem::take(&mut *waiting);
        drop(waiting);
        self.start_readback(next);
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("size", &self.descriptor.size)
            .field("usage", &self.descriptor.usage)
            .field("label", &self.descriptor.label)
            .finish()
    }
}

// Ensure Buffer is Send + Sync
static_assertions::assert_impl_all!(Buffer: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;
    use crate::types::BufferUsage;

    fn create_test_device() -> Arc<GraphicsDevice> {
        let instance = GraphicsInstance::new().unwrap();
        instance.create_device().unwrap()
    }

    #[test]
    fn test_buffer_debug() {
        let device = create_test_device();
        let buffer = device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::VERTEX))
            .unwrap();
        let debug = format!("{:?}", buffer);
        assert!(debug.contains("Buffer"));
        assert!(debug.contains("1024"));
    }

    #[test]
    fn test_buffer_size() {
        let device = create_test_device();
        let buffer = device
            .create_buffer(&BufferDescriptor::new(2048, BufferUsage::UNIFORM))
            .unwrap();
        assert_eq!(buffer.size(), 2048);
    }
    #[cfg(feature = "wgpu-backend")]
    #[test]
    fn queued_readback_batches_and_errors_complete() {
        use crate::{BackendType, InstanceParameters, Readback, ReadbackStatus};
        let Ok(instance) = GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Wgpu),
        ) else {
            return;
        };
        let device = instance.create_device().unwrap();
        let buffer = device
            .create_buffer(&BufferDescriptor::new(
                32,
                BufferUsage::MAP_READ | BufferUsage::COPY_DST,
            ))
            .unwrap();
        let first = Readback::new();
        let second = Readback::new();
        buffer.read_mapped_batch(vec![(0..16, first.clone())]);
        assert!(buffer.is_map_pending());
        buffer.read_mapped_batch(vec![(8..32, second.clone())]);
        let native_device = {
            let backend = instance.backend();
            let crate::backend::GpuBackend::Wgpu(backend) = &*backend else {
                unreachable!()
            };
            backend.device().clone()
        };
        for _ in 0..3 {
            native_device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(10)),
                })
                .unwrap();
        }
        assert_eq!(first.take_result().unwrap().unwrap(), vec![0; 16]);
        assert_eq!(second.take_result().unwrap().unwrap(), vec![0; 24]);
        assert!(!buffer.is_map_pending());

        let invalid = device
            .create_buffer(&BufferDescriptor::new(16, BufferUsage::COPY_DST))
            .unwrap();
        let result = Readback::new();
        invalid.read_mapped_batch(vec![(0..4, result.clone())]);
        assert_eq!(result.status(), ReadbackStatus::Ready);
        assert!(result.take_result().unwrap().is_err());
        assert!(!invalid.is_map_pending());
    }
}
