//! GPU buffer resource.

use std::sync::Arc;

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
        }
    }

    /// Get the GPU handle for this buffer.
    pub fn gpu_handle(&self) -> &GpuBuffer {
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

    /// Crate-internal read of this buffer's host-visible mapped memory.
    ///
    /// Used by the frame pipeline to drain [`ReadbackBuffer`] operations after
    /// the slot's fence has signalled (so the GPU has finished writing). Not
    /// public so external code cannot read GPU memory that may still be in use.
    ///
    /// [`ReadbackBuffer`]: crate::TransferOperation::ReadbackBuffer
    pub(crate) fn read_mapped(&self, offset: u64, size: u64) -> Result<Vec<u8>, GraphicsError> {
        self.device
            .instance()
            .backend()
            .read_buffer(&self.gpu_handle, offset, size)
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
}
