//! GPU buffer resource.

use std::sync::Arc;

use crate::device::GraphicsDevice;
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
    device: Arc<GraphicsDevice>,
    descriptor: BufferDescriptor,
}

impl Buffer {
    /// Create a new buffer (called by GraphicsDevice).
    pub(crate) fn new(device: Arc<GraphicsDevice>, descriptor: BufferDescriptor) -> Self {
        Self { device, descriptor }
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
        let desc = BufferDescriptor::new(1024, BufferUsage::VERTEX);
        let buffer = Buffer::new(create_test_device(), desc);
        let debug = format!("{:?}", buffer);
        assert!(debug.contains("Buffer"));
        assert!(debug.contains("1024"));
    }

    #[test]
    fn test_buffer_size() {
        let desc = BufferDescriptor::new(2048, BufferUsage::UNIFORM);
        let buffer = Buffer::new(create_test_device(), desc);
        assert_eq!(buffer.size(), 2048);
    }
}
