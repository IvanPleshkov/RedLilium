//! GPU buffer resource.

use std::sync::{Arc, Weak};

use crate::device::GraphicsDevice;
use crate::types::BufferDescriptor;

/// A GPU buffer resource.
///
/// Buffers are created by [`GraphicsDevice::create_buffer`] and are reference-counted.
/// They hold a weak reference back to their parent device.
///
/// # Example
///
/// ```ignore
/// let buffer = device.create_buffer(&BufferDescriptor::new(1024, BufferUsage::VERTEX))?;
/// println!("Buffer size: {}", buffer.size());
/// ```
pub struct Buffer {
    device: Weak<GraphicsDevice>,
    descriptor: BufferDescriptor,
}

impl Buffer {
    /// Create a new buffer (called by GraphicsDevice).
    pub(crate) fn new(device: Weak<GraphicsDevice>, descriptor: BufferDescriptor) -> Self {
        Self { device, descriptor }
    }

    /// Get the parent device, if it still exists.
    pub fn device(&self) -> Option<Arc<GraphicsDevice>> {
        self.device.upgrade()
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
    use crate::types::BufferUsage;

    #[test]
    fn test_buffer_debug() {
        let desc = BufferDescriptor::new(1024, BufferUsage::VERTEX);
        let buffer = Buffer::new(Weak::new(), desc);
        let debug = format!("{:?}", buffer);
        assert!(debug.contains("Buffer"));
        assert!(debug.contains("1024"));
    }

    #[test]
    fn test_buffer_size() {
        let desc = BufferDescriptor::new(2048, BufferUsage::UNIFORM);
        let buffer = Buffer::new(Weak::new(), desc);
        assert_eq!(buffer.size(), 2048);
    }
}
