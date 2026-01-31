//! Buffer types and descriptors.

use bitflags::bitflags;

bitflags! {
    /// Usage flags for buffers.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        /// Buffer can be used as a vertex buffer.
        const VERTEX = 1 << 0;
        /// Buffer can be used as an index buffer.
        const INDEX = 1 << 1;
        /// Buffer can be used as a uniform buffer.
        const UNIFORM = 1 << 2;
        /// Buffer can be used as a storage buffer.
        const STORAGE = 1 << 3;
        /// Buffer can be used as an indirect buffer.
        const INDIRECT = 1 << 4;
        /// Buffer can be copied from.
        const COPY_SRC = 1 << 5;
        /// Buffer can be copied to.
        const COPY_DST = 1 << 6;
        /// Buffer is mappable for CPU access.
        const MAP_READ = 1 << 7;
        /// Buffer is mappable for CPU write.
        const MAP_WRITE = 1 << 8;
    }
}

impl Default for BufferUsage {
    fn default() -> Self {
        Self::empty()
    }
}

/// Descriptor for creating a buffer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct BufferDescriptor {
    /// Debug label for the buffer.
    pub label: Option<String>,
    /// Size in bytes.
    pub size: u64,
    /// Usage flags.
    pub usage: BufferUsage,
}

impl BufferDescriptor {
    /// Create a new buffer descriptor.
    pub fn new(size: u64, usage: BufferUsage) -> Self {
        Self {
            label: None,
            size,
            usage,
        }
    }

    /// Set the debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}
