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

// ============================================================================
// Indirect Drawing Arguments
// ============================================================================

/// Arguments for a non-indexed indirect draw call.
///
/// This struct matches the GPU layout for `vkCmdDrawIndirect` / `wgpu::DrawIndirect`.
/// The buffer containing these arguments must have [`BufferUsage::INDIRECT`].
///
/// # Memory Layout
///
/// The struct is `#[repr(C)]` to ensure GPU-compatible memory layout:
/// - Total size: 16 bytes
/// - Alignment: 4 bytes
///
/// # Example
///
/// ```ignore
/// // Create indirect arguments for drawing 36 vertices as 100 instances
/// let args = DrawIndirectArgs {
///     vertex_count: 36,
///     instance_count: 100,
///     first_vertex: 0,
///     first_instance: 0,
/// };
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DrawIndirectArgs {
    /// Number of vertices to draw.
    pub vertex_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Index of the first vertex to draw.
    pub first_vertex: u32,
    /// Instance ID of the first instance to draw.
    pub first_instance: u32,
}

impl DrawIndirectArgs {
    /// Size of the struct in bytes.
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Create new indirect draw arguments.
    pub fn new(vertex_count: u32, instance_count: u32) -> Self {
        Self {
            vertex_count,
            instance_count,
            first_vertex: 0,
            first_instance: 0,
        }
    }

    /// Set the first vertex index.
    pub fn with_first_vertex(mut self, first_vertex: u32) -> Self {
        self.first_vertex = first_vertex;
        self
    }

    /// Set the first instance index.
    pub fn with_first_instance(mut self, first_instance: u32) -> Self {
        self.first_instance = first_instance;
        self
    }

    /// Convert to bytes for uploading to a buffer.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, Self::SIZE as usize) }
    }
}

/// Arguments for an indexed indirect draw call.
///
/// This struct matches the GPU layout for `vkCmdDrawIndexedIndirect` / `wgpu::DrawIndexedIndirect`.
/// The buffer containing these arguments must have [`BufferUsage::INDIRECT`].
///
/// # Memory Layout
///
/// The struct is `#[repr(C)]` to ensure GPU-compatible memory layout:
/// - Total size: 20 bytes
/// - Alignment: 4 bytes
///
/// # Example
///
/// ```ignore
/// // Create indirect arguments for drawing 36 indices as 100 instances
/// let args = DrawIndexedIndirectArgs {
///     index_count: 36,
///     instance_count: 100,
///     first_index: 0,
///     base_vertex: 0,
///     first_instance: 0,
/// };
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DrawIndexedIndirectArgs {
    /// Number of indices to draw.
    pub index_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Index of the first index to draw.
    pub first_index: u32,
    /// Value added to each index before reading from the vertex buffer.
    pub base_vertex: i32,
    /// Instance ID of the first instance to draw.
    pub first_instance: u32,
}

impl DrawIndexedIndirectArgs {
    /// Size of the struct in bytes.
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

    /// Create new indexed indirect draw arguments.
    pub fn new(index_count: u32, instance_count: u32) -> Self {
        Self {
            index_count,
            instance_count,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }

    /// Set the first index.
    pub fn with_first_index(mut self, first_index: u32) -> Self {
        self.first_index = first_index;
        self
    }

    /// Set the base vertex offset.
    pub fn with_base_vertex(mut self, base_vertex: i32) -> Self {
        self.base_vertex = base_vertex;
        self
    }

    /// Set the first instance index.
    pub fn with_first_instance(mut self, first_instance: u32) -> Self {
        self.first_instance = first_instance;
        self
    }

    /// Convert to bytes for uploading to a buffer.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, Self::SIZE as usize) }
    }
}
