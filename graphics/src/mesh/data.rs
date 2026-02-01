//! Mesh definition with vertex/index buffers.
//!
//! A [`Mesh`] contains the GPU resources needed for rendering geometry:
//! one or more vertex buffers, optional index buffer, topology, and vertex layout.
//!
//! # Multiple Vertex Buffers
//!
//! Meshes can have multiple vertex buffers to support:
//! - **Animation**: Static data (texcoords) separate from dynamic data (positions)
//! - **Skinning**: Bone indices/weights in their own buffer
//! - **Instancing**: Per-instance data in a separate buffer
//!
//! Each buffer corresponds to a slot defined in the [`VertexLayout`].

use std::sync::Arc;

use crate::device::GraphicsDevice;
use crate::resources::Buffer;

use super::layout::VertexLayout;

/// Primitive topology describing how vertices are assembled into primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrimitiveTopology {
    /// Each vertex is a separate point.
    PointList,
    /// Every two vertices form a line.
    LineList,
    /// Vertices form a connected strip of lines.
    LineStrip,
    /// Every three vertices form a triangle.
    #[default]
    TriangleList,
    /// Vertices form a connected strip of triangles.
    TriangleStrip,
}

impl PrimitiveTopology {
    /// Get the number of vertices per primitive (for non-strip topologies).
    pub fn vertices_per_primitive(&self) -> Option<u32> {
        match self {
            Self::PointList => Some(1),
            Self::LineList => Some(2),
            Self::TriangleList => Some(3),
            Self::LineStrip | Self::TriangleStrip => None, // Variable
        }
    }
}

/// Index format for indexed drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IndexFormat {
    /// 16-bit unsigned integers (max 65535 vertices).
    #[default]
    Uint16,
    /// 32-bit unsigned integers (max ~4 billion vertices).
    Uint32,
}

impl IndexFormat {
    /// Get the size in bytes of each index.
    pub fn size(&self) -> usize {
        match self {
            Self::Uint16 => 2,
            Self::Uint32 => 4,
        }
    }
}

/// Descriptor for creating a mesh.
///
/// # Example - Single Buffer
///
/// ```ignore
/// let layout = VertexLayout::position_normal_uv();
/// let desc = MeshDescriptor::new(layout)
///     .with_vertex_count(24)
///     .with_indices(IndexFormat::Uint16, 36)
///     .with_label("cube");
/// ```
///
/// # Example - Multiple Buffers (Animation)
///
/// ```ignore
/// let layout = VertexLayout::animated_dynamic(); // 2 buffers
/// let desc = MeshDescriptor::new(layout)
///     .with_vertex_count(1000)
///     .with_indices(IndexFormat::Uint16, 3000)
///     .with_label("character");
/// ```
#[derive(Debug)]
pub struct MeshDescriptor {
    /// Vertex layout (shared via Arc).
    pub layout: Arc<VertexLayout>,
    /// Primitive topology.
    pub topology: PrimitiveTopology,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Index format (None for non-indexed).
    pub index_format: Option<IndexFormat>,
    /// Number of indices (0 for non-indexed).
    pub index_count: u32,
    /// Optional label for debugging.
    pub label: Option<String>,
}

impl MeshDescriptor {
    /// Create a new mesh descriptor with the given layout.
    pub fn new(layout: Arc<VertexLayout>) -> Self {
        Self {
            layout,
            topology: PrimitiveTopology::TriangleList,
            vertex_count: 0,
            index_format: None,
            index_count: 0,
            label: None,
        }
    }

    /// Set the primitive topology.
    pub fn with_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Set the vertex count.
    pub fn with_vertex_count(mut self, count: u32) -> Self {
        self.vertex_count = count;
        self
    }

    /// Set indexed drawing with the given format and count.
    pub fn with_indices(mut self, format: IndexFormat, count: u32) -> Self {
        self.index_format = Some(format);
        self.index_count = count;
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Check if this mesh uses indexed drawing.
    pub fn is_indexed(&self) -> bool {
        self.index_format.is_some() && self.index_count > 0
    }

    /// Compute the required size for a specific vertex buffer.
    pub fn vertex_buffer_size(&self, buffer_index: usize) -> u64 {
        let stride = self.layout.buffer_stride(buffer_index);
        self.vertex_count as u64 * stride as u64
    }

    /// Compute the required index buffer size in bytes.
    pub fn index_buffer_size(&self) -> u64 {
        if let Some(format) = self.index_format {
            self.index_count as u64 * format.size() as u64
        } else {
            0
        }
    }

    /// Get the number of vertex buffers needed.
    pub fn buffer_count(&self) -> usize {
        self.layout.buffer_count()
    }
}

/// A GPU mesh with vertex and optional index buffers.
///
/// Meshes can have multiple vertex buffers to support animation, skinning,
/// and instancing. The buffer layout is defined by the [`VertexLayout`].
///
/// The vertex layout is shared via `Arc` since there are typically only a few
/// layout combinations across many meshes. This reduces allocations and enables
/// fast pointer comparison for batching.
///
/// # Example - Static Mesh
///
/// ```ignore
/// let layout = VertexLayout::position_normal_uv();
/// let mesh = device.create_mesh(&MeshDescriptor::new(layout)
///     .with_vertex_count(24)
///     .with_indices(IndexFormat::Uint16, 36)
///     .with_label("cube"))?;
///
/// // Access the single vertex buffer
/// let vb = mesh.vertex_buffer(0).unwrap();
/// ```
///
/// # Example - Animated Mesh
///
/// ```ignore
/// let layout = VertexLayout::animated_dynamic();
/// let mesh = device.create_mesh(&MeshDescriptor::new(layout)
///     .with_vertex_count(1000)
///     .with_label("character"))?;
///
/// // Buffer 0: static texcoords (upload once)
/// let static_buffer = mesh.vertex_buffer(0).unwrap();
///
/// // Buffer 1: dynamic positions/normals (update each frame)
/// let dynamic_buffer = mesh.vertex_buffer(1).unwrap();
/// ```
pub struct Mesh {
    device: Arc<GraphicsDevice>,
    layout: Arc<VertexLayout>,
    topology: PrimitiveTopology,
    vertex_buffers: Vec<Arc<Buffer>>,
    vertex_count: u32,
    index_buffer: Option<Arc<Buffer>>,
    index_format: Option<IndexFormat>,
    index_count: u32,
    label: Option<String>,
}

impl Mesh {
    /// Create a new mesh (called by GraphicsDevice).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: Arc<GraphicsDevice>,
        layout: Arc<VertexLayout>,
        topology: PrimitiveTopology,
        vertex_buffers: Vec<Arc<Buffer>>,
        vertex_count: u32,
        index_buffer: Option<Arc<Buffer>>,
        index_format: Option<IndexFormat>,
        index_count: u32,
        label: Option<String>,
    ) -> Self {
        debug_assert_eq!(
            vertex_buffers.len(),
            layout.buffer_count(),
            "Mesh buffer count must match layout buffer count"
        );

        Self {
            device,
            layout,
            topology,
            vertex_buffers,
            vertex_count,
            index_buffer,
            index_format,
            index_count,
            label,
        }
    }

    /// Get the parent device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Get the vertex layout.
    pub fn layout(&self) -> &Arc<VertexLayout> {
        &self.layout
    }

    /// Get the primitive topology.
    pub fn topology(&self) -> PrimitiveTopology {
        self.topology
    }

    /// Get a vertex buffer by index.
    pub fn vertex_buffer(&self, index: usize) -> Option<&Arc<Buffer>> {
        self.vertex_buffers.get(index)
    }

    /// Get all vertex buffers.
    pub fn vertex_buffers(&self) -> &[Arc<Buffer>] {
        &self.vertex_buffers
    }

    /// Get the number of vertex buffers.
    pub fn vertex_buffer_count(&self) -> usize {
        self.vertex_buffers.len()
    }

    /// Get the number of vertices.
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Get the index buffer, if any.
    pub fn index_buffer(&self) -> Option<&Arc<Buffer>> {
        self.index_buffer.as_ref()
    }

    /// Get the index format, if indexed.
    pub fn index_format(&self) -> Option<IndexFormat> {
        self.index_format
    }

    /// Get the number of indices.
    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    /// Check if this mesh uses indexed drawing.
    pub fn is_indexed(&self) -> bool {
        self.index_buffer.is_some()
    }

    /// Get the mesh label, if set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get the number of primitives based on topology and vertex/index count.
    pub fn primitive_count(&self) -> u32 {
        let count = if self.is_indexed() {
            self.index_count
        } else {
            self.vertex_count
        };

        match self.topology {
            PrimitiveTopology::PointList => count,
            PrimitiveTopology::LineList => count / 2,
            PrimitiveTopology::LineStrip => count.saturating_sub(1),
            PrimitiveTopology::TriangleList => count / 3,
            PrimitiveTopology::TriangleStrip => count.saturating_sub(2),
        }
    }
}

impl std::fmt::Debug for Mesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mesh")
            .field("label", &self.label)
            .field("topology", &self.topology)
            .field("vertex_count", &self.vertex_count)
            .field("vertex_buffer_count", &self.vertex_buffers.len())
            .field("index_count", &self.index_count)
            .field("layout", &self.layout.label)
            .finish()
    }
}

// Ensure Mesh is Send + Sync
static_assertions::assert_impl_all!(Mesh: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_topology_vertices() {
        assert_eq!(
            PrimitiveTopology::PointList.vertices_per_primitive(),
            Some(1)
        );
        assert_eq!(
            PrimitiveTopology::LineList.vertices_per_primitive(),
            Some(2)
        );
        assert_eq!(
            PrimitiveTopology::TriangleList.vertices_per_primitive(),
            Some(3)
        );
        assert_eq!(
            PrimitiveTopology::TriangleStrip.vertices_per_primitive(),
            None
        );
    }

    #[test]
    fn test_index_format_size() {
        assert_eq!(IndexFormat::Uint16.size(), 2);
        assert_eq!(IndexFormat::Uint32.size(), 4);
    }

    #[test]
    fn test_mesh_descriptor_single_buffer() {
        let layout = VertexLayout::position_normal_uv();
        let desc = MeshDescriptor::new(layout)
            .with_topology(PrimitiveTopology::TriangleList)
            .with_vertex_count(24)
            .with_indices(IndexFormat::Uint16, 36)
            .with_label("test_mesh");

        assert!(desc.is_indexed());
        assert_eq!(desc.buffer_count(), 1);
        assert_eq!(desc.vertex_buffer_size(0), 24 * 32); // 32 bytes per vertex
        assert_eq!(desc.index_buffer_size(), 36 * 2); // 2 bytes per index
    }

    #[test]
    fn test_mesh_descriptor_non_indexed() {
        let layout = VertexLayout::position_only();
        let desc = MeshDescriptor::new(layout).with_vertex_count(100);

        assert!(!desc.is_indexed());
        assert_eq!(desc.index_buffer_size(), 0);
    }
}
