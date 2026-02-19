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

// Re-export CPU-side types from core
pub use redlilium_core::mesh::{CpuMesh, IndexFormat, MeshDescriptor, PrimitiveTopology};

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
    pub fn new(
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
