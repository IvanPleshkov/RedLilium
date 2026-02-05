//! CPU-side mesh data structures.
//!
//! This module provides:
//! - [`PrimitiveTopology`] - How vertices are assembled into primitives
//! - [`IndexFormat`] - Index data format (u16 or u32)
//! - [`MeshDescriptor`] - Descriptor for creating GPU meshes
//! - [`CpuMesh`] - CPU-side mesh holding raw vertex and index data

use std::sync::Arc;

use crate::material::CpuMaterial;

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

/// A CPU-side mesh holding raw vertex and index data.
///
/// This is the GPU-agnostic representation of a mesh. It can be
/// created by generators (sphere, quad) or loaded from files,
/// and then uploaded to the GPU via `GraphicsDevice::create_mesh_from_cpu`.
///
/// # Multi-Buffer Support
///
/// Like [`VertexLayout`] and [`MeshDescriptor`], `CpuMesh` supports multiple
/// vertex buffers. Each buffer slot stores its raw byte data. The number
/// of buffers must match the layout's buffer count.
#[derive(Clone)]
pub struct CpuMesh {
    layout: Arc<VertexLayout>,
    topology: PrimitiveTopology,
    vertex_buffers: Vec<Vec<u8>>,
    vertex_count: u32,
    index_data: Option<Vec<u8>>,
    index_format: Option<IndexFormat>,
    index_count: u32,
    material: Option<Arc<CpuMaterial>>,
    label: Option<String>,
}

impl CpuMesh {
    /// Create a new empty CpuMesh with the given layout.
    ///
    /// Vertex buffers are initialized as empty vectors matching
    /// the layout's buffer count.
    pub fn new(layout: Arc<VertexLayout>) -> Self {
        let buffer_count = layout.buffer_count();
        Self {
            layout,
            topology: PrimitiveTopology::TriangleList,
            vertex_buffers: vec![Vec::new(); buffer_count],
            vertex_count: 0,
            index_data: None,
            index_format: None,
            index_count: 0,
            material: None,
            label: None,
        }
    }

    /// Set raw vertex data for a specific buffer slot.
    ///
    /// Vertex count is inferred from the data length and stride.
    pub fn with_vertex_data(mut self, buffer_index: usize, data: Vec<u8>) -> Self {
        let stride = self.layout.buffer_stride(buffer_index) as usize;
        if stride > 0 {
            self.vertex_count = (data.len() / stride) as u32;
        }
        if buffer_index < self.vertex_buffers.len() {
            self.vertex_buffers[buffer_index] = data;
        }
        self
    }

    /// Set index data as u16 indices.
    pub fn with_indices_u16(mut self, indices: &[u16]) -> Self {
        self.index_data = Some(bytemuck::cast_slice(indices).to_vec());
        self.index_format = Some(IndexFormat::Uint16);
        self.index_count = indices.len() as u32;
        self
    }

    /// Set index data as u32 indices.
    pub fn with_indices_u32(mut self, indices: &[u32]) -> Self {
        self.index_data = Some(bytemuck::cast_slice(indices).to_vec());
        self.index_format = Some(IndexFormat::Uint32);
        self.index_count = indices.len() as u32;
        self
    }

    /// Set raw index data bytes directly with format and count.
    pub fn with_raw_index_data(mut self, data: Vec<u8>, format: IndexFormat, count: u32) -> Self {
        self.index_data = Some(data);
        self.index_format = Some(format);
        self.index_count = count;
        self
    }

    /// Set the primitive topology.
    pub fn with_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Set the material.
    pub fn with_material(mut self, material: Arc<CpuMaterial>) -> Self {
        self.material = Some(material);
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the vertex layout.
    pub fn layout(&self) -> &Arc<VertexLayout> {
        &self.layout
    }

    /// Get the primitive topology.
    pub fn topology(&self) -> PrimitiveTopology {
        self.topology
    }

    /// Get raw vertex data for a specific buffer slot.
    pub fn vertex_buffer_data(&self, index: usize) -> Option<&[u8]> {
        self.vertex_buffers.get(index).map(|v| v.as_slice())
    }

    /// Get the number of vertices.
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Get the raw index data.
    pub fn index_data(&self) -> Option<&[u8]> {
        self.index_data.as_deref()
    }

    /// Get the index format.
    pub fn index_format(&self) -> Option<IndexFormat> {
        self.index_format
    }

    /// Get the number of indices.
    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    /// Check if this mesh uses indexed drawing.
    pub fn is_indexed(&self) -> bool {
        self.index_data.is_some()
    }

    /// Get the material, if set.
    pub fn material(&self) -> Option<&Arc<CpuMaterial>> {
        self.material.as_ref()
    }

    /// Get the debug label.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Get the number of vertex buffers.
    pub fn buffer_count(&self) -> usize {
        self.vertex_buffers.len()
    }

    /// Create a [`MeshDescriptor`] matching this CpuMesh.
    ///
    /// Useful when creating a GPU mesh from this CPU mesh.
    pub fn to_descriptor(&self) -> MeshDescriptor {
        let mut desc = MeshDescriptor::new(self.layout.clone())
            .with_topology(self.topology)
            .with_vertex_count(self.vertex_count);
        if let Some(format) = self.index_format {
            desc = desc.with_indices(format, self.index_count);
        }
        if let Some(label) = &self.label {
            desc = desc.with_label(label.clone());
        }
        desc
    }
}

impl std::fmt::Debug for CpuMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuMesh")
            .field("label", &self.label)
            .field("topology", &self.topology)
            .field("vertex_count", &self.vertex_count)
            .field("buffer_count", &self.vertex_buffers.len())
            .field("index_count", &self.index_count)
            .field(
                "material",
                &self.material.as_ref().map(|m| m.name.as_deref()),
            )
            .field("layout", &self.layout.label)
            .finish()
    }
}

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

    #[test]
    fn test_cpu_mesh_basic() {
        let layout = VertexLayout::position_only();
        // 3 vertices * 12 bytes = 36 bytes
        let vertex_data = vec![0u8; 36];
        let mesh = CpuMesh::new(layout)
            .with_vertex_data(0, vertex_data)
            .with_label("test");

        assert_eq!(mesh.vertex_count(), 3);
        assert!(!mesh.is_indexed());
        assert_eq!(mesh.buffer_count(), 1);
        assert_eq!(mesh.label(), Some("test"));
    }

    #[test]
    fn test_cpu_mesh_indexed() {
        let layout = VertexLayout::position_only();
        let vertex_data = vec![0u8; 48]; // 4 vertices
        let indices: [u32; 6] = [0, 1, 2, 2, 3, 0];
        let mesh = CpuMesh::new(layout)
            .with_vertex_data(0, vertex_data)
            .with_indices_u32(&indices);

        assert_eq!(mesh.vertex_count(), 4);
        assert!(mesh.is_indexed());
        assert_eq!(mesh.index_count(), 6);
        assert_eq!(mesh.index_format(), Some(IndexFormat::Uint32));
    }

    #[test]
    fn test_cpu_mesh_to_descriptor() {
        let layout = VertexLayout::position_normal_uv();
        let vertex_data = vec![0u8; 320]; // 10 vertices * 32 bytes
        let indices: [u32; 12] = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4];
        let mesh = CpuMesh::new(layout)
            .with_vertex_data(0, vertex_data)
            .with_indices_u32(&indices)
            .with_label("desc_test");

        let desc = mesh.to_descriptor();
        assert_eq!(desc.vertex_count, 10);
        assert_eq!(desc.index_count, 12);
        assert_eq!(desc.index_format, Some(IndexFormat::Uint32));
        assert_eq!(desc.label.as_deref(), Some("desc_test"));
    }
}
