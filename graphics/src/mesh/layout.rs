//! Vertex layout definitions for meshes.
//!
//! Vertex layouts describe the structure of vertex data. Layouts are shared via `Arc`
//! since there are typically only a few combinations across many meshes. This enables
//! efficient batching via pointer comparison.

use std::sync::Arc;

/// Semantic meaning of a vertex attribute.
///
/// Semantics are used to match mesh attributes with shader inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexAttributeSemantic {
    /// Vertex position (typically float3).
    Position,
    /// Vertex normal (typically float3).
    Normal,
    /// Vertex tangent (typically float4, w = handedness).
    Tangent,
    /// Texture coordinates set 0 (typically float2).
    TexCoord0,
    /// Texture coordinates set 1 (typically float2).
    TexCoord1,
    /// Vertex color (typically float4 or unorm4).
    Color,
    /// Bone indices for skinning (typically uint4).
    Joints,
    /// Bone weights for skinning (typically float4).
    Weights,
    /// Custom attribute with a user-defined index.
    Custom(u32),
}

impl VertexAttributeSemantic {
    /// Get a unique index for this semantic (used for matching).
    pub fn index(&self) -> u32 {
        match self {
            Self::Position => 0,
            Self::Normal => 1,
            Self::Tangent => 2,
            Self::TexCoord0 => 3,
            Self::TexCoord1 => 4,
            Self::Color => 5,
            Self::Joints => 6,
            Self::Weights => 7,
            Self::Custom(i) => 100 + i,
        }
    }
}

/// Format of a vertex attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexAttributeFormat {
    /// Single 32-bit float.
    Float,
    /// Two 32-bit floats.
    Float2,
    /// Three 32-bit floats.
    Float3,
    /// Four 32-bit floats.
    Float4,
    /// Single 32-bit signed integer.
    Int,
    /// Two 32-bit signed integers.
    Int2,
    /// Three 32-bit signed integers.
    Int3,
    /// Four 32-bit signed integers.
    Int4,
    /// Single 32-bit unsigned integer.
    Uint,
    /// Two 32-bit unsigned integers.
    Uint2,
    /// Three 32-bit unsigned integers.
    Uint3,
    /// Four 32-bit unsigned integers.
    Uint4,
    /// Four 8-bit unsigned integers (normalized to 0.0-1.0).
    Unorm8x4,
    /// Four 8-bit signed integers (normalized to -1.0-1.0).
    Snorm8x4,
}

impl VertexAttributeFormat {
    /// Get the size in bytes of this format.
    pub fn size(&self) -> usize {
        match self {
            Self::Float => 4,
            Self::Float2 => 8,
            Self::Float3 => 12,
            Self::Float4 => 16,
            Self::Int | Self::Uint => 4,
            Self::Int2 | Self::Uint2 => 8,
            Self::Int3 | Self::Uint3 => 12,
            Self::Int4 | Self::Uint4 => 16,
            Self::Unorm8x4 | Self::Snorm8x4 => 4,
        }
    }
}

/// A single vertex attribute description.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    /// Semantic meaning of this attribute.
    pub semantic: VertexAttributeSemantic,
    /// Data format of this attribute.
    pub format: VertexAttributeFormat,
    /// Byte offset within the vertex.
    pub offset: u32,
}

impl VertexAttribute {
    /// Create a new vertex attribute.
    pub fn new(
        semantic: VertexAttributeSemantic,
        format: VertexAttributeFormat,
        offset: u32,
    ) -> Self {
        Self {
            semantic,
            format,
            offset,
        }
    }

    /// Create a position attribute (float3).
    pub fn position(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Position,
            VertexAttributeFormat::Float3,
            offset,
        )
    }

    /// Create a normal attribute (float3).
    pub fn normal(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Normal,
            VertexAttributeFormat::Float3,
            offset,
        )
    }

    /// Create a tangent attribute (float4).
    pub fn tangent(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Tangent,
            VertexAttributeFormat::Float4,
            offset,
        )
    }

    /// Create a texcoord0 attribute (float2).
    pub fn texcoord0(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::TexCoord0,
            VertexAttributeFormat::Float2,
            offset,
        )
    }

    /// Create a color attribute (float4).
    pub fn color(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Color,
            VertexAttributeFormat::Float4,
            offset,
        )
    }
}

/// Describes the layout of vertex data in a mesh.
///
/// Layouts are typically wrapped in `Arc` and shared between meshes
/// to reduce allocations and enable efficient batching by pointer comparison.
///
/// # Example
///
/// ```ignore
/// let layout = Arc::new(VertexLayout::new()
///     .with_attribute(VertexAttribute::position(0))
///     .with_attribute(VertexAttribute::normal(12))
///     .with_attribute(VertexAttribute::texcoord0(24)));
///
/// // Create multiple meshes sharing the same layout
/// let mesh1 = Mesh::new(device, MeshDescriptor::new(layout.clone(), ...));
/// let mesh2 = Mesh::new(device, MeshDescriptor::new(layout.clone(), ...));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexLayout {
    /// The vertex attributes in this layout.
    pub attributes: Vec<VertexAttribute>,
    /// Total stride (bytes per vertex). If None, computed from attributes.
    stride: Option<u32>,
    /// Optional label for debugging.
    pub label: Option<String>,
}

impl VertexLayout {
    /// Create a new empty vertex layout.
    pub fn new() -> Self {
        Self {
            attributes: Vec::new(),
            stride: None,
            label: None,
        }
    }

    /// Add a vertex attribute.
    pub fn with_attribute(mut self, attribute: VertexAttribute) -> Self {
        self.attributes.push(attribute);
        self
    }

    /// Set an explicit stride (bytes per vertex).
    pub fn with_stride(mut self, stride: u32) -> Self {
        self.stride = Some(stride);
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the stride (bytes per vertex).
    ///
    /// Returns the explicit stride if set, otherwise computes from attributes.
    pub fn stride(&self) -> u32 {
        if let Some(stride) = self.stride {
            return stride;
        }
        // Compute from attributes: max(offset + size)
        self.attributes
            .iter()
            .map(|attr| attr.offset + attr.format.size() as u32)
            .max()
            .unwrap_or(0)
    }

    /// Check if this layout has a specific semantic.
    pub fn has_semantic(&self, semantic: VertexAttributeSemantic) -> bool {
        self.attributes.iter().any(|attr| attr.semantic == semantic)
    }

    /// Get an attribute by semantic.
    pub fn get_attribute(&self, semantic: VertexAttributeSemantic) -> Option<&VertexAttribute> {
        self.attributes
            .iter()
            .find(|attr| attr.semantic == semantic)
    }

    /// Check if this layout is compatible with another layout.
    ///
    /// A layout is compatible if the other layout has all the semantics this one has,
    /// with matching formats.
    pub fn is_compatible_with(&self, other: &VertexLayout) -> bool {
        self.attributes.iter().all(|attr| {
            other.attributes.iter().any(|other_attr| {
                other_attr.semantic == attr.semantic && other_attr.format == attr.format
            })
        })
    }

    /// Get a set of semantic indices for fast comparison.
    pub fn semantic_set(&self) -> std::collections::HashSet<u32> {
        self.attributes
            .iter()
            .map(|attr| attr.semantic.index())
            .collect()
    }
}

impl Default for VertexLayout {
    fn default() -> Self {
        Self::new()
    }
}

/// Common vertex layouts for convenience.
impl VertexLayout {
    /// Position-only layout (12 bytes per vertex).
    pub fn position_only() -> Arc<Self> {
        Arc::new(
            Self::new()
                .with_attribute(VertexAttribute::position(0))
                .with_label("position_only"),
        )
    }

    /// Position + normal layout (24 bytes per vertex).
    pub fn position_normal() -> Arc<Self> {
        Arc::new(
            Self::new()
                .with_attribute(VertexAttribute::position(0))
                .with_attribute(VertexAttribute::normal(12))
                .with_label("position_normal"),
        )
    }

    /// Position + normal + texcoord layout (32 bytes per vertex).
    pub fn position_normal_uv() -> Arc<Self> {
        Arc::new(
            Self::new()
                .with_attribute(VertexAttribute::position(0))
                .with_attribute(VertexAttribute::normal(12))
                .with_attribute(VertexAttribute::texcoord0(24))
                .with_label("position_normal_uv"),
        )
    }

    /// Full PBR layout: position + normal + tangent + texcoord (48 bytes per vertex).
    pub fn pbr() -> Arc<Self> {
        Arc::new(
            Self::new()
                .with_attribute(VertexAttribute::position(0))
                .with_attribute(VertexAttribute::normal(12))
                .with_attribute(VertexAttribute::tangent(24))
                .with_attribute(VertexAttribute::texcoord0(40))
                .with_label("pbr"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_attribute_format_size() {
        assert_eq!(VertexAttributeFormat::Float.size(), 4);
        assert_eq!(VertexAttributeFormat::Float3.size(), 12);
        assert_eq!(VertexAttributeFormat::Float4.size(), 16);
        assert_eq!(VertexAttributeFormat::Unorm8x4.size(), 4);
    }

    #[test]
    fn test_vertex_layout_stride() {
        let layout = VertexLayout::new()
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12));
        assert_eq!(layout.stride(), 24);

        let layout_explicit = VertexLayout::new()
            .with_attribute(VertexAttribute::position(0))
            .with_stride(32);
        assert_eq!(layout_explicit.stride(), 32);
    }

    #[test]
    fn test_vertex_layout_has_semantic() {
        let layout = VertexLayout::new()
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12));

        assert!(layout.has_semantic(VertexAttributeSemantic::Position));
        assert!(layout.has_semantic(VertexAttributeSemantic::Normal));
        assert!(!layout.has_semantic(VertexAttributeSemantic::TexCoord0));
    }

    #[test]
    fn test_vertex_layout_compatibility() {
        let required = VertexLayout::new()
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12));

        let provided = VertexLayout::new()
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12))
            .with_attribute(VertexAttribute::texcoord0(24));

        // Required is compatible with provided (provided has everything required needs)
        assert!(required.is_compatible_with(&provided));

        // Provided is NOT compatible with required (required lacks texcoord)
        assert!(!provided.is_compatible_with(&required));
    }

    #[test]
    fn test_common_layouts() {
        let pos_only = VertexLayout::position_only();
        assert_eq!(pos_only.stride(), 12);
        assert_eq!(pos_only.attributes.len(), 1);

        let pbr = VertexLayout::pbr();
        assert_eq!(pbr.stride(), 48);
        assert_eq!(pbr.attributes.len(), 4);
    }
}
