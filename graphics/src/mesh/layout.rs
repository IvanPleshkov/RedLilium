//! Vertex layout definitions for meshes.
//!
//! Vertex layouts describe the structure of vertex data across multiple buffers.
//! This design supports:
//!
//! - **Static/Dynamic separation**: Keep frequently-updated data (animated positions)
//!   in a separate buffer from static data (texcoords, colors).
//! - **Skinning**: Bone indices/weights in their own buffer.
//! - **Instancing**: Per-instance data in a buffer with instance step mode.
//!
//! Layouts are shared via `Arc` since there are typically only a few combinations
//! across many meshes. This enables efficient batching via pointer comparison.
//!
//! # Buffer Slots
//!
//! Each vertex buffer is bound to a slot (0, 1, 2, ...). Attributes reference
//! which slot they read from via `buffer_index`.
//!
//! # Example
//!
//! ```ignore
//! // Animated mesh with separate static and dynamic buffers:
//! // Buffer 0: Static data (texcoords, colors) - uploaded once
//! // Buffer 1: Dynamic data (position, normal) - updated each frame
//!
//! let layout = Arc::new(VertexLayout::new()
//!     // Buffer 0: static data
//!     .with_buffer(VertexBufferLayout::new(8))  // stride = 8 (float2 texcoord)
//!     // Buffer 1: dynamic data
//!     .with_buffer(VertexBufferLayout::new(24)) // stride = 24 (float3 + float3)
//!     // Attributes
//!     .with_attribute(VertexAttribute::new(
//!         VertexAttributeSemantic::TexCoord0,
//!         VertexAttributeFormat::Float2,
//!         0,  // offset
//!         0,  // buffer_index (static)
//!     ))
//!     .with_attribute(VertexAttribute::new(
//!         VertexAttributeSemantic::Position,
//!         VertexAttributeFormat::Float3,
//!         0,  // offset
//!         1,  // buffer_index (dynamic)
//!     ))
//!     .with_attribute(VertexAttribute::new(
//!         VertexAttributeSemantic::Normal,
//!         VertexAttributeFormat::Float3,
//!         12, // offset
//!         1,  // buffer_index (dynamic)
//!     )));
//! ```

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

/// How the vertex buffer advances: per-vertex or per-instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VertexStepMode {
    /// Buffer advances once per vertex (default).
    #[default]
    Vertex,
    /// Buffer advances once per instance (for instanced rendering).
    Instance,
}

/// Describes a single vertex buffer binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexBufferLayout {
    /// Stride in bytes between consecutive elements.
    pub stride: u32,
    /// How the buffer advances (per-vertex or per-instance).
    pub step_mode: VertexStepMode,
}

impl VertexBufferLayout {
    /// Create a new vertex buffer layout with the given stride.
    pub fn new(stride: u32) -> Self {
        Self {
            stride,
            step_mode: VertexStepMode::Vertex,
        }
    }

    /// Set the step mode to per-instance.
    pub fn with_instance_step(mut self) -> Self {
        self.step_mode = VertexStepMode::Instance;
        self
    }

    /// Create a per-instance buffer layout.
    pub fn per_instance(stride: u32) -> Self {
        Self {
            stride,
            step_mode: VertexStepMode::Instance,
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
    /// Byte offset within the vertex buffer.
    pub offset: u32,
    /// Index of the vertex buffer this attribute reads from.
    pub buffer_index: u32,
}

impl VertexAttribute {
    /// Create a new vertex attribute.
    pub fn new(
        semantic: VertexAttributeSemantic,
        format: VertexAttributeFormat,
        offset: u32,
        buffer_index: u32,
    ) -> Self {
        Self {
            semantic,
            format,
            offset,
            buffer_index,
        }
    }

    /// Create a position attribute (float3) at buffer 0.
    pub fn position(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Position,
            VertexAttributeFormat::Float3,
            offset,
            0,
        )
    }

    /// Create a normal attribute (float3) at buffer 0.
    pub fn normal(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Normal,
            VertexAttributeFormat::Float3,
            offset,
            0,
        )
    }

    /// Create a tangent attribute (float4) at buffer 0.
    pub fn tangent(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Tangent,
            VertexAttributeFormat::Float4,
            offset,
            0,
        )
    }

    /// Create a texcoord0 attribute (float2) at buffer 0.
    pub fn texcoord0(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::TexCoord0,
            VertexAttributeFormat::Float2,
            offset,
            0,
        )
    }

    /// Create a color attribute (float4) at buffer 0.
    pub fn color(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Color,
            VertexAttributeFormat::Float4,
            offset,
            0,
        )
    }

    /// Create a joints attribute (uint4) at buffer 0.
    pub fn joints(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Joints,
            VertexAttributeFormat::Uint4,
            offset,
            0,
        )
    }

    /// Create a weights attribute (float4) at buffer 0.
    pub fn weights(offset: u32) -> Self {
        Self::new(
            VertexAttributeSemantic::Weights,
            VertexAttributeFormat::Float4,
            offset,
            0,
        )
    }

    /// Set the buffer index for this attribute.
    pub fn at_buffer(mut self, buffer_index: u32) -> Self {
        self.buffer_index = buffer_index;
        self
    }
}

/// Describes the layout of vertex data across one or more buffers.
///
/// Layouts support multiple vertex buffers, enabling separation of:
/// - Static vs dynamic data (for animation)
/// - Skinning data (joints/weights)
/// - Per-instance data (for instancing)
///
/// Layouts are typically wrapped in `Arc` and shared between meshes
/// to reduce allocations and enable efficient batching by pointer comparison.
///
/// # Single Buffer Example
///
/// ```ignore
/// // Simple interleaved layout (one buffer)
/// let layout = Arc::new(VertexLayout::new()
///     .with_buffer(VertexBufferLayout::new(32))
///     .with_attribute(VertexAttribute::position(0))
///     .with_attribute(VertexAttribute::normal(12))
///     .with_attribute(VertexAttribute::texcoord0(24)));
/// ```
///
/// # Multi-Buffer Example (Animation)
///
/// ```ignore
/// // Animated mesh: static data in buffer 0, dynamic in buffer 1
/// let layout = Arc::new(VertexLayout::new()
///     .with_buffer(VertexBufferLayout::new(8))   // Buffer 0: texcoord (static)
///     .with_buffer(VertexBufferLayout::new(24))  // Buffer 1: pos+normal (dynamic)
///     .with_attribute(VertexAttribute::texcoord0(0).at_buffer(0))
///     .with_attribute(VertexAttribute::position(0).at_buffer(1))
///     .with_attribute(VertexAttribute::normal(12).at_buffer(1)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexLayout {
    /// Descriptions of each vertex buffer binding.
    pub buffers: Vec<VertexBufferLayout>,
    /// The vertex attributes, each referencing a buffer by index.
    pub attributes: Vec<VertexAttribute>,
    /// Optional label for debugging.
    pub label: Option<String>,
}

impl VertexLayout {
    /// Create a new empty vertex layout.
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            attributes: Vec::new(),
            label: None,
        }
    }

    /// Add a vertex buffer binding.
    pub fn with_buffer(mut self, buffer: VertexBufferLayout) -> Self {
        self.buffers.push(buffer);
        self
    }

    /// Add a vertex attribute.
    pub fn with_attribute(mut self, attribute: VertexAttribute) -> Self {
        self.attributes.push(attribute);
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the number of vertex buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Get a buffer layout by index.
    pub fn buffer(&self, index: usize) -> Option<&VertexBufferLayout> {
        self.buffers.get(index)
    }

    /// Get the stride for a specific buffer.
    pub fn buffer_stride(&self, buffer_index: usize) -> u32 {
        self.buffers
            .get(buffer_index)
            .map(|b| b.stride)
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

    /// Get all attributes for a specific buffer.
    pub fn attributes_for_buffer(
        &self,
        buffer_index: u32,
    ) -> impl Iterator<Item = &VertexAttribute> {
        self.attributes
            .iter()
            .filter(move |attr| attr.buffer_index == buffer_index)
    }

    /// Check if this layout is compatible with another layout.
    ///
    /// A layout is compatible if the other layout has all the semantics this one has,
    /// with matching formats. Buffer indices don't need to match.
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

    /// Validate the layout (check that all attributes reference valid buffers).
    pub fn validate(&self) -> Result<(), String> {
        for attr in &self.attributes {
            if attr.buffer_index as usize >= self.buffers.len() {
                return Err(format!(
                    "Attribute {:?} references buffer {} but only {} buffers defined",
                    attr.semantic,
                    attr.buffer_index,
                    self.buffers.len()
                ));
            }
        }
        Ok(())
    }
}

impl Default for VertexLayout {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Common Layouts - Static Meshes (Single Buffer)
// ============================================================================

impl VertexLayout {
    /// Position-only layout (12 bytes per vertex, single buffer).
    pub fn position_only() -> Arc<Self> {
        Arc::new(
            Self::new()
                .with_buffer(VertexBufferLayout::new(12))
                .with_attribute(VertexAttribute::position(0))
                .with_label("position_only"),
        )
    }

    /// Position + normal layout (24 bytes per vertex, single buffer).
    pub fn position_normal() -> Arc<Self> {
        Arc::new(
            Self::new()
                .with_buffer(VertexBufferLayout::new(24))
                .with_attribute(VertexAttribute::position(0))
                .with_attribute(VertexAttribute::normal(12))
                .with_label("position_normal"),
        )
    }

    /// Position + normal + texcoord layout (32 bytes per vertex, single buffer).
    pub fn position_normal_uv() -> Arc<Self> {
        Arc::new(
            Self::new()
                .with_buffer(VertexBufferLayout::new(32))
                .with_attribute(VertexAttribute::position(0))
                .with_attribute(VertexAttribute::normal(12))
                .with_attribute(VertexAttribute::texcoord0(24))
                .with_label("position_normal_uv"),
        )
    }

    /// Full PBR layout: position + normal + tangent + texcoord (48 bytes, single buffer).
    pub fn pbr() -> Arc<Self> {
        Arc::new(
            Self::new()
                .with_buffer(VertexBufferLayout::new(48))
                .with_attribute(VertexAttribute::position(0))
                .with_attribute(VertexAttribute::normal(12))
                .with_attribute(VertexAttribute::tangent(24))
                .with_attribute(VertexAttribute::texcoord0(40))
                .with_label("pbr"),
        )
    }

    /// Full animated PBR layout with static, dynamic, and skinning buffers.
    ///
    /// - Buffer 0 (static, 8 bytes): texcoord
    /// - Buffer 1 (dynamic, 40 bytes): position, normal, tangent
    /// - Buffer 2 (static, 32 bytes): joints, weights
    ///
    /// For CPU skinning: update buffer 1 each frame.
    /// For GPU skinning: shader reads buffer 2 and transforms buffer 1.
    pub fn animated_pbr() -> Arc<Self> {
        Arc::new(
            Self::new()
                // Buffer 0: static UV data
                .with_buffer(VertexBufferLayout::new(8))
                // Buffer 1: dynamic geometry (position/normal/tangent)
                .with_buffer(VertexBufferLayout::new(40))
                // Buffer 2: skinning data
                .with_buffer(VertexBufferLayout::new(32))
                // Static attributes (buffer 0)
                .with_attribute(VertexAttribute::texcoord0(0).at_buffer(0))
                // Dynamic attributes (buffer 1)
                .with_attribute(VertexAttribute::position(0).at_buffer(1))
                .with_attribute(VertexAttribute::normal(12).at_buffer(1))
                .with_attribute(VertexAttribute::tangent(24).at_buffer(1))
                // Skinning attributes (buffer 2)
                .with_attribute(VertexAttribute::joints(0).at_buffer(2))
                .with_attribute(VertexAttribute::weights(16).at_buffer(2))
                .with_label("animated_pbr"),
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
    fn test_vertex_buffer_layout() {
        let buffer = VertexBufferLayout::new(32);
        assert_eq!(buffer.stride, 32);
        assert_eq!(buffer.step_mode, VertexStepMode::Vertex);

        let instance_buffer = VertexBufferLayout::per_instance(64);
        assert_eq!(instance_buffer.step_mode, VertexStepMode::Instance);
    }

    #[test]
    fn test_vertex_layout_single_buffer() {
        let layout = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(24))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12));

        assert_eq!(layout.buffer_count(), 1);
        assert_eq!(layout.buffer_stride(0), 24);
        assert!(layout.has_semantic(VertexAttributeSemantic::Position));
        assert!(layout.has_semantic(VertexAttributeSemantic::Normal));
        assert!(!layout.has_semantic(VertexAttributeSemantic::TexCoord0));
        assert!(layout.validate().is_ok());
    }

    #[test]
    fn test_vertex_layout_multi_buffer() {
        let layout = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(8)) // texcoord
            .with_buffer(VertexBufferLayout::new(24)) // pos + normal
            .with_attribute(VertexAttribute::texcoord0(0).at_buffer(0))
            .with_attribute(VertexAttribute::position(0).at_buffer(1))
            .with_attribute(VertexAttribute::normal(12).at_buffer(1));

        assert_eq!(layout.buffer_count(), 2);
        assert_eq!(layout.buffer_stride(0), 8);
        assert_eq!(layout.buffer_stride(1), 24);
        assert!(layout.validate().is_ok());

        // Check attributes_for_buffer
        let buffer0_attrs: Vec<_> = layout.attributes_for_buffer(0).collect();
        assert_eq!(buffer0_attrs.len(), 1);
        assert_eq!(
            buffer0_attrs[0].semantic,
            VertexAttributeSemantic::TexCoord0
        );

        let buffer1_attrs: Vec<_> = layout.attributes_for_buffer(1).collect();
        assert_eq!(buffer1_attrs.len(), 2);
    }

    #[test]
    fn test_vertex_layout_validation() {
        let invalid_layout = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(12))
            .with_attribute(VertexAttribute::position(0).at_buffer(5)); // Invalid!

        assert!(invalid_layout.validate().is_err());
    }

    #[test]
    fn test_vertex_layout_compatibility() {
        let required = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(24))
            .with_attribute(VertexAttribute::position(0))
            .with_attribute(VertexAttribute::normal(12));

        // Different buffer layout but same semantics
        let provided = VertexLayout::new()
            .with_buffer(VertexBufferLayout::new(12))
            .with_buffer(VertexBufferLayout::new(12))
            .with_attribute(VertexAttribute::position(0).at_buffer(0))
            .with_attribute(VertexAttribute::normal(0).at_buffer(1))
            .with_attribute(VertexAttribute::texcoord0(0).at_buffer(0)); // Extra

        // Required is compatible with provided (semantics match)
        assert!(required.is_compatible_with(&provided));

        // Provided is NOT compatible with required (required lacks texcoord)
        assert!(!provided.is_compatible_with(&required));
    }

    #[test]
    fn test_common_layouts() {
        let pos_only = VertexLayout::position_only();
        assert_eq!(pos_only.buffer_count(), 1);
        assert_eq!(pos_only.buffer_stride(0), 12);

        let pbr = VertexLayout::pbr();
        assert_eq!(pbr.buffer_count(), 1);
        assert_eq!(pbr.buffer_stride(0), 48);
        assert_eq!(pbr.attributes.len(), 4);
    }
}
