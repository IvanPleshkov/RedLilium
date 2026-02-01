//! Mesh types for the graphics engine.
//!
//! This module provides mesh data structures for rendering:
//!
//! - [`VertexLayout`] - Describes vertex attributes across multiple buffers
//! - [`VertexBufferLayout`] - Describes a single vertex buffer binding
//! - [`Mesh`] - GPU mesh with one or more vertex buffers and topology
//!
//! # Multiple Vertex Buffers
//!
//! Meshes support multiple vertex buffers to enable:
//! - **Animation**: Separate static data (texcoords) from dynamic data (positions)
//! - **Skinning**: Bone indices/weights in their own buffer
//! - **Instancing**: Per-instance data with instance step mode
//!
//! # Efficient Sharing via Arc
//!
//! Vertex layouts are wrapped in `Arc` to reduce allocations since there are
//! typically only a few layout combinations across many meshes. This also
//! enables fast pointer comparison for batching.

mod data;
mod layout;

pub use data::{IndexFormat, Mesh, MeshDescriptor, PrimitiveTopology};
pub use layout::{
    VertexAttribute, VertexAttributeFormat, VertexAttributeSemantic, VertexBufferLayout,
    VertexLayout, VertexStepMode,
};
