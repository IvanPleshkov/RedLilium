//! Mesh types for the graphics engine.
//!
//! This module provides mesh data structures for rendering:
//!
//! - [`VertexLayout`] - Describes vertex attributes (shared via `Arc`)
//! - [`Mesh`] - GPU mesh with vertex/index buffers and topology
//!
//! # Efficient Sharing via Arc
//!
//! Vertex layouts are wrapped in `Arc` to reduce allocations since there are
//! typically only a few layout combinations across many meshes. This also
//! enables fast pointer comparison for batching.

mod data;
mod layout;

pub use data::{IndexFormat, Mesh, MeshDescriptor, PrimitiveTopology};
pub use layout::{VertexAttribute, VertexAttributeFormat, VertexAttributeSemantic, VertexLayout};
