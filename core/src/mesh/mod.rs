//! CPU-side mesh types and generators.
//!
//! This module provides GPU-agnostic mesh data structures:
//!
//! - [`VertexLayout`] - Describes vertex attributes across multiple buffers
//! - [`CpuMesh`] - CPU-side mesh data (vertex bytes, index bytes, layout)
//! - [`MeshDescriptor`] - Descriptor for creating GPU meshes
//! - Generators for common shapes (sphere, quad)
//!
//! These types are re-exported by `redlilium-graphics` for convenience.

mod data;
pub mod generators;
mod layout;

pub use data::{CpuMesh, IndexFormat, MeshDescriptor, PrimitiveTopology};
pub use layout::{
    VertexAttribute, VertexAttributeFormat, VertexAttributeSemantic, VertexBufferLayout,
    VertexLayout, VertexStepMode,
};
