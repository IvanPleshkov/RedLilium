//! Asset loaders for rendering resources, living next to the managers that
//! consume them. Gated by the `assets` feature.
//!
//! These implement [`AssetLoader`](redlilium_assets::AssetLoader) from the asset
//! framework. Loaders only produce the resident asset; deduplication / sharing
//! is a consumer concern (typically a manager — see
//! [`VertexLayoutManager`](super::VertexLayoutManager)).

mod mesh;
mod vertex_layout;

pub use mesh::{MeshGenerator, MeshLoader, MeshSource};
pub use vertex_layout::{VertexLayoutLoader, VertexLayoutSource};
