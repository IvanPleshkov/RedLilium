//! Asset loaders for rendering resources, living next to the managers that
//! consume them. Gated by the `assets` feature.
//!
//! These implement [`AssetLoader`](redlilium_assets::AssetLoader) from the asset
//! framework. Loaders only produce the resident asset; deduplication / sharing
//! is a consumer concern (typically a manager — see
//! [`VertexLayoutManager`](super::VertexLayoutManager)).

mod material;
mod material_instance;
mod mesh;
mod shader;
mod vertex_layout;

pub use material::{MaterialData, MaterialLoader, MaterialSource};
pub use material_instance::{MaterialInstanceData, MaterialInstanceLoader, MaterialInstanceSource};
pub use mesh::{MeshGenerator, MeshLoader, MeshSource};
pub use shader::{Shader, ShaderLoader, ShaderSource};
pub use vertex_layout::{VertexLayoutLoader, VertexLayoutSource};
