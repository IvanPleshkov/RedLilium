//! Rendering component types.

mod camera_target;
mod material_bundle;
mod render_material;
mod render_mesh;

pub use camera_target::{CameraTarget, PerEntityBuffers};
pub use material_bundle::{MaterialBundle, RenderPassType};
pub use render_material::RenderMaterial;
pub use render_mesh::RenderMesh;
