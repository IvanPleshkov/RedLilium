//! Rendering component types.

mod camera_target;
mod material_bundle;
mod mesh_renderer;

pub use camera_target::CameraTarget;
pub use material_bundle::{MaterialBundle, RenderPassType};
pub use mesh_renderer::{MeshRenderer, Primitive, PrimitiveMaterial};
