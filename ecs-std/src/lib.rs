//! # ECS Standard Components and Systems
//!
//! Standard game engine components and systems that bridge the generic ECS
//! infrastructure with the rendering and game layers.
//!
//! ## Components
//!
//! - [`Transform`] / [`GlobalTransform`] — Entity positioning (local TRS + world matrix)
//! - [`Camera`] — Camera with projection configuration and computed matrices
//! - [`Visibility`] — Render visibility toggle
//! - [`Name`] — Debug entity name
//! - [`MeshRenderer`] — Mesh + material rendering data
//! - [`DirectionalLight`] / [`PointLight`] / [`SpotLight`] — Light types
//!
//! ## Systems
//!
//! - [`update_global_transforms`] — Computes world matrices from local transforms
//! - [`update_camera_matrices`] — Computes view/projection matrices for cameras
//!
//! ## Scene Loading
//!
//! - [`spawn_scene`] — Converts a loaded [`Scene`](redlilium_core::scene::Scene) into ECS entities

pub mod components;
mod spawn;
pub mod systems;

pub use components::*;
pub use spawn::spawn_scene;
pub use systems::{update_camera_matrices, update_global_transforms};

/// Register all standard component types with the world.
///
/// Call this before running systems that query these types,
/// especially if no entities with these components have been spawned yet.
pub fn register_std_components(world: &mut redlilium_ecs::World) {
    world.register_component::<Transform>();
    world.register_component::<GlobalTransform>();
    world.register_component::<Camera>();
    world.register_component::<Visibility>();
    world.register_component::<Name>();
    world.register_component::<MeshRenderer>();
    world.register_component::<DirectionalLight>();
    world.register_component::<PointLight>();
    world.register_component::<SpotLight>();
}
