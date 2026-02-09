//! # ECS Standard Components and Systems
//!
//! Standard game engine components and systems that bridge the generic ECS
//! infrastructure with the rendering and game layers.
//!
//! ## Components
//!
//! - [`Transform`] / [`GlobalTransform`] — Entity positioning (local TRS + world matrix)
//! - [`Camera`] — Camera with computed view and projection matrices
//! - [`Visibility`] — Render visibility toggle
//! - [`Name`] — Debug entity name (via [`StringId`](redlilium_ecs::StringId))
//! - [`DirectionalLight`] / [`PointLight`] / [`SpotLight`] — Light types
//! - [`Parent`] / [`Children`] — Entity hierarchy
//!
//! ## Systems
//!
//! - [`update_global_transforms`] — Computes world matrices (hierarchy-aware)
//! - [`update_camera_matrices`] — Computes view matrices for cameras
//!
//! ## Hierarchy
//!
//! - [`set_parent`] / [`remove_parent`] / [`despawn_recursive`] — Hierarchy operations
//!
//! ## Scene Loading
//!
//! - [`spawn_scene`] — Converts a loaded [`Scene`](redlilium_core::scene::Scene) into ECS entities

pub mod components;
pub mod hierarchy;
mod spawn;
pub mod systems;

pub use components::*;
pub use hierarchy::{HierarchyCommands, despawn_recursive, remove_parent, set_parent};
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
    world.register_component::<DirectionalLight>();
    world.register_component::<PointLight>();
    world.register_component::<SpotLight>();
    world.register_component::<Parent>();
    world.register_component::<Children>();
}
