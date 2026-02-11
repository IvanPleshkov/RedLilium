#![allow(refining_impl_trait)]

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
//! - [`Name`] — Debug entity name (owned String)
//! - [`DirectionalLight`] / [`PointLight`] / [`SpotLight`] — Light types
//! - [`Parent`] / [`Children`] — Entity hierarchy
//!
//! ## Systems
//!
//! - [`UpdateGlobalTransforms`] — Computes world matrices (hierarchy-aware)
//! - [`UpdateCameraMatrices`] — Computes view matrices for cameras
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
#[cfg(any(
    feature = "physics-3d",
    feature = "physics-3d-f32",
    feature = "physics-2d",
    feature = "physics-2d-f32"
))]
pub mod physics;
mod spawn;
pub mod systems;
#[cfg(feature = "inspector")]
pub mod ui;

pub use components::*;
pub use hierarchy::{HierarchyCommands, despawn_recursive, remove_parent, set_parent};
pub use spawn::spawn_scene;
pub use systems::{UpdateCameraMatrices, UpdateGlobalTransforms};

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

    // Physics descriptor + handle components (feature-gated)
    #[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
    {
        world.register_component::<physics::components3d::RigidBody3D>();
        world.register_component::<physics::components3d::Collider3D>();
        world.register_component::<physics::physics3d::RigidBody3DHandle>();
    }
    #[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
    {
        world.register_component::<physics::components2d::RigidBody2D>();
        world.register_component::<physics::components2d::Collider2D>();
        world.register_component::<physics::physics2d::RigidBody2DHandle>();
    }
}
