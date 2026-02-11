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
/// Registers storage, inspector metadata, and (where applicable) default
/// insertion support. Call this before running systems or using the inspector.
pub fn register_std_components(world: &mut redlilium_ecs::World) {
    // Inspector-enabled components (support "Add Component" via Default)
    world.register_inspector_default::<Transform>();
    world.register_inspector_default::<GlobalTransform>();
    world.register_inspector_default::<Visibility>();
    world.register_inspector_default::<Name>();
    world.register_inspector_default::<DirectionalLight>();
    world.register_inspector_default::<PointLight>();
    world.register_inspector_default::<SpotLight>();

    // Inspector-enabled, readonly (no Default — constructed with parameters)
    world.register_inspector::<Camera>();

    // Hierarchy components (storage only — managed by hierarchy functions)
    world.register_component::<Parent>();
    world.register_component::<Children>();

    // Physics descriptor + handle components (feature-gated)
    #[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
    {
        world.register_inspector_default::<physics::components3d::RigidBody3D>();
        world.register_inspector_default::<physics::components3d::Collider3D>();
        world.register_component::<physics::physics3d::RigidBody3DHandle>();
    }
    #[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
    {
        world.register_inspector_default::<physics::components2d::RigidBody2D>();
        world.register_inspector_default::<physics::components2d::Collider2D>();
        world.register_component::<physics::physics2d::RigidBody2DHandle>();
    }
}
