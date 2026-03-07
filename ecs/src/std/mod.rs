//! Standard components and systems for the ECS.
//!
//! This module provides the built-in component types (Transform, Camera,
//! lights, hierarchy, etc.) and the systems that operate on them
//! (global transform propagation, camera matrix updates).

pub mod components;
pub mod hierarchy;
#[cfg(any(
    feature = "physics-3d",
    feature = "physics-3d-f32",
    feature = "physics-2d",
    feature = "physics-2d-f32"
))]
pub mod physics;
#[cfg(feature = "rendering")]
pub mod rendering;
pub mod spawn;
pub mod systems;

pub use components::*;
pub use hierarchy::{
    HierarchyCommands, despawn_recursive, disable, enable, remove_parent, set_parent,
};
pub use spawn::spawn_scene;
#[cfg(feature = "rendering")]
pub use systems::DrawGrid;
pub use systems::{UpdateCameraMatrices, UpdateFreeFlyCamera, UpdateGlobalTransforms};
