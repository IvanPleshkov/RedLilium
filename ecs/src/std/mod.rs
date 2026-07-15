//! Standard components and systems for the ECS.
//!
//! This module provides the built-in component types (Transform, Camera,
//! lights, hierarchy, etc.) and the systems that operate on them
//! (global transform propagation, camera matrix updates).

#[cfg(feature = "rendering")]
pub mod assets;
pub mod components;
pub mod hierarchy;
#[cfg(any(
    feature = "physics-3d",
    feature = "physics-3d-f32",
    feature = "physics-2d",
    feature = "physics-2d-f32"
))]
pub mod physics;
pub mod play_mode;
// Scene assets (#101): needs the asset system (`rendering` pulls
// redlilium-assets) and RON.
#[cfg(all(feature = "rendering", feature = "serialize-ron"))]
pub mod scene;
// Native-only: TCP-based remote-control transport (tokio); no wasm equivalent.
#[cfg(all(feature = "rendering", not(target_arch = "wasm32")))]
pub mod remote;
#[cfg(feature = "rendering")]
pub mod rendering;
pub mod spawn;
pub mod systems;
pub mod time;

pub use components::*;
pub use hierarchy::{
    HierarchyCommands, despawn_recursive, disable, enable, remove_parent, set_parent,
};
pub use play_mode::{
    ManagePlayModeTransitions, PanicInfo, PlayControl, PlayModeAwareRegistry, PlayModeTransition,
    PlayStartTick, PlayState, PlayTasks,
};
pub use spawn::spawn_scene;
#[cfg(feature = "rendering")]
pub use systems::DrawGrid;
pub use systems::{UpdateCameraMatrices, UpdateFreeFlyCamera, UpdateGlobalTransforms};
pub use time::{GameTime, RealTime};
