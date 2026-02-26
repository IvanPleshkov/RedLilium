//! Graphics integration for the ECS.
//!
//! This module provides components, resources, and systems for rendering
//! ECS entities using the `redlilium-graphics` crate.
//!
//! # Components
//!
//! - [`RenderMesh`] — GPU mesh attached to an entity
//! - [`RenderMaterial`] — GPU material instance attached to an entity
//! - [`CameraTarget`] — Render target textures for a camera entity
//!
//! # Resources
//!
//! - [`TextureManager`] — Caches GPU textures and samplers
//! - [`RenderSchedule`] — Holds the current frame's [`FrameSchedule`](redlilium_graphics::FrameSchedule)
//!
//! # Systems
//!
//! - [`ForwardRenderSystem`] — Collects renderable entities and submits
//!   draw commands for each camera with a render target
//!
//! # Feature Gate
//!
//! This module is only available when the `rendering` feature is enabled.

mod components;
#[cfg(feature = "inspector")]
mod material_inspector;
mod resources;
pub mod shaders;
mod systems;

pub use components::{
    CameraTarget, MaterialBundle, PerEntityBuffers, RenderMaterial, RenderMesh, RenderPassType,
};
pub use resources::{
    CpuBundleInfo, MaterialManager, MaterialManagerError, MeshManager, RenderSchedule,
    TextureManager, TextureManagerError, pack_uniform_bytes,
};
pub use systems::{
    EditorForwardRenderSystem, ForwardRenderSystem, InitializeRenderEntities, SyncMaterialUniforms,
    UpdatePerEntityUniforms,
};

use crate::World;

/// Register rendering component types with the world.
///
/// Call this after [`register_std_components`](crate::register_std_components)
/// to enable rendering support.
pub fn register_rendering_components(world: &mut World) {
    world.register_inspector::<RenderMesh>();
    world.register_inspector::<RenderMaterial>();
    world.register_component::<CameraTarget>();
    world.register_component::<PerEntityBuffers>();
}
