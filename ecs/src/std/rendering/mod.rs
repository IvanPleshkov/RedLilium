//! Graphics integration for the ECS.
//!
//! This module provides components, resources, and systems for rendering
//! ECS entities using the `redlilium-graphics` crate.
//!
//! # Components
//!
//! - [`MeshRenderer`] — list of (mesh, material) primitives on an entity
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

pub mod components;
pub(crate) mod material_inspector;
pub mod resources;
pub mod shaders;

pub use components::{
    CameraTarget, MaterialBundle, MeshRenderer, Primitive, PrimitiveMaterial, RenderPassType,
};
pub use resources::{
    CpuBundleInfo, MaterialManager, MaterialManagerError, MeshManager, RenderSchedule,
    TextureManager, TextureManagerError, pack_uniform_bytes,
};

use crate::World;

/// Register rendering component types with the world.
///
/// Call this after [`register_std_components`](crate::register_std_components)
/// to enable rendering support.
pub fn register_rendering_components(world: &mut World) {
    world.register_inspector::<MeshRenderer>();
    world.register_component::<CameraTarget>();
}
