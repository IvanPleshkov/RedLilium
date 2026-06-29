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

mod asset_inspect;
pub mod components;
#[cfg(feature = "rendering")]
pub mod loaders;
pub(crate) mod material_inspector;
pub mod resources;
pub mod shaders;
pub mod systems;

pub use asset_inspect::inspect_asset_settings;
pub use components::{
    CameraTarget, MaterialBundle, MeshRenderer, Primitive, PrimitiveMaterial, RenderPassType,
};
#[cfg(feature = "rendering")]
pub use loaders::{MeshGenerator, MeshLoader, MeshSource, VertexLayoutLoader, VertexLayoutSource};
#[cfg(feature = "rendering")]
pub use resources::VertexLayoutManager;
pub use resources::{
    CpuBundleInfo, FrameRing, MaterialManager, MaterialManagerError, MeshHandle, MeshManager,
    RenderSchedule, TextureManager, TextureManagerError, pack_uniform_bytes,
};
pub use systems::{
    DebugRender, EguiRender, FlushUploads, ForwardRender, FrameTarget, MeshLoad, ScenePass,
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
