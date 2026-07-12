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
//! - [`TextureManager`] — Owns and shares resident GPU textures (asset-based)
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

#[cfg(feature = "rendering")]
mod asset_actions;
#[cfg(feature = "rendering")]
mod asset_drag;
mod asset_inspect;
#[cfg(feature = "rendering")]
mod asset_ref_field;
pub mod components;
#[cfg(feature = "rendering")]
pub mod loaders;
pub(crate) mod material_inspector;
pub mod resources;
pub mod shaders;
#[cfg(feature = "rendering")]
pub mod shading;
pub mod systems;

#[cfg(feature = "rendering")]
pub use asset_actions::{DirtyMounts, SetAssetReferenceAction, SetAssetSettingsAction};
#[cfg(feature = "rendering")]
pub use asset_drag::{AssetDragPayload, asset_drop_target};
#[cfg(feature = "rendering")]
pub use asset_inspect::{NewAssetSpec, new_asset_spec};
pub use asset_inspect::{inspect_asset_settings, reference_accepted_kind};
pub use components::{
    CameraOutput, CameraTarget, CameraTargetSpec, MeshRenderer, OutputFormat, Primitive, SizePolicy,
};
#[cfg(feature = "rendering")]
pub use loaders::{
    MaterialData, MaterialInstanceData, MaterialInstanceLoader, MaterialInstanceSource,
    MaterialLoader, MaterialSource, MeshGenerator, MeshLoader, MeshSource, Shader, ShaderLoader,
    ShaderSource, TextureLoader, TextureSettings, TextureSource, VertexLayoutLoader,
    VertexLayoutSource,
};
#[cfg(feature = "rendering")]
pub use redlilium_assets::{AssetRef, AssetRefSource};
#[cfg(feature = "rendering")]
pub use resources::{
    ChangedAssets, MaterialAssetManager, MaterialInstanceManager, PipelineCache, ResolvedInstance,
    ResolvedMaterial, ResolvedTexture, ShaderManager, VertexLayoutManager,
};
pub use resources::{FrameRing, MainViewport, MeshManager, RenderSchedule, TextureManager};
#[cfg(feature = "rendering")]
pub use shading::{PropDef, PropValue, ShadingModel, ShadingRegistry, pack_props, texture_props};
#[cfg(feature = "rendering")]
pub use systems::HotReload;
pub use systems::{
    DebugRender, EguiRender, EnsureCameraTargets, FlushUploads, ForwardRender, FrameTarget,
    MaterialInstanceLoad, MeshLoad, ScenePass,
};

use crate::World;

/// Register rendering component types with the world.
///
/// Call this after [`register_std_components`](crate::register_std_components)
/// to enable rendering support.
pub fn register_rendering_components(world: &mut World) {
    world.register_inspector::<MeshRenderer>();
    // Serializable spec (ADR-029); its derived CameraTarget below is
    // runtime-only storage.
    world.register_inspector::<CameraOutput>();
    world.register_component::<CameraTarget>();
}
