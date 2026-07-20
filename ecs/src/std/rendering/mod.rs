//! Graphics integration for the ECS.
//!
//! This module provides components, resources, and systems for rendering
//! ECS entities using the `redlilium-graphics` crate.
//!
//! # Components
//!
//! - [`MeshRenderer`] — list of (mesh, material) primitives on an entity
//! - [`CameraOutput`] / [`RenderPath`] — a camera's serializable specs:
//!   *where* the image goes and *how* it is produced (ADR-029/ADR-035)
//! - [`CameraTarget`] / [`PipelineTargets`] — the runtime-derived GPU targets
//!
//! # Resources
//!
//! - [`TextureManager`] — Owns and shares resident GPU textures (asset-based)
//! - [`RenderSchedule`] — Holds the current frame's [`FrameSchedule`](redlilium_graphics::FrameSchedule)
//! - [`PipelineRegistry`] — Resolves [`RenderPath`] names to registered
//!   [`CameraRenderPipeline`]s
//!
//! # Systems
//!
//! - [`CameraRender`] — the per-camera dispatcher: resolves each camera's
//!   pipeline and has it record the camera's passes into the frame graph
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
pub mod deferred;
#[cfg(feature = "rendering")]
pub mod loaders;
pub(crate) mod material_inspector;
pub mod pipeline;
pub mod resources;
pub mod scene_drawer;
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
    CameraEnvironment, CameraExposure, CameraOutput, CameraTarget, CameraTargetSpec,
    DEFERRED_PIPELINE, FORWARD_PIPELINE, MeshRenderer, OutputFormat, PipelineTargets, Primitive,
    RenderPath, STD_ENVIRONMENT, SizePolicy, TemporalJitter,
};
pub use deferred::DeferredPipeline;
#[cfg(feature = "rendering")]
pub use loaders::{
    EnvironmentData, EnvironmentLoader, EnvironmentSource, MaterialData, MaterialInstanceData,
    MaterialInstanceLoader, MaterialInstanceSource, MaterialLoader, MaterialSource, MeshGenerator,
    MeshLoader, MeshSource, Shader, ShaderLoader, ShaderSource, TextureLoader, TextureSettings,
    TextureSource, VertexLayoutLoader, VertexLayoutSource,
};
pub use pipeline::{
    CameraRenderPipeline, CameraView, ForwardPipeline, PipelineRegistry, RecordCtx,
};
#[cfg(feature = "rendering")]
pub use redlilium_assets::{AssetRef, AssetRefSource};
#[cfg(feature = "rendering")]
pub use resources::{
    ChangedAssets, EnvironmentManager, MaterialAssetManager, MaterialInstanceManager,
    PipelineCache, ResolvedEnvironment, ResolvedInstance, ResolvedMaterial, ResolvedTexture,
    ShaderManager, VertexLayoutManager,
};
pub use resources::{
    FrameRing, MainViewport, MeshManager, RenderSchedule, TemporalState, TextureManager,
    jitter_pixels,
};
pub use scene_drawer::{DrawArgs, RenderPhase, SceneDrawer, VisibleScene};
#[cfg(feature = "rendering")]
pub use shading::{
    FeatureValue, PropDef, PropValue, ShadingModel, ShadingRegistry, pack_props, texture_props,
};
#[cfg(feature = "rendering")]
pub use systems::HotReload;
pub use systems::{
    CameraRender, DebugRender, EguiRender, EnsureCameraTargets, FlushUploads, FrameTarget,
    MaterialInstanceLoad, MeshLoad, ScenePass,
};

use crate::World;

/// Register rendering component types with the world.
///
/// Call this after [`register_std_components`](crate::register_std_components)
/// to enable rendering support.
pub fn register_rendering_components(world: &mut World) {
    world.register_inspector::<MeshRenderer>();
    // Serializable specs (ADR-029: where the image goes; ADR-035: how it is
    // produced); their derived CameraTarget/PipelineTargets below are
    // runtime-only storage.
    world.register_inspector::<CameraOutput>();
    world.register_inspector::<RenderPath>();
    world.register_inspector_default::<CameraExposure>();
    world.register_inspector_default::<TemporalJitter>();
    world.register_inspector_default::<CameraEnvironment>();
    world.register_component::<CameraTarget>();
    world.register_component::<PipelineTargets>();
}
