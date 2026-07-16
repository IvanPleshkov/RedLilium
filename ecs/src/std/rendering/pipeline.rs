//! Per-camera render pipelines (ADR-035, #128).
//!
//! A camera's [`RenderPath`](super::RenderPath) component names *how* its
//! frame is produced; this module owns the implementations behind those
//! names:
//!
//! - [`CameraRenderPipeline`] — the recipe: derive auxiliary targets, record
//!   the camera's passes into the frame graph.
//! - [`PipelineRegistry`] — the world resource resolving stable names to
//!   registered pipelines. The engine registers [`ForwardPipeline`] under
//!   [`FORWARD_PIPELINE`](super::FORWARD_PIPELINE); game plugins register
//!   their own paths at startup.
//! - [`ForwardPipeline`] — the built-in forward path: one pass per camera
//!   into its [`CameraTarget`], drawing the [`VisibleScene`] via the shared
//!   [`SceneDrawer`].
//!
//! Pipelines never declare cross-pass ordering: passes order themselves
//! through the graph's resource-dependency derivation (#47), exactly like
//! cross-camera ordering in ADR-029. A pipeline that records several passes
//! (a shadow pass feeding a main pass) relies on the same mechanism.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use redlilium_graphics::{
    ColorAttachment, DepthStencilAttachment, GraphicsPass, PassHandle, RenderGraph,
    RenderTargetConfig,
};

use crate::{Entity, World};

use super::scene_drawer::{DrawArgs, SceneDrawer, VisibleScene};
use super::{CameraTarget, FORWARD_PIPELINE, FrameRing, PipelineCache, shaders};

/// One camera's view of the frame, prepared by the render dispatcher
/// ([`CameraRender`](super::systems::CameraRender)) for its pipeline.
pub struct CameraView {
    /// The camera entity — pipelines read their own components off it
    /// (settings, [`PipelineTargets`](super::PipelineTargets)).
    pub entity: Entity,
    /// The camera's view-projection matrix, ready for
    /// [`CameraUniforms`](shaders::CameraUniforms).
    pub view_projection: [[f32; 4]; 4],
    /// The camera's derived color+depth target.
    pub target: CameraTarget,
    /// Whether this is the primary camera — the one whose surface the host
    /// composites and whose main pass becomes the
    /// [`ScenePass`](super::ScenePass).
    pub primary: bool,
}

/// Frame-wide context handed to [`CameraRenderPipeline::record`].
pub struct RecordCtx<'a> {
    /// The world — pipelines pull the resources they need
    /// ([`FrameRing`], [`PipelineCache`]) through interior mutability.
    pub world: &'a World,
    /// The frame's visible renderables, gathered once by the dispatcher.
    pub scene: &'a VisibleScene,
}

/// A registered per-camera render pipeline (ADR-035) — the implementation
/// behind a [`RenderPath`](super::RenderPath) name.
///
/// Implementations must be `Send + Sync`: one instance serves every camera
/// naming it, potentially from the multi-threaded schedule runner. Per-frame
/// state belongs in world resources; persistent caches (like the
/// [`SceneDrawer`]'s binding groups) behind interior mutability.
pub trait CameraRenderPipeline: Send + Sync {
    /// The stable registry name — what [`RenderPath`](super::RenderPath)
    /// serializes.
    fn name(&self) -> &str;

    /// Derive the pipeline's auxiliary targets for `camera` into its
    /// [`PipelineTargets`](super::PipelineTargets) (a shadow map, an HDR
    /// intermediate), re-deriving whenever they disagree with the spec —
    /// the same discipline as
    /// [`EnsureCameraTargets`](super::EnsureCameraTargets), which invokes
    /// this hook. The default pipeline needs none.
    fn ensure_targets(&self, _world: &mut World, _camera: Entity) {}

    /// Record the camera's passes into the frame graph. Returns the camera's
    /// **main** pass handle — the last writer of its
    /// [`CameraTarget`](super::CameraTarget) color, which overlays (debug
    /// lines, egui) and the host composite order after.
    fn record(
        &self,
        ctx: &RecordCtx<'_>,
        view: &CameraView,
        graph: &mut RenderGraph,
    ) -> Option<PassHandle>;
}

/// World resource resolving [`RenderPath`](super::RenderPath) names to
/// registered [`CameraRenderPipeline`]s.
///
/// Constructed with the engine's [`ForwardPipeline`] registered; game plugins
/// add their own via [`register`](Self::register) at startup. An unknown name
/// resolves to the forward path with a warning (logged once per name), so a
/// scene referencing an absent plugin's pipeline stays loadable.
pub struct PipelineRegistry {
    pipelines: HashMap<String, Arc<dyn CameraRenderPipeline>>,
    /// Names already warned about — resolve() is per-frame, the log is not.
    warned: Mutex<HashSet<String>>,
}

impl Default for PipelineRegistry {
    fn default() -> Self {
        let mut registry = Self {
            pipelines: HashMap::new(),
            warned: Mutex::new(HashSet::new()),
        };
        registry.register(Arc::new(ForwardPipeline::default()));
        registry
    }
}

impl PipelineRegistry {
    /// Register a pipeline under its [`name`](CameraRenderPipeline::name),
    /// replacing any previous registration of that name.
    pub fn register(&mut self, pipeline: Arc<dyn CameraRenderPipeline>) {
        self.pipelines.insert(pipeline.name().to_owned(), pipeline);
    }

    /// The pipeline registered under `name`, if any.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn CameraRenderPipeline>> {
        self.pipelines.get(name)
    }

    /// Resolve `name`, degrading to the forward path (with a once-per-name
    /// warning) when it is not registered. `None` only if the forward
    /// pipeline itself has been removed.
    pub fn resolve(&self, name: &str) -> Option<Arc<dyn CameraRenderPipeline>> {
        if let Some(pipeline) = self.pipelines.get(name) {
            return Some(pipeline.clone());
        }
        if self
            .warned
            .lock()
            .expect("pipeline registry warn set poisoned")
            .insert(name.to_owned())
        {
            log::warn!(
                "render pipeline {name:?} is not registered; \
                 falling back to {FORWARD_PIPELINE:?}"
            );
        }
        self.pipelines.get(FORWARD_PIPELINE).cloned()
    }
}

/// The engine's built-in forward path: one [`GraphicsPass`] per camera into
/// its [`CameraTarget`], drawing every visible primitive via the shared
/// [`SceneDrawer`] (which owns the frame-invariant binding-group cache — the
/// pipeline instance persists in the registry across frames, so the cache
/// does too).
#[derive(Default)]
pub struct ForwardPipeline {
    drawer: SceneDrawer,
}

impl CameraRenderPipeline for ForwardPipeline {
    fn name(&self) -> &str {
        FORWARD_PIPELINE
    }

    fn record(
        &self,
        ctx: &RecordCtx<'_>,
        view: &CameraView,
        graph: &mut RenderGraph,
    ) -> Option<PassHandle> {
        let world = ctx.world;
        // The camera set is `external` (a dynamic uniform): bound at offset 0,
        // this view's ring offset supplied per draw.
        let (camera_offset, ring_buffer) = {
            let mut ring = world.resource_mut::<FrameRing>();
            let offset = ring.push(bytemuck::bytes_of(&shaders::CameraUniforms {
                view_projection: view.view_projection,
            }));
            (offset, ring.buffer().clone())
        };
        let clear = view.target.clear_color;
        let mut pass = GraphicsPass::new(
            if view.primary {
                "scene_view"
            } else {
                "camera_view"
            }
            .into(),
        );
        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(
                    ColorAttachment::from_texture(view.target.color.clone())
                        .with_clear_color(clear[0], clear[1], clear[2], clear[3]),
                )
                .with_depth_stencil(
                    DepthStencilAttachment::from_texture(view.target.depth.clone())
                        .with_clear_depth(1.0),
                ),
        );
        {
            let mut pipelines = world.resource_mut::<PipelineCache>();
            self.drawer.record(
                &mut pass,
                ctx.scene,
                &ring_buffer,
                &mut pipelines,
                &DrawArgs {
                    camera_offset,
                    color_format: view.target.color.format(),
                    depth_format: view.target.depth.format(),
                },
            );
        }
        Some(graph.add_graphics_pass(pass))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A game-defined pipeline stub — records nothing.
    struct NullPipeline(&'static str);
    impl CameraRenderPipeline for NullPipeline {
        fn name(&self) -> &str {
            self.0
        }
        fn record(
            &self,
            _ctx: &RecordCtx<'_>,
            _view: &CameraView,
            _graph: &mut RenderGraph,
        ) -> Option<PassHandle> {
            None
        }
    }

    /// The default registry carries the engine's forward path under the name
    /// `RenderPath` components default to.
    #[test]
    fn default_registry_has_forward() {
        let registry = PipelineRegistry::default();
        let pipeline = registry.resolve(FORWARD_PIPELINE).expect("forward exists");
        assert_eq!(pipeline.name(), FORWARD_PIPELINE);
        assert_eq!(
            super::super::RenderPath::default().pipeline,
            FORWARD_PIPELINE
        );
    }

    /// An unknown name degrades to the forward path (a scene naming an absent
    /// plugin's pipeline must stay renderable).
    #[test]
    fn unknown_name_falls_back_to_forward() {
        let registry = PipelineRegistry::default();
        let pipeline = registry.resolve("portal").expect("fallback exists");
        assert_eq!(pipeline.name(), FORWARD_PIPELINE);
        assert!(
            registry.get("portal").is_none(),
            "fallback must not register"
        );
    }

    /// Registration resolves by the pipeline's own name and replaces a
    /// previous registration of that name.
    #[test]
    fn registration_resolves_and_replaces() {
        let mut registry = PipelineRegistry::default();
        registry.register(Arc::new(NullPipeline("portal")));
        assert_eq!(registry.resolve("portal").unwrap().name(), "portal");

        // Re-registering a name replaces the implementation.
        let replacement: Arc<dyn CameraRenderPipeline> = Arc::new(NullPipeline("portal"));
        let ptr = Arc::as_ptr(&replacement);
        registry.register(replacement);
        assert!(std::ptr::eq(
            Arc::as_ptr(registry.get("portal").unwrap()),
            ptr
        ));
    }
}
