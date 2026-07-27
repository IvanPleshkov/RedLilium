//! Render systems — they run in the [`Render`](crate::Render) schedule and
//! contribute passes to the frame graph held in the [`RenderSchedule`] resource.

use std::sync::Arc;

use redlilium_core::math::mat4_to_cols_array_2d;
use redlilium_graphics::{PassHandle, RenderTarget};

use redlilium_graphics::egui::EguiController;

use redlilium_assets::{AssetDb, AssetProcessor};

use crate::std::components::Camera;
use crate::system::SystemError;
use crate::{DebugDrawer, DebugDrawerRenderer, ExclusiveSystem, System, SystemContext, World};

use super::pipeline::{CameraRenderPipeline, CameraView, PipelineRegistry, RecordCtx};
use super::scene_drawer::VisibleScene;
use super::{
    CameraOutput, CameraTarget, CameraTargetSpec, EnvironmentManager, FORWARD_PIPELINE, FrameRing,
    MainViewport, MaterialAssetManager, MaterialInstanceManager, MeshManager, PipelineCache,
    RenderPath, RenderSchedule, ShaderManager, ShadingRegistry, SizePolicy, TemporalJitter,
    TemporalState, TextureManager, VertexLayoutManager, jitter_pixels,
};

/// Apply a sub-pixel jitter to a view-projection: a clip-space translation
/// `T(jx, jy) * VP` — post-projection, so every object shifts by the same
/// on-screen amount regardless of depth (jittering the camera *position*
/// would parallax near geometry). Offsets are in NDC units.
fn apply_jitter(view_projection: &redlilium_core::math::Mat4, jx: f32, jy: f32) -> [[f32; 4]; 4] {
    let mut translation = redlilium_core::math::Mat4::identity();
    translation[(0, 3)] = jx;
    translation[(1, 3)] = jy;
    mat4_to_cols_array_2d(&(translation * view_projection))
}

/// Holds the scene's main pass handle so other passes (an egui overlay,
/// debug lines) can depend on it. Written by [`CameraRender`] each frame (set to
/// `None` if it produced no pass).
#[derive(Default)]
pub struct ScenePass(pub Option<PassHandle>);

/// Holds this frame's egui overlay pass handle, reset by [`EguiRender`] each
/// frame (handles are graph-local, so a stale one must never leak across
/// frames). Optional resource: passes appended AFTER the Render schedule that
/// write a texture egui samples — the editor's selection outline writing the
/// scene camera's target — use it to order themselves before the egui draw.
#[derive(Default)]
pub struct EguiPass(pub Option<PassHandle>);

/// Derives each camera's GPU render target from its serializable
/// [`CameraOutput`] spec (ADR-029, #74).
///
/// For every entity with `Camera + CameraOutput`, (re)creates the runtime-only
/// [`CameraTarget`] whenever it disagrees with the spec (missing, wrong size,
/// or stale clear color), and publishes `Offscreen { output: Some(guid) }`
/// color textures as virtual texture assets
/// ([`TextureManager::publish_virtual`]) so materials can sample them.
///
/// Cameras **without** `CameraOutput` are untouched — their targets stay
/// host-managed. After the color+depth derivation it invokes each
/// [`RenderPath`] camera's pipeline
/// [`ensure_targets`](CameraRenderPipeline::ensure_targets) hook so pipelines
/// derive their auxiliary targets (ADR-035) under the same discipline. Runs
/// as an exclusive barrier (it inserts components), ordered before
/// [`CameraRender`].
#[derive(Default)]
pub struct EnsureCameraTargets;

impl ExclusiveSystem for EnsureCameraTargets {
    type Result = ();
    fn run(&mut self, world: &mut World) -> Result<Self::Result, SystemError> {
        use redlilium_graphics::{TextureDescriptor, TextureFormat, TextureUsage};

        if !world.has_resource::<TextureManager>() {
            return Ok(());
        }
        // The temporal contract's history resource (#147) — ensured here (an
        // exclusive barrier every render host already runs) so CameraRender
        // and VisibleScene::gather can rely on it without host wiring.
        if !world.has_resource::<TemporalState>() {
            world.insert_resource(TemporalState::default());
        }
        let viewport = world
            .has_resource::<MainViewport>()
            .then(|| *world.resource::<MainViewport>());

        /// One camera whose derived target disagrees with its spec.
        enum Work {
            /// (Re)create the textures at this size.
            Recreate {
                entity: crate::Entity,
                width: u32,
                height: u32,
                clear: [f32; 4],
                format: TextureFormat,
                publish: Option<redlilium_assets::Guid>,
            },
            /// Textures are fine; only the clear color changed.
            Reclear {
                entity: crate::Entity,
                clear: [f32; 4],
            },
        }

        // Phase 1: diff specs against derived targets under read borrows.
        let mut work: Vec<Work> = Vec::new();
        {
            let Ok(outputs) = world.read_all::<CameraOutput>() else {
                return Ok(());
            };
            for (idx, out) in outputs.iter() {
                let Some(entity) = world.entity_at_index(idx) else {
                    continue;
                };
                if world.get::<Camera>(entity).is_none() {
                    continue;
                }
                let (size, publish) = match &out.target {
                    CameraTargetSpec::Screen => (SizePolicy::Viewport, None),
                    CameraTargetSpec::Offscreen { size, output } => (*size, *output),
                };
                let (width, height) = match size {
                    SizePolicy::Viewport => {
                        let Some(v) = viewport else { continue };
                        (v.width, v.height)
                    }
                    SizePolicy::ViewportScale(scale) => {
                        let Some(v) = viewport else { continue };
                        (
                            (v.width as f32 * scale) as u32,
                            (v.height as f32 * scale) as u32,
                        )
                    }
                    SizePolicy::Fixed(w, h) => (w, h),
                };
                let (width, height) = (width.max(1), height.max(1));
                let format = out.format.color_format();

                match world.get::<CameraTarget>(entity) {
                    Some(target)
                        if target.color.width() == width
                            && target.color.height() == height
                            && target.color.format() == format =>
                    {
                        if target.clear_color != out.clear_color {
                            work.push(Work::Reclear {
                                entity,
                                clear: out.clear_color,
                            });
                        }
                    }
                    _ => work.push(Work::Recreate {
                        entity,
                        width,
                        height,
                        clear: out.clear_color,
                        format,
                        publish,
                    }),
                }
            }
        }
        // Phase 2: create textures and update components.
        let device = world.resource::<TextureManager>().device().clone();
        for item in work {
            match item {
                Work::Reclear { entity, clear } => {
                    if let Some(target) = world.get::<CameraTarget>(entity) {
                        let updated =
                            CameraTarget::new(target.color.clone(), target.depth.clone(), clear);
                        let _ = world.insert(entity, updated);
                    }
                }
                Work::Recreate {
                    entity,
                    width,
                    height,
                    clear,
                    format,
                    publish,
                } => {
                    // Engine-standard formats (see CameraOutput docs). The
                    // color target is sampleable (the host composites it,
                    // offscreen outputs are consumed as virtual textures) and
                    // copyable (screenshots / readbacks — same flags as the
                    // editor's scene-view target).
                    let color = device.create_texture(
                        &TextureDescriptor::new_2d(
                            width,
                            height,
                            format,
                            TextureUsage::RENDER_ATTACHMENT
                                | TextureUsage::TEXTURE_BINDING
                                | TextureUsage::COPY_SRC,
                        )
                        .with_label("camera_color"),
                    );
                    let depth = device.create_texture(
                        &TextureDescriptor::new_2d(
                            width,
                            height,
                            TextureFormat::Depth32Float,
                            // TEXTURE_BINDING: the deferred path reconstructs
                            // world position from the depth buffer (resolve,
                            // SSAO, TAA, motion blur sample it via Load).
                            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                        )
                        .with_label("camera_depth"),
                    );
                    let (color, depth) = match (color, depth) {
                        (Ok(c), Ok(d)) => (c, d),
                        (c, d) => {
                            log::warn!(
                                "EnsureCameraTargets: target creation failed \
                                 ({width}x{height}): {:?}",
                                c.err().or(d.err())
                            );
                            continue;
                        }
                    };
                    if let Some(guid) = publish
                        && let Err(e) = world
                            .resource_mut::<TextureManager>()
                            .publish_virtual(guid, color.clone())
                    {
                        log::warn!("EnsureCameraTargets: publish_virtual({guid:?}) failed: {e}");
                    }
                    let _ = world.insert(entity, CameraTarget::new(color, depth, clear));
                }
            }
        }

        // ADR-035: pipelines derive their auxiliary targets (PipelineTargets)
        // after the color+depth pair exists. Resolve under read borrows, then
        // hand each pipeline the world exclusively.
        if world.has_resource::<PipelineRegistry>() {
            let hooks: Vec<(crate::Entity, Arc<dyn CameraRenderPipeline>)> = {
                let Ok(paths) = world.read_all::<RenderPath>() else {
                    return Ok(());
                };
                let registry = world.resource::<PipelineRegistry>();
                paths
                    .iter()
                    .filter_map(|(idx, path)| {
                        let entity = world.entity_at_index(idx)?;
                        world.get::<Camera>(entity)?;
                        Some((entity, registry.resolve(&path.pipeline)?))
                    })
                    .collect()
            };
            for (entity, pipeline) in hooks {
                pipeline.ensure_targets(world, entity);
            }
        }
        Ok(())
    }
}

/// Flushes the GPU-upload managers' pending transfers into the frame graph
/// (deferred, graph-ordered — never a synchronous write). Passes that read the
/// uploaded meshes/textures/materials should depend on this system so their
/// transfer pass is scheduled first.
pub struct FlushUploads;

impl System for FlushUploads {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        let mut schedule = world.resource_mut::<RenderSchedule>();
        let Some(graph) = schedule.graph_mut() else {
            return Ok(()); // no frame graph bound (not in the render bracket)
        };
        // Mesh and texture uploads flow through the asset pipeline (the loaders'
        // GPU stages + AssetGpuFlush), so only the material manager flushes here.
        // Asset-based material instances flush their static property buffers here.
        if world.has_resource::<MaterialInstanceManager>() {
            world
                .resource_mut::<MaterialInstanceManager>()
                .flush_uploads(graph);
        }
        Ok(())
    }
}

/// The per-camera render dispatcher (ADR-035, #128): resolves each camera's
/// [`RenderPath`] against the [`PipelineRegistry`] and has the pipeline record
/// the camera's passes into the frame graph, then records the primary main
/// pass handle in [`ScenePass`] so dependent passes (egui, debug) can order
/// after it.
///
/// Camera + CameraTarget are read via `read_all` (the editor camera is
/// EDITOR-flagged, which the filtered `read` iterator skips). The frame's
/// visible renderables are gathered **once** ([`VisibleScene::gather`]) and
/// every recorded view replays the prepared list.
#[derive(Default)]
pub struct CameraRender;

impl System for CameraRender {
    /// The scene's main pass handle (so a dependent render system can order
    /// after it via
    /// [`SystemContext::system_result`](crate::SystemContext::system_result)).
    type Result = Option<PassHandle>;
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<RenderSchedule>()
            || !world.has_resource::<FrameRing>()
            || !world.has_resource::<ScenePass>()
            || !world.has_resource::<PipelineCache>()
            || !world.has_resource::<PipelineRegistry>()
        {
            return Ok(None);
        }
        world.resource_mut::<ScenePass>().0 = None;

        // Rotate the temporal history exactly once per frame (#147): current
        // matrices become "previous" for velocity; the frame index advances
        // the jitter sequence.
        let temporal_frame = world.has_resource::<TemporalState>().then(|| {
            let mut state = world.resource_mut::<TemporalState>();
            state.begin_frame();
            state.frame()
        });

        // Hand this frame its own region of the uniform ring, once per frame and
        // before any pass records a push. Every `FrameRing::push` in the engine
        // runs underneath this dispatcher (both pipelines' `record` and
        // `VisibleScene::gather`), so this is the one place that has to know.
        world.resource_mut::<FrameRing>().begin_frame();

        // Every entity with a Camera AND a CameraTarget renders (ADR-029):
        // its RenderPath's pipeline records into its own target. Offscreen
        // cameras are emitted first, the primary (screen) camera last — so
        // ScenePass, the handle overlays (debug lines, egui) attach to, is
        // the last writer of the surface the host composites. Cross-camera
        // (and cross-pass) ordering by resource use is derived by the graph
        // compiler automatically.
        struct PlannedView {
            view: CameraView,
            pipeline: Arc<dyn CameraRenderPipeline>,
        }
        let mut views: Vec<PlannedView> = {
            let (Ok(cams), Ok(targets)) =
                (world.read_all::<Camera>(), world.read_all::<CameraTarget>())
            else {
                return Ok(None);
            };
            let outputs = world.read_all::<CameraOutput>().ok();
            let paths = world.read_all::<RenderPath>().ok();
            let jitters = world.read_all::<TemporalJitter>().ok();
            let registry = world.resource::<PipelineRegistry>();
            targets
                .iter()
                .filter_map(|(idx, target)| {
                    let cam = cams.get(idx)?;
                    let entity = world.entity_at_index(idx)?;
                    // Primary = renders the surface the host composites: a
                    // Screen spec, or a host-managed target (no spec at all —
                    // the editor's scene view).
                    let primary = outputs
                        .as_ref()
                        .and_then(|o| o.get(idx))
                        .is_none_or(|out| matches!(out.target, CameraTargetSpec::Screen));
                    // No RenderPath component = the forward default (the same
                    // defaulting discipline as CameraOutput::screen()).
                    let name = paths
                        .as_ref()
                        .and_then(|p| p.get(idx))
                        .map_or(FORWARD_PIPELINE, |path| path.pipeline.as_str());
                    let pipeline = registry.resolve(name)?;

                    // The temporal contract's camera matrices (#147): raster
                    // uses the (optionally jittered) view-projection; the
                    // unjittered pair feeds velocity. Cameras without
                    // TemporalJitter rasterize the unjittered matrix —
                    // bit-identical to the pre-temporal path.
                    let unjittered = mat4_to_cols_array_2d(&cam.view_projection());
                    let (jittered, prev) =
                        match (temporal_frame, world.has_resource::<TemporalState>()) {
                            (Some(frame), true) => {
                                let mut state = world.resource_mut::<TemporalState>();
                                let prev = state.prev_view_proj(entity).unwrap_or(unjittered);
                                state.record_view_proj(entity, unjittered);
                                let jittered = jitters
                                    .as_ref()
                                    .and_then(|j| j.get(idx))
                                    .map(|jitter| {
                                        let size = target.color.size();
                                        let [jx, jy] = jitter_pixels(frame, jitter.cycle);
                                        apply_jitter(
                                            &cam.view_projection(),
                                            jx * 2.0 / size.width.max(1) as f32,
                                            jy * 2.0 / size.height.max(1) as f32,
                                        )
                                    })
                                    .unwrap_or(unjittered);
                                (jittered, prev)
                            }
                            _ => (unjittered, unjittered),
                        };
                    Some(PlannedView {
                        view: CameraView {
                            entity,
                            view_projection: jittered,
                            view_projection_unjittered: unjittered,
                            prev_view_projection: prev,
                            target: target.clone(),
                            primary,
                        },
                        pipeline,
                    })
                })
                .collect()
        };
        if views.is_empty() {
            return Ok(None);
        }
        views.sort_by_key(|planned| planned.view.primary);

        // Gather the frame's visible set once; every view records from it.
        let Some(scene) = VisibleScene::gather(world) else {
            return Ok(None);
        };

        // Take the graph out of the resource so pipelines can freely borrow
        // world resources (FrameRing, PipelineCache) while recording into it.
        let Some(mut graph) = world.resource_mut::<RenderSchedule>().take() else {
            return Ok(None);
        };
        let record_ctx = RecordCtx {
            world,
            scene: &scene,
        };
        // ScenePass (and this system's result) is the primary camera's main
        // pass — the one overlays attach to and the host composites.
        let mut primary_handle = None;
        for planned in &views {
            let handle = planned
                .pipeline
                .record(&record_ctx, &planned.view, &mut graph);
            if let Some(handle) = handle
                && (planned.view.primary || primary_handle.is_none())
            {
                primary_handle = Some(handle);
            }
        }
        world.resource_mut::<RenderSchedule>().set(graph);

        if let Some(handle) = primary_handle {
            world.resource_mut::<ScenePass>().0 = Some(handle);
        }
        Ok(primary_handle)
    }
}

/// Renders debug-drawer lines as a separate pass that loads the camera's
/// [`CameraTarget`] and is ordered after the scene's main pass — whose handle it
/// reads the native way, via [`system_result`](SystemContext::system_result)
/// (requires a `CameraRender -> DebugRender` edge). Updates [`ScenePass`] to its
/// own handle so it becomes the CameraTarget's last writer (egui depends on that).
pub struct DebugRender;

impl System for DebugRender {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        // The scene pass we order after (its handle comes from system_result).
        let Some(scene_handle) = *ctx.system_result::<CameraRender>() else {
            return Ok(());
        };
        if !world.has_resource::<DebugDrawer>()
            || !world.has_resource::<DebugDrawerRenderer>()
            || !world.has_resource::<RenderSchedule>()
        {
            return Ok(());
        }
        let vertices = world.resource::<DebugDrawer>().take_render_data();
        if vertices.is_empty() {
            return Ok(());
        }
        // The PRIMARY camera's target and matching view-projection, from the
        // same entity (the dispatcher's definition: a Screen spec, or no
        // CameraOutput at all — the editor's scene view). Taking the first
        // arbitrary CameraTarget would draw the lines into an offscreen
        // camera's target with a mismatched matrix when several cameras exist.
        let picked = world.read_all::<CameraTarget>().ok().and_then(|targets| {
            let cams = world.read_all::<Camera>().ok()?;
            let outputs = world.read_all::<CameraOutput>().ok();
            let mut fallback = None;
            for (idx, target) in targets.iter() {
                let Some(cam) = cams.get(idx) else {
                    continue;
                };
                let entry = (
                    target.color.clone(),
                    target.depth.clone(),
                    mat4_to_cols_array_2d(&cam.view_projection()),
                );
                let primary = outputs
                    .as_ref()
                    .and_then(|o| o.get(idx))
                    .is_none_or(|out| matches!(out.target, CameraTargetSpec::Screen));
                if primary {
                    return Some(entry);
                }
                if fallback.is_none() {
                    fallback = Some(entry);
                }
            }
            fallback
        });
        let Some((color, depth, vp)) = picked else {
            return Ok(());
        };

        let debug_handle = {
            let mut renderer = world.resource_mut::<DebugDrawerRenderer>();
            renderer.update_view_proj(vp);
            let rt = RenderTarget::from_texture(color);
            let Some(pass) = renderer.create_graphics_pass(&vertices, &rt, Some(&depth)) else {
                return Ok(());
            };
            let mut schedule = world.resource_mut::<RenderSchedule>();
            schedule.graph_mut().map(|graph| {
                let h = graph.add_graphics_pass(pass);
                graph.add_dependency(h, scene_handle);
                h
            })
        };
        if let Some(h) = debug_handle {
            world.resource_mut::<ScenePass>().0 = Some(h);
        }
        Ok(())
    }
}

/// The frame's final (swapchain) render target plus its size, set by the app each
/// frame BEFORE running the [`Render`](crate::Render) schedule. Render systems
/// that composite to the screen — currently [`EguiRender`] — read it. `RenderTarget`
/// is owned + Send + Sync, so this lives in the world; the app replaces it each
/// frame. (Off-screen passes target a [`CameraTarget`] texture instead.)
pub struct FrameTarget {
    pub target: RenderTarget,
    pub width: u32,
    pub height: u32,
}

/// Finishes the egui frame (the app calls `begin_frame` + builds UI before the
/// Render schedule) and contributes its draw pass to the frame graph, targeting
/// [`FrameTarget`] (the swapchain). Edge `DebugRender -> EguiRender` orders it last
/// so it can depend on the CameraTarget's last writer ([`ScenePass`]) — an egui
/// overlay samples that texture, so its draw must come after the scene/debug passes.
///
/// The egui controller is the [`EguiController`] resource (see the editor's
/// `on_init`); this system is a no-op when any of its inputs are absent.
pub struct EguiRender;

impl System for EguiRender {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        // Handles are graph-local: clear before any early return so a stale
        // handle never leaks into a later frame's graph.
        if world.has_resource::<EguiPass>() {
            world.resource_mut::<EguiPass>().0 = None;
        }
        if !world.has_resource::<EguiController>()
            || !world.has_resource::<FrameTarget>()
            || !world.has_resource::<RenderSchedule>()
        {
            return Ok(());
        }
        // The pass egui must order after (the CameraTarget's last writer), if any.
        let scene_handle = world
            .has_resource::<ScenePass>()
            .then(|| world.resource::<ScenePass>().0)
            .flatten();

        let mut egui = world.resource_mut::<EguiController>();
        let pass = {
            let ft = world.resource::<FrameTarget>();
            egui.end_frame(&ft.target, ft.width, ft.height)
        };
        let mut schedule = world.resource_mut::<RenderSchedule>();
        let mut egui_handle = None;
        if let Some(graph) = schedule.graph_mut() {
            // Atlas uploads first (graph-ordered before the egui draw).
            egui.flush_uploads(graph);
            if let Some(pass) = pass {
                let handle = graph.add_graphics_pass(pass);
                if let Some(scene_handle) = scene_handle {
                    graph.add_dependency(handle, scene_handle);
                }
                egui_handle = Some(handle);
            }
        }
        drop(schedule);
        if egui_handle.is_some() && world.has_resource::<EguiPass>() {
            world.resource_mut::<EguiPass>().0 = egui_handle;
        }
        Ok(())
    }
}

/// Drives mesh loading and syncs component [`AssetRef`](redlilium_assets::AssetRef)s.
///
/// First advances every in-flight [`MeshManager`] load (resolving each one's
/// shared vertex layout via [`VertexLayoutManager`]). Then scans the asset refs
/// of **every registered component** (via
/// [`World::scan_asset_refs`] — the derive-generated
/// [`Component::visit_asset_refs`](crate::Component::visit_asset_refs) hooks)
/// and resolves them against the managers: demand-driven (an unresolved source
/// gets requested) and reload-aware (a ref whose `Arc` no longer matches the
/// resident one is re-resolved; `Arc` pointer identity is the version). Writes
/// go through `Mut`, so a re-resolve marks the component dirty and stateful
/// consumers can react through ordinary change detection.
///
/// A user component with an `AssetRef` field gets loading + hot reload with no
/// extra code. No-op if any input is absent.
///
/// The scan is gen-gated: while every manager's `generation()` is unchanged
/// (no resident asset appeared or reloaded), only components changed since the
/// previous scan are visited — new/edited refs still get demand-requested, but
/// an idle scene costs nothing per frame. Any generation bump falls back to a
/// full walk (a reload must re-resolve refs anywhere in the world).
///
/// Syncs asset references across all components as asset managers change their
/// state. Runs as an exclusive barrier to avoid contending with component-writing
/// systems (e.g., `UpdateGlobalTransforms`) under the multi-threaded runner.
#[derive(Default)]
pub struct MeshLoad {
    /// Manager generations at the previous scan (mesh, instance, texture,
    /// environment). Used to gate: unchanged generations → skip the expensive
    /// scan pass.
    last_gens: Option<[u64; 4]>,
}

impl ExclusiveSystem for MeshLoad {
    type Result = ();
    fn run(&mut self, world: &mut World) -> Result<Self::Result, SystemError> {
        use redlilium_assets::AssetRef;

        use super::loaders::{
            EnvironmentSource, MaterialInstanceSource, MeshSource, TextureSource,
        };

        if !world.has_resource::<MeshManager>()
            || !world.has_resource::<VertexLayoutManager>()
            || !world.has_resource::<AssetProcessor>()
            || !world.has_resource::<AssetDb>()
        {
            return Ok(());
        }
        let mut mesh_mgr = world.resource_mut::<MeshManager>();
        {
            let mut layout_mgr = world.resource_mut::<VertexLayoutManager>();
            let mut processor = world.resource_mut::<AssetProcessor>();
            let db = world.resource::<AssetDb>();
            mesh_mgr.drive(&mut processor, &db, &mut layout_mgr);
        }

        let mut instance_mgr = world
            .has_resource::<MaterialInstanceManager>()
            .then(|| world.resource_mut::<MaterialInstanceManager>());
        let mut texture_mgr = world
            .has_resource::<TextureManager>()
            .then(|| world.resource_mut::<TextureManager>());
        let mut env_mgr = world
            .has_resource::<EnvironmentManager>()
            .then(|| world.resource_mut::<EnvironmentManager>());

        let gens = [
            mesh_mgr.generation(),
            instance_mgr.as_ref().map_or(0, |m| m.generation()),
            texture_mgr.as_ref().map_or(0, |m| m.generation()),
            env_mgr.as_ref().map_or(0, |m| m.generation()),
        ];
        let changed = self.last_gens != Some(gens);
        self.last_gens = Some(gens);

        if !changed {
            return Ok(());
        }

        let mut stale: Vec<(&'static str, u32)> = Vec::new();
        world.scan_asset_refs(None, &mut |component, idx, any| {
            if let Some(r) = any.downcast_ref::<AssetRef<MeshSource>>() {
                match mesh_mgr.get(r.source()) {
                    Some(mesh) if !r.is_current(mesh) => stale.push((component, idx)),
                    Some(_) => {}
                    None => mesh_mgr.request(r.source()),
                }
            } else if let Some(r) = any.downcast_ref::<AssetRef<MaterialInstanceSource>>()
                && let Some(instance_mgr) = instance_mgr.as_mut()
            {
                match instance_mgr.get(r.source().guid) {
                    Some(instance) if !r.is_current(instance) => stale.push((component, idx)),
                    Some(_) => {}
                    None => instance_mgr.request(r.source()),
                }
            } else if let Some(r) = any.downcast_ref::<AssetRef<TextureSource>>()
                && let Some(texture_mgr) = texture_mgr.as_mut()
            {
                match texture_mgr.get(r.source()) {
                    Some(texture) if !r.is_current(texture) => stale.push((component, idx)),
                    Some(_) => {}
                    None => texture_mgr.request(r.source()),
                }
            } else if let Some(r) = any.downcast_ref::<AssetRef<EnvironmentSource>>()
                && let Some(env_mgr) = env_mgr.as_mut()
            {
                match env_mgr.get(r.source().guid) {
                    Some(env) if !r.is_current(env) => stale.push((component, idx)),
                    Some(_) => {}
                    None => env_mgr.request(r.source()),
                }
            }
        });

        stale.sort_unstable();
        stale.dedup();
        for (component, idx) in stale {
            world.patch_asset_refs(component, idx, &mut |any| {
                if let Some(r) = any.downcast_mut::<AssetRef<MeshSource>>() {
                    if let Some(mesh) = mesh_mgr.get(r.source())
                        && !r.is_current(mesh)
                    {
                        r.resolve(mesh.clone());
                    }
                } else if let Some(r) = any.downcast_mut::<AssetRef<MaterialInstanceSource>>()
                    && let Some(instance_mgr) = instance_mgr.as_mut()
                    && let Some(instance) = instance_mgr.get(r.source().guid)
                    && !r.is_current(instance)
                {
                    r.resolve(instance.clone());
                } else if let Some(r) = any.downcast_mut::<AssetRef<TextureSource>>()
                    && let Some(texture_mgr) = texture_mgr.as_mut()
                    && let Some(texture) = texture_mgr.get(r.source())
                    && !r.is_current(texture)
                {
                    r.resolve(texture.clone());
                } else if let Some(r) = any.downcast_mut::<AssetRef<EnvironmentSource>>()
                    && let Some(env_mgr) = env_mgr.as_mut()
                    && let Some(env) = env_mgr.get(r.source().guid)
                    && !r.is_current(env)
                {
                    r.resolve(env.clone());
                }
            });
        }
        Ok(())
    }
}

/// Drives material-instance loading: advances every in-flight
/// [`MaterialInstanceManager`] request — loading its data, resolving its parent
/// template (via [`MaterialAssetManager`], which itself pulls the shader through
/// [`ShaderManager`]), and building the static property binding. Co-locks the
/// managers + [`ShadingRegistry`] + processor + DB so consumers only ever touch
/// `MaterialInstanceManager::request`. No-op if any input is absent.
pub struct MaterialInstanceLoad;

impl System for MaterialInstanceLoad {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<MaterialInstanceManager>()
            || !world.has_resource::<MaterialAssetManager>()
            || !world.has_resource::<ShaderManager>()
            || !world.has_resource::<TextureManager>()
            || !world.has_resource::<ShadingRegistry>()
            || !world.has_resource::<AssetProcessor>()
            || !world.has_resource::<AssetDb>()
        {
            return Ok(());
        }
        let mut instance_mgr = world.resource_mut::<MaterialInstanceManager>();
        let mut material_mgr = world.resource_mut::<MaterialAssetManager>();
        let mut shader_mgr = world.resource_mut::<ShaderManager>();
        let mut texture_mgr = world.resource_mut::<TextureManager>();
        let registry = world.resource::<ShadingRegistry>();
        let mut processor = world.resource_mut::<AssetProcessor>();
        let db = world.resource::<AssetDb>();
        // Textures first, so instances demanding them this frame poll fresh state.
        texture_mgr.drive(&mut processor, &db);
        instance_mgr.drive(
            &mut processor,
            &db,
            &mut material_mgr,
            &mut shader_mgr,
            &mut texture_mgr,
            &registry,
        );
        // IBL environments (#145) resolve against the same textures — drive
        // them here too, while the texture manager is already locked.
        if world.has_resource::<EnvironmentManager>() {
            world
                .resource_mut::<EnvironmentManager>()
                .drive(&mut processor, &db, &mut texture_mgr);
        }
        Ok(())
    }
}

/// Hot reload: drains [`ChangedAssets`](super::ChangedAssets) and invalidates
/// the **owning** manager per guid (routed by the DB record's kind). Everything
/// downstream catches up by itself — dependent managers pull-validate their
/// input `Arc`s, the pipeline cache revalidates its shader, and the `MeshLoad`
/// sync re-resolves component refs (`docs/ASSETS.md` §9). Consumers keep serving
/// the last-good resource while the new one loads.
pub struct HotReload;

impl System for HotReload {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<super::ChangedAssets>() || !world.has_resource::<AssetDb>() {
            return Ok(());
        }
        let changed = world.resource_mut::<super::ChangedAssets>().drain();
        if changed.is_empty() {
            return Ok(());
        }
        let db = world.resource::<AssetDb>();
        for guid in changed {
            let Some(record) = db.record(&guid) else {
                continue; // not a registered asset (e.g. the db file itself)
            };
            log::info!("hot reload: {} ({guid:?})", record.path.path);
            match record.kind.as_str() {
                "mesh" if world.has_resource::<MeshManager>() => {
                    world.resource_mut::<MeshManager>().invalidate_file(guid);
                }
                "vertex_layout" if world.has_resource::<VertexLayoutManager>() => {
                    world.resource_mut::<VertexLayoutManager>().invalidate(guid);
                }
                "texture" if world.has_resource::<TextureManager>() => {
                    world.resource_mut::<TextureManager>().invalidate_file(guid);
                }
                "shader" if world.has_resource::<ShaderManager>() => {
                    world.resource_mut::<ShaderManager>().invalidate(guid);
                }
                "material" if world.has_resource::<MaterialAssetManager>() => {
                    world
                        .resource_mut::<MaterialAssetManager>()
                        .invalidate(guid);
                }
                "material_instance" if world.has_resource::<MaterialInstanceManager>() => {
                    world
                        .resource_mut::<MaterialInstanceManager>()
                        .invalidate(guid);
                }
                "environment" if world.has_resource::<EnvironmentManager>() => {
                    world.resource_mut::<EnvironmentManager>().invalidate(guid);
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod camera_target_tests {
    use super::*;
    use crate::std::rendering::loaders::TextureSource;

    fn world_with_texture_manager() -> World {
        let instance = redlilium_graphics::GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("device");
        let mut world = World::new();
        world.register_component::<Camera>();
        world.register_component::<CameraOutput>();
        world.register_component::<CameraTarget>();
        world.insert_resource(TextureManager::new(device));
        world
    }

    fn run(world: &mut World) {
        EnsureCameraTargets
            .run(world)
            .expect("EnsureCameraTargets runs");
    }

    /// ADR-029: a Screen spec derives a viewport-sized CameraTarget, and a
    /// viewport change re-derives it at the new size.
    #[test]
    fn screen_target_follows_main_viewport() {
        let mut world = world_with_texture_manager();
        world.insert_resource(MainViewport::new(64, 32));
        let camera = world.spawn();
        world
            .insert(camera, Camera::perspective(1.0, 2.0, 0.1, 100.0))
            .unwrap();
        world.insert(camera, CameraOutput::screen()).unwrap();

        run(&mut world);
        {
            let target = world.get::<CameraTarget>(camera).expect("target derived");
            assert_eq!((target.color.width(), target.color.height()), (64, 32));
            assert_eq!((target.depth.width(), target.depth.height()), (64, 32));
        }

        // Steady state: same spec + same viewport → same textures (no churn).
        let before = world
            .get::<CameraTarget>(camera)
            .map(|t| Arc::as_ptr(&t.color))
            .unwrap();
        run(&mut world);
        let after = world
            .get::<CameraTarget>(camera)
            .map(|t| Arc::as_ptr(&t.color))
            .unwrap();
        assert_eq!(before, after, "unchanged spec must not recreate textures");

        // Resize re-derives.
        world.insert_resource(MainViewport::new(128, 128));
        run(&mut world);
        let target = world
            .get::<CameraTarget>(camera)
            .expect("target re-derived");
        assert_eq!((target.color.width(), target.color.height()), (128, 128));
    }

    /// A clear-color edit updates the derived target without recreating the
    /// GPU textures.
    #[test]
    fn clear_color_edit_keeps_textures() {
        let mut world = world_with_texture_manager();
        world.insert_resource(MainViewport::new(16, 16));
        let camera = world.spawn();
        world
            .insert(camera, Camera::perspective(1.0, 1.0, 0.1, 100.0))
            .unwrap();
        world.insert(camera, CameraOutput::screen()).unwrap();
        run(&mut world);
        let before = world
            .get::<CameraTarget>(camera)
            .map(|t| Arc::as_ptr(&t.color))
            .unwrap();

        world
            .insert(
                camera,
                CameraOutput::screen().with_clear_color([1.0, 0.0, 0.0, 1.0]),
            )
            .unwrap();
        run(&mut world);
        let target = world.get::<CameraTarget>(camera).unwrap();
        assert_eq!(target.clear_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(Arc::as_ptr(&target.color), before, "textures reused");
    }

    /// An Offscreen spec with an `output` guid publishes the color texture as
    /// a virtual texture asset (resolvable via `TextureSource::Virtual`), and
    /// a resize re-publishes the new texture.
    #[test]
    fn offscreen_output_publishes_virtual_texture() {
        let mut world = world_with_texture_manager();
        let guid = redlilium_assets::Guid::stable("test/mirror_output");
        let camera = world.spawn();
        world
            .insert(camera, Camera::perspective(1.0, 1.0, 0.1, 100.0))
            .unwrap();
        world
            .insert(
                camera,
                CameraOutput::offscreen(SizePolicy::Fixed(32, 32), Some(guid)),
            )
            .unwrap();

        // No MainViewport needed for Fixed sizing.
        run(&mut world);
        let color = world
            .get::<CameraTarget>(camera)
            .map(|t| t.color.clone())
            .expect("offscreen target derived");
        {
            let textures = world.resource::<TextureManager>();
            let resolved = textures
                .get(&TextureSource::Virtual(guid))
                .expect("virtual texture published");
            assert!(Arc::ptr_eq(&resolved.texture, &color));
        }
        let gen_before = world.resource::<TextureManager>().generation();

        // Resize → new texture published under the same identity.
        world
            .insert(
                camera,
                CameraOutput::offscreen(SizePolicy::Fixed(64, 64), Some(guid)),
            )
            .unwrap();
        run(&mut world);
        let textures = world.resource::<TextureManager>();
        let resolved = textures
            .get(&TextureSource::Virtual(guid))
            .expect("still published");
        assert_eq!(resolved.texture.width(), 64);
        assert!(
            textures.generation() > gen_before,
            "re-publish must bump the generation so AssetRef holders re-resolve"
        );
    }

    /// Cameras without a CameraOutput spec are host-managed: the system must
    /// leave them alone (the editor's scene view relies on this).
    #[test]
    fn cameras_without_spec_are_untouched() {
        let mut world = world_with_texture_manager();
        world.insert_resource(MainViewport::new(16, 16));
        let camera = world.spawn();
        world
            .insert(camera, Camera::perspective(1.0, 1.0, 0.1, 100.0))
            .unwrap();
        run(&mut world);
        assert!(world.get::<CameraTarget>(camera).is_none());
    }
}
