//! The editor core shared by both shells: the windowed editor
//! ([`crate::editor::Editor`]) and the headless one ([`crate::headless`]).
//!
//! Owns world construction (demo scene, asset system, schedules) and the
//! frame-tick primitives every shell runs: draining the [`ActionQueue`]
//! through the undo history and persisting dirty asset DBs. Shell-specific
//! concerns (egui, winit input, picking) stay in the shells.

use std::f32::consts::FRAC_PI_4;
use std::sync::Arc;

use redlilium_ecs::sync::RwLock;

use redlilium_assets::{AssetDb, AssetPath, Guid};
use redlilium_core::abstract_editor::{ActionQueue, DEFAULT_MAX_UNDO, EditActionHistory};
use redlilium_core::math::Vec3;
use redlilium_debug_drawer::DebugDrawer;
use redlilium_ecs::{
    AssetGpuFlush, AssetPump, Camera, CameraRender, DebugRender, DrawGrid, DrawSelectionAabb,
    EguiRender, Entity, FlushUploads, FrameRing, FreeFlyCamera, GameTime, GlobalTransform,
    GridConfig, HotReload, MaterialInstanceLoad, MaterialInstanceSource, MeshGenerator, MeshLoad,
    MeshRenderer, MeshSource, Name, PipelineRegistry, PostUpdate, Primitive, RealTime, Render,
    RenderSchedule, ScenePass, Schedules, Transform, Update, UpdateCameraMatrices,
    UpdateFreeFlyCamera, UpdateGlobalTransforms, Visibility, WindowInput, World,
    register_std_components,
};
use redlilium_runtime::EngineContext;

use crate::scene_view::SceneViewState;

/// An independent ECS world managed by the editor.
pub struct EditorWorld {
    pub world: World,
    pub schedules: Schedules,
    /// Undo/redo history for editor actions.
    pub history: EditActionHistory<World>,
    /// The editor camera entity (flagged as EDITOR).
    pub editor_camera: Entity,
    /// Handle to the WindowInput resource for updating from app events.
    pub window_input: Arc<RwLock<WindowInput>>,
    /// Handle to the DebugDrawer resource for advance_tick / take_render_data.
    pub debug_drawer: Arc<RwLock<DebugDrawer>>,
}

impl EditorWorld {
    /// Drain the [`ActionQueue`] and execute every action through the undo
    /// history. Run once per frame, before the ECS schedules, so `Changed<T>`
    /// filters see the mutations the same frame.
    pub fn drain_actions(&mut self) {
        let actions = self.world.resource::<ActionQueue<World>>().drain();
        for action in actions {
            log::debug!("history: executing '{}'", action.description());
            if let Err(e) = self.history.execute(action, &mut self.world) {
                log::warn!("Action failed: {e}");
            }
        }
    }

    /// Persist mounts dirtied by the undoable asset-edit actions to their
    /// local `<dir>/assets.db`. Shells with extra dirt sources (the browser's
    /// file operations) fold them in before calling.
    pub fn persist_dirty_mounts(&self, local_mounts: &[(&'static str, &'static str)]) {
        let dirty: Vec<String> = self
            .world
            .resource_mut::<redlilium_ecs::DirtyMounts>()
            .drain();
        for mount in dirty {
            match local_mounts.iter().find(|(name, _)| *name == mount) {
                Some((mount, dir)) => {
                    persist_mount_db(&self.world.resource::<AssetDb>(), mount, dir);
                }
                None => log::warn!("no persistence wired for mount '{mount}'"),
            }
        }
    }
}

/// The scene asset the editor's world content came from / was last saved to
/// (#2), as a VFS path `"mount/path"` (e.g. `"game/scenes/level1.scene"`).
/// A world resource: File > Save writes here without asking, and remote
/// `save_scene` defaults here; `None` until the first save/load picks a path.
#[derive(Default)]
pub struct CurrentScene(pub Option<String>);

/// What to build into a new editor world — the windowed shell wants the egui
/// overlay system, the headless shell wants the remote channel unconditionally.
pub struct EditorWorldParams {
    /// Insert the `RemoteTransport` resource and `RemoteServe` system.
    pub remote: bool,
    /// Register the `EguiRender` overlay system (windowed shell only).
    pub egui: bool,
}

/// Load each local mount's asset DB into the shared in-memory DB, then scan
/// the mount: files added/changed/removed while the editor was closed get
/// registered (routed by the loaders' extensions), rehashed, or dropped. A
/// non-empty delta is persisted right back.
///
/// Call once per editor session, before the first world is created — the
/// database lives in the persistent [`EngineContext`], not in a world.
pub fn scan_local_mounts(engine: &EngineContext, local_mounts: &[(&'static str, &'static str)]) {
    // Load every mount's DB first: load_mount_db locks the shared asset DB
    // internally, so it must not run under the write guard taken below.
    for &(mount, dir) in local_mounts {
        engine.load_mount_db(mount, dir);
    }
    let processor = engine.processor().read();
    let mut asset_db = engine.asset_db().write();
    for &(mount, dir) in local_mounts {
        match pollster::block_on(processor.scan(&mut asset_db, mount, None)) {
            Ok(report) => {
                let changed = report.added.len() + report.modified.len() + report.removed.len();
                log::info!(
                    "scanned mount '{mount}': +{} ~{} -{} ({} unchanged, {} unrouted)",
                    report.added.len(),
                    report.modified.len(),
                    report.removed.len(),
                    report.unchanged,
                    report.unrouted,
                );
                if changed > 0 {
                    persist_mount_db(&asset_db, mount, dir);
                }
            }
            Err(e) => log::error!("scan of mount '{mount}' failed: {e}"),
        }
    }
}

/// Create a new editor world with a simple demo scene.
///
/// Persistent state (GPU managers, asset DB/processor) comes from the
/// [`EngineContext`] as shared resources; everything created here is
/// per-world and dies with it (ADR-020).
pub fn create_editor_world(
    params: &EditorWorldParams,
    engine: &EngineContext,
    scene_view: &mut SceneViewState,
    aspect: f32,
) -> EditorWorld {
    let mut ew = create_editor_world_base(params, engine, scene_view);
    ew.editor_camera = spawn_editor_camera(&mut ew.world, aspect);
    spawn_demo_scene(&mut ew.world, engine);
    ew
}

/// [`create_editor_world`] minus the demo scene: editor camera only, for a
/// world whose content arrives later via [`instantiate_scene`]. The split
/// exists so the shells can host the game module (registering its component
/// types) **between** world creation and scene instantiation — instantiating
/// a scene that uses game components before `register_types` silently drops
/// them (restore skips unknown types by design).
pub fn create_editor_world_empty(
    params: &EditorWorldParams,
    engine: &EngineContext,
    scene_view: &mut SceneViewState,
    aspect: f32,
) -> EditorWorld {
    let mut ew = create_editor_world_base(params, engine, scene_view);
    ew.editor_camera = spawn_editor_camera(&mut ew.world, aspect);
    ew
}

/// Instantiate a `.scene` asset (#2) into `ew` and record `path` as the
/// world's [`CurrentScene`]. Call only after every plugin's `register_types`
/// ran against this world (see [`create_editor_world_empty`]). On
/// instantiation failure the world stays empty (not demo content — a broken
/// scene should look broken, not silently morph into the demo), with the
/// error logged.
pub fn instantiate_scene(
    ew: &mut EditorWorld,
    path: &str,
    scene: &redlilium_ecs::serialize::SerializedWorld,
) {
    match ew.world.deserialize_world_into(scene) {
        Ok(spawned) => {
            log::info!("opened scene '{path}' ({} entities)", spawned.len());
            // Fresh reference holders, steady-state managers — same reason
            // as scene transitions (#106).
            redlilium_ecs::rescan_asset_managers(&mut ew.world);
        }
        Err(e) => log::error!("failed to instantiate scene '{path}': {e}"),
    }
    ew.world.resource_mut::<CurrentScene>().0 = Some(path.to_string());
}

/// [`create_editor_world`] minus the editor camera and the demo scene: all
/// resources and schedules, **zero entities**. The game-reload path builds a
/// replacement world with this and then restores every entity (editor camera
/// included) from a whole-world snapshot — spawning anything here would
/// duplicate it on restore. `editor_camera` is [`Entity::DANGLING`] until the
/// caller resolves it.
pub fn create_editor_world_base(
    params: &EditorWorldParams,
    engine: &EngineContext,
    scene_view: &mut SceneViewState,
) -> EditorWorld {
    let mut world = World::new();
    register_std_components(&mut world);
    redlilium_ecs::register_rendering_components(&mut world);

    // Shared engine state: GPU resource managers, shading registry, pipeline
    // cache, ChangedAssets inbox, asset processor + database.
    engine.inject_into(&mut world);

    // Mounts with un-persisted asset-DB edits (written by the undoable
    // asset-edit actions); drained + persisted once per frame.
    world.insert_resource(redlilium_ecs::DirtyMounts::new());
    // The scene asset this world's content belongs to (#2); set by the
    // startup scene load / save / remote load_scene.
    world.insert_resource(CurrentScene::default());
    // Remote-control channel (docs/REMOTE.md): served by RemoteServe on
    // the IO runtime; the editor pumps commands each frame. Opt-in.
    if params.remote {
        world.insert_resource(redlilium_ecs::RemoteTransport::new(
            ".redlilium/editor.port",
        ));
    }

    // Holds the per-frame render graph while the `Render` schedule runs.
    world.insert_resource(RenderSchedule::empty());

    // Scene forward-pass dynamic-uniform ring (an ECS resource). Give the
    // scene view its buffer so per-primitive materials bind it; the buffer is
    // needed before any entity is created below.
    let frame_ring = FrameRing::new(scene_view.device(), 1 << 20, "scene_frame_ring")
        .expect("Failed to create scene frame ring");
    scene_view.set_frame_ring_buffer(frame_ring.buffer().clone());
    world.insert_resource(frame_ring);
    // The CameraRender dispatcher records its scene-pass handle here so the
    // egui overlay + debug pass can depend on it; the registry resolves each
    // camera's RenderPath (ADR-035).
    world.insert_resource(ScenePass::default());
    world.insert_resource(PipelineRegistry::default());

    // Insert WindowInput resource
    let window_input_handle = world.insert_resource(WindowInput::default());

    // Dual-clock time resources. The editing world's own game systems (if
    // any) tick RealTime/GameTime like any other world; Play runs a
    // completely separate game world (editor/src/play.rs) with its own clocks.
    world.insert_resource(RealTime::default());
    world.insert_resource(GameTime::default());

    // Insert debug drawing resources
    let debug_drawer_handle = world.insert_resource(DebugDrawer::new());
    world.insert_resource(GridConfig::new());

    // Latest per-pass GPU timings (#95); the render shell copies the device's
    // most recent retired-slot timings here each frame and the GPU stats
    // window reads it like any other panel.
    world.insert_resource(redlilium_graphics::FrameGpuTimings::default());

    // Latest GPU-memory stats (#98): the render shell copies the device's
    // per-frame sample (driver heap budgets + allocator totals + resource
    // counts) here and the GPU stats window's "Memory" section reads it.
    world.insert_resource(redlilium_graphics::GpuMemoryStats::default());

    // Insert ActionQueue for editor action dispatch
    world.insert_resource(ActionQueue::<World>::new());

    // The name -> constructor registry over EditActions: the remote channel's
    // generic `action` command builds from it (docs/REMOTE.md).
    world.insert_resource(redlilium_ecs::ui::ActionRegistry::with_builtins());

    // Insert Selection resource for tracking selected entities
    world.insert_resource(redlilium_ecs::ui::Selection::new());

    // Viewport extension registries: right-click menu ops (editor built-ins
    // now; game plugins append via Plugin::build_editing_view) and
    // interactive viewport tools.
    world.insert_resource(redlilium_ecs::ui::ViewportOps::with_builtins());
    world.insert_resource(redlilium_ecs::ui::ViewportTools::default());

    // Translate gizmo (#85): interaction state + focus/dots + the scene-rect
    // mapping the shells publish each frame. The GPU renderer resource is
    // shell-owned (needs device/format) and inserted next to
    // DebugDrawerRenderer.
    world.insert_resource(redlilium_gizmo::TransformGizmo::new(
        redlilium_gizmo::GizmoConfig::default(),
    ));
    world.insert_resource(crate::gizmo_system::GizmoUiState::default());
    world.insert_resource(crate::gizmo_system::SceneViewRect::default());

    let schedules = build_editor_schedules(params.egui);

    EditorWorld {
        world,
        schedules,
        history: EditActionHistory::new(DEFAULT_MAX_UNDO),
        editor_camera: Entity::DANGLING,
        window_input: window_input_handle,
        debug_drawer: debug_drawer_handle,
    }
}

/// Spawn the editor camera (marked EDITOR) and return its entity.
pub fn spawn_editor_camera(world: &mut World, aspect: f32) -> Entity {
    let editor_camera = world.spawn();

    let camera = Camera::perspective(FRAC_PI_4, aspect, 0.1, 500.0);
    let free_fly = FreeFlyCamera::new(Vec3::new(0.0, 0.5, 0.0), 5.0)
        .with_yaw(0.6)
        .with_pitch(0.3);

    world.insert(editor_camera, camera).unwrap();
    world.insert(editor_camera, free_fly).unwrap();
    let transform = free_fly.to_transform();
    world.insert(editor_camera, transform).unwrap();
    world
        .insert(editor_camera, GlobalTransform(transform.to_matrix()))
        .unwrap();
    world.insert(editor_camera, Visibility::VISIBLE).unwrap();

    // Mark the editor camera as an editor-only entity so it is hidden
    // from game queries and the world inspector by default.
    redlilium_ecs::mark_editor(world, editor_camera);
    editor_camera
}

/// Spawn the demo scene (ground plane, cubes, spheres).
fn spawn_demo_scene(world: &mut World, engine: &EngineContext) {
    // Resolve the demo meshes by path (fall back to generated if absent).
    let (cube_source, sphere_source) = {
        let asset_db = engine.asset_db().read();
        (
            asset_db
                .guid_of(&AssetPath::new("std", "meshes/cube.rmesh"))
                .map(MeshSource::File)
                .unwrap_or_else(|| MeshSource::Generated(MeshGenerator::cube(0.5))),
            asset_db
                .guid_of(&AssetPath::new("std", "meshes/sphere.rmesh"))
                .map(MeshSource::File)
                .unwrap_or_else(|| MeshSource::Generated(MeshGenerator::sphere(0.5, 32, 16))),
        )
    };
    // The std `default` material instance every demo primitive binds. Bound by
    // its stable guid (not a path lookup) so it survives a rename/move of the
    // asset and merely fails to resolve — rather than crashing the editor — if
    // it is deleted.
    let material_source = MaterialInstanceSource {
        guid: Guid::stable("materials/default.matinst"),
    };

    // --- Demo scene entities ---
    // `cube_source` / `sphere_source` were resolved above (File assets from the
    // std mount, or generated fallbacks). Entities sharing a source share one
    // Arc<Mesh> (loaded asynchronously by the MeshManager).

    // Ground plane (scaled flat cube)
    {
        let entity = world.spawn();
        let transform = Transform::new(
            Vec3::new(0.0, -0.05, 0.0),
            redlilium_core::math::Quat::identity(),
            Vec3::new(10.0, 0.1, 10.0),
        );
        world.insert(entity, transform).unwrap();
        world
            .insert(entity, GlobalTransform(transform.to_matrix()))
            .unwrap();
        world.insert(entity, Visibility::VISIBLE).unwrap();

        // Both refs resolve asynchronously once the MeshLoad sync system runs.
        let primitive = Primitive::new(cube_source.clone(), material_source.clone());
        world
            .insert(entity, MeshRenderer::single(primitive))
            .unwrap();
    }

    // 3 cubes at different positions
    let cube_positions = [
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(-2.0, 0.5, 1.0),
        Vec3::new(1.5, 0.5, -1.0),
    ];
    for pos in &cube_positions {
        let entity = world.spawn();
        let transform = Transform::from_translation(*pos);
        world.insert(entity, transform).unwrap();
        world
            .insert(entity, GlobalTransform(transform.to_matrix()))
            .unwrap();
        world.insert(entity, Visibility::VISIBLE).unwrap();

        // Both refs resolve asynchronously once the MeshLoad sync system runs.
        let primitive = Primitive::new(cube_source.clone(), material_source.clone());
        world
            .insert(entity, MeshRenderer::single(primitive))
            .unwrap();
    }

    // A sphere (a different mesh + a different vertex layout) — exercises the
    // second File asset and the layout switch.
    {
        let entity = world.spawn();
        let transform = Transform::from_translation(Vec3::new(2.0, 0.7, -0.5));
        world.insert(entity, transform).unwrap();
        world
            .insert(entity, GlobalTransform(transform.to_matrix()))
            .unwrap();
        world.insert(entity, Visibility::VISIBLE).unwrap();

        let primitive = Primitive::new(sphere_source.clone(), material_source.clone());
        world
            .insert(entity, MeshRenderer::single(primitive))
            .unwrap();
    }

    // A textured sphere (generated — its layout carries UVs) binding the std
    // `textured` material instance. Only spawned if the asset exists, so the
    // demo doesn't add an invisible entity on a stripped-down mount.
    if let Some(guid) = {
        let db = world.resource::<AssetDb>();
        db.guid_of(&AssetPath::new("std", "materials/textured.matinst"))
    } {
        let entity = world.spawn();
        let transform = Transform::from_translation(Vec3::new(-1.0, 0.7, -2.0));
        world.insert(entity, transform).unwrap();
        world
            .insert(entity, GlobalTransform(transform.to_matrix()))
            .unwrap();
        world.insert(entity, Visibility::VISIBLE).unwrap();

        let primitive = Primitive::new(
            MeshSource::Generated(MeshGenerator::sphere(0.5, 32, 16)),
            MaterialInstanceSource { guid },
        );
        world
            .insert(entity, MeshRenderer::single(primitive))
            .unwrap();
        world.insert(entity, Name::new("Textured Sphere")).unwrap();
    }
}

/// Build the editor's schedule graph. Pure of world state — the game-reload
/// path calls this to stand up a fresh `Schedules` (dropping the previous
/// generation's game systems wholesale) and then re-runs `Plugin::build`
/// against it.
pub fn build_editor_schedules(egui: bool) -> Schedules {
    let mut schedules = Schedules::new();

    // Update: read-only editor systems (debug grid, future interaction systems).
    // Systems here cannot mutate the world directly — they must push actions
    // through the ActionQueue resource. Nothing gates game systems anymore
    // (#67 is moot): Play boots a wholly separate game world (editor/src/play.rs),
    // so these editor-only systems always run against the editing world.
    schedules.get_mut::<Update>().add(DrawGrid);
    schedules
        .get_mut::<Update>()
        .add(DrawSelectionAabb::default());
    schedules.get_mut::<Update>().set_read_only(true);

    // Render schedule: flush uploads -> render the forward scene -> overlay
    // debug lines (each ordered after the previous via dependency edges; the
    // debug pass reads the forward pass handle through system_result).
    schedules.get_mut::<Render>().add(FlushUploads);
    schedules.get_mut::<Render>().add(AssetGpuFlush);
    // Derives the scene camera's CameraTarget from its CameraOutput spec
    // (ADR-029) before the forward pass consumes it.
    schedules
        .get_mut::<Render>()
        .add_exclusive(redlilium_ecs::EnsureCameraTargets);
    schedules.get_mut::<Render>().add(CameraRender);
    schedules.get_mut::<Render>().add(DebugRender);
    // Both FlushUploads and AssetGpuFlush use raw RenderSchedule access, so
    // they must not run in parallel under the multi-threaded runner.
    schedules
        .get_mut::<Render>()
        .add_edge::<FlushUploads, AssetGpuFlush>()
        .expect("FlushUploads -> AssetGpuFlush edge");
    schedules
        .get_mut::<Render>()
        .add_edge::<FlushUploads, CameraRender>()
        .expect("FlushUploads -> CameraRender edge");
    // Asset GPU uploads (e.g. freshly loaded meshes) must land before the
    // scene draw that uses them.
    schedules
        .get_mut::<Render>()
        .add_edge::<AssetGpuFlush, CameraRender>()
        .expect("AssetGpuFlush -> CameraRender edge");
    schedules
        .get_mut::<Render>()
        .add_edge::<redlilium_ecs::EnsureCameraTargets, CameraRender>()
        .expect("EnsureCameraTargets -> CameraRender edge");
    schedules
        .get_mut::<Render>()
        .add_edge::<CameraRender, DebugRender>()
        .expect("CameraRender -> DebugRender edge");
    // Gizmo overlay draws over the scene+debug image (#85).
    schedules
        .get_mut::<Render>()
        .add(crate::gizmo_system::GizmoRender);
    schedules
        .get_mut::<Render>()
        .add_edge::<DebugRender, crate::gizmo_system::GizmoRender>()
        .expect("DebugRender -> GizmoRender edge");
    // egui composites on top of the scene/debug passes; it reads the last
    // CameraTarget writer (ScenePass) to depend on it, so order it last.
    // The headless shell has no window to composite onto and skips it.
    if egui {
        schedules.get_mut::<Render>().add(EguiRender);
        schedules
            .get_mut::<Render>()
            .add_edge::<DebugRender, EguiRender>()
            .expect("DebugRender -> EguiRender edge");
        schedules
            .get_mut::<Render>()
            .add_edge::<crate::gizmo_system::GizmoRender, EguiRender>()
            .expect("GizmoRender -> EguiRender edge");
    }

    // PostUpdate: camera input -> transform propagation -> camera matrices.
    // Camera movement is viewport navigation, not a scene mutation, so it
    // lives in the non-read-only PostUpdate schedule. UpdateFreeFlyCamera is
    // editor-only, but there is nothing to gate it against anymore — Play
    // never runs in this world.
    schedules.get_mut::<PostUpdate>().add(UpdateFreeFlyCamera);
    schedules
        .get_mut::<PostUpdate>()
        .add(UpdateGlobalTransforms);
    schedules.get_mut::<PostUpdate>().add(UpdateCameraMatrices);
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<UpdateFreeFlyCamera, UpdateGlobalTransforms>()
        .expect("No cycle");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>()
        .expect("No cycle");

    // Asset loading: HotReload -> MaterialInstanceLoad -> MeshLoad -> AssetPump.
    // All share asset managers + ChangedAssets; raw access requires serialization.
    // MeshLoad is an ExclusiveSystem (barrier) so it doesn't contend with
    // component-writing systems under the multi-threaded runner.
    schedules.get_mut::<PostUpdate>().add(HotReload);
    schedules
        .get_mut::<PostUpdate>()
        .add_exclusive(MeshLoad::default());
    schedules
        .get_mut::<PostUpdate>()
        .add(redlilium_ecs::RemoteServe);
    schedules.get_mut::<PostUpdate>().add(MaterialInstanceLoad);
    schedules.get_mut::<PostUpdate>().add(AssetPump);
    // The asset systems reach the world raw (`ctx.raw_world()`), so the
    // scheduler cannot see their true footprint — the ambiguity detector
    // (#54) treats them as owning the whole world. Give the schedule a total
    // order: camera chain first, then the asset pipeline as a chain, then
    // the remote pump. `core::tests::editor_schedules_have_no_ambiguities`
    // enforces this stays complete.
    // Gizmo interaction (#85): editor-only, needs fresh camera matrices,
    // raw world access (anchors + action queue) -> chained explicitly per #54.
    schedules
        .get_mut::<PostUpdate>()
        .add(crate::gizmo_system::GizmoInteract);
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<UpdateCameraMatrices, crate::gizmo_system::GizmoInteract>()
        .expect("UpdateCameraMatrices -> GizmoInteract edge");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<crate::gizmo_system::GizmoInteract, HotReload>()
        .expect("GizmoInteract -> HotReload edge");
    // Viewport tool driver: raw world like the gizmo, chained right after it
    // (tool previews want the same fresh camera matrices; #54 total order).
    schedules
        .get_mut::<PostUpdate>()
        .add(crate::tool_system::RunViewportTool);
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<crate::gizmo_system::GizmoInteract, crate::tool_system::RunViewportTool>()
        .expect("GizmoInteract -> RunViewportTool edge");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<crate::tool_system::RunViewportTool, HotReload>()
        .expect("RunViewportTool -> HotReload edge");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<UpdateCameraMatrices, HotReload>()
        .expect("UpdateCameraMatrices -> HotReload edge");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<HotReload, MeshLoad>()
        .expect("HotReload -> MeshLoad edge");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<MeshLoad, MaterialInstanceLoad>()
        .expect("MeshLoad -> MaterialInstanceLoad edge");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<HotReload, MaterialInstanceLoad>()
        .expect("HotReload -> MaterialInstanceLoad edge");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<MaterialInstanceLoad, AssetPump>()
        .expect("MaterialInstanceLoad -> AssetPump edge");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<MeshLoad, AssetPump>()
        .expect("MeshLoad -> AssetPump edge");
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<AssetPump, redlilium_ecs::RemoteServe>()
        .expect("AssetPump -> RemoteServe edge");

    // All per-draw uniforms (group 0 transforms, group 1 material props) are
    // written into the scene-view rings in `on_draw`, so there are no
    // per-frame GPU-sync ECS systems (UpdatePerEntityUniforms,
    // InitializeRenderEntities, SyncMaterialUniforms) in the editor schedule.

    schedules
}

/// Write the mount's records to its local `<dir>/assets.db` (RON, mount-relative).
pub fn persist_mount_db(db: &AssetDb, mount: &str, dir: &str) {
    match db.to_ron_for_mount(mount) {
        Ok(text) => {
            if let Err(e) = std::fs::write(format!("{dir}/assets.db"), text) {
                log::error!("failed to persist {mount} assets.db: {e}");
            }
        }
        Err(e) => log::error!("failed to serialize {mount} assets.db: {e}"),
    }
}

/// A free asset path `dir/new[.N].<ext>` under `source` not already in the DB.
pub fn unique_asset_path(world: &World, source: &str, dir: &str, ext: &str) -> String {
    let db = world.resource::<AssetDb>();
    (0u32..)
        .find_map(|i| {
            let name = if i == 0 {
                format!("new.{ext}")
            } else {
                format!("new_{i}.{ext}")
            };
            let path = if dir.is_empty() {
                name
            } else {
                format!("{dir}/{name}")
            };
            db.guid_of(&AssetPath::new(source, &path))
                .is_none()
                .then_some(path)
        })
        .expect("infinite range yields a free name")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_view::SceneViewState;
    use redlilium_ecs::{EcsRunner, PreUpdate, RunDiagnostics};
    use redlilium_graphics::{GraphicsInstance, TextureFormat};
    use redlilium_vfs::Vfs;

    /// #54: the ambiguity detector — now able to see raw `ctx.world()`
    /// access — must report ZERO ambiguities on the editor's CPU schedules.
    /// Every raw-access system is either an exclusive barrier (#52) or
    /// ordered by explicit edges (#53); a warning here is a real regression
    /// (a new system missing its dependency edge), which is exactly what
    /// this test is for.
    #[test]
    fn editor_schedules_have_no_ambiguities() {
        let instance = GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("graphics device");
        let engine = redlilium_runtime::EngineContext::with_vfs(device.clone(), Vfs::new());
        let mut scene_view = SceneViewState::new(device, TextureFormat::Bgra8UnormSrgb);

        let mut ew = create_editor_world(
            &EditorWorldParams {
                remote: false,
                egui: false,
            },
            &engine,
            &mut scene_view,
            1.0,
        );

        let runner = EcsRunner::single_thread();
        let diagnostics = RunDiagnostics {
            detect_ambiguities: true,
            ..Default::default()
        };

        // The CPU schedules that run under the MT runner every frame. Render
        // is exercised separately by the shells (it needs GPU-backed
        // resources like DebugDrawerRenderer that the shells insert).
        let containers = [
            ("PreUpdate", ew.schedules.get::<PreUpdate>()),
            ("Update", ew.schedules.get::<Update>()),
            ("PostUpdate", ew.schedules.get::<PostUpdate>()),
        ];
        for (name, container) in containers {
            let Some(container) = container else { continue };
            let result = runner.run_with(&mut ew.world, container, &diagnostics);
            let ambiguities = result.report.ambiguities.expect("requested");
            assert!(
                ambiguities.is_empty(),
                "editor {name} schedule has unordered conflicting systems \
                 (add a dependency edge or an exclusive barrier):\n{ambiguities:#?}"
            );
        }
    }
}
