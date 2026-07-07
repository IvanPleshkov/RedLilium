//! The editor core shared by both shells: the windowed editor
//! ([`crate::editor::Editor`]) and the headless one ([`crate::headless`]).
//!
//! Owns world construction (demo scene, asset system, schedules) and the
//! frame-tick primitives every shell runs: draining the [`ActionQueue`]
//! through the undo history and persisting dirty asset DBs. Shell-specific
//! concerns (egui, winit input, picking) stay in the shells.

use std::f32::consts::FRAC_PI_4;
use std::sync::Arc;

use parking_lot::RwLock;

use redlilium_assets::{AssetDb, AssetPath, AssetProcessor, Guid};
use redlilium_core::abstract_editor::{ActionQueue, DEFAULT_MAX_UNDO, EditActionHistory};
use redlilium_core::math::Vec3;
use redlilium_debug_drawer::DebugDrawer;
use redlilium_ecs::{
    AssetGpuFlush, AssetPump, Camera, ChangedAssets, DebugRender, DrawGrid, DrawSelectionAabb,
    EguiRender, Entity, FlushUploads, ForwardRender, FrameRing, FreeFlyCamera, GlobalTransform,
    GridConfig, HotReload, MaterialAssetManager, MaterialInstanceLoad, MaterialInstanceLoader,
    MaterialInstanceManager, MaterialInstanceSource, MaterialLoader, MeshGenerator, MeshLoad,
    MeshLoader, MeshManager, MeshRenderer, MeshSource, Name, PipelineCache, PostUpdate, Primitive,
    Render, RenderSchedule, ScenePass, Schedules, ShaderLoader, ShaderManager, ShadingRegistry,
    TextureLoader, TextureManager, Transform, Update, UpdateCameraMatrices, UpdateFreeFlyCamera,
    UpdateGlobalTransforms, VertexLayoutLoader, VertexLayoutManager, Visibility, WindowInput,
    World, register_std_components,
};
use redlilium_vfs::Vfs;

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

/// What to build into a new editor world — the windowed shell wants the egui
/// overlay system, the headless shell wants the remote channel unconditionally.
pub struct EditorWorldParams<'a> {
    pub vfs: &'a Vfs,
    /// Local asset-pack mounts `(name, dir)`, each with its own `assets.db`.
    pub local_mounts: &'a [(&'static str, &'static str)],
    /// Insert the `RemoteTransport` resource and `RemoteServe` system.
    pub remote: bool,
    /// Register the `EguiRender` overlay system (windowed shell only).
    pub egui: bool,
}

/// Create a new editor world with a simple demo scene.
pub fn create_editor_world(
    params: &EditorWorldParams<'_>,
    scene_view: &mut SceneViewState,
    aspect: f32,
) -> EditorWorld {
    let mut world = World::new();
    register_std_components(&mut world);
    redlilium_ecs::register_rendering_components(&mut world);

    // Insert rendering manager resources
    world.insert_resource(TextureManager::new(scene_view.device().clone()));
    world.insert_resource(MeshManager::new());
    world.insert_resource(VertexLayoutManager::new());
    world.insert_resource(ShaderManager::new());
    // Asset-based material system: the shading registry (engine code), the
    // template + instance resolvers, and the draw-time pipeline cache.
    world.insert_resource(ShadingRegistry::with_builtins());
    world.insert_resource(MaterialAssetManager::new());
    world.insert_resource(MaterialInstanceManager::new(scene_view.device().clone()));
    world.insert_resource(PipelineCache::new(scene_view.device().clone()));
    // Hot-reload inbox: inspector edits + fs-watcher changes land here; the
    // HotReload system drains it and invalidates the owning managers.
    world.insert_resource(ChangedAssets::new());
    // Mounts with un-persisted asset-DB edits (written by the undoable
    // asset-edit actions); drained + persisted once per frame.
    world.insert_resource(redlilium_ecs::DirtyMounts::new());
    // Remote-control channel (docs/REMOTE.md): served by RemoteServe on
    // the IO runtime; the editor pumps commands each frame. Opt-in.
    if params.remote {
        world.insert_resource(redlilium_ecs::RemoteTransport::new(
            ".redlilium/editor.port",
        ));
    }

    // Asset system: one processor (with the rendering loaders) + one DB. The
    // AssetPump / MeshLoad / AssetGpuFlush systems drive these each frame.
    let processor = AssetProcessor::builder(params.vfs.clone(), scene_view.device().clone())
        .with_loader::<MeshLoader>()
        .with_loader::<VertexLayoutLoader>()
        .with_loader::<ShaderLoader>()
        .with_loader::<MaterialLoader>()
        .with_loader::<MaterialInstanceLoader>()
        .with_loader::<TextureLoader>()
        .build();

    // Load each local mount's asset DB into the one merged in-memory DB,
    // then scan the mount: files added/changed/removed while the editor was
    // closed get registered (routed by the loaders' extensions), rehashed,
    // or dropped. A non-empty delta is persisted right back.
    let mut asset_db = AssetDb::new();
    for (mount, dir) in params.local_mounts {
        match std::fs::read_to_string(format!("{dir}/assets.db")) {
            Ok(text) => {
                if let Err(e) = asset_db.merge_ron(mount, &text) {
                    log::error!("failed to parse {mount} assets.db: {e}");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => log::warn!("{mount} assets.db not readable: {e}"),
        }
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
    world.insert_resource(processor);
    // Resolve the demo meshes by path (fall back to generated if absent).
    let cube_source = asset_db
        .guid_of(&AssetPath::new("std", "meshes/cube.rmesh"))
        .map(MeshSource::File)
        .unwrap_or_else(|| MeshSource::Generated(MeshGenerator::cube(0.5)));
    let sphere_source = asset_db
        .guid_of(&AssetPath::new("std", "meshes/sphere.rmesh"))
        .map(MeshSource::File)
        .unwrap_or_else(|| MeshSource::Generated(MeshGenerator::sphere(0.5, 32, 16)));
    // The std `default` material instance every demo primitive binds. Bound by
    // its stable guid (not a path lookup) so it survives a rename/move of the
    // asset and merely fails to resolve — rather than crashing the editor — if
    // it is deleted.
    let material_source = MaterialInstanceSource {
        guid: Guid::stable("materials/default.matinst"),
    };
    world.insert_resource(asset_db);

    // Holds the per-frame render graph while the `Render` schedule runs.
    world.insert_resource(RenderSchedule::empty());

    // Scene forward-pass dynamic-uniform ring (an ECS resource). Give the
    // scene view its buffer so per-primitive materials bind it; the buffer is
    // needed before any entity is created below.
    let frame_ring = FrameRing::new(scene_view.device(), 1 << 20, "scene_frame_ring")
        .expect("Failed to create scene frame ring");
    scene_view.set_frame_ring_buffer(frame_ring.buffer().clone());
    world.insert_resource(frame_ring);
    // The ForwardRender system records its scene-pass handle here so the egui
    // overlay + debug pass can depend on it.
    world.insert_resource(ScenePass::default());

    // Insert WindowInput resource
    let window_input_handle = world.insert_resource(WindowInput::default());

    // Insert debug drawing resources
    let debug_drawer_handle = world.insert_resource(DebugDrawer::new());
    world.insert_resource(GridConfig::new());

    // --- Editor camera ---
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
    redlilium_ecs::mark_editor(&mut world, editor_camera);

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

    // Insert ActionQueue for editor action dispatch
    world.insert_resource(ActionQueue::<World>::new());

    // The name -> constructor registry over EditActions: the remote channel's
    // generic `action` command builds from it (docs/REMOTE.md).
    world.insert_resource(redlilium_ecs::ui::ActionRegistry::with_builtins());

    // Insert Selection resource for tracking selected entities
    world.insert_resource(redlilium_ecs::ui::Selection::new());

    // --- Setup schedules ---
    let mut schedules = Schedules::new();

    // Update: read-only editor systems (debug grid, future interaction systems).
    // Systems here cannot mutate the world directly — they must push actions
    // through the ActionQueue resource.
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
    schedules.get_mut::<Render>().add(ForwardRender::default());
    schedules.get_mut::<Render>().add(DebugRender);
    schedules
        .get_mut::<Render>()
        .add_edge::<FlushUploads, ForwardRender>()
        .expect("FlushUploads -> ForwardRender edge");
    // Asset GPU uploads (e.g. freshly loaded meshes) must land before the
    // scene draw that uses them.
    schedules
        .get_mut::<Render>()
        .add_edge::<AssetGpuFlush, ForwardRender>()
        .expect("AssetGpuFlush -> ForwardRender edge");
    schedules
        .get_mut::<Render>()
        .add_edge::<ForwardRender, DebugRender>()
        .expect("ForwardRender -> DebugRender edge");
    // egui composites on top of the scene/debug passes; it reads the last
    // CameraTarget writer (ScenePass) to depend on it, so order it last.
    // The headless shell has no window to composite onto and skips it.
    if params.egui {
        schedules.get_mut::<Render>().add(EguiRender);
        schedules
            .get_mut::<Render>()
            .add_edge::<DebugRender, EguiRender>()
            .expect("DebugRender -> EguiRender edge");
    }

    // PostUpdate: camera input -> transform propagation -> camera matrices.
    // Camera movement is viewport navigation, not a scene mutation, so it
    // lives in the non-read-only PostUpdate schedule.
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

    // Asset loading: MeshLoad resolves layouts + requests meshes, then
    // AssetPump drains the async stages onto the compute/IO pools + collects.
    schedules.get_mut::<PostUpdate>().add(HotReload);
    schedules.get_mut::<PostUpdate>().add(MeshLoad::default());
    schedules
        .get_mut::<PostUpdate>()
        .add(redlilium_ecs::RemoteServe);
    schedules.get_mut::<PostUpdate>().add(MaterialInstanceLoad);
    schedules.get_mut::<PostUpdate>().add(AssetPump);
    schedules
        .get_mut::<PostUpdate>()
        .add_edge::<MeshLoad, AssetPump>()
        .expect("MeshLoad -> AssetPump edge");

    // All per-draw uniforms (group 0 transforms, group 1 material props) are
    // written into the scene-view rings in `on_draw`, so there are no
    // per-frame GPU-sync ECS systems (UpdatePerEntityUniforms,
    // InitializeRenderEntities, SyncMaterialUniforms) in the editor schedule.

    EditorWorld {
        world,
        schedules,
        history: EditActionHistory::new(DEFAULT_MAX_UNDO),
        editor_camera,
        window_input: window_input_handle,
        debug_drawer: debug_drawer_handle,
    }
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
