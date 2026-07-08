use std::f32::consts::FRAC_PI_4;
use std::sync::Arc;

use parking_lot::RwLock;

use egui_dock::DockState;
use redlilium_app::{AppContext, AppHandler, DrawContext};
use redlilium_assets::{AssetDb, AssetPath, AssetProcessor};
use redlilium_debug_drawer::DebugDrawerRenderer;
use redlilium_ecs::ui::{
    ComponentDragPayload, ComponentFileDragPayload, ImportComponentAction, InspectorState,
    PrefabFileDragPayload, SelectAction, SpawnPrefabAction,
};
use redlilium_ecs::{
    Camera, ChangedAssets, EcsRunner, Entity, FrameTarget, MeshRenderer, Name, Render,
    RenderSchedule, World,
};
use redlilium_graphics::egui::{EguiApp, EguiController};
use redlilium_graphics::{FrameSchedule, RenderTarget, TextureFormat};
use redlilium_vfs::{FileSystemProvider, Vfs};
use winit::event::{KeyEvent, MouseButton, MouseScrollDelta};
use winit::keyboard::PhysicalKey;

use crate::asset_browser::AssetBrowser;
use crate::console::ConsolePanel;
use crate::core::{EditorWorld, EditorWorldParams, create_editor_world, unique_asset_path};
use crate::dock::{self, EditorTabViewer, Tab};
#[cfg(not(target_os = "macos"))]
use crate::menu;
#[cfg(target_os = "macos")]
use crate::menu::NativeMenu;
use crate::scene_view::SceneViewState;
use crate::status_bar;
use crate::toolbar::PlayState;

/// A minimal EguiApp that does nothing.
///
/// All actual UI rendering happens in [`Editor::on_draw`] using the egui
/// context directly between `begin_frame` / `end_frame`.
struct NullEguiApp;

impl EguiApp for NullEguiApp {
    fn update(&mut self, _ctx: &egui::Context) {}

    fn setup(&mut self, ctx: &egui::Context) {
        crate::theme::apply(ctx);
    }
}

pub struct Editor {
    // The single editor world. Created lazily once graphics is available
    // (see `on_resume`); `None` until then. The engine is single-world by
    // design — one scene is resident at a time.
    world: Option<EditorWorld>,
    runner: EcsRunner,
    /// Persistent engine state (GPU managers, asset DB/processor) — outlives
    /// any world (ADR-020). Created with the graphics device in `on_init`.
    engine: Option<redlilium_runtime::EngineContext>,

    // VFS and asset browser
    vfs: Vfs,
    asset_browser: AssetBrowser,
    /// Local asset-pack mounts `(name, dir)` — each carries its own
    /// `<dir>/assets.db`, loaded + scanned at world creation and persisted
    /// when edited.
    local_mounts: Vec<(&'static str, &'static str)>,
    /// Remote-control protocol state (`REDLILIUM_REMOTE=1`, docs/REMOTE.md).
    remote: Option<crate::remote_commands::RemoteCommands>,
    console: ConsolePanel,

    // UI (the egui controller is an ECS resource — see on_init)
    dock_state: DockState<Tab>,
    inspector_state: InspectorState,
    play_state: PlayState,
    #[cfg(target_os = "macos")]
    native_menu: Option<NativeMenu>,

    // Scene rendering
    scene_view: Option<SceneViewState>,
    /// egui texture id for the scene color target shown in the SceneView panel.
    scene_texture_id: Option<egui::TextureId>,

    // Input state for egui feedback
    egui_wants_pointer: bool,
    egui_wants_keyboard: bool,

    // Scene view interaction
    /// Scene view rect in physical pixels (x, y, w, h).
    scene_view_rect_phys: Option<[f32; 4]>,
    /// Current cursor position in physical pixels (for hit-testing).
    cursor_pos: [f32; 2],

    // Box selection drag state
    /// Physical pixel position where LMB was pressed (drag origin).
    drag_start: Option<[f32; 2]>,
    /// Whether the drag has exceeded the threshold and is now a box selection.
    dragging_box: bool,

    /// Smoothed frames-per-second for the status bar.
    fps: f32,

    /// Pending component import from asset browser (VFS read in progress).
    pending_import: Option<PendingImport>,
    /// Pending prefab import from asset browser (VFS read in progress).
    pending_prefab_import: Option<PendingPrefabImport>,

    /// Whether the "unsaved changes" dialog is currently shown.
    show_close_dialog: bool,
    /// Set to `true` when the user confirms closing (with or without saving).
    should_close: bool,
    /// An asset pending delete-confirmation: `(source, path)`. Shows a modal.
    pending_delete_confirm: Option<(String, String)>,

    /// Last frame's entity selection — when it changes to a non-empty set, the
    /// asset selection is cleared so the inspector switches back to the entity.
    last_selection: Vec<Entity>,
}

/// Tracks an in-flight VFS read for component import.
struct PendingImport {
    vfs_path: String,
    entity: Entity,
}

/// Tracks an in-flight VFS read for prefab import.
struct PendingPrefabImport {
    vfs_path: String,
    parent: Option<Entity>,
}

impl Editor {
    pub fn new() -> Self {
        let project_path = std::path::Path::new("project.toml");
        let (config, mut vfs) = crate::project::load_or_default(project_path);
        // Local asset-pack mounts: the engine's built-ins plus the project
        // sandbox (user-authored / experimental assets, gitignored — so playing
        // with materials never dirties the committed std pack). Each carries
        // its own assets.db, is watched for external edits, and is scanned on
        // startup.
        let local_mounts = [("std", "std-assets"), ("project", "project-assets")];
        let mut asset_browser = AssetBrowser::new(&config);
        for (name, dir) in local_mounts {
            if let Err(e) = std::fs::create_dir_all(dir) {
                log::error!("failed to create mount dir '{dir}': {e}");
            }
            vfs.mount(name, FileSystemProvider::new(dir));
            asset_browser.add_mount(name);
            asset_browser.watch_local_mount(name, dir);
        }
        let console = ConsolePanel::new(crate::log_capture::log_buffer());

        Self {
            world: None,
            // Single-threaded for now. Switching to the multi-threaded runner
            // Multi-threaded runner enabled after Track 2 MT-hardening (#45).
            runner: EcsRunner::multi_thread(
                std::thread::available_parallelism()
                    .map(|p| p.get().saturating_sub(2).max(1))
                    .unwrap_or(1),
            ),
            engine: None,
            vfs,
            asset_browser,
            local_mounts: local_mounts.to_vec(),
            remote: std::env::var("REDLILIUM_REMOTE")
                .is_ok_and(|v| v == "1")
                .then(crate::remote_commands::RemoteCommands::default),
            console,
            dock_state: dock::create_default_layout(),
            inspector_state: InspectorState::new(),
            play_state: PlayState::Editing,
            #[cfg(target_os = "macos")]
            native_menu: None,
            scene_view: None,
            scene_texture_id: None,
            egui_wants_pointer: false,
            egui_wants_keyboard: false,
            scene_view_rect_phys: None,
            cursor_pos: [0.0, 0.0],
            drag_start: None,
            dragging_box: false,
            fps: 0.0,
            pending_import: None,
            pending_prefab_import: None,
            show_close_dialog: false,
            pending_delete_confirm: None,
            should_close: false,
            last_selection: Vec::new(),
        }
    }

    /// Get the active editor world (immutable).
    fn active_world(&self) -> &EditorWorld {
        self.world.as_ref().unwrap()
    }

    /// Returns `true` if any editor world has unsaved changes.
    fn has_unsaved_changes(&self) -> bool {
        self.world
            .as_ref()
            .is_some_and(|ew| ew.history.has_unsaved_changes())
    }

    /// Apply a play action (Play/Pause/Resume/Stop) to the game.
    fn apply_play_action(&mut self, action: crate::toolbar::PlayAction) {
        if let Some(ew) = self.world.as_mut() {
            let mut play_control = ew.world.resource_mut::<redlilium_ecs::PlayControl>();
            match action {
                crate::toolbar::PlayAction::Play => play_control.play(),
                crate::toolbar::PlayAction::Pause => play_control.pause(),
                crate::toolbar::PlayAction::Resume => play_control.resume(),
                crate::toolbar::PlayAction::Stop => play_control.stop(),
            }
        }
        // Update display state to match PlayControl (will sync on next frame)
        self.sync_play_state();
    }

    /// Sync toolbar's play_state to match PlayControl's current state.
    fn sync_play_state(&mut self) {
        if let Some(ew) = self.world.as_ref() {
            let pc_state = ew.world.resource::<redlilium_ecs::PlayControl>().state();
            self.play_state = match pc_state {
                redlilium_ecs::PlayState::Stopped => crate::toolbar::PlayState::Editing,
                redlilium_ecs::PlayState::Playing => crate::toolbar::PlayState::Playing,
                redlilium_ecs::PlayState::Paused => crate::toolbar::PlayState::Paused,
            };
        }
    }

    /// Run `f` with the egui controller resource, if the world (and thus the
    /// controller) exists. The controller is an ECS resource (see `on_init`).
    fn with_egui(&self, f: impl FnOnce(&mut EguiController)) {
        if let Some(ew) = self.world.as_ref() {
            f(&mut ew.world.resource_mut::<EguiController>());
        }
    }

    /// Whether the cursor is currently inside the scene view panel.
    fn cursor_in_scene_view(&self) -> bool {
        if let Some([x, y, w, h]) = self.scene_view_rect_phys {
            let [cx, cy] = self.cursor_pos;
            cx >= x && cx <= x + w && cy >= y && cy <= y + h
        } else {
            false
        }
    }

    /// Request a pixel-perfect box selection by reading from the entity index
    /// texture. The actual selection is deferred until the GPU readback completes
    /// (resolved in `on_update` via `resolve_rect_pick`).
    fn perform_box_selection(&mut self, start: [f32; 2], end: [f32; 2]) {
        let Some(scene_view) = &mut self.scene_view else {
            return;
        };

        // Build the drag rectangle in physical pixels (normalize min/max).
        let x = start[0].min(end[0]).max(0.0) as u32;
        let y = start[1].min(end[1]).max(0.0) as u32;
        let x2 = start[0].max(end[0]).max(0.0) as u32;
        let y2 = start[1].max(end[1]).max(0.0) as u32;
        let w = x2.saturating_sub(x).max(1);
        let h = y2.saturating_sub(y).max(1);

        scene_view.request_rect_pick(x, y, w, h);
    }

    /// Update WindowInput's ui_wants_input flag from egui state.
    fn sync_input_flags(&self) {
        let ew = self.active_world();
        {
            let mut input = ew.window_input.write();
            input.ui_wants_input = !self.cursor_in_scene_view();
        }
    }

    /// Update the editor camera's projection on resize.
    fn update_camera_projection(&mut self, aspect: f32) {
        let ew = self.world.as_mut().unwrap();
        if let Ok(mut cameras) = ew.world.write_all::<Camera>()
            && let Some(mut cam) = cameras.get_mut(ew.editor_camera.index())
        {
            cam.projection_matrix =
                redlilium_core::math::perspective_rh(FRAC_PI_4, aspect, 0.1, 500.0);
        }
    }

    /// Apply a rename/move: rebind the DB record (if the file is a registered
    /// asset) and move the VFS file. Both are same-mount path changes; the guid is
    /// preserved so references by guid survive.
    fn perform_asset_move(&mut self, source: &str, old_path: &str, new_path: &str) {
        if new_path == old_path {
            return;
        }
        let rebound = {
            let ew = self.world.as_mut().unwrap();
            let guid = ew
                .world
                .resource::<AssetDb>()
                .guid_of(&AssetPath::new(source, old_path));
            match guid {
                Some(g) => match ew
                    .world
                    .resource_mut::<AssetDb>()
                    .rebind(g, AssetPath::new(source, new_path))
                {
                    Ok(()) => true,
                    Err(e) => {
                        log::error!("rebind {old_path} -> {new_path} failed: {e:?}");
                        return;
                    }
                },
                None => false,
            }
        };
        if rebound {
            self.asset_browser.mark_db_dirty(source);
        }
        let from = format!("{source}/{old_path}");
        let to = format!("{source}/{new_path}");
        self.asset_browser.dispatch_move(&self.vfs, &from, &to);
        self.asset_browser.notify_asset_moved(source, new_path);
        log::info!("Asset {from} -> {to}");
    }

    /// Delete an asset: remove its DB record (if registered) and the VFS file.
    fn perform_asset_delete(&mut self, source: &str, path: &str) {
        let removed = {
            let ew = self.world.as_mut().unwrap();
            let guid = ew
                .world
                .resource::<AssetDb>()
                .guid_of(&AssetPath::new(source, path));
            match guid {
                Some(g) => {
                    ew.world.resource_mut::<AssetDb>().remove(&g);
                    true
                }
                None => false,
            }
        };
        if removed {
            self.asset_browser.mark_db_dirty(source);
        }
        let vfs_path = format!("{source}/{path}");
        self.asset_browser.dispatch_delete(&self.vfs, &vfs_path);
        self.asset_browser.notify_asset_deleted(source, path);
        log::info!("Deleted asset {vfs_path}");
    }
}

impl AppHandler for Editor {
    fn on_init(&mut self, ctx: &mut AppContext) {
        log::info!("Editor initialized");

        let null_app: Arc<RwLock<dyn EguiApp>> = Arc::new(RwLock::new(NullEguiApp));
        let egui_controller = EguiController::new(
            ctx.device().clone(),
            null_app,
            ctx.width(),
            ctx.height(),
            ctx.scale_factor(),
            ctx.surface_format(),
        );

        // Create scene view GPU resources
        let mut scene_view = SceneViewState::new(ctx.device().clone(), ctx.surface_format());
        scene_view.resize_if_needed(ctx.width(), ctx.height());

        // Persistent engine state (shared managers + asset DB), then the
        // startup mount scan — both outlive the editor world.
        let engine =
            redlilium_runtime::EngineContext::with_vfs(ctx.device().clone(), self.vfs.clone());
        crate::core::scan_local_mounts(&engine, &self.local_mounts);

        // Create the editor world with a demo scene
        let aspect = ctx.aspect_ratio();
        let editor_world = create_editor_world(
            &EditorWorldParams {
                remote: self.remote.is_some(),
                egui: true,
            },
            &engine,
            &mut scene_view,
            aspect,
        );
        self.engine = Some(engine);
        self.world = Some(editor_world);

        self.scene_view = Some(scene_view);

        // The egui controller is an ECS resource so the egui render pass can be a
        // render system (and future systems can build egui UI).
        if let Some(ew) = self.world.as_mut() {
            ew.world.insert_resource(egui_controller);
        }

        // Debug drawer renderer as a resource — the DebugRender system draws it
        // each frame into the camera target, after the forward pass.
        if let Some(ew) = self.world.as_mut() {
            ew.world.insert_resource(DebugDrawerRenderer::new(
                ctx.device().clone(),
                ctx.surface_format(),
                Some(TextureFormat::Depth32Float),
            ));
        }

        // Run startup schedules
        let runner = &self.runner;
        let ew = self.world.as_mut().unwrap();
        ew.schedules.run_startup(&mut ew.world, runner);

        // Create native menu after the event loop / NSApplication is initialized
        #[cfg(target_os = "macos")]
        {
            self.native_menu = Some(NativeMenu::new());
        }
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        let (w, h) = (ctx.width(), ctx.height());
        self.with_egui(|egui| egui.on_resize(w, h));
        if let Some(sv) = &mut self.scene_view {
            sv.resize_if_needed(ctx.width(), ctx.height());
        }
        // Update camera projection for new aspect ratio
        if self.world.is_some()
            && let Some(sv) = &self.scene_view
        {
            self.update_camera_projection(sv.aspect_ratio());
        }
    }

    fn on_close_requested(&mut self, _ctx: &mut AppContext) -> bool {
        if self.has_unsaved_changes() {
            self.show_close_dialog = true;
            return false; // don't close yet — show dialog first
        }
        true
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        if self.should_close {
            return false;
        }

        // Persist edited asset DBs — any local mount writes back to its own
        // `<dir>/assets.db`. Two dirt sources: browser file operations
        // (create/move/delete) and the undoable asset-edit actions.
        if let Some(ew) = self.world.as_ref() {
            if let Some(mount) = self.asset_browser.take_db_dirty() {
                ew.world
                    .resource_mut::<redlilium_ecs::DirtyMounts>()
                    .push(&mount);
            }
            ew.persist_dirty_mounts(&self.local_mounts);
        }

        if self.world.is_none() {
            return true;
        }

        // When the entity selection changes to a non-empty set, drop the asset
        // selection so the inspector switches back to the entity.
        if let Some(ew) = self.world.as_ref() {
            let current = ew
                .world
                .resource::<redlilium_ecs::ui::Selection>()
                .entities()
                .to_vec();
            if current != self.last_selection {
                if !current.is_empty() {
                    self.asset_browser.clear_selected_file();
                }
                self.last_selection = current;
            }
        }

        // Resolve GPU pick from the previous frame's readback
        if let Some(scene_view) = &mut self.scene_view
            && let Some(hit) = scene_view.resolve_pick()
        {
            let ew = self.world.as_mut().unwrap();
            // A remote `pick` owns this readback — answer it, don't select.
            if self
                .remote
                .as_ref()
                .is_some_and(crate::remote_commands::pick_in_flight)
            {
                let rc = self.remote.as_mut().unwrap();
                crate::remote_commands::complete_point_pick(rc, &ew.world, hit);
            } else {
                // Find entity whose sparse-set index matches the picked index
                let target = hit.filter(|&entity_index| {
                    ew.world
                        .read::<MeshRenderer>()
                        .ok()
                        .is_some_and(|buffers| buffers.iter().any(|(idx, _)| idx == entity_index))
                });
                // Reconstruct full Entity from world (need spawn_tick etc.)
                let target_entity =
                    target.and_then(|entity_index| ew.world.entity_at_index(entity_index));
                let action: Box<dyn redlilium_core::abstract_editor::EditAction<World>> =
                    if let Some(entity) = target_entity {
                        Box::new(SelectAction::single(entity))
                    } else {
                        Box::new(SelectAction::clear())
                    };
                if let Err(e) = ew.history.execute(action, &mut ew.world) {
                    log::warn!("Pick selection failed: {e}");
                }
            }
        }

        // Resolve GPU rect pick from the previous frame's readback
        if let Some(scene_view) = &mut self.scene_view
            && let Some(entity_indices) = scene_view.resolve_rect_pick()
        {
            let ew = self.world.as_mut().unwrap();
            // A remote `pick_rect` owns this readback — answer it, don't select.
            if self
                .remote
                .as_ref()
                .is_some_and(crate::remote_commands::pick_in_flight)
            {
                let rc = self.remote.as_mut().unwrap();
                crate::remote_commands::complete_rect_pick(rc, &ew.world, &entity_indices);
            } else {
                let selected: Vec<Entity> = if let Ok(renderers) = ew.world.read::<MeshRenderer>() {
                    entity_indices
                        .iter()
                        .filter_map(|&idx| {
                            // Verify the index is a renderable entity.
                            if renderers.get(idx).is_some() {
                                ew.world.entity_at_index(idx)
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                let action: Box<dyn redlilium_core::abstract_editor::EditAction<World>> =
                    if selected.is_empty() {
                        Box::new(SelectAction::clear())
                    } else {
                        Box::new(SelectAction::set(selected))
                    };
                if let Err(e) = ew.history.execute(action, &mut ew.world) {
                    log::warn!("Rect selection failed: {e}");
                }
            }
        }

        // Poll native menu events (macOS only)
        #[cfg(target_os = "macos")]
        if let Some(menu) = &self.native_menu
            && let Some(action) = menu.poll_event()
        {
            use crate::menu::MenuAction;
            let ew = self.world.as_mut().unwrap();
            match action {
                MenuAction::Save => {
                    ew.history.mark_saved();
                    log::info!("Saved");
                }
                MenuAction::Undo => {
                    if let Err(e) = ew.history.undo(&mut ew.world) {
                        log::warn!("Undo failed: {e}");
                    }
                }
                MenuAction::Redo => {
                    if let Err(e) = ew.history.redo(&mut ew.world) {
                        log::warn!("Redo failed: {e}");
                    }
                }
                _ => log::info!("Menu action: {action:?}"),
            }
        }

        // Sync ui_wants_input flag from previous frame's egui state
        self.sync_input_flags();

        // Poll completed background VFS results for the asset browser
        self.asset_browser.poll();

        // Externally changed files → hot reload (map VFS path → guid via the DB).
        let changed_files = self.asset_browser.take_changed_files();
        if !changed_files.is_empty()
            && let Some(ew) = self.world.as_mut()
            && ew.world.has_resource::<AssetDb>()
        {
            for vfs_path in changed_files {
                let Some((source, rel)) = vfs_path.split_once('/') else {
                    continue;
                };
                let asset_path = AssetPath::new(source, rel);
                let guid = ew.world.resource::<AssetDb>().guid_of(&asset_path);
                if let Some(guid) = guid {
                    ew.world.resource_mut::<ChangedAssets>().push(guid);
                } else {
                    // A new file appeared on disk: if its extension routes to a
                    // loader (same table the startup scan uses), register it so
                    // it becomes referenceable immediately, and persist the DB.
                    let kind = rel.rsplit_once('.').and_then(|(_, ext)| {
                        ew.world
                            .resource::<AssetProcessor>()
                            .kind_for_extension(ext)
                            .map(str::to_owned)
                    });
                    if let Some(kind) = kind {
                        let guid = ew
                            .world
                            .resource_mut::<AssetDb>()
                            .register_path(asset_path, &kind, 0);
                        log::info!("registered new {kind} asset: {vfs_path} ({guid:?})");
                        self.asset_browser.mark_db_dirty(source);
                    }
                }
            }
        }

        // Advance debug drawer tick (systems will write to the new tick)
        {
            let ew = self.active_world();
            {
                let drawer = ew.debug_drawer.read();
                drawer.advance_tick();
            }
        }

        // Drain action queue and execute through history (before systems so
        // that Changed<T> filters can detect the mutations this frame).
        self.world.as_mut().unwrap().drain_actions();

        // Remote protocol pump: completes last frame's parked responses (their
        // actions just drained), then executes newly arrived commands.
        if let Some(rc) = &mut self.remote
            && let Some(ew) = self.world.as_mut()
        {
            crate::remote_commands::pump(rc, &mut ew.world, &mut ew.history);
            // Remote `shutdown` closes without the unsaved-changes dialog —
            // the caller asked for an exit, there is no one to click it.
            if rc.take_shutdown() {
                self.should_close = true;
            }
            // Submit a remote pick to the scene view. Wire coordinates are in
            // scene-image space (what `screenshot` shows); the pick pass works
            // in window space, so offset by the SceneView panel origin.
            if let Some(request) = crate::remote_commands::take_pick_request(rc) {
                match (self.scene_view_rect_phys, &mut self.scene_view) {
                    (Some([ox, oy, pw, ph]), Some(sv)) => {
                        use crate::remote_commands::PickRequest;
                        let (ox, oy) = (ox as u32, oy as u32);
                        match request {
                            PickRequest::Point { x, y } if x < pw as u32 && y < ph as u32 => {
                                sv.request_pick(ox + x, oy + y);
                            }
                            PickRequest::Rect { x, y, w, h } if x < pw as u32 && y < ph as u32 => {
                                let w = w.min(pw as u32 - x);
                                let h = h.min(ph as u32 - y);
                                sv.request_rect_pick(ox + x, oy + y, w, h);
                            }
                            _ => crate::remote_commands::fail_pick(
                                rc,
                                &ew.world,
                                "pick coordinates outside the scene image",
                            ),
                        }
                    }
                    _ => crate::remote_commands::fail_pick(
                        rc,
                        &ew.world,
                        "scene view is not visible",
                    ),
                }
            }
        }

        // Run ECS schedules (always run in editing mode for camera/transforms)
        {
            let Editor { world, runner, .. } = self;
            let ew = world.as_mut().unwrap();
            ew.schedules
                .run_frame(&mut ew.world, runner, ctx.delta_time() as f64);
        }

        // Process component export (inspector → asset browser)
        if let Some((entity, comp_name, vfs_dir)) =
            self.asset_browser.pending_component_export.take()
        {
            let ew = self.world.as_ref().unwrap();
            match ew.world.serialize_component_by_name(entity, comp_name) {
                Ok(Some(serialized)) => {
                    match redlilium_ecs::serialize::encode(
                        &serialized,
                        redlilium_ecs::serialize::Format::Ron,
                    ) {
                        Ok(data) => {
                            let vfs_path = format!("{vfs_dir}/{comp_name}.component");
                            log::info!("Exporting component to: {vfs_path}");
                            self.asset_browser
                                .dispatch_write(&self.vfs, &vfs_path, data);
                        }
                        Err(e) => log::error!("Failed to encode component: {e}"),
                    }
                }
                Ok(None) => log::warn!("Component '{comp_name}' not found or not serializable"),
                Err(e) => log::error!("Failed to serialize component: {e}"),
            }
        }

        // Process "New asset" creation (asset browser context menu → file + record).
        if let Some((source, dir, kind)) = self.asset_browser.take_pending_new() {
            let ew = self.world.as_mut().unwrap();
            // A material instance parents to the std opaque material by default.
            let parent = if kind == "material_instance" {
                ew.world
                    .resource::<AssetDb>()
                    .guid_of(&AssetPath::new("std", "materials/opaque.material"))
            } else {
                None
            };
            match redlilium_ecs::new_asset_spec(&kind, parent) {
                Some(spec) => {
                    let path = unique_asset_path(&ew.world, &source, &dir, spec.extension);
                    let asset_path = AssetPath::new(&source, &path);
                    let guid = ew
                        .world
                        .resource_mut::<AssetDb>()
                        .register_path(asset_path, &kind, 0);
                    ew.world
                        .resource_mut::<AssetDb>()
                        .set_settings(&guid, spec.settings);
                    let vfs_path = format!("{source}/{path}");
                    self.asset_browser
                        .dispatch_write(&self.vfs, &vfs_path, Vec::new());
                    self.asset_browser.mark_db_dirty(&source);
                    self.asset_browser
                        .notify_asset_created(&source, &dir, &path);
                    log::info!("Created asset: {vfs_path}");
                }
                None => log::warn!("Cannot create asset of kind '{kind}' (missing parent?)"),
            }
        }

        // Rename an asset (browser context menu → inline edit): same dir, new name.
        if let Some((source, old_path, new_name)) = self.asset_browser.take_pending_rename() {
            let dir = old_path.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
            let new_path = if dir.is_empty() {
                new_name
            } else {
                format!("{dir}/{new_name}")
            };
            self.perform_asset_move(&source, &old_path, &new_path);
        }

        // Move an asset (drag onto a directory): keep name, new directory.
        if let Some((source, old_path, new_dir)) = self.asset_browser.take_pending_move() {
            let name = old_path.rsplit('/').next().unwrap_or(&old_path).to_owned();
            let new_path = if new_dir.is_empty() {
                name
            } else {
                format!("{new_dir}/{name}")
            };
            self.perform_asset_move(&source, &old_path, &new_path);
        }

        // Delete an asset (browser context menu): ask for confirmation first.
        if let Some((source, path)) = self.asset_browser.take_pending_delete() {
            self.pending_delete_confirm = Some((source, path));
        }

        // Process component import (asset browser → inspector): dispatch read
        if let Some((vfs_path, entity)) = self.inspector_state.pending_component_import.take() {
            self.asset_browser.dispatch_read(&self.vfs, &vfs_path);
            self.pending_import = Some(PendingImport { vfs_path, entity });
        }

        // Process completed VFS reads for component import
        if let Some(pending) = &self.pending_import
            && let Some(idx) = self
                .asset_browser
                .completed_reads
                .iter()
                .position(|(path, _)| path == &pending.vfs_path)
        {
            let entity = pending.entity;
            let (path, data) = self.asset_browser.completed_reads.remove(idx);
            log::info!("Importing component from: {path}");
            self.pending_import = None;

            match redlilium_ecs::serialize::decode::<redlilium_ecs::serialize::SerializedComponent>(
                &data,
                redlilium_ecs::serialize::Format::Ron,
            ) {
                Ok(serialized) => {
                    let action = ImportComponentAction::new(entity, serialized);
                    let ew = self.world.as_mut().unwrap();
                    if let Err(e) = ew.history.execute(Box::new(action), &mut ew.world) {
                        log::warn!("Import action failed: {e}");
                    }
                }
                Err(e) => log::error!("Failed to decode .component file: {e}"),
            }
        }

        // Process prefab export (world inspector → asset browser)
        if let Some((root_entity, vfs_dir)) = self.asset_browser.pending_prefab_export.take() {
            let ew = self.world.as_ref().unwrap();
            if ew.world.is_alive(root_entity) {
                // Asset export: the prefab must be self-contained (external
                // entity references are rejected with a clear error).
                match ew.world.serialize_prefab_asset(root_entity) {
                    Ok(serialized) => {
                        match redlilium_ecs::serialize::encode(
                            &serialized,
                            redlilium_ecs::serialize::Format::Ron,
                        ) {
                            Ok(data) => {
                                let name = ew
                                    .world
                                    .get::<Name>(root_entity)
                                    .map(|n| n.as_str().to_owned())
                                    .unwrap_or_else(|| format!("Entity_{}", root_entity.index()));
                                let vfs_path = format!("{vfs_dir}/{name}.prefab");
                                log::info!("Exporting prefab to: {vfs_path}");
                                self.asset_browser
                                    .dispatch_write(&self.vfs, &vfs_path, data);
                            }
                            Err(e) => log::error!("Failed to encode prefab: {e}"),
                        }
                    }
                    Err(e) => log::error!("Failed to serialize prefab: {e}"),
                }
            }
        }

        // Process prefab import (asset browser → world inspector): dispatch read
        if let Some((vfs_path, parent)) = self.inspector_state.pending_prefab_import.take() {
            self.asset_browser.dispatch_read(&self.vfs, &vfs_path);
            self.pending_prefab_import = Some(PendingPrefabImport { vfs_path, parent });
        }

        // Process completed VFS reads for prefab import
        if let Some(pending) = &self.pending_prefab_import
            && let Some(idx) = self
                .asset_browser
                .completed_reads
                .iter()
                .position(|(path, _)| path == &pending.vfs_path)
        {
            let parent = pending.parent;
            let (path, data) = self.asset_browser.completed_reads.remove(idx);
            log::info!("Importing prefab from: {path}");
            self.pending_prefab_import = None;

            match redlilium_ecs::serialize::decode::<redlilium_ecs::serialize::SerializedPrefab>(
                &data,
                redlilium_ecs::serialize::Format::Ron,
            ) {
                Ok(serialized) => {
                    let action = SpawnPrefabAction::new(serialized, parent);
                    let ew = self.world.as_mut().unwrap();
                    if let Err(e) = ew.history.execute(Box::new(action), &mut ew.world) {
                        log::warn!("Prefab spawn action failed: {e}");
                    }
                }
                Err(e) => log::error!("Failed to decode .prefab file: {e}"),
            }
        }

        // Clear per-frame deltas *after* systems have consumed them
        {
            let ew = self.active_world();
            {
                let mut input = ew.window_input.write();
                input.begin_frame();
            }
        }

        // GPU uniform buffers are now updated automatically by the
        // UpdatePerEntityUniforms and SyncMaterialUniforms systems.

        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        #[allow(unused_variables)]
        let window = ctx.window().clone();
        let custom_titlebar = ctx.custom_titlebar();
        // One render graph per frame: scene passes and the egui overlay pass all
        // go into this graph; ordering is an explicit intra-graph dependency.
        let mut graph = ctx.acquire_graph();

        // The `Render` schedule (FlushUploads + ForwardRender) is run further
        // below, AFTER ensure_camera_target + the scene mesh uploads, so the
        // ForwardRender system renders into this frame's CameraTarget with the
        // meshes already uploaded.

        // Size the off-screen scene target to the SceneView panel (last frame's
        // physical rect) and make sure egui has a texture id for this frame's
        // image. The scene pass below fills this texture; egui samples it (the
        // egui pass depends on the scene pass).
        if let (Some(scene_view), Some(ew)) = (self.scene_view.as_mut(), self.world.as_mut()) {
            let (w, h) = self
                .scene_view_rect_phys
                .map_or((256, 256), |[_, _, w, h]| (w as u32, h as u32));
            let color = scene_view.ensure_camera_target(&mut ew.world, ew.editor_camera, w, h);
            // The `ensure_camera_target` mutable-world borrow has ended; access the
            // egui controller resource to (re)register the scene texture.
            let mut egui = ew.world.resource_mut::<EguiController>();
            match self.scene_texture_id {
                Some(id) => egui.update_user_texture(id, color),
                None => self.scene_texture_id = Some(egui.register_user_texture(color)),
            }
        }

        let mut scene_view_rect = None;
        let mut pixels_per_point = 1.0;

        if self.world.is_some() {
            let elapsed = ctx.elapsed_time() as f64;

            // begin_frame + grab the (cloned) egui context; the dock below builds
            // UI through the context, so we don't hold the controller resource
            // across it. (end_frame happens at the bottom, also via the resource.)
            let egui_ctx = {
                let ew = self.world.as_ref().unwrap();
                let mut egui = ew.world.resource_mut::<EguiController>();
                egui.begin_frame(elapsed);
                egui.context().clone()
            };
            pixels_per_point = egui_ctx.pixels_per_point();

            // macOS: reserve space for the native titlebar area (traffic lights)
            // Contains centered play controls. Double-click toggles maximize.
            #[cfg(target_os = "macos")]
            if custom_titlebar {
                egui::TopBottomPanel::top("macos_titlebar_spacer")
                    .exact_height(28.0)
                    .show_separator_line(false)
                    .show(&egui_ctx, |ui| {
                        ui.horizontal_centered(|ui| {
                            // Traffic lights occupy ~70px on the left
                            let available = ui.available_width();
                            ui.add_space((available / 2.0 - 40.0).max(0.0));

                            let ew = self.world.as_ref().unwrap();
                            let paused_due_to_panic = ew
                                .world
                                .resource::<redlilium_ecs::PlayControl>()
                                .panic_info()
                                .is_some();

                            if let Some(play_action) = crate::toolbar::draw_play_controls(
                                ui,
                                self.play_state,
                                paused_due_to_panic,
                            ) {
                                self.apply_play_action(play_action);
                            }

                            // Phase 6: Play mode indicator badge
                            ui.add_space(8.0);
                            crate::toolbar::draw_play_mode_indicator(ui, self.play_state);
                        });

                        // Double-click on background toggles maximize
                        let bar_rect = ui.min_rect();
                        let pointer_in_bar = ui
                            .input(|i| i.pointer.interact_pos())
                            .is_some_and(|pos| bar_rect.contains(pos));
                        if pointer_in_bar
                            && !ui.ctx().is_using_pointer()
                            && ui.input(|i| {
                                i.pointer
                                    .button_double_clicked(egui::PointerButton::Primary)
                            })
                        {
                            window.set_maximized(!window.is_maximized());
                        }
                    });
            }

            // Menu bar with play controls (egui fallback for non-macOS platforms)
            #[cfg(not(target_os = "macos"))]
            {
                let ew = self.world.as_ref().unwrap();
                let paused_due_to_panic = ew
                    .world
                    .resource::<redlilium_ecs::PlayControl>()
                    .panic_info()
                    .is_some();

                let result = menu::draw_menu_bar(
                    &egui_ctx,
                    &window,
                    custom_titlebar,
                    self.play_state,
                    paused_due_to_panic,
                );
                if let Some(play_action) = result.play_action {
                    self.apply_play_action(play_action);
                }
                if let Some(action) = result.action {
                    use crate::menu::MenuAction;
                    match action {
                        MenuAction::CloseWindow => {
                            if self.has_unsaved_changes() {
                                self.show_close_dialog = true;
                            } else {
                                self.should_close = true;
                            }
                        }
                        _ if self.world.is_some() => {
                            let ew = self.world.as_mut().unwrap();
                            match action {
                                MenuAction::Save => {
                                    ew.history.mark_saved();
                                    log::info!("Saved");
                                }
                                MenuAction::Undo => {
                                    if let Err(e) = ew.history.undo(&mut ew.world) {
                                        log::warn!("Undo failed: {e}");
                                    }
                                }
                                MenuAction::Redo => {
                                    if let Err(e) = ew.history.redo(&mut ew.world) {
                                        log::warn!("Redo failed: {e}");
                                    }
                                }
                                _ => log::info!("Menu action: {action:?}"),
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Update smoothed FPS from frame delta
            let dt = elapsed as f32;
            if dt > 0.0 {
                let instant_fps = 1.0 / dt;
                if self.fps == 0.0 {
                    self.fps = instant_fps;
                } else {
                    self.fps += (instant_fps - self.fps) * 0.05;
                }
            }

            // Phase 6: Full-screen Play mode
            // During Playing, hide editor UI and show only the game viewport
            // During Paused, show editor UI but lock editing (read-only inspection)
            let is_playing = self.play_state == PlayState::Playing;

            // Status bar (bottom) — hidden during Play, visible during Pause and Edit
            if !is_playing {
                status_bar::draw_status_bar(&egui_ctx, self.fps);
            }

            // Dock area fills remaining space (transparent, no margin so it spans edge-to-edge)
            let panel_frame = egui::Frame::NONE.fill(egui::Color32::TRANSPARENT);
            egui::CentralPanel::default()
                .frame(panel_frame)
                .show(&egui_ctx, |ui| {
                    let ew = self.world.as_mut();
                    if let Some(ew) = ew {
                        // Compute drag selection rect in egui logical points.
                        let drag_rect = if self.dragging_box
                            && let Some(start) = self.drag_start
                        {
                            let inv = 1.0 / pixels_per_point;
                            let x0 = start[0] * inv;
                            let y0 = start[1] * inv;
                            let x1 = self.cursor_pos[0] * inv;
                            let y1 = self.cursor_pos[1] * inv;
                            Some(egui::Rect::from_two_pos(
                                egui::pos2(x0, y0),
                                egui::pos2(x1, y1),
                            ))
                        } else {
                            None
                        };

                        if is_playing {
                            // Full-screen game viewport during Play mode
                            if let Some(texture_id) = self.scene_texture_id {
                                let available = ui.available_rect_before_wrap();
                                ui.painter().image(
                                    texture_id,
                                    available,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    egui::Color32::WHITE,
                                );
                                scene_view_rect = Some(available);
                            }
                        } else {
                            // Normal dock layout during Edit and Pause modes
                            let mut tab_viewer = EditorTabViewer {
                                world: &mut ew.world,
                                inspector_state: &mut self.inspector_state,
                                vfs: &self.vfs,
                                asset_browser: &mut self.asset_browser,
                                console: &mut self.console,
                                history: &ew.history,
                                scene_view_rect: None,
                                drag_rect,
                                scene_texture: self.scene_texture_id,
                            };
                            let mut dock_style = egui_dock::Style::from_egui(ui.style().as_ref());
                            dock_style.tab_bar.corner_radius = egui::CornerRadius::ZERO;
                            dock_style.tab_bar.bg_fill = crate::theme::BG;
                            dock_style.tab.active.corner_radius = egui::CornerRadius::ZERO;
                            dock_style.tab.active.bg_fill = crate::theme::SURFACE1;
                            dock_style.tab.active.text_color = crate::theme::TEXT_PRIMARY;
                            dock_style.tab.inactive.corner_radius = egui::CornerRadius::ZERO;
                            dock_style.tab.inactive.bg_fill = crate::theme::BG;
                            dock_style.tab.inactive.text_color = crate::theme::TEXT_MUTED;
                            dock_style.tab.focused.corner_radius = egui::CornerRadius::ZERO;
                            dock_style.tab.focused.bg_fill = crate::theme::SURFACE1;
                            dock_style.tab.focused.text_color = crate::theme::TEXT_PRIMARY;
                            dock_style.tab.hovered.corner_radius = egui::CornerRadius::ZERO;
                            dock_style.tab.hovered.bg_fill = crate::theme::SURFACE3;
                            dock_style.tab.hovered.text_color = crate::theme::TEXT_PRIMARY;
                            dock_style.tab.inactive_with_kb_focus.corner_radius =
                                egui::CornerRadius::ZERO;
                            dock_style.tab.active_with_kb_focus.corner_radius =
                                egui::CornerRadius::ZERO;
                            dock_style.tab.focused_with_kb_focus.corner_radius =
                                egui::CornerRadius::ZERO;
                            dock_style.tab.tab_body.corner_radius = egui::CornerRadius::ZERO;
                            dock_style.main_surface_border_rounding = egui::CornerRadius::ZERO;
                            dock_style.separator.color_idle = crate::theme::BORDER;
                            dock_style.separator.color_hovered = crate::theme::ACCENT_HOVER;
                            dock_style.separator.color_dragged = crate::theme::ACCENT;

                            egui_dock::DockArea::new(&mut self.dock_state)
                                .style(dock_style)
                                .show_leaf_collapse_buttons(false)
                                .show_inside(ui, &mut tab_viewer);
                            scene_view_rect = tab_viewer.scene_view_rect;

                            // Floating label near cursor while dragging
                            show_drag_overlay(ui.ctx(), tab_viewer.world);
                        }
                    }
                });

            // Modal "Unsaved Changes" dialog
            if self.show_close_dialog {
                // Full-screen dimming overlay that captures all interaction.
                egui::Area::new("close_dialog_overlay".into())
                    .fixed_pos(egui::pos2(0.0, 0.0))
                    .order(egui::Order::Foreground)
                    .interactable(true)
                    .show(&egui_ctx, |ui| {
                        let screen = ui.ctx().input(|i| i.viewport_rect());
                        ui.allocate_rect(screen, egui::Sense::click());
                        ui.painter().rect_filled(
                            screen,
                            egui::CornerRadius::ZERO,
                            egui::Color32::from_black_alpha(128),
                        );
                    });

                // Dialog window, centered, above the overlay.
                egui::Window::new("Unsaved Changes")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                    .order(egui::Order::Foreground)
                    .show(&egui_ctx, |ui| {
                        ui.label("You have unsaved changes. What would you like to do?");
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button("Save").clicked() {
                                if let Some(ew) = &mut self.world {
                                    ew.history.mark_saved();
                                }
                                self.show_close_dialog = false;
                                self.should_close = true;
                            }
                            if ui.button("Don't Save").clicked() {
                                self.show_close_dialog = false;
                                self.should_close = true;
                            }
                            if ui.button("Cancel").clicked() {
                                self.show_close_dialog = false;
                            }
                        });
                    });
            }

            // Modal "Delete asset?" confirmation.
            if let Some((source, path)) = self.pending_delete_confirm.clone() {
                egui::Area::new("delete_dialog_overlay".into())
                    .fixed_pos(egui::pos2(0.0, 0.0))
                    .order(egui::Order::Foreground)
                    .interactable(true)
                    .show(&egui_ctx, |ui| {
                        let screen = ui.ctx().input(|i| i.viewport_rect());
                        ui.allocate_rect(screen, egui::Sense::click());
                        ui.painter().rect_filled(
                            screen,
                            egui::CornerRadius::ZERO,
                            egui::Color32::from_black_alpha(128),
                        );
                    });

                egui::Window::new("Delete asset?")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
                    .order(egui::Order::Foreground)
                    .show(&egui_ctx, |ui| {
                        ui.label(format!("Delete \"{source}/{path}\"?"));
                        ui.weak("This deletes the file and cannot be undone.");
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            if ui.button("Delete").clicked() {
                                self.perform_asset_delete(&source, &path);
                                self.pending_delete_confirm = None;
                            }
                            if ui.button("Cancel").clicked() {
                                self.pending_delete_confirm = None;
                            }
                        });
                    });
            }

            // Store egui input state for next frame.
            // `wants_pointer_input()` is too broad — it returns true for ALL egui
            // areas including dock panels that contain the scene view. Instead,
            // check if a popup/foreground/tooltip layer is at the pointer position.
            self.egui_wants_pointer = egui_ctx
                .pointer_latest_pos()
                .and_then(|pos| egui_ctx.layer_id_at(pos))
                .is_some_and(|layer| {
                    matches!(
                        layer.order,
                        egui::Order::Foreground | egui::Order::Tooltip | egui::Order::Debug
                    )
                });
            self.egui_wants_keyboard = egui_ctx.wants_keyboard_input();
            // end_frame (finishing the egui frame, producing its draw pass, and
            // flushing atlas uploads) is now done by the `EguiRender` system in the
            // Render schedule below — it reads the `FrameTarget` resource set there.
        }

        // Update viewport/scissor from SceneView panel rect (outside egui block)
        if let Some(rect) = scene_view_rect {
            // Store physical-pixel rect for input hit-testing
            self.scene_view_rect_phys = Some([
                rect.min.x * pixels_per_point,
                rect.min.y * pixels_per_point,
                rect.width() * pixels_per_point,
                rect.height() * pixels_per_point,
            ]);
            if let Some(sv) = &mut self.scene_view {
                sv.set_viewport(rect, pixels_per_point);
            }
            if self.world.is_some() {
                let aspect = self
                    .scene_view
                    .as_ref()
                    .map(|sv| sv.aspect_ratio())
                    .unwrap_or(1.0);
                self.update_camera_projection(aspect);
            }
        } else {
            // SceneView tab not visible — clear viewport so scene rendering is skipped.
            self.scene_view_rect_phys = None;
            if let Some(sv) = &mut self.scene_view {
                sv.clear_viewport();
            }
        }

        // Scene passes clear the swapchain; the egui overlay draws on top. Both
        // are in the single frame graph; the egui pass is made to depend on the
        // scene pass so the order is deterministic for the Strict compiler.

        // Take pending pick/rect coordinates before the immutable borrow of scene_view.
        let pending_pick = self
            .scene_view
            .as_mut()
            .and_then(|sv| sv.take_pending_pick());
        let pending_rect = self
            .scene_view
            .as_mut()
            .and_then(|sv| sv.take_pending_rect_pick());

        // The forward fill is done by the ForwardRender system; here we fill only
        // the picking ring and upload scene meshes (before the scene pass).
        if let (Some(scene_view), Some(ew)) = (&mut self.scene_view, self.world.as_ref())
            && scene_view.has_viewport()
        {
            scene_view.fill_picking_rings(&ew.world);
            scene_view.flush_uploads(&mut graph);
        }

        let render_active =
            self.scene_view.as_ref().is_some_and(|sv| sv.has_viewport()) && self.world.is_some();

        // Provide this frame's swapchain target to the render systems — the
        // EguiRender system composites the egui overlay onto it. Set every frame
        // (the surface texture is per-frame).
        if let Some(ew) = self.world.as_mut() {
            let target = RenderTarget::from_surface(ctx.swapchain_texture());
            ew.world.insert_resource(FrameTarget {
                target,
                width: ctx.width(),
                height: ctx.height(),
            });
        }

        // Run the `Render` schedule (FlushUploads -> ForwardRender -> DebugRender ->
        // EguiRender). Always run it so the egui overlay is drawn even when the
        // scene view is hidden; the scene/debug systems no-op without a viewport.
        // EguiRender adds the egui pass and depends it on the last CameraTarget
        // writer (`ScenePass`), so no editor-side dependency wiring is needed.
        if let Some(ew) = self.world.as_mut() {
            ew.world.resource_mut::<RenderSchedule>().set(graph);
            ew.schedules
                .run_schedule::<Render>(&mut ew.world, &self.runner);
            graph = ew
                .world
                .resource_mut::<RenderSchedule>()
                .take()
                .expect("RenderSchedule must hold the graph after the Render schedule");
        }

        // Remote screenshot: copy this frame's scene target through the graph.
        if let (Some(rc), Some(sv), Some(ew)) = (
            &mut self.remote,
            self.scene_view.as_ref(),
            self.world.as_ref(),
        ) {
            crate::remote_commands::inject_screenshot_pass(rc, &ew.world, sv.device(), &mut graph);
        }

        if render_active && let Some(ew) = self.world.as_ref() {
            let scene_view = self.scene_view.as_ref().unwrap();

            // Entity index pass (picking) — independent target + depth now.
            if let Some(ei_pass) = scene_view.build_entity_index_pass(&ew.world) {
                let ei_handle = graph.add_graphics_pass(ei_pass);
                if let Some([px, py]) = pending_pick {
                    let readback_pass = scene_view.build_pick_readback(px, py);
                    let readback_handle = graph.add_transfer_pass(readback_pass);
                    graph.add_dependency(readback_handle, ei_handle);
                }
                if let Some([rx, ry, rw, rh]) = pending_rect {
                    let rect_readback_pass = scene_view.build_rect_readback(rx, ry, rw, rh);
                    let rect_rb_handle = graph.add_transfer_pass(rect_readback_pass);
                    graph.add_dependency(rect_rb_handle, ei_handle);
                }
            }
        }

        // Record the rect readback layout (for decoding the async result) after
        // the immutable borrow is released. Single-pixel picks need no state —
        // their result is polled from `pick_result`.
        if let Some([_, _, rw, rh]) = pending_rect
            && let Some(scene_view) = &mut self.scene_view
        {
            scene_view.set_rect_layout(rw, rh);
        }

        ctx.render(graph)
    }

    fn on_mouse_move(&mut self, _ctx: &mut AppContext, x: f64, y: f64) {
        self.cursor_pos = [x as f32, y as f32];
        self.with_egui(|egui| {
            egui.on_mouse_move(x, y);
        });
        if self.world.is_some() {
            let ew = self.active_world();
            {
                let mut input = ew.window_input.write();
                input.on_mouse_move(x, y);
            }
        }

        // Detect drag threshold for box selection
        if let Some(start) = self.drag_start
            && !self.dragging_box
        {
            let dx = self.cursor_pos[0] - start[0];
            let dy = self.cursor_pos[1] - start[1];
            if (dx * dx + dy * dy).sqrt() > 5.0 {
                self.dragging_box = true;
            }
        }
    }

    fn on_mouse_button(&mut self, _ctx: &mut AppContext, button: MouseButton, pressed: bool) {
        self.with_egui(|egui| {
            egui.on_mouse_button(button, pressed);
        });

        // Phase 6: Block edit operations during Play mode (game input still flows through)
        let is_playing = self.play_state == PlayState::Playing;

        // Releasing the primary button ends an edit gesture: the next recorded
        // action starts a fresh undo entry (drag-long edits keep coalescing
        // while the button is held).
        if button == MouseButton::Left
            && !pressed
            && !is_playing
            && let Some(ew) = &mut self.world
        {
            ew.history.break_merge();
        }
        // Only forward presses when cursor is in scene view and egui doesn't
        // want the pointer (e.g. color picker popup over scene view).
        // Always forward releases so buttons don't get stuck.
        if self.world.is_some()
            && (!pressed || (self.cursor_in_scene_view() && !self.egui_wants_pointer))
        {
            let idx = match button {
                MouseButton::Left => 0,
                MouseButton::Right => 1,
                MouseButton::Middle => 2,
                _ => return,
            };
            let ew = self.active_world();
            {
                let mut input = ew.window_input.write();
                input.on_mouse_button(idx, pressed);
            }

            // LMB press: start potential drag for box selection.
            // LMB release: if it was a small movement → single-click GPU pick,
            // otherwise → box selection of all entities in the rectangle.
            // Phase 6: Block drag selection during Play mode
            if button == MouseButton::Left && self.scene_view_rect_phys.is_some() && !is_playing {
                if pressed && self.cursor_in_scene_view() && !self.egui_wants_pointer {
                    self.drag_start = Some(self.cursor_pos);
                    self.dragging_box = false;
                    // Clear selection immediately on click; the GPU pick or
                    // box selection will re-select if anything is hit.
                    if let Some(ew) = &mut self.world {
                        let action: Box<dyn redlilium_core::abstract_editor::EditAction<World>> =
                            Box::new(SelectAction::clear());
                        let _ = ew.history.execute(action, &mut ew.world);
                    }
                } else if !pressed {
                    if self.dragging_box {
                        // Box selection: select entities whose screen AABBs
                        // intersect the drag rectangle.
                        if let Some(start) = self.drag_start {
                            self.perform_box_selection(start, self.cursor_pos);
                        }
                    } else if let Some(scene_view) = &mut self.scene_view
                        && self.drag_start.is_some()
                    {
                        // Single click: GPU pick at cursor position.
                        let px = self.cursor_pos[0].max(0.0) as u32;
                        let py = self.cursor_pos[1].max(0.0) as u32;
                        scene_view.request_pick(px, py);
                    }
                    self.drag_start = None;
                    self.dragging_box = false;
                }
            }
        }
    }

    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, dx: f32, dy: f32) {
        self.with_egui(|egui| {
            egui.on_mouse_scroll(MouseScrollDelta::LineDelta(dx, dy));
        });
        if self.world.is_some() && self.cursor_in_scene_view() && !self.egui_wants_pointer {
            let ew = self.active_world();
            {
                let mut input = ew.window_input.write();
                input.on_scroll(dx, dy);
            }
        }
    }

    fn on_file_dropped(&mut self, _ctx: &mut AppContext, path: std::path::PathBuf) {
        self.with_egui(|egui| egui.on_file_dropped(path));
    }

    fn on_file_hovered(&mut self, _ctx: &mut AppContext, path: std::path::PathBuf) {
        self.with_egui(|egui| egui.on_file_hovered(path));
    }

    fn on_file_hover_cancelled(&mut self, _ctx: &mut AppContext) {
        self.with_egui(|egui| egui.on_file_hover_cancelled());
    }

    fn on_modifiers_changed(
        &mut self,
        _ctx: &mut AppContext,
        modifiers: winit::keyboard::ModifiersState,
    ) {
        self.with_egui(|egui| egui.on_modifiers_changed(modifiers));
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        self.with_egui(|egui| {
            egui.on_key(event);
        });
        // Only forward key presses when egui doesn't want keyboard; always
        // forward releases so keys don't get stuck.
        if self.world.is_some()
            && (!event.state.is_pressed() || !self.egui_wants_keyboard)
            && let PhysicalKey::Code(winit_key) = event.physical_key
            && let Some(key) = redlilium_app::input::map_winit_key(winit_key)
        {
            let ew = self.active_world();
            {
                let mut input = ew.window_input.write();
                if event.state.is_pressed() {
                    input.on_key_pressed(key);
                } else {
                    input.on_key_released(key);
                }
            }
        }
    }
}

/// Show a floating label near the cursor for any active drag payload.
fn show_drag_overlay(ctx: &egui::Context, world: &World) {
    let label = if let Some(entity) = egui::DragAndDrop::payload::<Entity>(ctx) {
        let name = world
            .get::<Name>(*entity)
            .map(|n| n.as_str().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("Entity({})", entity.index()));
        Some(name)
    } else if let Some(comp) = egui::DragAndDrop::payload::<ComponentDragPayload>(ctx) {
        Some(comp.name.to_string())
    } else if let Some(file) = egui::DragAndDrop::payload::<ComponentFileDragPayload>(ctx) {
        Some(
            file.vfs_path
                .rsplit('/')
                .next()
                .unwrap_or(&file.vfs_path)
                .to_string(),
        )
    } else if let Some(file) = egui::DragAndDrop::payload::<PrefabFileDragPayload>(ctx) {
        Some(
            file.vfs_path
                .rsplit('/')
                .next()
                .unwrap_or(&file.vfs_path)
                .to_string(),
        )
    } else {
        egui::DragAndDrop::payload::<redlilium_ecs::AssetDragPayload>(ctx).map(|file| {
            let name = file.vfs_path.rsplit('/').next().unwrap_or(&file.vfs_path);
            match &file.asset {
                Some((_, kind)) => format!("{name} ({kind})"),
                None => name.to_string(),
            }
        })
    };

    if let Some(label) = label
        && let Some(pos) = ctx.input(|i| i.pointer.hover_pos())
    {
        egui::Area::new(egui::Id::new("drag_overlay"))
            .fixed_pos(pos + egui::vec2(12.0, 4.0))
            .order(egui::Order::Tooltip)
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(label);
                });
            });
    }
}
