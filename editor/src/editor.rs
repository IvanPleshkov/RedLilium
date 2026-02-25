use std::f32::consts::FRAC_PI_4;
use std::sync::{Arc, RwLock};

use egui_dock::DockState;
use redlilium_app::{AppContext, AppHandler, DrawContext};
use redlilium_core::abstract_editor::{ActionQueue, DEFAULT_MAX_UNDO, EditActionHistory};
use redlilium_core::math::{Vec3, mat4_to_cols_array_2d};
use redlilium_core::mesh::generators;
use redlilium_debug_drawer::{DebugDrawer, DebugDrawerRenderer};
use redlilium_ecs::ui::{
    ComponentDragPayload, ComponentFileDragPayload, ImportComponentAction, InspectorState,
    PrefabFileDragPayload, SelectAction, SpawnPrefabAction,
};
use redlilium_ecs::{
    Camera, DrawGrid, DrawSelectionAabb, EcsRunner, Entity, FreeFlyCamera, GlobalTransform,
    GridConfig, MaterialManager, Name, PerEntityBuffers, PostUpdate, RenderMesh, Schedules,
    SyncMaterialUniforms, Transform, Update, UpdateCameraMatrices, UpdateFreeFlyCamera,
    UpdateGlobalTransforms, UpdatePerEntityUniforms, Visibility, WindowInput, World,
    register_std_components,
};
use redlilium_graphics::egui::{EguiApp, EguiController};
use redlilium_graphics::{FrameSchedule, RenderTarget, TextureFormat};
use redlilium_vfs::Vfs;
use winit::event::{KeyEvent, MouseButton, MouseScrollDelta};
use winit::keyboard::PhysicalKey;

use crate::asset_browser::AssetBrowser;
use crate::console::ConsolePanel;
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

pub struct Editor {
    // Multi-world support
    worlds: Vec<EditorWorld>,
    active_world: usize,
    runner: EcsRunner,

    // VFS and asset browser
    vfs: Vfs,
    asset_browser: AssetBrowser,
    console: ConsolePanel,

    // UI
    egui_controller: Option<EguiController>,
    dock_state: DockState<Tab>,
    inspector_state: InspectorState,
    play_state: PlayState,
    #[cfg(target_os = "macos")]
    native_menu: Option<NativeMenu>,

    // Scene rendering
    scene_view: Option<SceneViewState>,
    debug_drawer_renderer: Option<DebugDrawerRenderer>,

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
        let (config, vfs) = crate::project::load_or_default(project_path);
        let asset_browser = AssetBrowser::new(&config);
        let console = ConsolePanel::new(crate::log_capture::log_buffer());

        Self {
            worlds: Vec::new(),
            active_world: 0,
            runner: EcsRunner::single_thread(),
            vfs,
            asset_browser,
            console,
            egui_controller: None,
            dock_state: dock::create_default_layout(),
            inspector_state: InspectorState::new(),
            play_state: PlayState::Editing,
            #[cfg(target_os = "macos")]
            native_menu: None,
            scene_view: None,
            debug_drawer_renderer: None,
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
            should_close: false,
        }
    }

    /// Create a new editor world with a simple demo scene.
    fn create_editor_world(&self, scene_view: &SceneViewState, aspect: f32) -> EditorWorld {
        let mut world = World::new();
        register_std_components(&mut world);
        redlilium_ecs::register_rendering_components(&mut world);

        // Insert MaterialManager resource (provides GraphicsDevice to GPU sync systems)
        world.insert_resource(MaterialManager::new(scene_view.device().clone()));

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
        let cpu_cube = generators::generate_cube(0.5);
        let cube_aabb = cpu_cube.compute_aabb();

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

            let (per_entity, render_mat, mesh) = scene_view.create_entity_resources(&cpu_cube);
            let render_mesh = match cube_aabb {
                Some(aabb) => RenderMesh::with_aabb(mesh, aabb),
                None => RenderMesh::new(mesh),
            };
            world.insert(entity, render_mesh).unwrap();
            world.insert(entity, render_mat).unwrap();
            world.insert(entity, per_entity).unwrap();
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

            let (per_entity, render_mat, mesh) = scene_view.create_entity_resources(&cpu_cube);
            let render_mesh = match cube_aabb {
                Some(aabb) => RenderMesh::with_aabb(mesh, aabb),
                None => RenderMesh::new(mesh),
            };
            world.insert(entity, render_mesh).unwrap();
            world.insert(entity, render_mat).unwrap();
            world.insert(entity, per_entity).unwrap();
        }

        // Insert ActionQueue for editor action dispatch
        world.insert_resource(ActionQueue::<World>::new());

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

        // Automatic GPU sync systems — run after camera matrices are computed.
        schedules
            .get_mut::<PostUpdate>()
            .add(UpdatePerEntityUniforms);
        schedules.get_mut::<PostUpdate>().add(SyncMaterialUniforms);
        schedules
            .get_mut::<PostUpdate>()
            .add_edge::<UpdateCameraMatrices, UpdatePerEntityUniforms>()
            .expect("No cycle");
        schedules
            .get_mut::<PostUpdate>()
            .add_edge::<UpdateCameraMatrices, SyncMaterialUniforms>()
            .expect("No cycle");

        EditorWorld {
            world,
            schedules,
            history: EditActionHistory::new(DEFAULT_MAX_UNDO),
            editor_camera,
            window_input: window_input_handle,
            debug_drawer: debug_drawer_handle,
        }
    }

    /// Get the active editor world (immutable).
    fn active_world(&self) -> &EditorWorld {
        &self.worlds[self.active_world]
    }

    /// Returns `true` if any editor world has unsaved changes.
    fn has_unsaved_changes(&self) -> bool {
        self.worlds
            .iter()
            .any(|ew| ew.history.has_unsaved_changes())
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
        if let Ok(mut input) = ew.window_input.write() {
            input.ui_wants_input = !self.cursor_in_scene_view();
        }
    }

    /// Update the editor camera's projection on resize.
    fn update_camera_projection(&self, aspect: f32) {
        let ew = self.active_world();
        if let Ok(mut cameras) = ew.world.write_all::<Camera>()
            && let Some(cam) = cameras.get_mut(ew.editor_camera.index())
        {
            cam.projection_matrix =
                redlilium_core::math::perspective_rh(FRAC_PI_4, aspect, 0.1, 500.0);
        }
    }
}

impl AppHandler for Editor {
    fn on_init(&mut self, ctx: &mut AppContext) {
        log::info!("Editor initialized");

        let null_app: Arc<RwLock<dyn EguiApp>> = Arc::new(RwLock::new(NullEguiApp));
        self.egui_controller = Some(EguiController::new(
            ctx.device().clone(),
            null_app,
            ctx.width(),
            ctx.height(),
            ctx.scale_factor(),
            ctx.surface_format(),
        ));

        // Create scene view GPU resources
        let mut scene_view = SceneViewState::new(ctx.device().clone(), ctx.surface_format());
        scene_view.resize_if_needed(ctx.width(), ctx.height());

        // Create the first editor world with a demo scene
        let aspect = ctx.aspect_ratio();
        let editor_world = self.create_editor_world(&scene_view, aspect);
        self.worlds.push(editor_world);

        self.scene_view = Some(scene_view);

        // Create debug drawer renderer (with depth testing against scene)
        self.debug_drawer_renderer = Some(DebugDrawerRenderer::new(
            ctx.device().clone(),
            ctx.surface_format(),
            Some(TextureFormat::Depth32Float),
        ));

        // Run startup schedules
        let runner = &self.runner;
        let ew = &mut self.worlds[0];
        ew.schedules.run_startup(&mut ew.world, runner);

        // Create native menu after the event loop / NSApplication is initialized
        #[cfg(target_os = "macos")]
        {
            self.native_menu = Some(NativeMenu::new());
        }
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_resize(ctx.width(), ctx.height());
        }
        if let Some(sv) = &mut self.scene_view {
            sv.resize_if_needed(ctx.width(), ctx.height());
        }
        // Update camera projection for new aspect ratio
        if !self.worlds.is_empty()
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

        if self.worlds.is_empty() {
            return true;
        }

        // Resolve GPU pick from the previous frame's readback
        if let Some(scene_view) = &mut self.scene_view
            && let Some(entity_index) = scene_view.resolve_pick()
        {
            let ew = &mut self.worlds[self.active_world];
            // Find entity whose sparse-set index matches the picked index
            let target = ew
                .world
                .read::<PerEntityBuffers>()
                .ok()
                .and_then(|buffers| {
                    buffers
                        .iter()
                        .find(|(idx, _)| *idx == entity_index)
                        .map(|(_, _)| ())
                });
            // Reconstruct full Entity from world (need spawn_tick etc.)
            let target_entity = if target.is_some() {
                ew.world.entity_at_index(entity_index)
            } else {
                None
            };
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

        // Resolve GPU rect pick from the previous frame's readback
        if let Some(scene_view) = &mut self.scene_view
            && let Some(entity_indices) = scene_view.resolve_rect_pick()
        {
            let ew = &mut self.worlds[self.active_world];
            let selected: Vec<Entity> = if let Ok(buffers) = ew.world.read::<PerEntityBuffers>() {
                entity_indices
                    .iter()
                    .filter_map(|&idx| {
                        // Verify the index has a PerEntityBuffers component
                        if buffers.get(idx).is_some() {
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

        // Poll native menu events (macOS only)
        #[cfg(target_os = "macos")]
        if let Some(menu) = &self.native_menu
            && let Some(action) = menu.poll_event()
        {
            use crate::menu::MenuAction;
            let ew = &mut self.worlds[self.active_world];
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

        // Advance debug drawer tick (systems will write to the new tick)
        {
            let ew = self.active_world();
            if let Ok(drawer) = ew.debug_drawer.read() {
                drawer.advance_tick();
            }
        }

        // Run ECS schedules (always run in editing mode for camera/transforms)
        {
            let Editor {
                worlds,
                active_world,
                runner,
                ..
            } = self;
            let ew = &mut worlds[*active_world];
            ew.schedules
                .run_frame(&mut ew.world, runner, ctx.delta_time() as f64);
        }

        // Drain action queue and execute through history
        {
            let ew = &mut self.worlds[self.active_world];
            let actions = ew.world.resource::<ActionQueue<World>>().drain();
            for action in actions {
                if let Err(e) = ew.history.execute(action, &mut ew.world) {
                    log::warn!("Action failed: {e}");
                }
            }
        }

        // Process component export (inspector → asset browser)
        if let Some((entity, comp_name, vfs_dir)) =
            self.asset_browser.pending_component_export.take()
        {
            let ew = &self.worlds[self.active_world];
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
                    let ew = &mut self.worlds[self.active_world];
                    if let Err(e) = ew.history.execute(Box::new(action), &mut ew.world) {
                        log::warn!("Import action failed: {e}");
                    }
                }
                Err(e) => log::error!("Failed to decode .component file: {e}"),
            }
        }

        // Process prefab export (world inspector → asset browser)
        if let Some((root_entity, vfs_dir)) = self.asset_browser.pending_prefab_export.take() {
            let ew = &self.worlds[self.active_world];
            if ew.world.is_alive(root_entity) {
                match ew.world.serialize_prefab(root_entity) {
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
                    let ew = &mut self.worlds[self.active_world];
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
            if let Ok(mut input) = ew.window_input.write() {
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
        let mut ui_graph = ctx.acquire_graph();
        let mut scene_view_rect = None;
        let mut pixels_per_point = 1.0;

        if let Some(egui) = &mut self.egui_controller {
            let width = ctx.width();
            let height = ctx.height();
            let elapsed = ctx.elapsed_time() as f64;
            let render_target = RenderTarget::from_surface(ctx.swapchain_texture());

            egui.begin_frame(elapsed);

            let egui_ctx = egui.context().clone();
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
                            self.play_state =
                                crate::toolbar::draw_play_controls(ui, self.play_state);
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
                let result =
                    menu::draw_menu_bar(&egui_ctx, &window, custom_titlebar, self.play_state);
                self.play_state = result.play_state;
                if let Some(action) = result.action {
                    use crate::menu::MenuAction;
                    match action {
                        MenuAction::CloseWindow => {
                            if self
                                .worlds
                                .iter()
                                .any(|ew| ew.history.has_unsaved_changes())
                            {
                                self.show_close_dialog = true;
                            } else {
                                self.should_close = true;
                            }
                        }
                        _ if !self.worlds.is_empty() => {
                            let ew = &mut self.worlds[self.active_world];
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

            // Status bar (bottom)
            status_bar::draw_status_bar(&egui_ctx, self.fps);

            // Dock area fills remaining space (transparent, no margin so it spans edge-to-edge)
            let panel_frame = egui::Frame::NONE.fill(egui::Color32::TRANSPARENT);
            egui::CentralPanel::default()
                .frame(panel_frame)
                .show(&egui_ctx, |ui| {
                    let ew = if self.worlds.is_empty() {
                        None
                    } else {
                        Some(&mut self.worlds[self.active_world])
                    };
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
                        let mut tab_viewer = EditorTabViewer {
                            world: &mut ew.world,
                            inspector_state: &mut self.inspector_state,
                            vfs: &self.vfs,
                            asset_browser: &mut self.asset_browser,
                            console: &mut self.console,
                            history: &ew.history,
                            scene_view_rect: None,
                            drag_rect,
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
                                for ew in &mut self.worlds {
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

            // Store egui input state for next frame
            self.egui_wants_pointer = egui_ctx.wants_pointer_input();
            self.egui_wants_keyboard = egui_ctx.wants_keyboard_input();

            if let Some(egui_pass) = egui.end_frame(&render_target, width, height) {
                ui_graph.add_graphics_pass(egui_pass);
            }
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
            if !self.worlds.is_empty() {
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

        // Submit scene first (clears swapchain), then egui on top
        let mut deps = Vec::new();

        // Take pending pick/rect coordinates before the immutable borrow of scene_view.
        let pending_pick = self
            .scene_view
            .as_mut()
            .and_then(|sv| sv.take_pending_pick());
        let pending_rect = self
            .scene_view
            .as_mut()
            .and_then(|sv| sv.take_pending_rect_pick());

        if let Some(scene_view) = &self.scene_view
            && scene_view.has_viewport()
            && !self.worlds.is_empty()
        {
            let ew = &self.worlds[self.active_world];
            if let Some(mut scene_pass) =
                scene_view.build_scene_pass(&ew.world, ctx.swapchain_texture())
            {
                // Append debug draw lines into the scene pass if available
                if let Some(renderer) = &mut self.debug_drawer_renderer
                    && let Ok(drawer) = ew.debug_drawer.read()
                {
                    let vertices = drawer.take_render_data();
                    if !vertices.is_empty() {
                        if let Ok(cameras) = ew.world.read_all::<Camera>()
                            && let Some((_, camera)) = cameras.iter().next()
                        {
                            renderer
                                .update_view_proj(mat4_to_cols_array_2d(&camera.view_projection()));
                        }
                        renderer.append_to_pass(&mut scene_pass, &vertices);
                    }
                }

                let mut scene_graph = ctx.acquire_graph();
                let scene_pass_handle = scene_graph.add_graphics_pass(scene_pass);

                // Entity index pass (for picking) — renders to R32Uint texture
                // Depends on scene pass because both write to the shared depth texture.
                if let Some(ei_pass) = scene_view.build_entity_index_pass(&ew.world) {
                    let ei_handle = scene_graph.add_graphics_pass(ei_pass);
                    scene_graph.add_dependency(ei_handle, scene_pass_handle);

                    // Single-pixel readback transfer.
                    if let Some([px, py]) = pending_pick {
                        log::info!("Submitting pick readback at ({px}, {py})");
                        let readback_pass = scene_view.build_pick_readback(px, py);
                        let readback_handle = scene_graph.add_transfer_pass(readback_pass);
                        scene_graph.add_dependency(readback_handle, ei_handle);
                    }

                    // Rect selection readback transfer.
                    if let Some([rx, ry, rw, rh]) = pending_rect {
                        let rect_readback_pass = scene_view.build_rect_readback(rx, ry, rw, rh);
                        let rect_rb_handle = scene_graph.add_transfer_pass(rect_readback_pass);
                        scene_graph.add_dependency(rect_rb_handle, ei_handle);
                    }
                }

                let scene_handle = ctx.submit("scene", scene_graph, &[]);
                deps.push(scene_handle);
            }
        }

        // Mark picks in flight after the immutable borrow is released.
        if pending_pick.is_some()
            && let Some(scene_view) = &mut self.scene_view
        {
            scene_view.set_pick_in_flight();
        }
        if let Some([_, _, rw, rh]) = pending_rect
            && let Some(scene_view) = &mut self.scene_view
        {
            scene_view.set_rect_pick_in_flight(rw, rh);
        }

        let _ui_handle = ctx.submit("editor_ui", ui_graph, &deps);

        ctx.finish(&[])
    }

    fn on_mouse_move(&mut self, _ctx: &mut AppContext, x: f64, y: f64) {
        self.cursor_pos = [x as f32, y as f32];
        if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_move(x, y);
        }
        if !self.worlds.is_empty() {
            let ew = self.active_world();
            if let Ok(mut input) = ew.window_input.write() {
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
        if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_button(button, pressed);
        }
        // Only forward presses when cursor is in scene view; always forward
        // releases so buttons don't get stuck.
        if !self.worlds.is_empty() && (!pressed || self.cursor_in_scene_view()) {
            let idx = match button {
                MouseButton::Left => 0,
                MouseButton::Right => 1,
                MouseButton::Middle => 2,
                _ => return,
            };
            let ew = self.active_world();
            if let Ok(mut input) = ew.window_input.write() {
                input.on_mouse_button(idx, pressed);
            }

            // LMB press: start potential drag for box selection.
            // LMB release: if it was a small movement → single-click GPU pick,
            // otherwise → box selection of all entities in the rectangle.
            if button == MouseButton::Left && self.scene_view_rect_phys.is_some() {
                if pressed && self.cursor_in_scene_view() {
                    self.drag_start = Some(self.cursor_pos);
                    self.dragging_box = false;
                    // Clear selection immediately on click; the GPU pick or
                    // box selection will re-select if anything is hit.
                    if !self.worlds.is_empty() {
                        let ew = &mut self.worlds[self.active_world];
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
        if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_scroll(MouseScrollDelta::LineDelta(dx, dy));
        }
        if !self.worlds.is_empty() && self.cursor_in_scene_view() {
            let ew = self.active_world();
            if let Ok(mut input) = ew.window_input.write() {
                input.on_scroll(dx, dy);
            }
        }
    }

    fn on_file_dropped(&mut self, _ctx: &mut AppContext, path: std::path::PathBuf) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_file_dropped(path);
        }
    }

    fn on_file_hovered(&mut self, _ctx: &mut AppContext, path: std::path::PathBuf) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_file_hovered(path);
        }
    }

    fn on_file_hover_cancelled(&mut self, _ctx: &mut AppContext) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_file_hover_cancelled();
        }
    }

    fn on_modifiers_changed(
        &mut self,
        _ctx: &mut AppContext,
        modifiers: winit::keyboard::ModifiersState,
    ) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_modifiers_changed(modifiers);
        }
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_key(event);
        }
        // Only forward key presses when egui doesn't want keyboard; always
        // forward releases so keys don't get stuck.
        if !self.worlds.is_empty()
            && (!event.state.is_pressed() || !self.egui_wants_keyboard)
            && let PhysicalKey::Code(winit_key) = event.physical_key
            && let Some(key) = redlilium_app::input::map_winit_key(winit_key)
        {
            let ew = self.active_world();
            if let Ok(mut input) = ew.window_input.write() {
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
    } else {
        egui::DragAndDrop::payload::<PrefabFileDragPayload>(ctx).map(|file| {
            file.vfs_path
                .rsplit('/')
                .next()
                .unwrap_or(&file.vfs_path)
                .to_string()
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
