use std::f32::consts::FRAC_PI_4;
use std::sync::{Arc, RwLock};

use egui_dock::DockState;
use redlilium_app::{AppContext, AppHandler, DrawContext};
use redlilium_core::abstract_editor::{ActionQueue, DEFAULT_MAX_UNDO, EditActionHistory};
use redlilium_core::math::{Vec3, mat4_to_cols_array_2d};
use redlilium_core::mesh::generators;
use redlilium_debug_drawer::{DebugDrawer, DebugDrawerRenderer};
use redlilium_ecs::ui::{ImportComponentAction, InspectorState, SpawnPrefabAction};
use redlilium_ecs::{
    Camera, DrawGrid, EcsRunner, Entity, FreeFlyCamera, GlobalTransform, GridConfig, Name,
    PostUpdate, RenderMaterial, RenderMesh, Schedules, Transform, Update, UpdateCameraMatrices,
    UpdateFreeFlyCamera, UpdateGlobalTransforms, Visibility, WindowInput, World,
    register_std_components,
};
use redlilium_graphics::egui::{EguiApp, EguiController};
use redlilium_graphics::{Buffer, FrameSchedule, RenderTarget, TextureFormat};
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
use crate::toolbar::{self, PlayState};

/// A minimal EguiApp that does nothing.
///
/// All actual UI rendering happens in [`Editor::on_draw`] using the egui
/// context directly between `begin_frame` / `end_frame`.
struct NullEguiApp;

impl EguiApp for NullEguiApp {
    fn update(&mut self, _ctx: &egui::Context) {}

    fn setup(&mut self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.window_corner_radius = egui::CornerRadius::same(4);
        ctx.set_style(style);
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
    /// Per-entity uniform buffers for scene rendering.
    pub entity_buffers: Vec<(Entity, Arc<Buffer>)>,
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

    /// Smoothed frames-per-second for the status bar.
    fps: f32,

    /// Pending component import from asset browser (VFS read in progress).
    pending_import: Option<PendingImport>,
    /// Pending prefab import from asset browser (VFS read in progress).
    pending_prefab_import: Option<PendingPrefabImport>,
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
            fps: 0.0,
            pending_import: None,
            pending_prefab_import: None,
        }
    }

    /// Create a new editor world with a simple demo scene.
    fn create_editor_world(&self, scene_view: &SceneViewState, aspect: f32) -> EditorWorld {
        let mut world = World::new();
        register_std_components(&mut world);
        redlilium_ecs::register_rendering_components(&mut world);

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

        // NOTE: We intentionally do NOT mark the editor camera as EDITOR here.
        // Standard systems (UpdateFreeFlyCamera, UpdateGlobalTransforms,
        // UpdateCameraMatrices) use Read/Write which skip editor-flagged entities.
        // Since this is an isolated editor world, the flag is unnecessary.

        // --- Demo scene entities ---
        let cpu_cube = generators::generate_cube(0.5);
        let mut entity_buffers = Vec::new();

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

            let (buffer, mesh, mat_inst) = scene_view.create_entity_resources(&cpu_cube);
            world.insert(entity, RenderMesh::new(mesh)).unwrap();
            world.insert(entity, RenderMaterial::new(mat_inst)).unwrap();
            entity_buffers.push((entity, buffer));
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

            let (buffer, mesh, mat_inst) = scene_view.create_entity_resources(&cpu_cube);
            world.insert(entity, RenderMesh::new(mesh)).unwrap();
            world.insert(entity, RenderMaterial::new(mat_inst)).unwrap();
            entity_buffers.push((entity, buffer));
        }

        // Insert ActionQueue for editor action dispatch
        world.insert_resource(ActionQueue::<World>::new());

        // --- Setup schedules ---
        let mut schedules = Schedules::new();

        // Update: read-only editor systems (debug grid, future interaction systems).
        // Systems here cannot mutate the world directly — they must push actions
        // through the ActionQueue resource.
        schedules.get_mut::<Update>().add(DrawGrid);
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

        EditorWorld {
            world,
            schedules,
            history: EditActionHistory::new(DEFAULT_MAX_UNDO),
            editor_camera,
            window_input: window_input_handle,
            entity_buffers,
            debug_drawer: debug_drawer_handle,
        }
    }

    /// Get the active editor world (immutable).
    fn active_world(&self) -> &EditorWorld {
        &self.worlds[self.active_world]
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

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        if self.worlds.is_empty() {
            return true;
        }

        // Poll native menu events (macOS only)
        #[cfg(target_os = "macos")]
        if let Some(menu) = &self.native_menu
            && let Some(action) = menu.poll_event()
        {
            use crate::menu::MenuAction;
            let ew = &mut self.worlds[self.active_world];
            match action {
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

        // Update GPU uniform buffers from ECS data
        if let Some(scene_view) = &self.scene_view {
            let ew = self.active_world();
            scene_view.update_uniforms(&ew.world, &ew.entity_buffers);
        }

        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
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

            // Menu bar (egui fallback for non-macOS platforms)
            #[cfg(not(target_os = "macos"))]
            menu::draw_menu_bar(&egui_ctx);

            // Toolbar (below menu bar)
            self.play_state = toolbar::draw_toolbar(&egui_ctx, self.play_state);

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
                        let mut tab_viewer = EditorTabViewer {
                            world: &mut ew.world,
                            inspector_state: &mut self.inspector_state,
                            vfs: &self.vfs,
                            asset_browser: &mut self.asset_browser,
                            console: &mut self.console,
                            history: &ew.history,
                            scene_view_rect: None,
                        };
                        egui_dock::DockArea::new(&mut self.dock_state)
                            .show_leaf_collapse_buttons(false)
                            .show_inside(ui, &mut tab_viewer);
                        scene_view_rect = tab_viewer.scene_view_rect;
                    }
                });

            // Store egui input state for next frame
            pixels_per_point = egui_ctx.pixels_per_point();
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

        if let Some(scene_view) = &self.scene_view
            && scene_view.has_viewport()
            && !self.worlds.is_empty()
        {
            let ew = self.active_world();
            if let Some(mut scene_pass) =
                scene_view.build_scene_pass(&ew.world, ctx.swapchain_texture())
            {
                // Append debug draw lines into the scene pass if available
                if let Some(renderer) = &mut self.debug_drawer_renderer {
                    let ew = &self.worlds[self.active_world];
                    if let Ok(drawer) = ew.debug_drawer.read() {
                        let vertices = drawer.take_render_data();
                        if !vertices.is_empty() {
                            if let Ok(cameras) = ew.world.read_all::<Camera>()
                                && let Some((_, camera)) = cameras.iter().next()
                            {
                                renderer.update_view_proj(mat4_to_cols_array_2d(
                                    &camera.view_projection(),
                                ));
                            }
                            renderer.append_to_pass(&mut scene_pass, &vertices);
                        }
                    }
                }

                let mut scene_graph = ctx.acquire_graph();
                scene_graph.add_graphics_pass(scene_pass);
                let scene_handle = ctx.submit("scene", scene_graph, &[]);
                deps.push(scene_handle);
            }
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
