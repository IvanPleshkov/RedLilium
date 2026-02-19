use std::f32::consts::FRAC_PI_4;
use std::sync::{Arc, RwLock};

use egui_dock::DockState;
use redlilium_app::{AppContext, AppHandler, DrawContext};
use redlilium_core::math::Vec3;
use redlilium_core::mesh::generators;
use redlilium_ecs::ui::InspectorState;
use redlilium_ecs::{
    Camera, EcsRunner, Entity, FreeFlyCamera, GlobalTransform, PostUpdate, RenderMaterial,
    RenderMesh, Schedules, Transform, Update, UpdateCameraMatrices, UpdateFreeFlyCamera,
    UpdateGlobalTransforms, Visibility, WindowInput, World, register_std_components,
};
use redlilium_graphics::egui::{EguiApp, EguiController};
use redlilium_graphics::{Buffer, FrameSchedule, RenderTarget};
use winit::event::{KeyEvent, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::dock::{self, EditorTabViewer, Tab};
#[cfg(not(target_os = "macos"))]
use crate::menu;
#[cfg(target_os = "macos")]
use crate::menu::NativeMenu;
use crate::scene_view::SceneViewState;
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
    /// The editor camera entity (flagged as EDITOR).
    pub editor_camera: Entity,
    /// Handle to the WindowInput resource for updating from app events.
    pub window_input: Arc<RwLock<WindowInput>>,
    /// Per-entity uniform buffers for scene rendering.
    pub entity_buffers: Vec<(Entity, Arc<Buffer>)>,
}

pub struct Editor {
    // Multi-world support
    worlds: Vec<EditorWorld>,
    active_world: usize,
    runner: EcsRunner,

    // UI
    egui_controller: Option<EguiController>,
    dock_state: DockState<Tab>,
    inspector_state: InspectorState,
    play_state: PlayState,
    #[cfg(target_os = "macos")]
    native_menu: Option<NativeMenu>,

    // Scene rendering
    scene_view: Option<SceneViewState>,

    // Input state for egui feedback
    egui_wants_pointer: bool,
    egui_wants_keyboard: bool,
}

impl Editor {
    pub fn new() -> Self {
        Self {
            worlds: Vec::new(),
            active_world: 0,
            runner: EcsRunner::single_thread(),
            egui_controller: None,
            dock_state: dock::create_default_layout(),
            inspector_state: InspectorState::new(),
            play_state: PlayState::Editing,
            #[cfg(target_os = "macos")]
            native_menu: None,
            scene_view: None,
            egui_wants_pointer: false,
            egui_wants_keyboard: false,
        }
    }

    /// Create a new editor world with a simple demo scene.
    fn create_editor_world(&self, scene_view: &SceneViewState, aspect: f32) -> EditorWorld {
        let mut world = World::new();
        register_std_components(&mut world);
        redlilium_ecs::register_rendering_components(&mut world);

        // Insert WindowInput resource
        let window_input_handle = world.insert_resource(WindowInput::default());

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

        // --- Setup schedules ---
        let mut schedules = Schedules::new();

        // Update: camera input
        schedules.get_mut::<Update>().add(UpdateFreeFlyCamera);

        // PostUpdate: transform propagation -> camera matrices
        schedules
            .get_mut::<PostUpdate>()
            .add(UpdateGlobalTransforms);
        schedules.get_mut::<PostUpdate>().add(UpdateCameraMatrices);
        schedules
            .get_mut::<PostUpdate>()
            .add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>()
            .expect("No cycle");

        EditorWorld {
            world,
            schedules,
            editor_camera,
            window_input: window_input_handle,
            entity_buffers,
        }
    }

    /// Get the active editor world (immutable).
    fn active_world(&self) -> &EditorWorld {
        &self.worlds[self.active_world]
    }

    /// Update WindowInput's ui_wants_input flag from egui state.
    fn sync_input_flags(&self) {
        let ew = self.active_world();
        if let Ok(mut input) = ew.window_input.write() {
            input.ui_wants_input = self.egui_wants_pointer || self.egui_wants_keyboard;
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
            log::info!("Menu action: {:?}", action);
        }

        // Sync ui_wants_input flag from previous frame's egui state
        self.sync_input_flags();

        // Begin input frame (clears per-frame deltas)
        {
            let ew = self.active_world();
            if let Ok(mut input) = ew.window_input.write() {
                input.begin_frame();
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

            // Dock area fills remaining space (transparent so scene renders through)
            let panel_frame =
                egui::Frame::central_panel(&egui_ctx.style()).fill(egui::Color32::TRANSPARENT);
            egui::CentralPanel::default()
                .frame(panel_frame)
                .show(&egui_ctx, |ui| {
                    let active_world = if self.worlds.is_empty() {
                        None
                    } else {
                        Some(&mut self.worlds[self.active_world].world)
                    };
                    if let Some(world) = active_world {
                        let mut tab_viewer = EditorTabViewer {
                            world,
                            inspector_state: &mut self.inspector_state,
                            scene_view_rect: None,
                        };
                        egui_dock::DockArea::new(&mut self.dock_state)
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
        }

        // Submit scene first (clears swapchain), then egui on top
        let mut deps = Vec::new();

        if let Some(scene_view) = &self.scene_view
            && scene_view.has_viewport()
            && !self.worlds.is_empty()
        {
            let ew = self.active_world();
            if let Some(scene_pass) =
                scene_view.build_scene_pass(&ew.world, ctx.swapchain_texture())
            {
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
        if !self.worlds.is_empty() {
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
        if !self.worlds.is_empty() {
            let ew = self.active_world();
            if let Ok(mut input) = ew.window_input.write() {
                input.on_scroll(dx, dy);
            }
        }
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_key(event);
        }
        if !self.worlds.is_empty() {
            let pressed = event.state.is_pressed();
            let ew = self.active_world();
            if let Ok(mut input) = ew.window_input.write() {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => input.key_w = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => input.key_a = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => input.key_s = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => input.key_d = pressed,
                    PhysicalKey::Code(KeyCode::KeyQ) => input.key_q = pressed,
                    PhysicalKey::Code(KeyCode::KeyE) => input.key_e = pressed,
                    PhysicalKey::Code(KeyCode::ControlLeft | KeyCode::ControlRight) => {
                        input.key_ctrl = pressed;
                    }
                    PhysicalKey::Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) => {
                        input.key_shift = pressed;
                    }
                    _ => {}
                }
            }
        }
    }
}
