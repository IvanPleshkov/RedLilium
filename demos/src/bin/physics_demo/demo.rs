//! Physics demo application — AppHandler implementation.

use std::sync::{Arc, RwLock};

use redlilium_app::{AppContext, AppHandler, DrawContext};
use redlilium_core::math::{Vec3, perspective_rh};
use redlilium_core::profiling::{profile_function, profile_scope};
use redlilium_graphics::{
    ColorAttachment, DepthStencilAttachment, FrameSchedule, GraphicsPass, RenderTarget,
    RenderTargetConfig, egui::EguiController,
};
use winit::event::KeyEvent;
use winit::keyboard::{KeyCode, PhysicalKey};

use redlilium_ecs::{
    Camera, EcsRunner, Entity, FreeFlyCamera, GlobalTransform, SystemsContainer,
    UpdateFreeFlyCamera, WindowInput, World,
};

use crate::renderer::PhysicsRenderer;
use crate::scenes_2d::{self, PhysicsScene2D};
use crate::scenes_3d::{self, PhysicsScene3D};
use crate::ui::{Dimension, PhysicsUi};

// ---------------------------------------------------------------------------
// PhysicsDemoApp
// ---------------------------------------------------------------------------

pub struct PhysicsDemoApp {
    // Scenes
    scenes_3d: Vec<Box<dyn PhysicsScene3D>>,
    scenes_2d: Vec<Box<dyn PhysicsScene2D>>,

    // ECS
    world: Option<World>,
    systems: Option<SystemsContainer>,
    runner: Option<EcsRunner>,
    camera_entity: Option<Entity>,

    // Rendering
    renderer: Option<PhysicsRenderer>,

    // Input
    window_input: Option<Arc<RwLock<WindowInput>>>,

    // UI
    egui_controller: Option<EguiController>,
    ui: Arc<RwLock<PhysicsUi>>,

    // Inspector
    inspector_state: redlilium_ecs::ui::InspectorState,
}

impl PhysicsDemoApp {
    pub fn new() -> Self {
        let scenes_3d = scenes_3d::all_scenes_3d();
        let scenes_2d = scenes_2d::all_scenes_2d();

        let ui = Arc::new(RwLock::new(PhysicsUi::new()));

        // Populate scene names in UI
        if let Ok(mut ui) = ui.write() {
            ui.scene_names_3d = scenes_3d.iter().map(|s| s.name().to_string()).collect();
            ui.scene_names_2d = scenes_2d.iter().map(|s| s.name().to_string()).collect();
        }

        Self {
            scenes_3d,
            scenes_2d,
            world: None,
            systems: None,
            runner: None,
            camera_entity: None,
            renderer: None,
            window_input: None,
            egui_controller: None,
            ui,
            inspector_state: redlilium_ecs::ui::InspectorState::new(),
        }
    }

    /// Create a fresh ECS world and populate it with the active scene.
    fn setup_active_scene(&mut self, ctx: &AppContext) {
        let (dim, index) = if let Ok(ui) = self.ui.read() {
            (ui.active_dim, ui.active_index)
        } else {
            (Dimension::ThreeD, 0)
        };

        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);

        // Insert WindowInput resource
        let input = WindowInput {
            window_width: ctx.width() as f32,
            window_height: ctx.height() as f32,
            ..WindowInput::default()
        };
        let input_handle = world.insert_resource(input);
        self.window_input = Some(input_handle);

        // Build systems container for the appropriate dimension
        let mut systems = SystemsContainer::new();

        // Free-fly camera system (always runs, even when paused)
        systems.add(UpdateFreeFlyCamera);

        match dim {
            Dimension::ThreeD => {
                use redlilium_ecs::physics::physics3d::*;
                systems.add_exclusive(SyncPhysicsBodies3D);
                systems.add_exclusive(SyncPhysicsJoints3D);
                systems.add(StepPhysics3D);
                systems.add(redlilium_ecs::UpdateGlobalTransforms);
                systems.add(redlilium_ecs::UpdateCameraMatrices);
                let _ = systems.add_edge::<SyncPhysicsBodies3D, SyncPhysicsJoints3D>();
                let _ = systems.add_edge::<SyncPhysicsJoints3D, StepPhysics3D>();
                let _ = systems.add_edge::<StepPhysics3D, redlilium_ecs::UpdateGlobalTransforms>();
                let _ = systems
                    .add_edge::<UpdateFreeFlyCamera, redlilium_ecs::UpdateGlobalTransforms>();
                let _ = systems.add_edge::<redlilium_ecs::UpdateGlobalTransforms, redlilium_ecs::UpdateCameraMatrices>();

                if let Some(scene) = self.scenes_3d.get(index) {
                    scene.setup(&mut world);
                }
            }
            Dimension::TwoD => {
                use redlilium_ecs::physics::physics2d::*;
                systems.add_exclusive(SyncPhysicsBodies2D);
                systems.add_exclusive(SyncPhysicsJoints2D);
                systems.add(StepPhysics2D);
                systems.add(redlilium_ecs::UpdateGlobalTransforms);
                systems.add(redlilium_ecs::UpdateCameraMatrices);
                let _ = systems.add_edge::<SyncPhysicsBodies2D, SyncPhysicsJoints2D>();
                let _ = systems.add_edge::<SyncPhysicsJoints2D, StepPhysics2D>();
                let _ = systems.add_edge::<StepPhysics2D, redlilium_ecs::UpdateGlobalTransforms>();
                let _ = systems
                    .add_edge::<UpdateFreeFlyCamera, redlilium_ecs::UpdateGlobalTransforms>();
                let _ = systems.add_edge::<redlilium_ecs::UpdateGlobalTransforms, redlilium_ecs::UpdateCameraMatrices>();

                if let Some(scene) = self.scenes_2d.get(index) {
                    scene.setup(&mut world);
                }
            }
        }

        // Spawn camera entity with FreeFlyCamera component
        let cam_entity = world.spawn();
        let fly_cam = FreeFlyCamera::new(Vec3::new(0.0, 3.0, 0.0), 20.0)
            .with_yaw(0.5)
            .with_pitch(0.4);
        let _ = world.insert(cam_entity, fly_cam);
        let _ = world.insert(
            cam_entity,
            Camera::perspective(std::f32::consts::FRAC_PI_4, ctx.aspect_ratio(), 0.1, 200.0),
        );
        let _ = world.insert(cam_entity, fly_cam.to_transform());
        let _ = world.insert(cam_entity, GlobalTransform::IDENTITY);
        self.camera_entity = Some(cam_entity);

        // Update UI stats
        self.update_ui_stats(&world, dim);

        // Reset inspector selection since entities changed
        self.inspector_state.selected = None;

        self.world = Some(world);
        self.systems = Some(systems);
    }

    fn update_ui_stats(&self, world: &World, dim: Dimension) {
        if let Ok(mut ui) = self.ui.write() {
            match dim {
                Dimension::ThreeD => {
                    if world.has_resource::<redlilium_ecs::physics::physics3d::PhysicsWorld3D>() {
                        let physics =
                            world.resource::<redlilium_ecs::physics::physics3d::PhysicsWorld3D>();
                        ui.body_count = physics.bodies.len();
                        ui.collider_count = physics.colliders.len();
                    }
                }
                Dimension::TwoD => {
                    if world.has_resource::<redlilium_ecs::physics::physics2d::PhysicsWorld2D>() {
                        let physics =
                            world.resource::<redlilium_ecs::physics::physics2d::PhysicsWorld2D>();
                        ui.body_count = physics.bodies.len();
                        ui.collider_count = physics.colliders.len();
                    }
                }
            }
        }
    }

    /// Set physics timestep to zero (freeze) or restore default.
    fn set_physics_paused(&self, paused: bool) {
        let Some(world) = &self.world else { return };
        let dim = self
            .ui
            .read()
            .map(|ui| ui.active_dim)
            .unwrap_or(Dimension::ThreeD);

        match dim {
            Dimension::ThreeD => {
                if world.has_resource::<redlilium_ecs::physics::physics3d::PhysicsWorld3D>() {
                    let mut physics =
                        world.resource_mut::<redlilium_ecs::physics::physics3d::PhysicsWorld3D>();
                    physics.integration_parameters.dt = if paused { 0.0 } else { 1.0 / 60.0 };
                }
            }
            Dimension::TwoD => {
                if world.has_resource::<redlilium_ecs::physics::physics2d::PhysicsWorld2D>() {
                    let mut physics =
                        world.resource_mut::<redlilium_ecs::physics::physics2d::PhysicsWorld2D>();
                    physics.integration_parameters.dt = if paused { 0.0 } else { 1.0 / 60.0 };
                }
            }
        }
    }
}

impl AppHandler for PhysicsDemoApp {
    fn on_init(&mut self, ctx: &mut AppContext) {
        profile_function!();

        log::info!("Initializing Physics Demo");
        log::info!(
            "  {} 3D scenes, {} 2D scenes",
            self.scenes_3d.len(),
            self.scenes_2d.len()
        );
        log::info!(
            "Controls: drag=orbit, Ctrl+drag=free look, WASD=move, QE=up/down, scroll=zoom, H=toggle UI, Space=pause"
        );

        // Create ECS runner
        self.runner = Some(EcsRunner::single_thread());

        // Create renderer
        self.renderer = Some(PhysicsRenderer::new(
            ctx.device(),
            ctx.width(),
            ctx.height(),
            ctx.surface_format(),
        ));

        // Create egui
        self.egui_controller = Some(EguiController::new(
            ctx.device().clone(),
            self.ui.clone(),
            ctx.width(),
            ctx.height(),
            ctx.scale_factor(),
            ctx.surface_format(),
        ));

        // Setup initial scene
        self.setup_active_scene(ctx);
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        if let Some(renderer) = &mut self.renderer {
            renderer.resize(ctx.device(), ctx.width(), ctx.height());
        }
        if let Some(egui) = &mut self.egui_controller {
            egui.on_resize(ctx.width(), ctx.height());
        }

        // Update WindowInput dimensions
        if let Some(handle) = &self.window_input
            && let Ok(mut input) = handle.write()
        {
            input.window_width = ctx.width() as f32;
            input.window_height = ctx.height() as f32;
        }

        // Update camera projection for new aspect ratio
        if let (Some(world), Some(cam_entity)) = (&self.world, self.camera_entity) {
            let mut cameras = world.write::<Camera>().unwrap();
            if let Some(cam) = cameras.get_mut(cam_entity.index()) {
                cam.projection_matrix =
                    perspective_rh(std::f32::consts::FRAC_PI_4, ctx.aspect_ratio(), 0.1, 200.0);
            }
        }
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        profile_scope!("on_update");

        // Check for scene change or reset
        let (scene_changed, reset) = if let Ok(mut ui) = self.ui.write() {
            (ui.take_scene_changed(), ui.take_reset_requested())
        } else {
            (false, false)
        };

        if scene_changed || reset {
            self.setup_active_scene(ctx);
        }

        // Freeze physics when paused (dt=0), but still run all systems
        // so orbit camera + transforms + camera matrices update
        let paused = self.ui.read().map(|ui| ui.paused).unwrap_or(false);
        self.set_physics_paused(paused);

        if let (Some(world), Some(systems), Some(runner)) =
            (&mut self.world, &self.systems, &self.runner)
        {
            runner.run(world, systems);
        }

        // Read camera data from ECS
        let dim = self
            .ui
            .read()
            .map(|ui| ui.active_dim)
            .unwrap_or(Dimension::ThreeD);

        let camera_pos = if let (Some(world), Some(cam_entity)) = (&self.world, self.camera_entity)
        {
            let fly_cams = world.read::<FreeFlyCamera>().unwrap();
            if let Some(cam) = fly_cams.get(cam_entity.index()) {
                // Update camera debug info in UI
                if let Ok(mut ui) = self.ui.write() {
                    ui.camera_distance = cam.distance;
                    ui.camera_speed = cam.move_speed * cam.speed_multiplier;
                }
                cam.eye_position()
            } else {
                Vec3::zeros()
            }
        } else {
            Vec3::zeros()
        };

        let view_proj = if let (Some(world), Some(cam_entity)) = (&self.world, self.camera_entity) {
            let cameras = world.read::<Camera>().unwrap();
            if let Some(cam) = cameras.get(cam_entity.index()) {
                cam.view_projection()
            } else {
                redlilium_core::math::Mat4::identity()
            }
        } else {
            redlilium_core::math::Mat4::identity()
        };

        if let (Some(renderer), Some(world)) = (&mut self.renderer, &self.world) {
            let device = ctx.device();
            match dim {
                Dimension::ThreeD => {
                    if world.has_resource::<redlilium_ecs::physics::physics3d::PhysicsWorld3D>() {
                        let physics =
                            world.resource::<redlilium_ecs::physics::physics3d::PhysicsWorld3D>();
                        renderer.update_3d(device, &physics, view_proj, camera_pos);
                    }
                }
                Dimension::TwoD => {
                    if world.has_resource::<redlilium_ecs::physics::physics2d::PhysicsWorld2D>() {
                        let physics =
                            world.resource::<redlilium_ecs::physics::physics2d::PhysicsWorld2D>();
                        renderer.update_2d(device, &physics, view_proj, camera_pos);
                    }
                }
            }
        }

        // Clear per-frame deltas after systems have consumed them,
        // so next frame's events accumulate fresh deltas.
        if let Some(handle) = &self.window_input
            && let Ok(mut input) = handle.write()
        {
            input.begin_frame();
        }

        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        profile_scope!("on_draw");

        let mut graph = ctx.acquire_graph();

        // Shape rendering pass with depth buffer
        let mut shape_pass = GraphicsPass::new("shapes".into());
        if let Some(renderer) = &self.renderer {
            shape_pass.set_render_targets(
                RenderTargetConfig::new()
                    .with_color(
                        ColorAttachment::from_surface(ctx.swapchain_texture())
                            .with_clear_color(0.15, 0.15, 0.2, 1.0),
                    )
                    .with_depth_stencil(
                        DepthStencilAttachment::from_texture(renderer.depth_texture().clone())
                            .with_clear_depth(1.0),
                    ),
            );
            renderer.add_draws(&mut shape_pass);
        }
        let shape_handle = graph.add_graphics_pass(shape_pass);

        // Egui pass (draws on top without depth, preserves color)
        if let Some(egui) = &mut self.egui_controller {
            let width = ctx.width();
            let height = ctx.height();
            let elapsed = ctx.elapsed_time() as f64;
            let render_target = RenderTarget::from_surface(ctx.swapchain_texture());

            egui.begin_frame(elapsed);

            // Draw inspector UI between begin_frame and end_frame
            let show_inspector = self.ui.read().map(|ui| ui.show_inspector).unwrap_or(false);
            if show_inspector {
                let egui_ctx = egui.context().clone();
                if let Some(world) = &self.world {
                    redlilium_ecs::ui::show_world_inspector(
                        &egui_ctx,
                        world,
                        &mut self.inspector_state,
                    );
                }
                if let Some(world) = &mut self.world {
                    redlilium_ecs::ui::show_component_inspector(
                        &egui_ctx,
                        world,
                        &mut self.inspector_state,
                    );
                }
            }

            if let Some(egui_pass) = egui.end_frame(&render_target, width, height) {
                let egui_handle = graph.add_graphics_pass(egui_pass);
                graph.add_dependency(egui_handle, shape_handle);
            }
        }

        let _handle = ctx.submit("main", graph, &[]);

        ctx.finish(&[])
    }

    fn on_mouse_move(&mut self, _ctx: &mut AppContext, x: f64, y: f64) {
        let egui_wants = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_move(x, y)
        } else {
            false
        };

        if let Some(handle) = &self.window_input
            && let Ok(mut input) = handle.write()
        {
            input.on_mouse_move(x, y);
            input.ui_wants_input = egui_wants;
        }
    }

    fn on_mouse_button(
        &mut self,
        _ctx: &mut AppContext,
        button: winit::event::MouseButton,
        pressed: bool,
    ) {
        let egui_wants = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_button(button, pressed)
        } else {
            false
        };

        if let Some(handle) = &self.window_input
            && let Ok(mut input) = handle.write()
        {
            let idx = match button {
                winit::event::MouseButton::Left => 0,
                winit::event::MouseButton::Right => 1,
                winit::event::MouseButton::Middle => 2,
                _ => return,
            };
            input.on_mouse_button(idx, pressed);
            input.ui_wants_input = egui_wants;
        }
    }

    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, _dx: f32, dy: f32) {
        let egui_wants = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, dy))
        } else {
            false
        };

        if let Some(handle) = &self.window_input
            && let Ok(mut input) = handle.write()
        {
            input.on_scroll(0.0, dy);
            input.ui_wants_input = egui_wants;
        }
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_key(event);
        }

        let pressed = event.state.is_pressed();

        // Forward movement/modifier keys to WindowInput (track held state)
        if let Some(handle) = &self.window_input
            && let Ok(mut input) = handle.write()
        {
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

        // Actions on key press only
        if !pressed {
            return;
        }

        match event.physical_key {
            // H: Toggle UI
            PhysicalKey::Code(KeyCode::KeyH) => {
                if let Ok(mut ui) = self.ui.write() {
                    ui.toggle_visibility();
                }
            }
            // Space: Toggle pause
            PhysicalKey::Code(KeyCode::Space) => {
                if let Ok(mut ui) = self.ui.write() {
                    ui.paused = !ui.paused;
                }
            }
            _ => {}
        }
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down Physics Demo");
    }
}
