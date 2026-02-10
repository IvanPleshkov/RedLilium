//! Physics demo application — AppHandler implementation.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use redlilium_app::{AppContext, AppHandler, DrawContext};
use redlilium_core::profiling::{profile_function, profile_scope};
use redlilium_graphics::{
    ColorAttachment, DepthStencilAttachment, FrameSchedule, GraphicsPass, RenderTarget,
    RenderTargetConfig, egui::EguiController,
};
use winit::event::KeyEvent;
use winit::keyboard::{KeyCode, PhysicalKey};

use redlilium_ecs::{EcsRunner, SystemsContainer, World};

use crate::renderer::PhysicsRenderer;
use crate::scenes_2d::{self, PhysicsScene2D};
use crate::scenes_3d::{self, PhysicsScene3D};
use crate::ui::{Dimension, PhysicsUi};

// ---------------------------------------------------------------------------
// Orbit camera
// ---------------------------------------------------------------------------

struct OrbitCamera {
    target: glam::Vec3,
    distance: f32,
    azimuth: f32,
    elevation: f32,
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            target: glam::Vec3::new(0.0, 3.0, 0.0),
            distance: 20.0,
            azimuth: 0.5,
            elevation: 0.4,
        }
    }

    fn rotate(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        self.azimuth += delta_azimuth;
        self.elevation = (self.elevation + delta_elevation).clamp(-1.5, 1.5);
    }

    fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta).clamp(2.0, 60.0);
    }

    fn position(&self) -> glam::Vec3 {
        let x = self.distance * self.elevation.cos() * self.azimuth.sin();
        let y = self.distance * self.elevation.sin();
        let z = self.distance * self.elevation.cos() * self.azimuth.cos();
        self.target + glam::Vec3::new(x, y, z)
    }
}

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

    // Rendering
    renderer: Option<PhysicsRenderer>,
    camera: OrbitCamera,

    // UI
    egui_controller: Option<EguiController>,
    ui: Arc<RwLock<PhysicsUi>>,

    // Input state
    mouse_pressed: bool,
    last_mouse_x: f64,
    last_mouse_y: f64,
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
            renderer: None,
            camera: OrbitCamera::new(),
            egui_controller: None,
            ui,
            mouse_pressed: false,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
        }
    }

    /// Create a fresh ECS world and populate it with the active scene.
    fn setup_active_scene(&mut self) {
        let (dim, index) = if let Ok(ui) = self.ui.read() {
            (ui.active_dim, ui.active_index)
        } else {
            (Dimension::ThreeD, 0)
        };

        let mut world = World::new();
        ecs_std::register_std_components(&mut world);

        // Register physics handle components
        world.register_component::<ecs_std::physics::physics3d::RigidBody3DHandle>();
        world.register_component::<ecs_std::physics::physics2d::RigidBody2DHandle>();

        // Build systems container for the appropriate dimension
        let mut systems = SystemsContainer::new();

        match dim {
            Dimension::ThreeD => {
                systems.add(ecs_std::physics::physics3d::StepPhysics3D);
                systems.add(ecs_std::UpdateGlobalTransforms);
                let _ = systems.add_edge::<
                    ecs_std::physics::physics3d::StepPhysics3D,
                    ecs_std::UpdateGlobalTransforms,
                >();

                if let Some(scene) = self.scenes_3d.get(index) {
                    scene.setup(&mut world);
                }
            }
            Dimension::TwoD => {
                systems.add(ecs_std::physics::physics2d::StepPhysics2D);
                systems.add(ecs_std::UpdateGlobalTransforms);
                let _ = systems.add_edge::<
                    ecs_std::physics::physics2d::StepPhysics2D,
                    ecs_std::UpdateGlobalTransforms,
                >();

                if let Some(scene) = self.scenes_2d.get(index) {
                    scene.setup(&mut world);
                }
            }
        }

        // Update UI stats
        self.update_ui_stats(&world, dim);

        self.world = Some(world);
        self.systems = Some(systems);
    }

    fn update_ui_stats(&self, world: &World, dim: Dimension) {
        if let Ok(mut ui) = self.ui.write() {
            match dim {
                Dimension::ThreeD => {
                    if world.has_resource::<ecs_std::physics::physics3d::PhysicsWorld3D>() {
                        let physics =
                            world.resource::<ecs_std::physics::physics3d::PhysicsWorld3D>();
                        ui.body_count = physics.bodies.len();
                        ui.collider_count = physics.colliders.len();
                    }
                }
                Dimension::TwoD => {
                    if world.has_resource::<ecs_std::physics::physics2d::PhysicsWorld2D>() {
                        let physics =
                            world.resource::<ecs_std::physics::physics2d::PhysicsWorld2D>();
                        ui.body_count = physics.bodies.len();
                        ui.collider_count = physics.colliders.len();
                    }
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
        log::info!("Controls: LMB drag=orbit, scroll=zoom, H=toggle UI, Space=pause");

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
        self.setup_active_scene();
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        if let Some(renderer) = &mut self.renderer {
            renderer.resize(ctx.device(), ctx.width(), ctx.height());
        }
        if let Some(egui) = &mut self.egui_controller {
            egui.on_resize(ctx.width(), ctx.height());
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
            self.setup_active_scene();
        }

        // Step physics if not paused
        let paused = self.ui.read().map(|ui| ui.paused).unwrap_or(false);

        if !paused
            && let (Some(world), Some(systems), Some(runner)) =
                (&mut self.world, &self.systems, &self.runner)
        {
            runner.run(world, systems, Duration::from_millis(16));
        }

        // Update renderer with current physics state
        let dim = self
            .ui
            .read()
            .map(|ui| ui.active_dim)
            .unwrap_or(Dimension::ThreeD);
        let camera_pos = self.camera.position();
        let aspect = ctx.width() as f32 / ctx.height().max(1) as f32;
        let view = glam::Mat4::look_at_rh(camera_pos, self.camera.target, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0);
        let view_proj = proj * view;

        if let (Some(renderer), Some(world)) = (&mut self.renderer, &self.world) {
            let device = ctx.device();
            match dim {
                Dimension::ThreeD => {
                    if world.has_resource::<ecs_std::physics::physics3d::PhysicsWorld3D>() {
                        let physics =
                            world.resource::<ecs_std::physics::physics3d::PhysicsWorld3D>();
                        renderer.update_3d(device, &physics, view_proj, camera_pos);
                    }
                }
                Dimension::TwoD => {
                    if world.has_resource::<ecs_std::physics::physics2d::PhysicsWorld2D>() {
                        let physics =
                            world.resource::<ecs_std::physics::physics2d::PhysicsWorld2D>();
                        renderer.update_2d(device, &physics, view_proj, camera_pos);
                    }
                }
            }
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

        if self.mouse_pressed && !egui_wants {
            let dx = (x - self.last_mouse_x) as f32 * 0.005;
            let dy = (y - self.last_mouse_y) as f32 * 0.005;
            self.camera.rotate(-dx, -dy);
        }
        self.last_mouse_x = x;
        self.last_mouse_y = y;
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

        if button == winit::event::MouseButton::Left && !egui_wants {
            self.mouse_pressed = pressed;
        }
    }

    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, _dx: f32, dy: f32) {
        let egui_wants = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, dy))
        } else {
            false
        };

        if !egui_wants {
            self.camera.zoom(dy * 0.5);
        }
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_key(event);
        }

        if !event.state.is_pressed() {
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
