//! # Window Demo
//!
//! Basic window creation demo with ECS and rendering integration.
//! Supports both native and web targets.

use redlilium_ecs::bevy_ecs::prelude::*;
#[allow(unused_imports)]
use redlilium_ecs::prelude::*;
use redlilium_graphics::{
    DummyBackend, ExtractedMaterial, ExtractedMesh, ExtractedTransform, SceneRenderer,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// Main application state
struct App {
    window: Option<Window>,
    /// ECS world containing all entities and components.
    world: World,
    /// Scene renderer managing the render graph.
    scene_renderer: SceneRenderer,
    /// Graphics backend (dummy for now).
    backend: DummyBackend,
}

impl App {
    fn new() -> Self {
        let mut world = World::new();
        setup_scene(&mut world);

        Self {
            window: None,
            world,
            scene_renderer: SceneRenderer::new(),
            backend: DummyBackend::new(),
        }
    }

    /// Extracts render data from ECS world into the render world.
    fn extract_render_data(&mut self) {
        let render_world = self.scene_renderer.render_world_mut();

        // Query all entities with Transform, GlobalTransform, RenderMesh, and Material
        let mut query = self
            .world
            .query::<(Entity, &GlobalTransform, &RenderMesh, &Material)>();

        for (entity, global_transform, render_mesh, material) in query.iter(&self.world) {
            let transform = ExtractedTransform::from_matrix(global_transform.to_matrix());

            let mesh = ExtractedMesh {
                mesh_id: render_mesh.mesh.id(),
                cast_shadows: render_mesh.cast_shadows,
                receive_shadows: render_mesh.receive_shadows,
                render_layers: render_mesh.render_layers.bits(),
            };

            let extracted_material = ExtractedMaterial {
                base_color: material.base_color,
                metallic: material.metallic,
                roughness: material.roughness,
                emissive: material.emissive,
                alpha_mode: match material.alpha_mode {
                    AlphaMode::Opaque => 0,
                    AlphaMode::Mask { .. } => 1,
                    AlphaMode::Blend => 2,
                },
                alpha_cutoff: material
                    .alpha_mode
                    .cutoff()
                    .map(|c| (c * 255.0) as u8)
                    .unwrap_or(127),
                double_sided: material.double_sided,
                base_color_texture: material.base_color_texture.map(|h| h.id()).unwrap_or(0),
                normal_texture: material.normal_texture.map(|h| h.id()).unwrap_or(0),
                metallic_roughness_texture: material
                    .metallic_roughness_texture
                    .map(|h| h.id())
                    .unwrap_or(0),
            };

            render_world.add(entity.to_bits(), transform, mesh, extracted_material);
        }
    }

    /// Runs the transform propagation systems.
    fn update_transforms(&mut self) {
        // Run transform propagation systems
        run_transform_systems(&mut self.world);
    }

    /// Renders a single frame.
    fn render_frame(&mut self) {
        // Begin frame
        self.scene_renderer.begin_frame();

        // Update transforms first
        self.update_transforms();

        // Extract render data from ECS
        self.extract_render_data();

        // Prepare and render
        self.scene_renderer.prepare();
        self.scene_renderer.setup_basic_graph();

        if let Err(e) = self.scene_renderer.render(&self.backend) {
            log::error!("Render error: {}", e);
        }

        // End frame
        self.scene_renderer.end_frame();

        // Log frame info periodically
        let frame = self.scene_renderer.frame_count();
        if frame.is_multiple_of(60) {
            log::debug!(
                "Frame {}: {} render items",
                frame,
                self.scene_renderer.render_world().total_items()
            );
        }
    }
}

/// Sets up a simple test scene with several entities.
///
/// This creates a flat scene (no hierarchy) to demonstrate the ECS-Rendering integration.
/// Hierarchical scenes with parent-child relationships will be added when we implement
/// proper hierarchy synchronization systems.
fn setup_scene(world: &mut World) {
    log::info!("Setting up scene...");

    // Create a red cube in the center
    world.spawn((
        Transform::from_xyz(0.0, 0.0, -5.0),
        GlobalTransform::IDENTITY,
        RenderMesh::new(MeshHandle::new(1)), // Placeholder mesh ID
        Material::default().with_base_color(Vec4::new(0.8, 0.2, 0.2, 1.0)), // Red
    ));

    // Create a green cube to the left
    world.spawn((
        Transform::from_xyz(-3.0, 0.0, -5.0),
        GlobalTransform::IDENTITY,
        RenderMesh::new(MeshHandle::new(1)),
        Material::default()
            .with_base_color(Vec4::new(0.2, 0.8, 0.2, 1.0)) // Green
            .with_metallic(0.8),
    ));

    // Create a blue cube to the right
    world.spawn((
        Transform::from_xyz(3.0, 0.0, -5.0),
        GlobalTransform::IDENTITY,
        RenderMesh::new(MeshHandle::new(1)),
        Material::default()
            .with_base_color(Vec4::new(0.2, 0.2, 0.8, 1.0)) // Blue
            .with_roughness(0.1),
    ));

    // Create a floor plane
    world.spawn((
        Transform::from_xyz(0.0, -2.0, -5.0).with_scale(Vec3::new(10.0, 0.1, 10.0)),
        GlobalTransform::IDENTITY,
        RenderMesh::new(MeshHandle::new(2)), // Different mesh for floor
        Material::default()
            .with_base_color(Vec4::new(0.5, 0.5, 0.5, 1.0)) // Gray
            .with_roughness(0.9),
    ));

    // Create a transparent sphere
    world.spawn((
        Transform::from_xyz(0.0, 1.0, -3.0),
        GlobalTransform::IDENTITY,
        RenderMesh::new(MeshHandle::new(3)), // Sphere mesh
        Material::default()
            .with_base_color(Vec4::new(1.0, 1.0, 1.0, 0.5)) // Semi-transparent white
            .with_alpha_mode(AlphaMode::Blend),
    ));

    log::info!("Scene setup complete: 5 entities created");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("RedLilium Engine - Scene Demo")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

            match event_loop.create_window(window_attributes) {
                Ok(window) => {
                    log::info!("Window created successfully");
                    self.window = Some(window);
                }
                Err(e) => {
                    log::error!("Failed to create window: {}", e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, exiting...");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                log::info!("Window resized to {}x{}", size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                // Render the frame
                self.render_frame();

                // Request another redraw for continuous rendering
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting RedLilium Engine Scene Demo");
    log::info!("Core version: {}", redlilium_core::VERSION);
    log::info!("Graphics version: {}", redlilium_graphics::VERSION);

    redlilium_core::init();
    redlilium_graphics::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("Event loop error");
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Entry point for wasm - actual initialization happens in start()
}

/// WASM entry point called from JavaScript
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("Failed to initialize logger");

    log::info!("Starting RedLilium Engine Scene Demo (Web)");
    log::info!("Core version: {}", redlilium_core::VERSION);
    log::info!("Graphics version: {}", redlilium_graphics::VERSION);

    redlilium_core::init();
    redlilium_graphics::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("Event loop error");
}
