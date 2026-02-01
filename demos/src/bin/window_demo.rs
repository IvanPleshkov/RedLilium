//! # Window Demo
//!
//! Basic window creation demo with ECS and rendering integration.
//! Supports both native and web targets.
//!
//! This demo showcases the camera-based rendering architecture:
//! - Camera entities define viewpoints and render targets
//! - CameraSystem orchestrates per-camera render graphs
//! - Multiple cameras can render to textures or the main window

use redlilium_ecs::bevy_ecs::prelude::*;
#[allow(unused_imports)]
use redlilium_ecs::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// Main application state
struct App {
    window: Option<Window>,
    /// ECS world containing all entities and components.
    world: World,
    /// Current window size.
    window_size: (u32, u32),
}

impl App {
    fn new() -> Self {
        let mut world = World::new();
        setup_scene(&mut world);

        Self {
            window: None,
            world,
            window_size: (1280, 720),
        }
    }

    /// Runs the transform propagation systems.
    fn update_transforms(&mut self) {
        run_transform_systems(&mut self.world);
    }

    /// Renders a single frame.
    fn render_frame(&mut self) {
        Self::update_transforms(self);

        // TODO
    }
}

/// Sets up a simple test scene with several entities.
///
/// This creates a flat scene (no hierarchy) to demonstrate the ECS-Rendering integration.
/// Hierarchical scenes with parent-child relationships will be added when we implement
/// proper hierarchy synchronization systems.
fn setup_scene(world: &mut World) {
    log::info!("Setting up scene...");

    // Main camera looking at the scene
    world.spawn((
        Camera::new()
            .with_priority(0)
            .with_clear_color(Vec4::new(0.1, 0.1, 0.15, 1.0)),
        Transform::from_xyz(0.0, 3.0, 8.0).looking_at(Vec3::new(0.0, 0.0, -5.0), Vec3::Y),
        GlobalTransform::IDENTITY,
    ));

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

    log::info!("Scene setup complete: 1 camera + 5 renderable entities");
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("RedLilium Engine - Camera Demo")
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
                self.window_size = (size.width, size.height);
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

    log::info!("Starting RedLilium Engine Camera Demo");
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

    log::info!("Starting RedLilium Engine Camera Demo (Web)");
    log::info!("Core version: {}", redlilium_core::VERSION);
    log::info!("Graphics version: {}", redlilium_graphics::VERSION);

    redlilium_core::init();
    redlilium_graphics::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("Event loop error");
}
