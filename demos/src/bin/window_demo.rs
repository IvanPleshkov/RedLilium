//! # Window Demo
//!
//! Basic window creation demo.
//! Supports both native and web targets.

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// Main application state
struct App {
    window: Option<Window>,
    /// Current window size.
    window_size: (u32, u32),
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            window_size: (1280, 720),
        }
    }

    /// Renders a single frame.
    fn render_frame(&mut self) {
        // TODO
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("RedLilium Engine - Window Demo")
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

    log::info!("Starting RedLilium Engine Window Demo");
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

    log::info!("Starting RedLilium Engine Window Demo (Web)");
    log::info!("Core version: {}", redlilium_core::VERSION);
    log::info!("Graphics version: {}", redlilium_graphics::VERSION);

    redlilium_core::init();
    redlilium_graphics::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();

    event_loop.run_app(&mut app).expect("Event loop error");
}
