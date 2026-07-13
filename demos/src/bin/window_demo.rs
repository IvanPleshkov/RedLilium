//! # Window Demo
//!
//! Basic window creation demo: opens a window and renders an animated clear
//! color through the frame pipeline (surface → device → `FramePipeline` →
//! swapchain present). Supports both native and web targets.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use redlilium_graphics::{
    ColorAttachment, FramePipeline, GraphicsDevice, GraphicsError, GraphicsInstance, GraphicsPass,
    InstanceParameters, LoadOp, PresentMode, RenderTargetConfig, StoreOp, Surface,
    SurfaceConfiguration,
};

/// Main application state
struct App {
    window: Option<Arc<Window>>,
    instance: Option<Arc<GraphicsInstance>>,
    device: Option<Arc<GraphicsDevice>>,
    surface: Option<Arc<Surface>>,
    pipeline: Option<FramePipeline>,
    /// Current window size.
    window_size: (u32, u32),
    /// Frames rendered so far (drives the clear-color animation).
    frame_count: u64,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            instance: None,
            device: None,
            surface: None,
            pipeline: None,
            window_size: (1280, 720),
            frame_count: 0,
        }
    }

    /// Create the instance, surface, device and frame pipeline for the window.
    fn init_graphics(&mut self) -> Result<(), GraphicsError> {
        let window = self.window.as_ref().expect("window created before init");

        let instance = GraphicsInstance::with_parameters(InstanceParameters::new())?;
        let surface = instance.create_surface(window.clone())?;
        let device = instance.create_device_for_surface(&surface)?;

        let config = SurfaceConfiguration::new(self.window_size.0, self.window_size.1)
            .with_format(surface.preferred_format())
            .with_present_mode(PresentMode::Fifo);
        surface.configure(&device, &config)?;

        let pipeline = device.create_pipeline(2);

        log::info!("Graphics initialized on {}", device.name());

        self.instance = Some(instance);
        self.device = Some(device);
        self.surface = Some(surface);
        self.pipeline = Some(pipeline);
        Ok(())
    }

    /// Renders a single frame: an animated clear straight to the swapchain.
    fn render_frame(&mut self) {
        let (Some(surface), Some(pipeline)) = (&self.surface, &mut self.pipeline) else {
            return;
        };

        let swapchain_texture = match surface.acquire_texture() {
            Ok(t) => t,
            Err(GraphicsError::SurfaceOutdated | GraphicsError::SurfaceLost) => {
                let config = SurfaceConfiguration::new(self.window_size.0, self.window_size.1)
                    .with_format(surface.preferred_format())
                    .with_present_mode(PresentMode::Fifo);
                if let Some(device) = &self.device {
                    let _ = surface.configure(device, &config);
                }
                return; // skip this frame; next acquire uses the new swapchain
            }
            Err(e) => {
                log::error!("Failed to acquire swapchain texture: {e}");
                return;
            }
        };

        let hue = (self.frame_count % 360) as f32;
        let (r, g, b) = hue_to_rgb(hue);

        let mut schedule = pipeline.begin_frame().expect("begin_frame failed");
        let mut graph = schedule.acquire_graph();
        let mut pass = GraphicsPass::new("window_demo_clear".into());
        pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_surface(&swapchain_texture)
                    .with_load_op(LoadOp::clear_color(r, g, b, 1.0))
                    .with_store_op(StoreOp::Store),
            ),
        );
        graph.add_graphics_pass(pass);
        schedule.render(graph);
        pipeline.end_frame(schedule);

        if let Err(e) = swapchain_texture.present() {
            // SurfaceOutdated here is non-fatal: the next acquire reconfigures.
            log::warn!("Present reported: {e}");
        }

        self.frame_count += 1;
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
                    self.window = Some(Arc::new(window));
                    if let Err(e) = self.init_graphics() {
                        log::error!("Graphics initialization failed: {e}");
                        event_loop.exit();
                    }
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
                if let Some(pipeline) = &self.pipeline {
                    let _ = pipeline.wait_idle();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                log::info!("Window resized to {}x{}", size.width, size.height);
                self.window_size = (size.width.max(1), size.height.max(1));

                // Reconfigure surface on resize
                if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
                    let config = SurfaceConfiguration::new(self.window_size.0, self.window_size.1)
                        .with_format(surface.preferred_format())
                        .with_present_mode(PresentMode::Fifo);
                    let _ = surface.configure(device, &config);
                }
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

/// Convert hue (0-360) to RGB (0-1).
fn hue_to_rgb(hue: f32) -> (f32, f32, f32) {
    let h = hue / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
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
