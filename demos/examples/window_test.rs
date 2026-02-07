//! Window integration test example.
//!
//! This example runs the same window/swapchain test as `graphics/tests/window_test.rs`
//! but can be run on the main thread, which is required for macOS.
//!
//! # Usage
//!
//! ```bash
//! # Run with Vulkan backend
//! cargo run -p redlilium-demos --example window_test -- vulkan
//!
//! # Run with wgpu backend
//! cargo run -p redlilium-demos --example window_test -- wgpu
//!
//! # Run with default (wgpu) backend
//! cargo run -p redlilium-demos --example window_test
//! ```
//!
//! # Why This Exists
//!
//! On macOS, the Cocoa framework requires that `EventLoop` be created on the main thread.
//! Rust's test framework runs tests on worker threads, making it impossible to run
//! window tests as regular `#[test]` functions on macOS.
//!
//! This example provides a way to run the window tests manually on all platforms.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;
use winit::window::{Window, WindowId};

use redlilium_graphics::{
    BackendType, ColorAttachment, FramePipeline, GraphicsDevice, GraphicsInstance, GraphicsPass,
    InstanceParameters, LoadOp, PresentMode, RenderGraph, RenderTargetConfig, StoreOp, Surface,
    SurfaceConfiguration, WgpuBackendType,
};

/// Number of frames to render before exiting.
const FRAMES_TO_RENDER: u32 = 5;

/// Test result that can be shared across the event loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestResult {
    Running,
    Passed,
    Skipped,
    Failed,
}

/// Application state for the window test.
struct WindowTestApp {
    result: TestResult,
    params: InstanceParameters,
    window: Option<Window>,
    instance: Option<Arc<GraphicsInstance>>,
    device: Option<Arc<GraphicsDevice>>,
    surface: Option<Arc<Surface>>,
    pipeline: Option<FramePipeline>,
    frame_count: u32,
    window_size: (u32, u32),
    resumed: bool,
}

impl WindowTestApp {
    fn new(params: InstanceParameters) -> Self {
        Self {
            result: TestResult::Running,
            params,
            window: None,
            instance: None,
            device: None,
            surface: None,
            pipeline: None,
            frame_count: 0,
            window_size: (320, 240),
            resumed: false,
        }
    }

    fn init_graphics(&mut self) -> bool {
        let window = match &self.window {
            Some(w) => w,
            None => {
                log::warn!("No window available for graphics init");
                return false;
            }
        };

        let instance = match GraphicsInstance::with_parameters(self.params.clone()) {
            Ok(i) => i,
            Err(e) => {
                log::warn!("Failed to create graphics instance: {}", e);
                return false;
            }
        };

        // Create surface first (needed to select compatible adapter)
        let surface = match instance.create_surface(window) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("Failed to create surface: {}", e);
                return false;
            }
        };

        // Create device that is compatible with the surface
        let device = match instance.create_device_for_surface(&surface) {
            Ok(d) => d,
            Err(e) => {
                log::warn!(
                    "Failed to create graphics device compatible with surface: {}",
                    e
                );
                return false;
            }
        };

        let config = SurfaceConfiguration::new(self.window_size.0, self.window_size.1)
            .with_format(surface.preferred_format())
            .with_present_mode(PresentMode::Fifo);

        if let Err(e) = surface.configure(&device, &config) {
            log::warn!("Failed to configure surface: {}", e);
            return false;
        }

        let pipeline = device.create_pipeline(2);

        log::info!(
            "Graphics initialized: {} ({}x{})",
            device.name(),
            self.window_size.0,
            self.window_size.1
        );

        self.instance = Some(instance);
        self.device = Some(device);
        self.surface = Some(surface);
        self.pipeline = Some(pipeline);

        true
    }

    fn render_frame(&mut self) -> bool {
        let _device = match &self.device {
            Some(d) => d,
            None => return false,
        };
        let surface = match &self.surface {
            Some(s) => s,
            None => return false,
        };
        let pipeline = match &mut self.pipeline {
            Some(p) => p,
            None => return false,
        };

        let swapchain_texture = match surface.acquire_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Failed to acquire swapchain texture: {}", e);
                return false;
            }
        };

        let hue = (self.frame_count as f32 / FRAMES_TO_RENDER as f32) * 360.0;
        let (r, g, b) = hue_to_rgb(hue);

        let mut graph = RenderGraph::default();
        let mut pass = GraphicsPass::new(format!("frame_{}", self.frame_count));
        pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_surface(&swapchain_texture)
                    .with_load_op(LoadOp::clear_color(r, g, b, 1.0))
                    .with_store_op(StoreOp::Store),
            ),
        );
        let _pass_handle = graph.add_graphics_pass(pass);

        let mut schedule = pipeline.begin_frame();
        let graph_handle = schedule.submit(format!("frame_{}", self.frame_count), graph, &[]);
        schedule.finish(&[graph_handle]);
        pipeline.end_frame(schedule);

        swapchain_texture.present();

        log::info!(
            "Frame {} rendered (clear color: RGB({:.2}, {:.2}, {:.2}))",
            self.frame_count,
            r,
            g,
            b
        );

        self.frame_count += 1;
        true
    }

    fn is_complete(&self) -> bool {
        !matches!(self.result, TestResult::Running)
    }
}

impl ApplicationHandler for WindowTestApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.resumed = true;

            let window_attributes = Window::default_attributes()
                .with_title("RedLilium Window Test")
                .with_inner_size(winit::dpi::LogicalSize::new(
                    self.window_size.0,
                    self.window_size.1,
                ))
                .with_visible(true);

            match event_loop.create_window(window_attributes) {
                Ok(window) => {
                    log::info!("Test window created successfully");
                    self.window = Some(window);

                    if !self.init_graphics() {
                        log::info!("Graphics initialization failed, skipping test");
                        self.result = TestResult::Skipped;
                        event_loop.exit();
                    }
                }
                Err(e) => {
                    log::info!("Window creation failed: {}", e);
                    self.result = TestResult::Skipped;
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested");
                self.result = TestResult::Failed;
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.window_size = (size.width.max(1), size.height.max(1));

                if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
                    let config = SurfaceConfiguration::new(self.window_size.0, self.window_size.1)
                        .with_format(surface.preferred_format())
                        .with_present_mode(PresentMode::Fifo);
                    let _ = surface.configure(device, &config);
                }
            }
            WindowEvent::RedrawRequested => {
                if self.pipeline.is_some() {
                    if !self.render_frame() {
                        log::warn!("Frame rendering failed");
                        self.result = TestResult::Failed;
                        event_loop.exit();
                        return;
                    }

                    if self.frame_count >= FRAMES_TO_RENDER {
                        log::info!(
                            "Successfully rendered {} frames, test passed!",
                            FRAMES_TO_RENDER
                        );
                        self.result = TestResult::Passed;

                        if let Some(pipeline) = &self.pipeline {
                            pipeline.wait_idle();
                        }

                        event_loop.exit();
                        return;
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

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

fn run_window_test(params: InstanceParameters) -> bool {
    log::info!(
        "Starting window integration test with backend: {:?}, wgpu_backend: {:?}",
        params.backend,
        params.wgpu_backend
    );

    #[cfg(target_os = "windows")]
    let mut event_loop = match EventLoop::builder().with_any_thread(true).build() {
        Ok(el) => el,
        Err(e) => {
            log::error!("Event loop creation failed: {}", e);
            return false;
        }
    };

    #[cfg(not(target_os = "windows"))]
    let mut event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            log::error!("Event loop creation failed: {}", e);
            return false;
        }
    };

    let mut app = WindowTestApp::new(params);

    let max_iterations = 1000;
    let mut iterations = 0;

    loop {
        let status = event_loop.pump_app_events(None, &mut app);

        match status {
            PumpStatus::Exit(_code) => {
                log::info!("Event loop exited");
                break;
            }
            PumpStatus::Continue => {
                if app.is_complete() {
                    break;
                }

                iterations += 1;
                if iterations >= max_iterations {
                    log::warn!("Test timed out after {} iterations", max_iterations);
                    app.result = TestResult::Failed;
                    break;
                }

                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    if let Some(pipeline) = &app.pipeline {
        pipeline.wait_idle();
    }

    match app.result {
        TestResult::Passed => {
            log::info!("Window test PASSED");
            true
        }
        TestResult::Skipped => {
            log::info!("Window test SKIPPED (no display available)");
            true
        }
        TestResult::Failed => {
            log::error!("Window test FAILED");
            false
        }
        TestResult::Running => {
            log::warn!("Window test ended in Running state (timeout)");
            false
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    let backend = args.get(1).map(|s| s.as_str()).unwrap_or("wgpu");

    let params = match backend {
        "vulkan" => {
            log::info!("Using Vulkan backend");
            InstanceParameters::new().with_backend(BackendType::Vulkan)
        }
        "wgpu" => {
            log::info!("Using wgpu backend with Auto mode");
            InstanceParameters::new()
                .with_backend(BackendType::Wgpu)
                .with_wgpu_backend(WgpuBackendType::Auto)
        }
        other => {
            log::info!("Unknown backend '{}', defaulting to wgpu with Auto", other);
            InstanceParameters::new()
                .with_backend(BackendType::Wgpu)
                .with_wgpu_backend(WgpuBackendType::Auto)
        }
    };

    let success = run_window_test(params);

    if success {
        log::info!("Test completed successfully!");
        std::process::exit(0);
    } else {
        log::error!("Test failed!");
        std::process::exit(1);
    }
}
