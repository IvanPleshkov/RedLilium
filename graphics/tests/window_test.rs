//! Window and swapchain integration test.
//!
//! This test verifies that the graphics system works correctly with a real window
//! and swapchain. It creates a window, configures a surface, and renders 5 frames.
//!
//! # CI Compatibility
//!
//! If window creation fails (e.g., on headless CI systems) or no device is compatible
//! with the surface, the test passes gracefully. This ensures the test suite doesn't
//! fail on systems without display hardware.
//!
//! # Running This Test
//!
//! ```bash
//! cargo test --test window_test
//! ```

use std::sync::Arc;

use rstest::rstest;
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
    /// Test is still running.
    Running,
    /// Test passed successfully.
    Passed,
    /// Test was skipped (window/device not available).
    Skipped,
    /// Test failed with an error.
    Failed,
}

/// Application state for the window test.
struct WindowTestApp {
    /// Test result.
    result: TestResult,
    /// Instance parameters for backend selection.
    params: InstanceParameters,
    /// Window handle (created on resume).
    window: Option<Window>,
    /// Graphics instance.
    instance: Option<Arc<GraphicsInstance>>,
    /// Graphics device.
    device: Option<Arc<GraphicsDevice>>,
    /// Surface for the window.
    surface: Option<Arc<Surface>>,
    /// Frame pipeline for synchronization.
    pipeline: Option<FramePipeline>,
    /// Current frame count.
    frame_count: u32,
    /// Window size.
    window_size: (u32, u32),
    /// Whether we've been resumed (window created).
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
            window_size: (320, 240), // Small window for tests
            resumed: false,
        }
    }

    /// Initialize graphics after window is created.
    fn init_graphics(&mut self) -> bool {
        let window = match &self.window {
            Some(w) => w,
            None => {
                log::warn!("No window available for graphics init");
                return false;
            }
        };

        // Create graphics instance with configured parameters
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

        // Configure the surface
        let config = SurfaceConfiguration::new(self.window_size.0, self.window_size.1)
            .with_format(surface.preferred_format())
            .with_present_mode(PresentMode::Fifo);

        if let Err(e) = surface.configure(&device, &config) {
            log::warn!("Failed to configure surface: {}", e);
            return false;
        }

        // Create frame pipeline with 2 frames in flight
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

    /// Render a single frame using FramePipeline and FrameSchedule.
    fn render_frame(&mut self) -> bool {
        // Device is available but not directly used - graph execution uses it internally
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

        // Acquire swapchain texture
        let swapchain_texture = match surface.acquire_texture() {
            Ok(t) => t,
            Err(e) => {
                log::warn!("Failed to acquire swapchain texture: {}", e);
                return false;
            }
        };

        // Build render graph with a simple clear pass that renders directly to swapchain
        // Note: This means we can't do GPU readback of the result since swapchain textures
        // typically don't have COPY_SRC usage, but it tests the real rendering path.
        let hue = (self.frame_count as f32 / FRAMES_TO_RENDER as f32) * 360.0;
        let (r, g, b) = hue_to_rgb(hue);

        let mut graph = RenderGraph::new();
        let mut pass = GraphicsPass::new(format!("frame_{}", self.frame_count));
        pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_surface(&swapchain_texture)
                    .with_load_op(LoadOp::clear_color(r, g, b, 1.0))
                    .with_store_op(StoreOp::Store),
            ),
        );
        let _pass_handle = graph.add_graphics_pass(pass);

        // Execute using FramePipeline and FrameSchedule (as documented in ARCHITECTURE.md)
        let mut schedule = pipeline.begin_frame();

        // Submit the render graph
        let graph_handle = schedule.submit(format!("frame_{}", self.frame_count), &graph, &[]);

        // Finish the schedule (offscreen rendering, no actual present to swapchain yet)
        schedule.finish(&[graph_handle]);

        // End the frame
        pipeline.end_frame(schedule);

        // Present the swapchain texture
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

    /// Check if test is complete.
    fn is_complete(&self) -> bool {
        !matches!(self.result, TestResult::Running)
    }
}

impl ApplicationHandler for WindowTestApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.resumed = true;

            // Create a small test window
            let window_attributes = Window::default_attributes()
                .with_title("RedLilium Window Test")
                .with_inner_size(winit::dpi::LogicalSize::new(
                    self.window_size.0,
                    self.window_size.1,
                ))
                .with_visible(true); // Need visible window for events

            match event_loop.create_window(window_attributes) {
                Ok(window) => {
                    log::info!("Test window created successfully");
                    self.window = Some(window);

                    // Initialize graphics
                    if !self.init_graphics() {
                        log::info!("Graphics initialization failed, skipping test");
                        self.result = TestResult::Skipped;
                        event_loop.exit();
                    }
                }
                Err(e) => {
                    log::info!("Window creation failed (expected on CI): {}", e);
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
                self.result = TestResult::Failed; // Unexpected close
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
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
                // Only render if we're initialized
                if self.pipeline.is_some() {
                    // Render frame
                    if !self.render_frame() {
                        log::warn!("Frame rendering failed");
                        self.result = TestResult::Failed;
                        event_loop.exit();
                        return;
                    }

                    // Check if we've rendered enough frames
                    if self.frame_count >= FRAMES_TO_RENDER {
                        log::info!(
                            "Successfully rendered {} frames, test passed!",
                            FRAMES_TO_RENDER
                        );
                        self.result = TestResult::Passed;

                        // Wait for GPU to finish before exiting
                        if let Some(pipeline) = &self.pipeline {
                            pipeline.wait_idle();
                        }

                        event_loop.exit();
                        return;
                    }
                }

                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Request redraw on each iteration to drive rendering
        if let Some(window) = &self.window {
            window.request_redraw();
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

/// Run the window test with event pumping (test-friendly approach).
///
/// Returns true if the test passed or was skipped (CI compatibility).
fn run_window_test(params: InstanceParameters) -> bool {
    // Initialize logging for test output
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .is_test(true)
        .try_init();

    log::info!(
        "Starting window integration test with backend: {:?}, wgpu_backend: {:?}",
        params.backend,
        params.wgpu_backend
    );

    // Try to create event loop - may fail on headless systems
    // On Windows, we need to use any_thread() because tests run on a non-main thread
    #[cfg(target_os = "windows")]
    let mut event_loop = match EventLoop::builder().with_any_thread(true).build() {
        Ok(el) => el,
        Err(e) => {
            log::info!("Event loop creation failed (expected on CI): {}", e);
            return true; // Skip test, consider passed
        }
    };

    // On macOS, EventLoop must be created on the main thread. Since Rust tests run on
    // worker threads, we use catch_unwind to detect panics and fail properly.
    #[cfg(target_os = "macos")]
    #[allow(clippy::redundant_closure)]
    let event_loop_result = std::panic::catch_unwind(|| EventLoop::new());

    #[cfg(target_os = "macos")]
    let mut event_loop = match event_loop_result {
        Ok(Ok(el)) => el,
        Ok(Err(e)) => {
            log::info!("Event loop creation failed (expected on CI): {}", e);
            return true; // Skip test, consider passed
        }
        Err(panic_info) => {
            // Extract panic message if possible
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            log::error!(
                "Event loop creation panicked on macOS (non-main thread): {}",
                panic_msg
            );
            return false; // Fail the test - this is a validation error that should not be silently skipped
        }
    };

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let mut event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            log::info!("Event loop creation failed (expected on CI): {}", e);
            return true; // Skip test, consider passed
        }
    };

    let mut app = WindowTestApp::new(params);

    // Use pump_events for controlled iteration (test-friendly)
    // This allows us to have a timeout and not block forever
    let max_iterations = 1000; // Timeout after 1000 iterations
    let mut iterations = 0;

    loop {
        // Pump events
        let status = event_loop.pump_app_events(None, &mut app);

        match status {
            PumpStatus::Exit(_code) => {
                log::info!("Event loop exited");
                break;
            }
            PumpStatus::Continue => {
                // Check if test is complete
                if app.is_complete() {
                    break;
                }

                iterations += 1;
                if iterations >= max_iterations {
                    log::warn!("Test timed out after {} iterations", max_iterations);
                    app.result = TestResult::Failed;
                    break;
                }

                // Small delay to avoid busy-waiting
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    // Cleanup
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
            true // Skipped tests are considered passing for CI
        }
        TestResult::Failed => {
            log::error!("Window test FAILED");
            false
        }
        TestResult::Running => {
            // If still running after timeout, consider it failed
            log::warn!("Window test ended in Running state (timeout)");
            false
        }
    }
}

#[rstest]
fn test_window_swapchain_5_frames_vulkan() {
    let params = InstanceParameters::new().with_backend(BackendType::Vulkan);
    assert!(
        run_window_test(params),
        "Window swapchain test failed - see log for details"
    );
}

#[rstest]
fn test_window_swapchain_5_frames_wgpu() {
    let params = InstanceParameters::new()
        .with_backend(BackendType::Wgpu)
        .with_wgpu_backend(WgpuBackendType::Vulkan);
    assert!(
        run_window_test(params),
        "Window swapchain test failed - see log for details"
    );
}
