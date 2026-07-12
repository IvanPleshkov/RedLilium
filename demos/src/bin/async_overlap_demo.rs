//! # Async Compute Overlap Demo
//!
//! Minimal harness for profiling async compute queue overlap (#47 phase 4,
//! #82 step 5): every frame submits a transfer-only graph flagged
//! `set_prefer_async_compute(true)` alongside the main render graph. The two
//! graphs share no resources, so on hardware with a dedicated compute queue
//! family the copy traffic runs concurrently with the graphics work — capture
//! a frame with Radeon GPU Profiler (or Nsight) and the async queue's copies
//! should overlap the graphics queue's render pass on the timeline.
//!
//! On single-queue devices the async hint is ignored and everything runs on
//! the graphics queue — the demo still works, there is just nothing to
//! overlap.
//!
//! Run with `RUST_LOG=info` to see the queue plan
//! (`Async compute queue planned: ...`) and a once-a-second stats line.

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use redlilium_graphics::{
    BackendType, Buffer, BufferDescriptor, BufferUsage, ColorAttachment, FramePipeline,
    GraphicsDevice, GraphicsError, GraphicsInstance, GraphicsPass, InstanceParameters, LoadOp,
    PresentMode, RenderTargetConfig, StoreOp, Surface, SurfaceConfiguration, TransferConfig,
    TransferOperation, TransferPass,
};

/// Size of each ping-pong buffer. Large enough that the copies take a
/// measurable slice of GPU time next to a trivial clear pass.
const BUFFER_SIZE: u64 = 64 * 1024 * 1024;

/// Whole-buffer copies per frame (alternating A→B / B→A). At 64 MiB each
/// this moves 1 GiB of DMA traffic per frame — clearly visible on a
/// profiler timeline.
const COPIES_PER_FRAME: usize = 16;

struct App {
    window: Option<Arc<Window>>,
    instance: Option<Arc<GraphicsInstance>>,
    device: Option<Arc<GraphicsDevice>>,
    surface: Option<Arc<Surface>>,
    pipeline: Option<FramePipeline>,
    /// Ping-pong buffers the async graph copies between.
    buffers: Option<(Arc<Buffer>, Arc<Buffer>)>,
    window_size: (u32, u32),
    frame_count: u64,
    last_stats: Instant,
    frames_since_stats: u32,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            instance: None,
            device: None,
            surface: None,
            pipeline: None,
            buffers: None,
            window_size: (1280, 720),
            frame_count: 0,
            last_stats: Instant::now(),
            frames_since_stats: 0,
        }
    }

    fn init_graphics(&mut self) -> Result<(), GraphicsError> {
        let window = self.window.as_ref().expect("window created before init");

        // The native Vulkan backend is the only one with an async compute
        // queue (#47 phase 4) — the default (wgpu) would silently run
        // everything on one queue and there would be nothing to profile.
        let params = InstanceParameters::new().with_backend(BackendType::Vulkan);
        let instance = GraphicsInstance::with_parameters(params)?;
        let surface = instance.create_surface(window.clone())?;
        let device = instance.create_device_for_surface(&surface)?;

        let config = SurfaceConfiguration::new(self.window_size.0, self.window_size.1)
            .with_format(surface.preferred_format())
            .with_present_mode(PresentMode::Fifo);
        surface.configure(&device, &config)?;

        let pipeline = device.create_pipeline(2);

        let usage = BufferUsage::COPY_SRC | BufferUsage::COPY_DST;
        let buffer_a = device.create_buffer(&BufferDescriptor::new(BUFFER_SIZE, usage))?;
        let buffer_b = device.create_buffer(&BufferDescriptor::new(BUFFER_SIZE, usage))?;

        log::info!(
            "Async overlap demo on {}: {} x {} MiB copies per frame alongside the render graph",
            device.name(),
            COPIES_PER_FRAME,
            BUFFER_SIZE / (1024 * 1024),
        );

        self.instance = Some(instance);
        self.device = Some(device);
        self.surface = Some(surface);
        self.pipeline = Some(pipeline);
        self.buffers = Some((buffer_a, buffer_b));
        Ok(())
    }

    fn render_frame(&mut self) {
        let (Some(surface), Some(pipeline), Some((buffer_a, buffer_b))) =
            (&self.surface, &mut self.pipeline, &self.buffers)
        else {
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

        let mut schedule = pipeline.begin_frame().expect("begin_frame failed");

        // Async graph: transfer-only ping-pong copies, no resources shared
        // with the render graph — eligible for the async queue and free to
        // overlap it.
        let mut async_graph = schedule.acquire_graph();
        async_graph.set_prefer_async_compute(true);
        let mut copy_pass = TransferPass::new("async_pingpong_copies".into());
        let mut config = TransferConfig::new();
        for i in 0..COPIES_PER_FRAME {
            let (src, dst) = if i % 2 == 0 {
                (buffer_a, buffer_b)
            } else {
                (buffer_b, buffer_a)
            };
            config = config.with_operation(TransferOperation::copy_buffer_whole(
                src.clone(),
                dst.clone(),
            ));
        }
        copy_pass.set_transfer_config(config);
        async_graph.add_transfer_pass(copy_pass);
        schedule.submit(async_graph);

        // Main render graph: animated clear straight to the swapchain.
        let hue = (self.frame_count % 360) as f32;
        let (r, g, b) = hue_to_rgb(hue);
        let mut render_graph = schedule.acquire_graph();
        let mut pass = GraphicsPass::new("main_render".into());
        pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_surface(&swapchain_texture)
                    .with_load_op(LoadOp::clear_color(r, g, b, 1.0))
                    .with_store_op(StoreOp::Store),
            ),
        );
        render_graph.add_graphics_pass(pass);
        schedule.submit(render_graph);

        pipeline.end_frame(schedule);

        if let Err(e) = swapchain_texture.present() {
            log::warn!("Present reported: {e}");
        }

        self.frame_count += 1;
        self.frames_since_stats += 1;
        let elapsed = self.last_stats.elapsed();
        if elapsed.as_secs_f32() >= 1.0 {
            let fps = self.frames_since_stats as f32 / elapsed.as_secs_f32();
            let gib_per_sec =
                fps * (COPIES_PER_FRAME as f32 * BUFFER_SIZE as f32) / (1024.0 * 1024.0 * 1024.0);
            log::info!(
                "{fps:.0} fps, async copy traffic ~{gib_per_sec:.1} GiB/s (frame {})",
                self.frame_count
            );
            self.last_stats = Instant::now();
            self.frames_since_stats = 0;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attributes = Window::default_attributes()
                .with_title("RedLilium — Async Compute Overlap Demo")
                .with_inner_size(winit::dpi::LogicalSize::new(
                    self.window_size.0,
                    self.window_size.1,
                ));
            match event_loop.create_window(attributes) {
                Ok(window) => {
                    self.window = Some(Arc::new(window));
                    if let Err(e) = self.init_graphics() {
                        log::error!("Graphics initialization failed: {e}");
                        event_loop.exit();
                    }
                }
                Err(e) => {
                    log::error!("Failed to create window: {e}");
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(pipeline) = &self.pipeline {
                    let _ = pipeline.wait_idle();
                }
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
                self.render_frame();
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

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting RedLilium async compute overlap demo");
    redlilium_core::init();
    redlilium_graphics::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");
}
