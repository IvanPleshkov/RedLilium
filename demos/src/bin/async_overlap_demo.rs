//! # Async Compute Overlap Demo
//!
//! Minimal harness for profiling async compute queue overlap (#47 phase 4,
//! #82 step 5): every frame submits a transfer-only graph flagged
//! `QueuePreference::AsyncCompute` alongside the main render graph. The two
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
    MaterialDescriptor, MaterialInstance, Mesh, MeshDescriptor, PresentMode, QueuePreference,
    RenderTargetConfig, ShaderSource, StoreOp, Surface, SurfaceConfiguration, Texture,
    TextureDescriptor, TextureFormat, TextureUsage, TransferConfig, TransferOperation,
    TransferPass, VertexAttribute, VertexBufferLayout, VertexLayout,
};

/// Solid-color fullscreen shader (WGSL): the render/depth-target views of
/// GPU profilers only list targets that receive DRAW calls, so the
/// compression-inspection targets need real draws, not just clears.
const SOLID_COLOR_SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.2, 0.6, 0.9, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

/// Fullscreen quad (two CCW triangles) in clip space.
const FULLSCREEN_QUAD: [QuadVertex; 6] = [
    QuadVertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        position: [1.0, -1.0, 0.0],
        uv: [1.0, 1.0],
    },
    QuadVertex {
        position: [1.0, 1.0, 0.0],
        uv: [1.0, 0.0],
    },
    QuadVertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 1.0],
    },
    QuadVertex {
        position: [1.0, 1.0, 0.0],
        uv: [1.0, 0.0],
    },
    QuadVertex {
        position: [-1.0, 1.0, 0.0],
        uv: [0.0, 0.0],
    },
];

/// Size of each ping-pong buffer. Large enough that the copies take a
/// measurable slice of GPU time next to a trivial clear pass.
const BUFFER_SIZE: u64 = 64 * 1024 * 1024;

/// Whole-buffer copies per frame (alternating A→B / B→A). At 64 MiB each
/// this moves 1 GiB of DMA traffic per frame — clearly visible on a
/// profiler timeline.
const COPIES_PER_FRAME: usize = 16;

/// Size of the EXCLUSIVE (undeclared) offscreen render target — in a
/// profiler's resource/render-target view compare its solid-fill duration
/// against the cross-queue one (#88 validation). 8192² RGBA8 = 256 MiB,
/// deliberately larger than any Infinity Cache so the fill is
/// bandwidth-bound and a compression difference shows up as fill time.
const RT_EXCLUSIVE_SIZE: u32 = 8192;

/// Size of the `cross_queue`-declared render target (CONCURRENT on AMD, or
/// EXCLUSIVE via the maintenance9 fast path). Same size as the EXCLUSIVE one
/// so their solid-fill draw durations compare directly — solid color is the
/// best case for framebuffer compression, so a compression difference shows
/// up as a fill-rate difference. (RT #0 = exclusive, drawn first; RT #1 =
/// cross-queue.) Read back by the async graph every frame, so the per-frame
/// cross-queue ping-pong synchronization is also exercised.
const RT_CROSS_QUEUE_SIZE: u32 = 8192;

struct App {
    window: Option<Arc<Window>>,
    instance: Option<Arc<GraphicsInstance>>,
    device: Option<Arc<GraphicsDevice>>,
    surface: Option<Arc<Surface>>,
    pipeline: Option<FramePipeline>,
    /// Ping-pong buffers the async graph copies between.
    buffers: Option<(Arc<Buffer>, Arc<Buffer>)>,
    /// Offscreen render targets for compression inspection (#88): the
    /// undeclared EXCLUSIVE one and the `cross_queue`-declared one, plus the
    /// buffer the async graph copies the latter into each frame.
    render_targets: Option<(Arc<Texture>, Arc<Texture>, Arc<Buffer>)>,
    /// Fullscreen quad drawn into both inspection targets (profilers only
    /// list render targets that receive draw calls).
    quad: Option<(Arc<Mesh>, Arc<MaterialInstance>)>,
    /// One-shot vertex upload, folded into the first frame's render graph.
    pending_upload: Option<TransferOperation>,
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
            render_targets: None,
            quad: None,
            pending_upload: None,
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

        // Compression-inspection targets (#88): identical usage, different
        // declaration — compare their compression state in a GPU profiler
        // (they are distinguishable by size: 2048² EXCLUSIVE, 1024² shared).
        let rt_usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC;
        let rt_exclusive = device.create_texture(
            &TextureDescriptor::new_2d(
                RT_EXCLUSIVE_SIZE,
                RT_EXCLUSIVE_SIZE,
                TextureFormat::Rgba8Unorm,
                rt_usage,
            )
            .with_label("rt_exclusive"),
        )?;
        let rt_cross_queue = device.create_texture(
            &TextureDescriptor::new_2d(
                RT_CROSS_QUEUE_SIZE,
                RT_CROSS_QUEUE_SIZE,
                TextureFormat::Rgba8Unorm,
                rt_usage,
            )
            .with_cross_queue(true)
            .with_label("rt_cross_queue"),
        )?;
        // 1024 * 4 bytes per row is already 256-aligned.
        let rt_readback = device.create_buffer(&BufferDescriptor::new(
            (RT_CROSS_QUEUE_SIZE * RT_CROSS_QUEUE_SIZE * 4) as u64,
            BufferUsage::COPY_DST,
        ))?;

        // Fullscreen quad + solid-color material for the inspection targets.
        let layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(
                    std::mem::size_of::<QuadVertex>() as u32
                ))
                .with_attribute(VertexAttribute::position(0))
                .with_attribute(VertexAttribute::texcoord0(12))
                .with_label("quad_layout"),
        );
        let mesh = device.create_mesh(
            &MeshDescriptor::new(layout.clone())
                .with_vertex_count(FULLSCREEN_QUAD.len() as u32)
                .with_label("fullscreen_quad"),
        )?;
        let material = device.create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(
                    SOLID_COLOR_SHADER.as_bytes().to_vec(),
                    "vs_main",
                ))
                .with_shader(ShaderSource::fragment(
                    SOLID_COLOR_SHADER.as_bytes().to_vec(),
                    "fs_main",
                ))
                .with_vertex_layout(layout)
                .with_color_format(TextureFormat::Rgba8Unorm)
                .with_label("solid_color"),
        )?;
        let material_instance =
            Arc::new(MaterialInstance::new(material).with_label("solid_color_instance"));
        let vertex_buffer = mesh
            .vertex_buffer(0)
            .expect("quad mesh has a vertex buffer")
            .clone();
        let vertex_bytes: Arc<[u8]> = bytemuck::cast_slice(&FULLSCREEN_QUAD).to_vec().into();
        self.pending_upload = Some(TransferOperation::write_buffer(
            vertex_buffer,
            0,
            vertex_bytes,
        ));
        self.quad = Some((mesh, material_instance));

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
        self.render_targets = Some((rt_exclusive, rt_cross_queue, rt_readback));
        Ok(())
    }

    fn render_frame(&mut self) {
        let pending_upload = self.pending_upload.take();
        let (
            Some(surface),
            Some(pipeline),
            Some((buffer_a, buffer_b)),
            Some((rt_exclusive, rt_cross_queue, rt_readback)),
            Some((quad_mesh, quad_material)),
        ) = (
            &self.surface,
            &mut self.pipeline,
            &self.buffers,
            &self.render_targets,
            &self.quad,
        )
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

        // Async graph: transfer-only ping-pong copies (no shared resources,
        // free to overlap) plus a copy of last frame's cross-queue render
        // target — a per-frame cross-queue RAW ordered by tracker-emitted
        // timeline waits.
        let mut async_graph = schedule.acquire_graph();
        async_graph.set_queue_preference(QueuePreference::AsyncCompute);
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
        config = config.with_operation(TransferOperation::readback_texture_whole(
            rt_cross_queue.clone(),
            rt_readback.clone(),
        ));
        copy_pass.set_transfer_config(config);
        async_graph.add_transfer_pass(copy_pass);
        schedule.submit(async_graph);

        // Main render graph: draw a fullscreen quad into both offscreen
        // compression-inspection targets (profilers only list targets with
        // draw calls), then an animated clear straight to the swapchain. The
        // first frame prepends the quad's one-shot vertex upload.
        let hue = (self.frame_count % 360) as f32;
        let (r, g, b) = hue_to_rgb(hue);
        let mut render_graph = schedule.acquire_graph();
        if let Some(upload) = pending_upload {
            let mut upload_pass = TransferPass::new("quad_vertex_upload".into());
            upload_pass.set_transfer_config(TransferConfig::new().with_operation(upload));
            render_graph.add_transfer_pass(upload_pass);
        }
        let mut pass = GraphicsPass::new("rt_exclusive_draw".into());
        pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_texture(rt_exclusive.clone())
                    .with_load_op(LoadOp::clear_color(b, r, g, 1.0))
                    .with_store_op(StoreOp::Store),
            ),
        );
        pass.add_draw(quad_mesh.clone(), quad_material.clone());
        render_graph.add_graphics_pass(pass);
        let mut pass = GraphicsPass::new("rt_cross_queue_draw".into());
        pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_texture(rt_cross_queue.clone())
                    .with_load_op(LoadOp::clear_color(g, b, r, 1.0))
                    .with_store_op(StoreOp::Store),
            ),
        );
        pass.add_draw(quad_mesh.clone(), quad_material.clone());
        render_graph.add_graphics_pass(pass);
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
