//! # Textured Quad Demo
//!
//! Demonstrates:
//! - Simple quad mesh rendering (centered, not fullscreen)
//! - Texture downloading from URL
//! - Texture upload via render graph TransferPass
//! - Frame scheduler and frame pipeline usage
//! - Resize manager from graphics crate
//! - App framework usage
//!
//! The demo downloads the famous Lenna test image and renders it on a quad.

use std::sync::Arc;

use redlilium_app::{App, AppArgs, AppContext, AppHandler, DefaultAppArgs, DrawContext};
use redlilium_core::math::{Mat4, Vec3, look_at_rh, mat4_to_cols_array_2d, orthographic_rh};
use redlilium_core::mesh::generators;
use redlilium_graphics::{
    AddressMode, BindingGroupDescriptor, BufferUsage, ColorAttachment, DepthConvention, DepthState,
    DepthStencilAttachment, DrawCommand, FilterMode, FrameSchedule, GraphicsPass, Material,
    MaterialDescriptor, MaterialInstance, Mesh, RenderTargetConfig, RingBuffer, SamplerDescriptor,
    ShaderSource, ShaderStage, TextureDescriptor, TextureFormat, TextureUsage, TransferConfig,
    TransferOperation, TransferPass, VertexLayout,
    resize::{ResizeManager, ResizeStrategy},
};

// === Slang Shader ===

/// Simple textured quad shader with MVP matrix (Slang)
const QUAD_SHADER_SLANG: &str = r#"
[[vk::binding(0, 0)]]
cbuffer Uniforms {
    float4x4 mvp;
};

[[vk::binding(1, 0)]]
Texture2D quad_texture;
[[vk::binding(2, 0)]]
SamplerState quad_sampler;

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
};

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    output.position = mul(mvp, float4(input.position, 1.0));
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    return quad_texture.Sample(quad_sampler, input.uv);
}
"#;

// === Uniform Data ===

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
}

// === Image Loading ===

const LENNA_URL: &str = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png";

fn load_image_from_url(url: &str) -> Result<(u32, u32, Vec<u8>), String> {
    use std::io::Read;

    log::info!("Downloading image from: {}", url);

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Failed to download image: {e}"))?;

    let mut data = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut data)
        .map_err(|e| format!("Failed to read image data: {e}"))?;

    log::info!("Downloaded {} bytes, parsing PNG...", data.len());

    let img = image::load_from_memory(&data).map_err(|e| format!("Failed to decode image: {e}"))?;

    let width = img.width();
    let height = img.height();

    log::info!("Image loaded: {}x{}", width, height);

    // Convert to RGBA8
    let rgba = img.to_rgba8();
    Ok((width, height, rgba.into_raw()))
}

/// Generate a tightly-packed RGBA8 checkerboard `(width, height, data)`.
fn generate_checkerboard(width: u32, height: u32, cell: u32) -> (u32, u32, Vec<u8>) {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let on = ((x / cell) + (y / cell)).is_multiple_of(2);
            let (r, g, b) = if on { (220, 90, 90) } else { (40, 40, 60) };
            data.extend_from_slice(&[r, g, b, 255]);
        }
    }
    (width, height, data)
}

// === Demo Application ===

struct TexturedQuadDemo {
    // GPU resources
    material: Option<Arc<Material>>,
    material_instance: Option<Arc<MaterialInstance>>,
    mesh: Option<Arc<Mesh>>,
    /// Per-frame MVP uniform ring (dynamic offset per draw) — race-free across
    /// frames in flight.
    uniform_ring: Option<RingBuffer>,
    /// This frame's offset into `uniform_ring`.
    uniform_offset: u32,
    depth_texture: Option<Arc<redlilium_graphics::Texture>>,
    texture: Option<Arc<redlilium_graphics::Texture>>,

    /// Mesh/texture uploads queued at init, flushed into the first frame's graph.
    pending_uploads: Vec<TransferOperation>,

    // Resize manager
    resize_manager: ResizeManager,
}

impl TexturedQuadDemo {
    fn new() -> Self {
        Self {
            material: None,
            material_instance: None,
            mesh: None,
            uniform_ring: None,
            uniform_offset: 0,
            depth_texture: None,
            texture: None,
            pending_uploads: Vec::new(),
            // Initial size will be updated in on_init
            resize_manager: ResizeManager::new((1280, 720), 50, ResizeStrategy::Stretch),
        }
    }

    fn create_gpu_resources(&mut self, ctx: &mut AppContext) {
        let device = ctx.device();

        // Create vertex layout for position + uv
        let vertex_layout = VertexLayout::position_uv();

        // Load image from URL, falling back to a generated checkerboard if the
        // download fails (e.g. offline or rate-limited).
        log::info!("Loading Lenna test image...");
        let (tex_width, tex_height, rgba_data) =
            load_image_from_url(LENNA_URL).unwrap_or_else(|e| {
                log::warn!("Image download failed ({e}); using generated checkerboard");
                generate_checkerboard(512, 512, 32)
            });

        // Create texture and queue its upload through the frame graph (Lenna is
        // 512×512, so its rows are already 256-byte aligned — no padding needed).
        let texture = device
            .create_texture(
                &TextureDescriptor::new_2d(
                    tex_width,
                    tex_height,
                    TextureFormat::Rgba8UnormSrgb,
                    TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                )
                .with_label("lenna_texture"),
            )
            .expect("Failed to create texture");
        self.pending_uploads.push(
            TransferOperation::upload_texture_data(device, texture.clone(), &rgba_data)
                .expect("Failed to stage texture upload"),
        );
        self.texture = Some(texture);

        // Create sampler
        let sampler = device
            .create_sampler(&SamplerDescriptor {
                label: Some("quad_sampler".into()),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Linear,
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                ..Default::default()
            })
            .expect("Failed to create sampler");

        // Create material using Slang shader source
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        QUAD_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        QUAD_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_vertex_layout(vertex_layout.clone())
                    .with_color_format(ctx.surface_format())
                    // This demo deliberately runs the CLASSIC depth convention
                    // (#125 opt-out proof): classic projection above, clear
                    // depth 1.0 in the pass, and LessEqual here — the three
                    // must flip together (one convention per depth target,
                    // ADR-038). Everything else in the engine is reversed-Z.
                    .with_depth(Some(DepthState::for_convention(
                        DepthConvention::Classic,
                        TextureFormat::Depth32Float,
                    )))
                    // Binding 0 (MVP) is a per-draw dynamic offset into a ring.
                    .with_dynamic_uniform(0, 0)
                    .with_label("quad_material"),
            )
            .expect("Failed to create material");
        self.material = Some(material.clone());

        // Per-frame MVP ring (one element bound; the draw picks the frame's slot
        // via a dynamic offset). Sized well above frames-in-flight.
        let uniform_ring = RingBuffer::new(
            device,
            1 << 16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "quad_mvp_ring",
        )
        .expect("Failed to create uniform ring");

        // Create material instance with bindings
        let layout = material.binding_layouts()[0].clone();
        let binding_group = device
            .create_binding_group(
                layout,
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        uniform_ring.buffer().clone(),
                        0,
                        size_of::<Uniforms>() as u64,
                    )
                    .with_texture(1, self.texture.clone().unwrap())
                    .with_sampler(2, sampler),
            )
            .expect("create binding group");
        self.uniform_ring = Some(uniform_ring);

        let material_instance =
            Arc::new(MaterialInstance::new(material).with_binding_group(binding_group));
        self.material_instance = Some(material_instance);

        // Create quad mesh (centered, aspect-ratio correct); its data uploads
        // through the frame graph on the first frame.
        let aspect = tex_width as f32 / tex_height as f32;
        let quad_cpu = generators::generate_quad(0.5 * aspect, 0.5);
        let (mesh, mesh_ops) = device
            .create_mesh_deferred(&quad_cpu)
            .expect("Failed to create quad mesh");
        self.pending_uploads.extend(mesh_ops);
        self.mesh = Some(mesh);

        // Create depth texture
        self.create_depth_texture(ctx);

        log::info!("GPU resources created successfully");
    }

    fn create_depth_texture(&mut self, ctx: &AppContext) {
        let depth_texture = ctx
            .device()
            .create_texture(
                &TextureDescriptor::new_2d(
                    ctx.width(),
                    ctx.height(),
                    TextureFormat::Depth32Float,
                    TextureUsage::RENDER_ATTACHMENT,
                )
                .with_label("depth_texture"),
            )
            .expect("Failed to create depth texture");
        self.depth_texture = Some(depth_texture);
    }

    fn update_uniform_buffer(&mut self, ctx: &AppContext) {
        // Create orthographic projection that keeps the quad centered and visible
        // The quad is sized with aspect ratio consideration, so we use a simple ortho
        let aspect = ctx.aspect_ratio();

        // Scale to show the quad at a reasonable size (not fullscreen)
        let scale = 1.5;

        // Classic-convention projection — see the material creation comment
        // (#125 opt-out proof).
        let proj = if aspect > 1.0 {
            // Wider than tall
            orthographic_rh(-scale * aspect, scale * aspect, -scale, scale, -1.0, 1.0)
        } else {
            // Taller than wide
            orthographic_rh(-scale, scale, -scale / aspect, scale / aspect, -1.0, 1.0)
        };

        let eye = Vec3::new(0.0, 0.0, 1.0);
        let target = Vec3::zeros();
        let up = Vec3::new(0.0, 1.0, 0.0);
        let view = look_at_rh(&eye, &target, &up);
        let model = Mat4::identity();
        let mvp = proj * view * model;

        let uniforms = Uniforms {
            mvp: mat4_to_cols_array_2d(&mvp),
        };

        // Write this frame's MVP into the next ring slot; the draw binds it via
        // a dynamic offset (race-free across frames in flight).
        if let Some(ring) = &mut self.uniform_ring {
            let bytes = bytemuck::bytes_of(&uniforms);
            let size = bytes.len() as u64;
            let alloc = ring.allocate(size).unwrap_or_else(|| {
                ring.reset();
                ring.allocate(size).expect("uniform ring too small")
            });
            ring.write(&alloc, bytes)
                .expect("Failed to write uniform ring");
            self.uniform_offset = alloc.offset as u32;
        }
    }
}

impl AppHandler for TexturedQuadDemo {
    fn on_init(&mut self, ctx: &mut AppContext) {
        log::info!("Initializing Textured Quad Demo");
        log::info!("This demo downloads the Lenna test image and renders it on a centered quad.");

        // Initialize resize manager with actual window size
        self.resize_manager =
            ResizeManager::new((ctx.width(), ctx.height()), 50, ResizeStrategy::Stretch);

        self.create_gpu_resources(ctx);
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        // Notify resize manager of the resize event
        self.resize_manager
            .on_resize_event(ctx.width(), ctx.height());

        // Apply resize immediately since App already handles debouncing
        if self.resize_manager.update().is_some() {
            self.create_depth_texture(ctx);
        } else {
            // Force resize since we know the window actually resized
            self.resize_manager.force_resize();
            self.create_depth_texture(ctx);
        }
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        // Check for pending resize from resize manager
        if self.resize_manager.update().is_some() {
            self.create_depth_texture(ctx);
        }

        self.update_uniform_buffer(ctx);
        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        let mut graph = ctx.acquire_graph();

        // Flush queued mesh/texture uploads through the frame graph (first frame).
        if !self.pending_uploads.is_empty() {
            let ops = std::mem::take(&mut self.pending_uploads);
            let mut transfer_pass = TransferPass::new("resource_uploads".into());
            transfer_pass.set_transfer_config(TransferConfig::new().with_operations(ops));
            graph.add_transfer_pass(transfer_pass);
            log::info!("Mesh + texture uploaded via transfer pass");
        }

        // Create main render pass
        let mut render_pass = GraphicsPass::new("main".into());

        if let Some(depth) = &self.depth_texture {
            render_pass.set_render_targets(
                RenderTargetConfig::new()
                    .with_color(
                        ColorAttachment::from_surface(ctx.swapchain_texture())
                            .with_clear_color(0.1, 0.1, 0.15, 1.0), // Dark background
                    )
                    .with_depth_stencil(
                        // Classic clear (1.0) — this demo is the #125 classic
                        // opt-out proof; see the material creation comment.
                        DepthStencilAttachment::from_texture(depth.clone())
                            .with_clear_depth(DepthConvention::Classic.clear_depth()),
                    ),
            );
        }

        // Draw the quad, selecting this frame's MVP ring slot via dynamic offset.
        if let (Some(mesh), Some(material_instance)) = (&self.mesh, &self.material_instance) {
            render_pass.add_draw_command(
                DrawCommand::new(mesh.clone(), material_instance.clone())
                    .with_dynamic_offsets(vec![vec![self.uniform_offset]]),
            );
        }

        graph.add_graphics_pass(render_pass);

        ctx.render(graph)
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down Textured Quad Demo");
    }
}

// === Entry Point ===

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // Parse command line arguments and set the window title
    let args = DefaultAppArgs::parse().with_title_str("Textured Quad Demo");
    App::run(TexturedQuadDemo::new(), args);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Entry point for wasm
}
