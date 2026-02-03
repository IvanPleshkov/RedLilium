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

use glam::{Mat4, Vec3};
use redlilium_app::{App, AppContext, AppHandler, DefaultAppArgs, DrawContext};
use redlilium_graphics::{
    AddressMode, BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BufferDescriptor,
    BufferUsage, ColorAttachment, DepthStencilAttachment, Extent3d, FilterMode, FrameSchedule,
    GraphicsPass, IndexFormat, Material, MaterialDescriptor, MaterialInstance, Mesh,
    MeshDescriptor, RenderGraph, RenderTargetConfig, SamplerDescriptor, ShaderSource, ShaderStage,
    ShaderStageFlags, TextureDescriptor, TextureFormat, TextureUsage, TransferConfig,
    TransferOperation, TransferPass, VertexAttribute, VertexAttributeFormat,
    VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
    resize::{ResizeManager, ResizeStrategy},
};
use redlilium_graphics::{BufferTextureCopyRegion, BufferTextureLayout, TextureCopyLocation};

// === WGSL Shader ===

/// Simple textured quad shader with MVP matrix
const QUAD_SHADER_WGSL: &str = r#"
// Camera uniforms with MVP matrix
struct Uniforms {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var texture_image: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(3) uv: vec2<f32>,  // TexCoord0 maps to location 3
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(texture_image, texture_sampler, in.uv);
}
"#;

// === Vertex Data ===

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

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

// === Demo Application ===

struct TexturedQuadDemo {
    // GPU resources
    material: Option<Arc<Material>>,
    material_instance: Option<Arc<MaterialInstance>>,
    mesh: Option<Arc<Mesh>>,
    uniform_buffer: Option<Arc<redlilium_graphics::Buffer>>,
    depth_texture: Option<Arc<redlilium_graphics::Texture>>,
    texture: Option<Arc<redlilium_graphics::Texture>>,

    // Staging buffer for texture upload
    staging_buffer: Option<Arc<redlilium_graphics::Buffer>>,
    texture_size: (u32, u32),
    aligned_bytes_per_row: u32,
    needs_texture_upload: bool,

    // Resize manager
    resize_manager: ResizeManager,
}

impl TexturedQuadDemo {
    fn new() -> Self {
        Self {
            material: None,
            material_instance: None,
            mesh: None,
            uniform_buffer: None,
            depth_texture: None,
            texture: None,
            staging_buffer: None,
            texture_size: (0, 0),
            aligned_bytes_per_row: 0,
            needs_texture_upload: false,
            // Initial size will be updated in on_init
            resize_manager: ResizeManager::new((1280, 720), 50, ResizeStrategy::Stretch),
        }
    }

    fn create_gpu_resources(&mut self, ctx: &mut AppContext) {
        let device = ctx.device();

        // Create vertex layout for position + uv
        let vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(
                    std::mem::size_of::<QuadVertex>() as u32
                ))
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::Position,
                    format: VertexAttributeFormat::Float3,
                    offset: 0,
                    buffer_index: 0,
                })
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::TexCoord0,
                    format: VertexAttributeFormat::Float2,
                    offset: 12,
                    buffer_index: 0,
                })
                .with_label("quad_vertex_layout"),
        );

        // Load image from URL
        log::info!("Loading Lenna test image...");
        let (tex_width, tex_height, rgba_data) =
            load_image_from_url(LENNA_URL).expect("Failed to load Lenna image");
        self.texture_size = (tex_width, tex_height);

        // Create texture
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
        self.texture = Some(texture);

        // Create staging buffer with aligned bytes per row (256-byte alignment for WebGPU)
        const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
        let bytes_per_pixel = 4u32; // RGBA8
        let bytes_per_row = tex_width * bytes_per_pixel;
        let aligned_bytes_per_row =
            bytes_per_row.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
        self.aligned_bytes_per_row = aligned_bytes_per_row;

        // Pad data if alignment is needed
        let padded_data = if aligned_bytes_per_row != bytes_per_row {
            let mut padded = vec![0u8; (aligned_bytes_per_row * tex_height) as usize];
            for y in 0..tex_height {
                let src_start = (y * bytes_per_row) as usize;
                let src_end = src_start + bytes_per_row as usize;
                let dst_start = (y * aligned_bytes_per_row) as usize;
                padded[dst_start..dst_start + bytes_per_row as usize]
                    .copy_from_slice(&rgba_data[src_start..src_end]);
            }
            padded
        } else {
            rgba_data
        };

        let staging_buffer = device
            .create_buffer(&BufferDescriptor::new(
                padded_data.len() as u64,
                BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create staging buffer");
        device.write_buffer(&staging_buffer, 0, &padded_data);
        self.staging_buffer = Some(staging_buffer);
        self.needs_texture_upload = true;

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

        // Create binding layout
        let binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::Sampler)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(2, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_label("quad_bindings"),
        );

        // Create material
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        QUAD_SHADER_WGSL.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        QUAD_SHADER_WGSL.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(binding_layout)
                    .with_vertex_layout(vertex_layout.clone())
                    .with_label("quad_material"),
            )
            .expect("Failed to create material");
        self.material = Some(material.clone());

        // Create uniform buffer
        let uniform_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<Uniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create uniform buffer");
        self.uniform_buffer = Some(uniform_buffer.clone());

        // Create material instance with bindings
        #[allow(clippy::arc_with_non_send_sync)]
        let binding_group = Arc::new(
            BindingGroup::new()
                .with_buffer(0, uniform_buffer)
                .with_sampler(1, sampler)
                .with_texture(2, self.texture.clone().unwrap()),
        );

        let material_instance =
            Arc::new(MaterialInstance::new(material).with_binding_group(binding_group));
        self.material_instance = Some(material_instance);

        // Create quad mesh (centered, aspect-ratio correct)
        // The quad will be sized to display the texture with correct aspect ratio
        let aspect = tex_width as f32 / tex_height as f32;
        let half_width = 0.5 * aspect;
        let half_height = 0.5;

        let vertices = [
            QuadVertex {
                position: [-half_width, -half_height, 0.0],
                uv: [0.0, 1.0],
            },
            QuadVertex {
                position: [half_width, -half_height, 0.0],
                uv: [1.0, 1.0],
            },
            QuadVertex {
                position: [half_width, half_height, 0.0],
                uv: [1.0, 0.0],
            },
            QuadVertex {
                position: [-half_width, half_height, 0.0],
                uv: [0.0, 0.0],
            },
        ];

        let indices: [u32; 6] = [0, 1, 2, 2, 3, 0];

        let mesh = device
            .create_mesh(
                &MeshDescriptor::new(vertex_layout)
                    .with_vertex_count(vertices.len() as u32)
                    .with_indices(IndexFormat::Uint32, indices.len() as u32)
                    .with_label("quad"),
            )
            .expect("Failed to create mesh");

        if let Some(vb) = mesh.vertex_buffer(0) {
            device.write_buffer(vb, 0, bytemuck::cast_slice(&vertices));
        }
        if let Some(ib) = mesh.index_buffer() {
            device.write_buffer(ib, 0, bytemuck::cast_slice(&indices));
        }
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

    fn update_uniform_buffer(&self, ctx: &AppContext) {
        // Create orthographic projection that keeps the quad centered and visible
        // The quad is sized with aspect ratio consideration, so we use a simple ortho
        let aspect = ctx.aspect_ratio();

        // Scale to show the quad at a reasonable size (not fullscreen)
        let scale = 1.5;

        let proj = if aspect > 1.0 {
            // Wider than tall
            Mat4::orthographic_rh(-scale * aspect, scale * aspect, -scale, scale, -1.0, 1.0)
        } else {
            // Taller than wide
            Mat4::orthographic_rh(-scale, scale, -scale / aspect, scale / aspect, -1.0, 1.0)
        };

        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO, Vec3::Y);
        let model = Mat4::IDENTITY;
        let mvp = proj * view * model;

        let uniforms = Uniforms {
            mvp: mvp.to_cols_array_2d(),
        };

        if let Some(buffer) = &self.uniform_buffer {
            ctx.device()
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    fn create_texture_transfer_config(&self) -> TransferConfig {
        let mut config = TransferConfig::new();

        if let (Some(staging), Some(texture)) = (&self.staging_buffer, &self.texture) {
            let (width, height) = self.texture_size;
            let region = BufferTextureCopyRegion::new(
                BufferTextureLayout::new(0, Some(self.aligned_bytes_per_row), None),
                TextureCopyLocation::base(),
                Extent3d::new_2d(width, height),
            );
            config = config.with_operation(TransferOperation::upload_texture(
                staging.clone(),
                texture.clone(),
                vec![region],
            ));
        }

        config
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
        let mut graph = RenderGraph::new();

        // Upload texture on first frame via TransferPass
        if self.needs_texture_upload {
            let transfer_config = self.create_texture_transfer_config();
            let mut transfer_pass = TransferPass::new("texture_upload".into());
            transfer_pass.set_transfer_config(transfer_config);
            graph.add_transfer_pass(transfer_pass);
            self.needs_texture_upload = false;
            log::info!("Texture uploaded via transfer pass");
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
                        DepthStencilAttachment::from_texture(depth.clone()).with_clear_depth(1.0),
                    ),
            );
        }

        // Draw the quad
        if let (Some(mesh), Some(material_instance)) = (&self.mesh, &self.material_instance) {
            render_pass.add_draw(mesh.clone(), material_instance.clone());
        }

        graph.add_graphics_pass(render_pass);

        let _handle = ctx.submit("main", &graph, &[]);
        ctx.finish(&[])
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down Textured Quad Demo");
    }
}

// === Entry Point ===

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args = DefaultAppArgs::with_title("Textured Quad Demo");
    App::run(TexturedQuadDemo::new(), args);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Entry point for wasm
}
