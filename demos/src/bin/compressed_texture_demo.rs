//! # Compressed Texture Demo (#120)
//!
//! Renders checked-in KTX2 fixtures with full mip chains next to their PNG
//! references:
//!
//! - top row: BC7 sRGB albedo (KTX2) | the same albedo from PNG | BC4
//!   grayscale (KTX2)
//! - bottom row: BC5 normal map lit with a fixed light (KTX2) | the same
//!   lighting from the PNG normal map
//!
//! Each KTX2 pair should be visually indistinguishable from its PNG
//! reference (block compression artifacts aside). The fixtures are produced
//! by `scripts/gen-compressed-fixtures.sh` (UASTC → `ktx transcode` route,
//! Zstd-supercompressed, 7 mips) and parsed by the core KTX2 module; every
//! (mip, layer) image uploads through the frame graph via
//! `TransferOperation::upload_texture_level` — the same path the asset
//! loader takes.
//!
//! On a device without BC support the compressed quads are skipped with a
//! warning (capability gating, #119); the PNG references still render.

use std::sync::Arc;

use redlilium_app::{App, AppArgs, AppContext, AppHandler, DefaultAppArgs, DrawContext};
use redlilium_core::math::{
    Mat4, Vec3, look_at_rh, mat4_to_cols_array_2d, orthographic_rh_reversed,
};
use redlilium_core::mesh::generators;
use redlilium_core::texture::CpuTexture;
use redlilium_core::texture::ktx2::parse_ktx2;
use redlilium_graphics::{
    AddressMode, BindingGroupDescriptor, BufferUsage, ColorAttachment, DepthConvention, DepthState,
    DepthStencilAttachment, DrawCommand, FilterMode, FrameSchedule, GraphicsPass, Material,
    MaterialDescriptor, MaterialInstance, Mesh, RenderTargetConfig, RingBuffer, SamplerDescriptor,
    ShaderSource, ShaderStage, Texture, TextureDescriptor, TextureFormat, TextureUsage,
    TransferConfig, TransferOperation, TransferPass, VertexLayout,
    resize::{ResizeManager, ResizeStrategy},
};

// === Checked-in fixtures ===

const ALBEDO_BC7: &[u8] = include_bytes!("../../assets/compressed/albedo_bc7.ktx2");
const NORMAL_BC5: &[u8] = include_bytes!("../../assets/compressed/normal_bc5.ktx2");
const GRAY_BC4: &[u8] = include_bytes!("../../assets/compressed/gray_bc4.ktx2");
const ALBEDO_PNG: &[u8] = include_bytes!("../../assets/compressed/albedo.png");
const NORMAL_PNG: &[u8] = include_bytes!("../../assets/compressed/normal.png");

// === Slang shader ===
//
// One shader, three display modes selected per draw through the dynamic
// uniform: 0 = sample color as-is, 1 = replicate the red channel (single-
// channel BC4), 2 = reconstruct a tangent-space normal from RG (BC5 stores
// two channels; z is rebuilt, so the PNG reference shades identically) and
// light it with a fixed directional light.
const SHADER_SLANG: &str = include_str!("../../shaders/compressed_quad.slang");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    mode: u32,
    _pad: [u32; 3],
}

/// One quad on screen: its texture's material instance, display mode, and
/// grid position (column, row).
struct Quad {
    instance: Arc<MaterialInstance>,
    mode: u32,
    cell: (f32, f32),
    /// This frame's dynamic offset into the uniform ring.
    uniform_offset: u32,
}

/// Create the GPU texture for a parsed [`CpuTexture`] and stage every
/// (mip, layer) image through the frame graph — the same shape the asset
/// loader's upload stage uses (#120).
fn create_texture_from_cpu(
    device: &Arc<redlilium_graphics::GraphicsDevice>,
    cpu: &CpuTexture,
    label: &str,
) -> Result<(Arc<Texture>, Vec<TransferOperation>), redlilium_graphics::GraphicsError> {
    let descriptor = TextureDescriptor::new_2d(
        cpu.width,
        cpu.height,
        cpu.format,
        TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
    )
    .with_mip_levels(cpu.mip_level_count)
    .with_label(label);
    let texture = device.create_texture(&descriptor)?;
    let mut ops = Vec::new();
    for mip in 0..cpu.mip_level_count {
        for layer in 0..cpu.layer_count() {
            ops.push(TransferOperation::upload_texture_level(
                device,
                Arc::clone(&texture),
                mip,
                layer,
                &cpu.data[cpu.byte_range(mip, layer)],
            )?);
        }
    }
    Ok((texture, ops))
}

/// Decode a PNG reference into a single-mip RGBA8 [`CpuTexture`].
fn decode_png(bytes: &[u8], srgb: bool, name: &str) -> CpuTexture {
    let img = image::load_from_memory(bytes).expect("fixture PNG decodes");
    let (width, height) = (img.width(), img.height());
    let format = if srgb {
        TextureFormat::Rgba8UnormSrgb
    } else {
        TextureFormat::Rgba8Unorm
    };
    CpuTexture::new(width, height, format, img.to_rgba8().into_raw()).with_name(name)
}

struct CompressedTextureDemo {
    material: Option<Arc<Material>>,
    mesh: Option<Arc<Mesh>>,
    quads: Vec<Quad>,
    uniform_ring: Option<RingBuffer>,
    depth_texture: Option<Arc<Texture>>,
    pending_uploads: Vec<TransferOperation>,
    resize_manager: ResizeManager,
}

impl CompressedTextureDemo {
    fn new() -> Self {
        Self {
            material: None,
            mesh: None,
            quads: Vec::new(),
            uniform_ring: None,
            depth_texture: None,
            pending_uploads: Vec::new(),
            resize_manager: ResizeManager::new((1280, 720), 50, ResizeStrategy::Stretch),
        }
    }

    fn create_gpu_resources(&mut self, ctx: &mut AppContext) {
        let device = ctx.device();

        // Top row compares BC7 against its PNG source and shows BC4; bottom
        // row lights the BC5 normal map next to the PNG-normal reference.
        struct Fixture {
            bytes: &'static [u8],
            is_ktx2: bool,
            /// sRGB decode for the PNG path (KTX2 carries its own format).
            srgb: bool,
            mode: u32,
            cell: (f32, f32),
            name: &'static str,
        }
        const fn fixture(
            bytes: &'static [u8],
            is_ktx2: bool,
            srgb: bool,
            mode: u32,
            cell: (f32, f32),
            name: &'static str,
        ) -> Fixture {
            Fixture {
                bytes,
                is_ktx2,
                srgb,
                mode,
                cell,
                name,
            }
        }
        let sources = [
            fixture(ALBEDO_BC7, true, true, 0, (-1.1, 0.55), "albedo_bc7"),
            fixture(ALBEDO_PNG, false, true, 0, (0.0, 0.55), "albedo_png"),
            fixture(GRAY_BC4, true, false, 1, (1.1, 0.55), "gray_bc4"),
            fixture(NORMAL_BC5, true, false, 2, (-0.55, -0.55), "normal_bc5"),
            fixture(NORMAL_PNG, false, false, 2, (0.55, -0.55), "normal_png"),
        ];

        let sampler = device
            .create_sampler(&SamplerDescriptor {
                label: Some("fixture_sampler".into()),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Linear,
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                ..Default::default()
            })
            .expect("create sampler");

        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_vertex_layout(VertexLayout::position_uv())
                    .with_color_format(ctx.surface_format())
                    .with_depth(Some(DepthState::new(TextureFormat::Depth32Float)))
                    .with_dynamic_uniform(0, 0)
                    .with_label("compressed_quad_material"),
            )
            .expect("create material");
        self.material = Some(material.clone());

        let uniform_ring = RingBuffer::new(
            device,
            1 << 16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "compressed_quad_mvp_ring",
        )
        .expect("create uniform ring");

        for Fixture {
            bytes,
            is_ktx2,
            srgb,
            mode,
            cell,
            name,
        } in sources
        {
            let cpu = if is_ktx2 {
                match parse_ktx2(bytes) {
                    Ok(cpu) => cpu.with_name(name),
                    Err(e) => {
                        log::error!("fixture {name}: {e}");
                        continue;
                    }
                }
            } else {
                decode_png(bytes, srgb, name)
            };

            // Capability gate (#119): a device without this family renders
            // the PNG references only. `create_texture` would reject the
            // format anyway — this just keeps the demo running.
            if let Some(family) = cpu.format.compression_family()
                && !device.capabilities().supports_compression_family(family)
            {
                log::warn!(
                    "fixture {name}: device lacks {family} texture compression — quad skipped"
                );
                continue;
            }

            let (texture, ops) =
                create_texture_from_cpu(device, &cpu, name).expect("create fixture texture");
            log::info!(
                "fixture {name}: {:?}, {} mips, {}x{}",
                cpu.format,
                cpu.mip_level_count,
                cpu.width,
                cpu.height
            );
            self.pending_uploads.extend(ops);

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
                        .with_texture(1, texture)
                        .with_sampler(2, sampler.clone()),
                )
                .expect("create binding group");
            self.quads.push(Quad {
                instance: Arc::new(
                    MaterialInstance::new(material.clone()).with_binding_group(binding_group),
                ),
                mode,
                cell,
                uniform_offset: 0,
            });
        }

        self.uniform_ring = Some(uniform_ring);

        let quad_cpu = generators::generate_quad(0.5, 0.5);
        let (mesh, mesh_ops) = device
            .create_mesh_deferred(&quad_cpu)
            .expect("create quad mesh");
        self.pending_uploads.extend(mesh_ops);
        self.mesh = Some(mesh);

        self.create_depth_texture(ctx);
    }

    fn create_depth_texture(&mut self, ctx: &AppContext) {
        self.depth_texture = Some(
            ctx.device()
                .create_texture(
                    &TextureDescriptor::new_2d(
                        ctx.width(),
                        ctx.height(),
                        TextureFormat::Depth32Float,
                        TextureUsage::RENDER_ATTACHMENT,
                    )
                    .with_label("depth_texture"),
                )
                .expect("create depth texture"),
        );
    }

    fn update_uniforms(&mut self, ctx: &AppContext) {
        let aspect = ctx.aspect_ratio();
        let scale = 1.4;
        let proj = if aspect > 1.0 {
            orthographic_rh_reversed(-scale * aspect, scale * aspect, -scale, scale, -1.0, 1.0)
        } else {
            orthographic_rh_reversed(-scale, scale, -scale / aspect, scale / aspect, -1.0, 1.0)
        };
        let view = look_at_rh(
            &Vec3::new(0.0, 0.0, 1.0),
            &Vec3::zeros(),
            &Vec3::new(0.0, 1.0, 0.0),
        );

        let Some(ring) = &mut self.uniform_ring else {
            return;
        };
        for quad in &mut self.quads {
            let model = Mat4::new_translation(&Vec3::new(quad.cell.0, quad.cell.1, 0.0));
            let uniforms = Uniforms {
                mvp: mat4_to_cols_array_2d(&(proj * view * model)),
                mode: quad.mode,
                _pad: [0; 3],
            };
            let bytes = bytemuck::bytes_of(&uniforms);
            let size = bytes.len() as u64;
            let alloc = ring.allocate(size).unwrap_or_else(|| {
                ring.reset();
                ring.allocate(size).expect("uniform ring too small")
            });
            ring.write(&alloc, bytes).expect("write uniform ring");
            quad.uniform_offset = alloc.offset as u32;
        }
    }
}

impl AppHandler for CompressedTextureDemo {
    fn on_init(&mut self, ctx: &mut AppContext) {
        log::info!("Initializing Compressed Texture Demo (#120)");
        self.resize_manager =
            ResizeManager::new((ctx.width(), ctx.height()), 50, ResizeStrategy::Stretch);
        self.create_gpu_resources(ctx);
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        self.resize_manager
            .on_resize_event(ctx.width(), ctx.height());
        if self.resize_manager.update().is_none() {
            self.resize_manager.force_resize();
        }
        self.create_depth_texture(ctx);
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        if self.resize_manager.update().is_some() {
            self.create_depth_texture(ctx);
        }
        self.update_uniforms(ctx);
        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        let mut graph = ctx.acquire_graph();

        if !self.pending_uploads.is_empty() {
            let ops = std::mem::take(&mut self.pending_uploads);
            let mut transfer_pass = TransferPass::new("fixture_uploads".into());
            transfer_pass.set_transfer_config(TransferConfig::new().with_operations(ops));
            graph.add_transfer_pass(transfer_pass);
            log::info!("fixture mips uploaded via transfer pass");
        }

        let mut render_pass = GraphicsPass::new("main".into());
        if let Some(depth) = &self.depth_texture {
            render_pass.set_render_targets(
                RenderTargetConfig::new()
                    .with_color(
                        ColorAttachment::from_surface(ctx.swapchain_texture())
                            .with_clear_color(0.1, 0.1, 0.15, 1.0),
                    )
                    .with_depth_stencil(
                        DepthStencilAttachment::from_texture(depth.clone())
                            .with_clear_depth(DepthConvention::default().clear_depth()),
                    ),
            );
        }

        if let Some(mesh) = &self.mesh {
            for quad in &self.quads {
                render_pass.add_draw_command(
                    DrawCommand::new(mesh.clone(), quad.instance.clone())
                        .with_dynamic_offsets(vec![vec![quad.uniform_offset]]),
                );
            }
        }

        graph.add_graphics_pass(render_pass);
        ctx.render(graph)
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down Compressed Texture Demo");
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args = DefaultAppArgs::parse().with_title_str("Compressed Texture Demo (#120)");
    App::run(CompressedTextureDemo::new(), args);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Entry point for wasm
}
