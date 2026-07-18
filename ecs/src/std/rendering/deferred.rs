//! Standard deferred PBR/IBL render pipeline (#144, ADR-035).
//!
//! [`DeferredPipeline`] is the engine's second built-in
//! [`CameraRenderPipeline`], registered under
//! [`DEFERRED_PIPELINE`](super::DEFERRED_PIPELINE). Per camera it records
//! three passes:
//!
//! 1. `gbuffer` — MRT (albedo, normal+metallic, position+roughness) + the
//!    camera's depth, drawing every visible `pbr`-model primitive via the
//!    shared [`SceneDrawer`];
//! 2. `skybox` — the environment cubemap as background, straight into the
//!    camera's color target;
//! 3. `deferred_resolve` — a fullscreen pass reading the G-buffer + IBL set,
//!    compositing lit geometry over the background (returned as the camera's
//!    main pass — the [`ScenePass`](super::ScenePass) contract).
//!
//! The G-buffer textures live in the camera's
//! [`PipelineTargets`](super::PipelineTargets), re-derived on resize by
//! [`ensure_targets`](CameraRenderPipeline::ensure_targets). Materials whose
//! shading model is not `pbr` (single-output shaders) fail MRT pipeline
//! specialization and are skipped with a warning — scenes migrate by
//! switching materials to the `pbr` model.
//!
//! The IBL set (irradiance/prefilter cubemaps, BRDF LUT, sky cubemap) is the
//! baked KTX2 pack from `std-assets/textures/ibl/`, embedded here as a
//! stopgap until IBL environments are first-class assets (#145). Light
//! directions are constants until lights come from ECS components (#146);
//! shadows arrive with #130.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use redlilium_core::math::Mat4;
use redlilium_core::profiling::profile_scope;
use redlilium_core::texture::ktx2::parse_ktx2;
use redlilium_graphics::{
    Buffer, ColorAttachment, CpuSampler, CpuTexture, DepthConvention, DepthStencilAttachment,
    DrawCommand, GraphicsDevice, GraphicsPass, LoadOp, MaterialDescriptor, MaterialInstance, Mesh,
    MeshDescriptor, PassHandle, RenderGraph, RenderTargetConfig, Sampler, ShaderSource,
    ShaderStage, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsage,
    TransferConfig, TransferOperation, TransferPass, VertexBufferLayout, VertexLayout,
};

use crate::{Entity, World};

use super::pipeline::{CameraRenderPipeline, CameraView, RecordCtx};
use super::scene_drawer::{DrawArgs, SceneDrawer};
use super::{
    CameraTarget, DEFERRED_PIPELINE, FrameRing, PipelineCache, PipelineTargets, TextureManager,
    shaders,
};

// The fullscreen passes' sources — the same bytes the shader bake registry
// reads (xtask/src/bake.rs), so the runtime hash finds the baked variants on
// Slang-less builds.
const RESOLVE_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/deferred_resolve.slang");
const SKYBOX_SHADER_SLANG: &str = include_str!("../../../../std-assets/shaders/skybox.slang");

// Baked IBL set (#137) — embedded stopgap until #145 loads it through the
// asset system.
const BRDF_LUT_KTX2: &[u8] = include_bytes!("../../../../std-assets/textures/ibl/brdf_lut.ktx2");
const IRRADIANCE_KTX2: &[u8] =
    include_bytes!("../../../../std-assets/textures/ibl/irradiance_cube.ktx2");
const PREFILTER_KTX2: &[u8] =
    include_bytes!("../../../../std-assets/textures/ibl/prefilter_cube.ktx2");
const SKY_KTX2: &[u8] = include_bytes!("../../../../std-assets/textures/ibl/sky_cube.ktx2");

/// [`PipelineTargets`] keys of the G-buffer textures.
pub const GBUFFER_ALBEDO: &str = "gbuffer_albedo";
pub const GBUFFER_NORMAL_METALLIC: &str = "gbuffer_normal_metallic";
pub const GBUFFER_POSITION_ROUGHNESS: &str = "gbuffer_position_roughness";

/// G-buffer attachment formats, in attachment order (must match
/// `deferred_gbuffer.slang`'s `FsOutput`).
const GBUFFER_FORMATS: [TextureFormat; 3] = [
    TextureFormat::Rgba8UnormSrgb,
    TextureFormat::Rgba16Float,
    TextureFormat::Rgba16Float,
];

/// Direction toward the key light — a constant until #146 sources lights
/// from ECS components (value matches the pbr_ibl demo's sun).
const SUN_DIR_TO_LIGHT: [f32; 3] = [0.75, 0.40, 0.75];

/// Uniforms of the resolve pass (must match `deferred_resolve.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveUniforms {
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
}

/// Uniforms of the skybox pass (must match `skybox.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyboxUniforms {
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    mip_level: f32,
    _pad: [f32; 3],
}

/// The engine's built-in deferred PBR/IBL path — see the module docs.
#[derive(Default)]
pub struct DeferredPipeline {
    drawer: SceneDrawer,
    /// Device-wide resources (IBL textures, samplers, fullscreen mesh),
    /// created on the first [`ensure_targets`](Self::ensure_targets) and
    /// shared by every camera.
    shared: Mutex<Option<SharedResources>>,
    /// Per-camera resolve/skybox materials, bound to that camera's G-buffer
    /// and color-target format; rebuilt when either changes.
    cameras: Mutex<HashMap<Entity, CameraResources>>,
}

/// Device-wide resources shared by every deferred camera.
struct SharedResources {
    irradiance_cubemap: Arc<Texture>,
    prefilter_cubemap: Arc<Texture>,
    brdf_lut: Arc<Texture>,
    sky_cubemap: Arc<Texture>,
    ibl_sampler: Arc<Sampler>,
    gbuffer_sampler: Arc<Sampler>,
    /// Fullscreen triangle (the shaders use `SV_VertexID` only; the buffer
    /// exists to satisfy the mesh contract).
    fullscreen_mesh: Arc<Mesh>,
    /// Staged one-time uploads (IBL mip levels + the dummy vertex buffer),
    /// drained into the first recorded frame's graph.
    pending_uploads: Vec<TransferOperation>,
}

/// One camera's resolve/skybox materials and the identities they were built
/// against.
struct CameraResources {
    resolve: Arc<MaterialInstance>,
    skybox: Arc<MaterialInstance>,
    /// `Arc::as_ptr` of the G-buffer albedo the binding groups reference —
    /// resize re-derives the G-buffer, invalidating these materials.
    albedo_ptr: usize,
    /// The color-target format the materials were specialized for (drives
    /// the `HDR_OUTPUT` variant and the pipelines' color state).
    color_format: TextureFormat,
}

impl SharedResources {
    fn create(device: &Arc<GraphicsDevice>, ring_buffer: &Arc<Buffer>) -> Self {
        // ring_buffer is unused here today, but taking it keeps the call site
        // honest about the dependency: camera materials bind it, and it must
        // be the same ring every frame.
        let _ = ring_buffer;
        profile_scope!("DeferredPipeline::shared_create");

        let mut pending_uploads = Vec::new();
        let brdf_lut = create_ibl_texture(
            device,
            BRDF_LUT_KTX2,
            TextureDimension::D2,
            "ibl_brdf_lut",
            &mut pending_uploads,
        );
        let irradiance_cubemap = create_ibl_texture(
            device,
            IRRADIANCE_KTX2,
            TextureDimension::Cube,
            "ibl_irradiance",
            &mut pending_uploads,
        );
        let prefilter_cubemap = create_ibl_texture(
            device,
            PREFILTER_KTX2,
            TextureDimension::Cube,
            "ibl_prefilter",
            &mut pending_uploads,
        );
        let sky_cubemap = create_ibl_texture(
            device,
            SKY_KTX2,
            TextureDimension::Cube,
            "ibl_sky",
            &mut pending_uploads,
        );

        let ibl_sampler = device
            .create_sampler_from_cpu(&CpuSampler::linear().with_name("ibl_sampler"))
            .expect("create IBL sampler");
        let gbuffer_sampler = device
            .create_sampler_from_cpu(&CpuSampler::nearest().with_name("gbuffer_sampler"))
            .expect("create G-buffer sampler");

        // Minimal fullscreen-triangle mesh (vertex data unused by the
        // shaders, uploaded once so the buffer is initialized).
        let layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(4))
                .with_label("deferred_fullscreen_layout"),
        );
        let fullscreen_mesh = device
            .create_mesh(
                &MeshDescriptor::new(layout)
                    .with_vertex_count(3)
                    .with_label("deferred_fullscreen"),
            )
            .expect("create fullscreen mesh");
        if let Some(vb) = fullscreen_mesh.vertex_buffer(0) {
            let dummy: [f32; 3] = [0.0; 3];
            pending_uploads.push(TransferOperation::write_buffer(
                vb.clone(),
                0,
                Arc::from(bytemuck::cast_slice(&dummy)),
            ));
        }

        log::info!("deferred pipeline: shared resources created (baked IBL set)");
        Self {
            irradiance_cubemap,
            prefilter_cubemap,
            brdf_lut,
            sky_cubemap,
            ibl_sampler,
            gbuffer_sampler,
            fullscreen_mesh,
            pending_uploads,
        }
    }
}

/// Parse one baked KTX2 blob, create its GPU texture, and stage every
/// (mip, layer) upload — the same path the asset loader uses (#120).
fn create_ibl_texture(
    device: &Arc<GraphicsDevice>,
    bytes: &[u8],
    dimension: TextureDimension,
    name: &str,
    ops: &mut Vec<TransferOperation>,
) -> Arc<Texture> {
    let cpu: CpuTexture = parse_ktx2(bytes)
        .unwrap_or_else(|e| panic!("baked IBL asset {name} failed to parse: {e}"))
        .with_name(name);
    assert_eq!(cpu.dimension, dimension, "baked IBL asset {name}");
    let usage = TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST;
    let descriptor = match cpu.dimension {
        TextureDimension::Cube => TextureDescriptor::new_cube(cpu.width, cpu.format, usage),
        _ => TextureDescriptor::new_2d(cpu.width, cpu.height, cpu.format, usage),
    }
    .with_mip_levels(cpu.mip_level_count)
    .with_label(name);
    let texture = device
        .create_texture(&descriptor)
        .unwrap_or_else(|e| panic!("create IBL texture {name}: {e}"));
    for mip in 0..cpu.mip_level_count {
        for layer in 0..cpu.layer_count() {
            ops.push(
                TransferOperation::upload_texture_level(
                    device,
                    Arc::clone(&texture),
                    mip,
                    layer,
                    &cpu.data[cpu.byte_range(mip, layer)],
                )
                .unwrap_or_else(|e| panic!("stage IBL upload {name}: {e}")),
            );
        }
    }
    texture
}

/// Whether a color-target format holds linear HDR (selects the shaders'
/// `HDR_OUTPUT` variant: raw linear out, 1.0 = SDR white).
fn hdr_active(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Rgba16Float | TextureFormat::Rgba32Float
    )
}

impl CameraResources {
    fn create(
        device: &Arc<GraphicsDevice>,
        shared: &SharedResources,
        gbuffer: [&Arc<Texture>; 3],
        color_format: TextureFormat,
        ring_buffer: &Arc<Buffer>,
    ) -> Option<Self> {
        profile_scope!("DeferredPipeline::camera_create");
        use redlilium_graphics::{BindingGroupDescriptor, ShaderVariantSpace};

        let hdr = hdr_active(color_format);

        // --- Resolve material ---
        let variant = ShaderVariantSpace::parse(RESOLVE_SHADER_SLANG)
            .ok()?
            .select()
            .system("HDR_OUTPUT", hdr)
            .build()
            .ok()?;
        let resolve_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        RESOLVE_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        RESOLVE_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_variant(variant)
                    .with_color_format(color_format)
                    .with_dynamic_uniform(0, 0)
                    .with_label("deferred_resolve"),
            )
            .inspect_err(|e| log::error!("deferred: resolve material failed: {e}"))
            .ok()?;

        let uniform_group = device
            .create_binding_group(
                resolve_material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new().with_buffer_range(
                    0,
                    ring_buffer.clone(),
                    0,
                    std::mem::size_of::<ResolveUniforms>() as u64,
                ),
            )
            .ok()?;
        let gbuffer_group = device
            .create_binding_group(
                resolve_material.binding_layouts()[1].clone(),
                BindingGroupDescriptor::new()
                    .with_texture(0, gbuffer[0].clone())
                    .with_texture(1, gbuffer[1].clone())
                    .with_texture(2, gbuffer[2].clone())
                    .with_sampler(3, shared.gbuffer_sampler.clone()),
            )
            .ok()?;
        let ibl_group = device
            .create_binding_group(
                resolve_material.binding_layouts()[2].clone(),
                BindingGroupDescriptor::new()
                    .with_texture(0, shared.irradiance_cubemap.clone())
                    .with_texture(1, shared.prefilter_cubemap.clone())
                    .with_texture(2, shared.brdf_lut.clone())
                    .with_sampler(3, shared.ibl_sampler.clone()),
            )
            .ok()?;
        let resolve = Arc::new(
            MaterialInstance::new(resolve_material)
                .with_binding_group(uniform_group)
                .with_binding_group(gbuffer_group)
                .with_binding_group(ibl_group),
        );

        // --- Skybox material ---
        let variant = ShaderVariantSpace::parse(SKYBOX_SHADER_SLANG)
            .ok()?
            .select()
            .system("HDR_OUTPUT", hdr)
            .build()
            .ok()?;
        let skybox_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        SKYBOX_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        SKYBOX_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_variant(variant)
                    .with_color_format(color_format)
                    .with_dynamic_uniform(0, 0)
                    .with_label("deferred_skybox"),
            )
            .inspect_err(|e| log::error!("deferred: skybox material failed: {e}"))
            .ok()?;
        let skybox_group = device
            .create_binding_group(
                skybox_material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        ring_buffer.clone(),
                        0,
                        std::mem::size_of::<SkyboxUniforms>() as u64,
                    )
                    .with_texture(1, shared.sky_cubemap.clone())
                    .with_sampler(2, shared.ibl_sampler.clone()),
            )
            .ok()?;
        let skybox =
            Arc::new(MaterialInstance::new(skybox_material).with_binding_group(skybox_group));

        Some(Self {
            resolve,
            skybox,
            albedo_ptr: Arc::as_ptr(gbuffer[0]) as usize,
            color_format,
        })
    }
}

impl CameraRenderPipeline for DeferredPipeline {
    fn name(&self) -> &str {
        DEFERRED_PIPELINE
    }

    fn ensure_targets(&self, world: &mut World, camera: Entity) {
        if !world.has_resource::<TextureManager>() || !world.has_resource::<FrameRing>() {
            return;
        }
        let Some(target) = world.get::<CameraTarget>(camera).cloned() else {
            return;
        };
        let device = world.resource::<TextureManager>().device().clone();
        let ring_buffer = world.resource::<FrameRing>().buffer().clone();
        let size = target.color.size();
        let (width, height) = (size.width, size.height);

        let mut shared_slot = self
            .shared
            .lock()
            .expect("deferred shared resources poisoned");
        let shared =
            shared_slot.get_or_insert_with(|| SharedResources::create(&device, &ring_buffer));

        // (Re-)derive the G-buffer at the camera target's size.
        let mut targets = world
            .get::<PipelineTargets>(camera)
            .cloned()
            .unwrap_or_default();
        let stale = targets
            .get(GBUFFER_ALBEDO)
            .map(|t| t.size().width != width || t.size().height != height)
            .unwrap_or(true);
        if stale {
            let names = [
                GBUFFER_ALBEDO,
                GBUFFER_NORMAL_METALLIC,
                GBUFFER_POSITION_ROUGHNESS,
            ];
            for (&name, format) in names.iter().zip(GBUFFER_FORMATS) {
                let texture = device
                    .create_texture(
                        &TextureDescriptor::new_2d(
                            width,
                            height,
                            format,
                            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                        )
                        .with_label(name),
                    )
                    .expect("create G-buffer texture");
                targets.set(name, texture);
            }
            let _ = world.insert(camera, targets.clone());
        }

        // (Re-)build the camera's materials when the G-buffer or the color
        // format changed.
        let (Some(albedo), Some(normal), Some(position)) = (
            targets.get(GBUFFER_ALBEDO),
            targets.get(GBUFFER_NORMAL_METALLIC),
            targets.get(GBUFFER_POSITION_ROUGHNESS),
        ) else {
            return;
        };
        let albedo_ptr = Arc::as_ptr(albedo) as usize;
        let color_format = target.color.format();
        let mut cameras = self.cameras.lock().expect("deferred camera map poisoned");
        let fresh = cameras
            .get(&camera)
            .is_some_and(|c| c.albedo_ptr == albedo_ptr && c.color_format == color_format);
        if !fresh {
            match CameraResources::create(
                &device,
                shared,
                [albedo, normal, position],
                color_format,
                &ring_buffer,
            ) {
                Some(resources) => {
                    cameras.insert(camera, resources);
                }
                None => {
                    cameras.remove(&camera);
                }
            }
        }
    }

    fn record(
        &self,
        ctx: &RecordCtx<'_>,
        view: &CameraView,
        graph: &mut RenderGraph,
    ) -> Option<PassHandle> {
        profile_scope!("DeferredPipeline::record");
        let world = ctx.world;

        let mut shared_slot = self
            .shared
            .lock()
            .expect("deferred shared resources poisoned");
        let shared = shared_slot.as_mut()?;
        let cameras = self.cameras.lock().expect("deferred camera map poisoned");
        let camera_resources = cameras.get(&view.entity)?;
        let targets = world.get::<PipelineTargets>(view.entity)?;
        let (Some(albedo), Some(normal), Some(position)) = (
            targets.get(GBUFFER_ALBEDO),
            targets.get(GBUFFER_NORMAL_METALLIC),
            targets.get(GBUFFER_POSITION_ROUGHNESS),
        ) else {
            return None;
        };

        // One-time uploads (IBL mips, fullscreen vertex buffer) ride the
        // first recorded frame's graph.
        if !shared.pending_uploads.is_empty() {
            let ops = std::mem::take(&mut shared.pending_uploads);
            let mut upload = TransferPass::new("deferred_ibl_upload".into());
            upload.set_transfer_config(TransferConfig::new().with_operations(ops));
            graph.add_transfer_pass(upload);
        }

        // Camera math: view-projection from the dispatcher; its inverse and
        // the camera position for the fullscreen passes.
        let view_proj = Mat4::from(view.view_projection);
        let inv_view_proj = view_proj.try_inverse()?;
        let camera_pos = world
            .get::<crate::std::components::Camera>(view.entity)
            .and_then(|cam| cam.view_matrix.try_inverse())
            .map(|m| [m[(0, 3)], m[(1, 3)], m[(2, 3)]])
            .unwrap_or([0.0; 3]);

        // Push this view's uniform slots into the frame ring.
        let (camera_offset, skybox_offset, resolve_offset, ring_buffer) = {
            let mut ring = world.resource_mut::<FrameRing>();
            let camera_offset = ring.push(bytemuck::bytes_of(&shaders::CameraUniforms {
                view_projection: view.view_projection,
            }));
            let skybox_offset = ring.push(bytemuck::bytes_of(&SkyboxUniforms {
                inv_view_proj: redlilium_core::math::mat4_to_cols_array_2d(&inv_view_proj),
                camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
                mip_level: 0.0,
                _pad: [0.0; 3],
            }));
            let resolve_offset = ring.push(bytemuck::bytes_of(&ResolveUniforms {
                camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
                light_dir: [
                    SUN_DIR_TO_LIGHT[0],
                    SUN_DIR_TO_LIGHT[1],
                    SUN_DIR_TO_LIGHT[2],
                    0.0,
                ],
            }));
            (
                camera_offset,
                skybox_offset,
                resolve_offset,
                ring.buffer().clone(),
            )
        };

        // --- 1. G-buffer pass (MRT + camera depth, reversed-Z ADR-038) ---
        let mut gbuffer_pass = GraphicsPass::new("gbuffer".into());
        gbuffer_pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(
                    // Alpha clears to 0 — the resolve's background marker.
                    ColorAttachment::from_texture(albedo.clone())
                        .with_clear_color(0.0, 0.0, 0.0, 0.0),
                )
                .with_color(
                    ColorAttachment::from_texture(normal.clone())
                        .with_clear_color(0.0, 0.0, 0.0, 0.0),
                )
                .with_color(
                    ColorAttachment::from_texture(position.clone())
                        .with_clear_color(0.0, 0.0, 0.0, 0.0),
                )
                .with_depth_stencil(
                    DepthStencilAttachment::from_texture(view.target.depth.clone())
                        .with_clear_depth(DepthConvention::default().clear_depth()),
                ),
        );
        {
            let mut pipelines = world.resource_mut::<PipelineCache>();
            self.drawer.record(
                &mut gbuffer_pass,
                ctx.scene,
                &ring_buffer,
                &mut pipelines,
                &DrawArgs::mrt(
                    camera_offset,
                    GBUFFER_FORMATS.to_vec(),
                    view.target.depth.format(),
                ),
            );
        }
        graph.add_graphics_pass(gbuffer_pass);

        // --- 2. Skybox background into the camera's color target ---
        let clear = view.target.clear_color;
        let mut skybox_pass = GraphicsPass::new("skybox".into());
        skybox_pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_texture(view.target.color.clone())
                    .with_clear_color(clear[0], clear[1], clear[2], clear[3]),
            ),
        );
        skybox_pass.add_draw_command(
            DrawCommand::new(
                shared.fullscreen_mesh.clone(),
                camera_resources.skybox.clone(),
            )
            .with_dynamic_offsets(vec![vec![skybox_offset]]),
        );
        let skybox_handle = graph.add_graphics_pass(skybox_pass);

        // --- 3. Resolve: lit geometry composited over the background ---
        let mut resolve_pass = GraphicsPass::new("deferred_resolve".into());
        resolve_pass.set_render_targets(RenderTargetConfig::new().with_color(
            ColorAttachment::from_texture(view.target.color.clone()).with_load_op(LoadOp::Load),
        ));
        resolve_pass.add_draw_command(
            DrawCommand::new(
                shared.fullscreen_mesh.clone(),
                camera_resources.resolve.clone(),
            )
            .with_dynamic_offsets(vec![vec![resolve_offset], vec![], vec![]]),
        );
        let resolve_handle = graph.add_graphics_pass(resolve_pass);

        // Skybox and resolve both write the color target with no read-based
        // direction — order them explicitly (background first).
        graph.add_dependency(resolve_handle, skybox_handle);

        Some(resolve_handle)
    }
}
