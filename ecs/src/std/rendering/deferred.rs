//! Standard deferred PBR/IBL render pipeline (#144, ADR-035).
//!
//! [`DeferredPipeline`] is the engine's second built-in
//! [`CameraRenderPipeline`], registered under
//! [`DEFERRED_PIPELINE`](super::DEFERRED_PIPELINE). Per camera it records
//! four passes:
//!
//! 1. `gbuffer` — MRT (albedo, normal+metallic, position+roughness) + the
//!    camera's depth, drawing every visible `pbr`-model primitive via the
//!    shared [`SceneDrawer`];
//! 2. `skybox` — the environment cubemap as background into the
//!    scene-referred [`SCENE_COLOR`] intermediate;
//! 3. `deferred_resolve` — a fullscreen pass reading the G-buffer + IBL set,
//!    compositing lit geometry over the background, still scene-referred
//!    linear;
//! 4. `display_output` (#142) — exposure ([`CameraExposure`](super::CameraExposure)),
//!    tonemap, and display encoding from [`SCENE_COLOR`] into the camera's
//!    color target (returned as the camera's main pass — the
//!    [`ScenePass`](super::ScenePass) contract). The scene/UI white-point
//!    contract on EDR surfaces is 1.0 = SDR white, shared with egui.
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
//! stopgap until IBL environments are first-class assets (#145). Direct
//! lights come from the ECS light components (#146) — see [`gather_lights`];
//! spot lights are not consumed yet. Shadows arrive with #130.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use redlilium_core::math::{Mat4, Vec3};
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
const DISPLAY_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/display_output.slang");

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
/// NDC-space motion current→previous frame, both unjittered (#147). Cleared
/// to zero; background pixels keep it (camera-only reprojection is the TAA
/// resolve's job, #148). `COPY_SRC` so tests can read the contract back.
pub const GBUFFER_VELOCITY: &str = "gbuffer_velocity";

/// [`PipelineTargets`] key of the scene-referred linear intermediate (#142):
/// skybox + resolve render here; the display-output pass reads it. Future
/// post effects (TAA, bloom) slot in between.
pub const SCENE_COLOR: &str = "scene_color";

/// Format of the [`SCENE_COLOR`] intermediate — linear f16 radiance,
/// 1.0 = SDR white.
const SCENE_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// G-buffer attachment formats, in attachment order (must match
/// `deferred_gbuffer.slang`'s `FsOutput`).
const GBUFFER_FORMATS: [TextureFormat; 4] = [
    TextureFormat::Rgba8UnormSrgb,
    TextureFormat::Rgba16Float,
    TextureFormat::Rgba16Float,
    TextureFormat::Rg16Float,
];

/// Uniform-array capacities of the resolve pass — must match the
/// `MAX_*_LIGHTS` constants in `deferred_resolve.slang`. Lights beyond the
/// capacity are dropped for the frame (warned once).
const MAX_DIRECTIONAL_LIGHTS: usize = 4;
const MAX_POINT_LIGHTS: usize = 16;

/// One directional light as the resolve shader consumes it.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDirectionalLight {
    /// xyz = normalized direction toward the light.
    dir_to_light: [f32; 4],
    /// rgb = linear color premultiplied by intensity.
    color: [f32; 4],
}

/// One point light as the resolve shader consumes it.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPointLight {
    /// xyz = world position, w = range (0 = unbounded).
    position_range: [f32; 4],
    /// rgb = linear color premultiplied by intensity.
    color: [f32; 4],
}

/// Uniforms of the resolve pass (must match `deferred_resolve.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveUniforms {
    camera_pos: [f32; 4],
    /// x = directional count, y = point count.
    light_counts: [u32; 4],
    dir_lights: [GpuDirectionalLight; MAX_DIRECTIONAL_LIGHTS],
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
}

/// Snapshot the world's visible light components into the resolve uniforms
/// (#146). Direction/position come from [`GlobalTransform`] (forward = the
/// direction the light travels); [`Visibility`] off drops the light.
fn gather_lights(world: &World) -> ResolveUniforms {
    use crate::std::components::{DirectionalLight, GlobalTransform, PointLight, Visibility};
    let mut uniforms = ResolveUniforms {
        camera_pos: [0.0; 4],
        light_counts: [0; 4],
        dir_lights: [GpuDirectionalLight::default(); MAX_DIRECTIONAL_LIGHTS],
        point_lights: [GpuPointLight::default(); MAX_POINT_LIGHTS],
    };
    let (Ok(globals), Ok(visibilities)) =
        (world.read::<GlobalTransform>(), world.read::<Visibility>())
    else {
        return uniforms;
    };
    let visible = |idx| {
        visibilities
            .get(idx)
            .is_none_or(|v: &Visibility| v.is_visible())
    };

    let mut dropped = 0usize;
    if let Ok(lights) = world.read::<DirectionalLight>() {
        for (idx, light) in lights.iter() {
            if !visible(idx) {
                continue;
            }
            let count = uniforms.light_counts[0] as usize;
            if count == MAX_DIRECTIONAL_LIGHTS {
                dropped += 1;
                continue;
            }
            let forward = globals
                .get(idx)
                .map(|g| g.forward())
                .unwrap_or_else(|| GlobalTransform::IDENTITY.forward());
            let to_light = -forward;
            uniforms.dir_lights[count] = GpuDirectionalLight {
                dir_to_light: [to_light.x, to_light.y, to_light.z, 0.0],
                color: [
                    light.color.x * light.intensity,
                    light.color.y * light.intensity,
                    light.color.z * light.intensity,
                    0.0,
                ],
            };
            uniforms.light_counts[0] += 1;
        }
    }
    if let Ok(lights) = world.read::<PointLight>() {
        for (idx, light) in lights.iter() {
            if !visible(idx) {
                continue;
            }
            let count = uniforms.light_counts[1] as usize;
            if count == MAX_POINT_LIGHTS {
                dropped += 1;
                continue;
            }
            let position = globals
                .get(idx)
                .map(|g| g.translation())
                .unwrap_or_else(Vec3::zeros);
            uniforms.point_lights[count] = GpuPointLight {
                position_range: [position.x, position.y, position.z, light.range],
                color: [
                    light.color.x * light.intensity,
                    light.color.y * light.intensity,
                    light.color.z * light.intensity,
                    0.0,
                ],
            };
            uniforms.light_counts[1] += 1;
        }
    }
    if dropped > 0 {
        static OVERFLOW_WARNED: std::sync::Once = std::sync::Once::new();
        OVERFLOW_WARNED.call_once(|| {
            log::warn!(
                "deferred: {dropped} light(s) beyond the uniform capacity \
                 ({MAX_DIRECTIONAL_LIGHTS} directional / {MAX_POINT_LIGHTS} point) are dropped"
            );
        });
    }
    uniforms
}

/// Uniforms of the display-output pass (must match `display_output.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DisplayOutputUniforms {
    /// x = linear exposure multiplier ([`CameraExposure`]), yzw unused.
    exposure: [f32; 4],
}

/// Uniforms of the skybox pass (must match `skybox.slang`). The WGSL uniform
/// layout aligns the shader's trailing `float3 _pad` to 16 bytes, rounding
/// the cbuffer to 112 — the extra `_pad1` matches that (a 96-byte buffer
/// fails wgpu's bind-size validation).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyboxUniforms {
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    mip_level: f32,
    _pad: [f32; 3],
    _pad1: [f32; 4],
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

/// One camera's fullscreen-pass materials and the identities they were built
/// against.
struct CameraResources {
    resolve: Arc<MaterialInstance>,
    skybox: Arc<MaterialInstance>,
    /// The display-output pass (#142): scene-referred `scene_color` →
    /// exposure → tonemap/encode into the camera's color target.
    display: Arc<MaterialInstance>,
    /// `Arc::as_ptr` of the G-buffer albedo the binding groups reference —
    /// resize re-derives the G-buffer (and `scene_color` with it),
    /// invalidating these materials.
    albedo_ptr: usize,
    /// The color-target format the display material was specialized for
    /// (drives the `HDR_OUTPUT` variant and its pipeline's color state).
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

/// Select the display-output shader's transform variant from the
/// color-target format (the egui renderer's scheme): linear-HDR targets get
/// raw linear (`HDR_OUTPUT`), sRGB-typed targets get tonemapped linear and
/// let the hardware encode (`SRGB_FRAMEBUFFER` — a manual encode would
/// double-gamma), anything else gets tonemap + manual gamma.
fn output_variant(source: &str, format: TextureFormat) -> Option<redlilium_graphics::VariantKey> {
    redlilium_graphics::ShaderVariantSpace::parse(source)
        .ok()?
        .select()
        .system("HDR_OUTPUT", format.is_hdr())
        .system("SRGB_FRAMEBUFFER", format.is_srgb())
        .build()
        .ok()
}

impl CameraResources {
    fn create(
        device: &Arc<GraphicsDevice>,
        shared: &SharedResources,
        gbuffer: [&Arc<Texture>; 3],
        scene_color: &Arc<Texture>,
        color_format: TextureFormat,
        ring_buffer: &Arc<Buffer>,
    ) -> Option<Self> {
        profile_scope!("DeferredPipeline::camera_create");
        use redlilium_graphics::BindingGroupDescriptor;

        // --- Resolve material (scene-referred: always targets scene_color) ---
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
                    .with_color_format(SCENE_COLOR_FORMAT)
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

        // --- Skybox material (scene-referred: always targets scene_color) ---
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
                    .with_color_format(SCENE_COLOR_FORMAT)
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

        // --- Display-output material (#142): the only format-variant pass ---
        let variant = output_variant(DISPLAY_SHADER_SLANG, color_format)?;
        let display_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        DISPLAY_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        DISPLAY_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_variant(variant)
                    .with_color_format(color_format)
                    .with_dynamic_uniform(0, 0)
                    .with_label("deferred_display_output"),
            )
            .inspect_err(|e| log::error!("deferred: display-output material failed: {e}"))
            .ok()?;
        let display_group = device
            .create_binding_group(
                display_material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        ring_buffer.clone(),
                        0,
                        std::mem::size_of::<DisplayOutputUniforms>() as u64,
                    )
                    .with_texture(1, scene_color.clone())
                    .with_sampler(2, shared.gbuffer_sampler.clone()),
            )
            .ok()?;
        let display =
            Arc::new(MaterialInstance::new(display_material).with_binding_group(display_group));

        Some(Self {
            resolve,
            skybox,
            display,
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

        // (Re-)derive the G-buffer + scene_color at the camera target's size.
        let mut targets = world
            .get::<PipelineTargets>(camera)
            .cloned()
            .unwrap_or_default();
        let stale = targets
            .get(GBUFFER_ALBEDO)
            .map(|t| t.size().width != width || t.size().height != height)
            .unwrap_or(true)
            || targets.get(SCENE_COLOR).is_none()
            || targets.get(GBUFFER_VELOCITY).is_none();
        if stale {
            let names = [
                GBUFFER_ALBEDO,
                GBUFFER_NORMAL_METALLIC,
                GBUFFER_POSITION_ROUGHNESS,
                GBUFFER_VELOCITY,
                SCENE_COLOR,
            ];
            let formats = [
                GBUFFER_FORMATS[0],
                GBUFFER_FORMATS[1],
                GBUFFER_FORMATS[2],
                GBUFFER_FORMATS[3],
                SCENE_COLOR_FORMAT,
            ];
            for (&name, format) in names.iter().zip(formats) {
                let mut usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING;
                if name == GBUFFER_VELOCITY {
                    // The temporal contract's observable — tests read it back.
                    usage |= TextureUsage::COPY_SRC;
                }
                let texture = device
                    .create_texture(
                        &TextureDescriptor::new_2d(width, height, format, usage).with_label(name),
                    )
                    .expect("create G-buffer texture");
                targets.set(name, texture);
            }
            let _ = world.insert(camera, targets.clone());
        }

        // (Re-)build the camera's materials when the targets or the color
        // format changed.
        let (Some(albedo), Some(normal), Some(position), Some(scene_color)) = (
            targets.get(GBUFFER_ALBEDO),
            targets.get(GBUFFER_NORMAL_METALLIC),
            targets.get(GBUFFER_POSITION_ROUGHNESS),
            targets.get(SCENE_COLOR),
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
                scene_color,
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
        let (Some(albedo), Some(normal), Some(position), Some(velocity), Some(scene_color)) = (
            targets.get(GBUFFER_ALBEDO),
            targets.get(GBUFFER_NORMAL_METALLIC),
            targets.get(GBUFFER_POSITION_ROUGHNESS),
            targets.get(GBUFFER_VELOCITY),
            targets.get(SCENE_COLOR),
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

        // This frame's direct lights, from the ECS light components (#146).
        let mut resolve_uniforms = gather_lights(world);
        resolve_uniforms.camera_pos = [camera_pos[0], camera_pos[1], camera_pos[2], 1.0];

        // Manual exposure (#142): the camera's CameraExposure, neutral when
        // absent.
        let exposure = world
            .get::<super::CameraExposure>(view.entity)
            .map_or(1.0, |e| e.exposure);

        // Push this view's uniform slots into the frame ring.
        let (camera_offset, skybox_offset, resolve_offset, display_offset, ring_buffer) = {
            let mut ring = world.resource_mut::<FrameRing>();
            let camera_offset = ring.push(bytemuck::bytes_of(&shaders::CameraUniforms {
                view_projection: view.view_projection,
                view_projection_unjittered: view.view_projection_unjittered,
                prev_view_projection: view.prev_view_projection,
            }));
            let skybox_offset = ring.push(bytemuck::bytes_of(&SkyboxUniforms {
                inv_view_proj: redlilium_core::math::mat4_to_cols_array_2d(&inv_view_proj),
                camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
                mip_level: 0.0,
                _pad: [0.0; 3],
                _pad1: [0.0; 4],
            }));
            let resolve_offset = ring.push(bytemuck::bytes_of(&resolve_uniforms));
            let display_offset = ring.push(bytemuck::bytes_of(&DisplayOutputUniforms {
                exposure: [exposure, 0.0, 0.0, 0.0],
            }));
            (
                camera_offset,
                skybox_offset,
                resolve_offset,
                display_offset,
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
                .with_color(
                    // Velocity clears to zero — background pixels stay still.
                    ColorAttachment::from_texture(velocity.clone())
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
                // Only pbr-model materials belong in the G-buffer: a
                // single-output shader would be accepted by the MRT pipeline
                // (writing only target 0) and land with garbage normals.
                &DrawArgs::mrt(
                    camera_offset,
                    GBUFFER_FORMATS.to_vec(),
                    view.target.depth.format(),
                )
                .with_shader_allowlist(vec![
                    redlilium_assets::Guid::stable("shaders/deferred_gbuffer.slang"),
                    redlilium_assets::Guid::stable("shaders/deferred_gbuffer_textured.slang"),
                ]),
            );
        }
        graph.add_graphics_pass(gbuffer_pass);

        // --- 2. Skybox background into the scene-referred intermediate ---
        let clear = view.target.clear_color;
        let mut skybox_pass = GraphicsPass::new("skybox".into());
        skybox_pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_texture(scene_color.clone())
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
            ColorAttachment::from_texture(scene_color.clone()).with_load_op(LoadOp::Load),
        ));
        resolve_pass.add_draw_command(
            DrawCommand::new(
                shared.fullscreen_mesh.clone(),
                camera_resources.resolve.clone(),
            )
            .with_dynamic_offsets(vec![vec![resolve_offset], vec![], vec![]]),
        );
        let resolve_handle = graph.add_graphics_pass(resolve_pass);

        // Skybox and resolve both write scene_color with no read-based
        // direction — order them explicitly (background first).
        graph.add_dependency(resolve_handle, skybox_handle);

        // --- 4. Display output (#142): scene-referred -> the camera target ---
        let mut display_pass = GraphicsPass::new("display_output".into());
        display_pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_texture(view.target.color.clone())
                    .with_clear_color(0.0, 0.0, 0.0, 1.0),
            ),
        );
        display_pass.add_draw_command(
            DrawCommand::new(
                shared.fullscreen_mesh.clone(),
                camera_resources.display.clone(),
            )
            .with_dynamic_offsets(vec![vec![display_offset]]),
        );
        let display_handle = graph.add_graphics_pass(display_pass);
        // scene_color has two writers above; anchor the read explicitly on
        // the last of them (the graph would otherwise derive an order against
        // *a* writer, not necessarily the resolve).
        graph.add_dependency(display_handle, resolve_handle);

        Some(display_handle)
    }
}
