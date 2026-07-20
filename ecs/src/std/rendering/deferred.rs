//! Standard deferred PBR/IBL render pipeline (#144, ADR-035).
//!
//! [`DeferredPipeline`] is the engine's second built-in
//! [`CameraRenderPipeline`], registered under
//! [`DEFERRED_PIPELINE`](super::DEFERRED_PIPELINE). Per camera it records:
//!
//! 1. `gbuffer` — MRT (albedo, normal+metallic, position+roughness,
//!    velocity #147) + the camera's depth, drawing every visible
//!    `pbr`-model primitive via the shared [`SceneDrawer`];
//! 2. `ssao` + `ssao_blur` (#150, only for cameras with a
//!    [`CameraAmbientOcclusion`](super::CameraAmbientOcclusion)) — a GTAO-lite
//!    horizon pass over the G-buffer and its bilateral denoise; the result
//!    multiplies the resolve's image-based ambient. Absent ⇒ the resolve
//!    binds a 1×1 white AO and the image is unchanged;
//! 3. `skybox` — the environment cubemap as background into the
//!    scene-referred [`SCENE_COLOR`] intermediate;
//! 4. `deferred_resolve` — a fullscreen pass reading the G-buffer + IBL set,
//!    compositing lit geometry over the background, still scene-referred
//!    linear;
//! 5. `taa_resolve` (#148, only for cameras with
//!    [`TemporalJitter`](super::TemporalJitter)) — accumulates the jittered
//!    frames into the [`TAA_HISTORY`] ping-pong (variance clipping in
//!    YCoCg; frame parity picks read/write);
//! 6. `bloom_down`/`bloom_up` (#151, only for cameras with a
//!    [`CameraBloom`](super::CameraBloom)) — the Jimenez dual-filter mip
//!    chain over `scene_color`; the display composites the accumulated glow.
//!    Absent ⇒ the display binds a 1×1 black bloom and the image is unchanged;
//! 7. `display_output` (#142) — exposure ([`CameraExposure`](super::CameraExposure)),
//!    tonemap, and display encoding from [`SCENE_COLOR`] (or the fresh TAA
//!    history) into the camera's color target (returned as the camera's
//!    main pass — the [`ScenePass`](super::ScenePass) contract). The
//!    scene/UI white-point contract on EDR surfaces is 1.0 = SDR white,
//!    shared with egui.
//!
//! The G-buffer textures live in the camera's
//! [`PipelineTargets`](super::PipelineTargets), re-derived on resize by
//! [`ensure_targets`](CameraRenderPipeline::ensure_targets). Materials whose
//! shading model is not `pbr` (single-output shaders) fail MRT pipeline
//! specialization and are skipped with a warning — scenes migrate by
//! switching materials to the `pbr` model.
//!
//! The IBL environment (irradiance/prefilter/sky cubemaps) is a per-camera
//! [`CameraEnvironment`](super::CameraEnvironment) asset (#145), resolved by
//! the [`EnvironmentManager`](super::EnvironmentManager); a camera without one
//! renders no skybox and zero image-based ambient (analytic lights only). The
//! BRDF integration LUT stays embedded — it is environment-independent. Direct
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
const TAA_SHADER_SLANG: &str = include_str!("../../../../std-assets/shaders/taa_resolve.slang");
const SSAO_SHADER_SLANG: &str = include_str!("../../../../std-assets/shaders/ssao.slang");
const SSAO_BLUR_SHADER_SLANG: &str = include_str!("../../../../std-assets/shaders/ssao_blur.slang");
const BLOOM_DOWN_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/bloom_down.slang");
const BLOOM_UP_SHADER_SLANG: &str = include_str!("../../../../std-assets/shaders/bloom_up.slang");

// The BRDF integration LUT (#137) stays embedded: it is environment-
// independent (the same table for every sky), so it is a device-wide resource
// rather than part of any environment asset. The sky/irradiance/prefilter
// cubemaps now come from a `CameraEnvironment` asset (#145).
const BRDF_LUT_KTX2: &[u8] = include_bytes!("../../../../std-assets/textures/ibl/brdf_lut.ktx2");

/// [`PipelineTargets`] keys of the G-buffer textures.
pub const GBUFFER_ALBEDO: &str = "gbuffer_albedo";
pub const GBUFFER_NORMAL_METALLIC: &str = "gbuffer_normal_metallic";
pub const GBUFFER_POSITION_ROUGHNESS: &str = "gbuffer_position_roughness";
/// NDC-space motion current→previous frame, both unjittered (#147). Cleared
/// to zero; background pixels keep it (camera-only reprojection is the TAA
/// resolve's job, #148). `COPY_SRC` so tests can read the contract back.
pub const GBUFFER_VELOCITY: &str = "gbuffer_velocity";

/// [`PipelineTargets`] keys of the SSAO targets (#150), present only for
/// cameras with a [`CameraAmbientOcclusion`](super::CameraAmbientOcclusion):
/// the raw horizon AO and its bilateral-denoised result (the one the resolve
/// reads). Both `R8Unorm`.
pub const SSAO_RAW: &str = "ssao_raw";
pub const SSAO_AO: &str = "ssao_ao";

/// Format of the SSAO targets — a single occlusion channel.
const SSAO_FORMAT: TextureFormat = TextureFormat::R8Unorm;

/// Max bloom mip levels (#151). Six half-steps from a half-res start reach a
/// few texels wide at 1080p — a wide, smooth glow without over-spending.
const MAX_BLOOM_MIPS: usize = 6;

/// [`PipelineTargets`] key of bloom mip `i` (#151): `bloom_0` is half the
/// camera resolution, each further mip half the previous. Present only for
/// cameras with a [`CameraBloom`](super::CameraBloom). `Rgba16Float`.
fn bloom_mip_key(i: usize) -> String {
    format!("bloom_{i}")
}

/// Number of bloom mips for a camera of `width`×`height`: half-res start,
/// halving until a dimension would drop below 2 texels, capped at
/// [`MAX_BLOOM_MIPS`].
fn bloom_mip_count(width: u32, height: u32) -> usize {
    let mut n = 0;
    let (mut w, mut h) = (width / 2, height / 2);
    while n < MAX_BLOOM_MIPS && w >= 2 && h >= 2 {
        n += 1;
        w /= 2;
        h /= 2;
    }
    n
}

/// [`PipelineTargets`] key of the scene-referred linear intermediate (#142):
/// skybox + resolve render here; the display-output pass reads it. Future
/// post effects (TAA, bloom) slot in between.
pub const SCENE_COLOR: &str = "scene_color";

/// [`PipelineTargets`] keys of the TAA accumulation ping-pong (#148):
/// frame parity selects which one is read (last frame's accumulation) and
/// which is written; the display pass then reads the written one. Only
/// cameras with [`TemporalJitter`](super::TemporalJitter) touch them.
pub const TAA_HISTORY: [&str; 2] = ["taa_history_a", "taa_history_b"];

/// Format of the [`SCENE_COLOR`] intermediate — linear f16 radiance,
/// 1.0 = SDR white.
const SCENE_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Fraction of the current frame in the TAA blend (history keeps the rest —
/// an exponential average remembering ~1/value frames).
const TAA_BLEND: f32 = 0.1;

/// Golden angle (radians) — the per-frame SSAO kernel rotation increment
/// (#150). Advancing the noise pattern by an irrational fraction of a turn
/// gives a temporal resolver a well-distributed sequence to average; applied
/// only when one is downstream (else the pattern is static — no flicker).
const GOLDEN_ANGLE: f32 = 2.399_963_2;

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
    /// x = linear exposure multiplier ([`CameraExposure`]); y = bloom
    /// intensity ([`CameraBloom`](super::CameraBloom), #151; 0 when the bloom
    /// input is the black fallback); zw unused.
    exposure: [f32; 4],
}

/// Uniforms of the TAA resolve pass (must match `taa_resolve.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaUniforms {
    /// Inverse of the current unjittered view-projection.
    inv_view_proj: [[f32; 4]; 4],
    /// Previous frame's unjittered view-projection.
    prev_view_proj: [[f32; 4]; 4],
    /// x = current-frame blend, y = history validity, zw = texel size.
    params: [f32; 4],
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

/// Uniforms of the SSAO pass (must match `ssao.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoUniforms {
    /// World → view (rigid: the upper 3×3 rotates normals).
    view: [[f32; 4]; 4],
    /// View → clip (for the depth-dependent screen-space search radius).
    proj: [[f32; 4]; 4],
    /// x = world radius, y = intensity, z = power, w = per-frame rotation
    /// (radians; 0 unless a temporal resolver is downstream, so a non-temporal
    /// camera's noise pattern is static — no flicker).
    params: [f32; 4],
    /// xy = texel size (1 / extent); zw unused.
    params2: [f32; 4],
}

/// Uniforms of the SSAO denoise pass (must match `ssao_blur.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoBlurUniforms {
    /// xy = texel size; z = world-space edge-stop sigma; w unused.
    params: [f32; 4],
}

/// Uniforms of the bloom down/up passes (must match `bloom_down.slang` /
/// `bloom_up.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomUniforms {
    /// xy = the *source* texel size (1 / source extent); zw unused.
    params: [f32; 4],
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
    /// Environment-independent BRDF integration LUT (embedded).
    brdf_lut: Arc<Texture>,
    /// 1×1 black cube bound for irradiance/prefilter/sky when a camera has no
    /// resolved environment (#145) — image-based lighting reads zero, so the
    /// scene is lit by analytic lights only.
    fallback_cube: Arc<Texture>,
    /// 1×1 white `R8` bound to the resolve's SSAO slot when a camera has no
    /// [`CameraAmbientOcclusion`](super::CameraAmbientOcclusion) (#150) — AO
    /// reads 1.0, so the ambient term is unchanged and the no-SSAO image is
    /// byte-identical.
    white_ao: Arc<Texture>,
    /// 1×1 black `Rgba16Float` bound to the display pass's bloom slot when a
    /// camera has no [`CameraBloom`](super::CameraBloom) (#151) — the composite
    /// adds zero, so the no-bloom image is byte-identical.
    black_bloom: Arc<Texture>,
    ibl_sampler: Arc<Sampler>,
    gbuffer_sampler: Arc<Sampler>,
    /// Fullscreen triangle (the shaders use `SV_VertexID` only; the buffer
    /// exists to satisfy the mesh contract).
    fullscreen_mesh: Arc<Mesh>,
    /// Staged one-time uploads (BRDF LUT, fallback cube, dummy vertex buffer),
    /// drained into the first recorded frame's graph.
    pending_uploads: Vec<TransferOperation>,
}

/// One camera's fullscreen-pass materials and the identities they were built
/// against.
struct CameraResources {
    resolve: Arc<MaterialInstance>,
    skybox: Arc<MaterialInstance>,
    /// The display-output pass (#142) reading `scene_color` directly — the
    /// TAA-off path.
    display_scene: Arc<MaterialInstance>,
    /// Display-output instances reading each TAA history texture — the
    /// TAA-on path reads the one the TAA pass just wrote.
    display_history: [Arc<MaterialInstance>; 2],
    /// TAA resolve instances (#148): instance `i` reads history `i` (and
    /// writes the other one — frame parity picks).
    taa: [Arc<MaterialInstance>; 2],
    /// SSAO passes (#150), present only when the camera has a
    /// [`CameraAmbientOcclusion`](super::CameraAmbientOcclusion): the horizon
    /// pass (writes `ssao_raw`) and its bilateral denoise (reads `ssao_raw`,
    /// writes `ssao_ao`). `None` ⇒ the resolve binds the shared white AO and
    /// no SSAO pass is recorded.
    ssao: Option<Arc<MaterialInstance>>,
    ssao_blur: Option<Arc<MaterialInstance>>,
    /// Whether SSAO is on — part of the fresh-check so toggling the component
    /// rebuilds the resolve (its AO binding) and the SSAO materials.
    ao_enabled: bool,
    /// Bloom passes (#151), present only when the camera has a
    /// [`CameraBloom`](super::CameraBloom). `bloom_down[i]` writes `bloom_i`
    /// (instance 0 Karis-samples `scene_color`, the rest sample the previous
    /// mip); `bloom_up[i]` additively upsamples `bloom_{i+1}` into `bloom_i`.
    /// Empty ⇒ the display binds the shared black bloom and no bloom pass runs.
    bloom_down: Vec<Arc<MaterialInstance>>,
    bloom_up: Vec<Arc<MaterialInstance>>,
    /// Whether bloom is on — part of the fresh-check (toggling rebuilds the
    /// display's bloom binding and the bloom materials).
    bloom_enabled: bool,
    /// Whether the TAA history holds a real frame yet — false right after
    /// (re)creation (resize/format change); the first TAA frame then takes
    /// the current frame wholesale instead of blending with garbage.
    history_primed: bool,
    /// `Arc::as_ptr` of the G-buffer albedo the binding groups reference —
    /// resize re-derives the G-buffer (and `scene_color` with it),
    /// invalidating these materials.
    albedo_ptr: usize,
    /// `Arc::as_ptr` of the resolved environment the IBL/skybox groups bind
    /// (`None` = the black fallback, #145). A changed environment (resolved,
    /// swapped, or hot-reloaded) rebuilds these materials. `is_some()` also
    /// gates whether the skybox pass is recorded (see `record`).
    env_ptr: Option<usize>,
    /// Highest prefilter mip index of the bound environment (`0` for the
    /// fallback) — fed to the resolve shader so roughness maps onto the
    /// actual baked chain instead of a hardcoded constant (#145).
    max_reflection_lod: f32,
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
        let fallback_cube = create_black_cube(device, &mut pending_uploads);
        let white_ao = create_white_ao(device, &mut pending_uploads);
        let black_bloom = create_black_bloom(device, &mut pending_uploads);

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

        log::info!("deferred pipeline: shared resources created (BRDF LUT + fallback cube)");
        Self {
            brdf_lut,
            fallback_cube,
            white_ao,
            black_bloom,
            ibl_sampler,
            gbuffer_sampler,
            fullscreen_mesh,
            pending_uploads,
        }
    }
}

/// Create the 1×1 black `Rgba16Float` cube used as the no-environment IBL
/// fallback (#145): sampling it returns zero, so image-based lighting drops
/// out and the scene is lit by analytic lights only.
fn create_black_cube(
    device: &Arc<GraphicsDevice>,
    ops: &mut Vec<TransferOperation>,
) -> Arc<Texture> {
    let usage = TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST;
    let texture = device
        .create_texture(
            &TextureDescriptor::new_cube(1, SCENE_COLOR_FORMAT, usage)
                .with_mip_levels(1)
                .with_label("ibl_fallback_black"),
        )
        .expect("create fallback cube");
    // One 1×1 Rgba16Float texel (4×f16) of zero per face.
    let black = [0u8; 8];
    for layer in 0..6 {
        ops.push(
            TransferOperation::upload_texture_level(device, Arc::clone(&texture), 0, layer, &black)
                .expect("stage fallback cube upload"),
        );
    }
    texture
}

/// Create the 1×1 white `R8` texture bound to the resolve's SSAO slot when a
/// camera has no [`CameraAmbientOcclusion`](super::CameraAmbientOcclusion)
/// (#150): sampling it returns 1.0, so ambient occlusion is a no-op.
fn create_white_ao(device: &Arc<GraphicsDevice>, ops: &mut Vec<TransferOperation>) -> Arc<Texture> {
    let usage = TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST;
    let texture = device
        .create_texture(
            &TextureDescriptor::new_2d(1, 1, SSAO_FORMAT, usage)
                .with_mip_levels(1)
                .with_label("ssao_white"),
        )
        .expect("create white AO");
    ops.push(
        TransferOperation::upload_texture_level(device, Arc::clone(&texture), 0, 0, &[255u8])
            .expect("stage white AO upload"),
    );
    texture
}

/// Create the 1×1 black `Rgba16Float` texture bound to the display pass's
/// bloom slot when a camera has no [`CameraBloom`](super::CameraBloom) (#151):
/// the composite adds zero.
fn create_black_bloom(
    device: &Arc<GraphicsDevice>,
    ops: &mut Vec<TransferOperation>,
) -> Arc<Texture> {
    let usage = TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST;
    let texture = device
        .create_texture(
            &TextureDescriptor::new_2d(1, 1, SCENE_COLOR_FORMAT, usage)
                .with_mip_levels(1)
                .with_label("bloom_black"),
        )
        .expect("create black bloom");
    // One 1×1 Rgba16Float texel (4×f16) of zero.
    let black = [0u8; 8];
    ops.push(
        TransferOperation::upload_texture_level(device, Arc::clone(&texture), 0, 0, &black)
            .expect("stage black bloom upload"),
    );
    texture
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
    #[allow(clippy::too_many_arguments)]
    fn create(
        device: &Arc<GraphicsDevice>,
        shared: &SharedResources,
        gbuffer: [&Arc<Texture>; 4],
        scene_color: &Arc<Texture>,
        history: [&Arc<Texture>; 2],
        env: Option<&Arc<super::ResolvedEnvironment>>,
        // The SSAO targets `[raw, denoised]` when the camera has a
        // `CameraAmbientOcclusion` (#150); `None` binds the shared white AO to
        // the resolve and records no SSAO pass.
        ssao_targets: Option<[&Arc<Texture>; 2]>,
        // The bloom mip chain (`bloom_0..N`) when the camera has a
        // `CameraBloom` (#151); `None`/empty binds the shared black bloom to
        // the display and records no bloom pass.
        bloom_targets: Option<&[&Arc<Texture>]>,
        color_format: TextureFormat,
        ring_buffer: &Arc<Buffer>,
    ) -> Option<Self> {
        profile_scope!("DeferredPipeline::camera_create");
        use redlilium_graphics::BindingGroupDescriptor;

        // The environment's cubemaps, or the shared black fallback (#145):
        // no environment ⇒ image-based lighting reads zero and the skybox
        // pass is skipped (see `record`).
        let (irradiance, prefilter, sky) = match env {
            Some(env) => (
                env.irradiance.texture.clone(),
                env.prefilter.texture.clone(),
                env.sky.texture.clone(),
            ),
            None => (
                shared.fallback_cube.clone(),
                shared.fallback_cube.clone(),
                shared.fallback_cube.clone(),
            ),
        };

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
                    .with_texture(0, irradiance.clone())
                    .with_texture(1, prefilter.clone())
                    .with_texture(2, shared.brdf_lut.clone())
                    .with_sampler(3, shared.ibl_sampler.clone()),
            )
            .ok()?;
        // SSAO group (#150): the denoised AO target, or the shared 1×1 white
        // (AO = 1) so this binding is present whether or not SSAO runs.
        let ao_texture = ssao_targets.map_or_else(|| shared.white_ao.clone(), |t| t[1].clone());
        let ao_group = device
            .create_binding_group(
                resolve_material.binding_layouts()[3].clone(),
                BindingGroupDescriptor::new()
                    .with_texture(0, ao_texture)
                    .with_sampler(1, shared.gbuffer_sampler.clone()),
            )
            .ok()?;
        let resolve = Arc::new(
            MaterialInstance::new(resolve_material)
                .with_binding_group(uniform_group)
                .with_binding_group(gbuffer_group)
                .with_binding_group(ibl_group)
                .with_binding_group(ao_group),
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
                    .with_texture(1, sky.clone())
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
        // Bloom the display composites: the accumulated top mip (#151), or the
        // shared black (adds zero) when the camera has no bloom. Sampled
        // linearly — the mip is lower-res than the scene.
        let bloom_top = bloom_targets
            .and_then(|m| m.first())
            .map_or_else(|| shared.black_bloom.clone(), |t| (*t).clone());
        // One instance per possible source: scene_color (TAA off) or either
        // history texture (TAA on — the one the TAA pass just wrote).
        let display_instance = |source: &Arc<Texture>| -> Option<Arc<MaterialInstance>> {
            let group = device
                .create_binding_group(
                    display_material.binding_layouts()[0].clone(),
                    BindingGroupDescriptor::new()
                        .with_buffer_range(
                            0,
                            ring_buffer.clone(),
                            0,
                            std::mem::size_of::<DisplayOutputUniforms>() as u64,
                        )
                        .with_texture(1, source.clone())
                        .with_sampler(2, shared.gbuffer_sampler.clone())
                        .with_texture(3, bloom_top.clone())
                        .with_sampler(4, shared.ibl_sampler.clone()),
                )
                .ok()?;
            Some(Arc::new(
                MaterialInstance::new(display_material.clone()).with_binding_group(group),
            ))
        };
        let display_scene = display_instance(scene_color)?;
        let display_history = [display_instance(history[0])?, display_instance(history[1])?];

        // --- TAA resolve material (#148): scene-referred in and out ---
        let taa_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        TAA_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        TAA_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_color_format(SCENE_COLOR_FORMAT)
                    .with_dynamic_uniform(0, 0)
                    .with_label("deferred_taa_resolve"),
            )
            .inspect_err(|e| log::error!("deferred: TAA material failed: {e}"))
            .ok()?;
        // Instance i reads history[i]; frame parity writes the other one.
        // History reprojection samples bilinearly (sub-pixel motion), the
        // rest nearest.
        let taa_instance = |read: &Arc<Texture>| -> Option<Arc<MaterialInstance>> {
            let group = device
                .create_binding_group(
                    taa_material.binding_layouts()[0].clone(),
                    BindingGroupDescriptor::new()
                        .with_buffer_range(
                            0,
                            ring_buffer.clone(),
                            0,
                            std::mem::size_of::<TaaUniforms>() as u64,
                        )
                        .with_texture(1, scene_color.clone())
                        .with_texture(2, read.clone())
                        .with_texture(3, gbuffer[3].clone())
                        .with_texture(4, gbuffer[0].clone())
                        .with_sampler(5, shared.gbuffer_sampler.clone())
                        .with_sampler(6, shared.ibl_sampler.clone())
                        // Camera-only velocity (adaptive blend) reprojects
                        // world position from the position G-buffer (RT2).
                        .with_texture(7, gbuffer[2].clone()),
                )
                .ok()?;
            Some(Arc::new(
                MaterialInstance::new(taa_material.clone()).with_binding_group(group),
            ))
        };
        let taa = [taa_instance(history[0])?, taa_instance(history[1])?];

        // --- SSAO passes (#150), only when the camera opted in ---
        // A tiny fullscreen material builder shared by the horizon and blur
        // passes (both target R8, dynamic uniform in group 0).
        let fullscreen_ssao_material = |source: &str, label: &str| {
            device
                .create_material(
                    &MaterialDescriptor::new()
                        .with_shader(ShaderSource::slang(
                            ShaderStage::Vertex,
                            source.as_bytes().to_vec(),
                            "vs_main",
                            vec![],
                        ))
                        .with_shader(ShaderSource::slang(
                            ShaderStage::Fragment,
                            source.as_bytes().to_vec(),
                            "fs_main",
                            vec![],
                        ))
                        .with_color_format(SSAO_FORMAT)
                        .with_dynamic_uniform(0, 0)
                        .with_label(label),
                )
                .inspect_err(|e| log::error!("deferred: {label} material failed: {e}"))
                .ok()
        };
        let (ssao, ssao_blur) = match ssao_targets {
            Some([raw, _ao]) => {
                let ssao_material = fullscreen_ssao_material(SSAO_SHADER_SLANG, "deferred_ssao")?;
                // Horizon pass: albedo (marker), normal, position + sampler.
                let ssao_group = device
                    .create_binding_group(
                        ssao_material.binding_layouts()[0].clone(),
                        BindingGroupDescriptor::new()
                            .with_buffer_range(
                                0,
                                ring_buffer.clone(),
                                0,
                                std::mem::size_of::<SsaoUniforms>() as u64,
                            )
                            .with_texture(1, gbuffer[0].clone())
                            .with_texture(2, gbuffer[1].clone())
                            .with_texture(3, gbuffer[2].clone())
                            .with_sampler(4, shared.gbuffer_sampler.clone()),
                    )
                    .ok()?;
                let ssao =
                    Arc::new(MaterialInstance::new(ssao_material).with_binding_group(ssao_group));

                let blur_material =
                    fullscreen_ssao_material(SSAO_BLUR_SHADER_SLANG, "deferred_ssao_blur")?;
                // Denoise pass: raw AO + position (edge stop) + sampler.
                let blur_group = device
                    .create_binding_group(
                        blur_material.binding_layouts()[0].clone(),
                        BindingGroupDescriptor::new()
                            .with_buffer_range(
                                0,
                                ring_buffer.clone(),
                                0,
                                std::mem::size_of::<SsaoBlurUniforms>() as u64,
                            )
                            .with_texture(1, raw.clone())
                            .with_texture(2, gbuffer[2].clone())
                            .with_sampler(3, shared.gbuffer_sampler.clone()),
                    )
                    .ok()?;
                let ssao_blur =
                    Arc::new(MaterialInstance::new(blur_material).with_binding_group(blur_group));
                (Some(ssao), Some(ssao_blur))
            }
            None => (None, None),
        };

        // --- Bloom passes (#151), only when the camera opted in ---
        let (bloom_down, bloom_up) = match bloom_targets {
            Some(mips) if !mips.is_empty() => {
                let n = mips.len();
                // Downsample material with the KARIS system variant selected.
                let down_material =
                    |karis: bool, label: &str| -> Option<Arc<redlilium_graphics::Material>> {
                        let variant =
                            redlilium_graphics::ShaderVariantSpace::parse(BLOOM_DOWN_SHADER_SLANG)
                                .ok()?
                                .select()
                                .system("KARIS", karis)
                                .build()
                                .ok()?;
                        device
                            .create_material(
                                &MaterialDescriptor::new()
                                    .with_shader(ShaderSource::slang(
                                        ShaderStage::Vertex,
                                        BLOOM_DOWN_SHADER_SLANG.as_bytes().to_vec(),
                                        "vs_main",
                                        vec![],
                                    ))
                                    .with_shader(ShaderSource::slang(
                                        ShaderStage::Fragment,
                                        BLOOM_DOWN_SHADER_SLANG.as_bytes().to_vec(),
                                        "fs_main",
                                        vec![],
                                    ))
                                    .with_variant(variant)
                                    .with_color_format(SCENE_COLOR_FORMAT)
                                    .with_dynamic_uniform(0, 0)
                                    .with_label(label),
                            )
                            .inspect_err(|e| log::error!("deferred: {label} material failed: {e}"))
                            .ok()
                    };
                let down_karis = down_material(true, "deferred_bloom_down_karis")?;
                let down_plain = down_material(false, "deferred_bloom_down")?;
                // Upsample material: OVER (alpha) blend + the shader's
                // alpha = BLOOM_MIX turns each step into a normalized
                // lerp(dst, tent, mix) accumulation (see bloom_up.slang).
                let up_material = device
                    .create_material(
                        &MaterialDescriptor::new()
                            .with_shader(ShaderSource::slang(
                                ShaderStage::Vertex,
                                BLOOM_UP_SHADER_SLANG.as_bytes().to_vec(),
                                "vs_main",
                                vec![],
                            ))
                            .with_shader(ShaderSource::slang(
                                ShaderStage::Fragment,
                                BLOOM_UP_SHADER_SLANG.as_bytes().to_vec(),
                                "fs_main",
                                vec![],
                            ))
                            .with_color_format(SCENE_COLOR_FORMAT)
                            .with_dynamic_uniform(0, 0)
                            .with_blend_state(redlilium_graphics::BlendState::alpha_blending())
                            .with_label("deferred_bloom_up"),
                    )
                    .inspect_err(|e| log::error!("deferred: bloom-up material failed: {e}"))
                    .ok()?;

                // One instance: dynamic uniform, one source texture, linear sampler.
                let bloom_instance = |material: &Arc<redlilium_graphics::Material>,
                                      source: &Arc<Texture>|
                 -> Option<Arc<MaterialInstance>> {
                    let group = device
                        .create_binding_group(
                            material.binding_layouts()[0].clone(),
                            BindingGroupDescriptor::new()
                                .with_buffer_range(
                                    0,
                                    ring_buffer.clone(),
                                    0,
                                    std::mem::size_of::<BloomUniforms>() as u64,
                                )
                                .with_texture(1, source.clone())
                                .with_sampler(2, shared.ibl_sampler.clone()),
                        )
                        .ok()?;
                    Some(Arc::new(
                        MaterialInstance::new(material.clone()).with_binding_group(group),
                    ))
                };

                // down[0] Karis-samples scene_color; down[i] samples mip i-1.
                let mut down = Vec::with_capacity(n);
                down.push(bloom_instance(&down_karis, scene_color)?);
                for i in 1..n {
                    down.push(bloom_instance(&down_plain, mips[i - 1])?);
                }
                // up[i] upsamples mip i+1 into mip i (additive), for i in 0..n-1.
                let mut up = Vec::with_capacity(n.saturating_sub(1));
                for i in 0..n.saturating_sub(1) {
                    up.push(bloom_instance(&up_material, mips[i + 1])?);
                }
                (down, up)
            }
            _ => (Vec::new(), Vec::new()),
        };

        Some(Self {
            resolve,
            skybox,
            display_scene,
            display_history,
            taa,
            ssao,
            ssao_blur,
            ao_enabled: ssao_targets.is_some(),
            bloom_down,
            bloom_up,
            bloom_enabled: bloom_targets.is_some_and(|m| !m.is_empty()),
            history_primed: false,
            albedo_ptr: Arc::as_ptr(gbuffer[0]) as usize,
            env_ptr: env.map(|e| Arc::as_ptr(e) as usize),
            max_reflection_lod: env.map_or(0.0, |e| e.max_reflection_lod),
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
                TAA_HISTORY[0],
                TAA_HISTORY[1],
            ];
            let formats = [
                GBUFFER_FORMATS[0],
                GBUFFER_FORMATS[1],
                GBUFFER_FORMATS[2],
                GBUFFER_FORMATS[3],
                SCENE_COLOR_FORMAT,
                SCENE_COLOR_FORMAT,
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

        // SSAO targets (#150): derived only for cameras with the component, at
        // the same size as the G-buffer (re-derived on resize). Left in place
        // when the component is absent — the resolve binds the shared white AO
        // regardless, so unused targets are harmless (memory is the only cost).
        let ao_enabled = world.get::<super::CameraAmbientOcclusion>(camera).is_some();
        if ao_enabled {
            let ao_stale = targets
                .get(SSAO_AO)
                .map(|t| t.size().width != width || t.size().height != height)
                .unwrap_or(true);
            if ao_stale {
                let usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING;
                for name in [SSAO_RAW, SSAO_AO] {
                    let texture = device
                        .create_texture(
                            &TextureDescriptor::new_2d(width, height, SSAO_FORMAT, usage)
                                .with_label(name),
                        )
                        .expect("create SSAO texture");
                    targets.set(name, texture);
                }
                let _ = world.insert(camera, targets.clone());
            }
        }

        // Bloom mip chain (#151): derived only for cameras with the component.
        // Rebuilt when the count changes or the top mip disagrees with a
        // half-res target (resize). Stale mips beyond the current count are
        // pruned so a shrink never leaves danglers behind.
        let bloom_enabled = world.get::<super::CameraBloom>(camera).is_some();
        let bloom_mips = if bloom_enabled {
            bloom_mip_count(width, height)
        } else {
            0
        };
        if bloom_enabled {
            let half = ((width >> 1).max(1), (height >> 1).max(1));
            let bloom_stale = bloom_mips == 0
                || targets
                    .get(&bloom_mip_key(0))
                    .map(|t| t.size().width != half.0 || t.size().height != half.1)
                    .unwrap_or(true)
                || targets.get(&bloom_mip_key(bloom_mips - 1)).is_none()
                || targets.get(&bloom_mip_key(bloom_mips)).is_some();
            if bloom_stale {
                for i in 0..MAX_BLOOM_MIPS {
                    targets.remove(&bloom_mip_key(i));
                }
                let usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING;
                for i in 0..bloom_mips {
                    let (w, h) = ((width >> (i + 1)).max(1), (height >> (i + 1)).max(1));
                    let texture = device
                        .create_texture(
                            &TextureDescriptor::new_2d(w, h, SCENE_COLOR_FORMAT, usage)
                                .with_label("bloom_mip"),
                        )
                        .expect("create bloom mip");
                    targets.set(bloom_mip_key(i), texture);
                }
                let _ = world.insert(camera, targets.clone());
            }
        }

        // (Re-)build the camera's materials when the targets or the color
        // format changed.
        let (
            Some(albedo),
            Some(normal),
            Some(position),
            Some(velocity),
            Some(scene_color),
            Some(history_a),
            Some(history_b),
        ) = (
            targets.get(GBUFFER_ALBEDO),
            targets.get(GBUFFER_NORMAL_METALLIC),
            targets.get(GBUFFER_POSITION_ROUGHNESS),
            targets.get(GBUFFER_VELOCITY),
            targets.get(SCENE_COLOR),
            targets.get(TAA_HISTORY[0]),
            targets.get(TAA_HISTORY[1]),
        )
        else {
            return;
        };
        let albedo_ptr = Arc::as_ptr(albedo) as usize;
        let color_format = target.color.format();
        // The camera's resolved IBL environment (#145), if it has one and it
        // has loaded — else the black fallback. Cloned out so the component
        // borrow ends before the camera-map lock.
        let env = world
            .get::<super::CameraEnvironment>(camera)
            .and_then(|c| c.environment.get().cloned());
        let env_ptr = env.as_ref().map(|e| Arc::as_ptr(e) as usize);
        // The SSAO input/output pair when the component is on and both targets
        // resolved (they were just ensured above) — else `None` (resolve binds
        // white AO, no SSAO pass). `ao_on` folds into the fresh-check so
        // toggling the component rebuilds the materials.
        let ssao_targets = match (ao_enabled, targets.get(SSAO_RAW), targets.get(SSAO_AO)) {
            (true, Some(raw), Some(ao)) => Some([raw, ao]),
            _ => None,
        };
        let ao_on = ssao_targets.is_some();
        // The bloom mip chain, gathered when the component is on and every mip
        // resolved (they were just ensured above). `bloom_on` folds into the
        // fresh-check so toggling rebuilds the display's bloom binding and the
        // bloom materials.
        let bloom_chain: Option<Vec<&Arc<Texture>>> = (bloom_mips > 0)
            .then(|| {
                (0..bloom_mips)
                    .map(|i| targets.get(&bloom_mip_key(i)))
                    .collect()
            })
            .flatten();
        let bloom_on = bloom_chain.is_some();
        let mut cameras = self.cameras.lock().expect("deferred camera map poisoned");
        let fresh = cameras.get(&camera).is_some_and(|c| {
            c.albedo_ptr == albedo_ptr
                && c.color_format == color_format
                && c.env_ptr == env_ptr
                && c.ao_enabled == ao_on
                && c.bloom_enabled == bloom_on
        });
        if !fresh {
            match CameraResources::create(
                &device,
                shared,
                [albedo, normal, position, velocity],
                scene_color,
                [history_a, history_b],
                env.as_ref(),
                ssao_targets,
                bloom_chain.as_deref(),
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
        let mut cameras = self.cameras.lock().expect("deferred camera map poisoned");
        let camera_resources = cameras.get_mut(&view.entity)?;
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
        // camera_pos.w carries the environment's reflection-LOD range (#145) —
        // the resolve shader reads only .xyz for position.
        let mut resolve_uniforms = gather_lights(world);
        resolve_uniforms.camera_pos = [
            camera_pos[0],
            camera_pos[1],
            camera_pos[2],
            camera_resources.max_reflection_lod,
        ];

        // Manual exposure (#142): the camera's CameraExposure, neutral when
        // absent.
        let exposure = world
            .get::<super::CameraExposure>(view.entity)
            .map_or(1.0, |e| e.exposure);

        // Bloom intensity (#151): the CameraBloom weight, but only when the
        // bloom materials/targets exist (else 0 — the display binds black).
        let bloom_intensity = if camera_resources.bloom_enabled {
            world
                .get::<super::CameraBloom>(view.entity)
                .map_or(0.0, |b| b.intensity)
        } else {
            0.0
        };

        // TAA (#148) runs for cameras that opted into the temporal contract;
        // frame parity picks the history ping-pong direction (read this
        // index, write the other).
        let taa_read_index = (world.get::<super::TemporalJitter>(view.entity).is_some()
            && world.has_resource::<super::TemporalState>())
        .then(|| (world.resource::<super::TemporalState>().frame() % 2) as usize);

        // SSAO uniforms (#150), when the camera opted in. Built before the
        // ring borrow (eager `.map`, so the world borrows end here). The
        // per-frame kernel rotation advances only when a temporal resolver
        // (TAA today) is downstream — else 0, so the noise is static.
        let ssao_uniforms = camera_resources
            .ao_enabled
            .then(|| {
                world
                    .get::<crate::std::components::Camera>(view.entity)
                    .zip(world.get::<super::CameraAmbientOcclusion>(view.entity))
                    .map(|(cam, ao)| {
                        let size = scene_color.size();
                        let texel = [
                            1.0 / size.width.max(1) as f32,
                            1.0 / size.height.max(1) as f32,
                        ];
                        let frame_rot = if taa_read_index.is_some() {
                            world.resource::<super::TemporalState>().frame() as f32 * GOLDEN_ANGLE
                        } else {
                            0.0
                        };
                        let ssao = SsaoUniforms {
                            view: redlilium_core::math::mat4_to_cols_array_2d(&cam.view_matrix),
                            proj: redlilium_core::math::mat4_to_cols_array_2d(
                                &cam.projection_matrix,
                            ),
                            params: [ao.radius, ao.intensity, ao.power, frame_rot],
                            params2: [texel[0], texel[1], 0.0, 0.0],
                        };
                        let blur = SsaoBlurUniforms {
                            // Edge-stop sigma = the sampling radius: neighbors
                            // within a radius blur together, silhouettes stop it.
                            params: [texel[0], texel[1], ao.radius, 0.0],
                        };
                        (ssao, blur)
                    })
            })
            .flatten();

        // Push this view's uniform slots into the frame ring.
        let (
            camera_offset,
            skybox_offset,
            resolve_offset,
            display_offset,
            taa_offset,
            ssao_offsets,
            bloom_down_offsets,
            bloom_up_offsets,
            ring_buffer,
        ) = {
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
                exposure: [exposure, bloom_intensity, 0.0, 0.0],
            }));
            let taa_offset = taa_read_index.map(|_| {
                // Reprojection math runs on the unjittered pair (the raster
                // inverse would bake the sub-pixel offset into every ray).
                let unjittered = Mat4::from(view.view_projection_unjittered);
                let inv_unjittered = unjittered.try_inverse().unwrap_or(unjittered);
                let size = scene_color.size();
                ring.push(bytemuck::bytes_of(&TaaUniforms {
                    inv_view_proj: redlilium_core::math::mat4_to_cols_array_2d(&inv_unjittered),
                    prev_view_proj: view.prev_view_projection,
                    params: [
                        TAA_BLEND,
                        if camera_resources.history_primed {
                            1.0
                        } else {
                            0.0
                        },
                        1.0 / size.width.max(1) as f32,
                        1.0 / size.height.max(1) as f32,
                    ],
                }))
            });
            let ssao_offsets = ssao_uniforms.map(|(ssao, blur)| {
                (
                    ring.push(bytemuck::bytes_of(&ssao)),
                    ring.push(bytemuck::bytes_of(&blur)),
                )
            });
            // Bloom uniforms (#151): one per down/up pass, carrying the SOURCE
            // texel size. Down pass i reads mip i-1 (i=0 reads scene, full
            // res), so its source is width>>i; up pass i reads mip i+1, source
            // width>>(i+2).
            let (bw, bh) = (scene_color.size().width, scene_color.size().height);
            let bloom_texel = |shift: u32| BloomUniforms {
                params: [
                    1.0 / (bw >> shift).max(1) as f32,
                    1.0 / (bh >> shift).max(1) as f32,
                    0.0,
                    0.0,
                ],
            };
            let n = camera_resources.bloom_down.len();
            let bloom_down_offsets: Vec<u32> = (0..n)
                .map(|i| ring.push(bytemuck::bytes_of(&bloom_texel(i as u32))))
                .collect();
            let bloom_up_offsets: Vec<u32> = (0..camera_resources.bloom_up.len())
                .map(|i| ring.push(bytemuck::bytes_of(&bloom_texel(i as u32 + 2))))
                .collect();
            (
                camera_offset,
                skybox_offset,
                resolve_offset,
                display_offset,
                taa_offset,
                ssao_offsets,
                bloom_down_offsets,
                bloom_up_offsets,
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
        let gbuffer_handle = graph.add_graphics_pass(gbuffer_pass);

        // --- 1b. SSAO (#150): horizon AO + bilateral denoise, when the camera
        // opted in. Reads the G-buffer, writes the denoised AO the resolve
        // multiplies into the ambient term. Independent of the skybox.
        let ssao_blur_handle = match (
            camera_resources.ssao.as_ref(),
            camera_resources.ssao_blur.as_ref(),
            ssao_offsets,
            targets.get(SSAO_RAW),
            targets.get(SSAO_AO),
        ) {
            (Some(ssao_mat), Some(blur_mat), Some((ssao_off, blur_off)), Some(raw), Some(ao)) => {
                let mut ssao_pass = GraphicsPass::new("ssao".into());
                ssao_pass.set_render_targets(RenderTargetConfig::new().with_color(
                    // Clear to 1 (open) — the pass overwrites every pixel.
                    ColorAttachment::from_texture(raw.clone()).with_clear_color(1.0, 1.0, 1.0, 1.0),
                ));
                ssao_pass.add_draw_command(
                    DrawCommand::new(shared.fullscreen_mesh.clone(), ssao_mat.clone())
                        .with_dynamic_offsets(vec![vec![ssao_off]]),
                );
                let ssao_handle = graph.add_graphics_pass(ssao_pass);
                graph.add_dependency(ssao_handle, gbuffer_handle);

                let mut blur_pass = GraphicsPass::new("ssao_blur".into());
                blur_pass.set_render_targets(RenderTargetConfig::new().with_color(
                    ColorAttachment::from_texture(ao.clone()).with_clear_color(1.0, 1.0, 1.0, 1.0),
                ));
                blur_pass.add_draw_command(
                    DrawCommand::new(shared.fullscreen_mesh.clone(), blur_mat.clone())
                        .with_dynamic_offsets(vec![vec![blur_off]]),
                );
                let blur_handle = graph.add_graphics_pass(blur_pass);
                graph.add_dependency(blur_handle, ssao_handle);
                Some(blur_handle)
            }
            _ => None,
        };

        // --- 2. Skybox background into the scene-referred intermediate ---
        // Only when the camera has a resolved environment (#145); with none,
        // the skybox is skipped and the resolve clears scene_color to the
        // camera clear color instead (fallback: analytic-lit, no sky).
        let clear = view.target.clear_color;
        let has_env = camera_resources.env_ptr.is_some();
        let skybox_handle = has_env.then(|| {
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
            graph.add_graphics_pass(skybox_pass)
        });

        // --- 3. Resolve: lit geometry composited over the background ---
        // Loads the skybox background when present, else clears scene_color.
        let background = match skybox_handle {
            Some(_) => {
                ColorAttachment::from_texture(scene_color.clone()).with_load_op(LoadOp::Load)
            }
            None => ColorAttachment::from_texture(scene_color.clone())
                .with_clear_color(clear[0], clear[1], clear[2], clear[3]),
        };
        let mut resolve_pass = GraphicsPass::new("deferred_resolve".into());
        resolve_pass.set_render_targets(RenderTargetConfig::new().with_color(background));
        resolve_pass.add_draw_command(
            DrawCommand::new(
                shared.fullscreen_mesh.clone(),
                camera_resources.resolve.clone(),
            )
            // Four groups now: uniforms (dynamic), G-buffer, IBL, SSAO — only
            // the first carries a dynamic offset; the rest are static.
            .with_dynamic_offsets(vec![vec![resolve_offset], vec![], vec![], vec![]]),
        );
        let resolve_handle = graph.add_graphics_pass(resolve_pass);

        // Skybox and resolve both write scene_color with no read-based
        // direction — order them explicitly (background first).
        if let Some(skybox_handle) = skybox_handle {
            graph.add_dependency(resolve_handle, skybox_handle);
        }
        // The resolve samples the denoised AO — order it after the blur (#150).
        if let Some(ssao_blur_handle) = ssao_blur_handle {
            graph.add_dependency(resolve_handle, ssao_blur_handle);
        }

        // --- 4. TAA resolve (#148), when the camera opted in: accumulate
        // scene_color into the history ping-pong; display then reads the
        // freshly written history instead of scene_color.
        let (display_source, taa_handle) = match (taa_read_index, taa_offset) {
            (Some(read), Some(taa_offset)) => {
                let write = 1 - read;
                let history_write = targets.get(TAA_HISTORY[write])?;
                let mut taa_pass = GraphicsPass::new("taa_resolve".into());
                taa_pass.set_render_targets(
                    RenderTargetConfig::new().with_color(
                        ColorAttachment::from_texture(history_write.clone())
                            .with_clear_color(0.0, 0.0, 0.0, 1.0),
                    ),
                );
                taa_pass.add_draw_command(
                    DrawCommand::new(
                        shared.fullscreen_mesh.clone(),
                        camera_resources.taa[read].clone(),
                    )
                    .with_dynamic_offsets(vec![vec![taa_offset]]),
                );
                let taa_handle = graph.add_graphics_pass(taa_pass);
                // TAA reads scene_color, which has two writers — anchor on
                // the last (the resolve), like the display pass does.
                graph.add_dependency(taa_handle, resolve_handle);
                camera_resources.history_primed = true;
                (
                    camera_resources.display_history[write].clone(),
                    Some(taa_handle),
                )
            }
            _ => (camera_resources.display_scene.clone(), None),
        };

        // --- 4b. Bloom (#151): down/up mip chain from scene_color, when the
        // camera opted in. Reads scene_color (post-resolve), independent of
        // TAA; the display composites the accumulated top mip (bloom_0).
        let bloom_final: Option<PassHandle> = if camera_resources.bloom_down.is_empty() {
            None
        } else {
            let n = camera_resources.bloom_down.len();
            let mips: Option<Vec<&Arc<Texture>>> =
                (0..n).map(|i| targets.get(&bloom_mip_key(i))).collect();
            mips.map(|mips| {
                // Downsample chain: down[i] writes mip i (replace).
                let mut down_handles = Vec::with_capacity(n);
                for i in 0..n {
                    let mut pass = GraphicsPass::new(format!("bloom_down_{i}"));
                    pass.set_render_targets(
                        RenderTargetConfig::new().with_color(
                            ColorAttachment::from_texture(mips[i].clone())
                                .with_clear_color(0.0, 0.0, 0.0, 1.0),
                        ),
                    );
                    pass.add_draw_command(
                        DrawCommand::new(
                            shared.fullscreen_mesh.clone(),
                            camera_resources.bloom_down[i].clone(),
                        )
                        .with_dynamic_offsets(vec![vec![bloom_down_offsets[i]]]),
                    );
                    let h = graph.add_graphics_pass(pass);
                    // down[0] reads scene_color (the resolve writes it); down[i]
                    // reads mip i-1.
                    graph.add_dependency(
                        h,
                        if i == 0 {
                            resolve_handle
                        } else {
                            down_handles[i - 1]
                        },
                    );
                    down_handles.push(h);
                }
                // Upsample chain (additive): up[i] adds mip i+1 into mip i,
                // walked from the smallest mip up to mip 0.
                let mut up_handles: Vec<Option<PassHandle>> = vec![None; n];
                for i in (0..camera_resources.bloom_up.len()).rev() {
                    let mut pass = GraphicsPass::new(format!("bloom_up_{i}"));
                    pass.set_render_targets(RenderTargetConfig::new().with_color(
                        // Additive blend onto mip i's own downsample — keep it.
                        ColorAttachment::from_texture(mips[i].clone()).with_load_op(LoadOp::Load),
                    ));
                    pass.add_draw_command(
                        DrawCommand::new(
                            shared.fullscreen_mesh.clone(),
                            camera_resources.bloom_up[i].clone(),
                        )
                        .with_dynamic_offsets(vec![vec![bloom_up_offsets[i]]]),
                    );
                    let h = graph.add_graphics_pass(pass);
                    // Reads mip i+1 (last writer: up[i+1] if it ran, else
                    // down[i+1]); writes mip i after its own downsample.
                    let src_writer = up_handles[i + 1].unwrap_or(down_handles[i + 1]);
                    graph.add_dependency(h, src_writer);
                    graph.add_dependency(h, down_handles[i]);
                    up_handles[i] = Some(h);
                }
                // Accumulated bloom lives in mip 0 (up[0] if it ran, else down[0]).
                up_handles[0].unwrap_or(down_handles[0])
            })
        };

        // --- 5. Display output (#142): scene-referred -> the camera target ---
        let mut display_pass = GraphicsPass::new("display_output".into());
        display_pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_texture(view.target.color.clone())
                    .with_clear_color(0.0, 0.0, 0.0, 1.0),
            ),
        );
        display_pass.add_draw_command(
            DrawCommand::new(shared.fullscreen_mesh.clone(), display_source)
                .with_dynamic_offsets(vec![vec![display_offset]]),
        );
        let display_handle = graph.add_graphics_pass(display_pass);
        // scene_color has two writers above; anchor the read explicitly on
        // the last of them (the graph would otherwise derive an order against
        // *a* writer, not necessarily the resolve). With TAA the display
        // reads the history the TAA pass just wrote instead.
        graph.add_dependency(display_handle, taa_handle.unwrap_or(resolve_handle));
        // The display composites the accumulated bloom — order it after the
        // chain's final write (#151).
        if let Some(bloom_final) = bloom_final {
            graph.add_dependency(display_handle, bloom_final);
        }

        Some(display_handle)
    }
}
