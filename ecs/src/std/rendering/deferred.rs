//! Standard deferred PBR/IBL render pipeline (#144, ADR-035).
//!
//! [`DeferredPipeline`] is the engine's second built-in
//! [`CameraRenderPipeline`], registered under
//! [`DEFERRED_PIPELINE`](super::DEFERRED_PIPELINE). Per camera it records:
//!
//! 1. `gbuffer` — MRT (albedo, normal+metallic, roughness, velocity #147) +
//!    the camera's depth, drawing every visible `pbr`-model primitive via the
//!    shared [`SceneDrawer`]; world position is reconstructed from the depth
//!    buffer by every consumer (full 32-bit precision — no f16 position RT);
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
//! 7. `histogram_build`/`histogram_resolve` (#153, only for cameras with a
//!    [`CameraAutoExposure`](super::CameraAutoExposure)) — two compute passes
//!    that build a luminance histogram of `scene_color` and meter an adapting
//!    exposure the display multiplies in. Absent ⇒ the display binds a neutral
//!    1.0 exposure buffer and the manual [`CameraExposure`] stands alone;
//! 8. `display_output` (#142) — exposure ([`CameraExposure`](super::CameraExposure)),
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
    Buffer, BufferDescriptor, BufferUsage, ColorAttachment, ComputePass, CpuSampler, CpuTexture,
    DepthConvention, DepthStencilAttachment, DrawCommand, GraphicsDevice, GraphicsPass, LoadOp,
    MaterialDescriptor, MaterialInstance, Mesh, MeshDescriptor, PassHandle, RenderGraph,
    RenderTargetConfig, Sampler, ShaderSource, ShaderStage, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsage, TransferConfig, TransferOperation, TransferPass,
    VertexBufferLayout, VertexLayout,
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
const VELOCITY_COMPLETE_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/velocity_complete.slang");
const MB_TILE_MAX_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/mb_tile_max.slang");
const MB_NEIGHBOR_MAX_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/mb_neighbor_max.slang");
const MB_RECONSTRUCT_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/mb_reconstruct.slang");
const SSAO_SHADER_SLANG: &str = include_str!("../../../../std-assets/shaders/ssao.slang");
const SSAO_BLUR_SHADER_SLANG: &str = include_str!("../../../../std-assets/shaders/ssao_blur.slang");
const BLOOM_DOWN_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/bloom_down.slang");
const BLOOM_UP_SHADER_SLANG: &str = include_str!("../../../../std-assets/shaders/bloom_up.slang");
const HISTOGRAM_BUILD_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/histogram_build.slang");
const HISTOGRAM_RESOLVE_SHADER_SLANG: &str =
    include_str!("../../../../std-assets/shaders/histogram_resolve.slang");

// The BRDF integration LUT (#137) stays embedded: it is environment-
// independent (the same table for every sky), so it is a device-wide resource
// rather than part of any environment asset. The sky/irradiance/prefilter
// cubemaps now come from a `CameraEnvironment` asset (#145).
const BRDF_LUT_KTX2: &[u8] = include_bytes!("../../../../std-assets/textures/ibl/brdf_lut.ktx2");

/// [`PipelineTargets`] keys of the G-buffer textures.
pub const GBUFFER_ALBEDO: &str = "gbuffer_albedo";
pub const GBUFFER_NORMAL_METALLIC: &str = "gbuffer_normal_metallic";
/// Single-channel roughness. World position is NOT stored in the G-buffer —
/// consumers reconstruct it from the camera depth buffer (full 32-bit
/// precision everywhere; the old f16 position target lost ~0.5 units of
/// precision at |coord| ~ 1000).
pub const GBUFFER_ROUGHNESS: &str = "gbuffer_roughness";
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

/// Auto-exposure metering range (#153), log2 luminance. 24 stops spans a night
/// scene to bright daylight; the histogram's bin 0 collects everything below
/// `2^AE_LOG_MIN` (so a dark background does not pull exposure down).
const AE_LOG_MIN: f32 = -8.0;
const AE_LOG_MAX: f32 = 16.0;
/// Side of the square grid the histogram build samples the scene over — the
/// metering work is resolution-independent (512×512 = 256k samples). A
/// multiple of the 16×16 workgroup so the dispatch tiles exactly.
const AE_SAMPLE_DIM: u32 = 512;
/// Histogram bin count — must match `histogram_build`/`histogram_resolve`.
const AE_BINS: usize = 256;
/// Middle-grey key the metered average luminance targets (#153).
const AE_KEY: f32 = 0.18;

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

/// Motion-blur tile size K in pixels (#149) — also the max blur radius the
/// reconstruction can gather. Velocity is downsampled into ⌈W/K⌉×⌈H/K⌉ tiles.
const MB_TILE_SIZE: u32 = 20;

/// [`PipelineTargets`] keys of the motion-blur intermediates (#149). Only
/// cameras with [`MotionBlur`](super::MotionBlur) allocate them: two
/// tile-resolution velocity reductions and the full-res blurred output that
/// bloom and the display pass then read.
pub const MB_TILE_MAX: &str = "mb_tile_max";
pub const MB_NEIGHBOR_MAX: &str = "mb_neighbor_max";
pub const MB_OUTPUT: &str = "mb_output";

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
    TextureFormat::R8Unorm,
    TextureFormat::Rg16Float,
];

/// Uniform-array capacities of the resolve pass — must match the
/// `MAX_*_LIGHTS` constants in `deferred_resolve.slang`. Lights beyond the
/// capacity are dropped for the frame (warned once).
const MAX_DIRECTIONAL_LIGHTS: usize = 4;
const MAX_POINT_LIGHTS: usize = 16;
const MAX_SPOT_LIGHTS: usize = 8;

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

/// One spot light as the resolve shader consumes it (#146; KHR-style cone).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSpotLight {
    /// xyz = world position, w = range (0 = unbounded).
    position_range: [f32; 4],
    /// xyz = normalized direction the light travels, w = cos(outer cone).
    direction_cos: [f32; 4],
    /// rgb = linear color premultiplied by intensity, a = cos(inner cone).
    color_cos: [f32; 4],
}

/// Uniforms of the resolve pass (must match `deferred_resolve.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveUniforms {
    /// Inverse of the view-projection the G-buffer was rasterized with (the
    /// jittered one under TAA) — reconstructs world position from depth.
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    /// x = directional count, y = point count, z = spot count.
    light_counts: [u32; 4],
    dir_lights: [GpuDirectionalLight; MAX_DIRECTIONAL_LIGHTS],
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
    spot_lights: [GpuSpotLight; MAX_SPOT_LIGHTS],
}

/// Snapshot the world's visible light components into the resolve uniforms
/// (#146). Direction/position come from [`GlobalTransform`] (forward = the
/// direction the light travels); [`Visibility`] off drops the light.
fn gather_lights(world: &World) -> ResolveUniforms {
    use crate::std::components::{
        DirectionalLight, GlobalTransform, PointLight, SpotLight, Visibility,
    };
    let mut uniforms = ResolveUniforms {
        inv_view_proj: [[0.0; 4]; 4],
        camera_pos: [0.0; 4],
        light_counts: [0; 4],
        dir_lights: [GpuDirectionalLight::default(); MAX_DIRECTIONAL_LIGHTS],
        point_lights: [GpuPointLight::default(); MAX_POINT_LIGHTS],
        spot_lights: [GpuSpotLight::default(); MAX_SPOT_LIGHTS],
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
    if let Ok(lights) = world.read::<SpotLight>() {
        for (idx, light) in lights.iter() {
            if !visible(idx) {
                continue;
            }
            let count = uniforms.light_counts[2] as usize;
            if count == MAX_SPOT_LIGHTS {
                dropped += 1;
                continue;
            }
            let (position, forward) = globals.get(idx).map_or_else(
                || (Vec3::zeros(), GlobalTransform::IDENTITY.forward()),
                |g| (g.translation(), g.forward()),
            );
            // Inner must not exceed outer, and the cosines must differ so the
            // shader's ramp denominator stays finite.
            let outer = light.outer_cone_angle.max(1e-3);
            let inner = light.inner_cone_angle.clamp(0.0, outer);
            uniforms.spot_lights[count] = GpuSpotLight {
                position_range: [position.x, position.y, position.z, light.range],
                direction_cos: [forward.x, forward.y, forward.z, outer.cos()],
                color_cos: [
                    light.color.x * light.intensity,
                    light.color.y * light.intensity,
                    light.color.z * light.intensity,
                    inner.cos(),
                ],
            };
            uniforms.light_counts[2] += 1;
        }
    }
    if dropped > 0 {
        static OVERFLOW_WARNED: std::sync::Once = std::sync::Once::new();
        OVERFLOW_WARNED.call_once(|| {
            log::warn!(
                "deferred: {dropped} light(s) beyond the uniform capacity \
                 ({MAX_DIRECTIONAL_LIGHTS} directional / {MAX_POINT_LIGHTS} point / \
                 {MAX_SPOT_LIGHTS} spot) are dropped"
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
    /// input is the black fallback); z = display headroom H (#154, only the
    /// `HDR_OUTPUT` variant reads it); w unused.
    exposure: [f32; 4],
}

/// Display headroom `H` (#154): the output surface's peak luminance over
/// paper-white, in ×SDR-white units (`H ≥ 1`). Highlights above paper-white
/// (1.0) roll off up to `H` on a linear-HDR target; at `H = 1` the display
/// output is byte-identical to the SDR path.
///
/// A frame resource the host sets from the real display (macOS
/// `NSScreen.maximumExtendedDynamicRangeColorComponentValue`, later swapchain
/// HDR metadata). **Absent ⇒ `H = 1`** — the safe SDR default every
/// headless/test path takes, so goldens never depend on a display query. The
/// deferred pass folds it into `DisplayOutputUniforms::exposure.z`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayHeadroom(pub f32);

impl DisplayHeadroom {
    /// The effective headroom, floored at 1.0 (a display never has less range
    /// than SDR, and the curve is undefined below 1).
    pub fn get(self) -> f32 {
        self.0.max(1.0)
    }
}

impl Default for DisplayHeadroom {
    fn default() -> Self {
        // SDR: paper-white is the peak, no extended range.
        Self(1.0)
    }
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
    /// Camera world position (xyz) for the linear-depth disocclusion proxy
    /// stored in the history alpha (#148 depth reject).
    camera_pos: [f32; 4],
}

/// Uniforms of the velocity-completion pass (must match
/// `velocity_complete.slang`) — the unjittered camera pair the background
/// reprojection runs on.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VelocityCompleteUniforms {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
}

/// Uniforms of the motion-blur TileMax pass (must match `mb_tile_max.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MbTileUniforms {
    inv_resolution: [f32; 2],
    resolution: [f32; 2],
    shutter: f32,
    tile_size: f32,
    _pad: [f32; 2],
}

/// Uniforms of the motion-blur NeighborMax pass (must match
/// `mb_neighbor_max.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MbNeighborUniforms {
    inv_tile_resolution: [f32; 2],
    _pad: [f32; 2],
}

/// Uniforms of the motion-blur reconstruction pass (must match
/// `mb_reconstruct.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MbReconstructUniforms {
    /// Inverse of the unjittered view-projection — the depth proxy's world
    /// position is reconstructed from the depth buffer.
    inv_view_proj: [[f32; 4]; 4],
    inv_resolution: [f32; 2],
    resolution: [f32; 2],
    camera_pos: [f32; 4],
    shutter: f32,
    tile_size: f32,
    samples: f32,
    /// Padding to a 16-byte boundary (cbuffer alignment).
    _pad: f32,
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
    /// Inverse of the rasterizing view-projection (jittered under TAA) —
    /// world position is reconstructed from the depth buffer.
    inv_view_proj: [[f32; 4]; 4],
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
    /// Inverse of the rasterizing view-projection (jittered under TAA) —
    /// edge-stop positions are reconstructed from the depth buffer.
    inv_view_proj: [[f32; 4]; 4],
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

/// Uniforms of the auto-exposure histogram build (must match
/// `histogram_build.slang`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AeBuildUniforms {
    /// x = [`AE_LOG_MIN`], y = [`AE_LOG_MAX`], z = [`AE_SAMPLE_DIM`], w unused.
    params: [f32; 4],
}

/// Uniforms of the auto-exposure histogram resolve (must match
/// `histogram_resolve.slang`).
///
/// Rate-slot semantics: the shader applies `params.z` when the metered
/// multiplier RISES (the image brightens because the scene darkened — the
/// eye's slow dark adaptation, [`CameraAutoExposure::speed_down`]) and
/// `limits.z` when it FALLS (the scene brightened — the fast stop-down,
/// [`CameraAutoExposure::speed_up`]). Getting this pairing backwards inverts
/// the eye-like asymmetry.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AeResolveUniforms {
    /// x = [`AE_LOG_MIN`], y = [`AE_LOG_MAX`], z = rate when the multiplier
    /// rises (scene darkened — dark adaptation), w = [`AE_KEY`].
    params: [f32; 4],
    /// x = min multiplier (`2^min_ev`), y = max (`2^max_ev`), z = rate when
    /// the multiplier falls (scene brightened — stop-down), w unused.
    limits: [f32; 4],
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
    /// Single-`f32` storage buffer holding `1.0`, bound to the display pass's
    /// exposure slot when a camera has no
    /// [`CameraAutoExposure`](super::CameraAutoExposure) (#153) — the multiply
    /// is a no-op, so the manual exposure stands alone.
    neutral_exposure: Arc<Buffer>,
    ibl_sampler: Arc<Sampler>,
    gbuffer_sampler: Arc<Sampler>,
    /// Fullscreen triangle (the shaders use `SV_VertexID` only; the buffer
    /// exists to satisfy the mesh contract).
    fullscreen_mesh: Arc<Mesh>,
    /// Staged one-time uploads (BRDF LUT, fallback cube, dummy vertex buffer),
    /// drained into the first recorded frame's graph.
    pending_uploads: Vec<TransferOperation>,
}

/// Motion-blur pass materials for one camera (#149), built only when the
/// camera carries a [`MotionBlur`](super::MotionBlur) component.
struct MotionBlurResources {
    /// Velocity → per-tile maximum blur vector.
    tile_max: Arc<MaterialInstance>,
    /// TileMax → 3×3 directional neighbour max.
    neighbor_max: Arc<MaterialInstance>,
    /// Reconstruction, one instance per colour source: `scene_color` (TAA off)
    /// or either TAA history (the one TAA just wrote). The history pair exists
    /// only when the camera has TAA history targets (#148 lazy allocation).
    reconstruct_scene: Arc<MaterialInstance>,
    reconstruct_history: Option<[Arc<MaterialInstance>; 2]>,
    /// Display-output instance reading the blurred `MB_OUTPUT`.
    display: Arc<MaterialInstance>,
}

/// One camera's auto-exposure compute resources (#153), present only when it
/// has a [`CameraAutoExposure`](super::CameraAutoExposure). The `exposure`
/// buffer is also bound (read-only) by the display pass; the histogram and
/// uniform buffers are compute-internal.
struct AutoExposureResources {
    /// [`AE_BINS`] × `u32` luminance histogram (`STORAGE | COPY_DST`): the
    /// build pass accumulates into it, the resolve pass reads it, the prep
    /// transfer clears it each frame.
    histogram: Arc<Buffer>,
    /// Single `f32` persistent exposure multiplier the resolve pass smooths
    /// and the display multiplies in. Survives frames (eye adaptation).
    exposure: Arc<Buffer>,
    /// Per-frame uniform buffers (the compute dispatch has no dynamic offset,
    /// so these can't ride the ring — they're written each frame by the prep
    /// transfer).
    build_uniform: Arc<Buffer>,
    resolve_uniform: Arc<Buffer>,
    build: Arc<MaterialInstance>,
    resolve: Arc<MaterialInstance>,
    /// Whether the persistent `exposure` slot has been seeded to neutral yet —
    /// false right after (re)creation, like [`CameraResources::history_primed`].
    primed: bool,
}

/// One camera's fullscreen-pass materials and the identities they were built
/// against.
struct CameraResources {
    resolve: Arc<MaterialInstance>,
    skybox: Arc<MaterialInstance>,
    /// Velocity-completion pass (#149): fills camera-only motion into the
    /// background of `GBUFFER_VELOCITY`. Always present; only motion blur (and
    /// a future unified TAA) consume the completed background.
    velocity_complete: Arc<MaterialInstance>,
    /// The display-output pass (#142) reading `scene_color` directly — the
    /// TAA-off path.
    display_scene: Arc<MaterialInstance>,
    /// Display-output instances reading each TAA history texture — the
    /// TAA-on path reads the one the TAA pass just wrote. `None` when the
    /// camera has no [`TemporalJitter`](super::TemporalJitter) (the history
    /// targets are not allocated — #148 lazy allocation).
    display_history: Option<[Arc<MaterialInstance>; 2]>,
    /// TAA resolve instances (#148): instance `i` reads history `i` (and
    /// writes the other one — frame parity picks). `None` without
    /// [`TemporalJitter`](super::TemporalJitter).
    taa: Option<[Arc<MaterialInstance>; 2]>,
    /// Motion-blur materials (#149), `Some` only while the camera carries a
    /// [`MotionBlur`](super::MotionBlur) component and the MB targets exist.
    motion_blur: Option<MotionBlurResources>,
    /// `Arc::as_ptr` of `MB_OUTPUT` these MB materials were built against
    /// (`None` when MB is off) — a mismatch (toggle or resize) rebuilds.
    mb_output_ptr: Option<usize>,
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
    /// (instance 0 Karis-samples the motion-blurred image when MB is on, else
    /// `scene_color`; the rest sample the previous mip); `bloom_up[i]`
    /// additively upsamples `bloom_{i+1}` into `bloom_i`. Empty ⇒ the display
    /// binds the shared black bloom and no bloom pass runs.
    bloom_down: Vec<Arc<MaterialInstance>>,
    /// TAA-on variants of `bloom_down[0]` reading each history texture, so the
    /// glow comes off the SAME anti-aliased image the display shows instead of
    /// the raw jittered `scene_color` (which would shimmer at edges). Built
    /// only when bloom + TAA history exist and MB is off (with MB on,
    /// `bloom_down[0]` already reads the post-TAA `MB_OUTPUT`).
    bloom_down0_history: Option<[Arc<MaterialInstance>; 2]>,
    bloom_up: Vec<Arc<MaterialInstance>>,
    /// Whether bloom is on — part of the fresh-check (toggling rebuilds the
    /// display's bloom binding and the bloom materials).
    bloom_enabled: bool,
    /// Auto-exposure compute resources (#153), present only when the camera has
    /// a [`CameraAutoExposure`](super::CameraAutoExposure). `None` ⇒ the display
    /// binds the shared neutral (1.0) exposure buffer and no compute pass runs.
    auto_exposure: Option<AutoExposureResources>,
    /// Whether auto-exposure is on — part of the fresh-check (toggling rebuilds
    /// the display's exposure binding and the compute materials).
    auto_enabled: bool,
    /// Whether the TAA history holds a real frame yet — false right after
    /// (re)creation (resize/format change); the first TAA frame then takes
    /// the current frame wholesale instead of blending with garbage.
    history_primed: bool,
    /// `Arc::as_ptr` of the G-buffer albedo the binding groups reference —
    /// resize re-derives the G-buffer (and `scene_color` with it),
    /// invalidating these materials.
    albedo_ptr: usize,
    /// `Arc::as_ptr` of the camera depth attachment the position-from-depth
    /// bindings reference — a host recreating the depth at the same size
    /// (editor scene view) must also rebuild these materials.
    depth_ptr: usize,
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
        let neutral_exposure = create_neutral_exposure(device, &mut pending_uploads);

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
            neutral_exposure,
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

/// Create the single-`f32` storage buffer holding `1.0` bound to the display
/// pass's exposure slot when a camera has no
/// [`CameraAutoExposure`](super::CameraAutoExposure) (#153): the display
/// multiplies its manual exposure by it, so the multiply is a no-op.
fn create_neutral_exposure(
    device: &Arc<GraphicsDevice>,
    ops: &mut Vec<TransferOperation>,
) -> Arc<Buffer> {
    let buffer = device
        .create_buffer(
            &BufferDescriptor::new(
                std::mem::size_of::<f32>() as u64,
                BufferUsage::STORAGE | BufferUsage::COPY_DST,
            )
            .with_label("exposure_neutral"),
        )
        .expect("create neutral exposure buffer");
    ops.push(TransferOperation::write_buffer(
        Arc::clone(&buffer),
        0,
        Arc::from(bytemuck::bytes_of(&1.0f32)),
    ));
    buffer
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
        // The camera's Depth32Float attachment: the resolve/SSAO/TAA/MB passes
        // reconstruct world position from it (unfilterable binding, Load).
        camera_depth: &Arc<Texture>,
        scene_color: &Arc<Texture>,
        // The TAA history ping-pong, present iff the camera has a
        // `TemporalJitter` (#148 lazy allocation): `None` skips the TAA and
        // history-display materials entirely.
        history: Option<[&Arc<Texture>; 2]>,
        // [tile_max, neighbor_max, output], present iff the camera has MotionBlur.
        mb_targets: Option<[&Arc<Texture>; 3]>,
        env: Option<&Arc<super::ResolvedEnvironment>>,
        // The SSAO targets `[raw, denoised]` when the camera has a
        // `CameraAmbientOcclusion` (#150); `None` binds the shared white AO to
        // the resolve and records no SSAO pass.
        ssao_targets: Option<[&Arc<Texture>; 2]>,
        // The bloom mip chain (`bloom_0..N`) when the camera has a
        // `CameraBloom` (#151); `None`/empty binds the shared black bloom to
        // the display and records no bloom pass.
        bloom_targets: Option<&[&Arc<Texture>]>,
        // Whether the camera has a `CameraAutoExposure` (#153): builds the two
        // compute passes and binds the metered exposure buffer to the display;
        // `false` binds the shared neutral (1.0) exposure and records nothing.
        auto_exposure_on: bool,
        // The previous resources' persistent exposure buffer + primed flag,
        // carried across a rebuild (resize/env change) so the eye adaptation
        // does not visibly reset to 1.0 on every window resize.
        prev_auto_exposure: Option<(Arc<Buffer>, bool)>,
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
                    // Depth buffer read via Load (position reconstruction).
                    .with_unfilterable_texture(1, 4)
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
                    .with_sampler(3, shared.gbuffer_sampler.clone())
                    .with_texture(4, camera_depth.clone()),
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
                    // Linear + clamp (ibl_sampler): the AO target is half-res,
                    // so the resolve bilinearly upsamples it to full res. The
                    // 1×1 white fallback samples identically.
                    .with_sampler(1, shared.ibl_sampler.clone()),
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

        // --- Velocity-completion material (#149): background camera motion ---
        // Writes into the velocity G-buffer (RG16F), so its color format is the
        // velocity attachment's, not scene_color's.
        let velocity_complete_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        VELOCITY_COMPLETE_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        VELOCITY_COMPLETE_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_color_format(GBUFFER_FORMATS[3])
                    .with_dynamic_uniform(0, 0)
                    .with_label("deferred_velocity_complete"),
            )
            .inspect_err(|e| log::error!("deferred: velocity-complete material failed: {e}"))
            .ok()?;
        let velocity_complete_group = device
            .create_binding_group(
                velocity_complete_material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        ring_buffer.clone(),
                        0,
                        std::mem::size_of::<VelocityCompleteUniforms>() as u64,
                    )
                    .with_texture(1, gbuffer[0].clone())
                    .with_sampler(2, shared.gbuffer_sampler.clone()),
            )
            .ok()?;
        let velocity_complete = Arc::new(
            MaterialInstance::new(velocity_complete_material)
                .with_binding_group(velocity_complete_group),
        );
        // --- Auto-exposure compute resources (#153), only when opted in ---
        // Built before the display so the display can bind the metered exposure
        // buffer. When off, the display binds the shared neutral (1.0) buffer —
        // the fallback-binding pattern keeps the display group present either
        // way. The two compute materials are single-group (group 0) and carry
        // no dynamic offset (dispatch has none), so their uniforms live in
        // dedicated per-camera buffers written each frame by the prep transfer.
        let (auto_exposure, exposure_display_buf) = if auto_exposure_on {
            let storage = BufferUsage::STORAGE | BufferUsage::COPY_DST;
            let uniform = BufferUsage::UNIFORM | BufferUsage::COPY_DST;
            let make_buf = |size: u64, usage: BufferUsage, label: &str| -> Option<Arc<Buffer>> {
                device
                    .create_buffer(&BufferDescriptor::new(size, usage).with_label(label))
                    .inspect_err(|e| log::error!("deferred: {label} buffer failed: {e}"))
                    .ok()
            };
            let histogram = make_buf((AE_BINS * 4) as u64, storage, "ae_histogram")?;
            // Reuse the previous resources' persistent exposure slot (and its
            // primed state) across a rebuild — the adapted value survives a
            // window resize instead of visibly popping back to neutral.
            let (exposure, primed) = match prev_auto_exposure {
                Some((buffer, primed)) => (buffer, primed),
                None => (
                    make_buf(std::mem::size_of::<f32>() as u64, storage, "ae_exposure")?,
                    false,
                ),
            };
            let build_uniform = make_buf(
                std::mem::size_of::<AeBuildUniforms>() as u64,
                uniform,
                "ae_build_uniform",
            )?;
            let resolve_uniform = make_buf(
                std::mem::size_of::<AeResolveUniforms>() as u64,
                uniform,
                "ae_resolve_uniform",
            )?;

            let compute_material = |source: &str, label: &str| {
                device
                    .create_material(
                        &MaterialDescriptor::new()
                            .with_shader(ShaderSource::slang(
                                ShaderStage::Compute,
                                source.as_bytes().to_vec(),
                                "cs_main",
                                vec![],
                            ))
                            .with_label(label),
                    )
                    .inspect_err(|e| log::error!("deferred: {label} material failed: {e}"))
                    .ok()
            };
            // Build: uniform, scene_color (linear-sampled over the grid),
            // histogram (read-write).
            let build_mat =
                compute_material(HISTOGRAM_BUILD_SHADER_SLANG, "deferred_histogram_build")?;
            let build_group = device
                .create_binding_group(
                    build_mat.binding_layouts()[0].clone(),
                    BindingGroupDescriptor::new()
                        .with_buffer(0, build_uniform.clone())
                        .with_texture(1, scene_color.clone())
                        .with_sampler(2, shared.ibl_sampler.clone())
                        .with_buffer(3, histogram.clone()),
                )
                .ok()?;
            let build = Arc::new(MaterialInstance::new(build_mat).with_binding_group(build_group));
            // Resolve: uniform, histogram (read-only), exposure (read-write).
            let resolve_mat =
                compute_material(HISTOGRAM_RESOLVE_SHADER_SLANG, "deferred_histogram_resolve")?;
            let resolve_group = device
                .create_binding_group(
                    resolve_mat.binding_layouts()[0].clone(),
                    BindingGroupDescriptor::new()
                        .with_buffer(0, resolve_uniform.clone())
                        .with_buffer(1, histogram.clone())
                        .with_buffer(2, exposure.clone()),
                )
                .ok()?;
            let resolve =
                Arc::new(MaterialInstance::new(resolve_mat).with_binding_group(resolve_group));

            let display_buf = exposure.clone();
            (
                Some(AutoExposureResources {
                    histogram,
                    exposure,
                    build_uniform,
                    resolve_uniform,
                    build,
                    resolve,
                    primed,
                }),
                display_buf,
            )
        } else {
            (None, shared.neutral_exposure.clone())
        };

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
                        .with_sampler(4, shared.ibl_sampler.clone())
                        // Auto-exposure multiplier (#153): the camera's metered
                        // buffer, or the shared neutral 1.0 when off.
                        .with_buffer(5, exposure_display_buf.clone()),
                )
                .ok()?;
            Some(Arc::new(
                MaterialInstance::new(display_material.clone()).with_binding_group(group),
            ))
        };
        let display_scene = display_instance(scene_color)?;
        let display_history = match history {
            Some([a, b]) => Some([display_instance(a)?, display_instance(b)?]),
            None => None,
        };

        // --- TAA resolve material (#148): scene-referred in and out. Built
        // (with its history targets) only for TemporalJitter cameras.
        let taa = match history {
            Some(history) => {
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
                            // Depth buffer read via Load (position
                            // reconstruction for the camera-only reprojection
                            // and the disocclusion proxy).
                            .with_unfilterable_texture(0, 7)
                            .with_label("deferred_taa_resolve"),
                    )
                    .inspect_err(|e| log::error!("deferred: TAA material failed: {e}"))
                    .ok()?;
                // Instance i reads history[i]; frame parity writes the other
                // one. History reprojection samples bilinearly (sub-pixel
                // motion), the rest nearest.
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
                                // Camera depth: the camera-only reprojection
                                // reconstructs world position from it.
                                .with_texture(7, camera_depth.clone()),
                        )
                        .ok()?;
                    Some(Arc::new(
                        MaterialInstance::new(taa_material.clone()).with_binding_group(group),
                    ))
                };
                Some([taa_instance(history[0])?, taa_instance(history[1])?])
            }
            None => None,
        };

        // --- Motion-blur materials (#149): built only when MB targets exist ---
        let motion_blur = if let Some([mb_tile, mb_neighbor, mb_output]) = mb_targets {
            // `depth_read`: the reconstruction pass reads the camera depth via
            // Load (unfilterable binding 4); the tile passes do not.
            let mb_material = |src: &str, fmt: TextureFormat, label: &'static str, depth_read| {
                let mut desc = MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        src.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        src.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_color_format(fmt)
                    .with_dynamic_uniform(0, 0);
                if depth_read {
                    desc = desc.with_unfilterable_texture(0, 4);
                }
                device
                    .create_material(&desc.with_label(label))
                    .inspect_err(|e| log::error!("deferred: {label} material failed: {e}"))
                    .ok()
            };

            // TileMax: velocity -> per-tile max blur vector (RG16F).
            let tile_material = mb_material(
                MB_TILE_MAX_SHADER_SLANG,
                GBUFFER_FORMATS[3],
                "deferred_mb_tile_max",
                false,
            )?;
            let tile_group = device
                .create_binding_group(
                    tile_material.binding_layouts()[0].clone(),
                    BindingGroupDescriptor::new()
                        .with_buffer_range(
                            0,
                            ring_buffer.clone(),
                            0,
                            std::mem::size_of::<MbTileUniforms>() as u64,
                        )
                        .with_texture(1, gbuffer[3].clone())
                        .with_sampler(2, shared.gbuffer_sampler.clone()),
                )
                .ok()?;
            let tile_max =
                Arc::new(MaterialInstance::new(tile_material).with_binding_group(tile_group));

            // NeighborMax: TileMax -> 3x3 directional max (RG16F).
            let neighbor_material = mb_material(
                MB_NEIGHBOR_MAX_SHADER_SLANG,
                GBUFFER_FORMATS[3],
                "deferred_mb_neighbor_max",
                false,
            )?;
            let neighbor_group = device
                .create_binding_group(
                    neighbor_material.binding_layouts()[0].clone(),
                    BindingGroupDescriptor::new()
                        .with_buffer_range(
                            0,
                            ring_buffer.clone(),
                            0,
                            std::mem::size_of::<MbNeighborUniforms>() as u64,
                        )
                        .with_texture(1, mb_tile.clone())
                        .with_sampler(2, shared.gbuffer_sampler.clone()),
                )
                .ok()?;
            let neighbor_max = Arc::new(
                MaterialInstance::new(neighbor_material).with_binding_group(neighbor_group),
            );

            // Reconstruction: colour + NeighborMax + velocity/position/albedo ->
            // blurred MB_OUTPUT. One instance per possible colour source.
            let reconstruct_material = mb_material(
                MB_RECONSTRUCT_SHADER_SLANG,
                SCENE_COLOR_FORMAT,
                "deferred_mb_reconstruct",
                true,
            )?;
            let reconstruct_instance = |color: &Arc<Texture>| -> Option<Arc<MaterialInstance>> {
                let group = device
                    .create_binding_group(
                        reconstruct_material.binding_layouts()[0].clone(),
                        BindingGroupDescriptor::new()
                            .with_buffer_range(
                                0,
                                ring_buffer.clone(),
                                0,
                                std::mem::size_of::<MbReconstructUniforms>() as u64,
                            )
                            .with_texture(1, color.clone())
                            .with_texture(2, mb_neighbor.clone())
                            .with_texture(3, gbuffer[3].clone())
                            // Camera depth (unfilterable, Load): the depth
                            // proxy reconstructs world position from it.
                            .with_texture(4, camera_depth.clone())
                            .with_texture(5, gbuffer[0].clone())
                            .with_sampler(6, shared.gbuffer_sampler.clone())
                            .with_sampler(7, shared.ibl_sampler.clone()),
                    )
                    .ok()?;
                Some(Arc::new(
                    MaterialInstance::new(reconstruct_material.clone()).with_binding_group(group),
                ))
            };
            let reconstruct_scene = reconstruct_instance(scene_color)?;
            let reconstruct_history = match history {
                Some([a, b]) => Some([reconstruct_instance(a)?, reconstruct_instance(b)?]),
                None => None,
            };

            // Display reads the blurred output instead of scene_color/history.
            let display = display_instance(mb_output)?;

            Some(MotionBlurResources {
                tile_max,
                neighbor_max,
                reconstruct_scene,
                reconstruct_history,
                display,
            })
        } else {
            None
        };
        let mb_output_ptr = mb_targets.map(|t| Arc::as_ptr(t[2]) as usize);

        // --- SSAO passes (#150), only when the camera opted in ---
        // A tiny fullscreen material builder shared by the horizon and blur
        // passes (both target R8, dynamic uniform in group 0).
        let fullscreen_ssao_material = |source: &str, label: &str, depth_binding: u32| {
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
                        // Both passes read the camera depth via Load
                        // (position reconstruction / edge stop).
                        .with_unfilterable_texture(0, depth_binding)
                        .with_label(label),
                )
                .inspect_err(|e| log::error!("deferred: {label} material failed: {e}"))
                .ok()
        };
        let (ssao, ssao_blur) = match ssao_targets {
            Some([raw, _ao]) => {
                let ssao_material =
                    fullscreen_ssao_material(SSAO_SHADER_SLANG, "deferred_ssao", 3)?;
                // Horizon pass: albedo (marker), normal, depth + sampler.
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
                            .with_texture(3, camera_depth.clone())
                            .with_sampler(4, shared.gbuffer_sampler.clone()),
                    )
                    .ok()?;
                let ssao =
                    Arc::new(MaterialInstance::new(ssao_material).with_binding_group(ssao_group));

                let blur_material =
                    fullscreen_ssao_material(SSAO_BLUR_SHADER_SLANG, "deferred_ssao_blur", 2)?;
                // Denoise pass: raw AO + depth (edge stop) + sampler.
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
                            .with_texture(2, camera_depth.clone())
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
        let (bloom_down, bloom_down0_history, bloom_up) = match bloom_targets {
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

                // down[0] Karis-samples the post-motion-blur image when motion
                // blur is on, so bloom glows off the *blurred* frame: blur is a
                // sensor-time effect and bloom bleeds from what the sensor saw,
                // hence TAA -> MB (#149) -> bloom (#151) -> exposure. Without
                // motion blur it samples scene_color. down[i] samples mip i-1.
                let mut down = Vec::with_capacity(n);
                let bloom_source = mb_targets.map_or(scene_color, |mb| mb[2]);
                down.push(bloom_instance(&down_karis, bloom_source)?);
                for i in 1..n {
                    down.push(bloom_instance(&down_plain, mips[i - 1])?);
                }
                // TAA-on, MB-off variants of down[0] reading each history: the
                // display shows the TAA history, so the glow must come off the
                // same anti-aliased image, not the raw jittered scene_color
                // (edge shimmer). With MB on the MB_OUTPUT source above is
                // already post-TAA, so no variants are needed.
                let down0_history = match (mb_targets.is_none(), history) {
                    (true, Some([a, b])) => Some([
                        bloom_instance(&down_karis, a)?,
                        bloom_instance(&down_karis, b)?,
                    ]),
                    _ => None,
                };
                // up[i] upsamples mip i+1 into mip i (additive), for i in 0..n-1.
                let mut up = Vec::with_capacity(n.saturating_sub(1));
                for i in 0..n.saturating_sub(1) {
                    up.push(bloom_instance(&up_material, mips[i + 1])?);
                }
                (down, down0_history, up)
            }
            _ => (Vec::new(), None, Vec::new()),
        };

        Some(Self {
            resolve,
            skybox,
            velocity_complete,
            display_scene,
            display_history,
            taa,
            motion_blur,
            mb_output_ptr,
            ssao,
            ssao_blur,
            ao_enabled: ssao_targets.is_some(),
            bloom_down,
            bloom_down0_history,
            bloom_up,
            bloom_enabled: bloom_targets.is_some_and(|m| !m.is_empty()),
            auto_enabled: auto_exposure.is_some(),
            auto_exposure,
            history_primed: false,
            albedo_ptr: Arc::as_ptr(gbuffer[0]) as usize,
            depth_ptr: Arc::as_ptr(camera_depth) as usize,
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
        // Motion blur (#149) allocates its intermediates only when opted in.
        let mb_on = world.get::<super::MotionBlur>(camera).is_some();

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
                GBUFFER_ROUGHNESS,
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

        // TAA history ping-pong (#148): two full-res Rgba16Float textures —
        // allocated only for TemporalJitter cameras (they are the single
        // biggest optional cost, ~33 MB at 1080p) and pruned when the
        // component leaves, so toggling never blends against a stale history
        // (the material rebuild below re-primes).
        let taa_on = world.get::<super::TemporalJitter>(camera).is_some();
        if taa_on {
            let history_stale = TAA_HISTORY.iter().any(|&name| {
                targets
                    .get(name)
                    .map(|t| t.size().width != width || t.size().height != height)
                    .unwrap_or(true)
            });
            if history_stale {
                let usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING;
                for name in TAA_HISTORY {
                    let texture = device
                        .create_texture(
                            &TextureDescriptor::new_2d(width, height, SCENE_COLOR_FORMAT, usage)
                                .with_label(name),
                        )
                        .expect("create TAA history texture");
                    targets.set(name, texture);
                }
                let _ = world.insert(camera, targets.clone());
            }
        } else if TAA_HISTORY.iter().any(|&name| targets.get(name).is_some()) {
            for name in TAA_HISTORY {
                targets.remove(name);
            }
            let _ = world.insert(camera, targets.clone());
        }

        // Motion-blur intermediates (#149): two tile-resolution velocity
        // reductions + a full-res output, allocated only when opted in and
        // pruned when the component leaves.
        if !mb_on
            && [MB_TILE_MAX, MB_NEIGHBOR_MAX, MB_OUTPUT]
                .iter()
                .any(|&name| targets.get(name).is_some())
        {
            for name in [MB_TILE_MAX, MB_NEIGHBOR_MAX, MB_OUTPUT] {
                targets.remove(name);
            }
            let _ = world.insert(camera, targets.clone());
        }
        if mb_on {
            let (tile_w, tile_h) = (width.div_ceil(MB_TILE_SIZE), height.div_ceil(MB_TILE_SIZE));
            let mb_stale = targets
                .get(MB_OUTPUT)
                .map(|t| t.size().width != width || t.size().height != height)
                .unwrap_or(true);
            if mb_stale {
                let usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING;
                for name in [MB_TILE_MAX, MB_NEIGHBOR_MAX] {
                    let tex = device
                        .create_texture(
                            &TextureDescriptor::new_2d(tile_w, tile_h, GBUFFER_FORMATS[3], usage)
                                .with_label(name),
                        )
                        .expect("create MB tile texture");
                    targets.set(name, tex);
                }
                let out = device
                    .create_texture(
                        &TextureDescriptor::new_2d(width, height, SCENE_COLOR_FORMAT, usage)
                            .with_label(MB_OUTPUT),
                    )
                    .expect("create MB output texture");
                targets.set(MB_OUTPUT, out);
                let _ = world.insert(camera, targets.clone());
            }
        }

        // SSAO targets (#150): derived only for cameras with the component, at
        // the same size as the G-buffer (re-derived on resize), pruned when the
        // component leaves (the resolve rebinds the shared white AO).
        let ao_enabled = world.get::<super::CameraAmbientOcclusion>(camera).is_some();
        if !ao_enabled
            && [SSAO_RAW, SSAO_AO]
                .iter()
                .any(|&name| targets.get(name).is_some())
        {
            for name in [SSAO_RAW, SSAO_AO] {
                targets.remove(name);
            }
            let _ = world.insert(camera, targets.clone());
        }
        if ao_enabled {
            // Half-res (#150 perf): SSAO is low-frequency and the bilateral
            // denoise + bilinear upsample in the resolve hide the lower
            // sampling rate, so both SSAO passes run at half the G-buffer
            // extent — ~4× fewer invocations, the dominant cost lever. The
            // per-frame texel the shaders sample by is derived from these same
            // dims (`div_ceil` keeps odd extents in step).
            let ao_w = width.div_ceil(2).max(1);
            let ao_h = height.div_ceil(2).max(1);
            let ao_stale = targets
                .get(SSAO_AO)
                .map(|t| t.size().width != ao_w || t.size().height != ao_h)
                .unwrap_or(true);
            if ao_stale {
                let usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING;
                for name in [SSAO_RAW, SSAO_AO] {
                    let texture = device
                        .create_texture(
                            &TextureDescriptor::new_2d(ao_w, ao_h, SSAO_FORMAT, usage)
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
        if !bloom_enabled && targets.get(&bloom_mip_key(0)).is_some() {
            // Pruned when the component leaves (the display rebinds the shared
            // black bloom).
            for i in 0..MAX_BLOOM_MIPS {
                targets.remove(&bloom_mip_key(i));
            }
            let _ = world.insert(camera, targets.clone());
        }
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
        let (Some(albedo), Some(normal), Some(roughness), Some(velocity), Some(scene_color)) = (
            targets.get(GBUFFER_ALBEDO),
            targets.get(GBUFFER_NORMAL_METALLIC),
            targets.get(GBUFFER_ROUGHNESS),
            targets.get(GBUFFER_VELOCITY),
            targets.get(SCENE_COLOR),
        ) else {
            return;
        };
        // TAA history pair present iff opted in and derived above (#148).
        let history = if taa_on {
            match (targets.get(TAA_HISTORY[0]), targets.get(TAA_HISTORY[1])) {
                (Some(a), Some(b)) => Some([a, b]),
                _ => None,
            }
        } else {
            None
        };
        // MB targets present iff opted in and derived above.
        let mb_targets = if mb_on {
            match (
                targets.get(MB_TILE_MAX),
                targets.get(MB_NEIGHBOR_MAX),
                targets.get(MB_OUTPUT),
            ) {
                (Some(t), Some(n), Some(o)) => Some([t, n, o]),
                _ => None,
            }
        } else {
            None
        };
        let albedo_ptr = Arc::as_ptr(albedo) as usize;
        let mb_output_ptr = mb_targets.map(|t| Arc::as_ptr(t[2]) as usize);
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
        // Auto-exposure (#153): buffers/materials live in CameraResources (no
        // pipeline targets), so this just gates the create call. `auto_on`
        // folds into the fresh-check so toggling rebuilds the display's
        // exposure binding and the compute materials.
        let auto_on = world.get::<super::CameraAutoExposure>(camera).is_some();
        let mut cameras = self.cameras.lock().expect("deferred camera map poisoned");
        // Evict cameras whose entity died (scene reload, despawn) — their
        // materials would otherwise pin GPU textures forever.
        cameras.retain(|entity, _| world.is_alive(*entity));
        let depth_ptr = Arc::as_ptr(&target.depth) as usize;
        let fresh = cameras.get(&camera).is_some_and(|c| {
            c.albedo_ptr == albedo_ptr
                && c.depth_ptr == depth_ptr
                && c.color_format == color_format
                && c.mb_output_ptr == mb_output_ptr
                && c.env_ptr == env_ptr
                && c.ao_enabled == ao_on
                // Toggling TemporalJitter reallocates the history pair, so the
                // rebuild also resets `history_primed` — a re-enabled TAA never
                // blends against a stale history.
                && c.taa.is_some() == history.is_some()
                && c.bloom_enabled == bloom_on
                && c.auto_enabled == auto_on
        });
        if !fresh {
            // Carry the adapted exposure across the rebuild (resize must not
            // visibly reset the eye adaptation).
            let prev_auto_exposure = cameras
                .get(&camera)
                .and_then(|c| c.auto_exposure.as_ref())
                .map(|ae| (ae.exposure.clone(), ae.primed));
            match CameraResources::create(
                &device,
                shared,
                [albedo, normal, roughness, velocity],
                &target.depth,
                scene_color,
                history,
                mb_targets,
                env.as_ref(),
                ssao_targets,
                bloom_chain.as_deref(),
                auto_on,
                prev_auto_exposure,
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
        let (Some(albedo), Some(normal), Some(roughness), Some(velocity), Some(scene_color)) = (
            targets.get(GBUFFER_ALBEDO),
            targets.get(GBUFFER_NORMAL_METALLIC),
            targets.get(GBUFFER_ROUGHNESS),
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
        // Unjittered inverse for the reprojection passes (velocity completion
        // #149, TAA #148): the jittered raster inverse would bake the sub-pixel
        // offset into every ray.
        let unjittered_vp = Mat4::from(view.view_projection_unjittered);
        let inv_unjittered_cols = redlilium_core::math::mat4_to_cols_array_2d(
            &unjittered_vp.try_inverse().unwrap_or(unjittered_vp),
        );
        let camera_pos = world
            .get::<crate::std::components::Camera>(view.entity)
            .and_then(|cam| cam.view_matrix.try_inverse())
            .map(|m| [m[(0, 3)], m[(1, 3)], m[(2, 3)]])
            .unwrap_or([0.0; 3]);

        // This frame's direct lights, from the ECS light components (#146).
        // camera_pos.w carries the environment's reflection-LOD range (#145) —
        // the resolve shader reads only .xyz for position.
        let mut resolve_uniforms = gather_lights(world);
        // The JITTERED inverse: the resolve reconstructs the raster sample's
        // world position from depth, and the raster used the jittered matrix.
        let inv_view_proj_cols = redlilium_core::math::mat4_to_cols_array_2d(&inv_view_proj);
        resolve_uniforms.inv_view_proj = inv_view_proj_cols;
        resolve_uniforms.camera_pos = [
            camera_pos[0],
            camera_pos[1],
            camera_pos[2],
            camera_resources.max_reflection_lod,
        ];

        // Exposure the display applies (#142). With auto-exposure (#153) on,
        // the metered multiplier comes from the GPU buffer and this carries
        // only the compensation bias (`2^compensation`); otherwise it is the
        // manual CameraExposure (neutral when absent).
        let exposure = if camera_resources.auto_exposure.is_some() {
            world
                .get::<super::CameraAutoExposure>(view.entity)
                .map_or(1.0, |a| 2f32.powf(a.compensation))
        } else {
            world
                .get::<super::CameraExposure>(view.entity)
                .map_or(1.0, |e| e.exposure)
        };
        // Frame delta for the eye-adaptation smoothing (#153); a 60 Hz step
        // when the clock is absent (headless/tests).
        let dt = if world.has_resource::<crate::RealTime>() {
            world.resource::<crate::RealTime>().delta() as f32
        } else {
            1.0 / 60.0
        };

        // Display headroom H (#154) the HDR display path rolls highlights off
        // to. Absent ⇒ 1.0 (SDR), so every headless/test path is display-
        // independent and the SDR curve is unchanged.
        let headroom = if world.has_resource::<DisplayHeadroom>() {
            world.resource::<DisplayHeadroom>().get()
        } else {
            1.0
        };

        // Bloom intensity (#151): the CameraBloom weight, but only when the
        // bloom materials/targets exist (else 0 — the display binds black).
        let bloom_intensity = if camera_resources.bloom_enabled {
            world
                .get::<super::CameraBloom>(view.entity)
                .map_or(0.0, |b| b.intensity)
        } else {
            0.0
        };

        // TAA (#148) runs for cameras that opted into the temporal contract
        // (and whose TAA materials/history exist — lazy allocation); frame
        // parity picks the history ping-pong direction (read this index,
        // write the other).
        let taa_read_index = (world.get::<super::TemporalJitter>(view.entity).is_some()
            && camera_resources.taa.is_some()
            && world.has_resource::<super::TemporalState>())
        .then(|| (world.resource::<super::TemporalState>().frame() % 2) as usize);

        // Motion blur (#149) runs when the camera opted in and its materials
        // were built (targets present).
        let mb = world.get::<super::MotionBlur>(view.entity).copied();
        let mb_on = mb.is_some() && camera_resources.motion_blur.is_some();

        // dt-normalized MB shutter (#149): blur length ~ per-frame velocity ~
        // dt, so scale the shutter by smoothed_dt / dt to tie blur to a *stable*
        // reference interval — frame-time jitter (the vsync 16/17ms beat) then no
        // longer strobes it sharp/blurred. Clamp is asymmetric: a generous floor
        // lets a genuine hitch's frame collapse back to normal blur length; a
        // tight ceiling guards the one blow-up direction (dt anomalously small).
        // Applied to the MB shutter only — GBUFFER_VELOCITY and the TAA/velocity
        // contract (#147) stay untouched.
        let mb_shutter_ratio = if mb_on && world.has_resource::<crate::Time>() {
            let time = world.resource::<crate::Time>();
            (time.smoothed_frame_delta() / time.frame_delta().max(1e-6)).clamp(0.15, 2.0) as f32
        } else {
            1.0
        };
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
                        // Texel of the half-res SSAO target (built at
                        // width.div_ceil(2) × height.div_ceil(2)): it sets the
                        // march step size, the sub-texel guard, the IGN pixel
                        // grid, and the blur tap stride — all in SSAO-target
                        // space, so it must match the half-res extent, not the
                        // full-res scene.
                        let size = scene_color.size();
                        let ao_w = size.width.div_ceil(2).max(1);
                        let ao_h = size.height.div_ceil(2).max(1);
                        let texel = [1.0 / ao_w as f32, 1.0 / ao_h as f32];
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
                            // Jittered inverse: positions reconstruct the
                            // raster samples exactly.
                            inv_view_proj: inv_view_proj_cols,
                            params: [ao.radius, ao.intensity, ao.power, frame_rot],
                            params2: [texel[0], texel[1], 0.0, 0.0],
                        };
                        let blur = SsaoBlurUniforms {
                            inv_view_proj: inv_view_proj_cols,
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
            velocity_complete_offset,
            resolve_offset,
            display_offset,
            taa_offset,
            mb_offsets,
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
            // Velocity completion (#149) — always on; the unjittered pair, so
            // the background velocity matches the G-buffer's convention.
            let velocity_complete_offset =
                ring.push(bytemuck::bytes_of(&VelocityCompleteUniforms {
                    inv_view_proj: inv_unjittered_cols,
                    prev_view_proj: view.prev_view_projection,
                }));
            let resolve_offset = ring.push(bytemuck::bytes_of(&resolve_uniforms));
            let display_offset = ring.push(bytemuck::bytes_of(&DisplayOutputUniforms {
                exposure: [exposure, bloom_intensity, headroom, 0.0],
            }));
            let taa_offset = taa_read_index.map(|_| {
                let size = scene_color.size();
                ring.push(bytemuck::bytes_of(&TaaUniforms {
                    inv_view_proj: inv_unjittered_cols,
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
                    camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
                }))
            });
            // Motion blur (#149): TileMax / NeighborMax / reconstruction slots.
            let mb_offsets = if mb_on {
                let mb = mb.expect("mb_on implies MotionBlur present");
                let size = scene_color.size();
                let (w, h) = (size.width.max(1) as f32, size.height.max(1) as f32);
                let inv_resolution = [1.0 / w, 1.0 / h];
                let resolution = [w, h];
                let tile_w = size.width.div_ceil(MB_TILE_SIZE).max(1) as f32;
                let tile_h = size.height.div_ceil(MB_TILE_SIZE).max(1) as f32;
                let tile = ring.push(bytemuck::bytes_of(&MbTileUniforms {
                    inv_resolution,
                    resolution,
                    shutter: mb.shutter * mb_shutter_ratio,
                    tile_size: MB_TILE_SIZE as f32,
                    _pad: [0.0; 2],
                }));
                let neighbor = ring.push(bytemuck::bytes_of(&MbNeighborUniforms {
                    inv_tile_resolution: [1.0 / tile_w, 1.0 / tile_h],
                    _pad: [0.0; 2],
                }));
                let reconstruct = ring.push(bytemuck::bytes_of(&MbReconstructUniforms {
                    // Unjittered inverse (matches the pass's velocity
                    // conventions; the jitter bias cancels in the depth
                    // comparison — both samples reconstruct the same way).
                    inv_view_proj: inv_unjittered_cols,
                    inv_resolution,
                    resolution,
                    camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
                    // Same shutter as TileMax — McGuire's invariant that no tile
                    // advertises more blur than reconstruction gathers.
                    shutter: mb.shutter * mb_shutter_ratio,
                    tile_size: MB_TILE_SIZE as f32,
                    samples: mb.samples as f32,
                    _pad: 0.0,
                }));
                Some((tile, neighbor, reconstruct))
            } else {
                None
            };
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
                velocity_complete_offset,
                resolve_offset,
                display_offset,
                taa_offset,
                mb_offsets,
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
                    // Roughness clears to zero (background; the resolve
                    // discards those pixels before reading it).
                    ColorAttachment::from_texture(roughness.clone())
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

        // --- 1b. Velocity completion (#149): fill background camera motion
        // into the velocity G-buffer. Loads (not clears) so geometry velocity
        // survives; the shader discards geometry fragments. Consumed by motion
        // blur's TileMax (and, later, a unified TAA).
        let mut velocity_complete_pass = GraphicsPass::new("velocity_complete".into());
        velocity_complete_pass.set_render_targets(RenderTargetConfig::new().with_color(
            ColorAttachment::from_texture(velocity.clone()).with_load_op(LoadOp::Load),
        ));
        velocity_complete_pass.add_draw_command(
            DrawCommand::new(
                shared.fullscreen_mesh.clone(),
                camera_resources.velocity_complete.clone(),
            )
            .with_dynamic_offsets(vec![vec![velocity_complete_offset]]),
        );
        let velocity_complete_handle = graph.add_graphics_pass(velocity_complete_pass);
        // Runs after the G-buffer: it reads albedo and rewrites the velocity
        // target the G-buffer just cleared/wrote (write-after-write).
        graph.add_dependency(velocity_complete_handle, gbuffer_handle);
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
        // scene_color into the history ping-pong. `color_writer` is the pass
        // whose output the display (or motion blur) then reads; `mb_color_index`
        // picks the matching reconstruction instance. Resolved up front so a
        // missing piece degrades to the TAA-off path instead of aborting the
        // record mid-graph (orphan passes, no display).
        let taa_setup = match (taa_read_index, taa_offset) {
            (Some(read), Some(taa_offset)) => {
                let write = 1 - read;
                match (
                    targets.get(TAA_HISTORY[write]),
                    camera_resources.taa.as_ref(),
                    camera_resources.display_history.as_ref(),
                ) {
                    (Some(history_write), Some(taa), Some(display_history)) => Some((
                        write,
                        taa_offset,
                        history_write.clone(),
                        taa[read].clone(),
                        display_history[write].clone(),
                    )),
                    _ => None,
                }
            }
            _ => None,
        };
        // A frame without a TAA pass leaves the history stale — un-prime it so
        // the next TAA frame takes the current frame wholesale instead of
        // blending against old content.
        if taa_setup.is_none() {
            camera_resources.history_primed = false;
        }
        let (taa_display, color_writer, taa_handle, mb_color_index) = match taa_setup {
            Some((write, taa_offset, history_write, taa_instance, display_instance)) => {
                let mut taa_pass = GraphicsPass::new("taa_resolve".into());
                taa_pass.set_render_targets(
                    RenderTargetConfig::new().with_color(
                        ColorAttachment::from_texture(history_write)
                            .with_clear_color(0.0, 0.0, 0.0, 1.0),
                    ),
                );
                taa_pass.add_draw_command(
                    DrawCommand::new(shared.fullscreen_mesh.clone(), taa_instance)
                        .with_dynamic_offsets(vec![vec![taa_offset]]),
                );
                let taa_handle = graph.add_graphics_pass(taa_pass);
                graph.add_dependency(taa_handle, resolve_handle);
                camera_resources.history_primed = true;
                (display_instance, taa_handle, Some(taa_handle), Some(write))
            }
            None => (
                camera_resources.display_scene.clone(),
                resolve_handle,
                None,
                None,
            ),
        };

        // --- 4b. Motion blur (#149): TileMax -> NeighborMax -> reconstruction
        // into MB_OUTPUT, which the display pass then reads. Opt-in only.
        // Inputs resolved up front so a missing target/instance degrades to the
        // MB-off path instead of aborting the record mid-graph.
        let mb_setup = match (mb_on, mb_offsets, camera_resources.motion_blur.as_ref()) {
            (true, Some(offsets), Some(mb)) => {
                let reconstruct = match mb_color_index {
                    Some(write) => mb.reconstruct_history.as_ref().map(|h| h[write].clone()),
                    None => Some(mb.reconstruct_scene.clone()),
                };
                match (
                    targets.get(MB_TILE_MAX),
                    targets.get(MB_NEIGHBOR_MAX),
                    targets.get(MB_OUTPUT),
                    reconstruct,
                ) {
                    (Some(tile), Some(neighbor), Some(output), Some(reconstruct)) => Some((
                        offsets,
                        mb,
                        tile.clone(),
                        neighbor.clone(),
                        output.clone(),
                        reconstruct,
                    )),
                    _ => None,
                }
            }
            _ => None,
        };
        let mb_active = mb_setup.is_some();
        let (display_source, display_dep) = match mb_setup {
            Some((
                (tile_off, neighbor_off, recon_off),
                mb,
                mb_tile,
                mb_neighbor,
                mb_output,
                reconstruct,
            )) => {
                let mut tile_pass = GraphicsPass::new("mb_tile_max".into());
                tile_pass.set_render_targets(
                    RenderTargetConfig::new()
                        .with_color(ColorAttachment::from_texture(mb_tile.clone())),
                );
                tile_pass.add_draw_command(
                    DrawCommand::new(shared.fullscreen_mesh.clone(), mb.tile_max.clone())
                        .with_dynamic_offsets(vec![vec![tile_off]]),
                );
                let tile_handle = graph.add_graphics_pass(tile_pass);
                // Reads the completed velocity (background filled in).
                graph.add_dependency(tile_handle, velocity_complete_handle);

                let mut neighbor_pass = GraphicsPass::new("mb_neighbor_max".into());
                neighbor_pass.set_render_targets(
                    RenderTargetConfig::new()
                        .with_color(ColorAttachment::from_texture(mb_neighbor.clone())),
                );
                neighbor_pass.add_draw_command(
                    DrawCommand::new(shared.fullscreen_mesh.clone(), mb.neighbor_max.clone())
                        .with_dynamic_offsets(vec![vec![neighbor_off]]),
                );
                let neighbor_handle = graph.add_graphics_pass(neighbor_pass);
                graph.add_dependency(neighbor_handle, tile_handle);

                let mut recon_pass = GraphicsPass::new("mb_reconstruct".into());
                recon_pass.set_render_targets(
                    RenderTargetConfig::new()
                        .with_color(ColorAttachment::from_texture(mb_output.clone())),
                );
                recon_pass.add_draw_command(
                    DrawCommand::new(shared.fullscreen_mesh.clone(), reconstruct)
                        .with_dynamic_offsets(vec![vec![recon_off]]),
                );
                let recon_handle = graph.add_graphics_pass(recon_pass);
                graph.add_dependency(recon_handle, neighbor_handle);
                // Reads the colour the display would otherwise show.
                graph.add_dependency(recon_handle, color_writer);

                (mb.display.clone(), recon_handle)
            }
            None => (taa_display, taa_handle.unwrap_or(resolve_handle)),
        };

        // --- 4c. Bloom (#151): down/up mip chain, when the camera opted in.
        // The glow sources the SAME image the display shows: the motion-blurred
        // MB_OUTPUT when MB is on (#149), else the TAA history just written
        // (#148 — the raw jittered scene_color would shimmer at edges), else
        // scene_color (post-resolve); the display composites the accumulated
        // top mip (bloom_0).
        let bloom_final: Option<PassHandle> = if camera_resources.bloom_down.is_empty() {
            None
        } else {
            let n = camera_resources.bloom_down.len();
            let mips: Option<Vec<&Arc<Texture>>> =
                (0..n).map(|i| targets.get(&bloom_mip_key(i))).collect();
            // down[0]'s instance + the pass that wrote its source.
            let (down0_instance, down0_dep) = if mb_active {
                (camera_resources.bloom_down[0].clone(), display_dep)
            } else {
                match (
                    mb_color_index,
                    taa_handle,
                    camera_resources.bloom_down0_history.as_ref(),
                ) {
                    (Some(write), Some(taa_handle), Some(down0_history)) => {
                        (down0_history[write].clone(), taa_handle)
                    }
                    _ => (camera_resources.bloom_down[0].clone(), resolve_handle),
                }
            };
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
                    let instance = if i == 0 {
                        down0_instance.clone()
                    } else {
                        camera_resources.bloom_down[i].clone()
                    };
                    pass.add_draw_command(
                        DrawCommand::new(shared.fullscreen_mesh.clone(), instance)
                            .with_dynamic_offsets(vec![vec![bloom_down_offsets[i]]]),
                    );
                    let h = graph.add_graphics_pass(pass);
                    // down[0] reads its source's writer (MB reconstruction, TAA
                    // resolve, or the deferred resolve); down[i] reads mip i-1.
                    graph.add_dependency(
                        h,
                        if i == 0 {
                            down0_dep
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

        // --- 4d. Auto-exposure (#153): histogram build + resolve compute, when
        // the camera opted in. Reads scene_color (post-resolve), writes the
        // persistent exposure buffer the display multiplies in. Independent of
        // TAA and bloom; the display waits on the resolve.
        let auto_handle: Option<PassHandle> = {
            let ae_params = world
                .get::<super::CameraAutoExposure>(view.entity)
                .map(|a| {
                    // Per-frame smoothing rate from dt + the eye-like speeds.
                    let rate = |speed: f32| 1.0 - (-dt * speed.max(0.0)).exp();
                    (
                        rate(a.speed_up),
                        rate(a.speed_down),
                        2f32.powf(a.min_ev),
                        2f32.powf(a.max_ev),
                    )
                });
            match (camera_resources.auto_exposure.as_mut(), ae_params) {
                (Some(ae), Some((rate_scene_brighten, rate_scene_darken, min_mult, max_mult))) => {
                    let build_u = AeBuildUniforms {
                        params: [AE_LOG_MIN, AE_LOG_MAX, AE_SAMPLE_DIM as f32, 0.0],
                    };
                    // Slot pairing (see AeResolveUniforms): a rising multiplier
                    // means the scene DARKENED, so params.z carries speed_down;
                    // a falling one means it brightened — limits.z carries
                    // speed_up (the fast, eye-like stop-down).
                    let resolve_u = AeResolveUniforms {
                        params: [AE_LOG_MIN, AE_LOG_MAX, rate_scene_darken, AE_KEY],
                        limits: [min_mult, max_mult, rate_scene_brighten, 0.0],
                    };
                    // Uniforms + histogram clear ride a transfer the build waits
                    // on (the dispatch has no dynamic offset, so uniforms can't
                    // ride the ring).
                    let mut ops = vec![
                        TransferOperation::write_buffer(
                            ae.build_uniform.clone(),
                            0,
                            Arc::from(bytemuck::bytes_of(&build_u)),
                        ),
                        TransferOperation::write_buffer(
                            ae.resolve_uniform.clone(),
                            0,
                            Arc::from(bytemuck::bytes_of(&resolve_u)),
                        ),
                        TransferOperation::write_buffer(
                            ae.histogram.clone(),
                            0,
                            Arc::from(vec![0u8; AE_BINS * 4]),
                        ),
                    ];
                    if !ae.primed {
                        // Seed the persistent exposure slot to neutral before the
                        // first read (create_buffer leaves it uninitialized).
                        ops.push(TransferOperation::write_buffer(
                            ae.exposure.clone(),
                            0,
                            Arc::from(bytemuck::bytes_of(&1.0f32)),
                        ));
                        ae.primed = true;
                    }
                    let mut prep = TransferPass::new("auto_exposure_prep".into());
                    prep.set_transfer_config(TransferConfig::new().with_operations(ops));
                    let prep_handle = graph.add_transfer_pass(prep);

                    // Build: one thread per grid sample (fixed 512×512).
                    let mut build_pass = ComputePass::new("histogram_build".into());
                    build_pass.add_dispatch(
                        ae.build.clone(),
                        AE_SAMPLE_DIM / 16,
                        AE_SAMPLE_DIM / 16,
                        1,
                    );
                    let build_handle = graph.add_compute_pass(build_pass);
                    graph.add_dependency(build_handle, prep_handle); // uniforms + cleared histogram
                    graph.add_dependency(build_handle, resolve_handle); // scene_color written

                    // Resolve: single thread meters + smooths the exposure.
                    let mut resolve_pass = ComputePass::new("histogram_resolve".into());
                    resolve_pass.add_dispatch(ae.resolve.clone(), 1, 1, 1);
                    let ae_resolve_handle = graph.add_compute_pass(resolve_pass);
                    graph.add_dependency(ae_resolve_handle, build_handle);
                    // The resolve reads the histogram and (first frame) the
                    // prep-seeded exposure — anchor after the prep too.
                    graph.add_dependency(ae_resolve_handle, prep_handle);
                    Some(ae_resolve_handle)
                }
                _ => None,
            }
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
        // The display reads whichever pass produced its source: the motion-blur
        // reconstruction, else the TAA history, else the resolve (scene_color
        // has two writers, so anchor explicitly on the last).
        graph.add_dependency(display_handle, display_dep);
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
        // The display reads the metered exposure buffer — order it after the
        // auto-exposure resolve wrote it (#153).
        if let Some(auto_handle) = auto_handle {
            graph.add_dependency(display_handle, auto_handle);
        }

        Some(display_handle)
    }
}
