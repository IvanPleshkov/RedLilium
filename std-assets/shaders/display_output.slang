// Standard display-output pass (#142) — the single place scene-referred
// linear radiance becomes display-referred output. Samples the deferred
// path's `scene_color` intermediate (Rgba16Float, 1.0 = SDR white), applies
// the manual exposure, and encodes for the actual output target. Everything
// upstream (G-buffer, skybox, resolve — and later TAA/post effects) works in
// scene-referred linear and never encodes.
//
// Output-transform axes, selected by the pipeline from the camera target's
// format (the same scheme egui uses for UI composition — both honor
// 1.0 = SDR white, so scene and UI agree on EDR surfaces):
//   HDR_OUTPUT       - highlight roll-off to the display headroom for a
//                      linear-HDR target: paper-white (1.0) is pinned, only
//                      highlights >1 extend up to H (exposure.z, in x-SDR-white
//                      units). At H=1 this equals the SDR curve below.
//   SRGB_FRAMEBUFFER - tonemapped linear for an sRGB-typed target (the
//                      hardware applies the linear->sRGB encode on write;
//                      encoding manually would double-gamma the image)
//   (both off)       - tonemap + manual exact sRGB encode for a non-sRGB
//                      SDR target
//#pragma variant_system HDR_OUTPUT
//#pragma variant_system SRGB_FRAMEBUFFER

import color;

// =============================================================================
// Vertex Shader
// =============================================================================

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    VsOutput output;

    // Fullscreen triangle
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };

    float2 uvs[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };

    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

// =============================================================================
// Fragment Shader
// =============================================================================

[[vk::binding(0, 0)]]
cbuffer DisplayOutputUniforms {
    // x = linear exposure multiplier (CameraExposure, #142);
    // y = bloom intensity (CameraBloom, #151; 0 when the bloom input is the
    //     1x1 black fallback);
    // z = display headroom H (#154), peak/paper-white in x-SDR-white units,
    //     >= 1; 1.0 for SDR (and every test/headless path). Only HDR_OUTPUT
    //     reads it. w unused.
    float4 exposure;
};

[[vk::binding(1, 0)]]
Texture2D scene_color;
[[vk::binding(2, 0)]]
SamplerState scene_sampler;
// Accumulated bloom (#151), or a 1x1 black texture when the camera has no
// CameraBloom — so this composite is a no-op there. Sampled with a linear
// sampler (the bloom mip is lower-res than the scene).
[[vk::binding(3, 0)]]
Texture2D bloom;
[[vk::binding(4, 0)]]
SamplerState bloom_sampler;
// Auto-exposure (#153): the metered/adapted exposure multiplier the compute
// passes wrote, or a 1x1 neutral 1.0 buffer when the camera has no
// CameraAutoExposure — so the multiply is a no-op there and exposure.x (the
// manual CameraExposure) stands alone.
[[vk::binding(5, 0)]]
StructuredBuffer<float> auto_exposure;

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    // Bloom composited in scene-referred linear, before exposure. A bounded
    // mix (the Jimenez/CoD composite), not an add: lerp can't blow the image
    // out the way `scene + bloom` does when the sky is very bright, and it is
    // a no-op when bloom is off (intensity 0, black input). exposure.y = 0
    // when the camera has no CameraBloom.
    float3 scene = scene_color.Sample(scene_sampler, input.uv).rgb;
    float3 glow = bloom.Sample(bloom_sampler, input.uv).rgb;
    // Manual exposure (exposure.x) times the auto-exposure multiplier (1.0 when
    // the camera has no CameraAutoExposure, #153). When auto is on, exposure.x
    // carries only the compensation bias so the metered value drives.
    float exposure_mul = exposure.x * auto_exposure[0];
    float3 result_color = lerp(scene, glow, exposure.y) * exposure_mul;

#ifdef HDR_OUTPUT
    // Linear-HDR (extended-sRGB-linear) target: roll highlights off to the
    // display's headroom H (exposure.z) instead of the old hard clamp at a
    // magic 10x SDR white. Paper-white (1.0) stays pinned; highlights extend up
    // to H. Output stays linear (the compositor decodes the extended-linear
    // surface — encoding here would double up). At H=1 this is the SDR PBR
    // Neutral shoulder, so an SDR-headroom EDR panel matches the SDR path.
    result_color = tonemap_pbr_neutral_headroom(result_color, exposure.z);
#else
    // SDR target: tonemap (Khronos PBR Neutral — hue-preserving, so
    // saturated albedos don't wash toward white); gamma-encode manually only
    // when the target is not sRGB-typed (an sRGB texture encodes on write by
    // itself).
    result_color = tonemap_pbr_neutral(result_color);
#ifndef SRGB_FRAMEBUFFER
    // Exact sRGB OETF — the same transfer the hardware applies on sRGB
    // targets, so both SDR paths encode identically.
    result_color = linear_to_srgb(result_color);
#endif
#endif

    return float4(result_color, 1.0);
}
