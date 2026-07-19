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
//   HDR_OUTPUT       - raw linear for a linear-HDR target (extended range,
//                      clamped to 10x SDR white)
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
    float4 exposure; // x = linear exposure multiplier (CameraExposure, #142)
};

[[vk::binding(1, 0)]]
Texture2D scene_color;
[[vk::binding(2, 0)]]
SamplerState scene_sampler;

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float3 result_color = scene_color.Sample(scene_sampler, input.uv).rgb * exposure.x;

#ifdef HDR_OUTPUT
    // Linear-HDR target: pass radiance through (encoding here would be
    // decoded a second time by the display). 1.0 = SDR white; clamp bounds
    // the extended range.
    result_color = clamp(result_color, float3(0.0), float3(10.0));
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
