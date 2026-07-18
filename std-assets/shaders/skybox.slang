// Standard skybox pass (#144) — renders the environment cubemap as the
// scene background, before the deferred resolve composites lit geometry
// over it.
//
// HDR_OUTPUT gates the output transform exactly like deferred_resolve.slang:
// off = tonemap + gamma encode for an sRGB-ish SDR target, on = raw linear
// color for a linear-HDR target (1.0 = SDR white) — encoding there would be
// decoded a second time by the display and wash the background out.
//#pragma variant_system HDR_OUTPUT

import color;

// =============================================================================
// Bindings
// =============================================================================

[[vk::binding(0, 0)]]
cbuffer SkyboxUniforms {
    column_major float4x4 inv_view_proj;
    float4 camera_pos;
    float mip_level;
    float3 _pad;
};

[[vk::binding(1, 0)]]
TextureCube env_map;
[[vk::binding(2, 0)]]
SamplerState env_sampler;

// =============================================================================
// Vertex Shader
// =============================================================================

struct VsOutput {
    float4 position : SV_Position;
    float3 view_dir : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    VsOutput output;

    // Fullscreen triangle
    float x = float((vertex_id & 1) << 2) - 1.0;
    float y = float((vertex_id & 2) << 1) - 1.0;

    // Arbitrary in-range clip depth: the skybox pass has no depth attachment
    // (the resolve pass composites lit geometry over this background), so the
    // value is depth-convention independent (#125/ADR-038).
    output.position = float4(x, y, 0.9999, 1.0);

    // Compute view direction from clip space
    float4 clip_pos = float4(x, y, 1.0, 1.0);
    float4 world_pos = mul(inv_view_proj, clip_pos);
    output.view_dir = normalize(world_pos.xyz / world_pos.w - camera_pos.xyz);
    return output;
}

// =============================================================================
// Fragment Shader
// =============================================================================

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float3 color = env_map.SampleLevel(env_sampler, input.view_dir, mip_level).rgb;

#ifdef HDR_OUTPUT
    // Linear-HDR target: pass radiance through (same clamp as the resolve).
    float3 result = clamp(color, float3(0.0), float3(10.0));
#else
    // SDR target: tonemap and gamma correct using library functions
    float3 result = gamma_correct(tonemap_reinhard(color));
#endif

    return float4(result, 1.0);
}
