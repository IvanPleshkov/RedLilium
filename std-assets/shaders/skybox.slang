// Standard skybox pass (#144) — renders the environment cubemap as the
// scene background, before the deferred resolve composites lit geometry
// over it.
//
// Output is scene-referred linear radiance (1.0 = SDR white) into the
// `scene_color` intermediate (#142) — exposure, tonemap, and display
// encoding all live in display_output.slang.

import math;

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

    // Compute view direction from clip space. Deliberately UNNORMALIZED: the
    // unprojected plane point is an affine function of NDC, so its linear
    // interpolation across the triangle is exact per pixel. Normalizing here
    // would weight the vertex blend by ray length and warp the whole sky
    // (>20 degrees off at screen center for a 90-degree FOV); cube sampling
    // does not need a unit vector.
    float4 clip_pos = float4(x, y, 1.0, 1.0);
    float4 world_pos = mul(inv_view_proj, clip_pos);
    output.view_dir = world_pos.xyz / world_pos.w - camera_pos.xyz;
    return output;
}

// =============================================================================
// Fragment Shader
// =============================================================================

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    // Scene-referred linear out — display transform happens downstream.
    // Sanitize: an environment cubemap can carry a non-finite / f16-overflowing
    // texel at the baked sun hot-spot (the IBL bake flags it as near-singular),
    // which would otherwise land as Inf/NaN in scene_color and tonemap into a
    // black or colored block on the sun (#152). Clamp at the write so every
    // scene_color consumer (bloom, TAA, the display composite) sees finite data.
    float3 color = env_map.SampleLevel(env_sampler, input.view_dir, mip_level).rgb;
    return float4(sanitize_finite3(color), 1.0);
}
