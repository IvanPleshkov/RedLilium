// PBR IBL Demo - Skybox Shader
// Renders environment cubemap as background with tone mapping and gamma correction.

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

    // Tonemap and gamma correct using library functions
    float3 mapped = tonemap_reinhard(color);
    float3 corrected = gamma_correct(mapped);

    return float4(corrected, 1.0);
}
