// Standard textured opaque shader — Blinn-Phong lighting with
// position + normal + uv vertex layout; base color texture times base color.
//
// Binding group 0: Per-entity transform uniforms (VP + model matrices).
// Binding group 1: Material properties — packed uniforms at binding 0, then
// texture/sampler pairs per texture property in schema order.

[[vk::binding(0, 0)]]
cbuffer Uniforms {
    column_major float4x4 view_projection;
    column_major float4x4 model;
};

[[vk::binding(0, 1)]]
cbuffer MaterialProps {
    float4 base_color;
};

[[vk::binding(1, 1)]]
Texture2D base_texture;

[[vk::binding(2, 1)]]
SamplerState base_texture_sampler;

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
};

struct VsOutput {
    float4 clip_position : SV_Position;
    float3 world_normal : NORMAL;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    float4 world_pos = mul(model, float4(input.position, 1.0));
    output.clip_position = mul(view_projection, world_pos);
    output.world_normal = mul(model, float4(input.normal, 0.0)).xyz;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float3 light_dir = normalize(float3(0.5, 1.0, 0.3));
    float3 n = normalize(input.world_normal);
    float ndotl = max(dot(n, light_dir), 0.0);
    float4 albedo = base_texture.Sample(base_texture_sampler, input.uv) * base_color;
    float3 ambient = float3(0.15, 0.15, 0.18);
    float3 color = ambient * albedo.rgb + albedo.rgb * ndotl;
    return float4(color, albedo.a);
}
