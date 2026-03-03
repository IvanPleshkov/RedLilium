// Standard opaque color shader — Blinn-Phong lighting with position + normal vertex layout.
//
// Binding group 0: Per-entity transform uniforms (VP + model matrices).
// Binding group 1: Material property uniforms (base_color).

[[vk::binding(0, 0)]]
cbuffer Uniforms {
    column_major float4x4 view_projection;
    column_major float4x4 model;
};

[[vk::binding(0, 1)]]
cbuffer MaterialProps {
    float4 base_color;
};

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct VsOutput {
    float4 clip_position : SV_Position;
    float3 world_normal : NORMAL;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    float4 world_pos = mul(model, float4(input.position, 1.0));
    output.clip_position = mul(view_projection, world_pos);
    output.world_normal = mul(model, float4(input.normal, 0.0)).xyz;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float3 light_dir = normalize(float3(0.5, 1.0, 0.3));
    float3 n = normalize(input.world_normal);
    float ndotl = max(dot(n, light_dir), 0.0);
    float3 bc = base_color.rgb;
    float3 ambient = float3(0.15, 0.15, 0.18);
    float3 color = ambient + bc * ndotl;
    return float4(color, 1.0);
}
