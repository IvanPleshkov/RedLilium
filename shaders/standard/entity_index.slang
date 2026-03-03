// Entity index shader — outputs entity index as u32 to an R32Uint target for picking.
//
// Writes (entity_index + 1) so that a cleared-to-zero texture means "no entity".
//
// Binding 0: Uniforms { view_projection, model, entity_index } — per-entity uniform buffer.

[[vk::binding(0, 0)]]
cbuffer Uniforms {
    column_major float4x4 view_projection;
    column_major float4x4 model;
    uint entity_index;
};

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

[shader("vertex")]
float4 vs_main(VsInput input) : SV_Position {
    float4 world_pos = mul(model, float4(input.position, 1.0));
    return mul(view_projection, world_pos);
}

[shader("fragment")]
uint fs_main() : SV_Target {
    return entity_index + 1u;
}
