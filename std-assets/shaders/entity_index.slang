// Entity index shader — MRT: entity index (R32Uint) for picking, the
// selection mask (R8Unorm) for the editor's selection outline, and the raw
// reversed-Z depth (R32Float) for reconstructing the pick's world-space point.
//
// Writes (entity_index + 1) so that a cleared-to-zero texture means "no entity".
// The mask receives 1 for selected entities and 0 otherwise — unselected
// occluders overwrite it under the shared depth test, so the mask holds the
// *visible* part of the selected silhouette. The depth target clears to 0
// (reversed-Z background), so any positive readback is a real surface hit.
//
// Binding 0: Uniforms { view_projection, model, entity_index, selected } —
// per-entity uniform buffer.

[[vk::binding(0, 0)]]
cbuffer Uniforms {
    column_major float4x4 view_projection;
    column_major float4x4 model;
    uint entity_index;
    uint selected;
};

// Position only — location 0 maps to the mesh layout's FIRST attribute (both
// backends number shader locations sequentially in layout attribute order,
// ADR-019), so the material must be specialized per mesh vertex layout and
// only layouts whose attribute 0 is the position are drawable.
struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
};

[shader("vertex")]
float4 vs_main(VsInput input) : SV_Position {
    float4 world_pos = mul(model, float4(input.position, 1.0));
    return mul(view_projection, world_pos);
}

struct FsOutput {
    uint index : SV_Target0;
    float mask : SV_Target1;
    float depth : SV_Target2;
};

[shader("fragment")]
FsOutput fs_main(float4 frag_coord : SV_Position) {
    FsOutput output;
    output.index = entity_index + 1u;
    output.mask = selected != 0u ? 1.0 : 0.0;
    // Raw [0,1] reversed-Z depth of the picked surface — read back with the
    // pick and unprojected to the exact world-space hit point.
    output.depth = frag_coord.z;
    return output;
}
