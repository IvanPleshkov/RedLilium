// Standard deferred G-buffer shader (#144) — writes the `pbr` shading
// model's material properties to MRT for the deferred resolve pass.
//
// Binding sets are ParameterBlocks classified by update frequency (the
// [UpdateRate] attribute from the engine library module — see
// docs/MATERIAL_ASSETS.md Decision 7). Declaration order fixes the set
// indices: 0 camera (external), 1 model (dynamic), 2 material (static).
import engine;

struct CameraParams {
    // Rasterization matrix — sub-pixel jittered when the camera runs a
    // temporal technique (#147).
    column_major float4x4 view_projection;
    // Unjittered pair for velocity: jitter in velocity would smear statics.
    column_major float4x4 view_projection_unjittered;
    column_major float4x4 prev_view_projection;
};

struct ModelParams {
    column_major float4x4 model;
    // Previous frame's model matrix (equals model on first visible frame).
    column_major float4x4 prev_model;
};

struct MaterialParams {
    float4 base_color;
    // x: metallic, y: roughness (z, w reserved).
    float4 pbr_params;
};

[UpdateRate("external")]
ParameterBlock<CameraParams> gCamera;

[UpdateRate("dynamic")]
ParameterBlock<ModelParams> gModel;

[UpdateRate("static")]
ParameterBlock<MaterialParams> gMaterial;

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct VsOutput {
    float4 clip_position : SV_Position;
    float3 world_normal : NORMAL;
    // Unjittered clip positions of this and the previous frame — the
    // fragment stage derives NDC velocity from the pair (#147).
    float4 curr_clip : TEXCOORD1;
    float4 prev_clip : TEXCOORD2;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    float4 world_pos = mul(gModel.model, float4(input.position, 1.0));
    output.clip_position = mul(gCamera.view_projection, world_pos);
    // Normal via the cofactor of the model's upper 3x3 (rows m[i]): equal to
    // the inverse-transpose up to a positive scale, so non-uniform scale bends
    // normals correctly without a CPU-side inverse. Normalized in the fragment
    // stage as before.
    float3x3 m = (float3x3)gModel.model;
    float3x3 cof = float3x3(
        cross(m[1], m[2]),
        cross(m[2], m[0]),
        cross(m[0], m[1]));
    output.world_normal = mul(cof, input.normal);
    output.curr_clip = mul(gCamera.view_projection_unjittered, world_pos);
    float4 prev_world = mul(gModel.prev_model, float4(input.position, 1.0));
    output.prev_clip = mul(gCamera.prev_view_projection, prev_world);
    return output;
}

struct FsOutput {
    // RT0: albedo (rgb) + AO-as-geometry-marker (a): the untextured model
    // has no AO map, so it writes a = 1 (full ambient); the resolve treats
    // a < 1/128 as background — the pass clears alpha to 0 (ADR-039).
    float4 albedo : SV_Target0;
    // RT1: world normal encoded to [0,1] (rgb) + metallic (a).
    float4 normal_metallic : SV_Target1;
    // RT2: roughness (R8Unorm). World position is NOT stored — consumers
    // reconstruct it from the depth buffer (full 32-bit precision everywhere;
    // an f16 position target loses ~0.5 units at |coord| ~ 1000).
    float roughness : SV_Target2;
    // RT3: NDC motion current -> previous frame, both unjittered (#147).
    float2 velocity : SV_Target3;
};

[shader("fragment")]
FsOutput fs_main(VsOutput input) {
    FsOutput output;
    float3 normal = normalize(input.world_normal);
    float metallic = saturate(gMaterial.pbr_params.x);
    float roughness = max(gMaterial.pbr_params.y, 0.04);
    output.albedo = float4(gMaterial.base_color.rgb, 1.0);
    output.normal_metallic = float4(normal * 0.5 + 0.5, metallic);
    output.roughness = roughness;
    output.velocity = input.curr_clip.xy / input.curr_clip.w
        - input.prev_clip.xy / input.prev_clip.w;
    return output;
}
