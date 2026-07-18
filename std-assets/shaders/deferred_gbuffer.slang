// Standard deferred G-buffer shader (#144) — writes the `pbr` shading
// model's material properties to MRT for the deferred resolve pass.
//
// Binding sets are ParameterBlocks classified by update frequency (the
// [UpdateRate] attribute from the engine library module — see
// docs/MATERIAL_ASSETS.md Decision 7). Declaration order fixes the set
// indices: 0 camera (external), 1 model (dynamic), 2 material (static).
import engine;

struct CameraParams {
    column_major float4x4 view_projection;
};

struct ModelParams {
    column_major float4x4 model;
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
    float3 world_position : TEXCOORD0;
    float3 world_normal : NORMAL;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    float4 world_pos = mul(gModel.model, float4(input.position, 1.0));
    output.clip_position = mul(gCamera.view_projection, world_pos);
    output.world_position = world_pos.xyz;
    output.world_normal = mul(gModel.model, float4(input.normal, 0.0)).xyz;
    return output;
}

struct FsOutput {
    // RT0: albedo (rgb) + geometry marker (a = 1; the resolve treats
    // a < 0.5 as background — the pass clears alpha to 0).
    float4 albedo : SV_Target0;
    // RT1: world normal encoded to [0,1] (rgb) + metallic (a).
    float4 normal_metallic : SV_Target1;
    // RT2: world position (rgb) + roughness (a).
    float4 position_roughness : SV_Target2;
};

[shader("fragment")]
FsOutput fs_main(VsOutput input) {
    FsOutput output;
    float3 normal = normalize(input.world_normal);
    float metallic = saturate(gMaterial.pbr_params.x);
    float roughness = max(gMaterial.pbr_params.y, 0.04);
    output.albedo = float4(gMaterial.base_color.rgb, 1.0);
    output.normal_metallic = float4(normal * 0.5 + 0.5, metallic);
    output.position_roughness = float4(input.world_position, roughness);
    return output;
}
