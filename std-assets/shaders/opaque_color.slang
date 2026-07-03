// Standard opaque color shader — Blinn-Phong lighting with position + normal
// vertex layout.
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
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    float4 world_pos = mul(gModel.model, float4(input.position, 1.0));
    output.clip_position = mul(gCamera.view_projection, world_pos);
    output.world_normal = mul(gModel.model, float4(input.normal, 0.0)).xyz;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float3 light_dir = normalize(float3(0.5, 1.0, 0.3));
    float3 n = normalize(input.world_normal);
    float ndotl = max(dot(n, light_dir), 0.0);
    float3 bc = gMaterial.base_color.rgb;
    float3 ambient = float3(0.15, 0.15, 0.18);
    float3 color = ambient + bc * ndotl;
    return float4(color, 1.0);
}
