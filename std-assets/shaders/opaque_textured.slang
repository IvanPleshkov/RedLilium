// Standard textured opaque shader — Blinn-Phong lighting with
// position + normal + uv vertex layout; base color texture times base color.
//
// Binding sets are ParameterBlocks classified by update frequency (the
// [UpdateRate] attribute from the engine library module — see
// docs/MATERIAL_ASSETS.md Decision 7). Declaration order fixes the set
// indices: 0 camera (external), 1 model (dynamic), 2 material (static).
// Within the material block: uniform fields pack into the implicit constant
// buffer at binding 0; each texture/sampler gets the next descriptor slot —
// the same order the engine's instance manager builds the static group in.
import engine;

struct CameraParams {
    column_major float4x4 view_projection;
};

struct ModelParams {
    column_major float4x4 model;
};

struct MaterialParams {
    float4 base_color;
    Texture2D base_texture;
    SamplerState base_texture_sampler;
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
    float4 world_pos = mul(gModel.model, float4(input.position, 1.0));
    output.clip_position = mul(gCamera.view_projection, world_pos);
    output.world_normal = mul(gModel.model, float4(input.normal, 0.0)).xyz;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float3 light_dir = normalize(float3(0.5, 1.0, 0.3));
    float3 n = normalize(input.world_normal);
    float ndotl = max(dot(n, light_dir), 0.0);
    float4 albedo =
        gMaterial.base_texture.Sample(gMaterial.base_texture_sampler, input.uv)
        * gMaterial.base_color;
    float3 ambient = float3(0.15, 0.15, 0.18);
    float3 color = ambient * albedo.rgb + albedo.rgb * ndotl;
    return float4(color, albedo.a);
}
