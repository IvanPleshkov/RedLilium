// Demo/reference shader for the two newer material-property kinds a plain
// scalar/vector/texture schema can't express (docs/MATERIAL_ASSETS.md):
//   - a `Texture2DArray` layered texture bound as a material property, and
//   - a read-only `StructuredBuffer` storage buffer bound as a material
//     property.
// Forward, single color target, position + normal + uv vertex layout. Backed by
// the engine `array_storage_demo` shading model, whose schema order matches the
// MaterialParams field order below (uniform, then array texture, then buffer).
//
// Binding sets are ParameterBlocks classified by update frequency (the
// [UpdateRate] attribute from the engine library module — see
// docs/MATERIAL_ASSETS.md Decision 7): 0 camera (external), 1 model (dynamic),
// 2 material (static). Within the material block: uniform fields pack into the
// implicit constant buffer at binding 0; each opaque field (the array texture,
// its sampler, then the structured buffer) takes the next descriptor slot — the
// same order the engine's instance manager builds the static group in.
import engine;

struct CameraParams {
    column_major float4x4 view_projection;
};

struct ModelParams {
    column_major float4x4 model;
};

struct MaterialParams {
    // Uniform field → constant buffer at binding 0.
    float4 base_color;
    // Opaque fields → consecutive descriptor slots (1: array, 2: sampler,
    // 3: structured buffer).
    Texture2DArray layers;
    SamplerState layers_sampler;
    StructuredBuffer<float4> tints;
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
    // Sample layer 0 of the array texture, tinted by the first structured-buffer
    // entry and the base color factor.
    float4 layer = gMaterial.layers.Sample(gMaterial.layers_sampler, float3(input.uv, 0.0));
    float4 albedo = layer * gMaterial.tints[0] * gMaterial.base_color;
    float3 ambient = float3(0.15, 0.15, 0.18);
    float3 color = ambient * albedo.rgb + albedo.rgb * ndotl;
    return float4(color, albedo.a);
}
