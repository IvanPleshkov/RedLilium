// Standard textured deferred G-buffer shader (ADR-039) — the `pbr_textured`
// shading model: glTF metallic-roughness convention, base color (sRGB) and
// packed ORM (R = AO, G = roughness, B = metallic) textures multiplying the
// factor properties. Needs a UV-carrying vertex layout.
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
    // Factor properties (glTF semantics: multiplied with the textures).
    float4 base_color;
    // x: metallic factor, y: roughness factor (z, w reserved).
    float4 pbr_params;
    Texture2D base_color_texture;
    SamplerState base_color_texture_sampler;
    // R = ambient occlusion, G = roughness, B = metallic (linear).
    Texture2D orm_texture;
    SamplerState orm_texture_sampler;
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
    float2 uv : TEXCOORD1;
    // Unjittered clip positions of this and the previous frame — the
    // fragment stage derives NDC velocity from the pair (#147).
    float4 curr_clip : TEXCOORD2;
    float4 prev_clip : TEXCOORD3;
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
    output.uv = input.uv;
    output.curr_clip = mul(gCamera.view_projection_unjittered, world_pos);
    float4 prev_world = mul(gModel.prev_model, float4(input.position, 1.0));
    output.prev_clip = mul(gCamera.prev_view_projection, prev_world);
    return output;
}

struct FsOutput {
    // RT0: albedo (rgb) + AO-as-geometry-marker (a): 0 = background (the
    // clear value), otherwise max(AO, 1/64) — the resolve multiplies the
    // IBL ambient by it. Alpha is never sRGB-encoded, so the value stays
    // linear (ADR-039).
    float4 albedo : SV_Target0;
    // RT1: world normal encoded to [0,1] (rgb) + metallic (a).
    float4 normal_metallic : SV_Target1;
    // RT2: roughness (R8Unorm). World position is NOT stored — consumers
    // reconstruct it from the depth buffer (full 32-bit precision everywhere).
    float roughness : SV_Target2;
    // RT3: NDC motion current -> previous frame, both unjittered (#147).
    float2 velocity : SV_Target3;
};

[shader("fragment")]
FsOutput fs_main(VsOutput input) {
    FsOutput output;
    float3 normal = normalize(input.world_normal);

    float4 base_sample =
        gMaterial.base_color_texture.Sample(gMaterial.base_color_texture_sampler, input.uv);
    float3 orm = gMaterial.orm_texture.Sample(gMaterial.orm_texture_sampler, input.uv).rgb;

    float3 albedo = gMaterial.base_color.rgb * base_sample.rgb;
    float ao = orm.r;
    float metallic = saturate(gMaterial.pbr_params.x * orm.b);
    float roughness = max(gMaterial.pbr_params.y * orm.g, 0.04);

    output.albedo = float4(albedo, max(ao, 1.0 / 64.0));
    output.normal_metallic = float4(normal * 0.5 + 0.5, metallic);
    output.roughness = roughness;
    output.velocity = input.curr_clip.xy / input.curr_clip.w
        - input.prev_clip.xy / input.prev_clip.w;
    return output;
}
