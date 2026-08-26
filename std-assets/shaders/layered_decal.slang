// Layered decal shader — baked-decal channel (procedural: design/decals-design.md).
// `opaque_textured` plus up to LAYERS baked decal albedo layers composited over the
// base albedo in declaration order. Each layer samples its own texture through a
// per-layer affine UV transform (surface-UV → decal-UV) and alpha-composites `Over`
// the accumulator, so an overlap region carrying an ordered layer-set renders as one
// opaque surface — no transparency pass, no unique-UV atlas. The "2-3 overlaps"
// guarantee is this shader's layer cap (LAYERS).
//
// A region covered by fewer layers leaves the unused slots at their white 1×1 /
// zero-strength defaults (see ShadingRegistry::layered_decal), so it renders exactly
// like the base material.
//
// Binding-set convention matches opaque_textured: set 0 camera (external), 1 model
// (dynamic), 2 material (static). Within the material block, uniforms pack into the
// implicit cbuffer at binding 0, then each texture/sampler takes the next descriptor
// slot in DECLARATION order — the ShadingRegistry schema lists the props in exactly
// this order.
import engine;

struct CameraParams {
    column_major float4x4 view_projection;
};

struct ModelParams {
    column_major float4x4 model;
};

// Field order fixes both the cbuffer byte layout AND the ShadingRegistry schema
// order — keep the two in lockstep. Per-layer fields are named (not arrays) so the
// instance manager's by-name override path (the viewer) maps 1:1 to a slot.
//   layerN_uv     = (offset.xy, scale.xy): decal_uv = surface_uv * scale + offset.
//   layerN_params = (strength, _, _, _): strength 0 disables the layer, so the white
//                   default texture never tints the base.
struct MaterialParams {
    float4 base_color;
    float4 layer0_uv;
    float4 layer0_params;
    float4 layer1_uv;
    float4 layer1_params;
    float4 layer2_uv;
    float4 layer2_params;
    Texture2D base_texture;
    SamplerState base_texture_sampler;
    Texture2D layer0_texture;
    SamplerState layer0_texture_sampler;
    Texture2D layer1_texture;
    SamplerState layer1_texture_sampler;
    Texture2D layer2_texture;
    SamplerState layer2_texture_sampler;
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

// Sample one decal layer and composite it `Over` the running albedo. `tex`/`smp` are
// the layer's texture and sampler; `xform` is (offset.xy, scale.xy); `strength` in
// `params.x` scales the layer alpha (0 => the layer contributes nothing).
float3 composite_layer(
    float3 albedo, float2 surface_uv,
    Texture2D tex, SamplerState smp, float4 xform, float4 params
) {
    float2 decal_uv = surface_uv * xform.zw + xform.xy;
    // Outside [0,1] the projector doesn't cover this fragment — clamp to zero
    // contribution rather than relying on wrap mode.
    float2 inside = step(float2(0.0, 0.0), decal_uv) * step(decal_uv, float2(1.0, 1.0));
    float4 c = tex.Sample(smp, decal_uv);
    float a = c.a * params.x * inside.x * inside.y;
    return lerp(albedo, c.rgb, a);
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float3 light_dir = normalize(float3(0.5, 1.0, 0.3));
    float3 n = normalize(input.world_normal);
    float ndotl = max(dot(n, light_dir), 0.0);

    float4 base =
        gMaterial.base_texture.Sample(gMaterial.base_texture_sampler, input.uv)
        * gMaterial.base_color;
    float3 albedo = base.rgb;

    albedo = composite_layer(albedo, input.uv,
        gMaterial.layer0_texture, gMaterial.layer0_texture_sampler,
        gMaterial.layer0_uv, gMaterial.layer0_params);
    albedo = composite_layer(albedo, input.uv,
        gMaterial.layer1_texture, gMaterial.layer1_texture_sampler,
        gMaterial.layer1_uv, gMaterial.layer1_params);
    albedo = composite_layer(albedo, input.uv,
        gMaterial.layer2_texture, gMaterial.layer2_texture_sampler,
        gMaterial.layer2_uv, gMaterial.layer2_params);

    float3 ambient = float3(0.15, 0.15, 0.18);
    float3 color = ambient * albedo + albedo * ndotl;
    return float4(color, base.a);
}
