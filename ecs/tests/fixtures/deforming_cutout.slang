// Application-owned example of GBufferContract::V1. No engine shader GUID.
import engine;
struct CameraParams {
    column_major float4x4 view_projection;
    column_major float4x4 view_projection_unjittered;
    column_major float4x4 prev_view_projection;
};
struct ModelParams {
    column_major float4x4 model;
    column_major float4x4 prev_model;
};
struct MaterialParams {
    float4 tint;
    Texture2D image;
    SamplerState image_sampler;
};
[UpdateRate("external")] ParameterBlock<CameraParams> camera;
[UpdateRate("dynamic")] ParameterBlock<ModelParams> object;
[UpdateRate("static")] ParameterBlock<MaterialParams> surface;
struct Input {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 previous_position : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
};
struct Varying {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 current : TEXCOORD1;
    float4 previous : TEXCOORD2;
};
[shader("vertex")]
Varying vs_main(Input v) {
    Varying o;
    float4 world = mul(object.model, float4(v.position, 1));
    o.position = mul(camera.view_projection, world);
    o.current = mul(camera.view_projection_unjittered, world);
    o.previous = mul(camera.prev_view_projection, mul(object.prev_model, float4(v.previous_position, 1)));
    o.uv = v.uv;
    return o;
}
struct GBuffer {
    float4 color : SV_Target0;
    float4 normal_mode : SV_Target1;
    float roughness : SV_Target2;
    float2 velocity : SV_Target3;
};
[shader("fragment")]
GBuffer fs_main(Varying v) {
    float4 sampled = surface.image.Sample(surface.image_sampler, v.uv) * surface.tint;
    if (sampled.a < 0.5) discard;
    GBuffer o;
    o.color = float4(sampled.rgb, 1);
    o.normal_mode = float4(0.5, 0.5, 1, -1); // unlit, +Z world normal
    o.roughness = 1;
    o.velocity = v.current.xy / v.current.w - v.previous.xy / v.previous.w;
    return o;
}
