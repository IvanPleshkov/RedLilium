struct Uniforms {
    float4x4 mvp;
    uint mode;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> uniforms;

[[vk::binding(1, 0)]]
Texture2D quad_texture;
[[vk::binding(2, 0)]]
SamplerState quad_sampler;

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
};

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    output.position = mul(uniforms.mvp, float4(input.position, 1.0));
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float4 texel = quad_texture.Sample(quad_sampler, input.uv);
    if (uniforms.mode == 1) {
        return float4(texel.rrr, 1.0);
    }
    if (uniforms.mode == 2) {
        float2 xy = texel.rg * 2.0 - 1.0;
        float z = sqrt(saturate(1.0 - dot(xy, xy)));
        float3 n = normalize(float3(xy, z));
        float3 l = normalize(float3(0.4, 0.5, 0.75));
        float shade = 0.08 + 0.92 * saturate(dot(n, l));
        return float4(shade, shade, shade, 1.0);
    }
    return texel;
}
