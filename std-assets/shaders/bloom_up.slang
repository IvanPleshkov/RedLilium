// Bloom upsample (#151) — the Jimenez / CoD:AW 3x3 tent filter.
//
// One step of the progressive upsample: reads the smaller (already-accumulated)
// mip and writes it, tent-filtered, into the next larger mip. The pass is drawn
// with OVER (alpha) blending and the shader outputs alpha = BLOOM_MIX, so the
// blend computes lerp(largerMip, tent(smallerMip), BLOOM_MIX). This *blended*
// accumulation (rather than a raw add) keeps the bloom energy-normalized —
// bloom stays about scene magnitude instead of growing ~N x brighter with the
// mip count, so a uniformly bright region does not self-brighten into a haze;
// only genuine highlights (where the blurred bloom exceeds the local scene)
// glow. The geometric weights (1/2, 1/4, 1/8, ...) sum to 1.
static const float BLOOM_MIX = 0.5;

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    VsOutput output;
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };
    float2 uvs[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

[[vk::binding(0, 0)]]
cbuffer BloomUniforms {
    // xy = source (smaller mip) texel size (1 / source extent); zw unused.
    float4 params;
};

[[vk::binding(1, 0)]]
Texture2D source;
[[vk::binding(2, 0)]]
SamplerState linear_sampler;

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float2 uv = input.uv;
    float2 t = params.xy;

    // 3x3 tent (weights 1 2 1 / 2 4 2 / 1 2 1, /16), offsets in source texels.
    float3 result = source.Sample(linear_sampler, uv + t * float2(-1.0, -1.0)).rgb * 1.0;
    result += source.Sample(linear_sampler, uv + t * float2(0.0, -1.0)).rgb * 2.0;
    result += source.Sample(linear_sampler, uv + t * float2(1.0, -1.0)).rgb * 1.0;
    result += source.Sample(linear_sampler, uv + t * float2(-1.0, 0.0)).rgb * 2.0;
    result += source.Sample(linear_sampler, uv + t * float2(0.0, 0.0)).rgb * 4.0;
    result += source.Sample(linear_sampler, uv + t * float2(1.0, 0.0)).rgb * 2.0;
    result += source.Sample(linear_sampler, uv + t * float2(-1.0, 1.0)).rgb * 1.0;
    result += source.Sample(linear_sampler, uv + t * float2(0.0, 1.0)).rgb * 2.0;
    result += source.Sample(linear_sampler, uv + t * float2(1.0, 1.0)).rgb * 1.0;
    result *= 1.0 / 16.0;

    // alpha = BLOOM_MIX drives the OVER blend into lerp(dst, result, BLOOM_MIX).
    return float4(result, BLOOM_MIX);
}
