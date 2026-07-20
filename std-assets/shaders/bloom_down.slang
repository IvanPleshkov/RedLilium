// Bloom downsample (#151) — the Jimenez / CoD:AW 13-tap dual filter.
//
// One step of the progressive downsample: reads the source (the scene-referred
// image for mip 0, the previous bloom mip afterwards) and writes the next,
// half-size mip. Thirteen bilinear taps are combined as five overlapping 2x2
// boxes so the result is smooth and stable (a plain box would alias).
//
// KARIS variant (mip 0 only): each box is weighted by 1/(1 + luma) before the
// combine — the partial Karis average that stops a single fireflies-bright
// pixel from flickering the whole glow. Later mips are already smooth, so they
// use the plain weights.
//#pragma variant_system KARIS

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
    // xy = source texel size (1 / source extent); zw unused here.
    float4 params;
};

[[vk::binding(1, 0)]]
Texture2D source;
[[vk::binding(2, 0)]]
SamplerState linear_sampler;

// Sanitize bloom taps to a finite range. A very bright sky/sun can exceed the
// f16 range and land as +Inf (or a NaN from upstream) in scene_color; the
// Karis weight 1/(1+luma) then hits 1/Inf = 0 and Inf*0 = NaN, which spreads a
// black block up the mip chain. `min()`/`clamp()` NaN semantics differ across
// backends (WGSL returns the other operand, SPIR-V FMin may propagate), so
// kill NaN explicitly via the self-inequality (NaN != NaN) and clamp Inf to
// the f16 max — every tap comes out finite regardless of backend.
static const float BLOOM_CLAMP = 65504.0;

float sanitize(float x) {
    // x == x is false only for NaN; clamp is well-defined once NaN is excluded.
    return (x == x) ? clamp(x, 0.0, BLOOM_CLAMP) : 0.0;
}

float3 fetch(float2 uv) {
    float3 c = source.Sample(linear_sampler, uv).rgb;
    return float3(sanitize(c.x), sanitize(c.y), sanitize(c.z));
}

float luma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

#ifdef KARIS
// Karis-weighted average of a 2x2 box (partial: weights by 1/(1+luma)).
float3 karis_box(float3 a, float3 b, float3 c, float3 d) {
    float wa = 1.0 / (1.0 + luma(a));
    float wb = 1.0 / (1.0 + luma(b));
    float wc = 1.0 / (1.0 + luma(c));
    float wd = 1.0 / (1.0 + luma(d));
    return (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
}
#endif

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float2 uv = input.uv;
    float2 t = params.xy;

    // 13 taps: a b c / j k / d e f / l m / g h i (offsets in source texels).
    float3 a = fetch(uv + t * float2(-2.0, -2.0));
    float3 b = fetch(uv + t * float2(0.0, -2.0));
    float3 c = fetch(uv + t * float2(2.0, -2.0));
    float3 d = fetch(uv + t * float2(-2.0, 0.0));
    float3 e = fetch(uv + t * float2(0.0, 0.0));
    float3 f = fetch(uv + t * float2(2.0, 0.0));
    float3 g = fetch(uv + t * float2(-2.0, 2.0));
    float3 h = fetch(uv + t * float2(0.0, 2.0));
    float3 i = fetch(uv + t * float2(2.0, 2.0));
    float3 j = fetch(uv + t * float2(-1.0, -1.0));
    float3 k = fetch(uv + t * float2(1.0, -1.0));
    float3 l = fetch(uv + t * float2(-1.0, 1.0));
    float3 m = fetch(uv + t * float2(1.0, 1.0));

#ifdef KARIS
    // Five boxes, Karis-averaged, with the dual-filter box weights (center
    // 0.5, the four corners 0.125 each).
    float3 b0 = karis_box(a, b, d, e); // top-left
    float3 b1 = karis_box(b, c, e, f); // top-right
    float3 b2 = karis_box(d, e, g, h); // bottom-left
    float3 b3 = karis_box(e, f, h, i); // bottom-right
    float3 b4 = karis_box(j, k, l, m); // center
    float3 result = b4 * 0.5 + (b0 + b1 + b2 + b3) * 0.125;
#else
    // Fixed dual-filter weights (sum to 1): center 0.125, edges 0.0625,
    // corners 0.03125, inner box 0.125 each.
    float3 result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;
#endif

    return float4(result, 1.0);
}
