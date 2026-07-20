// Motion blur — reconstruction (#149; McGuire 2012 + Guertin 2014).
//
// Gathers each pixel's smear along its tile's dominant motion (NeighborMax).
// For every tap along that line the weight asks "could this surface plausibly
// have swept across me during the shutter?" — a depth classification (which of
// the two is in front) times a velocity-reach falloff (cone/cylinder). This is
// what lets a fast foreground object blur over static background while a static
// pixel behind it still receives the streak. Taps are dithered per pixel to
// break banding.
//
// Runs full-resolution, after TAA; the colour input is whatever the display
// pass would otherwise read (scene_color, or the TAA history just written).

[[vk::binding(0, 0)]]
cbuffer MbReconstructUniforms {
    float2 inv_resolution;
    float2 resolution;
    // Camera world position (xyz) for a linear depth proxy.
    float4 camera_pos;
    float shutter;
    float tile_size;
    // Reconstruction tap count.
    float samples;
    float _pad;
};

[[vk::binding(1, 0)]]
Texture2D color;
[[vk::binding(2, 0)]]
Texture2D neighbor_max;
[[vk::binding(3, 0)]]
Texture2D gbuffer_velocity;
[[vk::binding(4, 0)]]
Texture2D gbuffer_position;
[[vk::binding(5, 0)]]
Texture2D gbuffer_albedo;
[[vk::binding(6, 0)]]
SamplerState nearest_sampler;
[[vk::binding(7, 0)]]
SamplerState linear_sampler;

// Soft depth-compare extent, as a fraction of the compared depth
// (scale-invariant). Ghosting-vs-halo knob at silhouettes.
static const float SOFT_Z_REL = 0.05;
// Background depth sentinel — sky sits behind everything.
static const float FAR_DEPTH = 1e7;

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    VsOutput output;
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    float2 uvs[3] = { float2(0.0, 1.0), float2(2.0, 1.0), float2(0.0, -1.0) };
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

// Per-pixel screen velocity in pixels (shutter applied), matching TileMax.
float2 velocity_px(float2 uv) {
    float2 v_ndc = gbuffer_velocity.Sample(nearest_sampler, uv).xy;
    return v_ndc * float2(0.5, -0.5) * resolution * shutter;
}

// Linear depth proxy: distance to camera; background sits at the far sentinel.
float view_depth(float2 uv) {
    if (gbuffer_albedo.Sample(nearest_sampler, uv).a < 1.0 / 128.0) {
        return FAR_DEPTH;
    }
    float3 world = gbuffer_position.Sample(nearest_sampler, uv).xyz;
    return length(camera_pos.xyz - world);
}

// 1 when `a` is in front of `b` (closer), softly.
float soft_z(float a, float b) {
    return saturate(1.0 - (a - b) / max(SOFT_Z_REL * max(a, b), 1e-5));
}

// Linear reach falloff: full at the sample, zero past the velocity's extent.
float cone(float dist, float v) {
    return saturate(1.0 - dist / max(v, 1e-5));
}

// ~1 inside the velocity's extent, rolling off at the edge.
float cylinder(float dist, float v) {
    return 1.0 - smoothstep(0.95 * v, 1.05 * v, dist);
}

// Interleaved gradient noise in [0, 1) — cheap per-pixel dither.
float ign(float2 p) {
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float2 uv = input.uv;
    float3 center_color = color.Sample(linear_sampler, uv).rgb;

    float2 v_max = neighbor_max.Sample(nearest_sampler, uv).xy;
    float v_max_len = length(v_max);
    // Nothing in this neighbourhood moves enough to blur.
    if (v_max_len < 0.5) {
        return float4(center_color, 1.0);
    }

    float2 v_center = velocity_px(uv);
    float v_center_len = max(length(v_center), 0.5);
    float center_depth = view_depth(uv);

    int s = max(int(samples), 1);
    float jitter = ign(input.position.xy) - 0.5;

    // Centre tap anchors the result so a still pixel keeps its own colour.
    float weight = 1.0 / v_center_len;
    float3 sum = center_color * weight;

    for (int i = 0; i < s; i++) {
        // Even spread over [-1, 1] around the centre, dithered.
        float t = lerp(-1.0, 1.0, (float(i) + jitter + 1.0) / (float(s) + 1.0));
        float2 offset = v_max * (0.5 * t);
        float dist = length(offset);
        float2 s_uv = uv + offset * inv_resolution;

        float3 s_color = color.Sample(linear_sampler, s_uv).rgb;
        float2 s_vel = velocity_px(s_uv);
        float s_vel_len = max(length(s_vel), 0.5);
        float s_depth = view_depth(s_uv);

        // f: centre in front of sample; b: sample in front of centre.
        float f = soft_z(center_depth, s_depth);
        float b = soft_z(s_depth, center_depth);
        float w = f * cone(dist, s_vel_len)          // moving background sample
                + b * cone(dist, v_center_len)        // foreground streaks over
                + cylinder(dist, s_vel_len) * cylinder(dist, v_center_len) * 2.0; // co-planar

        weight += w;
        sum += s_color * w;
    }

    return float4(sum / max(weight, 1e-5), 1.0);
}
