// Screen-space ambient occlusion (#150) — GTAO-lite, horizon-based.
//
// A fullscreen pass between the G-buffer and the deferred resolve. For each
// shaded pixel it estimates how much of the hemisphere above the surface is
// blocked by nearby geometry, from the G-buffer world position (RT2) and
// world normal (RT1). The result (1 = open, 0 = fully occluded) lands in a
// single-channel target that `ssao_blur` denoises and the resolve multiplies
// into the *image-based* ambient only (direct lights are shadowed by real
// shadows, #130 — never by this).
//
// Horizon-based / normal-aware: per screen-space direction it marches a few
// steps and keeps the highest a blocker rises above the surface tangent plane
// (`sin(elevation) = dot(sampleDir, N)`), then averages the per-direction
// horizons. A per-pixel interleaved-gradient rotation turns undersampling
// banding into high-frequency noise that `ssao_blur` (and TAA, when the camera
// has it) removes; that rotation only advances per frame when a temporal
// resolver is downstream (params.w != 0), so a non-temporal camera stays
// flicker-free.

import math;

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
cbuffer SsaoUniforms {
    column_major float4x4 view; // world -> view (rigid: upper 3x3 rotates normals)
    column_major float4x4 proj; // view -> clip (for the screen-space radius)
    // x = world radius, y = intensity, z = power, w = per-frame rotation
    // (radians; 0 when no temporal resolver, so the pattern is static).
    float4 params;
    float4 params2; // xy = texel size (1 / extent); zw unused
};

[[vk::binding(1, 0)]]
Texture2D gbuffer_albedo; // RT0: .a < 1/128 marks background (ADR-039)
[[vk::binding(2, 0)]]
Texture2D gbuffer_normal_metallic; // RT1: .rgb = encoded world normal
[[vk::binding(3, 0)]]
Texture2D gbuffer_position_roughness; // RT2: .rgb = world position
[[vk::binding(4, 0)]]
SamplerState point_sampler;

static const int DIR_COUNT = 6;
static const int STEP_COUNT = 4;
// Ignore near-tangent samples so a flat surface does not occlude itself.
static const float BIAS = 0.02;
// Cap the screen search so a near-camera pixel does not scan the whole image.
static const float MAX_RADIUS_UV = 0.25;

bool is_background(float2 uv) {
    return gbuffer_albedo.Sample(point_sampler, uv).a < 1.0 / 128.0;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float2 uv = input.uv;
    // Background carries no geometry — fully open (no darkening of the sky).
    if (is_background(uv)) {
        return float4(1.0, 0.0, 0.0, 1.0);
    }

    float3 pw = gbuffer_position_roughness.Sample(point_sampler, uv).rgb;
    float3 nw = normalize(gbuffer_normal_metallic.Sample(point_sampler, uv).rgb * 2.0 - 1.0);
    float3 pv = mul(view, float4(pw, 1.0)).xyz;
    float3 nv = normalize(mul((float3x3)view, nw));

    float radius = params.x;
    // Screen-space (uv) radius at this depth: project a horizontal world offset
    // of `radius` and measure the uv delta. Isotropic in uv — slightly
    // stretched on non-square targets (full-res v1; aspect correction rides the
    // half-res perf follow-up).
    float4 pc = mul(proj, float4(pv, 1.0));
    float4 rc = mul(proj, float4(pv + float3(radius, 0.0, 0.0), 1.0));
    float2 uv_c = pc.xy / pc.w * float2(0.5, -0.5) + 0.5;
    float2 uv_r = rc.xy / rc.w * float2(0.5, -0.5) + 0.5;
    float radius_uv = min(length(uv_r - uv_c), MAX_RADIUS_UV);
    if (radius_uv < params2.x) {
        return float4(1.0, 0.0, 0.0, 1.0); // sub-texel radius — nothing to sample
    }

    // Per-pixel rotation (interleaved gradient noise) + optional per-frame
    // advance for the temporal resolver to average over.
    float2 px = uv / params2.xy;
    float ign = frac(52.9829189 * frac(dot(px, float2(0.06711056, 0.00583715))));
    float base_rot = ign * TAU + params.w;

    float occlusion = 0.0;
    [unroll]
    for (int d = 0; d < DIR_COUNT; d++) {
        float angle = base_rot + (float(d) / float(DIR_COUNT)) * TAU;
        float2 dir = float2(cos(angle), sin(angle));
        float horizon = 0.0; // highest sin(elevation) of a blocker this slice
        [unroll]
        for (int s = 1; s <= STEP_COUNT; s++) {
            float t = float(s) / float(STEP_COUNT);
            float2 suv = uv + dir * radius_uv * t;
            if (any(suv < float2(0.0)) || any(suv > float2(1.0))) {
                continue;
            }
            if (is_background(suv)) {
                continue; // sky is not an occluder
            }
            float3 spw = gbuffer_position_roughness.Sample(point_sampler, suv).rgb;
            float3 dv = mul(view, float4(spw, 1.0)).xyz - pv;
            float dist = length(dv);
            if (dist < 1e-4) {
                continue;
            }
            float elev = dot(dv / dist, nv); // sin(elevation above tangent plane)
            float falloff = saturate(1.0 - dist / radius);
            horizon = max(horizon, (elev - BIAS) * falloff);
        }
        occlusion += saturate(horizon);
    }
    occlusion /= float(DIR_COUNT);

    float ao = saturate(1.0 - occlusion * params.y);
    ao = pow(ao, params.z);
    return float4(ao, 0.0, 0.0, 1.0);
}
