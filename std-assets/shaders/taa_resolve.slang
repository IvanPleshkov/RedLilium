// Temporal anti-aliasing resolve (#148) — accumulates the jittered
// scene-referred frames into a converged, anti-aliased image.
//
// Runs between the deferred resolve and the display-output pass, entirely
// scene-referred (pre-exposure linear, 1.0 = SDR white): reads the current
// `scene_color`, the previous frame's accumulated history, and the G-buffer
// velocity (#147); writes the new history, which the display pass then
// consumes. History rejection is variance clipping in YCoCg (Salvi): the
// 3x3 neighborhood's mean +- GAMMA * stddev bounds what a plausible history
// can look like; reprojected history outside the box is clipped along the
// line toward the current sample. YCoCg decorrelates luma from chroma, so
// the axis-aligned statistical box hugs the neighborhood tightly — the same
// box in RGB is inflated along the gray diagonal and lets ghosting through.
//
// Background pixels (no geometry — the G-buffer albedo-alpha marker,
// ADR-039) carry zero velocity; their history is reprojected through the
// camera matrices instead (sky moves with camera rotation, not geometry).

import motion; // shared reprojection helpers (#149)

// =============================================================================
// Vertex Shader
// =============================================================================

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    VsOutput output;

    // Fullscreen triangle
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

// =============================================================================
// Fragment Shader
// =============================================================================

[[vk::binding(0, 0)]]
cbuffer TaaUniforms {
    // Inverse of the current unjittered view-projection (background rays).
    column_major float4x4 inv_view_proj;
    // Previous frame's unjittered view-projection (background reprojection).
    column_major float4x4 prev_view_proj;
    // x = history blend (fraction of the *current* frame in the mix),
    // y = history validity (0 on the first frame after (re)creation),
    // zw = texel size (1 / target extent).
    float4 params;
};

[[vk::binding(1, 0)]]
Texture2D scene_color;
[[vk::binding(2, 0)]]
Texture2D history;
[[vk::binding(3, 0)]]
Texture2D gbuffer_velocity;
[[vk::binding(4, 0)]]
Texture2D gbuffer_albedo;
[[vk::binding(5, 0)]]
SamplerState nearest_sampler;
[[vk::binding(6, 0)]]
SamplerState linear_sampler;
// World position (G-buffer RT2, xyz). Reprojected through the camera pair to
// recover the *camera-only* screen motion at each pixel (below), independent
// of any object's own motion.
[[vk::binding(7, 0)]]
Texture2D gbuffer_position;

// History-rejection box width in standard deviations: the
// ghosting-vs-flicker knob (narrow clips history aggressively and flickers
// on thin detail; wide lets trails through).
static const float VARIANCE_GAMMA = 1.0;

// Velocity-adaptive blend (camera-only). The trail length in a moving shot is
// set by how much of the current frame the mix keeps: `params.x` (a long,
// stable memory) when the camera is still, ramping to `TAA_MAX_BLEND` (a
// short memory that converges in ~1/value frames) as the camera-induced
// screen motion grows. Driving this from the *camera* component — not the
// per-object velocity — keeps a screen-static object (e.g. a chase-cam
// subject) on the stable long memory while the environment it scrolls past
// converges fast, so the environment streak shrinks without adding noise to
// the subject.
static const float TAA_MAX_BLEND = 0.5;
// Pixels/frame of camera-induced motion at which the blend reaches
// TAA_MAX_BLEND (reciprocal ramp width).
static const float TAA_SPEED_SCALE = 1.0 / 20.0;

float3 rgb_to_ycocg(float3 c) {
    return float3(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

float3 ycocg_to_rgb(float3 c) {
    return float3(
        c.x + c.y - c.z,
        c.x + c.z,
        c.x - c.y - c.z);
}

// Salvi clip: pull `q` along the segment toward the box center until it
// enters the aabb — unlike a per-channel clamp this preserves the color
// direction (no hue shift).
float3 clip_to_aabb(float3 aabb_min, float3 aabb_max, float3 q) {
    float3 center = 0.5 * (aabb_max + aabb_min);
    float3 extents = 0.5 * (aabb_max - aabb_min) + 1e-6;
    float3 v = q - center;
    float3 t = abs(v / extents);
    float max_t = max(t.x, max(t.y, t.z));
    if (max_t > 1.0) {
        return center + v / max_t;
    }
    return q;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float2 texel = params.zw;
    float3 current = scene_color.Sample(nearest_sampler, input.uv).rgb;

    // Reprojected history coordinate: geometry pixels carry their own NDC
    // motion; background reprojects the view ray through the camera pair.
    // `cam_uv` is the same pixel reprojected using *only* the camera motion
    // (object-independent) — its screen delta drives the adaptive blend.
    float2 prev_uv;
    float2 cam_uv;
    bool background = gbuffer_albedo.Sample(nearest_sampler, input.uv).a < 1.0 / 128.0;
    if (background) {
        // Sky has no object motion, so its camera reprojection (far-plane ray
        // through the camera pair) *is* the history reprojection.
        float3 world = far_plane_world(inv_view_proj, input.uv);
        prev_uv = project_uv(prev_view_proj, world);
        cam_uv = prev_uv;
    } else {
        float2 velocity = gbuffer_velocity.Sample(nearest_sampler, input.uv).xy;
        prev_uv = reproject_prev_uv(input.uv, velocity);
        // Camera-only reprojection of this pixel's world position — unlike
        // `velocity` (which folds in the object's own motion) it depends solely
        // on the camera, so a moving object does not inflate the blend itself.
        float3 world = gbuffer_position.Sample(nearest_sampler, input.uv).xyz;
        cam_uv = project_uv(prev_view_proj, world);
    }

    // Camera-induced screen motion in pixels/frame → adaptive current-frame
    // weight. Still camera → params.x (long, stable memory); fast pan →
    // TAA_MAX_BLEND (short memory), shrinking the environment streak.
    float cam_speed = length((input.uv - cam_uv) / texel);
    float blend = lerp(params.x, TAA_MAX_BLEND, saturate(cam_speed * TAA_SPEED_SCALE));

    // Neighborhood statistics in YCoCg for the variance-clipping box.
    float3 mean = float3(0.0);
    float3 mean_sq = float3(0.0);
    [unroll]
    for (int dy = -1; dy <= 1; dy++) {
        [unroll]
        for (int dx = -1; dx <= 1; dx++) {
            float3 s = rgb_to_ycocg(
                scene_color.Sample(nearest_sampler, input.uv + float2(dx, dy) * texel).rgb);
            mean += s;
            mean_sq += s * s;
        }
    }
    mean /= 9.0;
    mean_sq /= 9.0;
    float3 sigma = sqrt(max(mean_sq - mean * mean, float3(0.0)));
    float3 aabb_min = mean - VARIANCE_GAMMA * sigma;
    float3 aabb_max = mean + VARIANCE_GAMMA * sigma;

    // Full-current fallback must be a BRANCH, not `lerp(history, current,
    // 1.0)`: the first frame's history texture is uninitialized and may
    // hold NaN — which survives any arithmetic (including lerp at alpha 1),
    // gets written back, and poisons the accumulation forever. Phrased so a
    // NaN prev_uv (degenerate reprojection) also lands here: comparisons
    // with NaN are false, so `on_screen` fails.
    bool on_screen = all(prev_uv >= float2(0.0)) && all(prev_uv <= float2(1.0));
    if (params.y < 0.5 || !on_screen) {
        return float4(current, 1.0);
    }

    float3 history_color = history.Sample(linear_sampler, prev_uv).rgb;
    // Belt and suspenders against the same feedback loop: a NaN that slips
    // into the history (driver-dependent contents, future passes) resets to
    // the current sample instead of propagating.
    if (any(isnan(history_color))) {
        history_color = current;
    }
    float3 clipped = ycocg_to_rgb(
        clip_to_aabb(aabb_min, aabb_max, rgb_to_ycocg(history_color)));

    float3 result = lerp(clipped, current, blend);
    return float4(result, 1.0);
}
