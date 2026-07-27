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
//
// Three refinements over the base accumulate/clip (#148 follow-up):
//   1. Karis anti-flicker: the current/history blend happens in a reversible
//      tonemapped domain (c / (1 + luma)), so a sub-pixel HDR highlight that
//      the jitter walks in and out of a pixel can no longer dominate the
//      average and strobe. Exactly the sun-bright class of value from #152.
//   2. Catmull-Rom history: the reprojected history is fetched with a 5-tap
//      bicubic (bilinear-optimised) instead of a single bilinear tap, which
//      undoes most of the per-frame reprojection blur that softens TAA.
//   3. Depth disocclusion: each frame stashes its clamped view-depth in the
//      history alpha (otherwise dead — always 1.0). Next frame the reprojected
//      alpha is the *previous* depth at that pixel; a large relative mismatch
//      under motion means the history came from a different surface (a moving
//      foreground uncovering background), so the history is rejected even when
//      its colour happens to fall inside the variance box. Gated by motion so
//      static silhouettes keep their temporal AA.

import math; // fast-math-proof NaN guards (any_nan2 / any_nan3, #152)
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
    // Camera world position (xyz) for the linear-depth disocclusion proxy the
    // history carries in its alpha (#148 depth reject).
    float4 camera_pos;
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
// Camera depth (reversed-Z), unfilterable-float, read via Load. World
// position is reconstructed from it (unjittered inverse — sub-pixel error
// only) and reprojected through the camera pair to recover the *camera-only*
// screen motion at each pixel (below), independent of any object's own
// motion; it also feeds the view-depth disocclusion proxy.
[[vk::binding(7, 0)]]
Texture2D<float> depth_texture;

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

// Non-negative luma for the Karis anti-flicker weight (any positive-weight
// combination works; this matches the YCoCg Y so it is consistent with the
// clip box). `max(0)` because a Catmull-Rom overshoot can push a channel
// slightly negative, and a negative luma would invert the weight.
float taa_luma(float3 c) {
    return max(dot(c, float3(0.25, 0.5, 0.25)), 0.0);
}

// Karis anti-flicker weight: the inverse of a sample's luma (Karis "High
// Quality Temporal Supersampling"). Blending current and history as an
// energy-preserving weighted average with this weight de-emphasises a bright
// sub-pixel highlight that the jitter walks in and out of a pixel — the
// firefly can no longer dominate the mean and strobe. A *steady* highlight
// weighs current and history down equally, so its intensity is preserved
// (unlike a tonemap round-trip, which is not exactly invertible at the f16
// extreme and would permanently dim it).
float taa_flicker_weight(float3 c) {
    return 1.0 / (1.0 + taa_luma(c));
}

// 5-tap bilinear-optimised Catmull-Rom fetch (Playdead/INSIDE, MJP). A single
// bilinear history tap blurs the accumulation a little more each frame; the
// bicubic reconstruction restores most of that lost sharpness for four extra
// samples. The Catmull-Rom lobes are negative, so the fetch can overshoot at
// high-contrast edges — harmless here because the caller clips the result to
// the neighbourhood AABB straight after. Returns rgb only; the alpha (depth
// proxy) is fetched separately with a plain bilinear tap, which needs no
// overshoot.
float3 sample_history_catmull_rom(float2 uv, float2 texel, float2 resolution) {
    float2 sample_pos = uv * resolution;
    float2 tex_pos1 = floor(sample_pos - 0.5) + 0.5;
    float2 f = sample_pos - tex_pos1;

    // Catmull-Rom weights (sharpness 0.5) per axis.
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);

    // Fuse the middle two taps into one bilinear fetch at their weighted
    // centroid — the trick that turns 4x4 taps into 3x3, then a cross into 5.
    float2 w12 = w1 + w2;
    float2 offset12 = w2 / w12;

    float2 tc0 = (tex_pos1 - 1.0) * texel;
    float2 tc3 = (tex_pos1 + 2.0) * texel;
    float2 tc12 = (tex_pos1 + offset12) * texel;

    float3 result = float3(0.0);
    result += history.Sample(linear_sampler, float2(tc12.x, tc0.y)).rgb * (w12.x * w0.y);
    result += history.Sample(linear_sampler, float2(tc0.x, tc12.y)).rgb * (w0.x * w12.y);
    result += history.Sample(linear_sampler, float2(tc12.x, tc12.y)).rgb * (w12.x * w12.y);
    result += history.Sample(linear_sampler, float2(tc3.x, tc12.y)).rgb * (w3.x * w12.y);
    result += history.Sample(linear_sampler, float2(tc12.x, tc3.y)).rgb * (w12.x * w3.y);
    return result;
}

// Far sentinel for the view-depth proxy, kept under F16_MAX (65504) so it
// stores losslessly enough in the history alpha and never rounds to +Inf
// (a +Inf there would make the disocclusion subtraction NaN, poisoning the
// blend — the #152 failure mode). Geometry farther than this simply reads as
// "far", which is exactly right for disocclusion.
static const float TAA_FAR_DEPTH = 60000.0;

// Relative view-depth mismatch that starts / completes history rejection at a
// disocclusion. Scale-invariant (fraction of the compared depth), matching the
// motion-blur SOFT_Z_REL knob. Generous so a static silhouette whose bilinear
// depth tap merely straddles an edge does not trip it (the motion gate below
// is the real guard).
static const float TAA_DEPTH_REJECT_LO = 0.05;
static const float TAA_DEPTH_REJECT_HI = 0.25;

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
    // This pixel's clamped view-depth, stashed into the history alpha at the
    // end so next frame can read the *previous* depth here and detect
    // disocclusion. Sky sits at the far sentinel.
    float depth;
    bool background = gbuffer_albedo.Sample(nearest_sampler, input.uv).a < 1.0 / 128.0;
    if (background) {
        // Sky has no object motion, so its camera reprojection (far-plane ray
        // through the camera pair) *is* the history reprojection.
        float3 world = far_plane_world(inv_view_proj, input.uv);
        prev_uv = project_uv(prev_view_proj, world);
        cam_uv = prev_uv;
        depth = TAA_FAR_DEPTH;
    } else {
        float2 velocity = gbuffer_velocity.Sample(nearest_sampler, input.uv).xy;
        prev_uv = reproject_prev_uv(input.uv, velocity);
        // Camera-only reprojection of this pixel's world position — unlike
        // `velocity` (which folds in the object's own motion) it depends solely
        // on the camera, so a moving object does not inflate the blend itself.
        float3 world = reconstruct_world(
            inv_view_proj, input.uv, load_texel_r(depth_texture, input.uv));
        cam_uv = project_uv(prev_view_proj, world);
        depth = min(length(camera_pos.xyz - world), TAA_FAR_DEPTH);
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
    // gets written back, and poisons the accumulation forever. A degenerate
    // reprojection can also make prev_uv NaN; rather than lean on "comparisons
    // with NaN are false" (Metal fast-math relaxes that on the Vulkan/MoltenVK
    // path, #152), reject a NaN prev_uv explicitly with the bit-test guard.
    bool on_screen = !any_nan2(prev_uv)
        && all(prev_uv >= float2(0.0)) && all(prev_uv <= float2(1.0));
    if (params.y < 0.5 || !on_screen) {
        // Carry this frame's depth in alpha so the very first accumulated
        // frame after priming has a valid previous depth to compare against.
        return float4(current, depth);
    }

    // Catmull-Rom history fetch (bicubic, 5 bilinear taps) instead of a single
    // bilinear tap — restores the sharpness a per-frame bilinear reprojection
    // otherwise erodes. `resolution = 1 / texel`.
    float3 history_color = sample_history_catmull_rom(prev_uv, texel, 1.0 / texel);
    // Belt and suspenders against the same feedback loop: a NaN that slips
    // into the history (driver-dependent contents, future passes) resets to
    // the current sample instead of propagating. `any_nan3`, not the `isnan()`
    // intrinsic — fast-math folds `isnan()` to false on MoltenVK (#152).
    if (any_nan3(history_color)) {
        history_color = current;
    }
    float3 clipped = ycocg_to_rgb(
        clip_to_aabb(aabb_min, aabb_max, rgb_to_ycocg(history_color)));

    // Depth disocclusion: the previous depth at the reprojected spot is the
    // history alpha (a plain bilinear tap — no bicubic overshoot on a depth).
    // A large *relative* mismatch means the reprojection landed on a different
    // surface (a moving foreground uncovering background), so reject the
    // history even when its colour happens to fall inside the variance box.
    // Gated by `saturate(motion_px)`: with no reprojected motion there is no
    // disocclusion, and a static silhouette whose bilinear depth tap straddles
    // an edge must keep its temporal AA rather than re-alias.
    float prev_depth = history.Sample(linear_sampler, prev_uv).a;
    float depth_diff = abs(depth - prev_depth) / max(depth, 1e-3);
    float motion_px = length((input.uv - prev_uv) / texel);
    float reject =
        smoothstep(TAA_DEPTH_REJECT_LO, TAA_DEPTH_REJECT_HI, depth_diff) * saturate(motion_px);
    float blend_eff = lerp(blend, 1.0, reject);

    // Karis anti-flicker: an energy-preserving weighted average, each endpoint
    // scaled by the inverse of its luma, so a bright sub-pixel highlight cannot
    // dominate the mix and strobe (a steady highlight is preserved — both
    // endpoints weigh down equally).
    float w_hist = (1.0 - blend_eff) * taa_flicker_weight(clipped);
    float w_curr = blend_eff * taa_flicker_weight(current);
    float3 result = (clipped * w_hist + current * w_curr) / max(w_hist + w_curr, 1e-5);
    return float4(result, depth);
}
