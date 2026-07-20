// Auto-exposure histogram build (#153) — first of the two compute passes.
//
// Samples the scene-referred image over a fixed square grid (resolution-
// independent work and atomic contention) and bins each sample's log2
// luminance into a 256-bucket histogram. `histogram_resolve.slang` then meters
// an adapting exposure from these counts.
//
// Bin 0 is the under-range bucket (luminance at or below `2^LOG_MIN`, plus
// true black); the resolve pass skips it so a large dark background does not
// drag the exposure down. The remaining 255 bins span [LOG_MIN, LOG_MAX] in
// log2 space, matching the CPU-side constants in `deferred.rs`.
//
// The counts are accumulated with global atomics for codegen robustness across
// the WGSL and SPIR-V targets; a group-shared reduction and a half-res source
// are the obvious perf follow-ups (see #153).

[[vk::binding(0, 0)]]
cbuffer BuildUniforms {
    // x = LOG_MIN (log2 luminance of bin 1's lower edge),
    // y = LOG_MAX (log2 luminance of the top bin's upper edge),
    // z = sample grid dimension (samples per side), w unused.
    float4 params;
};

[[vk::binding(1, 0)]]
Texture2D scene_color;
[[vk::binding(2, 0)]]
SamplerState scene_sampler;
// 256 luminance bins, cleared to zero by the prep transfer each frame. The
// element type is `Atomic<uint>` so the accumulation lowers to a real atomic
// on every target — Slang's WGSL backend rejects `InterlockedAdd` on a plain
// buffer (WGSL atomics require an `atomic<u32>` element).
[[vk::binding(3, 0)]]
RWStructuredBuffer<Atomic<uint> > histogram;

static const uint BINS = 256;
static const float LUMA_CLAMP = 65504.0; // f16 max — kills +Inf from a bright sky.

float luma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    float dim = params.z;
    if (float(tid.x) >= dim || float(tid.y) >= dim) {
        return;
    }

    // Sample the scene at this grid cell's center (linear filtering averages
    // the covered footprint, so the fixed grid still sees the whole image).
    float2 uv = (float2(tid.xy) + 0.5) / dim;
    float3 c = scene_color.SampleLevel(scene_sampler, uv, 0.0).rgb;
    float l = luma(c);
    // Kill NaN via the self-inequality, clamp +Inf to the f16 max — the same
    // guard the bloom path uses (backends disagree on min()/clamp() NaN).
    if (!(l == l)) {
        l = 0.0;
    }
    l = clamp(l, 0.0, LUMA_CLAMP);

    float log_min = params.x;
    float log_max = params.y;
    uint bin = 0; // under-range / black bucket
    if (l > 1e-4) {
        float t = (log2(l) - log_min) / (log_max - log_min);
        // Bins 1..255 span [log_min, log_max]; bin 0 stays the under-range one.
        bin = 1u + uint(clamp(t, 0.0, 1.0) * float(BINS - 2) + 0.5);
    }
    histogram[bin].add(1u);
}
