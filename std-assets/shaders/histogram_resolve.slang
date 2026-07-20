// Auto-exposure histogram resolve (#153) — second compute pass, one thread.
//
// Reads the 256-bin luminance histogram `histogram_build.slang` produced,
// meters an average scene luminance, converts it to a target exposure
// multiplier (clamped to the camera's adaptation window), and smooths toward
// it in a single persistent slot so the value carries across frames (eye
// adaptation). The display pass multiplies its manual exposure by this slot.
//
// One thread is plenty: the whole cost is a 255-iteration sum. The persistent
// slot is read-modify-written here, so this must be the sole writer.

[[vk::binding(0, 0)]]
cbuffer ResolveUniforms {
    // x = LOG_MIN, y = LOG_MAX (must match the build pass),
    // z = adaptation rate when brightening (0..1 this frame; folds dt + speed),
    // w = middle-grey key (target average, ~0.18).
    float4 params;
    // x = min exposure multiplier (2^min_ev), y = max (2^max_ev),
    // z = adaptation rate when darkening, w unused.
    float4 limits;
};

[[vk::binding(1, 0)]]
StructuredBuffer<uint> histogram;
// Single persistent slot: [0] = smoothed exposure multiplier (survives frames).
[[vk::binding(2, 0)]]
RWStructuredBuffer<float> exposure;

static const uint BINS = 256;

[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    float log_min = params.x;
    float log_max = params.y;
    float rate_up = params.z;
    float key = params.w;
    float min_mult = limits.x;
    float max_mult = limits.y;
    float rate_down = limits.z;

    // Weighted average of log-luminance over bins 1..255 (bin 0 is the
    // under-range/black bucket — skipped so dark areas don't drag exposure).
    float weighted = 0.0;
    float total = 0.0;
    for (uint i = 1; i < BINS; ++i) {
        float count = float(histogram[i]);
        // Bin i's center in [log_min, log_max] (bins 1..255 span the range).
        float t = (float(i) - 0.5) / float(BINS - 1);
        float log_lum = lerp(log_min, log_max, t);
        weighted += log_lum * count;
        total += count;
    }

    float prev = exposure[0];
    // Hold when nothing metered in range (fully dark / under-range frame).
    if (total < 1.0) {
        return;
    }

    float avg_lum = exp2(weighted / total);
    float target = key / max(avg_lum, 1e-4);
    target = clamp(target, min_mult, max_mult);

    // Asymmetric smoothing: brighten and darken at their own rates (the eye
    // stops down faster than it dark-adapts). Rates already fold dt + speed.
    float rate = (target > prev) ? rate_up : rate_down;
    exposure[0] = prev + (target - prev) * rate;
}
