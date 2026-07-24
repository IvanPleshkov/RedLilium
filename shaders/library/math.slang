// RedLilium Shader Library - Math Module
// Mathematical constants and utility functions.
//
// -----------------------------------------------------------------------------
// Numeric-safety policy for shaders (#152) — read before adding any NaN/Inf guard.
//
// 1. Fix at the source. When an unbounded value (bake, lighting, emissive) is
//    written into a bounded format (f16), clamp it there. Kill the bad value,
//    not its symptom downstream.
// 2. Guard only at trust boundaries — where data enters from something you do
//    not control (loaded/external assets, the scene_color writers). Once a
//    boundary guarantees finiteness, downstream passes may assume it; do not
//    re-sanitize every pass (that hides where the real boundary is).
// 3. NEVER test NaN/Inf with float ops. `x == x`, the `isnan()` intrinsic, and
//    `min`/`max` NaN semantics are folded away by Metal fast-math on the
//    Vulkan/MoltenVK path (and vary by backend), so the guard silently dies.
//    Use the bit-test helpers here (`sanitize_finite`, `is_nan`) — integer ops
//    survive fast-math on every backend.
// 4. Put every finiteness invariant under a test (e.g. xtask's sky_cube finite
//    scan). An unchecked invariant rots silently — that is why #152 lived so long.
// -----------------------------------------------------------------------------

// Pi constant.
static const float PI = 3.14159265359;

// Tau (2 * Pi) constant.
static const float TAU = 6.28318530718;

// Inverse of Pi.
static const float INV_PI = 0.31830988618;

// Inverse of Tau.
static const float INV_TAU = 0.15915494309;

// Small epsilon for avoiding division by zero.
static const float EPSILON = 0.0001;

// Clamp a value to [0, 1].
float saturate_f(float x) {
    return saturate(x);
}

// Clamp a float2 to [0, 1].
float2 saturate2(float2 v) {
    return saturate(v);
}

// Clamp a float3 to [0, 1].
float3 saturate3(float3 v) {
    return saturate(v);
}

// Clamp a float4 to [0, 1].
float4 saturate4(float4 v) {
    return saturate(v);
}

// Linear interpolation.
float lerp_f32(float a, float b, float t) {
    return lerp(a, b, t);
}

// Linear interpolation for float3.
float3 lerp3(float3 a, float3 b, float t) {
    return lerp(a, b, t);
}

// Smoothstep function.
float smoothstep_f32(float edge0, float edge1, float x) {
    return smoothstep(edge0, edge1, x);
}

// Square of a value.
float sq(float x) {
    return x * x;
}

// Square of a float3.
float3 sq3(float3 v) {
    return v * v;
}

// Safe normalize that handles zero vectors.
float3 safe_normalize(float3 v) {
    float len = length(v);
    if (len > EPSILON) {
        return v / len;
    }
    return float3(0.0, 1.0, 0.0);
}

// Largest magnitude representable in an f16 (Rgba16Float) texel. Both the
// scene_color intermediate and the bloom mip chain are f16, so this is the
// clamp ceiling below — a finite fp32 value above it would round to +Inf on
// store and poison every downstream pass.
static const float F16_MAX = 65504.0;

// Force a value finite and f16-storable: NaN -> 0, -Inf -> 0, +Inf -> F16_MAX,
// finite -> clamped to [0, F16_MAX] (radiance is non-negative). Used at every
// write into an f16 render target that can carry an extreme HDR highlight
// (skybox/resolve scene_color, the bloom taps) so a sun-bright Inf/NaN never
// spreads a black or colored block up the chain (#152).
//
// The guard tests the IEEE-754 bit pattern, NOT a float compare like `x == x`:
// on the Vulkan/MoltenVK path Metal compiles the final MSL with fast-math, which
// assumes operands are never NaN/Inf and folds `x == x` to a constant `true`, so
// a float NaN test is silently dropped before it reaches the GPU (this is why the
// artifact was Vulkan-only — the WGSL/naga source keeps the compare, Metal drops
// it). Integer bit ops are untouched by fast-math, so this holds on every backend.
float sanitize_finite(float x) {
    uint bits = asuint(x);
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // exponent all-ones: NaN (nonzero mantissa) or Inf (zero mantissa).
        if ((bits & 0x7FFFFFu) != 0u) {
            return 0.0; // NaN
        }
        return (bits >> 31) != 0u ? 0.0 : F16_MAX; // -Inf : +Inf
    }
    return clamp(x, 0.0, F16_MAX);
}

// Per-channel `sanitize_finite`.
float3 sanitize_finite3(float3 v) {
    return float3(sanitize_finite(v.x), sanitize_finite(v.y), sanitize_finite(v.z));
}

// True if `x` is NaN, tested via the IEEE-754 bit pattern (exponent all-ones,
// nonzero mantissa). Use this instead of the `isnan()` intrinsic or the `x != x`
// self-inequality: both are folded to a constant `false` by Metal fast-math on
// the Vulkan/MoltenVK path (fast-math assumes operands are never NaN), leaving a
// dead guard there (#152). Integer bit ops survive fast-math on every backend.
bool is_nan(float x) {
    uint bits = asuint(x);
    return (((bits >> 23) & 0xFFu) == 0xFFu) && ((bits & 0x7FFFFFu) != 0u);
}

// True if any component is NaN (fast-math-proof; see `is_nan`).
bool any_nan2(float2 v) {
    return is_nan(v.x) || is_nan(v.y);
}
bool any_nan3(float3 v) {
    return is_nan(v.x) || is_nan(v.y) || is_nan(v.z);
}
