// RedLilium Shader Library - Math Module
// Mathematical constants and utility functions.

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
