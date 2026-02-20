// RedLilium Shader Library - Math Module
// Mathematical constants and utility functions.

#ifndef REDLILIUM_MATH_GLSL
#define REDLILIUM_MATH_GLSL

// Pi constant.
const float PI = 3.14159265359;

// Tau (2 * Pi) constant.
const float TAU = 6.28318530718;

// Inverse of Pi.
const float INV_PI = 0.31830988618;

// Inverse of Tau.
const float INV_TAU = 0.15915494309;

// Small epsilon for avoiding division by zero.
const float EPSILON = 0.0001;

// Clamp a value to [0, 1].
float saturate_f(float x) {
    return clamp(x, 0.0, 1.0);
}

// Clamp a vec2 to [0, 1].
vec2 saturate2(vec2 v) {
    return clamp(v, vec2(0.0), vec2(1.0));
}

// Clamp a vec3 to [0, 1].
vec3 saturate3(vec3 v) {
    return clamp(v, vec3(0.0), vec3(1.0));
}

// Clamp a vec4 to [0, 1].
vec4 saturate4(vec4 v) {
    return clamp(v, vec4(0.0), vec4(1.0));
}

// Linear interpolation.
float lerp_f32(float a, float b, float t) {
    return mix(a, b, t);
}

// Linear interpolation for vec3.
vec3 lerp3(vec3 a, vec3 b, float t) {
    return mix(a, b, t);
}

// Smoothstep function.
float smoothstep_f32(float edge0, float edge1, float x) {
    return smoothstep(edge0, edge1, x);
}

// Square of a value.
float sq(float x) {
    return x * x;
}

// Square of a vec3.
vec3 sq3(vec3 v) {
    return v * v;
}

// Safe normalize that handles zero vectors.
vec3 safe_normalize(vec3 v) {
    float len = length(v);
    if (len > EPSILON) {
        return v / len;
    }
    return vec3(0.0, 1.0, 0.0);
}

#endif
