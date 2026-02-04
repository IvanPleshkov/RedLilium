// RedLilium Shader Library - Math Module
// Mathematical constants and utility functions.

#define_import_path redlilium::math

/// Pi constant.
const PI: f32 = 3.14159265359;

/// Tau (2 * Pi) constant.
const TAU: f32 = 6.28318530718;

/// Inverse of Pi.
const INV_PI: f32 = 0.31830988618;

/// Inverse of Tau.
const INV_TAU: f32 = 0.15915494309;

/// Small epsilon for avoiding division by zero.
const EPSILON: f32 = 0.0001;

/// Clamp a value to [0, 1].
fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

/// Clamp a vec2 to [0, 1].
fn saturate2(v: vec2<f32>) -> vec2<f32> {
    return clamp(v, vec2<f32>(0.0), vec2<f32>(1.0));
}

/// Clamp a vec3 to [0, 1].
fn saturate3(v: vec3<f32>) -> vec3<f32> {
    return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Clamp a vec4 to [0, 1].
fn saturate4(v: vec4<f32>) -> vec4<f32> {
    return clamp(v, vec4<f32>(0.0), vec4<f32>(1.0));
}

/// Linear interpolation.
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

/// Linear interpolation for vec3.
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

/// Smoothstep function.
fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

/// Square of a value.
fn sq(x: f32) -> f32 {
    return x * x;
}

/// Square of a vec3.
fn sq3(v: vec3<f32>) -> vec3<f32> {
    return v * v;
}

/// Safe normalize that handles zero vectors.
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if len > EPSILON {
        return v / len;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}
