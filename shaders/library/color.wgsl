// RedLilium Shader Library - Color Module
// Color space conversions and tone mapping functions.

#define_import_path redlilium::color

#import redlilium::math::{saturate3, lerp3}

/// Convert linear RGB to sRGB.
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = pow(color, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

/// Convert sRGB to linear RGB.
fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    let low = color / 12.92;
    let high = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, color <= vec3<f32>(0.04045));
}

/// Simple gamma correction (gamma = 2.2).
fn gamma_correct(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / 2.2));
}

/// Remove gamma correction (gamma = 2.2).
fn gamma_uncorrect(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(2.2));
}

/// Reinhard tone mapping.
fn tonemap_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3<f32>(1.0));
}

/// Reinhard extended tone mapping with white point.
fn tonemap_reinhard_extended(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let white_sq = white_point * white_point;
    let numerator = color * (vec3<f32>(1.0) + color / white_sq);
    return numerator / (vec3<f32>(1.0) + color);
}

/// ACES filmic tone mapping (approximation).
fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate3((color * (a * color + b)) / (color * (c * color + d) + e));
}

/// Uncharted 2 tone mapping curve.
fn tonemap_uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

/// Uncharted 2 tone mapping.
fn tonemap_uncharted2(color: vec3<f32>) -> vec3<f32> {
    let exposure_bias = 2.0;
    let W = vec3<f32>(11.2);
    let curr = tonemap_uncharted2_partial(color * exposure_bias);
    let white_scale = vec3<f32>(1.0) / tonemap_uncharted2_partial(W);
    return curr * white_scale;
}

/// Luminance of a linear RGB color.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Desaturate a color by a factor (0 = original, 1 = grayscale).
fn desaturate(color: vec3<f32>, factor: f32) -> vec3<f32> {
    let gray = luminance(color);
    return lerp3(color, vec3<f32>(gray), factor);
}
