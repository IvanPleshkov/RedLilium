// RedLilium Shader Library - Color Module
// Color space conversions and tone mapping functions.

#ifndef REDLILIUM_COLOR_GLSL
#define REDLILIUM_COLOR_GLSL

#include "redlilium/math.glsl"

// Convert linear RGB to sRGB.
vec3 linear_to_srgb(vec3 color) {
    vec3 low = color * 12.92;
    vec3 high = pow(color, vec3(1.0 / 2.4)) * 1.055 - 0.055;
    // step(edge, x) returns 1.0 where x >= edge; we want low where color <= threshold
    vec3 t = step(vec3(0.0031308), color);
    return mix(low, high, t);
}

// Convert sRGB to linear RGB.
vec3 srgb_to_linear(vec3 color) {
    vec3 low = color / 12.92;
    vec3 high = pow((color + 0.055) / 1.055, vec3(2.4));
    vec3 t = step(vec3(0.04045), color);
    return mix(low, high, t);
}

// Simple gamma correction (gamma = 2.2).
vec3 gamma_correct(vec3 color) {
    return pow(color, vec3(1.0 / 2.2));
}

// Remove gamma correction (gamma = 2.2).
vec3 gamma_uncorrect(vec3 color) {
    return pow(color, vec3(2.2));
}

// Reinhard tone mapping.
vec3 tonemap_reinhard(vec3 color) {
    return color / (color + vec3(1.0));
}

// Reinhard extended tone mapping with white point.
vec3 tonemap_reinhard_extended(vec3 color, float white_point) {
    float white_sq = white_point * white_point;
    vec3 numerator = color * (vec3(1.0) + color / white_sq);
    return numerator / (vec3(1.0) + color);
}

// ACES filmic tone mapping (approximation).
vec3 tonemap_aces(vec3 color) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate3((color * (a * color + b)) / (color * (c * color + d) + e));
}

// Uncharted 2 tone mapping curve.
vec3 tonemap_uncharted2_partial(vec3 x) {
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// Uncharted 2 tone mapping.
vec3 tonemap_uncharted2(vec3 color) {
    float exposure_bias = 2.0;
    vec3 W = vec3(11.2);
    vec3 curr = tonemap_uncharted2_partial(color * exposure_bias);
    vec3 white_scale = vec3(1.0) / tonemap_uncharted2_partial(W);
    return curr * white_scale;
}

// Luminance of a linear RGB color.
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// Desaturate a color by a factor (0 = original, 1 = grayscale).
vec3 desaturate(vec3 color, float factor) {
    float gray = luminance(color);
    return lerp3(color, vec3(gray), factor);
}

#endif
