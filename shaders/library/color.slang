// RedLilium Shader Library - Color Module
// Color space conversions and tone mapping functions.

import math;

// Convert linear RGB to sRGB.
float3 linear_to_srgb(float3 color) {
    float3 low = color * 12.92;
    float3 high = pow(color, float3(1.0 / 2.4)) * 1.055 - 0.055;
    float3 t = step(float3(0.0031308), color);
    return lerp(low, high, t);
}

// Convert sRGB to linear RGB.
float3 srgb_to_linear(float3 color) {
    float3 low = color / 12.92;
    float3 high = pow((color + 0.055) / 1.055, float3(2.4));
    float3 t = step(float3(0.04045), color);
    return lerp(low, high, t);
}

// Simple gamma correction (gamma = 2.2).
float3 gamma_correct(float3 color) {
    return pow(color, float3(1.0 / 2.2));
}

// Remove gamma correction (gamma = 2.2).
float3 gamma_uncorrect(float3 color) {
    return pow(color, float3(2.2));
}

// Reinhard tone mapping.
float3 tonemap_reinhard(float3 color) {
    return color / (color + float3(1.0));
}

// Reinhard extended tone mapping with white point.
float3 tonemap_reinhard_extended(float3 color, float white_point) {
    float white_sq = white_point * white_point;
    float3 numerator = color * (float3(1.0) + color / white_sq);
    return numerator / (float3(1.0) + color);
}

// ACES filmic tone mapping (approximation).
float3 tonemap_aces(float3 color) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate3((color * (a * color + b)) / (color * (c * color + d) + e));
}

// Uncharted 2 tone mapping curve.
float3 tonemap_uncharted2_partial(float3 x) {
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// Uncharted 2 tone mapping.
float3 tonemap_uncharted2(float3 color) {
    float exposure_bias = 2.0;
    float3 W = float3(11.2);
    float3 curr = tonemap_uncharted2_partial(color * exposure_bias);
    float3 white_scale = float3(1.0) / tonemap_uncharted2_partial(W);
    return curr * white_scale;
}

// Luminance of a linear RGB color.
float luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// Desaturate a color by a factor (0 = original, 1 = grayscale).
float3 desaturate(float3 color, float factor) {
    float gray = luminance(color);
    return lerp3(color, float3(gray), factor);
}
