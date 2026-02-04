// RedLilium Shader Library - Egui Module
// Shader for rendering egui UI elements.
// Handles vertex positions, texture coordinates, and vertex colors with alpha blending.

#define_import_path redlilium::egui

/// Screen size uniform for coordinate transformation.
struct EguiUniforms {
    screen_size: vec2<f32>,
    _padding: vec2<f32>,
}

/// Vertex input from egui mesh data.
/// Locations match VertexAttributeSemantic indices:
/// - Position = 0, TexCoord0 = 3, Color = 5
struct EguiVertexInput {
    @location(0) position: vec2<f32>,
    @location(3) tex_coords: vec2<f32>,
    @location(5) color: vec4<f32>,
}

/// Vertex output to fragment shader.
struct EguiVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
}

/// Convert sRGB color to linear color space.
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3<f32>(0.04045);
    let higher = pow((srgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let lower = srgb / vec3<f32>(12.92);
    return select(higher, lower, cutoff);
}

/// Convert linear color to sRGB color space.
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3<f32>(0.0031308);
    let higher = vec3<f32>(1.055) * pow(linear, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let lower = linear * vec3<f32>(12.92);
    return select(higher, lower, cutoff);
}
