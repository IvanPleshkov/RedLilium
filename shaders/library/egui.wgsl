// RedLilium Shader Library - Egui Module
// Complete shader for rendering egui UI elements.
// Handles vertex positions, texture coordinates, and vertex colors with alpha blending.
//
// Based on egui-wgpu's shader approach for proper alpha blending.
// Uses gamma framebuffer output since our surface format is Bgra8Unorm (non-sRGB).

#define_import_path redlilium::egui

// =============================================================================
// Types (importable by other shaders)
// =============================================================================

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

// =============================================================================
// Utility Functions (importable by other shaders)
// =============================================================================

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

// =============================================================================
// Resource Bindings
// =============================================================================

@group(0) @binding(0) var<uniform> uniforms: EguiUniforms;
@group(1) @binding(0) var egui_texture: texture_2d<f32>;
@group(1) @binding(1) var egui_sampler: sampler;

// =============================================================================
// Entry Points
// =============================================================================

@vertex
fn vs_main(in: EguiVertexInput) -> EguiVertexOutput {
    var out: EguiVertexOutput;

    // Transform from screen space [0, screen_size] to clip space [-1, 1]
    let pos = vec2<f32>(
        2.0 * in.position.x / uniforms.screen_size.x - 1.0,
        1.0 - 2.0 * in.position.y / uniforms.screen_size.y
    );

    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    out.color = in.color;

    return out;
}

@fragment
fn fs_main(in: EguiVertexOutput) -> @location(0) vec4<f32> {
    // Sample texture (hardware converts sRGB texture to linear)
    let texture_color_linear = textureSample(egui_texture, egui_sampler, in.tex_coords);

    // Premultiply texture color by its alpha in linear space
    let texture_color_linear_premultiplied = vec4<f32>(
        texture_color_linear.rgb * texture_color_linear.a,
        texture_color_linear.a
    );

    // Convert premultiplied texture to gamma/sRGB space
    let texture_color_gamma_premultiplied = vec4<f32>(
        linear_to_srgb(texture_color_linear_premultiplied.rgb),
        texture_color_linear_premultiplied.a
    );

    // Multiply with vertex color in GAMMA space (vertex colors are already in sRGB)
    let color_gamma = texture_color_gamma_premultiplied * in.color;

    // Output in gamma space directly (framebuffer is Bgra8Unorm, not sRGB)
    // This matches egui-wgpu's fs_main_gamma_framebuffer
    return color_gamma;
}
