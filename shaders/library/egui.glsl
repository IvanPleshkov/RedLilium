// RedLilium Shader Library - Egui Module
// Complete shader for rendering egui UI elements.
// Handles vertex positions, texture coordinates, and vertex colors with alpha blending.
//
// Based on egui-wgpu's shader approach for proper alpha blending.
//
// Defines:
//   HDR_OUTPUT        - Outputs linear-space values for HDR surfaces (Rgba16Float etc.)
//   SRGB_FRAMEBUFFER  - Outputs linear-space values for sRGB surfaces (Bgra8UnormSrgb)
//                       where hardware applies the linear→sRGB conversion automatically
//   (default)         - Outputs gamma-space values for non-sRGB SDR surfaces (Bgra8Unorm)

#version 450

// =============================================================================
// Vertex Shader
// =============================================================================

#ifdef VERTEX

layout(set = 0, binding = 0) uniform EguiUniforms {
    vec2 screen_size;
    vec2 _padding;
};

layout(location = 0) in vec2 position;
layout(location = 3) in vec2 tex_coords;
layout(location = 5) in vec4 color;

layout(location = 0) out vec2 v_tex_coords;
layout(location = 1) out vec4 v_color;

void main() {
    // Transform from screen space [0, screen_size] to clip space [-1, 1]
    vec2 pos = vec2(
        2.0 * position.x / screen_size.x - 1.0,
        1.0 - 2.0 * position.y / screen_size.y
    );

    gl_Position = vec4(pos, 0.0, 1.0);
    v_tex_coords = tex_coords;
    v_color = color;
}

#endif

// =============================================================================
// Fragment Shader
// =============================================================================

#ifdef FRAGMENT

layout(set = 1, binding = 0) uniform texture2D egui_texture;
layout(set = 1, binding = 1) uniform sampler egui_sampler;

layout(location = 0) in vec2 v_tex_coords;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 out_color;

// Convert sRGB color to linear color space.
vec3 srgb_to_linear(vec3 srgb) {
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    vec3 t = step(vec3(0.04045), srgb);
    return mix(lower, higher, t);
}

// Convert linear color to sRGB color space.
vec3 linear_to_srgb(vec3 linear_color) {
    vec3 lower = linear_color * vec3(12.92);
    vec3 higher = vec3(1.055) * pow(linear_color, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 t = step(vec3(0.0031308), linear_color);
    return mix(lower, higher, t);
}

void main() {
    // Sample texture (hardware converts sRGB texture to linear)
    vec4 texture_color_linear = texture(sampler2D(egui_texture, egui_sampler), v_tex_coords);

    // Premultiply texture color by its alpha in linear space
    vec4 texture_color_linear_premultiplied = vec4(
        texture_color_linear.rgb * texture_color_linear.a,
        texture_color_linear.a
    );

#if defined(HDR_OUTPUT) || defined(SRGB_FRAMEBUFFER)
    // Linear output: for HDR surfaces or sRGB framebuffers (hardware does linear→sRGB).
    // Convert vertex colors from sRGB to linear, then multiply in linear space.
    vec4 vertex_color_linear = vec4(srgb_to_linear(v_color.rgb), v_color.a);
    out_color = texture_color_linear_premultiplied * vertex_color_linear;
#else
    // Gamma output: for non-sRGB SDR framebuffer (Bgra8Unorm).
    // Convert texture from linear to gamma, multiply with sRGB vertex colors.
    vec4 texture_color_gamma_premultiplied = vec4(
        linear_to_srgb(texture_color_linear_premultiplied.rgb),
        texture_color_linear_premultiplied.a
    );
    out_color = texture_color_gamma_premultiplied * v_color;
#endif
}

#endif
