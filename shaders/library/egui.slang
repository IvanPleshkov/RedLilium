// RedLilium Shader Library - Egui Module
// Complete shader for rendering egui UI elements.
// Handles vertex positions, texture coordinates, and vertex colors with alpha blending.
//
// Based on egui-wgpu's shader approach for proper alpha blending.
//
// Defines:
//   HDR_OUTPUT        - Outputs linear-space values for HDR surfaces (Rgba16Float etc.)
//   SRGB_FRAMEBUFFER  - Outputs linear-space values for sRGB surfaces (Bgra8UnormSrgb)
//                       where hardware applies the linear->sRGB conversion automatically
//   (default)         - Outputs gamma-space values for non-sRGB SDR surfaces (Bgra8Unorm)

// =============================================================================
// Uniforms and Textures
// =============================================================================

[[vk::binding(0, 0)]]
cbuffer EguiUniforms {
    float2 screen_size;
    float2 _padding;
};

[[vk::binding(0, 1)]]
Texture2D egui_texture;
[[vk::binding(1, 1)]]
SamplerState egui_sampler;

// =============================================================================
// Vertex Shader
// =============================================================================

struct VsInput {
    [[vk::location(0)]] float2 position : POSITION;
    [[vk::location(3)]] float2 tex_coords : TEXCOORD0;
    [[vk::location(5)]] float4 color : COLOR;
};

struct VsOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 tex_coords : TEXCOORD0;
    [[vk::location(1)]] float4 color : COLOR;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;

    // Transform from screen space [0, screen_size] to clip space [-1, 1]
    float2 pos = float2(
        2.0 * input.position.x / screen_size.x - 1.0,
        1.0 - 2.0 * input.position.y / screen_size.y
    );

    output.position = float4(pos, 0.0, 1.0);
    output.tex_coords = input.tex_coords;
    output.color = input.color;
    return output;
}

// =============================================================================
// Fragment Shader Utilities
// =============================================================================

// Convert sRGB color to linear color space.
float3 srgb_to_linear(float3 srgb) {
    float3 lower = srgb / float3(12.92);
    float3 higher = pow((srgb + float3(0.055)) / float3(1.055), float3(2.4));
    float3 t = step(float3(0.04045), srgb);
    return lerp(lower, higher, t);
}

// Convert linear color to sRGB color space.
float3 linear_to_srgb(float3 linear_color) {
    float3 lower = linear_color * float3(12.92);
    float3 higher = float3(1.055) * pow(linear_color, float3(1.0 / 2.4)) - float3(0.055);
    float3 t = step(float3(0.0031308), linear_color);
    return lerp(lower, higher, t);
}

// =============================================================================
// Fragment Shader
// =============================================================================

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    // Sample texture (hardware converts sRGB texture to linear)
    float4 texture_color_linear = egui_texture.Sample(egui_sampler, input.tex_coords);

    // Premultiply texture color by its alpha in linear space
    float4 texture_color_linear_premultiplied = float4(
        texture_color_linear.rgb * texture_color_linear.a,
        texture_color_linear.a
    );

#if defined(HDR_OUTPUT) || defined(SRGB_FRAMEBUFFER)
    // Linear output: for HDR surfaces or sRGB framebuffers (hardware does linear->sRGB).
    // Convert vertex colors from sRGB to linear, then multiply in linear space.
    float4 vertex_color_linear = float4(srgb_to_linear(input.color.rgb), input.color.a);
    return texture_color_linear_premultiplied * vertex_color_linear;
#else
    // Gamma output: for non-sRGB SDR framebuffer (Bgra8Unorm).
    // Convert texture from linear to gamma, multiply with sRGB vertex colors.
    float4 texture_color_gamma_premultiplied = float4(
        linear_to_srgb(texture_color_linear_premultiplied.rgb),
        texture_color_linear_premultiplied.a
    );
    return texture_color_gamma_premultiplied * input.color;
#endif
}
