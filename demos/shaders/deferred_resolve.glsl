// Deferred Rendering - Resolve/Lighting Pass
// Reads G-buffer and applies IBL lighting to produce final image.

#version 450

#include "redlilium/math.glsl"
#include "redlilium/brdf.glsl"
#include "redlilium/color.glsl"

// =============================================================================
// Vertex Shader
// =============================================================================

#ifdef VERTEX

layout(location = 0) out vec2 v_uv;

void main() {
    // Generate fullscreen triangle
    vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );

    vec2 uvs[3] = vec2[3](
        vec2(0.0, 1.0),
        vec2(2.0, 1.0),
        vec2(0.0, -1.0)
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    v_uv = uvs[gl_VertexIndex];
}

#endif

// =============================================================================
// Fragment Shader
// =============================================================================

#ifdef FRAGMENT

layout(set = 0, binding = 0) uniform ResolveUniforms {
    vec4 camera_pos;
    vec4 screen_size; // xy = screen dimensions, zw = 1/screen dimensions
};

// G-buffer textures
layout(set = 1, binding = 0) uniform texture2D gbuffer_albedo;
layout(set = 1, binding = 1) uniform texture2D gbuffer_normal_metallic;
layout(set = 1, binding = 2) uniform texture2D gbuffer_position_roughness;
layout(set = 1, binding = 3) uniform sampler gbuffer_sampler;

// IBL textures
layout(set = 2, binding = 0) uniform textureCube irradiance_map;
layout(set = 2, binding = 1) uniform textureCube prefilter_map;
layout(set = 2, binding = 2) uniform texture2D brdf_lut;
layout(set = 2, binding = 3) uniform sampler ibl_sampler;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

// IBL constants
const float MAX_REFLECTION_LOD_VAL = 4.0;

void main() {
    // Sample G-buffer
    vec4 albedo_sample = texture(sampler2D(gbuffer_albedo, gbuffer_sampler), v_uv);
    vec4 normal_metallic_sample = texture(sampler2D(gbuffer_normal_metallic, gbuffer_sampler), v_uv);
    vec4 position_roughness_sample = texture(sampler2D(gbuffer_position_roughness, gbuffer_sampler), v_uv);

    // Check if this pixel has geometry (albedo alpha > 0)
    if (albedo_sample.a < 0.5) {
        // Background - return transparent/black
        discard;
    }

    // Decode G-buffer
    vec3 albedo = albedo_sample.rgb;
    // Decode normal from [0,1] back to [-1,1]
    vec3 normal = normalize(normal_metallic_sample.rgb * 2.0 - 1.0);
    float metallic = normal_metallic_sample.a;
    vec3 world_position = position_roughness_sample.rgb;
    float roughness = max(position_roughness_sample.a, 0.04);

    // Calculate view direction
    vec3 v = normalize(camera_pos.xyz - world_position);
    vec3 r = reflect(-v, normal);
    float n_dot_v = max(dot(normal, v), 0.0);

    // Calculate F0 using library function
    vec3 f0 = calculate_f0(albedo, metallic);

    // === Direct lighting ===
    // Simple directional light (sun-like)
    vec3 light_dir = normalize(vec3(1.0, 1.0, 0.5));
    vec3 light_color = vec3(1.0, 0.98, 0.95) * 3.0;

    // Use library function for direct lighting
    vec3 lo = pbr_direct_lighting(normal, v, light_dir, albedo, metallic, roughness, light_color);

    // Fill light (simple diffuse-only)
    vec3 fill_light_dir = normalize(vec3(-0.5, -0.3, -1.0));
    vec3 fill_light_color = vec3(0.3, 0.4, 0.5) * 0.5;
    float fill_n_dot_l = max(dot(normal, fill_light_dir), 0.0);
    float kd_fill = (1.0 - metallic);
    lo = lo + kd_fill * albedo * INV_PI * fill_light_color * fill_n_dot_l;

    // === IBL ambient lighting ===
    // Use library functions for Fresnel
    vec3 f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    vec3 ks_ibl = f_ibl;
    vec3 kd_ibl = (vec3(1.0) - ks_ibl) * (1.0 - metallic);

    // Diffuse IBL from irradiance map
    vec3 irradiance = texture(samplerCube(irradiance_map, ibl_sampler), normal).rgb;
    vec3 diffuse_ibl = irradiance * albedo;

    // Specular IBL from pre-filtered environment map + BRDF LUT
    vec3 prefiltered_color = textureLod(samplerCube(prefilter_map, ibl_sampler), r, roughness * MAX_REFLECTION_LOD_VAL).rgb;
    vec2 brdf_sample = texture(sampler2D(brdf_lut, ibl_sampler), vec2(n_dot_v, roughness)).rg;
    vec3 specular_ibl = prefiltered_color * (f_ibl * brdf_sample.x + brdf_sample.y);

    vec3 ambient = kd_ibl * diffuse_ibl + specular_ibl;

    // Combine
    vec3 color = ambient + lo;

#ifdef HDR_OUTPUT
    // HDR output: skip tone mapping and gamma correction
    // Clamp to reasonable HDR range (avoid extreme values)
    color = clamp(color, vec3(0.0), vec3(10.0));
#else
    // SDR output: apply tonemapping and gamma correction
    // HDR tonemapping using library function (Reinhard)
    color = tonemap_reinhard(color);

    // Gamma correction using library function
    color = gamma_correct(color);
#endif

    out_color = vec4(color, 1.0);
}

#endif
