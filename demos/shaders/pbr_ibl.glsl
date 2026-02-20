// PBR IBL Demo - Main PBR Shader
// Demonstrates PBR rendering with Image-Based Lighting using the RedLilium shader library.

#version 450

#include "redlilium/math.glsl"
#include "redlilium/brdf.glsl"
#include "redlilium/color.glsl"

// =============================================================================
// Shared Types
// =============================================================================

struct InstanceData {
    mat4 model;
    vec4 base_color;
    vec4 metallic_roughness;
};

// =============================================================================
// Vertex Shader
// =============================================================================

#ifdef VERTEX

layout(set = 0, binding = 0) uniform CameraUniforms {
    mat4 view_proj;
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
};

layout(std430, set = 0, binding = 1) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 3) in vec2 uv;

layout(location = 0) out vec3 v_world_position;
layout(location = 1) out vec3 v_world_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec4 v_base_color;
layout(location = 4) out float v_metallic;
layout(location = 5) out float v_roughness;

void main() {
    InstanceData instance = instances[gl_InstanceIndex];
    vec4 world_pos = instance.model * vec4(position, 1.0);
    mat3 normal_matrix = mat3(
        instance.model[0].xyz,
        instance.model[1].xyz,
        instance.model[2].xyz
    );

    gl_Position = view_proj * world_pos;
    v_world_position = world_pos.xyz;
    v_world_normal = normalize(normal_matrix * normal);
    v_uv = uv;
    v_base_color = instance.base_color;
    v_metallic = instance.metallic_roughness.x;
    v_roughness = instance.metallic_roughness.y;
}

#endif

// =============================================================================
// Fragment Shader
// =============================================================================

#ifdef FRAGMENT

layout(set = 0, binding = 0) uniform CameraUniforms {
    mat4 view_proj;
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
};

// IBL textures
layout(set = 1, binding = 0) uniform textureCube irradiance_map;
layout(set = 1, binding = 1) uniform textureCube prefilter_map;
layout(set = 1, binding = 2) uniform texture2D brdf_lut;
layout(set = 1, binding = 3) uniform sampler ibl_sampler;

layout(location = 0) in vec3 v_world_position;
layout(location = 1) in vec3 v_world_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_base_color;
layout(location = 4) in float v_metallic;
layout(location = 5) in float v_roughness;

// Fragment output structure for MRT (Multiple Render Targets)
layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_albedo;

// IBL constants
const float MAX_REFLECTION_LOD_VAL = 4.0;

void main() {
    vec3 albedo = v_base_color.rgb;
    float metallic = v_metallic;
    float roughness = max(v_roughness, 0.04);

    vec3 n = normalize(v_world_normal);
    vec3 v = normalize(camera_pos.xyz - v_world_position);
    vec3 r = reflect(-v, n);

    float n_dot_v = max(dot(n, v), 0.0);

    // Calculate F0 using library function
    vec3 f0 = calculate_f0(albedo, metallic);

    // === Direct lighting ===
    // Simple directional light (sun-like)
    vec3 light_dir = normalize(vec3(1.0, 1.0, 0.5));
    vec3 light_color = vec3(1.0, 0.98, 0.95) * 3.0;

    // Use library function for direct lighting
    vec3 lo = pbr_direct_lighting(n, v, light_dir, albedo, metallic, roughness, light_color);

    // Fill light (simple diffuse-only)
    vec3 fill_light_dir = normalize(vec3(-0.5, -0.3, -1.0));
    vec3 fill_light_color = vec3(0.3, 0.4, 0.5) * 0.5;
    float fill_n_dot_l = max(dot(n, fill_light_dir), 0.0);
    float kd_fill = (1.0 - metallic);
    lo = lo + kd_fill * albedo * INV_PI * fill_light_color * fill_n_dot_l;

    // === IBL ambient lighting ===
    // Use library functions for Fresnel
    vec3 f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    vec3 ks_ibl = f_ibl;
    vec3 kd_ibl = (vec3(1.0) - ks_ibl) * (1.0 - metallic);

    // Diffuse IBL from irradiance map
    vec3 irradiance = texture(samplerCube(irradiance_map, ibl_sampler), n).rgb;
    vec3 diffuse_ibl = irradiance * albedo;

    // Specular IBL from pre-filtered environment map + BRDF LUT
    vec3 prefiltered_color = textureLod(samplerCube(prefilter_map, ibl_sampler), r, roughness * MAX_REFLECTION_LOD_VAL).rgb;
    vec2 brdf_sample = texture(sampler2D(brdf_lut, ibl_sampler), vec2(n_dot_v, roughness)).rg;
    vec3 specular_ibl = prefiltered_color * (f_ibl * brdf_sample.x + brdf_sample.y);

    vec3 ambient = kd_ibl * diffuse_ibl + specular_ibl;

    // Combine
    vec3 color = ambient + lo;

    // HDR tonemapping using library function (Reinhard)
    color = tonemap_reinhard(color);

    // Gamma correction using library function
    color = gamma_correct(color);

    out_color = vec4(color, 1.0);
    out_albedo = vec4(albedo, 1.0);
}

#endif
