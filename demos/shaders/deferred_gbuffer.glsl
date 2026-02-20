// Deferred Rendering - G-Buffer Pass
// Outputs material properties to multiple render targets for later lighting calculation.

#version 450

#include "redlilium/math.glsl"

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

layout(location = 0) in vec3 v_world_position;
layout(location = 1) in vec3 v_world_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_base_color;
layout(location = 4) in float v_metallic;
layout(location = 5) in float v_roughness;

// G-Buffer output (Multiple Render Targets)
// RT0: Albedo (RGB) - sRGB color space
layout(location = 0) out vec4 out_albedo;
// RT1: World Normal (RGB) + Metallic (A) - linear, high precision
layout(location = 1) out vec4 out_normal_metallic;
// RT2: World Position (RGB) + Roughness (A) - linear, high precision
layout(location = 2) out vec4 out_position_roughness;

void main() {
    vec3 albedo = v_base_color.rgb;
    float metallic = v_metallic;
    float roughness = max(v_roughness, 0.04);
    vec3 normal = normalize(v_world_normal);

    // RT0: Albedo
    out_albedo = vec4(albedo, 1.0);

    // RT1: Normal (encoded to [0,1] range) + Metallic
    vec3 encoded_normal = normal * 0.5 + 0.5;
    out_normal_metallic = vec4(encoded_normal, metallic);

    // RT2: World Position + Roughness
    out_position_roughness = vec4(v_world_position, roughness);
}

#endif
