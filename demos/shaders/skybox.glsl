// PBR IBL Demo - Skybox Shader
// Renders environment cubemap as background with tone mapping and gamma correction.

#version 450

#include "redlilium/color.glsl"

// =============================================================================
// Vertex Shader
// =============================================================================

#ifdef VERTEX

layout(set = 0, binding = 0) uniform SkyboxUniforms {
    mat4 inv_view_proj;
    vec4 camera_pos;
    float mip_level;
    vec3 _pad;
};

layout(location = 0) out vec3 v_view_dir;

void main() {
    // Fullscreen triangle
    float x = float((gl_VertexIndex & 1) << 2) - 1.0;
    float y = float((gl_VertexIndex & 2) << 1) - 1.0;

    gl_Position = vec4(x, y, 0.9999, 1.0);

    // Compute view direction from clip space
    vec4 clip_pos = vec4(x, y, 1.0, 1.0);
    vec4 world_pos = inv_view_proj * clip_pos;
    v_view_dir = normalize(world_pos.xyz / world_pos.w - camera_pos.xyz);
}

#endif

// =============================================================================
// Fragment Shader
// =============================================================================

#ifdef FRAGMENT

layout(set = 0, binding = 0) uniform SkyboxUniforms {
    mat4 inv_view_proj;
    vec4 camera_pos;
    float mip_level;
    vec3 _pad;
};

layout(set = 0, binding = 1) uniform textureCube env_map;
layout(set = 0, binding = 2) uniform sampler env_sampler;

layout(location = 0) in vec3 v_view_dir;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 color = textureLod(samplerCube(env_map, env_sampler), v_view_dir, mip_level).rgb;

    // Tonemap and gamma correct using library functions
    vec3 mapped = tonemap_reinhard(color);
    vec3 corrected = gamma_correct(mapped);

    out_color = vec4(corrected, 1.0);
}

#endif
