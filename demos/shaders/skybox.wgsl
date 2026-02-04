// PBR IBL Demo - Skybox Shader
// Renders environment cubemap as background with tone mapping and gamma correction.

#import redlilium::color::{tonemap_reinhard, gamma_correct}

struct SkyboxUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    mip_level: f32,
    _pad: vec3<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: SkyboxUniforms;
@group(0) @binding(1) var env_map: texture_cube<f32>;
@group(0) @binding(2) var env_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) view_dir: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;

    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.9999, 1.0);

    // Compute view direction from clip space
    let clip_pos = vec4<f32>(x, y, 1.0, 1.0);
    let world_pos = uniforms.inv_view_proj * clip_pos;
    out.view_dir = normalize(world_pos.xyz / world_pos.w - uniforms.camera_pos.xyz);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(env_map, env_sampler, in.view_dir, uniforms.mip_level).rgb;

    // Tonemap and gamma correct using library functions
    let mapped = tonemap_reinhard(color);
    let corrected = gamma_correct(mapped);

    return vec4<f32>(corrected, 1.0);
}
