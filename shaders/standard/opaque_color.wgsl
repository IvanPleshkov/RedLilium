// Standard opaque color shader — Blinn-Phong lighting with position + normal vertex layout.
//
// Binding 0: Uniforms { view_projection, model } — per-entity uniform buffer.

struct Uniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.model * vec4<f32>(position, 1.0);
    out.clip_position = uniforms.view_projection * world_pos;
    out.world_normal = (uniforms.model * vec4<f32>(normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let n = normalize(in.world_normal);
    let ndotl = max(dot(n, light_dir), 0.0);
    let base_color = vec3<f32>(0.6, 0.6, 0.65);
    let ambient = vec3<f32>(0.15, 0.15, 0.18);
    let color = ambient + base_color * ndotl;
    return vec4<f32>(color, 1.0);
}
