// Entity index shader — outputs entity index as u32 to an R32Uint target for picking.
//
// Writes (entity_index + 1) so that a cleared-to-zero texture means "no entity".
//
// Binding 0: Uniforms { view_projection, model, entity_index } — per-entity uniform buffer.

struct Uniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    entity_index: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> @builtin(position) vec4<f32> {
    let world_pos = uniforms.model * vec4<f32>(position, 1.0);
    return uniforms.view_projection * world_pos;
}

@fragment
fn fs_main() -> @location(0) u32 {
    return uniforms.entity_index + 1u;
}
