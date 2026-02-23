//! Standard opaque color material with Blinn-Phong lighting.
//!
//! Provides a simple lit material using position + normal vertex layout.
//! The shader uses per-entity uniform buffers containing view-projection
//! and model matrices.
//!
//! # Usage
//!
//! ```ignore
//! // At init time:
//! let (material, _layout) = create_opaque_color_material(&device, color_fmt, depth_fmt);
//!
//! // Per entity:
//! let (buffer, bundle) = create_opaque_color_entity(&device, &material);
//! world.insert(entity, RenderMaterial::new(bundle));
//!
//! // Each frame:
//! update_opaque_color_uniforms(&device, &world, &entity_buffers);
//! ```

use std::sync::Arc;

use redlilium_core::math::{Mat4, mat4_to_cols_array_2d};
use redlilium_graphics::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, Buffer, BufferDescriptor,
    BufferUsage, GraphicsDevice, Material, MaterialDescriptor, MaterialInstance, ShaderSource,
    ShaderStage, ShaderStageFlags, TextureFormat, VertexLayout,
};

use crate::Entity;
use crate::std::components::{Camera, GlobalTransform};

use super::super::components::{MaterialBundle, RenderPassType};

/// WGSL shader for opaque color rendering with camera VP + model matrix uniforms.
const SHADER_WGSL: &str = r#"
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
"#;

/// Per-entity uniform data: view-projection matrix + model matrix.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OpaqueColorUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
}

/// Create the GPU [`Material`] for the opaque color shader.
///
/// Returns `(material, binding_layout)`. The binding layout has a single
/// uniform buffer at slot 0 visible to both vertex and fragment stages.
pub fn create_opaque_color_material(
    device: &Arc<GraphicsDevice>,
    color_format: TextureFormat,
    depth_format: TextureFormat,
) -> (Arc<Material>, Arc<BindingLayout>) {
    let binding_layout = Arc::new(
        BindingLayout::new().with_entry(
            BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
        ),
    );

    let material = device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::new(
                    ShaderStage::Vertex,
                    SHADER_WGSL.as_bytes().to_vec(),
                    "vs_main",
                ))
                .with_shader(ShaderSource::new(
                    ShaderStage::Fragment,
                    SHADER_WGSL.as_bytes().to_vec(),
                    "fs_main",
                ))
                .with_binding_layout(binding_layout.clone())
                .with_vertex_layout(VertexLayout::position_normal())
                .with_color_format(color_format)
                .with_depth_format(depth_format)
                .with_label("std_opaque_color"),
        )
        .expect("Failed to create opaque color material");

    (material, binding_layout)
}

/// Create per-entity GPU resources for the opaque color material.
///
/// Returns `(uniform_buffer, material_bundle)`. The uniform buffer should
/// be kept alongside the entity for per-frame updates via
/// [`update_opaque_color_uniforms`].
pub fn create_opaque_color_entity(
    device: &Arc<GraphicsDevice>,
    material: &Arc<Material>,
) -> (Arc<Buffer>, Arc<MaterialBundle>) {
    let uniform_buffer = device
        .create_buffer(&BufferDescriptor::new(
            std::mem::size_of::<OpaqueColorUniforms>() as u64,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        ))
        .expect("Failed to create opaque color uniform buffer");

    let binding_group = Arc::new(BindingGroup::new().with_buffer(0, uniform_buffer.clone()));

    let instance = Arc::new(
        MaterialInstance::new(Arc::clone(material)).with_binding_group(Arc::clone(&binding_group)),
    );

    let bundle = Arc::new(
        MaterialBundle::new()
            .with_pass(RenderPassType::Forward, instance)
            .with_shared_bindings(vec![binding_group]),
    );

    (uniform_buffer, bundle)
}

/// Update per-entity uniform buffers with the current camera VP and model matrices.
///
/// Uses `read_all::<Camera>` so editor-flagged cameras are included.
pub fn update_opaque_color_uniforms(
    device: &Arc<GraphicsDevice>,
    world: &crate::World,
    entity_buffers: &[(Entity, Arc<Buffer>)],
) {
    let Ok(cameras) = world.read_all::<Camera>() else {
        return;
    };
    let Some((_, camera)) = cameras.iter().next() else {
        return;
    };
    let vp = camera.view_projection();

    let Ok(globals) = world.read::<GlobalTransform>() else {
        return;
    };
    for (entity, buffer) in entity_buffers {
        let model = globals
            .get(entity.index())
            .map(|g| g.0)
            .unwrap_or_else(Mat4::identity);

        let uniforms = OpaqueColorUniforms {
            view_projection: mat4_to_cols_array_2d(&vp),
            model: mat4_to_cols_array_2d(&model),
        };

        let _ = device.write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}
