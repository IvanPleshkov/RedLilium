//! Entity index material — renders entity ID to an R32Uint target for picking.
//!
//! Uses the same position + normal vertex layout and view-projection / model
//! uniforms as [`super::opaque_color`], with an additional `entity_index: u32`
//! field. The fragment shader outputs the raw entity index so a readback or
//! copy from the picking texture can identify which entity was clicked.

use std::sync::Arc;

use redlilium_core::math::{Mat4, mat4_to_cols_array_2d};
use redlilium_graphics::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, Buffer, BufferDescriptor,
    BufferUsage, GraphicsDevice, Material, MaterialDescriptor, MaterialInstance, ShaderSource,
    ShaderStage, ShaderStageFlags, TextureFormat, VertexLayout,
};

use crate::Entity;
use crate::std::components::{Camera, GlobalTransform};

/// WGSL shader that outputs entity index as `u32` to an R32Uint color target.
const SHADER_WGSL: &str = include_str!("../../../../shaders/standard/entity_index.wgsl");

/// Per-entity uniform data: view-projection, model matrix, and entity index.
///
/// Layout must match the WGSL `Uniforms` struct. The `_padding` fields ensure
/// 16-byte alignment after the `u32` entity index.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntityIndexUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub entity_index: u32,
    pub _padding: [u32; 3],
}

/// Create the GPU [`Material`] for the entity index shader.
///
/// The material renders to an `R32Uint` color target (no blending) with the
/// given depth format. Returns `(material, binding_layout)`.
pub fn create_entity_index_material(
    device: &Arc<GraphicsDevice>,
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
                .with_color_format(TextureFormat::R32Uint)
                .with_depth_format(depth_format)
                .with_label("std_entity_index"),
        )
        .expect("Failed to create entity index material");

    (material, binding_layout)
}

/// Create per-entity GPU resources for the entity index material.
///
/// Returns `(uniform_buffer, material_instance)`. The buffer is written each
/// frame by [`update_entity_index_uniforms`].
pub fn create_entity_index_instance(
    device: &Arc<GraphicsDevice>,
    material: &Arc<Material>,
) -> (Arc<Buffer>, Arc<MaterialInstance>) {
    let uniform_buffer = device
        .create_buffer(&BufferDescriptor::new(
            std::mem::size_of::<EntityIndexUniforms>() as u64,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        ))
        .expect("Failed to create entity index uniform buffer");

    let binding_group = Arc::new(BindingGroup::new().with_buffer(0, uniform_buffer.clone()));

    let instance =
        Arc::new(MaterialInstance::new(Arc::clone(material)).with_binding_group(binding_group));

    (uniform_buffer, instance)
}

/// Update per-entity uniform buffers with camera VP, model matrix, and entity index.
pub fn update_entity_index_uniforms(
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

        let uniforms = EntityIndexUniforms {
            view_projection: mat4_to_cols_array_2d(&vp),
            model: mat4_to_cols_array_2d(&model),
            entity_index: entity.index(),
            _padding: [0; 3],
        };

        let _ = device.write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
    }
}
