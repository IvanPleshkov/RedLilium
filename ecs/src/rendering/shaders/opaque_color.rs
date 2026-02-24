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
const SHADER_WGSL: &str = include_str!("../../../../shaders/standard/opaque_color.wgsl");

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
///
/// If `entity_index_material` is provided, the bundle will also include an
/// [`EntityIndex`](RenderPassType::EntityIndex) pass for object picking.
/// In that case the returned buffer list contains two buffers: the forward
/// uniform buffer **and** the entity-index uniform buffer (in that order).
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

/// Extended version of [`create_opaque_color_entity`] that also adds an
/// [`EntityIndex`](RenderPassType::EntityIndex) pass for picking.
///
/// Returns `(forward_buffer, entity_index_buffer, material_bundle)`.
/// Both buffers must be updated each frame — forward via
/// [`update_opaque_color_uniforms`] and entity-index via
/// [`super::entity_index::update_entity_index_uniforms`].
pub fn create_opaque_color_entity_with_picking(
    device: &Arc<GraphicsDevice>,
    forward_material: &Arc<Material>,
    entity_index_material: &Arc<Material>,
) -> (Arc<Buffer>, Arc<Buffer>, Arc<MaterialBundle>) {
    let uniform_buffer = device
        .create_buffer(&BufferDescriptor::new(
            std::mem::size_of::<OpaqueColorUniforms>() as u64,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        ))
        .expect("Failed to create opaque color uniform buffer");

    let binding_group = Arc::new(BindingGroup::new().with_buffer(0, uniform_buffer.clone()));

    let forward_instance = Arc::new(
        MaterialInstance::new(Arc::clone(forward_material))
            .with_binding_group(Arc::clone(&binding_group)),
    );

    let (ei_buffer, ei_instance) =
        super::entity_index::create_entity_index_instance(device, entity_index_material);

    let bundle = Arc::new(
        MaterialBundle::new()
            .with_pass(RenderPassType::Forward, forward_instance)
            .with_pass(RenderPassType::EntityIndex, ei_instance)
            .with_shared_bindings(vec![binding_group]),
    );

    (uniform_buffer, ei_buffer, bundle)
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
