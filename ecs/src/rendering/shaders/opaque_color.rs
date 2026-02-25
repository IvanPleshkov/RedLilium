//! Standard opaque color material with Blinn-Phong lighting.
//!
//! Provides a simple lit material using position + normal vertex layout.
//! The shader uses per-entity uniform buffers containing view-projection
//! and model matrices, plus a material properties uniform with base color.
//!
//! # Usage
//!
//! ```ignore
//! // At init time:
//! let (material, _layout) = create_opaque_color_material(&device, color_fmt, depth_fmt);
//! let cpu_material = create_opaque_color_cpu_material();
//!
//! // Per entity:
//! let (per_entity, render_mat, bundle) =
//!     create_opaque_color_entity_full(&device, &material, &ei_material, &cpu_material);
//! world.insert(entity, render_mat);
//! world.insert(entity, per_entity);
//! ```

use std::sync::Arc;

use redlilium_core::material::{
    CpuMaterial, CpuMaterialInstance, MaterialBindingDef, MaterialValueType,
};
use redlilium_core::math::{Mat4, mat4_to_cols_array_2d};
use redlilium_graphics::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, Buffer, BufferDescriptor,
    BufferUsage, GraphicsDevice, Material, MaterialDescriptor, MaterialInstance, ShaderSource,
    ShaderStage, ShaderStageFlags, TextureFormat, VertexLayout,
};

use crate::Entity;
use crate::rendering::RenderMaterial;
use crate::std::components::{Camera, GlobalTransform};

use super::super::components::{MaterialBundle, PerEntityBuffers, RenderPassType};

/// WGSL shader for opaque color rendering with camera VP + model matrix uniforms.
const SHADER_WGSL: &str = include_str!("../../../../shaders/standard/opaque_color.wgsl");

/// Default base color: light gray matching the original hardcoded value.
const DEFAULT_BASE_COLOR: [f32; 4] = [0.6, 0.6, 0.65, 1.0];

/// Per-entity uniform data: view-projection matrix + model matrix.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OpaqueColorUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
}

/// Create the GPU [`Material`] for the opaque color shader.
///
/// Returns `(material, binding_layout)`. The material has two binding layouts:
/// - Group 0: per-entity transform uniforms (VP + model)
/// - Group 1: material property uniforms (base_color)
pub fn create_opaque_color_material(
    device: &Arc<GraphicsDevice>,
    color_format: TextureFormat,
    depth_format: TextureFormat,
) -> (Arc<Material>, Arc<BindingLayout>) {
    // Group 0: per-entity transform uniforms
    let binding_layout = Arc::new(
        BindingLayout::new().with_entry(
            BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
        ),
    );

    // Group 1: material property uniforms (base_color)
    let material_binding_layout = Arc::new(
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
                .with_binding_layout(binding_layout.clone()) // group 0
                .with_binding_layout(material_binding_layout) // group 1
                .with_vertex_layout(VertexLayout::position_normal())
                .with_color_format(color_format)
                .with_depth_format(depth_format)
                .with_label("std_opaque_color"),
        )
        .expect("Failed to create opaque color material");

    (material, binding_layout)
}

/// Create the CPU-side material definition for the opaque color shader.
///
/// Describes a single `base_color` Vec4 binding at slot 0. Used with
/// [`CpuMaterialInstance`] to provide inspector-editable material properties.
pub fn create_opaque_color_cpu_material() -> Arc<CpuMaterial> {
    Arc::new(CpuMaterial {
        name: Some("opaque_color".into()),
        bindings: vec![MaterialBindingDef {
            name: "base_color".into(),
            value_type: MaterialValueType::Vec4,
            binding: 0,
        }],
        ..CpuMaterial::new()
    })
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

    let transform_group = Arc::new(BindingGroup::new().with_buffer(0, uniform_buffer.clone()));

    // Material props buffer (base_color)
    let mat_props_buffer = create_material_props_buffer(device);
    let mat_props_group = Arc::new(BindingGroup::new().with_buffer(0, mat_props_buffer));

    let instance = Arc::new(
        MaterialInstance::new(Arc::clone(material))
            .with_binding_group(Arc::clone(&transform_group)) // group 0
            .with_binding_group(mat_props_group), // group 1
    );

    let bundle = Arc::new(
        MaterialBundle::new()
            .with_pass(RenderPassType::Forward, instance)
            .with_shared_bindings(vec![transform_group]),
    );

    (uniform_buffer, bundle)
}

/// Extended version of [`create_opaque_color_entity`] that also adds an
/// [`EntityIndex`](RenderPassType::EntityIndex) pass for picking.
///
/// Returns `(forward_buffer, entity_index_buffer, material_props_buffer, material_bundle)`.
pub fn create_opaque_color_entity_with_picking(
    device: &Arc<GraphicsDevice>,
    forward_material: &Arc<Material>,
    entity_index_material: &Arc<Material>,
) -> (Arc<Buffer>, Arc<Buffer>, Arc<Buffer>, Arc<MaterialBundle>) {
    let uniform_buffer = device
        .create_buffer(&BufferDescriptor::new(
            std::mem::size_of::<OpaqueColorUniforms>() as u64,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        ))
        .expect("Failed to create opaque color uniform buffer");

    let transform_group = Arc::new(BindingGroup::new().with_buffer(0, uniform_buffer.clone()));

    // Material props buffer (base_color)
    let mat_props_buffer = create_material_props_buffer(device);
    let mat_props_group = Arc::new(BindingGroup::new().with_buffer(0, mat_props_buffer.clone()));

    let forward_instance = Arc::new(
        MaterialInstance::new(Arc::clone(forward_material))
            .with_binding_group(Arc::clone(&transform_group)) // group 0
            .with_binding_group(mat_props_group), // group 1
    );

    let (ei_buffer, ei_instance) =
        super::entity_index::create_entity_index_instance(device, entity_index_material);

    let bundle = Arc::new(
        MaterialBundle::new()
            .with_pass(RenderPassType::Forward, forward_instance)
            .with_pass(RenderPassType::EntityIndex, ei_instance)
            .with_shared_bindings(vec![transform_group]),
    );

    (uniform_buffer, ei_buffer, mat_props_buffer, bundle)
}

/// Create per-entity GPU resources for opaque color with picking, returning
/// components ready for ECS insertion.
///
/// Returns `(per_entity_buffers, render_material, material_bundle)`.
/// The `PerEntityBuffers` and `RenderMaterial` should be inserted as
/// components; the [`UpdatePerEntityUniforms`](super::super::UpdatePerEntityUniforms)
/// and [`SyncMaterialUniforms`](super::super::SyncMaterialUniforms)
/// systems will handle GPU updates automatically.
pub fn create_opaque_color_entity_full(
    device: &Arc<GraphicsDevice>,
    forward_material: &Arc<Material>,
    entity_index_material: &Arc<Material>,
    cpu_material: &Arc<CpuMaterial>,
) -> (PerEntityBuffers, RenderMaterial, Arc<MaterialBundle>) {
    let (fwd_buf, ei_buf, mat_props_buf, bundle) =
        create_opaque_color_entity_with_picking(device, forward_material, entity_index_material);
    let per_entity = PerEntityBuffers::with_entity_index(fwd_buf, ei_buf);

    let cpu_instance = Arc::new(
        CpuMaterialInstance::new(Arc::clone(cpu_material)).with_value(
            0,
            redlilium_core::material::MaterialValue::Vec4(DEFAULT_BASE_COLOR),
        ),
    );

    let render_material = RenderMaterial::with_cpu_data(
        Arc::clone(&bundle),
        cpu_instance,
        vec![(RenderPassType::Forward, "opaque_color".into())],
    )
    .with_material_uniform_buffer(mat_props_buf);

    (per_entity, render_material, bundle)
}

/// Create the material properties GPU buffer with default base_color.
fn create_material_props_buffer(device: &Arc<GraphicsDevice>) -> Arc<Buffer> {
    let buffer = device
        .create_buffer(
            &BufferDescriptor::new(
                std::mem::size_of::<[f32; 4]>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            )
            .with_label("opaque_color_material_props"),
        )
        .expect("Failed to create material props buffer");
    let _ = device.write_buffer(&buffer, 0, bytemuck::bytes_of(&DEFAULT_BASE_COLOR));
    buffer
}

/// Update per-entity uniform buffers with the current camera VP and model matrices.
///
/// Uses `read_all::<Camera>` so editor-flagged cameras are included.
///
/// **Deprecated:** Use the [`UpdatePerEntityUniforms`](super::super::UpdatePerEntityUniforms)
/// system with [`PerEntityBuffers`] components instead.
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
