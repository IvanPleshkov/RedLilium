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
    BindingGroup, Buffer, BufferDescriptor, BufferUsage, GraphicsDevice, Material,
    MaterialDescriptor, MaterialInstance, ShaderSource, ShaderStage, TextureFormat, VertexLayout,
};

use crate::Entity;
use crate::std::components::{Camera, GlobalTransform};
use crate::std::rendering::components::{
    MaterialBundle, PerEntityBuffers, PrimitiveMaterial, RenderPassType,
};

/// Slang shader for opaque color rendering with camera VP + model matrix uniforms.
const SHADER_SLANG: &str = include_str!("../../../../../shaders/standard/opaque_color.slang");

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
/// The material has two binding groups auto-reflected from the Slang shader:
/// - Group 0: per-entity transform uniforms (VP + model)
/// - Group 1: material property uniforms (base_color)
pub fn create_opaque_color_material(
    device: &Arc<GraphicsDevice>,
    color_format: TextureFormat,
    depth_format: TextureFormat,
) -> Arc<Material> {
    device
        .create_material(
            &MaterialDescriptor::new()
                .with_shader(ShaderSource::slang(
                    ShaderStage::Vertex,
                    SHADER_SLANG.as_bytes().to_vec(),
                    "vs_main",
                    vec![],
                ))
                .with_shader(ShaderSource::slang(
                    ShaderStage::Fragment,
                    SHADER_SLANG.as_bytes().to_vec(),
                    "fs_main",
                    vec![],
                ))
                .with_vertex_layout(VertexLayout::position_normal())
                .with_color_format(color_format)
                .with_depth_format(depth_format)
                .with_label("std_opaque_color"),
        )
        .expect("Failed to create opaque color material")
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

/// Per-entity transform GPU resources, shared by every [`Primitive`] of one
/// entity. The forward + entity-index uniform buffers hold the (per-entity)
/// view-projection / model matrices, so all primitives bind the same group 0.
///
/// [`Primitive`]: crate::std::rendering::components::Primitive
pub struct EntityTransform {
    /// The per-entity uniform buffers (forward + optional entity-index).
    pub buffers: PerEntityBuffers,
    /// Binding group 0 for the forward pass.
    forward_group: Arc<BindingGroup>,
    /// Binding group 0 for the entity-index (picking) pass, if enabled.
    entity_index_group: Option<Arc<BindingGroup>>,
}

/// Create the per-entity transform buffers + their binding groups. When
/// `picking` is set, also creates the entity-index transform buffer used by the
/// picking pass.
pub fn create_entity_transform(device: &Arc<GraphicsDevice>, picking: bool) -> EntityTransform {
    let fwd_buf = device
        .create_buffer(&BufferDescriptor::new(
            std::mem::size_of::<OpaqueColorUniforms>() as u64,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        ))
        .expect("Failed to create opaque color uniform buffer");
    let forward_group = Arc::new(BindingGroup::new().with_buffer(0, fwd_buf.clone()));

    if picking {
        let ei_buf = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<super::entity_index::EntityIndexUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create entity index uniform buffer");
        let entity_index_group = Arc::new(BindingGroup::new().with_buffer(0, ei_buf.clone()));
        EntityTransform {
            buffers: PerEntityBuffers::with_entity_index(fwd_buf, ei_buf),
            forward_group,
            entity_index_group: Some(entity_index_group),
        }
    } else {
        EntityTransform {
            buffers: PerEntityBuffers::new(fwd_buf),
            forward_group,
            entity_index_group: None,
        }
    }
}

/// Build a single primitive's [`PrimitiveMaterial`] that shares the entity's
/// `transform` (group 0) and owns its own material-property buffer (group 1).
/// Adds an [`EntityIndex`](RenderPassType::EntityIndex) pass when both
/// `entity_index_material` and the transform's picking group are present.
pub fn create_opaque_color_primitive_material(
    device: &Arc<GraphicsDevice>,
    forward_material: &Arc<Material>,
    entity_index_material: Option<&Arc<Material>>,
    cpu_material: &Arc<CpuMaterial>,
    transform: &EntityTransform,
) -> PrimitiveMaterial {
    // Material props buffer (base_color) — per primitive.
    let mat_props_buffer = create_material_props_buffer(device);
    let mat_props_group = Arc::new(BindingGroup::new().with_buffer(0, mat_props_buffer.clone()));

    let forward_instance = Arc::new(
        MaterialInstance::new(Arc::clone(forward_material))
            .with_binding_group(Arc::clone(&transform.forward_group)) // group 0 (shared)
            .with_binding_group(mat_props_group), // group 1 (per-primitive)
    );

    let mut bundle = MaterialBundle::new()
        .with_pass(RenderPassType::Forward, forward_instance)
        .with_shared_bindings(vec![Arc::clone(&transform.forward_group)]);

    if let (Some(ei_material), Some(ei_group)) =
        (entity_index_material, &transform.entity_index_group)
    {
        let ei_instance = Arc::new(
            MaterialInstance::new(Arc::clone(ei_material)).with_binding_group(Arc::clone(ei_group)), // group 0 (shared)
        );
        bundle = bundle.with_pass(RenderPassType::EntityIndex, ei_instance);
    }

    let cpu_instance = Arc::new(
        CpuMaterialInstance::new(Arc::clone(cpu_material)).with_value(
            0,
            redlilium_core::material::MaterialValue::Vec4(DEFAULT_BASE_COLOR),
        ),
    );

    PrimitiveMaterial::with_cpu_data(
        Arc::new(bundle),
        cpu_instance,
        vec![(RenderPassType::Forward, "opaque_color".into())],
    )
    .with_material_uniform_buffer(mat_props_buffer)
}

/// Convenience builder for a single-primitive entity: creates the per-entity
/// transform (with picking) plus one [`PrimitiveMaterial`].
///
/// Returns `(per_entity_buffers, primitive_material)`. The caller wraps the
/// material in a [`Primitive`](crate::std::rendering::components::Primitive) +
/// [`MeshRenderer`](crate::std::rendering::components::MeshRenderer).
pub fn create_opaque_color_entity_full(
    device: &Arc<GraphicsDevice>,
    forward_material: &Arc<Material>,
    entity_index_material: &Arc<Material>,
    cpu_material: &Arc<CpuMaterial>,
) -> (PerEntityBuffers, PrimitiveMaterial) {
    let transform = create_entity_transform(device, true);
    let material = create_opaque_color_primitive_material(
        device,
        forward_material,
        Some(entity_index_material),
        cpu_material,
        &transform,
    );
    (transform.buffers, material)
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
