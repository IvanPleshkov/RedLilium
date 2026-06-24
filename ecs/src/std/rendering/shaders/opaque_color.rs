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
//! let material = create_opaque_color_material(&device, color_fmt, depth_fmt);
//! let cpu_material = create_opaque_color_cpu_material();
//!
//! // Per primitive — group 0 binds the shared transform rings as dynamic
//! // uniforms; the caller fills a ring slot per frame and draws with its offset.
//! let prim_mat = create_opaque_color_primitive_material_ring(
//!     &device, &material, Some(&ei_material), &cpu_material,
//!     forward_ring.buffer(), Some(entity_index_ring.buffer()),
//! );
//! ```

use std::sync::Arc;

use redlilium_core::material::{
    CpuMaterial, CpuMaterialInstance, MaterialBindingDef, MaterialValueType,
};
use redlilium_graphics::{
    BindingGroup, Buffer, GraphicsDevice, Material, MaterialDescriptor, MaterialInstance,
    ShaderSource, ShaderStage, TextureFormat, VertexLayout,
};

use crate::std::rendering::components::{MaterialBundle, PrimitiveMaterial, RenderPassType};

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
                // Group 0 binding 0 (per-entity transform) and group 1 binding 0
                // (material props) are both bound with per-draw dynamic offsets
                // (ring-allocated each frame).
                .with_dynamic_uniform(0, 0)
                .with_dynamic_uniform(1, 0)
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

/// Build a primitive material whose per-entity transform (group 0) is bound to
/// shared ring buffers as a **dynamic** uniform. Each draw selects its entity's
/// slot via `DrawCommand::dynamic_offsets`; the rings are filled per frame with
/// `RingBuffer::write` (no synchronous GPU writes, race-free across frames).
///
/// `forward_ring` holds [`OpaqueColorUniforms`] elements; `entity_index_ring`
/// (when picking) holds [`EntityIndexUniforms`](super::entity_index::EntityIndexUniforms).
pub fn create_opaque_color_primitive_material_ring(
    forward_material: &Arc<Material>,
    entity_index_material: Option<&Arc<Material>>,
    cpu_material: &Arc<CpuMaterial>,
    forward_ring: &Arc<Buffer>,
    entity_index_ring: Option<&Arc<Buffer>>,
    material_props_ring: &Arc<Buffer>,
) -> PrimitiveMaterial {
    // Group 1 (material props) binds one element of the props ring; the per-draw
    // dynamic offset selects this primitive's slot (filled each frame).
    let props_size = std::mem::size_of::<[f32; 4]>() as u64;
    let mat_props_group = Arc::new(BindingGroup::new().with_buffer_range(
        0,
        material_props_ring.clone(),
        0,
        props_size,
    ));

    // Group 0 binds one element of the forward ring; the per-draw dynamic offset
    // selects the entity's slot.
    let fwd_size = std::mem::size_of::<OpaqueColorUniforms>() as u64;
    let forward_group =
        Arc::new(BindingGroup::new().with_buffer_range(0, forward_ring.clone(), 0, fwd_size));
    let forward_instance = Arc::new(
        MaterialInstance::new(Arc::clone(forward_material))
            .with_binding_group(Arc::clone(&forward_group))
            .with_binding_group(mat_props_group),
    );

    let mut bundle = MaterialBundle::new()
        .with_pass(RenderPassType::Forward, forward_instance)
        .with_shared_bindings(vec![forward_group]);

    if let (Some(ei_material), Some(ei_ring)) = (entity_index_material, entity_index_ring) {
        let ei_size = std::mem::size_of::<super::entity_index::EntityIndexUniforms>() as u64;
        let ei_group =
            Arc::new(BindingGroup::new().with_buffer_range(0, ei_ring.clone(), 0, ei_size));
        let ei_instance =
            Arc::new(MaterialInstance::new(Arc::clone(ei_material)).with_binding_group(ei_group));
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
}
