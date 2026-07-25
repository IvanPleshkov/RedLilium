//! Entity index material — renders entity ID to an R32Uint target for picking,
//! plus an R8Unorm selection mask for the editor's selection outline (MRT).
//!
//! The fragment shader outputs the raw entity index so a readback or copy from
//! the picking texture can identify which entity was clicked, and writes the
//! selection flag to the mask target — unselected occluders overwrite it under
//! the shared depth test, so the mask holds the visible part of the selected
//! silhouette.
//!
//! The material is specialized **per mesh vertex layout** (the pipeline's
//! vertex-input state comes from the material's layout — a fixed layout
//! misreads any mesh with a different stride; a generated sphere used to pick
//! as a ~2x vertex soup). The shader consumes only location 0, and both
//! backends number shader locations sequentially in layout attribute order
//! (ADR-019), so a layout is drawable iff its first attribute is the position.

use std::sync::Arc;

use redlilium_graphics::{
    GraphicsDevice, Material, MaterialDescriptor, ShaderSource, ShaderStage, TextureFormat,
    VertexLayout,
};

/// Slang shader that outputs entity index as `u32` to an R32Uint color target.
const SHADER_SLANG: &str = include_str!("../../../../../std-assets/shaders/entity_index.slang");

/// Per-entity uniform data: view-projection, model matrix, entity index, and
/// the selection flag feeding the mask target.
///
/// Layout must match the Slang `Uniforms` cbuffer. The `_padding` fields ensure
/// 16-byte alignment after the `u32` fields.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntityIndexUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub entity_index: u32,
    /// Non-zero when the entity is selected (drives the R8Unorm mask output).
    pub selected: u32,
    pub _padding: [u32; 2],
}

/// Create the GPU [`Material`] for the entity index shader, specialized for
/// `vertex_layout` (which must have the position as its first attribute — see
/// the module docs).
///
/// The material renders to an `R32Uint` index target plus an `R8Unorm`
/// selection-mask target (no blending) with the given depth format. Binding
/// layout is auto-reflected from the Slang shader.
pub fn create_entity_index_material(
    device: &Arc<GraphicsDevice>,
    vertex_layout: &Arc<VertexLayout>,
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
                .with_vertex_layout(vertex_layout.clone())
                .with_color_format(TextureFormat::R32Uint)
                .with_color_format(TextureFormat::R8Unorm)
                .with_depth_format(depth_format)
                // Group 0 binding 0 (per-entity transform) uses a per-draw offset.
                .with_dynamic_uniform(0, 0)
                .with_label("std_entity_index"),
        )
        .expect("Failed to create entity index material")
}
