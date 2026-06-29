//! Entity index material — renders entity ID to an R32Uint target for picking.
//!
//! Uses the same position + normal vertex layout and view-projection / model
//! uniforms as [`super::opaque_color`], with an additional `entity_index: u32`
//! field. The fragment shader outputs the raw entity index so a readback or
//! copy from the picking texture can identify which entity was clicked.

use std::sync::Arc;

use redlilium_graphics::{
    GraphicsDevice, Material, MaterialDescriptor, ShaderSource, ShaderStage, TextureFormat,
    VertexLayout,
};

/// Slang shader that outputs entity index as `u32` to an R32Uint color target.
const SHADER_SLANG: &str = include_str!("../../../../../std-assets/shaders/entity_index.slang");

/// Per-entity uniform data: view-projection, model matrix, and entity index.
///
/// Layout must match the Slang `Uniforms` cbuffer. The `_padding` fields ensure
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
/// given depth format. Binding layout is auto-reflected from the Slang shader.
pub fn create_entity_index_material(
    device: &Arc<GraphicsDevice>,
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
                .with_color_format(TextureFormat::R32Uint)
                .with_depth_format(depth_format)
                // Group 0 binding 0 (per-entity transform) uses a per-draw offset.
                .with_dynamic_uniform(0, 0)
                .with_label("std_entity_index"),
        )
        .expect("Failed to create entity index material")
}
