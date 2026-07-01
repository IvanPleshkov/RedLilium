//! Per-entity transform uniforms for the opaque forward pass.
//!
//! The opaque pipeline itself is now specialized on demand by the
//! [`PipelineCache`](crate::std::rendering::PipelineCache) from the `opaque`
//! shading model's shader asset (`docs/MATERIAL_ASSETS.md`). This module only
//! carries the GPU uniform layout the forward renderer fills into the shared
//! per-entity ring (group 0, bound as a dynamic uniform).

/// Per-entity uniform data: view-projection matrix + model matrix (group 0).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OpaqueColorUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
}
