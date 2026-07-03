//! Forward-pass uniform layouts, split by update frequency
//! (`docs/MATERIAL_ASSETS.md` Decision 7).
//!
//! The opaque pipelines are specialized on demand by the
//! [`PipelineCache`](crate::std::rendering::PipelineCache) from the shading
//! models' shader assets; this module only carries the GPU uniform layouts the
//! forward renderer fills into the shared ring — the camera block (external,
//! pushed once per view) and the model block (dynamic, one slot per draw).

/// Per-view camera uniforms (the `external` set: `gCamera`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub view_projection: [[f32; 4]; 4],
}

/// Per-draw model uniforms (the `dynamic` set: `gModel`, ring-buffered).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelUniforms {
    pub model: [[f32; 4]; 4],
}
