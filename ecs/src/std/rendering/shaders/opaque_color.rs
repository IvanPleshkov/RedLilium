//! Forward-pass uniform layouts, split by update frequency
//! (`docs/MATERIAL_ASSETS.md` Decision 7).
//!
//! The opaque pipelines are specialized on demand by the
//! [`PipelineCache`](crate::std::rendering::PipelineCache) from the shading
//! models' shader assets; this module only carries the GPU uniform layouts the
//! forward renderer fills into the shared ring — the camera block (external,
//! pushed once per view) and the model block (dynamic, one slot per draw).

/// Per-view camera uniforms (the `external` set: `gCamera`).
///
/// Shaders that only rasterize declare just the leading `view_projection`
/// (a prefix of this block is a valid smaller cbuffer); the G-buffer shaders
/// additionally read the unjittered pair for velocity (#147).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    /// What rasterization uses — sub-pixel jittered when the camera opted
    /// into [`TemporalJitter`](crate::std::rendering::TemporalJitter).
    pub view_projection: [[f32; 4]; 4],
    /// This frame's view-projection without jitter (velocity math — jitter
    /// in velocity would smear static imagery by the jitter amplitude).
    pub view_projection_unjittered: [[f32; 4]; 4],
    /// Previous frame's unjittered view-projection.
    pub prev_view_projection: [[f32; 4]; 4],
}

/// Per-draw model uniforms (the `dynamic` set: `gModel`, ring-buffered).
///
/// As with the camera block, `model` alone is a valid prefix declaration.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelUniforms {
    pub model: [[f32; 4]; 4],
    /// Previous frame's model matrix (velocity, #147); equals `model` on the
    /// entity's first visible frame (zero velocity, no first-frame smear).
    pub prev_model: [[f32; 4]; 4],
}
