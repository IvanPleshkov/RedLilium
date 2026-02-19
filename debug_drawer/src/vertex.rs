/// A debug draw vertex: position + color.
///
/// Used for line-list rendering. Every pair of consecutive vertices
/// forms one line segment.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

/// Uniform buffer data for the debug draw shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugUniforms {
    /// Column-major 4x4 view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
}
