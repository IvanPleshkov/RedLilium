//! GPU uniform buffer structures for the PBR demo.

/// Camera uniform data for the vertex/fragment shaders.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

/// Per-sphere instance data for instanced rendering.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphereInstance {
    pub model: [[f32; 4]; 4],
    pub base_color: [f32; 4],
    pub metallic_roughness: [f32; 4],
}

/// Skybox uniform data.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyboxUniforms {
    pub inv_view_proj: [[f32; 4]; 4], // 64 bytes, offset 0
    pub camera_pos: [f32; 4],         // 16 bytes, offset 64
    pub mip_level: f32,               // 4 bytes, offset 80
    pub _pad0: [f32; 3],              // 12 bytes padding before vec3 (which has 16-byte alignment)
    pub _pad1: [f32; 4], // Additional padding to match WGSL vec3<f32> + struct alignment
}

/// Resolve pass uniform data.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResolveUniforms {
    pub camera_pos: [f32; 4],  // 16 bytes
    pub screen_size: [f32; 4], // xy = dimensions, zw = 1/dimensions
}
