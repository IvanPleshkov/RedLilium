//! GPU uniform buffer structures for the PBR demo.

/// Camera uniform data for the vertex/fragment shaders.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    /// Element index of this frame's first instance within the (whole-buffer
    /// bound) instance ring; the vertex shader adds it to `SV_InstanceID`. This
    /// lets the instance binding group stay frame-invariant (bound once, at
    /// offset 0) instead of being rebuilt each frame — see [`SphereInstance`].
    pub instance_base: u32,
    pub _pad: [u32; 3],
}

/// Per-sphere instance data for instanced rendering.
///
/// Padded to 128 bytes (a divisor of the ring's 256-byte allocation alignment)
/// so every per-frame instance block starts at an offset that is an exact
/// multiple of the stride. That makes `offset / 128` a valid element index,
/// which the shader receives as [`CameraUniforms::instance_base`] and adds to
/// `SV_InstanceID`. The `InstanceData` struct in `deferred_gbuffer.slang` is
/// padded to match this stride.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphereInstance {
    pub model: [[f32; 4]; 4],
    pub base_color: [f32; 4],
    pub metallic_roughness: [f32; 4],
    pub _pad: [[f32; 4]; 2],
}

/// Byte stride of one [`SphereInstance`] in the instance ring (also the element
/// granularity used to compute [`CameraUniforms::instance_base`]).
pub const SPHERE_INSTANCE_STRIDE: u64 = std::mem::size_of::<SphereInstance>() as u64;

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

/// Direction **toward** the key light (the sun). Shared by the resolve shader's
/// direct-lighting term and the ray-traced shadow pass (#110) so the shadow rays
/// line up with the light the surface is shaded against. Both shaders normalize
/// it, so it need not be unit length.
///
/// Grazing and diagonal: with the spheres on a flat grid and no ground plane, a
/// steep light would leave the inter-sphere shadows invisible. A low diagonal
/// light makes each sphere cast a soft shadow onto its row and column neighbors
/// without the artificial-looking hard-edged gashes a steeper light produced.
pub const SUN_DIR_TO_LIGHT: [f32; 3] = [0.75, 0.40, 0.75];

/// Resolve pass uniform data.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResolveUniforms {
    pub camera_pos: [f32; 4],  // 16 bytes
    pub screen_size: [f32; 4], // xy = dimensions, zw = 1/dimensions
    /// xyz = normalized direction toward the key light ([`SUN_DIR_TO_LIGHT`]).
    pub light_dir: [f32; 4],
}

/// Shadow-mask pass uniform data (#110): the key-light direction the shadow
/// rays are cast toward.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniforms {
    /// xyz = normalized direction toward the key light ([`SUN_DIR_TO_LIGHT`]).
    pub light_dir: [f32; 4],
}
