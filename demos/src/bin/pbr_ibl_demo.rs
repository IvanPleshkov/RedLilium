//! # PBR IBL Demo
//!
//! Demonstrates:
//! - Forward PBR rendering with Image-Based Lighting (IBL)
//! - HDR environment map converted to cubemap
//! - Irradiance cubemap for diffuse IBL
//! - Pre-filtered environment map for specular IBL
//! - BRDF Look-Up Table for split-sum approximation
//! - Orbit camera (no ECS)
//! - Grid of PBR spheres with varying metallic/roughness
//!
//! Based on LearnOpenGL IBL tutorials:
//! - https://learnopengl.com/PBR/IBL/Diffuse-irradiance
//! - https://learnopengl.com/PBR/IBL/Specular-IBL

use std::f32::consts::PI;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use redlilium_app::{App, AppArgs, AppContext, AppHandler, DefaultAppArgs, DrawContext};
use redlilium_graphics::{
    AddressMode, BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BufferDescriptor,
    BufferUsage, ColorAttachment, DepthStencilAttachment, Extent3d, FilterMode, FrameSchedule,
    GraphicsPass, IndexFormat, Material, MaterialDescriptor, MaterialInstance, Mesh,
    MeshDescriptor, RenderGraph, RenderTargetConfig, SamplerDescriptor, ShaderSource, ShaderStage,
    ShaderStageFlags, TextureDescriptor, TextureFormat, TextureUsage, TransferConfig,
    TransferOperation, TransferPass, VertexAttribute, VertexAttributeFormat,
    VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
};

// === WGSL Shaders ===

/// Main PBR shader with IBL
const PBR_SHADER_WGSL: &str = r#"
// Camera uniforms
struct CameraUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

// Per-instance data
struct InstanceData {
    model: mat4x4<f32>,
    base_color: vec4<f32>,
    metallic_roughness: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;

// IBL textures
@group(1) @binding(0) var irradiance_map: texture_cube<f32>;
@group(1) @binding(1) var prefilter_map: texture_cube<f32>;
@group(1) @binding(2) var brdf_lut: texture_2d<f32>;
@group(1) @binding(3) var ibl_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @builtin(instance_index) instance_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) base_color: vec4<f32>,
    @location(4) metallic: f32,
    @location(5) roughness: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let instance = instances[in.instance_id];
    let world_pos = instance.model * vec4<f32>(in.position, 1.0);
    let normal_matrix = mat3x3<f32>(
        instance.model[0].xyz,
        instance.model[1].xyz,
        instance.model[2].xyz
    );

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal = normalize(normal_matrix * in.normal);
    out.uv = in.uv;
    out.base_color = instance.base_color;
    out.metallic = instance.metallic_roughness.x;
    out.roughness = instance.metallic_roughness.y;
    return out;
}

const PI: f32 = 3.14159265359;
const MAX_REFLECTION_LOD: f32 = 4.0;

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;

    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo = in.base_color.rgb;
    let metallic = in.metallic;
    let roughness = max(in.roughness, 0.04);

    let n = normalize(in.world_normal);
    let v = normalize(camera.camera_pos.xyz - in.world_position);
    let r = reflect(-v, n);

    let n_dot_v = max(dot(n, v), 0.0);

    // Calculate reflectance at normal incidence
    var f0 = vec3<f32>(0.04);
    f0 = mix(f0, albedo, metallic);

    // === Direct lighting ===
    // Simple directional light (sun-like)
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 0.5));
    let light_color = vec3<f32>(1.0, 0.98, 0.95) * 3.0;

    let h = normalize(v + light_dir);
    let radiance = light_color;

    // Cook-Torrance BRDF
    let ndf = distribution_ggx(n, h, roughness);
    let g = geometry_smith(n, v, light_dir, roughness);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);

    let numerator = ndf * g * f;
    let denominator = 4.0 * max(dot(n, v), 0.0) * max(dot(n, light_dir), 0.0) + 0.0001;
    let specular_direct = numerator / denominator;

    let ks_direct = f;
    var kd_direct = vec3<f32>(1.0) - ks_direct;
    kd_direct = kd_direct * (1.0 - metallic);

    let n_dot_l = max(dot(n, light_dir), 0.0);
    var lo = (kd_direct * albedo / PI + specular_direct) * radiance * n_dot_l;

    // Fill light
    let fill_light_dir = normalize(vec3<f32>(-0.5, -0.3, -1.0));
    let fill_light_color = vec3<f32>(0.3, 0.4, 0.5) * 0.5;
    let fill_n_dot_l = max(dot(n, fill_light_dir), 0.0);
    lo = lo + kd_direct * albedo * fill_light_color * fill_n_dot_l;

    // === IBL ambient lighting ===
    let f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let ks_ibl = f_ibl;
    let kd_ibl = (vec3<f32>(1.0) - ks_ibl) * (1.0 - metallic);

    // Diffuse IBL from irradiance map
    let irradiance = textureSample(irradiance_map, ibl_sampler, n).rgb;
    let diffuse_ibl = irradiance * albedo;

    // Specular IBL from pre-filtered environment map + BRDF LUT
    let prefiltered_color = textureSampleLevel(prefilter_map, ibl_sampler, r, roughness * MAX_REFLECTION_LOD).rgb;
    let brdf = textureSample(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness)).rg;
    let specular_ibl = prefiltered_color * (f_ibl * brdf.x + brdf.y);

    let ambient = kd_ibl * diffuse_ibl + specular_ibl;

    // Combine
    var color = ambient + lo;

    // HDR tonemapping (Reinhard)
    color = color / (color + vec3<f32>(1.0));

    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, 1.0);
}
"#;

// === Orbit Camera ===

struct OrbitCamera {
    target: Vec3,
    distance: f32,
    azimuth: f32,
    elevation: f32,
    fov: f32,
    near: f32,
    far: f32,
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 8.0,
            azimuth: 0.5,
            elevation: 0.4,
            fov: PI / 4.0,
            near: 0.1,
            far: 100.0,
        }
    }

    fn rotate(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        self.azimuth += delta_azimuth;
        self.elevation = (self.elevation + delta_elevation).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);
    }

    fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta).clamp(2.0, 20.0);
    }

    fn position(&self) -> Vec3 {
        let x = self.distance * self.elevation.cos() * self.azimuth.sin();
        let y = self.distance * self.elevation.sin();
        let z = self.distance * self.elevation.cos() * self.azimuth.cos();
        self.target + Vec3::new(x, y, z)
    }

    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Y)
    }

    fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect_ratio, self.near, self.far)
    }
}

// === Vertex and Instance Data ===

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PbrVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereInstance {
    model: [[f32; 4]; 4],
    base_color: [f32; 4],
    metallic_roughness: [f32; 4],
}

// === Mesh Generation ===

fn generate_sphere(radius: f32, segments: u32, rings: u32) -> (Vec<PbrVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for ring in 0..=rings {
        let theta = ring as f32 * PI / rings as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for segment in 0..=segments {
            let phi = segment as f32 * 2.0 * PI / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = sin_theta * cos_phi;
            let y = cos_theta;
            let z = sin_theta * sin_phi;

            vertices.push(PbrVertex {
                position: [x * radius, y * radius, z * radius],
                normal: [x, y, z],
                uv: [segment as f32 / segments as f32, ring as f32 / rings as f32],
            });
        }
    }

    for ring in 0..rings {
        for segment in 0..segments {
            let current = ring * (segments + 1) + segment;
            let next = current + segments + 1;

            indices.push(current);
            indices.push(next);
            indices.push(current + 1);

            indices.push(current + 1);
            indices.push(next);
            indices.push(next + 1);
        }
    }

    (vertices, indices)
}

// === HDR Loading ===

const HDR_URL: &str = "https://raw.githubusercontent.com/JoeyDeVries/LearnOpenGL/master/resources/textures/hdr/newport_loft.hdr";
const BRDF_LUT_URL: &str = "https://learnopengl.com/img/pbr/ibl_brdf_lut.png";

fn load_brdf_lut_from_url(url: &str) -> Result<(u32, u32, Vec<u8>), String> {
    use std::io::Read;

    log::info!("Downloading BRDF LUT from: {}", url);

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Failed to download BRDF LUT: {e}"))?;

    let mut data = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut data)
        .map_err(|e| format!("Failed to read BRDF LUT data: {e}"))?;

    log::info!("Downloaded {} bytes, parsing PNG...", data.len());

    let img =
        image::load_from_memory(&data).map_err(|e| format!("Failed to decode BRDF LUT: {e}"))?;

    let width = img.width();
    let height = img.height();

    log::info!("BRDF LUT image: {}x{}", width, height);

    // Convert to RG8 (we only need red and green channels)
    let rgba = img.to_rgba8();
    let mut rg_data = Vec::with_capacity((width * height * 2) as usize);
    for pixel in rgba.pixels() {
        rg_data.push(pixel[0]); // R channel = scale
        rg_data.push(pixel[1]); // G channel = bias
    }

    Ok((width, height, rg_data))
}

fn load_hdr_from_url(url: &str) -> Result<(u32, u32, Vec<f32>), String> {
    use std::io::Read;

    log::info!("Downloading HDR texture from: {}", url);

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Failed to download HDR: {e}"))?;

    let mut data = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut data)
        .map_err(|e| format!("Failed to read HDR data: {e}"))?;

    log::info!("Downloaded {} bytes, parsing HDR...", data.len());

    let img = image::load_from_memory_with_format(&data, image::ImageFormat::Hdr)
        .map_err(|e| format!("Failed to decode HDR: {e}"))?;

    let width = img.width();
    let height = img.height();

    log::info!("HDR image: {}x{}", width, height);

    let rgba32f = img.to_rgba32f();
    let rgba_data: Vec<f32> = rgba32f.into_raw();

    Ok((width, height, rgba_data))
}

// === Demo Application ===

const GRID_SIZE: usize = 5;
const SPHERE_SPACING: f32 = 1.5;
const IRRADIANCE_SIZE: u32 = 32;
const PREFILTER_SIZE: u32 = 128;

struct PbrIblDemo {
    camera: OrbitCamera,
    mouse_pressed: bool,
    last_mouse_x: f64,
    last_mouse_y: f64,

    // GPU resources
    material: Option<Arc<Material>>,
    material_instance: Option<Arc<MaterialInstance>>,
    mesh: Option<Arc<Mesh>>,
    camera_buffer: Option<Arc<redlilium_graphics::Buffer>>,
    instance_buffer: Option<Arc<redlilium_graphics::Buffer>>,
    depth_texture: Option<Arc<redlilium_graphics::Texture>>,

    // IBL resources
    ibl_ready: bool,
    irradiance_cubemap: Option<Arc<redlilium_graphics::Texture>>,
    prefilter_cubemap: Option<Arc<redlilium_graphics::Texture>>,
    brdf_lut: Option<Arc<redlilium_graphics::Texture>>,
    ibl_sampler: Option<Arc<redlilium_graphics::Sampler>>,

    // Staging buffers for IBL upload (used on first frame)
    irradiance_staging: Option<Arc<redlilium_graphics::Buffer>>,
    prefilter_staging: Option<Vec<Arc<redlilium_graphics::Buffer>>>,
    prefilter_aligned_bytes_per_row: Vec<u32>,
    brdf_staging: Option<Arc<redlilium_graphics::Buffer>>,
    brdf_size: (u32, u32),
    brdf_aligned_bytes_per_row: u32,
    needs_ibl_upload: bool,
}

impl PbrIblDemo {
    fn new() -> Self {
        Self {
            camera: OrbitCamera::new(),
            mouse_pressed: false,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            material: None,
            material_instance: None,
            mesh: None,
            camera_buffer: None,
            instance_buffer: None,
            depth_texture: None,
            ibl_ready: false,
            irradiance_cubemap: None,
            prefilter_cubemap: None,
            brdf_lut: None,
            ibl_sampler: None,
            irradiance_staging: None,
            prefilter_staging: None,
            prefilter_aligned_bytes_per_row: Vec::new(),
            brdf_staging: None,
            brdf_size: (0, 0),
            brdf_aligned_bytes_per_row: 0,
            needs_ibl_upload: false,
        }
    }

    fn create_gpu_resources(&mut self, ctx: &mut AppContext) {
        let device = ctx.device();

        // Create vertex layout
        let vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(
                    std::mem::size_of::<PbrVertex>() as u32
                ))
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::Position,
                    format: VertexAttributeFormat::Float3,
                    offset: 0,
                    buffer_index: 0,
                })
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::Normal,
                    format: VertexAttributeFormat::Float3,
                    offset: 12,
                    buffer_index: 0,
                })
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::TexCoord0,
                    format: VertexAttributeFormat::Float2,
                    offset: 24,
                    buffer_index: 0,
                })
                .with_label("pbr_vertex_layout"),
        );

        // Load HDR environment and compute IBL data on CPU
        log::info!("Loading HDR environment map...");
        let (hdr_width, hdr_height, hdr_data) =
            load_hdr_from_url(HDR_URL).expect("Failed to load HDR texture");

        log::info!("Computing IBL cubemaps on CPU...");
        let (irradiance_data, prefilter_data) = compute_ibl_cpu(&hdr_data, hdr_width, hdr_height);

        // Load BRDF LUT from LearnOpenGL
        let (brdf_width, brdf_height, brdf_data) =
            load_brdf_lut_from_url(BRDF_LUT_URL).expect("Failed to load BRDF LUT");

        // Create IBL textures
        let irradiance_cubemap = device
            .create_texture(
                &TextureDescriptor::new_cube(
                    IRRADIANCE_SIZE,
                    TextureFormat::Rgba16Float,
                    TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                )
                .with_label("irradiance_cubemap"),
            )
            .expect("Failed to create irradiance cubemap");
        self.irradiance_cubemap = Some(irradiance_cubemap);

        let mip_levels = (PREFILTER_SIZE as f32).log2().floor() as u32 + 1;
        let prefilter_cubemap = device
            .create_texture(
                &TextureDescriptor::new_cube(
                    PREFILTER_SIZE,
                    TextureFormat::Rgba16Float,
                    TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                )
                .with_mip_levels(mip_levels)
                .with_label("prefilter_cubemap"),
            )
            .expect("Failed to create prefilter cubemap");
        self.prefilter_cubemap = Some(prefilter_cubemap);

        let brdf_lut = device
            .create_texture(
                &TextureDescriptor::new_2d(
                    brdf_width,
                    brdf_height,
                    TextureFormat::Rg8Unorm,
                    TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                )
                .with_label("brdf_lut"),
            )
            .expect("Failed to create BRDF LUT");
        self.brdf_lut = Some(brdf_lut);
        self.brdf_size = (brdf_width, brdf_height);

        // Create staging buffers for IBL data upload
        let irradiance_bytes: &[u8] = bytemuck::cast_slice(&irradiance_data);
        let irradiance_staging = device
            .create_buffer(&BufferDescriptor::new(
                irradiance_bytes.len() as u64,
                BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create irradiance staging buffer");
        device.write_buffer(&irradiance_staging, 0, irradiance_bytes);
        self.irradiance_staging = Some(irradiance_staging);

        // Create staging buffers for each mip level with aligned bytes per row
        const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
        let bytes_per_pixel = 8u32; // Rgba16Float = 4 channels * 2 bytes
        let mut prefilter_staging = Vec::new();
        let mut prefilter_aligned_bytes_per_row = Vec::new();

        for (mip, mip_data) in prefilter_data.iter().enumerate() {
            let mip_size = (PREFILTER_SIZE >> mip).max(1);
            let bytes_per_row = mip_size * bytes_per_pixel;
            let aligned_bytes_per_row =
                bytes_per_row.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
            prefilter_aligned_bytes_per_row.push(aligned_bytes_per_row);

            let bytes: &[u8] = bytemuck::cast_slice(mip_data);

            // Pad data if alignment is needed
            let padded_data = if aligned_bytes_per_row != bytes_per_row {
                let face_size = (mip_size * mip_size) as usize * bytes_per_pixel as usize;
                let padded_face_size = (aligned_bytes_per_row * mip_size) as usize;
                let mut padded = vec![0u8; padded_face_size * 6];
                for face in 0..6 {
                    for y in 0..mip_size {
                        let src_start = face * face_size + (y as usize * bytes_per_row as usize);
                        let src_end = src_start + bytes_per_row as usize;
                        let dst_start =
                            face * padded_face_size + (y as usize * aligned_bytes_per_row as usize);
                        padded[dst_start..dst_start + bytes_per_row as usize]
                            .copy_from_slice(&bytes[src_start..src_end]);
                    }
                }
                padded
            } else {
                bytes.to_vec()
            };

            let buffer = device
                .create_buffer(&BufferDescriptor::new(
                    padded_data.len() as u64,
                    BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                ))
                .expect("Failed to create prefilter staging buffer");
            device.write_buffer(&buffer, 0, &padded_data);
            prefilter_staging.push(buffer);
        }
        self.prefilter_staging = Some(prefilter_staging);
        self.prefilter_aligned_bytes_per_row = prefilter_aligned_bytes_per_row;

        // BRDF staging buffer needs aligned bytes per row (256-byte alignment for WebGPU)
        let brdf_bytes_per_row = brdf_width * 2; // 2 bytes per pixel (Rg8Unorm)
        let brdf_aligned_bytes_per_row = brdf_bytes_per_row.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT)
            * COPY_BYTES_PER_ROW_ALIGNMENT;

        // Create padded buffer if alignment is needed
        let brdf_padded_data = if brdf_aligned_bytes_per_row != brdf_bytes_per_row {
            let mut padded = vec![0u8; (brdf_aligned_bytes_per_row * brdf_height) as usize];
            for y in 0..brdf_height {
                let src_start = (y * brdf_bytes_per_row) as usize;
                let src_end = src_start + brdf_bytes_per_row as usize;
                let dst_start = (y * brdf_aligned_bytes_per_row) as usize;
                padded[dst_start..dst_start + brdf_bytes_per_row as usize]
                    .copy_from_slice(&brdf_data[src_start..src_end]);
            }
            padded
        } else {
            brdf_data
        };

        let brdf_staging = device
            .create_buffer(&BufferDescriptor::new(
                brdf_padded_data.len() as u64,
                BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create BRDF staging buffer");
        device.write_buffer(&brdf_staging, 0, &brdf_padded_data);
        self.brdf_staging = Some(brdf_staging);
        self.brdf_aligned_bytes_per_row = brdf_aligned_bytes_per_row;

        self.needs_ibl_upload = true;

        // Create IBL sampler
        let ibl_sampler = device
            .create_sampler(&SamplerDescriptor {
                label: Some("ibl_sampler".into()),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Linear,
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                ..Default::default()
            })
            .expect("Failed to create IBL sampler");
        self.ibl_sampler = Some(ibl_sampler);

        self.ibl_ready = true;
        log::info!("IBL resources created successfully");

        // Create binding layouts
        let camera_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::StorageBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX),
                )
                .with_label("camera_bindings"),
        );

        let ibl_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::TextureCube)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::TextureCube)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(2, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(3, BindingType::Sampler)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_label("ibl_bindings"),
        );

        // Create PBR material
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        PBR_SHADER_WGSL.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        PBR_SHADER_WGSL.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(camera_binding_layout)
                    .with_binding_layout(ibl_binding_layout)
                    .with_vertex_layout(vertex_layout.clone())
                    .with_label("pbr_material"),
            )
            .expect("Failed to create material");
        self.material = Some(material.clone());

        // Create camera uniform buffer
        let camera_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<CameraUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create camera buffer");
        self.camera_buffer = Some(camera_buffer.clone());

        // Create instance buffer
        let instances = self.create_sphere_instances();
        let instance_data = bytemuck::cast_slice(&instances);
        let instance_buffer = device
            .create_buffer(&BufferDescriptor::new(
                instance_data.len() as u64,
                BufferUsage::STORAGE | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create instance buffer");
        device.write_buffer(&instance_buffer, 0, instance_data);
        self.instance_buffer = Some(instance_buffer.clone());

        // Create material instance with bindings
        #[allow(clippy::arc_with_non_send_sync)]
        let camera_binding_group = Arc::new(
            BindingGroup::new()
                .with_buffer(0, camera_buffer)
                .with_buffer(1, instance_buffer),
        );

        #[allow(clippy::arc_with_non_send_sync)]
        let ibl_binding_group = Arc::new(
            BindingGroup::new()
                .with_texture(0, self.irradiance_cubemap.clone().unwrap())
                .with_texture(1, self.prefilter_cubemap.clone().unwrap())
                .with_texture(2, self.brdf_lut.clone().unwrap())
                .with_sampler(3, self.ibl_sampler.clone().unwrap()),
        );

        let material_instance = Arc::new(
            MaterialInstance::new(material)
                .with_binding_group(camera_binding_group)
                .with_binding_group(ibl_binding_group),
        );
        self.material_instance = Some(material_instance);

        // Create sphere mesh
        let (vertices, indices) = generate_sphere(0.5, 32, 16);
        let mesh = device
            .create_mesh(
                &MeshDescriptor::new(vertex_layout)
                    .with_vertex_count(vertices.len() as u32)
                    .with_indices(IndexFormat::Uint32, indices.len() as u32)
                    .with_label("sphere"),
            )
            .expect("Failed to create mesh");

        if let Some(vb) = mesh.vertex_buffer(0) {
            device.write_buffer(vb, 0, bytemuck::cast_slice(&vertices));
        }
        if let Some(ib) = mesh.index_buffer() {
            device.write_buffer(ib, 0, bytemuck::cast_slice(&indices));
        }
        self.mesh = Some(mesh);

        // Create depth texture
        self.create_depth_texture(ctx);
    }

    fn create_depth_texture(&mut self, ctx: &AppContext) {
        let depth_texture = ctx
            .device()
            .create_texture(&TextureDescriptor::new_2d(
                ctx.width(),
                ctx.height(),
                TextureFormat::Depth32Float,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .expect("Failed to create depth texture");
        self.depth_texture = Some(depth_texture);
    }

    fn create_sphere_instances(&self) -> Vec<SphereInstance> {
        let mut instances = Vec::with_capacity(GRID_SIZE * GRID_SIZE);
        let offset = (GRID_SIZE as f32 - 1.0) * SPHERE_SPACING / 2.0;

        for row in 0..GRID_SIZE {
            for col in 0..GRID_SIZE {
                let x = col as f32 * SPHERE_SPACING - offset;
                let z = row as f32 * SPHERE_SPACING - offset;

                let model = Mat4::from_translation(Vec3::new(x, 0.0, z));
                let metallic = col as f32 / (GRID_SIZE - 1) as f32;
                let roughness = (row as f32 / (GRID_SIZE - 1) as f32).max(0.05);

                instances.push(SphereInstance {
                    model: model.to_cols_array_2d(),
                    base_color: [0.9, 0.1, 0.1, 1.0], // Red
                    metallic_roughness: [metallic, roughness, 0.0, 0.0],
                });
            }
        }

        instances
    }

    fn update_camera_buffer(&self, ctx: &AppContext) {
        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix(ctx.aspect_ratio());
        let view_proj = proj * view;

        let uniforms = CameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            camera_pos: self.camera.position().extend(1.0).to_array(),
        };

        if let Some(buffer) = &self.camera_buffer {
            ctx.device()
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    fn create_ibl_transfer_config(&self) -> TransferConfig {
        use redlilium_graphics::{
            BufferTextureCopyRegion, BufferTextureLayout, TextureCopyLocation, TextureOrigin,
        };

        let mut config = TransferConfig::new();

        // Upload irradiance cubemap (6 faces)
        if let (Some(staging), Some(texture)) = (&self.irradiance_staging, &self.irradiance_cubemap)
        {
            let face_bytes = (IRRADIANCE_SIZE * IRRADIANCE_SIZE * 4 * 2) as u64; // 4 channels * 2 bytes (f16)
            for face in 0..6u32 {
                let region = BufferTextureCopyRegion::new(
                    BufferTextureLayout::new(
                        face as u64 * face_bytes,
                        Some(IRRADIANCE_SIZE * 4 * 2),
                        None,
                    ),
                    TextureCopyLocation::new(0, TextureOrigin::new(0, 0, face)),
                    Extent3d::new_2d(IRRADIANCE_SIZE, IRRADIANCE_SIZE),
                );
                config = config.with_operation(TransferOperation::upload_texture(
                    staging.clone(),
                    texture.clone(),
                    vec![region],
                ));
            }
        }

        // Upload prefilter cubemap (all mip levels, 6 faces each)
        if let (Some(staging_buffers), Some(texture)) =
            (&self.prefilter_staging, &self.prefilter_cubemap)
        {
            for (mip, staging) in staging_buffers.iter().enumerate() {
                let mip_size = (PREFILTER_SIZE >> mip).max(1);
                let aligned_bytes_per_row = self.prefilter_aligned_bytes_per_row[mip];
                let face_bytes = (aligned_bytes_per_row * mip_size) as u64;
                for face in 0..6u32 {
                    let region = BufferTextureCopyRegion::new(
                        BufferTextureLayout::new(
                            face as u64 * face_bytes,
                            Some(aligned_bytes_per_row),
                            None,
                        ),
                        TextureCopyLocation::new(mip as u32, TextureOrigin::new(0, 0, face)),
                        Extent3d::new_2d(mip_size, mip_size),
                    );
                    config = config.with_operation(TransferOperation::upload_texture(
                        staging.clone(),
                        texture.clone(),
                        vec![region],
                    ));
                }
            }
        }

        // Upload BRDF LUT
        if let (Some(staging), Some(texture)) = (&self.brdf_staging, &self.brdf_lut) {
            let (brdf_width, brdf_height) = self.brdf_size;
            let region = BufferTextureCopyRegion::new(
                BufferTextureLayout::new(0, Some(self.brdf_aligned_bytes_per_row), None),
                TextureCopyLocation::base(),
                Extent3d::new_2d(brdf_width, brdf_height),
            );
            config = config.with_operation(TransferOperation::upload_texture(
                staging.clone(),
                texture.clone(),
                vec![region],
            ));
        }

        config
    }
}

// === IBL Computation Helpers ===

fn compute_ibl_cpu(hdr_data: &[f32], hdr_width: u32, hdr_height: u32) -> (Vec<u16>, Vec<Vec<u16>>) {
    // Sample direction from equirectangular map
    let sample_equirect = |dir: Vec3| -> Vec3 {
        let inv_atan = glam::vec2(
            0.5 * std::f32::consts::FRAC_1_PI,
            std::f32::consts::FRAC_1_PI,
        );
        let uv = glam::vec2(dir.z.atan2(dir.x), dir.y.asin()) * inv_atan + 0.5;

        let x = ((uv.x * hdr_width as f32) as u32).min(hdr_width - 1);
        let y = (((1.0 - uv.y) * hdr_height as f32) as u32).min(hdr_height - 1);
        let idx = ((y * hdr_width + x) * 4) as usize;

        Vec3::new(hdr_data[idx], hdr_data[idx + 1], hdr_data[idx + 2])
    };

    // Compute irradiance cubemap
    log::info!("Computing irradiance cubemap...");
    let mut irradiance_data =
        Vec::with_capacity((IRRADIANCE_SIZE * IRRADIANCE_SIZE * 6 * 4) as usize);

    for face in 0..6 {
        for y in 0..IRRADIANCE_SIZE {
            for x in 0..IRRADIANCE_SIZE {
                let dir = cubemap_dir(face, x, y, IRRADIANCE_SIZE);
                let irradiance = compute_irradiance(dir, &sample_equirect);
                irradiance_data.push(f32_to_f16_bits(irradiance.x));
                irradiance_data.push(f32_to_f16_bits(irradiance.y));
                irradiance_data.push(f32_to_f16_bits(irradiance.z));
                irradiance_data.push(f32_to_f16_bits(1.0));
            }
        }
    }

    // Compute pre-filtered environment map with mipmaps
    log::info!("Computing pre-filtered environment map...");
    let mip_levels = (PREFILTER_SIZE as f32).log2().floor() as u32 + 1;
    let mut prefilter_data = Vec::with_capacity(mip_levels as usize);

    for mip in 0..mip_levels {
        let mip_size = (PREFILTER_SIZE >> mip).max(1);
        let roughness = mip as f32 / (mip_levels - 1) as f32;
        let mut mip_data = Vec::with_capacity((mip_size * mip_size * 6 * 4) as usize);

        for face in 0..6 {
            for y in 0..mip_size {
                for x in 0..mip_size {
                    let dir = cubemap_dir(face, x, y, mip_size);
                    let prefiltered = compute_prefiltered(dir, roughness, &sample_equirect);
                    mip_data.push(f32_to_f16_bits(prefiltered.x));
                    mip_data.push(f32_to_f16_bits(prefiltered.y));
                    mip_data.push(f32_to_f16_bits(prefiltered.z));
                    mip_data.push(f32_to_f16_bits(1.0));
                }
            }
        }
        prefilter_data.push(mip_data);
    }

    (irradiance_data, prefilter_data)
}

/// Convert f32 to f16 bits (IEEE 754 half-precision)
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0 {
        // Zero or denormalized
        0
    } else if exp == 0xFF {
        // Infinity or NaN
        ((sign << 15) | 0x7C00 | (mantissa >> 13).min(0x3FF)) as u16
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            // Overflow to infinity
            ((sign << 15) | 0x7C00) as u16
        } else if new_exp <= 0 {
            // Underflow to zero or denorm
            0
        } else {
            ((sign << 15) | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
        }
    }
}

/// Get cubemap direction for a given face, pixel, and size
fn cubemap_dir(face: u32, x: u32, y: u32, size: u32) -> Vec3 {
    let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
    let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;

    let dir = match face {
        0 => Vec3::new(1.0, -v, -u),  // +X
        1 => Vec3::new(-1.0, -v, u),  // -X
        2 => Vec3::new(u, 1.0, v),    // +Y
        3 => Vec3::new(u, -1.0, -v),  // -Y
        4 => Vec3::new(u, -v, 1.0),   // +Z
        _ => Vec3::new(-u, -v, -1.0), // -Z
    };

    dir.normalize()
}

/// Compute irradiance for a given normal direction
fn compute_irradiance<F: Fn(Vec3) -> Vec3>(normal: Vec3, sample_env: &F) -> Vec3 {
    let mut irradiance = Vec3::ZERO;

    let up = if normal.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let right = normal.cross(up).normalize();
    let up = normal.cross(right);

    let sample_delta = 0.05;
    let mut nr_samples = 0.0;

    let mut phi = 0.0f32;
    while phi < 2.0 * PI {
        let mut theta = 0.0f32;
        while theta < 0.5 * PI {
            let tangent_sample = Vec3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            );
            let sample_vec =
                tangent_sample.x * right + tangent_sample.y * up + tangent_sample.z * normal;

            irradiance += sample_env(sample_vec) * theta.cos() * theta.sin();
            nr_samples += 1.0;

            theta += sample_delta;
        }
        phi += sample_delta;
    }

    PI * irradiance / nr_samples
}

/// Compute pre-filtered environment map value
fn compute_prefiltered<F: Fn(Vec3) -> Vec3>(normal: Vec3, roughness: f32, sample_env: &F) -> Vec3 {
    let r = normal;
    let v = r;

    let sample_count = 128u32;
    let mut prefiltered = Vec3::ZERO;
    let mut total_weight = 0.0;

    for i in 0..sample_count {
        let xi = hammersley(i, sample_count);
        let h = importance_sample_ggx(xi, normal, roughness);
        let l = (2.0 * v.dot(h) * h - v).normalize();

        let n_dot_l = normal.dot(l).max(0.0);
        if n_dot_l > 0.0 {
            prefiltered += sample_env(l) * n_dot_l;
            total_weight += n_dot_l;
        }
    }

    prefiltered / total_weight.max(0.001)
}

/// Hammersley sequence for low-discrepancy sampling
fn hammersley(i: u32, n: u32) -> glam::Vec2 {
    glam::vec2(i as f32 / n as f32, radical_inverse_vdc(i))
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    bits as f32 * 2.328_306_4e-10
}

/// GGX importance sampling
fn importance_sample_ggx(xi: glam::Vec2, n: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;

    let phi = 2.0 * PI * xi.x;
    let cos_theta = ((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    let h = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    let up = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    let tangent = n.cross(up).normalize();
    let bitangent = n.cross(tangent);

    (tangent * h.x + bitangent * h.y + n * h.z).normalize()
}

impl AppHandler for PbrIblDemo {
    fn on_init(&mut self, ctx: &mut AppContext) {
        log::info!("Initializing PBR IBL Demo");
        log::info!(
            "Grid: {}x{} spheres with varying metallic/roughness",
            GRID_SIZE,
            GRID_SIZE
        );
        log::info!("IBL: Cubemap-based (irradiance + prefiltered + BRDF LUT)");
        log::info!("Controls:");
        log::info!("  - Left mouse drag: Rotate camera");
        log::info!("  - Scroll: Zoom");

        self.create_gpu_resources(ctx);
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        self.create_depth_texture(ctx);
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        if !self.mouse_pressed {
            self.camera.rotate(ctx.delta_time() * 0.15, 0.0);
        }
        self.update_camera_buffer(ctx);
        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        let mut graph = RenderGraph::new();

        // Upload IBL textures on first frame
        if self.needs_ibl_upload {
            let transfer_config = self.create_ibl_transfer_config();
            let mut transfer_pass = TransferPass::new("ibl_upload".into());
            transfer_pass.set_transfer_config(transfer_config);
            graph.add_transfer_pass(transfer_pass);
            self.needs_ibl_upload = false;
            log::info!("IBL textures uploaded via transfer pass");
        }

        let mut render_pass = GraphicsPass::new("main".into());

        if let Some(depth) = &self.depth_texture {
            render_pass.set_render_targets(
                RenderTargetConfig::new()
                    .with_color(
                        ColorAttachment::from_surface(ctx.swapchain_texture())
                            .with_clear_color(0.02, 0.02, 0.03, 1.0),
                    )
                    .with_depth_stencil(
                        DepthStencilAttachment::from_texture(depth.clone()).with_clear_depth(1.0),
                    ),
            );
        }

        if let (Some(mesh), Some(material_instance)) = (&self.mesh, &self.material_instance) {
            render_pass.add_draw_instanced(
                mesh.clone(),
                material_instance.clone(),
                (GRID_SIZE * GRID_SIZE) as u32,
            );
        }

        graph.add_graphics_pass(render_pass);

        let _handle = ctx.submit("main", &graph, &[]);
        ctx.finish(&[])
    }

    fn on_mouse_move(&mut self, _ctx: &mut AppContext, x: f64, y: f64) {
        if self.mouse_pressed {
            let dx = (x - self.last_mouse_x) as f32 * 0.005;
            let dy = (y - self.last_mouse_y) as f32 * 0.005;
            self.camera.rotate(-dx, -dy);
        }
        self.last_mouse_x = x;
        self.last_mouse_y = y;
    }

    fn on_mouse_button(
        &mut self,
        _ctx: &mut AppContext,
        button: winit::event::MouseButton,
        pressed: bool,
    ) {
        if button == winit::event::MouseButton::Left {
            self.mouse_pressed = pressed;
        }
    }

    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, _dx: f32, dy: f32) {
        self.camera.zoom(dy * 0.5);
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down PBR IBL Demo");
    }
}

// === Entry Point ===

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args = DefaultAppArgs::parse();
    App::run(PbrIblDemo::new(), args);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Entry point for wasm
}
