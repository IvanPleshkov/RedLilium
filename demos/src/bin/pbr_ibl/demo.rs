//! PBR IBL Demo application.

use std::sync::{Arc, RwLock};

use glam::{Mat4, Vec3};
use redlilium_app::{AppContext, AppHandler, DrawContext};
use redlilium_core::mesh::generators;
use redlilium_core::profiling::{
    profile_function, profile_memory_stats, profile_message, profile_scope,
};
use redlilium_graphics::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BufferDescriptor, BufferUsage,
    ColorAttachment, CpuSampler, DepthStencilAttachment, Extent3d, FrameSchedule, GraphicsPass,
    LoadOp, Material, MaterialDescriptor, MaterialInstance, Mesh, MeshDescriptor, RenderTarget,
    RenderTargetConfig, RingAllocation, ShaderComposer, ShaderDef, ShaderSource, ShaderStage,
    ShaderStageFlags, TextureDescriptor, TextureFormat, TextureUsage, TransferConfig,
    TransferOperation, TransferPass, VertexBufferLayout, VertexLayout,
    egui::{EguiController, egui},
};
use winit::event::KeyEvent;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::camera::OrbitCamera;
use crate::ibl::compute_ibl_cpu;
use crate::resources::{BRDF_LUT_URL, HDR_URL, load_brdf_lut_from_url, load_hdr_from_url};
use crate::ui::PbrUi;
use crate::uniforms::{CameraUniforms, ResolveUniforms, SkyboxUniforms, SphereInstance};
use crate::{GRID_SIZE, IRRADIANCE_SIZE, PREFILTER_SIZE, SPHERE_SPACING};

// Shaders are loaded from external files in demos/shaders/.
const GBUFFER_SHADER_WGSL: &str = include_str!("../../../shaders/deferred_gbuffer.wgsl");
const RESOLVE_SHADER_WGSL: &str = include_str!("../../../shaders/deferred_resolve.wgsl");
const SKYBOX_SHADER_WGSL: &str = include_str!("../../../shaders/skybox.wgsl");

/// Per-frame uniform allocation offsets from the ring buffer.
#[derive(Default)]
#[allow(dead_code)]
struct FrameUniformAllocations {
    camera: Option<RingAllocation>,
    skybox: Option<RingAllocation>,
    resolve: Option<RingAllocation>,
}

/// The main PBR IBL demo application.
pub struct PbrIblDemo {
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

    // Per-frame uniform allocations from ring buffer
    frame_allocations: FrameUniformAllocations,

    // Skybox resources
    skybox_material: Option<Arc<Material>>,
    skybox_material_instance: Option<Arc<MaterialInstance>>,
    skybox_uniform_buffer: Option<Arc<redlilium_graphics::Buffer>>,
    skybox_mesh: Option<Arc<Mesh>>,
    skybox_mip_level: f32,

    // Track shift key for MIP level switching
    shift_pressed: bool,

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
    needs_ibl_upload: bool,

    // Egui UI
    egui_controller: Option<EguiController>,
    egui_ui: Arc<RwLock<PbrUi>>,
    needs_instance_update: bool,

    // G-buffer textures for deferred rendering
    gbuffer_albedo: Option<Arc<redlilium_graphics::Texture>>,
    gbuffer_normal_metallic: Option<Arc<redlilium_graphics::Texture>>,
    gbuffer_position_roughness: Option<Arc<redlilium_graphics::Texture>>,
    gbuffer_albedo_id: Option<egui::TextureId>,
    gbuffer_normal_id: Option<egui::TextureId>,
    gbuffer_position_id: Option<egui::TextureId>,

    // Resolve pass resources
    resolve_material: Option<Arc<Material>>,
    resolve_material_instance: Option<Arc<MaterialInstance>>,
    resolve_uniform_buffer: Option<Arc<redlilium_graphics::Buffer>>,
    resolve_mesh: Option<Arc<Mesh>>,
    gbuffer_sampler: Option<Arc<redlilium_graphics::Sampler>>,

    // HDR output state
    hdr_active: bool,
}

impl PbrIblDemo {
    pub fn new() -> Self {
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
            frame_allocations: FrameUniformAllocations::default(),
            skybox_material: None,
            skybox_material_instance: None,
            skybox_uniform_buffer: None,
            skybox_mesh: None,
            skybox_mip_level: 0.0,
            shift_pressed: false,
            ibl_ready: false,
            irradiance_cubemap: None,
            prefilter_cubemap: None,
            brdf_lut: None,
            ibl_sampler: None,
            irradiance_staging: None,
            prefilter_staging: None,
            prefilter_aligned_bytes_per_row: Vec::new(),
            needs_ibl_upload: false,
            egui_controller: None,
            egui_ui: Arc::new(RwLock::new(PbrUi::new())),
            needs_instance_update: false,
            gbuffer_albedo: None,
            gbuffer_normal_metallic: None,
            gbuffer_position_roughness: None,
            gbuffer_albedo_id: None,
            gbuffer_normal_id: None,
            gbuffer_position_id: None,
            resolve_material: None,
            resolve_material_instance: None,
            resolve_uniform_buffer: None,
            resolve_mesh: None,
            gbuffer_sampler: None,
            hdr_active: false,
        }
    }

    fn create_gpu_resources(&mut self, ctx: &mut AppContext) {
        profile_function!();
        profile_message!("Creating GPU resources");

        let device = ctx.device();

        // Generate sphere mesh on CPU
        let sphere_cpu = generators::generate_sphere(0.5, 32, 16);
        let vertex_layout = sphere_cpu.layout().clone();

        // Load HDR environment and compute IBL data on CPU
        log::info!("Loading HDR environment map...");
        let (hdr_width, hdr_height, hdr_data) =
            load_hdr_from_url(HDR_URL).expect("Failed to load HDR texture");

        log::info!("Computing IBL cubemaps on CPU...");
        let (irradiance_data, prefilter_data) = compute_ibl_cpu(&hdr_data, hdr_width, hdr_height);

        // Load BRDF LUT from LearnOpenGL
        let brdf_cpu = load_brdf_lut_from_url(BRDF_LUT_URL).expect("Failed to load BRDF LUT");

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
            .create_texture_from_cpu(&brdf_cpu)
            .expect("Failed to create BRDF LUT");
        self.brdf_lut = Some(brdf_lut);

        // Create staging buffers for IBL data upload
        let irradiance_bytes: &[u8] = bytemuck::cast_slice(&irradiance_data);
        let irradiance_staging = device
            .create_buffer(&BufferDescriptor::new(
                irradiance_bytes.len() as u64,
                BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create irradiance staging buffer");
        device
            .write_buffer(&irradiance_staging, 0, irradiance_bytes)
            .expect("Failed to write irradiance staging buffer");
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
            device
                .write_buffer(&buffer, 0, &padded_data)
                .expect("Failed to write prefilter staging buffer");
            prefilter_staging.push(buffer);
        }
        self.prefilter_staging = Some(prefilter_staging);
        self.prefilter_aligned_bytes_per_row = prefilter_aligned_bytes_per_row;

        self.needs_ibl_upload = true;

        // Create IBL sampler
        let ibl_sampler = device
            .create_sampler_from_cpu(&CpuSampler::linear().with_name("ibl_sampler"))
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

        // Compose G-buffer shader using shader library
        let mut shader_composer =
            ShaderComposer::with_standard_library().expect("Failed to create shader composer");
        let composed_gbuffer_shader = shader_composer
            .compose(GBUFFER_SHADER_WGSL, &[])
            .expect("Failed to compose G-buffer shader");

        log::info!("G-buffer shader composed with library imports");

        // Create G-buffer material (only needs camera/instance bindings, no IBL)
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        composed_gbuffer_shader.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        composed_gbuffer_shader.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(camera_binding_layout.clone())
                    .with_vertex_layout(vertex_layout.clone())
                    .with_label("gbuffer_material"),
            )
            .expect("Failed to create G-buffer material");
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
        device
            .write_buffer(&instance_buffer, 0, instance_data)
            .expect("Failed to write instance buffer");
        self.instance_buffer = Some(instance_buffer.clone());

        // Create G-buffer material instance (only camera bindings, no IBL for G-buffer pass)
        #[allow(clippy::arc_with_non_send_sync)]
        let camera_binding_group = Arc::new(
            BindingGroup::new()
                .with_buffer(0, camera_buffer.clone())
                .with_buffer(1, instance_buffer),
        );

        let material_instance =
            Arc::new(MaterialInstance::new(material).with_binding_group(camera_binding_group));
        self.material_instance = Some(material_instance);

        // Create GPU mesh from CPU sphere
        let mesh = device
            .create_mesh_from_cpu(&sphere_cpu)
            .expect("Failed to create sphere mesh");
        self.mesh = Some(mesh);

        // Create skybox material and resources
        self.create_skybox_resources(ctx);

        // Create depth texture
        self.create_depth_texture(ctx);
    }

    fn create_skybox_resources(&mut self, ctx: &AppContext) {
        profile_scope!("create_skybox_resources");

        let device = ctx.device();

        // Create skybox binding layout
        let skybox_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::TextureCube)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(2, BindingType::Sampler)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_label("skybox_bindings"),
        );

        // Compose skybox shader using shader library
        let mut shader_composer =
            ShaderComposer::with_standard_library().expect("Failed to create shader composer");
        let composed_skybox_shader = shader_composer
            .compose(SKYBOX_SHADER_WGSL, &[])
            .expect("Failed to compose skybox shader");

        log::info!("Skybox shader composed with library imports");

        // Create skybox material (no vertex layout needed for fullscreen triangle)
        let skybox_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        composed_skybox_shader.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        composed_skybox_shader.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(skybox_binding_layout)
                    .with_label("skybox_material"),
            )
            .expect("Failed to create skybox material");
        self.skybox_material = Some(skybox_material.clone());

        // Create skybox uniform buffer
        let skybox_uniform_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<SkyboxUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create skybox uniform buffer");
        self.skybox_uniform_buffer = Some(skybox_uniform_buffer.clone());

        // Create skybox material instance
        #[allow(clippy::arc_with_non_send_sync)]
        let skybox_binding_group = Arc::new(
            BindingGroup::new()
                .with_buffer(0, skybox_uniform_buffer)
                .with_texture(1, self.prefilter_cubemap.clone().unwrap())
                .with_sampler(2, self.ibl_sampler.clone().unwrap()),
        );

        let skybox_material_instance = Arc::new(
            MaterialInstance::new(skybox_material).with_binding_group(skybox_binding_group),
        );
        self.skybox_material_instance = Some(skybox_material_instance);

        // Create a minimal mesh for fullscreen triangle (shader uses vertex_index only)
        let skybox_vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(4)) // Minimal stride, no attributes
                .with_label("skybox_vertex_layout"),
        );

        let skybox_mesh = device
            .create_mesh(
                &MeshDescriptor::new(skybox_vertex_layout)
                    .with_vertex_count(3)
                    .with_label("skybox_triangle"),
            )
            .expect("Failed to create skybox mesh");

        // Write minimal vertex data (shader doesn't use it, just needs valid buffer)
        if let Some(vb) = skybox_mesh.vertex_buffer(0) {
            let dummy_data: [f32; 3] = [0.0, 0.0, 0.0];
            device
                .write_buffer(vb, 0, bytemuck::cast_slice(&dummy_data))
                .expect("Failed to write skybox vertex buffer");
        }
        self.skybox_mesh = Some(skybox_mesh);

        log::info!("Skybox resources created");
    }

    fn create_depth_texture(&mut self, ctx: &AppContext) {
        profile_scope!("create_depth_texture");

        let device = ctx.device();
        let width = ctx.width();
        let height = ctx.height();

        // Create depth texture
        let depth_texture = device
            .create_texture(&TextureDescriptor::new_2d(
                width,
                height,
                TextureFormat::Depth32Float,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .expect("Failed to create depth texture");
        self.depth_texture = Some(depth_texture);

        // Create G-buffer textures for deferred rendering

        // RT0: Albedo (sRGB for correct color)
        let gbuffer_albedo = device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::Rgba8UnormSrgb,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("gbuffer_albedo"),
            )
            .expect("Failed to create G-buffer albedo");
        self.gbuffer_albedo = Some(gbuffer_albedo);

        // RT1: Normal (RGB) + Metallic (A) - high precision for normals
        let gbuffer_normal_metallic = device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::Rgba16Float,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("gbuffer_normal_metallic"),
            )
            .expect("Failed to create G-buffer normal/metallic");
        self.gbuffer_normal_metallic = Some(gbuffer_normal_metallic);

        // RT2: Position (RGB) + Roughness (A) - high precision for world positions
        let gbuffer_position_roughness = device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::Rgba16Float,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("gbuffer_position_roughness"),
            )
            .expect("Failed to create G-buffer position/roughness");
        self.gbuffer_position_roughness = Some(gbuffer_position_roughness);
    }

    fn create_resolve_resources(&mut self, ctx: &AppContext) {
        profile_scope!("create_resolve_resources");

        let device = ctx.device();

        // Create G-buffer sampler
        let gbuffer_sampler = device
            .create_sampler_from_cpu(&CpuSampler::nearest().with_name("gbuffer_sampler"))
            .expect("Failed to create G-buffer sampler");
        self.gbuffer_sampler = Some(gbuffer_sampler.clone());

        // Create resolve uniform buffer
        let resolve_uniform_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<ResolveUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create resolve uniform buffer");
        self.resolve_uniform_buffer = Some(resolve_uniform_buffer.clone());

        // Create binding layouts for resolve pass
        // Group 0: Resolve uniforms
        let resolve_uniform_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
                )
                .with_label("resolve_uniform_bindings"),
        );

        // Group 1: G-buffer textures
        let gbuffer_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::Texture)
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
                .with_label("gbuffer_bindings"),
        );

        // Group 2: IBL textures
        let ibl_layout = Arc::new(
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

        // Compose resolve shader with HDR define if HDR is active
        let mut shader_composer =
            ShaderComposer::with_standard_library().expect("Failed to create shader composer");

        let shader_defs: Vec<(&str, ShaderDef)> = if self.hdr_active {
            log::info!("Compiling resolve shader with HDR_OUTPUT define");
            vec![("HDR_OUTPUT", ShaderDef::Bool(true))]
        } else {
            log::info!("Compiling resolve shader for SDR output");
            vec![]
        };

        let composed_resolve_shader = shader_composer
            .compose(RESOLVE_SHADER_WGSL, &shader_defs)
            .expect("Failed to compose resolve shader");

        log::info!("Resolve shader composed with library imports");

        // Create resolve material
        let resolve_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        composed_resolve_shader.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        composed_resolve_shader.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(resolve_uniform_layout)
                    .with_binding_layout(gbuffer_layout)
                    .with_binding_layout(ibl_layout)
                    .with_label("resolve_material"),
            )
            .expect("Failed to create resolve material");
        self.resolve_material = Some(resolve_material.clone());

        // Create binding groups
        #[allow(clippy::arc_with_non_send_sync)]
        let resolve_uniform_binding =
            Arc::new(BindingGroup::new().with_buffer(0, resolve_uniform_buffer));

        #[allow(clippy::arc_with_non_send_sync)]
        let gbuffer_binding = Arc::new(
            BindingGroup::new()
                .with_texture(0, self.gbuffer_albedo.clone().unwrap())
                .with_texture(1, self.gbuffer_normal_metallic.clone().unwrap())
                .with_texture(2, self.gbuffer_position_roughness.clone().unwrap())
                .with_sampler(3, gbuffer_sampler),
        );

        #[allow(clippy::arc_with_non_send_sync)]
        let ibl_binding = Arc::new(
            BindingGroup::new()
                .with_texture(0, self.irradiance_cubemap.clone().unwrap())
                .with_texture(1, self.prefilter_cubemap.clone().unwrap())
                .with_texture(2, self.brdf_lut.clone().unwrap())
                .with_sampler(3, self.ibl_sampler.clone().unwrap()),
        );

        // Create material instance
        let resolve_material_instance = Arc::new(
            MaterialInstance::new(resolve_material)
                .with_binding_group(resolve_uniform_binding)
                .with_binding_group(gbuffer_binding)
                .with_binding_group(ibl_binding),
        );
        self.resolve_material_instance = Some(resolve_material_instance);

        // Create a minimal mesh for fullscreen triangle (shader generates vertices)
        let resolve_vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(4))
                .with_label("resolve_vertex_layout"),
        );

        let resolve_mesh = device
            .create_mesh(
                &MeshDescriptor::new(resolve_vertex_layout)
                    .with_vertex_count(3)
                    .with_label("resolve_triangle"),
            )
            .expect("Failed to create resolve mesh");

        if let Some(vb) = resolve_mesh.vertex_buffer(0) {
            let dummy_data: [f32; 3] = [0.0, 0.0, 0.0];
            device
                .write_buffer(vb, 0, bytemuck::cast_slice(&dummy_data))
                .expect("Failed to write resolve mesh vertex buffer");
        }
        self.resolve_mesh = Some(resolve_mesh);

        log::info!("Resolve pass resources created");
    }

    fn update_resolve_uniforms(&self, ctx: &AppContext) {
        let uniforms = ResolveUniforms {
            camera_pos: self.camera.position().extend(1.0).to_array(),
            screen_size: [
                ctx.width() as f32,
                ctx.height() as f32,
                1.0 / ctx.width() as f32,
                1.0 / ctx.height() as f32,
            ],
        };

        if let Some(buffer) = &self.resolve_uniform_buffer {
            ctx.device()
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms))
                .expect("Failed to write resolve uniform buffer");
        }
    }

    fn create_sphere_instances(&self) -> Vec<SphereInstance> {
        // Get base color from UI if available
        let base_color = if let Ok(ui) = self.egui_ui.read() {
            let c = ui.state().base_color;
            [c[0], c[1], c[2], 1.0]
        } else {
            [0.9, 0.1, 0.1, 1.0]
        };

        // Get spacing from UI if available
        let spacing = if let Ok(ui) = self.egui_ui.read() {
            ui.state().sphere_spacing
        } else {
            SPHERE_SPACING
        };

        let mut instances = Vec::with_capacity(GRID_SIZE * GRID_SIZE);
        let offset = (GRID_SIZE as f32 - 1.0) * spacing / 2.0;

        for row in 0..GRID_SIZE {
            for col in 0..GRID_SIZE {
                let x = col as f32 * spacing - offset;
                let z = row as f32 * spacing - offset;

                let model = Mat4::from_translation(Vec3::new(x, 0.0, z));
                let metallic = col as f32 / (GRID_SIZE - 1) as f32;
                let roughness = (row as f32 / (GRID_SIZE - 1) as f32).max(0.05);

                instances.push(SphereInstance {
                    model: model.to_cols_array_2d(),
                    base_color,
                    metallic_roughness: [metallic, roughness, 0.0, 0.0],
                });
            }
        }

        instances
    }

    fn update_sphere_instances(&mut self, ctx: &AppContext) {
        profile_scope!("update_sphere_instances");
        let instances = self.create_sphere_instances();
        let instance_data = bytemuck::cast_slice(&instances);
        if let Some(buffer) = &self.instance_buffer {
            ctx.device()
                .write_buffer(buffer, 0, instance_data)
                .expect("Failed to write instance buffer");
        }
    }

    fn update_camera_buffer(&self, ctx: &AppContext) {
        profile_scope!("update_camera_buffer");
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
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms))
                .expect("Failed to write camera uniform buffer");
        }
    }

    fn update_skybox_buffer(&self, ctx: &AppContext) {
        profile_scope!("update_skybox_buffer");
        let view = self.camera.view_matrix();
        let proj = self.camera.projection_matrix(ctx.aspect_ratio());
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        let uniforms = SkyboxUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: self.camera.position().extend(1.0).to_array(),
            mip_level: self.skybox_mip_level,
            _pad0: [0.0; 3],
            _pad1: [0.0; 4],
        };

        if let Some(buffer) = &self.skybox_uniform_buffer {
            ctx.device()
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms))
                .expect("Failed to write skybox uniform buffer");
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

        config
    }
}

impl Default for PbrIblDemo {
    fn default() -> Self {
        Self::new()
    }
}

impl AppHandler for PbrIblDemo {
    fn on_init(&mut self, ctx: &mut AppContext) {
        profile_function!();
        profile_message!("PBR Demo: Initializing");

        log::info!("Initializing Deferred PBR IBL Demo");
        log::info!(
            "Grid: {}x{} spheres with varying metallic/roughness",
            GRID_SIZE,
            GRID_SIZE
        );
        log::info!("Deferred rendering with G-buffer + IBL resolve pass");
        log::info!(
            "Surface format: {:?}, HDR: {}",
            ctx.surface_format(),
            ctx.hdr_active()
        );
        log::info!("Controls:");
        log::info!("  - Left mouse drag: Rotate camera");
        log::info!("  - Scroll: Zoom");
        log::info!("  - H: Toggle UI visibility");

        // Store HDR status for shader compilation
        self.hdr_active = ctx.hdr_active();

        self.create_gpu_resources(ctx);

        // Create resolve pass resources (needs G-buffer textures to exist)
        self.create_resolve_resources(ctx);

        // Initialize per-frame ring buffer for uniform data streaming
        ctx.pipeline_mut()
            .create_ring_buffers(
                4 * 1024, // 4 KB per frame
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                "per_frame_uniforms",
            )
            .expect("Failed to create per-frame ring buffers");
        log::info!(
            "Created per-frame ring buffers: {} frames x {} bytes",
            ctx.pipeline().frames_in_flight(),
            ctx.pipeline().ring_buffer_capacity().unwrap_or(0)
        );

        // Initialize egui controller
        let mut egui_controller = EguiController::new(
            ctx.device().clone(),
            self.egui_ui.clone(),
            ctx.width(),
            ctx.height(),
            ctx.scale_factor(),
        );

        // Register all G-buffer textures with egui for UI visualization
        if let Some(albedo) = &self.gbuffer_albedo {
            let albedo_id = egui_controller.register_user_texture(albedo.clone());
            self.gbuffer_albedo_id = Some(albedo_id);
        }
        if let Some(normal) = &self.gbuffer_normal_metallic {
            let normal_id = egui_controller.register_user_texture(normal.clone());
            self.gbuffer_normal_id = Some(normal_id);
        }
        if let Some(position) = &self.gbuffer_position_roughness {
            let position_id = egui_controller.register_user_texture(position.clone());
            self.gbuffer_position_id = Some(position_id);
        }

        // Pass the texture IDs to the UI
        if let Ok(mut ui) = self.egui_ui.write() {
            ui.set_gbuffer_texture_ids(
                self.gbuffer_albedo_id,
                self.gbuffer_normal_id,
                self.gbuffer_position_id,
            );
        }

        self.egui_controller = Some(egui_controller);
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        self.create_depth_texture(ctx);

        // Recreate resolve resources with new G-buffer textures
        self.create_resolve_resources(ctx);

        // Update the registered G-buffer textures with egui after resize
        if let Some(egui) = &mut self.egui_controller {
            if let (Some(id), Some(tex)) = (self.gbuffer_albedo_id, &self.gbuffer_albedo) {
                egui.update_user_texture(id, tex.clone());
            }
            if let (Some(id), Some(tex)) = (self.gbuffer_normal_id, &self.gbuffer_normal_metallic) {
                egui.update_user_texture(id, tex.clone());
            }
            if let (Some(id), Some(tex)) =
                (self.gbuffer_position_id, &self.gbuffer_position_roughness)
            {
                egui.update_user_texture(id, tex.clone());
            }
            egui.on_resize(ctx.width(), ctx.height());
        }
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        profile_scope!("on_update");

        // Process UI state changes
        if let Ok(mut ui) = self.egui_ui.write() {
            if ui.take_state_changed() {
                let state = ui.state().clone();

                // Update skybox mip level
                self.skybox_mip_level = state.skybox_mip_level;

                // Update camera distance
                self.camera.distance = state.camera_distance;

                // Update sphere instances if color or spacing changed
                self.needs_instance_update = true;
            }

            // Sync camera distance back to UI
            ui.set_camera_distance(self.camera.distance);

            // Auto-rotate if enabled
            if ui.state().auto_rotate && !self.mouse_pressed {
                self.camera.rotate(ctx.delta_time() * 0.15, 0.0);
            }
        }

        // Update instances if needed
        if self.needs_instance_update {
            self.update_sphere_instances(ctx);
            self.needs_instance_update = false;
        }

        self.update_camera_buffer(ctx);
        self.update_skybox_buffer(ctx);
        self.update_resolve_uniforms(ctx);
        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        profile_scope!("on_draw");

        // Allocate space for per-frame uniforms from the ring buffer.
        if ctx.has_ring_buffer() {
            self.frame_allocations = FrameUniformAllocations {
                camera: ctx.allocate(std::mem::size_of::<CameraUniforms>() as u64),
                skybox: ctx.allocate(std::mem::size_of::<SkyboxUniforms>() as u64),
                resolve: ctx.allocate(std::mem::size_of::<ResolveUniforms>() as u64),
            };

            // Log ring buffer usage on first few frames
            if let Some(ring) = ctx.ring_buffer()
                && ctx.frame_number() < 3
            {
                log::debug!(
                    "Ring buffer slot {}: allocated {} bytes, {} remaining",
                    ctx.frame_slot(),
                    ring.used(),
                    ring.remaining()
                );
            }
        }

        let mut graph = ctx.acquire_graph();

        // Upload IBL textures on first frame
        if self.needs_ibl_upload {
            let transfer_config = self.create_ibl_transfer_config();
            let mut transfer_pass = TransferPass::new("ibl_upload".into());
            transfer_pass.set_transfer_config(transfer_config);
            graph.add_transfer_pass(transfer_pass);
            self.needs_ibl_upload = false;
            log::info!("IBL textures uploaded via transfer pass");
        }

        // === Pass 1: G-Buffer Pass ===
        let mut gbuffer_pass = GraphicsPass::new("gbuffer".into());

        if let (Some(depth), Some(albedo), Some(normal), Some(position)) = (
            &self.depth_texture,
            &self.gbuffer_albedo,
            &self.gbuffer_normal_metallic,
            &self.gbuffer_position_roughness,
        ) {
            gbuffer_pass.set_render_targets(
                RenderTargetConfig::new()
                    .with_color(
                        ColorAttachment::from_texture(albedo.clone())
                            .with_clear_color(0.0, 0.0, 0.0, 0.0),
                    )
                    .with_color(
                        ColorAttachment::from_texture(normal.clone())
                            .with_clear_color(0.5, 0.5, 0.5, 0.0),
                    )
                    .with_color(
                        ColorAttachment::from_texture(position.clone())
                            .with_clear_color(0.0, 0.0, 0.0, 0.0),
                    )
                    .with_depth_stencil(
                        DepthStencilAttachment::from_texture(depth.clone()).with_clear_depth(1.0),
                    ),
            );
        }

        // Draw spheres to G-buffer
        if let (Some(mesh), Some(material_instance)) = (&self.mesh, &self.material_instance) {
            gbuffer_pass.add_draw_instanced(
                mesh.clone(),
                material_instance.clone(),
                (GRID_SIZE * GRID_SIZE) as u32,
            );
        }

        graph.add_graphics_pass(gbuffer_pass);

        // === Pass 2: Skybox Pass ===
        let mut skybox_pass = GraphicsPass::new("skybox".into());

        skybox_pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_surface(ctx.swapchain_texture())
                    .with_clear_color(0.02, 0.02, 0.03, 1.0),
            ),
        );

        if let (Some(skybox_mesh), Some(skybox_instance)) =
            (&self.skybox_mesh, &self.skybox_material_instance)
        {
            skybox_pass.add_draw(skybox_mesh.clone(), skybox_instance.clone());
        }

        graph.add_graphics_pass(skybox_pass);

        // === Pass 3: Resolve/Lighting Pass ===
        let mut resolve_pass = GraphicsPass::new("resolve".into());

        resolve_pass.set_render_targets(RenderTargetConfig::new().with_color(
            ColorAttachment::from_surface(ctx.swapchain_texture()).with_load_op(LoadOp::Load),
        ));

        if let (Some(mesh), Some(material_instance)) =
            (&self.resolve_mesh, &self.resolve_material_instance)
        {
            resolve_pass.add_draw(mesh.clone(), material_instance.clone());
        }

        let resolve_handle = graph.add_graphics_pass(resolve_pass);

        // === Pass 4: Egui Pass ===
        if let Some(egui) = &mut self.egui_controller {
            let width = ctx.width();
            let height = ctx.height();
            let elapsed = ctx.elapsed_time() as f64;
            let render_target = RenderTarget::from_surface(ctx.swapchain_texture());

            egui.begin_frame(elapsed);
            if let Some(egui_pass) = egui.end_frame(&render_target, width, height) {
                let egui_handle = graph.add_graphics_pass(egui_pass);
                // Both resolve and egui use LoadOp::Load on the surface (read-modify-write).
                // The auto-dependency system can't infer ordering between two LoadOp::Load
                // passes on the same resource, so we must specify it explicitly.
                graph.add_dependency(egui_handle, resolve_handle);
            }
        }

        let _handle = ctx.submit("main", graph, &[]);

        // Report memory stats to Tracy
        profile_memory_stats!();

        ctx.finish(&[])
    }

    fn on_mouse_move(&mut self, _ctx: &mut AppContext, x: f64, y: f64) {
        // Forward to egui
        let egui_wants_pointer = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_move(x, y)
        } else {
            false
        };

        // Only handle camera if egui doesn't want the input
        if self.mouse_pressed && !egui_wants_pointer {
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
        // Forward to egui
        let egui_wants_pointer = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_button(button, pressed)
        } else {
            false
        };

        // Only handle camera if egui doesn't want the input
        if button == winit::event::MouseButton::Left && !egui_wants_pointer {
            self.mouse_pressed = pressed;
        }
    }

    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, _dx: f32, dy: f32) {
        // Forward to egui
        let egui_wants_pointer = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, dy))
        } else {
            false
        };

        // Only handle camera if egui doesn't want the input
        if !egui_wants_pointer {
            self.camera.zoom(dy * 0.5);
        }
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        // Forward to egui
        let egui_wants_keyboard = if let Some(egui) = &mut self.egui_controller {
            egui.on_key(event)
        } else {
            false
        };

        // Track shift key state
        if let PhysicalKey::Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) = event.physical_key {
            self.shift_pressed = event.state.is_pressed();
            return;
        }

        // Only handle key press events
        if !event.state.is_pressed() {
            return;
        }

        // H key toggles UI visibility (always handled, not passed to egui)
        if let PhysicalKey::Code(KeyCode::KeyH) = event.physical_key {
            if let Ok(mut ui) = self.egui_ui.write() {
                ui.toggle_visibility();
            }
            return;
        }

        // Don't handle other keys if egui wants keyboard input
        if egui_wants_keyboard {
            return;
        }

        let mip_levels = (PREFILTER_SIZE as f32).log2().floor() as u32 + 1;

        match event.physical_key {
            // Shift + number: change skybox MIP level
            PhysicalKey::Code(KeyCode::Digit0) if self.shift_pressed => {
                self.skybox_mip_level = 0.0;
                log::info!("Skybox MIP level: 0");
            }
            PhysicalKey::Code(KeyCode::Digit1) if self.shift_pressed => {
                self.skybox_mip_level = 1.0_f32.min(mip_levels as f32 - 1.0);
                log::info!("Skybox MIP level: {}", self.skybox_mip_level);
            }
            PhysicalKey::Code(KeyCode::Digit2) if self.shift_pressed => {
                self.skybox_mip_level = 2.0_f32.min(mip_levels as f32 - 1.0);
                log::info!("Skybox MIP level: {}", self.skybox_mip_level);
            }
            PhysicalKey::Code(KeyCode::Digit3) if self.shift_pressed => {
                self.skybox_mip_level = 3.0_f32.min(mip_levels as f32 - 1.0);
                log::info!("Skybox MIP level: {}", self.skybox_mip_level);
            }
            PhysicalKey::Code(KeyCode::Digit4) if self.shift_pressed => {
                self.skybox_mip_level = 4.0_f32.min(mip_levels as f32 - 1.0);
                log::info!("Skybox MIP level: {}", self.skybox_mip_level);
            }
            PhysicalKey::Code(KeyCode::Digit5) if self.shift_pressed => {
                self.skybox_mip_level = 5.0_f32.min(mip_levels as f32 - 1.0);
                log::info!("Skybox MIP level: {}", self.skybox_mip_level);
            }
            PhysicalKey::Code(KeyCode::Digit6) if self.shift_pressed => {
                self.skybox_mip_level = 6.0_f32.min(mip_levels as f32 - 1.0);
                log::info!("Skybox MIP level: {}", self.skybox_mip_level);
            }
            PhysicalKey::Code(KeyCode::Digit7) if self.shift_pressed => {
                self.skybox_mip_level = 7.0_f32.min(mip_levels as f32 - 1.0);
                log::info!("Skybox MIP level: {}", self.skybox_mip_level);
            }

            _ => {}
        }
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down PBR IBL Demo");
    }
}
