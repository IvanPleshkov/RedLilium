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
use std::sync::{Arc, RwLock};

use glam::{Mat4, Vec3};
use redlilium_app::{App, AppArgs, AppContext, AppHandler, DefaultAppArgs, DrawContext};
use redlilium_graphics::{
    AddressMode, BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BufferDescriptor,
    BufferUsage, ColorAttachment, DepthStencilAttachment, Extent3d, FilterMode, FrameSchedule,
    GraphicsPass, IndexFormat, Material, MaterialDescriptor, MaterialInstance, Mesh,
    MeshDescriptor, RenderGraph, RenderTargetConfig, SamplerDescriptor, ShaderComposer,
    ShaderSource, ShaderStage, ShaderStageFlags, TextureDescriptor, TextureFormat, TextureUsage,
    TransferConfig, TransferOperation, TransferPass, VertexAttribute, VertexAttributeFormat,
    VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
    egui::{EguiApp, EguiController, egui},
};
use winit::event::KeyEvent;
use winit::keyboard::{KeyCode, PhysicalKey};

// === WGSL Shaders ===
//
// Shaders are loaded from external files in demos/shaders/.
// They use the RedLilium shader library via `#import` directives,
// which are resolved by ShaderComposer at runtime.

/// Main PBR shader with IBL - uses shader library imports
const PBR_SHADER_WGSL: &str = include_str!("../../shaders/pbr_ibl.wgsl");

/// Skybox shader - renders environment cubemap as background
const SKYBOX_SHADER_WGSL: &str = include_str!("../../shaders/skybox.wgsl");

// Note: G-buffer debug visualization would require multi-render-target output
// from the PBR shader, which is not implemented yet. The debug mode keys are
// reserved for future implementation.

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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyboxUniforms {
    inv_view_proj: [[f32; 4]; 4], // 64 bytes, offset 0
    camera_pos: [f32; 4],         // 16 bytes, offset 64
    mip_level: f32,               // 4 bytes, offset 80
    _pad0: [f32; 3],              // 12 bytes padding before vec3 (which has 16-byte alignment)
    _pad1: [f32; 4],              // Additional padding to match WGSL vec3<f32> + struct alignment
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

// === Egui UI ===

/// UI state shared between the demo and egui
#[derive(Clone)]
pub struct PbrUiState {
    /// Base color RGB (0.0-1.0)
    pub base_color: [f32; 3],
    /// Skybox MIP level (0-7)
    pub skybox_mip_level: f32,
    /// Camera auto-rotation enabled
    pub auto_rotate: bool,
    /// Camera zoom level
    pub camera_distance: f32,
    /// Whether the UI is visible
    pub ui_visible: bool,
    /// Grid spacing
    pub sphere_spacing: f32,
    /// Whether to show the info panel
    pub show_info: bool,
}

impl Default for PbrUiState {
    fn default() -> Self {
        Self {
            base_color: [0.9, 0.1, 0.1],
            skybox_mip_level: 0.0,
            auto_rotate: true,
            camera_distance: 8.0,
            ui_visible: true,
            sphere_spacing: SPHERE_SPACING,
            show_info: true,
        }
    }
}

/// Egui application for the PBR demo UI
pub struct PbrUi {
    state: PbrUiState,
    state_changed: bool,
}

impl Default for PbrUi {
    fn default() -> Self {
        Self::new()
    }
}

impl PbrUi {
    pub fn new() -> Self {
        Self {
            state: PbrUiState::default(),
            state_changed: true,
        }
    }

    pub fn state(&self) -> &PbrUiState {
        &self.state
    }

    pub fn take_state_changed(&mut self) -> bool {
        let changed = self.state_changed;
        self.state_changed = false;
        changed
    }

    pub fn set_camera_distance(&mut self, distance: f32) {
        self.state.camera_distance = distance;
    }

    pub fn toggle_visibility(&mut self) {
        self.state.ui_visible = !self.state.ui_visible;
    }
}

impl EguiApp for PbrUi {
    fn setup(&mut self, ctx: &egui::Context) {
        // Configure egui style
        let mut style = (*ctx.style()).clone();
        style.visuals.window_corner_radius = egui::CornerRadius::same(8);
        style.spacing.slider_width = 200.0;
        ctx.set_style(style);
    }

    fn update(&mut self, ctx: &egui::Context) {
        if !self.state.ui_visible {
            return;
        }

        egui::Window::new("PBR Controls")
            .default_pos([10.0, 10.0])
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Material");
                ui.separator();

                // Base color picker
                ui.horizontal(|ui| {
                    ui.label("Base Color:");
                    if ui
                        .color_edit_button_rgb(&mut self.state.base_color)
                        .changed()
                    {
                        self.state_changed = true;
                    }
                });

                ui.add_space(10.0);
                ui.heading("Environment");
                ui.separator();

                // Skybox mip level slider
                ui.horizontal(|ui| {
                    ui.label("Skybox Blur:");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.state.skybox_mip_level, 0.0..=7.0)
                                .step_by(0.5),
                        )
                        .changed()
                    {
                        self.state_changed = true;
                    }
                });

                ui.add_space(10.0);
                ui.heading("Camera");
                ui.separator();

                // Auto-rotate checkbox
                if ui
                    .checkbox(&mut self.state.auto_rotate, "Auto Rotate")
                    .changed()
                {
                    self.state_changed = true;
                }

                // Camera distance slider
                ui.horizontal(|ui| {
                    ui.label("Distance:");
                    if ui
                        .add(egui::Slider::new(
                            &mut self.state.camera_distance,
                            2.0..=20.0,
                        ))
                        .changed()
                    {
                        self.state_changed = true;
                    }
                });

                ui.add_space(10.0);
                ui.heading("Grid");
                ui.separator();

                // Sphere spacing slider
                ui.horizontal(|ui| {
                    ui.label("Spacing:");
                    if ui
                        .add(egui::Slider::new(&mut self.state.sphere_spacing, 1.0..=3.0))
                        .changed()
                    {
                        self.state_changed = true;
                    }
                });

                ui.add_space(10.0);

                // Show info checkbox
                ui.checkbox(&mut self.state.show_info, "Show Info Panel");
            });

        // Info panel
        if self.state.show_info {
            egui::Window::new("Info")
                .default_pos([10.0, 400.0])
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("PBR IBL Demo");
                    ui.separator();
                    ui.label(format!("Grid: {}x{} spheres", GRID_SIZE, GRID_SIZE));
                    ui.label("Rows: Roughness (top=smooth)");
                    ui.label("Cols: Metallic (left=dielectric)");
                    ui.separator();
                    ui.label("Controls:");
                    ui.label("  H: Toggle UI");
                    ui.label("  LMB Drag: Rotate camera");
                    ui.label("  Scroll: Zoom");
                });
        }
    }
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
    brdf_staging: Option<Arc<redlilium_graphics::Buffer>>,
    brdf_size: (u32, u32),
    brdf_aligned_bytes_per_row: u32,
    needs_ibl_upload: bool,

    // Egui UI
    egui_controller: Option<EguiController>,
    egui_ui: Arc<RwLock<PbrUi>>,
    needs_instance_update: bool,
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
            brdf_staging: None,
            brdf_size: (0, 0),
            brdf_aligned_bytes_per_row: 0,
            needs_ibl_upload: false,
            egui_controller: None,
            egui_ui: Arc::new(RwLock::new(PbrUi::new())),
            needs_instance_update: false,
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

        // Compose PBR shader using shader library
        // This resolves #import directives and inlines library functions
        let mut shader_composer =
            ShaderComposer::with_standard_library().expect("Failed to create shader composer");
        let composed_pbr_shader = shader_composer
            .compose(PBR_SHADER_WGSL, &[])
            .expect("Failed to compose PBR shader");

        log::info!("PBR shader composed with library imports");

        // Create PBR material with composed shader
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        composed_pbr_shader.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        composed_pbr_shader.as_bytes().to_vec(),
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

        // Create skybox material and resources
        self.create_skybox_resources(ctx);

        // Create depth texture
        self.create_depth_texture(ctx);
    }

    fn create_skybox_resources(&mut self, ctx: &AppContext) {
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
        // Need a buffer but no attributes since the shader doesn't consume any vertex inputs
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
            device.write_buffer(vb, 0, bytemuck::cast_slice(&dummy_data));
        }
        self.skybox_mesh = Some(skybox_mesh);

        log::info!("Skybox resources created");
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
        let instances = self.create_sphere_instances();
        let instance_data = bytemuck::cast_slice(&instances);
        if let Some(buffer) = &self.instance_buffer {
            ctx.device().write_buffer(buffer, 0, instance_data);
        }
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

    fn update_skybox_buffer(&self, ctx: &AppContext) {
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
        log::info!("  - H: Toggle UI visibility");

        self.create_gpu_resources(ctx);

        // Initialize egui controller
        self.egui_controller = Some(EguiController::new(
            ctx.device().clone(),
            self.egui_ui.clone(),
            ctx.width(),
            ctx.height(),
        ));
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        self.create_depth_texture(ctx);
        if let Some(egui) = &mut self.egui_controller {
            egui.on_resize(ctx.width(), ctx.height());
        }
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
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

        // Draw skybox first (fullscreen triangle at far plane)
        if let (Some(skybox_mesh), Some(skybox_instance)) =
            (&self.skybox_mesh, &self.skybox_material_instance)
        {
            render_pass.add_draw(skybox_mesh.clone(), skybox_instance.clone());
        }

        // Draw PBR spheres
        if let (Some(mesh), Some(material_instance)) = (&self.mesh, &self.material_instance) {
            render_pass.add_draw_instanced(
                mesh.clone(),
                material_instance.clone(),
                (GRID_SIZE * GRID_SIZE) as u32,
            );
        }

        graph.add_graphics_pass(render_pass);

        // Add egui pass
        if let Some(egui) = &mut self.egui_controller {
            let width = ctx.width();
            let height = ctx.height();
            let elapsed = ctx.elapsed_time() as f64;

            egui.begin_frame(elapsed);
            if let Some(egui_pass) = egui.end_frame(ctx.swapchain_texture(), width, height) {
                graph.add_graphics_pass(egui_pass);
            }
        }

        let _handle = ctx.submit("main", &graph, &[]);
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

            // Note: Number keys without shift reserved for future G-buffer debug visualization
            _ => {}
        }
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
