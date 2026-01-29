//! Main engine orchestrator

use bevy_ecs::prelude::*;
use crate::backend::traits::*;
use crate::backend::types::{
    BufferDescriptor, BufferUsage, CompareFunction, CullMode, FrontFace,
    PrimitiveTopology, SamplerDescriptor, TextureDescriptor, TextureFormat,
    TextureUsage, Vertex, FilterMode, AddressMode,
};
use crate::backend::wgpu_backend::WgpuBackend;
#[cfg(not(target_arch = "wasm32"))]
use crate::backend::vulkan::VulkanBackend;
use crate::pipeline::{build_deferred_graph, DeferredConfig};
use crate::render_graph::{RenderGraph, RenderGraphExecutor};
use crate::resources::{Material, Mesh};
use crate::scene::{Camera, MainCamera, MeshRenderer, Transform, CameraUniformData, TransformUniformData, AmbientLight};
use crate::{BackendType, EngineConfig};
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use std::collections::HashMap;
use std::sync::Arc;
use winit::window::Window as WinitWindow;

/// Debug visualization mode for G-buffer inspection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GBufferDebugMode {
    /// Normal lit output (default)
    #[default]
    Final,
    /// Show albedo/base color
    Albedo,
    /// Show world-space normals
    Normal,
    /// Show metallic (R) and roughness (G)
    Material,
    /// Show depth buffer (linearized)
    Depth,
}

impl GBufferDebugMode {
    /// Get all available modes for UI display
    pub const ALL: &'static [GBufferDebugMode] = &[
        GBufferDebugMode::Final,
        GBufferDebugMode::Albedo,
        GBufferDebugMode::Normal,
        GBufferDebugMode::Material,
        GBufferDebugMode::Depth,
    ];

    /// Get the display name for this mode
    pub fn name(&self) -> &'static str {
        match self {
            GBufferDebugMode::Final => "Final (Lit)",
            GBufferDebugMode::Albedo => "Albedo",
            GBufferDebugMode::Normal => "Normal",
            GBufferDebugMode::Material => "Material (M/R)",
            GBufferDebugMode::Depth => "Depth",
        }
    }

    /// Get the shader mode value (0-4)
    pub fn shader_value(&self) -> u32 {
        match self {
            GBufferDebugMode::Final => 0,
            GBufferDebugMode::Albedo => 1,
            GBufferDebugMode::Normal => 2,
            GBufferDebugMode::Material => 3,
            GBufferDebugMode::Depth => 4,
        }
    }
}

/// Backend wrapper to abstract over different backends
pub enum Backend {
    Wgpu(WgpuBackend),
    #[cfg(not(target_arch = "wasm32"))]
    Vulkan(VulkanBackend),
}

impl Backend {
    /// Create a new backend (native only - use new_async on web)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(
        window: Arc<WinitWindow>,
        backend_type: BackendType,
        vsync: bool,
    ) -> BackendResult<Self> {
        match backend_type {
            BackendType::Wgpu => Ok(Backend::Wgpu(WgpuBackend::new(window, vsync)?)),
            #[cfg(not(target_arch = "wasm32"))]
            BackendType::Vulkan => Ok(Backend::Vulkan(VulkanBackend::new(window, vsync)?)),
            #[cfg(target_arch = "wasm32")]
            BackendType::Vulkan => Err(BackendError::InitializationFailed(
                "Vulkan backend not available".into()
            )),
        }
    }

    /// Async backend creation (required on web, optional on native)
    pub async fn new_async(
        window: Arc<WinitWindow>,
        backend_type: BackendType,
        vsync: bool,
    ) -> BackendResult<Self> {
        match backend_type {
            BackendType::Wgpu => Ok(Backend::Wgpu(WgpuBackend::new_async(window, vsync).await?)),
            #[cfg(not(target_arch = "wasm32"))]
            BackendType::Vulkan => Ok(Backend::Vulkan(VulkanBackend::new(window, vsync)?)),
            #[cfg(target_arch = "wasm32")]
            BackendType::Vulkan => Err(BackendError::InitializationFailed(
                "Vulkan backend not available on web".into()
            )),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        match self {
            Backend::Wgpu(b) => b.resize(width, height),
            #[cfg(not(target_arch = "wasm32"))]
            Backend::Vulkan(b) => b.resize(width, height),
        }
    }

    pub fn begin_frame(&mut self) -> BackendResult<FrameContext> {
        match self {
            Backend::Wgpu(b) => b.begin_frame(),
            #[cfg(not(target_arch = "wasm32"))]
            Backend::Vulkan(b) => b.begin_frame(),
        }
    }

    pub fn end_frame(&mut self) -> BackendResult<()> {
        match self {
            Backend::Wgpu(b) => b.end_frame(),
            #[cfg(not(target_arch = "wasm32"))]
            Backend::Vulkan(b) => b.end_frame(),
        }
    }

    pub fn swapchain_format(&self) -> TextureFormat {
        match self {
            Backend::Wgpu(b) => b.swapchain_format(),
            #[cfg(not(target_arch = "wasm32"))]
            Backend::Vulkan(b) => b.swapchain_format(),
        }
    }

    /// Get the actual surface size (may be clamped by device limits)
    pub fn surface_size(&self) -> (u32, u32) {
        match self {
            Backend::Wgpu(b) => b.surface_size(),
            #[cfg(not(target_arch = "wasm32"))]
            Backend::Vulkan(b) => b.surface_size(),
        }
    }

    /// Get the wgpu backend (if using wgpu)
    pub fn as_wgpu(&self) -> Option<&WgpuBackend> {
        match self {
            Backend::Wgpu(b) => Some(b),
            #[cfg(not(target_arch = "wasm32"))]
            _ => None,
        }
    }

    /// Get mutable wgpu backend (if using wgpu)
    pub fn as_wgpu_mut(&mut self) -> Option<&mut WgpuBackend> {
        match self {
            Backend::Wgpu(b) => Some(b),
            #[cfg(not(target_arch = "wasm32"))]
            _ => None,
        }
    }

    /// Get the Vulkan backend (if using Vulkan)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn as_vulkan(&self) -> Option<&VulkanBackend> {
        match self {
            Backend::Vulkan(b) => Some(b),
            _ => None,
        }
    }

    /// Get mutable Vulkan backend (if using Vulkan)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn as_vulkan_mut(&mut self) -> Option<&mut VulkanBackend> {
        match self {
            Backend::Vulkan(b) => Some(b),
            _ => None,
        }
    }
}

/// Material uniform data sent to GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    base_color: Vec4,
    metallic: f32,
    roughness: f32,
    _padding: [f32; 2],
}

/// Lighting uniform data for deferred lighting pass
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LightingUniform {
    ambient: Vec4,      // RGB ambient, A unused
    light_dir: Vec4,    // XYZ direction, W unused
    light_color: Vec4,  // RGB color, W intensity
    debug_mode: u32,    // 0=Final, 1=Albedo, 2=Normal, 3=Material, 4=Depth
    near_plane: f32,    // Camera near plane for depth visualization
    far_plane: f32,     // Camera far plane for depth visualization
    _padding: f32,
}

/// GPU resources for a mesh
struct GpuMesh {
    vertex_buffer: BufferHandle,
    index_buffer: BufferHandle,
    index_count: u32,
}

/// Per-object GPU resources
struct GpuObject {
    transform_buffer: BufferHandle,
    transform_bind_group: BindGroupHandle,
}

/// Render state holding GPU resources for deferred rendering
struct RenderState {
    // G-Buffer pipeline (renders geometry to G-buffers)
    gbuffer_pipeline: RenderPipelineHandle,

    // Lighting pipeline (fullscreen deferred lighting)
    lighting_pipeline: RenderPipelineHandle,

    // Bind group layouts
    camera_layout: BindGroupLayoutHandle,
    object_layout: BindGroupLayoutHandle,
    material_layout: BindGroupLayoutHandle,
    gbuffer_layout: BindGroupLayoutHandle, // For reading G-buffers in lighting pass
    lighting_layout: BindGroupLayoutHandle, // For lights uniform

    // Camera resources
    camera_buffer: BufferHandle,
    camera_bind_group: BindGroupHandle,

    // Per-material resources
    material_buffers: HashMap<usize, BufferHandle>,
    material_bind_groups: HashMap<usize, BindGroupHandle>,

    // GPU meshes
    gpu_meshes: HashMap<usize, GpuMesh>,

    // Per-entity GPU resources (maps entity to transform buffer/bind group)
    entity_gpu_objects: HashMap<Entity, GpuObject>,

    // G-Buffer textures
    gbuffer_albedo: TextureHandle,
    gbuffer_albedo_view: TextureViewHandle,
    gbuffer_normal: TextureHandle,
    gbuffer_normal_view: TextureViewHandle,
    gbuffer_material: TextureHandle,
    gbuffer_material_view: TextureViewHandle,
    depth_texture: TextureHandle,
    depth_view: TextureViewHandle,

    // G-buffer bind group for lighting pass
    gbuffer_bind_group: BindGroupHandle,
    gbuffer_sampler: SamplerHandle,

    // Lighting resources
    lighting_uniform_buffer: BufferHandle,
    lighting_bind_group: BindGroupHandle,
}

/// The main graphics engine
pub struct Engine {
    backend: Backend,
    render_graph: RenderGraph,
    graph_executor: RenderGraphExecutor,
    world: World,
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
    width: u32,
    height: u32,
    config: EngineConfig,
    render_state: Option<RenderState>,
    /// Current G-buffer debug visualization mode
    gbuffer_debug_mode: GBufferDebugMode,
}

// G-Buffer shader for deferred rendering - outputs to multiple render targets
const GBUFFER_SHADER: &str = r#"
// G-Buffer generation shader for deferred rendering

struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    position: vec4<f32>,
    near_far: vec4<f32>,
}

struct ObjectUniform {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct MaterialUniform {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> object: ObjectUniform;
@group(2) @binding(0) var<uniform> material: MaterialUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_pos = object.model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = normalize((object.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.uv;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;

    // Albedo: base color
    out.albedo = material.base_color;

    // Normal: encode world-space normal to [0,1] range
    out.normal = vec4<f32>(in.world_normal * 0.5 + 0.5, 1.0);

    // Material: R = metallic, G = roughness
    out.material = vec4<f32>(material.metallic, material.roughness, 0.0, 1.0);

    return out;
}
"#;

// Deferred lighting shader - renders fullscreen quad and computes lighting
// Supports debug visualization of G-buffer contents
const DEFERRED_LIGHTING_SHADER: &str = r#"
// Deferred lighting shader with G-buffer debug visualization

struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    position: vec4<f32>,
    near_far: vec4<f32>,
}

struct LightingUniform {
    ambient: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    debug_mode: u32,    // 0=Final, 1=Albedo, 2=Normal, 3=Material, 4=Depth
    near_plane: f32,
    far_plane: f32,
    _padding: f32,
}

@group(0) @binding(0) var gbuffer_albedo: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(2) var gbuffer_material: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_depth: texture_depth_2d;
@group(0) @binding(4) var gbuffer_sampler: sampler;

@group(1) @binding(0) var<uniform> lighting: LightingUniform;
@group(2) @binding(0) var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Fullscreen triangle
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);

    return out;
}

fn reconstruct_world_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world_pos = camera.inv_view * camera.inv_proj * ndc;
    return world_pos.xyz / world_pos.w;
}

// Linearize depth for visualization
fn linearize_depth(depth: f32, near: f32, far: f32) -> f32 {
    let z = depth;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel_coord = vec2<i32>(in.position.xy);

    // Sample G-buffer
    let albedo = textureLoad(gbuffer_albedo, pixel_coord, 0).rgb;
    let normal_encoded = textureLoad(gbuffer_normal, pixel_coord, 0).rgb;
    let material_data = textureLoad(gbuffer_material, pixel_coord, 0);
    let depth = textureLoad(gbuffer_depth, pixel_coord, 0);

    // Debug mode: Albedo
    if lighting.debug_mode == 1u {
        if depth >= 1.0 {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        return vec4<f32>(albedo, 1.0);
    }

    // Debug mode: Normal
    if lighting.debug_mode == 2u {
        if depth >= 1.0 {
            return vec4<f32>(0.5, 0.5, 1.0, 1.0);
        }
        // Normal is stored as [0,1], show as-is for visualization
        return vec4<f32>(normal_encoded, 1.0);
    }

    // Debug mode: Material (Metallic=R, Roughness=G)
    if lighting.debug_mode == 3u {
        if depth >= 1.0 {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        let metallic = material_data.r;
        let roughness = material_data.g;
        return vec4<f32>(metallic, roughness, 0.0, 1.0);
    }

    // Debug mode: Depth (linearized)
    if lighting.debug_mode == 4u {
        if depth >= 1.0 {
            return vec4<f32>(1.0, 1.0, 1.0, 1.0);
        }
        let linear_depth = linearize_depth(depth, lighting.near_plane, lighting.far_plane);
        let normalized = saturate(linear_depth / lighting.far_plane);
        return vec4<f32>(vec3<f32>(normalized), 1.0);
    }

    // Default: Final lit output (mode 0)
    // Early out for sky
    if depth >= 1.0 {
        return vec4<f32>(0.05, 0.05, 0.08, 1.0);
    }

    // Decode normal
    let normal = normalize(normal_encoded * 2.0 - 1.0);
    let metallic = material_data.r;
    let roughness = material_data.g;

    // Reconstruct world position
    let world_pos = reconstruct_world_position(in.uv, depth);

    // View direction
    let view_dir = normalize(camera.position.xyz - world_pos);

    // Lighting calculation
    let light_dir = normalize(lighting.light_dir.xyz);
    let light_color = lighting.light_color.rgb;
    let ambient = lighting.ambient.rgb;

    // Diffuse
    let ndotl = max(dot(normal, light_dir), 0.0);
    let diffuse = albedo * (1.0 - metallic);

    // Specular
    let half_vec = normalize(light_dir + view_dir);
    let ndoth = max(dot(normal, half_vec), 0.0);
    let shininess = mix(16.0, 128.0, 1.0 - roughness);
    let spec_strength = pow(ndoth, shininess) * (1.0 - roughness);
    let spec_color = mix(vec3<f32>(0.04), albedo, metallic);

    let color = ambient * albedo
              + diffuse * light_color * ndotl
              + spec_color * spec_strength * light_color;

    return vec4<f32>(color, 1.0);
}
"#;

impl Engine {
    /// Create a new engine instance (native only - use new_async on web)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(window: Arc<WinitWindow>, config: EngineConfig) -> BackendResult<Self> {
        let backend = Backend::new(Arc::clone(&window), config.backend, config.vsync)?;
        Self::from_backend(backend, window, config)
    }

    /// Create a new engine instance asynchronously (required on web)
    pub async fn new_async(window: Arc<WinitWindow>, config: EngineConfig) -> BackendResult<Self> {
        let backend = Backend::new_async(Arc::clone(&window), config.backend, config.vsync).await?;
        Self::from_backend(backend, window, config)
    }

    /// Internal: create engine from initialized backend
    fn from_backend(backend: Backend, window: Arc<WinitWindow>, config: EngineConfig) -> BackendResult<Self> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Build the deferred render graph
        let deferred_config = DeferredConfig {
            max_lights: config.max_lights,
            enable_bloom: true,
            enable_fxaa: false,
        };

        let (render_graph, _resources) = build_deferred_graph(width, height, &deferred_config);

        // Create ECS world with default camera
        let mut world = World::new();
        world.insert_resource(AmbientLight::default());

        // Spawn default camera entity
        world.spawn((
            Camera::default(),
            MainCamera,
        ));

        Ok(Self {
            backend,
            render_graph,
            graph_executor: RenderGraphExecutor::new(),
            world,
            meshes: Vec::new(),
            materials: Vec::new(),
            width,
            height,
            config,
            render_state: None,
            gbuffer_debug_mode: GBufferDebugMode::default(),
        })
    }

    /// Initialize rendering resources (must be called after adding meshes/materials)
    fn initialize_render_state(&mut self) -> BackendResult<()> {
        let backend = match &mut self.backend {
            Backend::Wgpu(b) => b,
            #[cfg(not(target_arch = "wasm32"))]
            Backend::Vulkan(_) => return Ok(()), // Skip for Vulkan for now
        };

        // === Bind group layouts ===

        // Camera layout (used by both G-buffer and lighting passes)
        let camera_layout = backend.create_bind_group_layout(&[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::VERTEX_FRAGMENT,
            ty: BindingType::UniformBuffer,
        }])?;

        // Object transform layout (G-buffer pass)
        let object_layout = backend.create_bind_group_layout(&[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::VERTEX,
            ty: BindingType::UniformBuffer,
        }])?;

        // Material layout (G-buffer pass)
        let material_layout = backend.create_bind_group_layout(&[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::FRAGMENT,
            ty: BindingType::UniformBuffer,
        }])?;

        // G-buffer texture layout (lighting pass reads G-buffers)
        let gbuffer_layout = backend.create_bind_group_layout(&[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStageFlags::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                },
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStageFlags::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                },
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStageFlags::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                },
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStageFlags::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                },
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStageFlags::FRAGMENT,
                ty: BindingType::Sampler { comparison: false },
            },
        ])?;

        // Lighting uniform layout
        let lighting_layout = backend.create_bind_group_layout(&[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::FRAGMENT,
            ty: BindingType::UniformBuffer,
        }])?;

        // === Create camera buffer ===
        let camera_buffer = backend.create_buffer(&BufferDescriptor {
            label: Some("Camera Buffer".into()),
            size: std::mem::size_of::<CameraUniformData>() as u64,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            mapped_at_creation: false,
        })?;

        let camera_bind_group = backend.create_bind_group(camera_layout, &[(
            0,
            BindGroupEntry::Buffer {
                buffer: camera_buffer,
                offset: 0,
                size: None,
            },
        )])?;

        // === Create G-buffer textures ===

        // Albedo (base color)
        let gbuffer_albedo = backend.create_texture(&TextureDescriptor {
            label: Some("G-Buffer Albedo".into()),
            width: self.width,
            height: self.height,
            depth: 1,
            mip_levels: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        })?;
        let gbuffer_albedo_view = backend.create_texture_view(gbuffer_albedo)?;

        // Normal (world-space)
        let gbuffer_normal = backend.create_texture(&TextureDescriptor {
            label: Some("G-Buffer Normal".into()),
            width: self.width,
            height: self.height,
            depth: 1,
            mip_levels: 1,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        })?;
        let gbuffer_normal_view = backend.create_texture_view(gbuffer_normal)?;

        // Material (metallic, roughness)
        let gbuffer_material = backend.create_texture(&TextureDescriptor {
            label: Some("G-Buffer Material".into()),
            width: self.width,
            height: self.height,
            depth: 1,
            mip_levels: 1,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        })?;
        let gbuffer_material_view = backend.create_texture_view(gbuffer_material)?;

        // Depth buffer
        let depth_texture = backend.create_texture(&TextureDescriptor {
            label: Some("Depth Buffer".into()),
            width: self.width,
            height: self.height,
            depth: 1,
            mip_levels: 1,
            format: TextureFormat::Depth32Float,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
        })?;
        let depth_view = backend.create_texture_view(depth_texture)?;

        // === Create sampler for G-buffer textures ===
        let gbuffer_sampler = backend.create_sampler(&SamplerDescriptor {
            label: Some("G-Buffer Sampler".into()),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            compare: None,
        })?;

        // === Create G-buffer bind group for lighting pass ===
        let gbuffer_bind_group = backend.create_bind_group(gbuffer_layout, &[
            (0, BindGroupEntry::Texture(gbuffer_albedo_view)),
            (1, BindGroupEntry::Texture(gbuffer_normal_view)),
            (2, BindGroupEntry::Texture(gbuffer_material_view)),
            (3, BindGroupEntry::Texture(depth_view)),
            (4, BindGroupEntry::Sampler(gbuffer_sampler)),
        ])?;

        // === Create lighting uniform buffer ===
        let lighting_uniform_buffer = backend.create_buffer(&BufferDescriptor {
            label: Some("Lighting Uniform Buffer".into()),
            size: std::mem::size_of::<LightingUniform>() as u64,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            mapped_at_creation: false,
        })?;

        let lighting_bind_group = backend.create_bind_group(lighting_layout, &[(
            0,
            BindGroupEntry::Buffer {
                buffer: lighting_uniform_buffer,
                offset: 0,
                size: None,
            },
        )])?;

        // === Create G-buffer pipeline ===
        let gbuffer_pipeline = backend.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("G-Buffer Pipeline".into()),
            vertex_shader: GBUFFER_SHADER.into(),
            fragment_shader: Some(GBUFFER_SHADER.into()),
            vertex_layouts: vec![Vertex::layout()],
            bind_group_layouts: vec![camera_layout, object_layout, material_layout],
            primitive_topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::Back,
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
            }),
            // MRT: Albedo, Normal, Material
            color_targets: vec![
                ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                },
                ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                },
                ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                },
            ],
        })?;

        // === Create lighting pipeline ===
        let swapchain_format = backend.swapchain_format();
        let lighting_pipeline = backend.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Deferred Lighting Pipeline".into()),
            vertex_shader: DEFERRED_LIGHTING_SHADER.into(),
            fragment_shader: Some(DEFERRED_LIGHTING_SHADER.into()),
            vertex_layouts: vec![], // Fullscreen triangle, no vertex input
            bind_group_layouts: vec![gbuffer_layout, lighting_layout, camera_layout],
            primitive_topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            depth_stencil: None, // No depth test for fullscreen pass
            color_targets: vec![ColorTargetState {
                format: swapchain_format,
                blend: None,
                write_mask: ColorWrites::ALL,
            }],
        })?;

        // === Upload meshes to GPU ===
        let mut gpu_meshes = HashMap::new();
        for (id, mesh) in self.meshes.iter().enumerate() {
            let vertex_data = mesh.vertex_bytes();
            let index_data = mesh.index_bytes();

            let vertex_buffer = backend.create_buffer_init(
                &BufferDescriptor {
                    label: Some(format!("Vertex Buffer {}", id)),
                    size: vertex_data.len() as u64,
                    usage: BufferUsage::VERTEX,
                    mapped_at_creation: false,
                },
                vertex_data,
            )?;

            let index_buffer = backend.create_buffer_init(
                &BufferDescriptor {
                    label: Some(format!("Index Buffer {}", id)),
                    size: index_data.len() as u64,
                    usage: BufferUsage::INDEX,
                    mapped_at_creation: false,
                },
                index_data,
            )?;

            gpu_meshes.insert(id, GpuMesh {
                vertex_buffer,
                index_buffer,
                index_count: mesh.index_count() as u32,
            });
        }

        // === Create material resources ===
        let mut material_buffers = HashMap::new();
        let mut material_bind_groups = HashMap::new();

        for (id, material) in self.materials.iter().enumerate() {
            let uniform = MaterialUniform {
                base_color: material.base_color,
                metallic: material.metallic,
                roughness: material.roughness,
                _padding: [0.0; 2],
            };

            let buffer = backend.create_buffer_init(
                &BufferDescriptor {
                    label: Some(format!("Material Buffer {}", id)),
                    size: std::mem::size_of::<MaterialUniform>() as u64,
                    usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                },
                bytemuck::bytes_of(&uniform),
            )?;

            let bind_group = backend.create_bind_group(material_layout, &[(
                0,
                BindGroupEntry::Buffer {
                    buffer,
                    offset: 0,
                    size: None,
                },
            )])?;

            material_buffers.insert(id, buffer);
            material_bind_groups.insert(id, bind_group);
        }

        // === Create per-entity resources using ECS query ===
        let mut entity_gpu_objects = HashMap::new();
        let mut query = self.world.query::<(Entity, &MeshRenderer, &Transform)>();
        for (entity, _renderer, transform) in query.iter(&self.world) {
            let transform_buffer = backend.create_buffer_init(
                &BufferDescriptor {
                    label: Some(format!("Transform Buffer {:?}", entity)),
                    size: std::mem::size_of::<TransformUniformData>() as u64,
                    usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                },
                bytemuck::bytes_of(&transform.uniform_data()),
            )?;

            let transform_bind_group = backend.create_bind_group(object_layout, &[(
                0,
                BindGroupEntry::Buffer {
                    buffer: transform_buffer,
                    offset: 0,
                    size: None,
                },
            )])?;

            entity_gpu_objects.insert(entity, GpuObject {
                transform_buffer,
                transform_bind_group,
            });
        }

        self.render_state = Some(RenderState {
            gbuffer_pipeline,
            lighting_pipeline,
            camera_layout,
            object_layout,
            material_layout,
            gbuffer_layout,
            lighting_layout,
            camera_buffer,
            camera_bind_group,
            material_buffers,
            material_bind_groups,
            gpu_meshes,
            entity_gpu_objects,
            gbuffer_albedo,
            gbuffer_albedo_view,
            gbuffer_normal,
            gbuffer_normal_view,
            gbuffer_material,
            gbuffer_material_view,
            depth_texture,
            depth_view,
            gbuffer_bind_group,
            gbuffer_sampler,
            lighting_uniform_buffer,
            lighting_bind_group,
        });

        Ok(())
    }

    /// Get mutable reference to the ECS world
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Get reference to the ECS world
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Add a mesh and return its ID
    pub fn add_mesh(&mut self, mesh: Mesh) -> usize {
        let id = self.meshes.len();
        self.meshes.push(mesh);
        id
    }

    /// Add a material and return its ID
    pub fn add_material(&mut self, material: Material) -> usize {
        let id = self.materials.len();
        self.materials.push(material);
        id
    }

    /// Get a mesh by ID
    pub fn get_mesh(&self, id: usize) -> Option<&Mesh> {
        self.meshes.get(id)
    }

    /// Get a material by ID
    pub fn get_material(&self, id: usize) -> Option<&Material> {
        self.materials.get(id)
    }

    /// Get the current G-buffer debug visualization mode
    pub fn gbuffer_debug_mode(&self) -> GBufferDebugMode {
        self.gbuffer_debug_mode
    }

    /// Set the G-buffer debug visualization mode
    pub fn set_gbuffer_debug_mode(&mut self, mode: GBufferDebugMode) {
        self.gbuffer_debug_mode = mode;
    }

    /// Handle window resize
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.backend.resize(width, height);

            // Get actual surface size (may be clamped by device limits like WebGL2's 2048 max)
            let (actual_width, actual_height) = self.backend.surface_size();

            // Only proceed if size actually changed
            if actual_width == self.width && actual_height == self.height {
                return;
            }

            self.width = actual_width;
            self.height = actual_height;

            // Update camera aspect ratio using ECS query
            let mut query = self.world.query::<(&mut Camera, &MainCamera)>();
            for (mut camera, _) in query.iter_mut(&mut self.world) {
                camera.set_aspect(actual_width as f32, actual_height as f32);
            }

            // Rebuild render graph for new size
            let deferred_config = DeferredConfig {
                max_lights: self.config.max_lights,
                enable_bloom: true,
                enable_fxaa: false,
            };
            let (render_graph, _) = build_deferred_graph(actual_width, actual_height, &deferred_config);
            self.render_graph = render_graph;

            // Recreate G-buffer textures and depth buffer
            if let (Backend::Wgpu(backend), Some(ref mut state)) = (&mut self.backend, &mut self.render_state) {
                // Destroy old textures
                backend.destroy_texture(state.gbuffer_albedo);
                backend.destroy_texture(state.gbuffer_normal);
                backend.destroy_texture(state.gbuffer_material);
                backend.destroy_texture(state.depth_texture);

                // Create new G-buffer textures
                let gbuffer_albedo = backend.create_texture(&TextureDescriptor {
                    label: Some("G-Buffer Albedo".into()),
                    width: actual_width,
                    height: actual_height,
                    depth: 1,
                    mip_levels: 1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                });

                let gbuffer_normal = backend.create_texture(&TextureDescriptor {
                    label: Some("G-Buffer Normal".into()),
                    width: actual_width,
                    height: actual_height,
                    depth: 1,
                    mip_levels: 1,
                    format: TextureFormat::Rgba16Float,
                    usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                });

                let gbuffer_material = backend.create_texture(&TextureDescriptor {
                    label: Some("G-Buffer Material".into()),
                    width: actual_width,
                    height: actual_height,
                    depth: 1,
                    mip_levels: 1,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                });

                let depth_texture = backend.create_texture(&TextureDescriptor {
                    label: Some("Depth Buffer".into()),
                    width: actual_width,
                    height: actual_height,
                    depth: 1,
                    mip_levels: 1,
                    format: TextureFormat::Depth32Float,
                    usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                });

                // Update state if all textures created successfully
                if let (Ok(albedo), Ok(normal), Ok(material), Ok(depth)) =
                    (gbuffer_albedo, gbuffer_normal, gbuffer_material, depth_texture)
                {
                    if let (Ok(albedo_view), Ok(normal_view), Ok(material_view), Ok(depth_view)) = (
                        backend.create_texture_view(albedo),
                        backend.create_texture_view(normal),
                        backend.create_texture_view(material),
                        backend.create_texture_view(depth),
                    ) {
                        state.gbuffer_albedo = albedo;
                        state.gbuffer_albedo_view = albedo_view;
                        state.gbuffer_normal = normal;
                        state.gbuffer_normal_view = normal_view;
                        state.gbuffer_material = material;
                        state.gbuffer_material_view = material_view;
                        state.depth_texture = depth;
                        state.depth_view = depth_view;

                        // Recreate G-buffer bind group with new texture views
                        if let Ok(gbuffer_bind_group) = backend.create_bind_group(state.gbuffer_layout, &[
                            (0, BindGroupEntry::Texture(albedo_view)),
                            (1, BindGroupEntry::Texture(normal_view)),
                            (2, BindGroupEntry::Texture(material_view)),
                            (3, BindGroupEntry::Texture(depth_view)),
                            (4, BindGroupEntry::Sampler(state.gbuffer_sampler)),
                        ]) {
                            state.gbuffer_bind_group = gbuffer_bind_group;
                        }
                    }
                }
            }
        }
    }

    /// Render a frame (convenience method that calls render_scene + end_frame)
    pub fn render(&mut self) -> BackendResult<()> {
        self.render_scene()?;
        self.end_frame()
    }

    /// Render the scene without presenting. Call end_frame() after to present.
    /// Use this when you need to render additional content (like egui) before presenting.
    pub fn render_scene(&mut self) -> BackendResult<()> {
        // Initialize render state if not done yet
        if self.render_state.is_none() && !self.meshes.is_empty() {
            self.initialize_render_state()?;
        }

        // Begin frame
        let frame = self.backend.begin_frame()?;

        // Set swapchain view as external resource
        if let Some(swapchain_id) = self.render_graph.get_external("swapchain") {
            self.graph_executor
                .set_external_view(swapchain_id, frame.swapchain_view);
        }

        match &mut self.backend {
            Backend::Wgpu(backend) => {
                if let Some(ref state) = self.render_state {
                    // Get camera uniform from ECS query
                    let camera_uniform = {
                        let mut query = self.world.query::<(&Camera, &MainCamera)>();
                        query.iter(&self.world)
                            .next()
                            .map(|(camera, _)| camera.uniform_data())
                            .unwrap_or_else(|| Camera::default().uniform_data())
                    };
                    backend.write_buffer(
                        state.camera_buffer,
                        0,
                        bytemuck::bytes_of(&camera_uniform),
                    );

                    // Get ambient light from ECS resource
                    let ambient = self.world.get_resource::<AmbientLight>()
                        .map(|a| a.0)
                        .unwrap_or(Vec3::splat(0.03));

                    // Get camera near/far planes for depth visualization
                    let (near_plane, far_plane) = {
                        let mut query = self.world.query::<(&Camera, &MainCamera)>();
                        query.iter(&self.world)
                            .next()
                            .map(|(camera, _)| (camera.projection.near(), camera.projection.far()))
                            .unwrap_or((0.1, 1000.0))
                    };

                    // Update lighting uniform with debug mode
                    let lighting_uniform = LightingUniform {
                        ambient: Vec4::new(ambient.x, ambient.y, ambient.z, 1.0),
                        light_dir: Vec4::new(0.5, 1.0, 0.3, 0.0).normalize(),
                        light_color: Vec4::new(1.0, 0.98, 0.95, 1.0),
                        debug_mode: self.gbuffer_debug_mode.shader_value(),
                        near_plane,
                        far_plane,
                        _padding: 0.0,
                    };
                    backend.write_buffer(
                        state.lighting_uniform_buffer,
                        0,
                        bytemuck::bytes_of(&lighting_uniform),
                    );

                    // ============================================
                    // PASS 1: G-Buffer Pass (render geometry to MRT)
                    // ============================================
                    backend.begin_render_pass(&RenderPassDescriptor {
                        label: Some("G-Buffer Pass".into()),
                        color_attachments: vec![
                            // Albedo
                            ColorAttachment {
                                view: state.gbuffer_albedo_view,
                                resolve_target: None,
                                load_op: LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
                                store_op: StoreOp::Store,
                            },
                            // Normal
                            ColorAttachment {
                                view: state.gbuffer_normal_view,
                                resolve_target: None,
                                load_op: LoadOp::Clear([0.0, 0.0, 0.0, 0.0]),
                                store_op: StoreOp::Store,
                            },
                            // Material (metallic, roughness)
                            ColorAttachment {
                                view: state.gbuffer_material_view,
                                resolve_target: None,
                                load_op: LoadOp::Clear([0.0, 0.5, 0.0, 0.0]), // Default roughness 0.5
                                store_op: StoreOp::Store,
                            },
                        ],
                        depth_stencil_attachment: Some(DepthStencilAttachment {
                            view: state.depth_view,
                            depth_load_op: LoadOp::Clear([1.0, 0.0, 0.0, 0.0]),
                            depth_store_op: StoreOp::Store,
                            depth_clear_value: 1.0,
                        }),
                    });

                    backend.set_viewport(
                        0.0,
                        0.0,
                        frame.width as f32,
                        frame.height as f32,
                        0.0,
                        1.0,
                    );

                    backend.set_render_pipeline(state.gbuffer_pipeline);
                    backend.set_bind_group(0, state.camera_bind_group);

                    // Draw each entity with MeshRenderer and Transform to G-buffers
                    let mut query = self.world.query::<(Entity, &MeshRenderer, &Transform)>();
                    for (entity, renderer, _transform) in query.iter(&self.world) {
                        // Get GPU resources
                        let gpu_mesh = match state.gpu_meshes.get(&renderer.mesh_id) {
                            Some(m) => m,
                            None => continue,
                        };
                        let material_bind_group = match state.material_bind_groups.get(&renderer.material_id) {
                            Some(bg) => *bg,
                            None => continue,
                        };
                        let gpu_object = match state.entity_gpu_objects.get(&entity) {
                            Some(o) => o,
                            None => continue,
                        };

                        backend.set_bind_group(1, gpu_object.transform_bind_group);
                        backend.set_bind_group(2, material_bind_group);
                        backend.set_vertex_buffer(0, gpu_mesh.vertex_buffer, 0);
                        backend.set_index_buffer(gpu_mesh.index_buffer, 0, IndexFormat::Uint32);
                        backend.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
                    }

                    backend.end_render_pass();

                    // ============================================
                    // PASS 2: Lighting Pass (fullscreen deferred shading)
                    // ============================================
                    backend.begin_render_pass(&RenderPassDescriptor {
                        label: Some("Deferred Lighting Pass".into()),
                        color_attachments: vec![ColorAttachment {
                            view: frame.swapchain_view,
                            resolve_target: None,
                            load_op: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                            store_op: StoreOp::Store,
                        }],
                        depth_stencil_attachment: None,
                    });

                    backend.set_viewport(
                        0.0,
                        0.0,
                        frame.width as f32,
                        frame.height as f32,
                        0.0,
                        1.0,
                    );

                    backend.set_render_pipeline(state.lighting_pipeline);
                    backend.set_bind_group(0, state.gbuffer_bind_group);
                    backend.set_bind_group(1, state.lighting_bind_group);
                    backend.set_bind_group(2, state.camera_bind_group);

                    // Draw fullscreen triangle (3 vertices, no vertex buffer)
                    backend.draw(0..3, 0..1);

                    backend.end_render_pass();
                } else {
                    // No render state yet, just clear
                    backend.begin_render_pass(&RenderPassDescriptor {
                        label: Some("Clear Pass".into()),
                        color_attachments: vec![ColorAttachment {
                            view: frame.swapchain_view,
                            resolve_target: None,
                            load_op: LoadOp::Clear([0.1, 0.1, 0.15, 1.0]),
                            store_op: StoreOp::Store,
                        }],
                        depth_stencil_attachment: None,
                    });
                    backend.end_render_pass();
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            Backend::Vulkan(_backend) => {
                // Vulkan rendering not implemented yet
            }
        }

        Ok(())
    }

    /// End the frame and present to the screen.
    /// Call this after render_scene() and any overlay rendering (like egui).
    pub fn end_frame(&mut self) -> BackendResult<()> {
        self.backend.end_frame()
    }

    /// Get current dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the backend type
    pub fn backend_type(&self) -> BackendType {
        self.config.backend
    }

    /// Get access to the backend for advanced operations (like egui rendering)
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// Get mutable access to the backend
    pub fn backend_mut(&mut self) -> &mut Backend {
        &mut self.backend
    }
}
