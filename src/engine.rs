//! Main engine orchestrator

use crate::backend::traits::*;
use crate::backend::types::*;
use crate::backend::wgpu_backend::WgpuBackend;
#[cfg(not(target_arch = "wasm32"))]
use crate::backend::vulkan::VulkanBackend;
use crate::pipeline::{build_forward_plus_graph, ForwardPlusConfig};
use crate::render_graph::{RenderGraph, RenderGraphExecutor};
use crate::resources::{Material, Mesh};
use crate::scene::{Scene, CameraUniformData, TransformUniformData};
use crate::{BackendType, EngineConfig};
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use std::collections::HashMap;
use std::sync::Arc;
use winit::window::Window as WinitWindow;

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

/// Render state holding GPU resources
struct RenderState {
    // Pipeline
    pipeline: RenderPipelineHandle,

    // Bind group layouts
    camera_layout: BindGroupLayoutHandle,
    object_layout: BindGroupLayoutHandle,
    material_layout: BindGroupLayoutHandle,

    // Camera resources
    camera_buffer: BufferHandle,
    camera_bind_group: BindGroupHandle,

    // Per-material resources
    material_buffers: HashMap<usize, BufferHandle>,
    material_bind_groups: HashMap<usize, BindGroupHandle>,

    // GPU meshes
    gpu_meshes: HashMap<usize, GpuMesh>,

    // Per-object resources
    gpu_objects: Vec<GpuObject>,

    // Depth buffer
    depth_texture: TextureHandle,
    depth_view: TextureViewHandle,
}

/// The main graphics engine
pub struct Engine {
    backend: Backend,
    render_graph: RenderGraph,
    graph_executor: RenderGraphExecutor,
    scene: Scene,
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
    width: u32,
    height: u32,
    config: EngineConfig,
    render_state: Option<RenderState>,
}

// Basic shader embedded in code
const BASIC_SHADER: &str = r#"
// Basic shader for rendering meshes with simple lighting

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
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Directional light
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let light_color = vec3<f32>(1.0, 0.98, 0.95);
    let ambient = vec3<f32>(0.1, 0.1, 0.12);

    let normal = normalize(in.world_normal);

    // Diffuse: N dot L
    let ndotl = max(dot(normal, light_dir), 0.0);

    // View direction
    let view_dir = normalize(camera.position.xyz - in.world_position);

    // Phong specular: reflect light dir, compare with view dir
    let reflect_dir = reflect(-light_dir, normal);
    let spec_angle = max(dot(view_dir, reflect_dir), 0.0);
    let shininess = mix(16.0, 128.0, 1.0 - material.roughness);
    let specular = pow(spec_angle, shininess) * (1.0 - material.roughness);

    // Combine
    let diffuse = material.base_color.rgb * (1.0 - material.metallic);
    let spec_color = mix(vec3<f32>(1.0), material.base_color.rgb, material.metallic);

    let color = ambient * material.base_color.rgb
              + diffuse * light_color * ndotl
              + spec_color * specular;

    return vec4<f32>(color, material.base_color.a);
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

        // Build the render graph
        let forward_plus_config = ForwardPlusConfig {
            tile_size: config.tile_size,
            max_lights: config.max_lights,
            enable_bloom: true,
            enable_fxaa: false,
        };

        let (render_graph, _resources) = build_forward_plus_graph(width, height, &forward_plus_config);

        Ok(Self {
            backend,
            render_graph,
            graph_executor: RenderGraphExecutor::new(),
            scene: Scene::new(),
            meshes: Vec::new(),
            materials: Vec::new(),
            width,
            height,
            config,
            render_state: None,
        })
    }

    /// Initialize rendering resources (must be called after adding meshes/materials)
    fn initialize_render_state(&mut self) -> BackendResult<()> {
        let backend = match &mut self.backend {
            Backend::Wgpu(b) => b,
            #[cfg(not(target_arch = "wasm32"))]
            Backend::Vulkan(_) => return Ok(()), // Skip for Vulkan for now
        };

        // Create bind group layouts
        let camera_layout = backend.create_bind_group_layout(&[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::VERTEX_FRAGMENT,
            ty: BindingType::UniformBuffer,
        }])?;

        let object_layout = backend.create_bind_group_layout(&[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::VERTEX,
            ty: BindingType::UniformBuffer,
        }])?;

        let material_layout = backend.create_bind_group_layout(&[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStageFlags::FRAGMENT,
            ty: BindingType::UniformBuffer,
        }])?;

        // Create camera buffer
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

        // Create depth buffer
        let depth_texture = backend.create_texture(&TextureDescriptor {
            label: Some("Depth Buffer".into()),
            width: self.width,
            height: self.height,
            depth: 1,
            mip_levels: 1,
            format: TextureFormat::Depth32Float,
            usage: TextureUsage::RENDER_ATTACHMENT,
        })?;

        let depth_view = backend.create_texture_view(depth_texture)?;

        // Create pipeline
        let swapchain_format = backend.swapchain_format();
        let pipeline = backend.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Basic Pipeline".into()),
            vertex_shader: BASIC_SHADER.into(),
            fragment_shader: Some(BASIC_SHADER.into()),
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
            color_targets: vec![ColorTargetState {
                format: swapchain_format,
                blend: None,
                write_mask: ColorWrites::ALL,
            }],
        })?;

        // Upload meshes to GPU
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

        // Create material resources
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

        // Create per-object resources
        let mut gpu_objects = Vec::new();
        for (id, obj) in self.scene.objects.iter().enumerate() {
            let transform_buffer = backend.create_buffer_init(
                &BufferDescriptor {
                    label: Some(format!("Transform Buffer {}", id)),
                    size: std::mem::size_of::<TransformUniformData>() as u64,
                    usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                    mapped_at_creation: false,
                },
                bytemuck::bytes_of(&obj.transform.uniform_data()),
            )?;

            let transform_bind_group = backend.create_bind_group(object_layout, &[(
                0,
                BindGroupEntry::Buffer {
                    buffer: transform_buffer,
                    offset: 0,
                    size: None,
                },
            )])?;

            gpu_objects.push(GpuObject {
                transform_buffer,
                transform_bind_group,
            });
        }

        self.render_state = Some(RenderState {
            pipeline,
            camera_layout,
            object_layout,
            material_layout,
            camera_buffer,
            camera_bind_group,
            material_buffers,
            material_bind_groups,
            gpu_meshes,
            gpu_objects,
            depth_texture,
            depth_view,
        });

        Ok(())
    }

    /// Get mutable reference to the scene
    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// Get reference to the scene
    pub fn scene(&self) -> &Scene {
        &self.scene
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

            // Update camera aspect ratio
            self.scene.camera.set_aspect(actual_width as f32, actual_height as f32);

            // Rebuild render graph for new size
            let forward_plus_config = ForwardPlusConfig {
                tile_size: self.config.tile_size,
                max_lights: self.config.max_lights,
                enable_bloom: true,
                enable_fxaa: false,
            };
            let (render_graph, _) = build_forward_plus_graph(actual_width, actual_height, &forward_plus_config);
            self.render_graph = render_graph;

            // Recreate depth buffer
            if let (Backend::Wgpu(backend), Some(ref mut state)) = (&mut self.backend, &mut self.render_state) {
                // Destroy old depth texture
                backend.destroy_texture(state.depth_texture);

                // Create new depth buffer with actual (clamped) dimensions
                if let Ok(depth_texture) = backend.create_texture(&TextureDescriptor {
                    label: Some("Depth Buffer".into()),
                    width: actual_width,
                    height: actual_height,
                    depth: 1,
                    mip_levels: 1,
                    format: TextureFormat::Depth32Float,
                    usage: TextureUsage::RENDER_ATTACHMENT,
                }) {
                    if let Ok(depth_view) = backend.create_texture_view(depth_texture) {
                        state.depth_texture = depth_texture;
                        state.depth_view = depth_view;
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
                    // Update camera uniform
                    let camera_uniform = self.scene.camera.uniform_data();
                    backend.write_buffer(
                        state.camera_buffer,
                        0,
                        bytemuck::bytes_of(&camera_uniform),
                    );

                    // Begin render pass with depth buffer
                    backend.begin_render_pass(&RenderPassDescriptor {
                        label: Some("Main Pass".into()),
                        color_attachments: vec![ColorAttachment {
                            view: frame.swapchain_view,
                            resolve_target: None,
                            load_op: LoadOp::Clear([0.1, 0.1, 0.15, 1.0]),
                            store_op: StoreOp::Store,
                        }],
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

                    backend.set_render_pipeline(state.pipeline);
                    backend.set_bind_group(0, state.camera_bind_group);

                    // Draw each object
                    for (obj_idx, obj) in self.scene.objects.iter().enumerate() {
                        // Get GPU resources
                        let gpu_mesh = match state.gpu_meshes.get(&obj.mesh_id) {
                            Some(m) => m,
                            None => continue,
                        };
                        let material_bind_group = match state.material_bind_groups.get(&obj.material_id) {
                            Some(bg) => *bg,
                            None => continue,
                        };
                        let gpu_object = match state.gpu_objects.get(obj_idx) {
                            Some(o) => o,
                            None => continue,
                        };

                        // Update object transform (write outside render pass for proper timing)
                        // The transform is already initialized, but we update it here for dynamic objects
                        // Note: In a real engine you'd only update if transform changed

                        backend.set_bind_group(1, gpu_object.transform_bind_group);
                        backend.set_bind_group(2, material_bind_group);
                        backend.set_vertex_buffer(0, gpu_mesh.vertex_buffer, 0);
                        backend.set_index_buffer(gpu_mesh.index_buffer, 0, IndexFormat::Uint32);
                        backend.draw_indexed(0..gpu_mesh.index_count, 0, 0..1);
                    }

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
