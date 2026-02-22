//! Scene view state for rendering ECS entities into the editor's SceneView panel.
//!
//! Reads Camera, GlobalTransform, RenderMesh, RenderMaterial, and Visibility
//! from the ECS World and builds a forward rendering pass targeting the
//! swapchain with viewport/scissor matching the egui panel rect.

use std::sync::Arc;

use redlilium_core::math::{Mat4, mat4_to_cols_array_2d};
use redlilium_ecs::{
    Camera, Entity, GlobalTransform, RenderMaterial, RenderMesh, Visibility, World,
};
use redlilium_graphics::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, Buffer, BufferDescriptor,
    BufferUsage, ColorAttachment, DepthStencilAttachment, GraphicsDevice, GraphicsPass, Material,
    MaterialDescriptor, MaterialInstance, Mesh, RenderTargetConfig, ScissorRect, ShaderSource,
    ShaderStage, ShaderStageFlags, SurfaceTexture, TextureDescriptor, TextureFormat, TextureUsage,
    Viewport,
};

/// WGSL shader for lit scene rendering with camera VP + model matrix uniforms.
const SCENE_SHADER_WGSL: &str = r#"
struct Uniforms {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.model * vec4<f32>(position, 1.0);
    out.clip_position = uniforms.view_projection * world_pos;
    out.world_normal = (uniforms.model * vec4<f32>(normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let n = normalize(in.world_normal);
    let ndotl = max(dot(n, light_dir), 0.0);
    let base_color = vec3<f32>(0.6, 0.6, 0.65);
    let ambient = vec3<f32>(0.15, 0.15, 0.18);
    let color = ambient + base_color * ndotl;
    return vec4<f32>(color, 1.0);
}
"#;

/// Per-entity uniform data: VP + model matrix.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

/// Manages GPU resources and rendering for the editor's SceneView panel.
pub struct SceneViewState {
    device: Arc<GraphicsDevice>,
    depth_texture: Arc<redlilium_graphics::Texture>,
    scene_material: Arc<Material>,
    _binding_layout: Arc<BindingLayout>,
    viewport: Option<Viewport>,
    scissor: Option<ScissorRect>,
    last_size: (u32, u32),
}

impl SceneViewState {
    /// Create scene view resources.
    pub fn new(device: Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        // Binding layout: slot 0 = uniform buffer (vertex + fragment)
        let binding_layout = Arc::new(
            BindingLayout::new().with_entry(
                BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                    .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
            ),
        );

        let scene_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        SCENE_SHADER_WGSL.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        SCENE_SHADER_WGSL.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(binding_layout.clone())
                    .with_vertex_layout(redlilium_graphics::VertexLayout::position_normal())
                    .with_color_format(surface_format)
                    .with_depth_format(TextureFormat::Depth32Float)
                    .with_label("editor_scene_material"),
            )
            .expect("Failed to create editor scene material");

        let depth_texture = Self::create_depth_texture(&device, 256, 256);

        Self {
            device,
            depth_texture,
            scene_material,
            _binding_layout: binding_layout,
            viewport: None,
            scissor: None,
            last_size: (256, 256),
        }
    }

    /// Create GPU resources for a renderable entity: uniform buffer + MaterialInstance.
    ///
    /// Returns `(uniform_buffer, gpu_mesh, material_instance)`.
    pub fn create_entity_resources(
        &self,
        cpu_mesh: &redlilium_core::mesh::CpuMesh,
    ) -> (Arc<Buffer>, Arc<Mesh>, Arc<MaterialInstance>) {
        let uniform_buffer = self
            .device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<SceneUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create entity uniform buffer");

        let binding_group = Arc::new(BindingGroup::new().with_buffer(0, uniform_buffer.clone()));

        let material_instance = Arc::new(
            MaterialInstance::new(self.scene_material.clone()).with_binding_group(binding_group),
        );

        let gpu_mesh = self
            .device
            .create_mesh_from_cpu(cpu_mesh)
            .expect("Failed to create entity GPU mesh");

        (uniform_buffer, gpu_mesh, material_instance)
    }

    /// Update the viewport and scissor from an egui panel rect.
    pub fn set_viewport(&mut self, rect: egui::Rect, pixels_per_point: f32) {
        let x = rect.min.x * pixels_per_point;
        let y = rect.min.y * pixels_per_point;
        let w = rect.width() * pixels_per_point;
        let h = rect.height() * pixels_per_point;

        self.viewport = Some(Viewport::new(x, y, w, h));
        self.scissor = Some(ScissorRect::new(x as i32, y as i32, w as u32, h as u32));
    }

    /// Recreate the depth texture if the window size changed.
    pub fn resize_if_needed(&mut self, width: u32, height: u32) -> bool {
        let width = width.max(1);
        let height = height.max(1);

        if (width, height) == self.last_size {
            return false;
        }

        self.depth_texture = Self::create_depth_texture(&self.device, width, height);
        self.last_size = (width, height);
        true
    }

    /// Update per-entity uniform buffers with current camera VP and model matrices.
    pub fn update_uniforms(&self, world: &World, entity_buffers: &[(Entity, Arc<Buffer>)]) {
        // Get camera view-projection matrix (use read_all to include editor-flagged camera)
        let Ok(cameras) = world.read_all::<Camera>() else {
            return;
        };
        let Some((_, camera)) = cameras.iter().next() else {
            return;
        };
        let vp = camera.view_projection();

        // Update each entity's uniform buffer
        let Ok(globals) = world.read::<GlobalTransform>() else {
            return;
        };
        for (entity, buffer) in entity_buffers {
            let model = globals
                .get(entity.index())
                .map(|g| g.0)
                .unwrap_or_else(Mat4::identity);

            let uniforms = SceneUniforms {
                view_projection: mat4_to_cols_array_2d(&vp),
                model: mat4_to_cols_array_2d(&model),
            };

            let _ = self
                .device
                .write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }
    }

    /// Build a graphics pass that renders ECS entities to the swapchain
    /// at the SceneView panel's viewport.
    pub fn build_scene_pass(
        &self,
        world: &World,
        swapchain: &SurfaceTexture,
    ) -> Option<GraphicsPass> {
        let meshes = world.read::<RenderMesh>().ok()?;
        let materials = world.read::<RenderMaterial>().ok()?;
        let visibilities = world.read::<Visibility>().ok()?;

        let mut pass = GraphicsPass::new("scene_view".into());

        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(
                    ColorAttachment::from_surface(swapchain)
                        .with_clear_color(0.055, 0.063, 0.078, 1.0),
                )
                .with_depth_stencil(
                    DepthStencilAttachment::from_texture(self.depth_texture.clone())
                        .with_clear_depth(1.0),
                ),
        );

        if let Some(viewport) = &self.viewport {
            pass.set_viewport(*viewport);
        }
        if let Some(scissor) = &self.scissor {
            pass.set_scissor_rect(*scissor);
        }

        for (entity_idx, render_mesh) in meshes.iter() {
            let Some(render_material) = materials.get(entity_idx) else {
                continue;
            };
            if let Some(vis) = visibilities.get(entity_idx)
                && !vis.is_visible()
            {
                continue;
            }
            pass.add_draw(render_mesh.0.clone(), render_material.0.clone());
        }

        Some(pass)
    }

    /// Clear the viewport (e.g. when the SceneView tab is not visible).
    pub fn clear_viewport(&mut self) {
        self.viewport = None;
        self.scissor = None;
    }

    /// Whether the viewport has been set (i.e. the SceneView tab is visible).
    pub fn has_viewport(&self) -> bool {
        self.viewport.is_some()
    }

    /// Get the viewport aspect ratio, or 1.0 if no viewport is set.
    pub fn aspect_ratio(&self) -> f32 {
        if let Some(vp) = &self.viewport {
            vp.width / vp.height.max(1.0)
        } else {
            let (w, h) = self.last_size;
            w as f32 / h.max(1) as f32
        }
    }

    fn create_depth_texture(
        device: &Arc<GraphicsDevice>,
        width: u32,
        height: u32,
    ) -> Arc<redlilium_graphics::Texture> {
        device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::Depth32Float,
                    TextureUsage::RENDER_ATTACHMENT,
                )
                .with_label("scene_view_depth"),
            )
            .expect("Failed to create scene view depth texture")
    }
}
