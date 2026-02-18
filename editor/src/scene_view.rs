//! Scene view state for rendering 3D content into an egui dock panel.
//!
//! Renders directly to the swapchain using a viewport/scissor that matches
//! the egui SceneView panel rect. No offscreen color texture is used.

use std::sync::Arc;

use redlilium_core::math::{Mat4, Vec3, look_at_rh, mat4_to_cols_array_2d, perspective_rh};
use redlilium_core::mesh::generators;
use redlilium_graphics::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BufferDescriptor, BufferUsage,
    ColorAttachment, DepthStencilAttachment, GraphicsDevice, GraphicsPass, Material,
    MaterialDescriptor, MaterialInstance, Mesh, RenderTargetConfig, ScissorRect, ShaderSource,
    ShaderStage, ShaderStageFlags, SurfaceTexture, TextureDescriptor, TextureFormat, TextureUsage,
    Viewport,
};

/// WGSL shader for a flat-colored cube with simple directional lighting.
const SCENE_SHADER_WGSL: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4<f32>(position, 1.0);
    out.normal = normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ndotl = max(dot(in.normal, light_dir), 0.0);
    let base_color = vec3<f32>(0.2, 0.5, 0.8);
    let color = base_color * (0.3 + 0.7 * ndotl);
    return vec4<f32>(color, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
}

/// Manages GPU resources and rendering for the editor's SceneView panel.
pub struct SceneViewState {
    device: Arc<GraphicsDevice>,
    depth_texture: Arc<redlilium_graphics::Texture>,
    mesh: Arc<Mesh>,
    #[allow(dead_code)]
    material: Arc<Material>,
    material_instance: Arc<MaterialInstance>,
    uniform_buffer: Arc<redlilium_graphics::Buffer>,
    viewport: Option<Viewport>,
    scissor: Option<ScissorRect>,
    last_size: (u32, u32),
    rotation: f32,
}

impl SceneViewState {
    /// Create scene view resources.
    ///
    /// `surface_format` must match the swapchain format since we render directly
    /// to the surface.
    pub fn new(device: Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        // Cube mesh
        let cpu_cube = generators::generate_cube(0.5);
        let mesh = device
            .create_mesh_from_cpu(&cpu_cube)
            .expect("Failed to create cube mesh");

        // Uniform buffer for MVP matrix
        let uniform_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<Uniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create uniform buffer");

        // Binding layout: slot 0 = uniform buffer (vertex stage)
        #[allow(clippy::arc_with_non_send_sync)]
        let binding_layout = Arc::new(
            BindingLayout::new().with_entry(
                BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                    .with_visibility(ShaderStageFlags::VERTEX),
            ),
        );

        // Material (shader + pipeline config)
        let material = device
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
                    .with_binding_layout(binding_layout)
                    .with_vertex_layout(redlilium_graphics::VertexLayout::position_normal())
                    .with_color_format(surface_format)
                    .with_depth_format(TextureFormat::Depth32Float)
                    .with_label("scene_cube_material"),
            )
            .expect("Failed to create scene material");

        // Binding group with the uniform buffer
        #[allow(clippy::arc_with_non_send_sync)]
        let binding_group = Arc::new(BindingGroup::new().with_buffer(0, uniform_buffer.clone()));

        let material_instance =
            Arc::new(MaterialInstance::new(material.clone()).with_binding_group(binding_group));

        // Initial depth texture (will be resized on first frame)
        let initial_w = 256;
        let initial_h = 256;
        let depth_texture = Self::create_depth_texture(&device, initial_w, initial_h);

        Self {
            device,
            depth_texture,
            mesh,
            material,
            material_instance,
            uniform_buffer,
            viewport: None,
            scissor: None,
            last_size: (initial_w, initial_h),
            rotation: 0.0,
        }
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

    /// Recreate the depth texture if the window size changed. Returns `true` if resized.
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

    /// Animate the cube and upload the MVP matrix.
    pub fn update(&mut self, device: &Arc<GraphicsDevice>, delta_time: f32) {
        self.rotation += delta_time * 0.8;

        // Compute aspect from viewport (panel) dimensions
        let aspect = if let Some(vp) = &self.viewport {
            vp.width / vp.height.max(1.0)
        } else {
            let (w, h) = self.last_size;
            w as f32 / h.max(1) as f32
        };

        // Camera
        let proj = perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.1, 100.0);
        let eye = Vec3::new(2.0, 2.0, 2.0);
        let target = Vec3::zeros();
        let up = Vec3::new(0.0, 1.0, 0.0);
        let view = look_at_rh(&eye, &target, &up);

        // Rotation around Y axis
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        #[rustfmt::skip]
        let model = Mat4::new(
             cos_r, 0.0, sin_r, 0.0,
             0.0,   1.0, 0.0,   0.0,
            -sin_r, 0.0, cos_r, 0.0,
             0.0,   0.0, 0.0,   1.0,
        );

        let mvp = proj * view * model;
        let uniforms = Uniforms {
            mvp: mat4_to_cols_array_2d(&mvp),
        };

        let _ = device.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Build a graphics pass that renders the scene into the swapchain
    /// at the SceneView panel's viewport.
    pub fn build_scene_pass(&self, swapchain: &SurfaceTexture) -> GraphicsPass {
        let mut pass = GraphicsPass::new("scene_view".into());

        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(
                    ColorAttachment::from_surface(swapchain)
                        .with_clear_color(0.15, 0.15, 0.15, 1.0),
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

        pass.add_draw(self.mesh.clone(), self.material_instance.clone());
        pass
    }

    /// Whether the viewport has been set (i.e. the SceneView tab is visible).
    pub fn has_viewport(&self) -> bool {
        self.viewport.is_some()
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
