use std::sync::Arc;

use redlilium_graphics::device::GraphicsDevice;
use redlilium_graphics::graph::{
    ColorAttachment, GraphicsPass, LoadOp, RenderTarget, RenderTargetConfig,
};
use redlilium_graphics::materials::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, MaterialDescriptor,
    MaterialInstance, ShaderSource, ShaderStage, ShaderStageFlags,
};
use redlilium_graphics::mesh::{
    Mesh, PrimitiveTopology, VertexAttribute, VertexAttributeFormat, VertexAttributeSemantic,
    VertexBufferLayout, VertexLayout,
};
use redlilium_graphics::resources::Buffer;
use redlilium_graphics::shader::ShaderComposer;
use redlilium_graphics::types::{BufferDescriptor, BufferUsage, TextureFormat};

use redlilium_graphics::materials::{BlendState, Material};

use crate::shader::DEBUG_DRAW_SHADER_SOURCE;
use crate::vertex::{DebugUniforms, DebugVertex};

/// Default initial capacity for the vertex buffer (number of vertices).
const DEFAULT_VERTEX_CAPACITY: u32 = 4096;

/// Manages GPU resources for debug line rendering.
///
/// Create once at initialization. Each frame, call [`create_graphics_pass`](Self::create_graphics_pass)
/// with the accumulated vertex data to get a renderable pass.
pub struct DebugDrawerRenderer {
    device: Arc<GraphicsDevice>,
    material: Arc<Material>,
    vertex_layout: Arc<VertexLayout>,
    uniform_buffer: Arc<Buffer>,
    vertex_buffer: Arc<Buffer>,
    vertex_capacity: u32,
    #[allow(dead_code)]
    uniform_binding_layout: Arc<BindingLayout>,
}

impl DebugDrawerRenderer {
    /// Create a new debug drawer renderer.
    ///
    /// # Arguments
    /// * `device` - The graphics device
    /// * `surface_format` - The color format of the render target
    pub fn new(device: Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        // Vertex layout: Position (Float3) + Color (Float4)
        let vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(
                    std::mem::size_of::<DebugVertex>() as u32
                ))
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::Position,
                    format: VertexAttributeFormat::Float3,
                    offset: 0,
                    buffer_index: 0,
                })
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::Color,
                    format: VertexAttributeFormat::Float4,
                    offset: 12,
                    buffer_index: 0,
                })
                .with_label("debug_draw_vertex_layout"),
        );

        // Binding layout: uniform buffer at group 0, binding 0 (vertex stage)
        let uniform_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX),
                )
                .with_label("debug_draw_uniform_bindings"),
        );

        // Compose shader
        let mut shader_composer = ShaderComposer::new();
        let composed_shader = shader_composer
            .compose(DEBUG_DRAW_SHADER_SOURCE, &[])
            .expect("Failed to compose debug draw shader");
        let shader_bytes = composed_shader.as_bytes().to_vec();

        // Material with LineList topology and alpha blending
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        shader_bytes.clone(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        shader_bytes,
                        "fs_main",
                    ))
                    .with_binding_layout(uniform_binding_layout.clone())
                    .with_vertex_layout(vertex_layout.clone())
                    .with_topology(PrimitiveTopology::LineList)
                    .with_blend_state(BlendState::alpha_blending())
                    .with_color_format(surface_format)
                    .with_label("debug_draw_material"),
            )
            .expect("Failed to create debug draw material");

        // Uniform buffer (view-projection matrix)
        let uniform_buffer = device
            .create_buffer(
                &BufferDescriptor::new(
                    std::mem::size_of::<DebugUniforms>() as u64,
                    BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                )
                .with_label("debug_draw_uniforms"),
            )
            .expect("Failed to create debug draw uniform buffer");

        // Initial vertex buffer
        let vertex_stride = std::mem::size_of::<DebugVertex>() as u64;
        let vertex_buffer = device
            .create_buffer(
                &BufferDescriptor::new(
                    DEFAULT_VERTEX_CAPACITY as u64 * vertex_stride,
                    BufferUsage::VERTEX | BufferUsage::COPY_DST,
                )
                .with_label("debug_draw_vertices"),
            )
            .expect("Failed to create debug draw vertex buffer");

        Self {
            device,
            material,
            vertex_layout,
            uniform_buffer,
            vertex_buffer,
            vertex_capacity: DEFAULT_VERTEX_CAPACITY,
            uniform_binding_layout,
        }
    }

    /// Update the view-projection matrix uniform.
    ///
    /// Call once per frame before [`create_graphics_pass`](Self::create_graphics_pass).
    /// The matrix should be column-major `[[f32; 4]; 4]`.
    pub fn update_view_proj(&self, view_proj: [[f32; 4]; 4]) {
        let uniforms = DebugUniforms { view_proj };
        self.device
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms))
            .expect("Failed to write debug draw uniform buffer");
    }

    /// Create a graphics pass for the given debug vertices.
    ///
    /// Typically called with the result of [`DebugDrawer::take_render_data()`](crate::DebugDrawer::take_render_data).
    ///
    /// Returns `None` if `vertices` is empty.
    pub fn create_graphics_pass(
        &mut self,
        vertices: &[DebugVertex],
        render_target: &RenderTarget,
    ) -> Option<GraphicsPass> {
        if vertices.is_empty() {
            return None;
        }

        let vertex_count = vertices.len() as u32;

        // Grow vertex buffer if needed (2x strategy)
        if vertex_count > self.vertex_capacity {
            let new_capacity = vertex_count
                .max(self.vertex_capacity.saturating_mul(2))
                .max(DEFAULT_VERTEX_CAPACITY);

            let vertex_stride = std::mem::size_of::<DebugVertex>() as u64;
            self.vertex_buffer = self
                .device
                .create_buffer(
                    &BufferDescriptor::new(
                        new_capacity as u64 * vertex_stride,
                        BufferUsage::VERTEX | BufferUsage::COPY_DST,
                    )
                    .with_label("debug_draw_vertices"),
                )
                .expect("Failed to grow debug draw vertex buffer");
            self.vertex_capacity = new_capacity;
        }

        // Upload vertex data
        self.device
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices))
            .expect("Failed to write debug draw vertex buffer");

        // Construct Mesh (non-indexed, LineList)
        let gpu_mesh = Arc::new(Mesh::new(
            Arc::clone(&self.device),
            self.vertex_layout.clone(),
            PrimitiveTopology::LineList,
            vec![self.vertex_buffer.clone()],
            vertex_count,
            None,
            None,
            0,
            Some("debug_draw_mesh".into()),
        ));

        // Binding group and material instance
        #[allow(clippy::arc_with_non_send_sync)]
        let uniform_binding =
            Arc::new(BindingGroup::new().with_buffer(0, self.uniform_buffer.clone()));

        let material_instance = Arc::new(
            MaterialInstance::new(self.material.clone()).with_binding_group(uniform_binding),
        );

        // Build the pass (draws on top of existing content)
        let mut pass = GraphicsPass::new("debug_draw".into());
        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(ColorAttachment::new(render_target.clone()).with_load_op(LoadOp::Load)),
        );
        pass.add_draw(gpu_mesh, material_instance);

        Some(pass)
    }
}
