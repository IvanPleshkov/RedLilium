use std::sync::Arc;

use redlilium_graphics::device::GraphicsDevice;
use redlilium_graphics::graph::{
    ColorAttachment, DepthStencilAttachment, DrawCommand, GraphicsPass, LoadOp, RenderTarget,
    RenderTargetConfig,
};
use redlilium_graphics::materials::{
    BindingGroup, BindingGroupDescriptor, MaterialDescriptor, MaterialInstance, ShaderSource,
    ShaderStage,
};
use redlilium_graphics::mesh::{
    Mesh, PrimitiveTopology, VertexAttribute, VertexAttributeFormat, VertexAttributeSemantic,
    VertexBufferLayout, VertexLayout,
};
use redlilium_graphics::resources::{RingBuffer, Texture};
use redlilium_graphics::types::{BufferUsage, TextureFormat};

use redlilium_graphics::materials::{BlendState, Material};

use crate::vertex::{DebugUniforms, DebugVertex};

/// Debug draw shader source (Slang).
const SHADER_SOURCE: &str = include_str!("../../shaders/standard/debug_draw.slang");

/// Capacity (bytes) of the per-frame uniform ring. Sized well above the
/// frames-in-flight count so a slot is reused only long after its fence.
const UNIFORM_RING_CAPACITY: u64 = 1 << 16;

/// Capacity (bytes) of the per-frame vertex ring.
const VERTEX_RING_CAPACITY: u64 = 1 << 22;

/// Manages GPU resources for debug line rendering.
///
/// Create once at initialization. Each frame, call [`create_graphics_pass`](Self::create_graphics_pass)
/// with the accumulated vertex data to get a renderable pass.
///
/// Per-frame data (view-projection uniform, line vertices) is written into
/// persistent ring buffers so writes never race a frame still in flight; the
/// uniform is bound as a dynamic offset and the vertices via a mesh buffer
/// offset.
pub struct DebugDrawerRenderer {
    device: Arc<GraphicsDevice>,
    material: Arc<Material>,
    /// Lazily-created uniform binding group (bound with a per-draw dynamic offset).
    uniform_binding: Option<Arc<BindingGroup>>,
    vertex_layout: Arc<VertexLayout>,
    uniform_ring: RingBuffer,
    vertex_ring: RingBuffer,
    /// This frame's offset into `uniform_ring` (set by `update_view_proj`).
    uniform_offset: u32,
}

impl DebugDrawerRenderer {
    /// Create a new debug drawer renderer.
    ///
    /// # Arguments
    /// * `device` - The graphics device
    /// * `surface_format` - The color format of the render target
    /// * `depth_format` - Optional depth format for depth testing against the scene
    pub fn new(
        device: Arc<GraphicsDevice>,
        surface_format: TextureFormat,
        depth_format: Option<TextureFormat>,
    ) -> Self {
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

        // Material with LineList topology and alpha blending (Slang shader)
        let mut material_desc = MaterialDescriptor::new()
            .with_shader(ShaderSource::slang(
                ShaderStage::Vertex,
                SHADER_SOURCE.as_bytes().to_vec(),
                "vs_main",
                vec![],
            ))
            .with_shader(ShaderSource::slang(
                ShaderStage::Fragment,
                SHADER_SOURCE.as_bytes().to_vec(),
                "fs_main",
                vec![],
            ))
            .with_vertex_layout(vertex_layout.clone())
            .with_topology(PrimitiveTopology::LineList)
            .with_blend_state(BlendState::alpha_blending())
            .with_color_format(surface_format)
            // Binding 0 (view-projection) is bound with a per-draw dynamic offset.
            .with_dynamic_uniform(0, 0)
            .with_label("debug_draw_material");

        if let Some(fmt) = depth_format {
            material_desc = material_desc.with_depth_format(fmt);
        }

        let material = device
            .create_material(&material_desc)
            .expect("Failed to create debug draw material");

        let uniform_ring = RingBuffer::new(
            &device,
            UNIFORM_RING_CAPACITY,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "debug_draw_uniform_ring",
        )
        .expect("Failed to create debug draw uniform ring");

        let vertex_ring = RingBuffer::new(
            &device,
            VERTEX_RING_CAPACITY,
            BufferUsage::VERTEX | BufferUsage::COPY_DST,
            "debug_draw_vertex_ring",
        )
        .expect("Failed to create debug draw vertex ring");

        Self {
            device,
            material,
            uniform_binding: None,
            vertex_layout,
            uniform_ring,
            vertex_ring,
            uniform_offset: 0,
        }
    }

    /// Allocate a slot in `ring` (wrapping when full) and write `data`,
    /// returning its byte offset.
    fn ring_push(ring: &mut RingBuffer, data: &[u8]) -> u64 {
        let size = data.len() as u64;
        let alloc = ring.allocate(size).unwrap_or_else(|| {
            ring.reset();
            ring.allocate(size).expect("debug draw ring too small")
        });
        let _ = ring.write(&alloc, data);
        alloc.offset
    }

    /// Update the view-projection matrix uniform.
    ///
    /// Call once per frame before [`create_graphics_pass`](Self::create_graphics_pass).
    /// The matrix should be column-major `[[f32; 4]; 4]`.
    pub fn update_view_proj(&mut self, view_proj: [[f32; 4]; 4]) {
        let uniforms = DebugUniforms { view_proj };
        self.uniform_offset =
            Self::ring_push(&mut self.uniform_ring, bytemuck::bytes_of(&uniforms)) as u32;
    }

    /// Build a mesh + material instance for `vertices` written into the vertex
    /// ring, and the draw command selecting this frame's uniform slot.
    fn build_draw(&mut self, vertices: &[DebugVertex]) -> DrawCommand {
        let vertex_count = vertices.len() as u32;
        let v_off = Self::ring_push(&mut self.vertex_ring, bytemuck::cast_slice(vertices));

        // Non-indexed LineList mesh referencing this frame's vertex ring slot.
        let gpu_mesh = Arc::new(
            Mesh::new(
                Arc::clone(&self.device),
                self.vertex_layout.clone(),
                PrimitiveTopology::LineList,
                vec![self.vertex_ring.buffer().clone()],
                vertex_count,
                None,
                None,
                0,
                Some("debug_draw_mesh".into()),
            )
            .with_buffer_offsets(vec![v_off], 0),
        );

        let uniform_size = std::mem::size_of::<DebugUniforms>() as u64;
        let uniform_binding = match &self.uniform_binding {
            Some(g) => g.clone(),
            None => {
                let g = self
                    .device
                    .create_binding_group(
                        self.material.binding_layouts()[0].clone(),
                        BindingGroupDescriptor::new().with_buffer_range(
                            0,
                            self.uniform_ring.buffer().clone(),
                            0,
                            uniform_size,
                        ),
                    )
                    .expect("debug draw uniform binding group");
                self.uniform_binding = Some(g.clone());
                g
            }
        };

        let material_instance = Arc::new(
            MaterialInstance::new(self.material.clone()).with_binding_group(uniform_binding),
        );

        DrawCommand::new(gpu_mesh, material_instance)
            .with_dynamic_offsets(vec![vec![self.uniform_offset]])
    }

    /// Append debug draw commands to an existing graphics pass.
    ///
    /// Call [`update_view_proj`](Self::update_view_proj) before this method.
    /// Does nothing if `vertices` is empty.
    pub fn append_to_pass(&mut self, pass: &mut GraphicsPass, vertices: &[DebugVertex]) {
        if vertices.is_empty() {
            return;
        }
        let draw = self.build_draw(vertices);
        pass.add_draw_command(draw);
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
        depth_texture: Option<&Arc<Texture>>,
    ) -> Option<GraphicsPass> {
        if vertices.is_empty() {
            return None;
        }

        let draw = self.build_draw(vertices);

        // Build the pass (draws with depth testing against existing scene depth)
        let mut pass = GraphicsPass::new("debug_draw".into());
        let mut target_config = RenderTargetConfig::new()
            .with_color(ColorAttachment::new(render_target.clone()).with_load_op(LoadOp::Load));

        if let Some(depth) = depth_texture {
            target_config = target_config
                .with_depth_stencil(DepthStencilAttachment::from_texture(depth.clone()));
        }

        pass.set_render_targets(target_config);
        pass.add_draw_command(draw);

        Some(pass)
    }
}
