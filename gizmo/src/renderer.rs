//! GPU rendering for the gizmo: an overlay triangle pass patterned on
//! `DebugDrawerRenderer` — per-frame vertices through a persistent ring
//! buffer, view-projection as a dynamic-offset uniform, **no depth test**
//! (the gizmo draws on top of the scene).

use std::sync::Arc;

use redlilium_graphics::device::GraphicsDevice;
use redlilium_graphics::graph::{
    ColorAttachment, DrawCommand, GraphicsPass, LoadOp, RenderTarget, RenderTargetConfig,
};
use redlilium_graphics::materials::{
    BindingGroup, BindingGroupDescriptor, BlendState, Material, MaterialDescriptor,
    MaterialInstance, ShaderSource, ShaderStage,
};
use redlilium_graphics::mesh::{
    Mesh, PrimitiveTopology, VertexAttribute, VertexAttributeFormat, VertexAttributeSemantic,
    VertexBufferLayout, VertexLayout,
};
use redlilium_graphics::resources::RingBuffer;
use redlilium_graphics::types::{BufferUsage, TextureFormat};

use crate::mesh::{GizmoUniforms, GizmoVertex};

/// Same position+color contract as the debug drawer — the shader is shared.
const SHADER_SOURCE: &str = include_str!("../../shaders/standard/debug_draw.slang");

const UNIFORM_RING_CAPACITY: u64 = 1 << 14;
const VERTEX_RING_CAPACITY: u64 = 1 << 20;

/// GPU resources for gizmo rendering. Create once; call
/// [`update_view_proj`](Self::update_view_proj) then
/// [`append_to_pass`](Self::append_to_pass) /
/// [`create_graphics_pass`](Self::create_graphics_pass) each frame.
pub struct GizmoRenderer {
    device: Arc<GraphicsDevice>,
    material: Arc<Material>,
    uniform_binding: Option<Arc<BindingGroup>>,
    vertex_layout: Arc<VertexLayout>,
    uniform_ring: RingBuffer,
    vertex_ring: RingBuffer,
    uniform_offset: u32,
}

impl GizmoRenderer {
    /// `surface_format` — the color format of the target the gizmo draws
    /// into. No depth format: the gizmo intentionally renders over the scene.
    pub fn new(device: Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        let vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(
                    std::mem::size_of::<GizmoVertex>() as u32
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
                .with_label("gizmo_vertex_layout"),
        );

        let material_desc = MaterialDescriptor::new()
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
            .with_topology(PrimitiveTopology::TriangleList)
            .with_blend_state(BlendState::alpha_blending())
            .with_color_format(surface_format)
            .with_dynamic_uniform(0, 0)
            .with_label("gizmo_material");

        let material = device
            .create_material(&material_desc)
            .expect("Failed to create gizmo material");

        let uniform_ring = RingBuffer::new(
            &device,
            UNIFORM_RING_CAPACITY,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "gizmo_uniform_ring",
        )
        .expect("Failed to create gizmo uniform ring");

        let vertex_ring = RingBuffer::new(
            &device,
            VERTEX_RING_CAPACITY,
            BufferUsage::VERTEX | BufferUsage::COPY_DST,
            "gizmo_vertex_ring",
        )
        .expect("Failed to create gizmo vertex ring");

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

    fn ring_push(ring: &mut RingBuffer, data: &[u8]) -> u64 {
        let size = data.len() as u64;
        let alloc = ring.allocate(size).unwrap_or_else(|| {
            ring.reset();
            ring.allocate(size).expect("gizmo ring too small")
        });
        let _ = ring.write(&alloc, data);
        alloc.offset
    }

    /// Write this frame's view-projection (column-major). Call before
    /// building the pass.
    pub fn update_view_proj(&mut self, view_proj: [[f32; 4]; 4]) {
        let uniforms = GizmoUniforms { view_proj };
        self.uniform_offset =
            Self::ring_push(&mut self.uniform_ring, bytemuck::bytes_of(&uniforms)) as u32;
    }

    fn build_draw(&mut self, vertices: &[GizmoVertex]) -> DrawCommand {
        let vertex_count = vertices.len() as u32;
        let v_off = Self::ring_push(&mut self.vertex_ring, bytemuck::cast_slice(vertices));

        let gpu_mesh = Arc::new(
            Mesh::new(
                Arc::clone(&self.device),
                self.vertex_layout.clone(),
                PrimitiveTopology::TriangleList,
                vec![self.vertex_ring.buffer().clone()],
                vertex_count,
                None,
                None,
                0,
                Some("gizmo_mesh".into()),
            )
            .with_buffer_offsets(vec![v_off], 0),
        );

        let uniform_size = std::mem::size_of::<GizmoUniforms>() as u64;
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
                    .expect("gizmo uniform binding group");
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

    /// Append the gizmo draw to an existing pass (e.g. the debug overlay
    /// pass). No-op when `vertices` is empty.
    pub fn append_to_pass(&mut self, pass: &mut GraphicsPass, vertices: &[GizmoVertex]) {
        if vertices.is_empty() {
            return;
        }
        let draw = self.build_draw(vertices);
        pass.add_draw_command(draw);
    }

    /// Build a standalone gizmo pass over `render_target` (loads the
    /// existing image, draws on top, no depth attachment).
    pub fn create_graphics_pass(
        &mut self,
        vertices: &[GizmoVertex],
        render_target: &RenderTarget,
    ) -> Option<GraphicsPass> {
        if vertices.is_empty() {
            return None;
        }
        let draw = self.build_draw(vertices);
        let mut pass = GraphicsPass::new("gizmo".into());
        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(ColorAttachment::new(render_target.clone()).with_load_op(LoadOp::Load)),
        );
        pass.add_draw_command(draw);
        Some(pass)
    }
}
