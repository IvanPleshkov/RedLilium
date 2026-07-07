//! Final present pass: samples the main camera's off-screen target onto the
//! swapchain with a fullscreen triangle.
//!
//! [`ForwardRender`](redlilium_ecs::ForwardRender) draws into a
//! [`CameraTarget`](redlilium_ecs::CameraTarget) texture (render-to-texture,
//! ADR-009); in the editor the egui overlay composites that texture, in the
//! game runtime this blit does.

use std::sync::Arc;

use redlilium_graphics::{
    BindingGroupDescriptor, ColorAttachment, CpuSampler, DrawCommand, GraphicsDevice, GraphicsPass,
    Material, MaterialDescriptor, MaterialInstance, Mesh, MeshDescriptor, PassHandle, RenderGraph,
    RenderTargetConfig, Sampler, ShaderSource, ShaderStage, SurfaceTexture, Texture, TextureFormat,
    TransferConfig, TransferOperation, TransferPass, VertexBufferLayout, VertexLayout,
};

const BLIT_SHADER_SLANG: &str = r#"
// Present blit: fullscreen triangle sampling the scene texture.

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };
    // V flipped: NDC -1 is the bottom of the swapchain, texel v=0 is the top
    // of the scene texture.
    float2 uvs[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };
    VsOutput output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

[[vk::binding(0, 0)]]
Texture2D source_texture;
[[vk::binding(1, 0)]]
SamplerState source_sampler;

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    return source_texture.Sample(source_sampler, input.uv);
}
"#;

/// Host-side resources for the present blit (material, sampler, and the
/// 3-vertex dummy mesh backing the fullscreen triangle).
pub(crate) struct PresentBlit {
    /// Device handle, kept to build the source binding group eagerly (#40).
    device: Arc<GraphicsDevice>,
    material: Arc<Material>,
    mesh: Arc<Mesh>,
    sampler: Arc<Sampler>,
    /// Material instance cached per source texture (keyed by `Arc` pointer
    /// identity — the camera target is recreated on resize).
    instance: Option<(Arc<MaterialInstance>, usize)>,
    /// Dummy vertex-buffer upload, flushed into the first frame's graph.
    pending_uploads: Vec<TransferOperation>,
}

impl PresentBlit {
    pub(crate) fn new(device: &Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        let sampler = device
            .create_sampler_from_cpu(&CpuSampler::nearest().with_name("present_blit_sampler"))
            .expect("failed to create present blit sampler");

        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        BLIT_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        BLIT_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_color_format(surface_format)
                    .with_label("present_blit"),
            )
            .expect("failed to create present blit material");

        // The fullscreen triangle is generated from SV_VertexID; the mesh
        // only supplies a vertex count (plus a minimal dummy buffer so every
        // backend has something to bind).
        let vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(4))
                .with_label("present_blit_layout"),
        );
        let mesh = device
            .create_mesh(
                &MeshDescriptor::new(vertex_layout)
                    .with_vertex_count(3)
                    .with_label("present_blit_triangle"),
            )
            .expect("failed to create present blit mesh");

        let mut pending_uploads = Vec::new();
        if let Some(vb) = mesh.vertex_buffer(0) {
            let dummy: [f32; 3] = [0.0; 3];
            pending_uploads.push(TransferOperation::write_buffer(
                vb.clone(),
                0,
                Arc::from(bytemuck::cast_slice(&dummy)),
            ));
        }

        Self {
            device: device.clone(),
            material,
            mesh,
            sampler,
            instance: None,
            pending_uploads,
        }
    }

    /// Flush the one-time dummy vertex upload into the frame graph.
    pub(crate) fn flush_uploads(&mut self, graph: &mut RenderGraph) {
        if self.pending_uploads.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.pending_uploads);
        let mut pass = TransferPass::new("present_blit_upload".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(ops));
        graph.add_transfer_pass(pass);
    }

    /// Append the present pass to the frame graph: clears the swapchain and,
    /// if a scene texture exists, draws it fullscreen. `depends_on` orders
    /// the blit after the scene's last writer.
    pub(crate) fn encode(
        &mut self,
        graph: &mut RenderGraph,
        swapchain: &SurfaceTexture,
        source: Option<Arc<Texture>>,
        depends_on: Option<PassHandle>,
        clear_color: [f32; 4],
    ) {
        let mut pass = GraphicsPass::new("present_blit".into());
        pass.set_render_targets(RenderTargetConfig::new().with_color(
            ColorAttachment::from_surface(swapchain).with_clear_color(
                clear_color[0],
                clear_color[1],
                clear_color[2],
                clear_color[3],
            ),
        ));

        if let Some(source) = source {
            let instance = self.instance_for(&source);
            pass.add_draw_command(DrawCommand::new(self.mesh.clone(), instance));
        }

        let handle = graph.add_graphics_pass(pass);
        if let Some(dep) = depends_on {
            graph.add_dependency(handle, dep);
        }
    }

    /// The cached material instance for `source`, rebuilt when the camera
    /// target texture changes (resize).
    fn instance_for(&mut self, source: &Arc<Texture>) -> Arc<MaterialInstance> {
        let key = Arc::as_ptr(source) as usize;
        if let Some((instance, cached_key)) = &self.instance
            && *cached_key == key
        {
            return instance.clone();
        }
        // Eagerly-built binding group (#40): descriptor set created once here,
        // cached per source texture and rebuilt only when the target changes.
        let group = self
            .device
            .create_binding_group(
                self.material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new()
                    .with_texture(0, source.clone())
                    .with_sampler(1, self.sampler.clone()),
            )
            .expect("create present blit binding group");
        #[allow(clippy::arc_with_non_send_sync)]
        let instance =
            Arc::new(MaterialInstance::new(self.material.clone()).with_binding_group(group));
        self.instance = Some((instance.clone(), key));
        instance
    }
}
