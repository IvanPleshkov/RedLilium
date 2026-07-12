//! Deferred resolve/lighting pass resources.

use std::sync::Arc;

use redlilium_core::math::Vec3;
use redlilium_core::profiling::profile_scope;
use redlilium_graphics::{
    BindingGroupDescriptor, BufferUsage, CpuSampler, GraphicsDevice, MaterialDescriptor,
    MaterialInstance, MeshDescriptor, RenderGraph, RingBuffer, ShaderSource, ShaderStage,
    TextureFormat, TransferConfig, TransferOperation, TransferPass, VertexBufferLayout,
    VertexLayout,
};

use crate::gbuffer::GBuffer;
use crate::ibl_textures::IblTextures;
use crate::uniforms::ResolveUniforms;

const RESOLVE_SHADER_SLANG: &str = include_str!("../../../shaders/deferred_resolve.slang");

/// Deferred resolve/lighting pass: reads G-buffer + IBL textures, outputs lit pixels.
pub struct ResolvePass {
    pub material_instance: Arc<MaterialInstance>,
    pub mesh: Arc<redlilium_graphics::Mesh>,

    /// Per-frame uniform ring (dynamic offset per draw).
    uniform_ring: RingBuffer,
    /// This frame's offset into `uniform_ring`.
    uniform_offset: u32,

    /// Dummy vertex buffer data queued at init, flushed into the first frame's graph.
    pending_uploads: Vec<TransferOperation>,
}

impl ResolvePass {
    /// Create resolve pass material, bindings, and fullscreen triangle mesh.
    pub fn create(
        device: &Arc<GraphicsDevice>,
        gbuffer: &GBuffer,
        ibl: &IblTextures,
        surface_format: TextureFormat,
        hdr_active: bool,
    ) -> Self {
        profile_scope!("ResolvePass::create");

        // Create G-buffer sampler
        let gbuffer_sampler = device
            .create_sampler_from_cpu(&CpuSampler::nearest().with_name("gbuffer_sampler"))
            .expect("Failed to create G-buffer sampler");

        // Per-frame uniform ring (one slot written per frame).
        let uniform_ring = RingBuffer::new(
            device,
            1 << 16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "resolve_uniform_ring",
        )
        .expect("Failed to create resolve uniform ring");

        // Select the HDR variant (#6): the shader declares the axis
        // (`//#pragma variant_system HDR_OUTPUT`), the pass sets it.
        log::info!(
            "Compiling resolve shader ({} output)",
            if hdr_active { "HDR" } else { "SDR" }
        );
        let variant = redlilium_graphics::ShaderVariantSpace::parse(RESOLVE_SHADER_SLANG)
            .expect("deferred_resolve.slang variant pragmas")
            .select()
            .system("HDR_OUTPUT", hdr_active)
            .build()
            .expect("resolve variant selection");

        // Create resolve material using Slang shader. Binding 0 (uniforms, group
        // 0) is a per-draw dynamic offset.
        let resolve_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        RESOLVE_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        Vec::new(),
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        RESOLVE_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        Vec::new(),
                    ))
                    .with_variant(variant)
                    .with_color_format(surface_format)
                    .with_dynamic_uniform(0, 0)
                    .with_label("resolve_material"),
            )
            .expect("Failed to create resolve material");

        // Create binding groups
        let resolve_uniform_binding = device
            .create_binding_group(
                resolve_material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new().with_buffer_range(
                    0,
                    uniform_ring.buffer().clone(),
                    0,
                    std::mem::size_of::<ResolveUniforms>() as u64,
                ),
            )
            .expect("create resolve uniform binding group");

        let gbuffer_binding = device
            .create_binding_group(
                resolve_material.binding_layouts()[1].clone(),
                BindingGroupDescriptor::new()
                    .with_texture(0, gbuffer.albedo.clone())
                    .with_texture(1, gbuffer.normal_metallic.clone())
                    .with_texture(2, gbuffer.position_roughness.clone())
                    .with_sampler(3, gbuffer_sampler),
            )
            .expect("create gbuffer binding group");

        let ibl_binding = device
            .create_binding_group(
                resolve_material.binding_layouts()[2].clone(),
                BindingGroupDescriptor::new()
                    .with_texture(0, ibl.irradiance_cubemap.clone())
                    .with_texture(1, ibl.prefilter_cubemap.clone())
                    .with_texture(2, ibl.brdf_lut.clone())
                    .with_sampler(3, ibl.sampler.clone()),
            )
            .expect("create ibl binding group");

        let material_instance = Arc::new(
            MaterialInstance::new(resolve_material)
                .with_binding_group(resolve_uniform_binding)
                .with_binding_group(gbuffer_binding)
                .with_binding_group(ibl_binding),
        );

        // Create fullscreen triangle mesh
        let resolve_vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(4))
                .with_label("resolve_vertex_layout"),
        );

        let mesh = device
            .create_mesh(
                &MeshDescriptor::new(resolve_vertex_layout)
                    .with_vertex_count(3)
                    .with_label("resolve_triangle"),
            )
            .expect("Failed to create resolve mesh");

        // Queue minimal vertex data upload through the frame graph.
        let mut pending_uploads = Vec::new();
        if let Some(vb) = mesh.vertex_buffer(0) {
            let dummy_data: [f32; 3] = [0.0, 0.0, 0.0];
            pending_uploads.push(TransferOperation::write_buffer(
                vb.clone(),
                0,
                Arc::from(bytemuck::cast_slice(&dummy_data)),
            ));
        }

        log::info!("Resolve pass resources created");

        Self {
            material_instance,
            mesh,
            uniform_ring,
            uniform_offset: 0,
            pending_uploads,
        }
    }

    /// This frame's dynamic offset into the uniform ring (for the draw command).
    pub fn uniform_offset(&self) -> u32 {
        self.uniform_offset
    }

    /// Flush queued vertex-buffer uploads into the current frame's render graph.
    pub fn flush_uploads(&mut self, graph: &mut RenderGraph) {
        if self.pending_uploads.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.pending_uploads);
        let mut pass = TransferPass::new("resolve_vertex_upload".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(ops));
        graph.add_transfer_pass(pass);
    }

    /// Update resolve uniforms by writing this frame's slot into the uniform ring.
    pub fn update_uniforms(&mut self, camera_pos: Vec3, width: u32, height: u32) {
        let uniforms = ResolveUniforms {
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            screen_size: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
        };

        let bytes = bytemuck::bytes_of(&uniforms);
        let size = bytes.len() as u64;
        let alloc = self.uniform_ring.allocate(size).unwrap_or_else(|| {
            self.uniform_ring.reset();
            self.uniform_ring
                .allocate(size)
                .expect("resolve uniform ring too small")
        });
        self.uniform_ring
            .write(&alloc, bytes)
            .expect("Failed to write resolve uniform ring");
        self.uniform_offset = alloc.offset as u32;
    }
}
