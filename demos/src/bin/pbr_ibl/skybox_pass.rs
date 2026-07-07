//! Skybox rendering resources.

use std::sync::Arc;

use redlilium_core::profiling::profile_scope;
use redlilium_graphics::{
    BindingGroupDescriptor, BufferUsage, GraphicsDevice, MaterialDescriptor, MaterialInstance,
    MeshDescriptor, RenderGraph, RingBuffer, ShaderSource, ShaderStage, TextureFormat,
    TransferConfig, TransferOperation, TransferPass, VertexBufferLayout, VertexLayout,
};

use redlilium_core::math::{Mat4, Vec3, mat4_to_cols_array_2d};

use crate::ibl_textures::IblTextures;
use crate::uniforms::SkyboxUniforms;

const SKYBOX_SHADER_SLANG: &str = include_str!("../../../shaders/skybox.slang");

/// Skybox rendering resources: material, fullscreen triangle mesh, and uniforms.
pub struct SkyboxPass {
    pub material_instance: Arc<MaterialInstance>,
    pub mesh: Arc<redlilium_graphics::Mesh>,
    pub mip_level: f32,

    /// Per-frame uniform ring (dynamic offset per draw).
    uniform_ring: RingBuffer,
    /// This frame's offset into `uniform_ring`.
    uniform_offset: u32,

    /// Dummy vertex buffer data queued at init, flushed into the first frame's graph.
    pending_uploads: Vec<TransferOperation>,
}

impl SkyboxPass {
    /// Create skybox material, fullscreen triangle mesh, and uniform ring.
    pub fn create(
        device: &Arc<GraphicsDevice>,
        ibl: &IblTextures,
        surface_format: TextureFormat,
    ) -> Self {
        profile_scope!("SkyboxPass::create");

        // Create skybox material using Slang shader (no vertex layout needed for
        // fullscreen triangle). Binding 0 (uniforms) is a per-draw dynamic offset.
        let skybox_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        SKYBOX_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        SKYBOX_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_color_format(surface_format)
                    .with_dynamic_uniform(0, 0)
                    .with_label("skybox_material"),
            )
            .expect("Failed to create skybox material");

        // Per-frame uniform ring (one slot written per frame).
        let uniform_ring = RingBuffer::new(
            device,
            1 << 16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "skybox_uniform_ring",
        )
        .expect("Failed to create skybox uniform ring");

        // Create material instance binding the ring by range.
        let skybox_binding_group = device
            .create_binding_group(
                skybox_material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        uniform_ring.buffer().clone(),
                        0,
                        std::mem::size_of::<SkyboxUniforms>() as u64,
                    )
                    .with_texture(1, ibl.prefilter_cubemap.clone())
                    .with_sampler(2, ibl.sampler.clone()),
            )
            .expect("create skybox binding group");

        let material_instance = Arc::new(
            MaterialInstance::new(skybox_material).with_binding_group(skybox_binding_group),
        );

        // Create a minimal mesh for fullscreen triangle (shader uses vertex_index only)
        let skybox_vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(4)) // Minimal stride, no attributes
                .with_label("skybox_vertex_layout"),
        );

        let mesh = device
            .create_mesh(
                &MeshDescriptor::new(skybox_vertex_layout)
                    .with_vertex_count(3)
                    .with_label("skybox_triangle"),
            )
            .expect("Failed to create skybox mesh");

        // Queue minimal vertex data upload through the frame graph (shader doesn't
        // use it, just needs a valid initialized buffer).
        let mut pending_uploads = Vec::new();
        if let Some(vb) = mesh.vertex_buffer(0) {
            let dummy_data: [f32; 3] = [0.0, 0.0, 0.0];
            pending_uploads.push(TransferOperation::write_buffer(
                vb.clone(),
                0,
                Arc::from(bytemuck::cast_slice(&dummy_data)),
            ));
        }

        log::info!("Skybox resources created");

        Self {
            material_instance,
            mesh,
            mip_level: 0.0,
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
        let mut pass = TransferPass::new("skybox_vertex_upload".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(ops));
        graph.add_transfer_pass(pass);
    }

    /// Update skybox uniforms by writing this frame's slot into the uniform ring.
    pub fn update_uniforms(&mut self, view: Mat4, proj: Mat4, camera_pos: Vec3) {
        profile_scope!("SkyboxPass::update_uniforms");
        let view_proj = proj * view;
        let inv_view_proj = view_proj.try_inverse().unwrap();

        let uniforms = SkyboxUniforms {
            inv_view_proj: mat4_to_cols_array_2d(&inv_view_proj),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            mip_level: self.mip_level,
            _pad0: [0.0; 3],
            _pad1: [0.0; 4],
        };

        let bytes = bytemuck::bytes_of(&uniforms);
        let size = bytes.len() as u64;
        let alloc = self.uniform_ring.allocate(size).unwrap_or_else(|| {
            self.uniform_ring.reset();
            self.uniform_ring
                .allocate(size)
                .expect("skybox uniform ring too small")
        });
        self.uniform_ring
            .write(&alloc, bytes)
            .expect("Failed to write skybox uniform ring");
        self.uniform_offset = alloc.offset as u32;
    }
}
