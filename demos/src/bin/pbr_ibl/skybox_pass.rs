//! Skybox rendering resources.

use std::sync::Arc;

use redlilium_core::profiling::profile_scope;
use redlilium_graphics::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BufferDescriptor, BufferUsage,
    GraphicsDevice, MaterialDescriptor, MaterialInstance, MeshDescriptor, ShaderComposer,
    ShaderSource, ShaderStage, ShaderStageFlags, TextureFormat, VertexBufferLayout, VertexLayout,
};

use crate::camera::OrbitCamera;
use crate::ibl_textures::IblTextures;
use crate::uniforms::SkyboxUniforms;

const SKYBOX_SHADER_WGSL: &str = include_str!("../../../shaders/skybox.wgsl");

/// Skybox rendering resources: material, fullscreen triangle mesh, and uniforms.
pub struct SkyboxPass {
    pub material_instance: Arc<MaterialInstance>,
    pub mesh: Arc<redlilium_graphics::Mesh>,
    pub uniform_buffer: Arc<redlilium_graphics::Buffer>,
    pub mip_level: f32,
}

impl SkyboxPass {
    /// Create skybox material, fullscreen triangle mesh, and uniform buffer.
    pub fn create(
        device: &Arc<GraphicsDevice>,
        ibl: &IblTextures,
        surface_format: TextureFormat,
    ) -> Self {
        profile_scope!("SkyboxPass::create");

        // Create skybox binding layout
        let skybox_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::TextureCube)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(2, BindingType::Sampler)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_label("skybox_bindings"),
        );

        // Compose skybox shader
        let mut shader_composer =
            ShaderComposer::with_standard_library().expect("Failed to create shader composer");
        let composed_skybox_shader = shader_composer
            .compose(SKYBOX_SHADER_WGSL, &[])
            .expect("Failed to compose skybox shader");
        log::info!("Skybox shader composed with library imports");

        // Create skybox material (no vertex layout needed for fullscreen triangle)
        let skybox_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        composed_skybox_shader.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        composed_skybox_shader.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(skybox_binding_layout)
                    .with_color_format(surface_format)
                    .with_label("skybox_material"),
            )
            .expect("Failed to create skybox material");

        // Create skybox uniform buffer
        let uniform_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<SkyboxUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create skybox uniform buffer");

        // Create material instance
        #[allow(clippy::arc_with_non_send_sync)]
        let skybox_binding_group = Arc::new(
            BindingGroup::new()
                .with_buffer(0, uniform_buffer.clone())
                .with_texture(1, ibl.prefilter_cubemap.clone())
                .with_sampler(2, ibl.sampler.clone()),
        );

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

        // Write minimal vertex data (shader doesn't use it, just needs valid buffer)
        if let Some(vb) = mesh.vertex_buffer(0) {
            let dummy_data: [f32; 3] = [0.0, 0.0, 0.0];
            device
                .write_buffer(vb, 0, bytemuck::cast_slice(&dummy_data))
                .expect("Failed to write skybox vertex buffer");
        }

        log::info!("Skybox resources created");

        Self {
            material_instance,
            mesh,
            uniform_buffer,
            mip_level: 0.0,
        }
    }

    /// Update skybox uniform buffer with current camera state.
    pub fn update_uniforms(
        &self,
        device: &Arc<GraphicsDevice>,
        camera: &OrbitCamera,
        aspect_ratio: f32,
    ) {
        profile_scope!("SkyboxPass::update_uniforms");
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect_ratio);
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        let uniforms = SkyboxUniforms {
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            camera_pos: camera.position().extend(1.0).to_array(),
            mip_level: self.mip_level,
            _pad0: [0.0; 3],
            _pad1: [0.0; 4],
        };

        device
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms))
            .expect("Failed to write skybox uniform buffer");
    }
}
