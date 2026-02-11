//! Deferred resolve/lighting pass resources.

use std::sync::Arc;

use redlilium_core::math::Vec3;
use redlilium_core::profiling::profile_scope;
use redlilium_graphics::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BufferDescriptor, BufferUsage,
    CpuSampler, GraphicsDevice, MaterialDescriptor, MaterialInstance, MeshDescriptor,
    ShaderComposer, ShaderDef, ShaderSource, ShaderStage, ShaderStageFlags, TextureFormat,
    VertexBufferLayout, VertexLayout,
};

use crate::gbuffer::GBuffer;
use crate::ibl_textures::IblTextures;
use crate::uniforms::ResolveUniforms;

const RESOLVE_SHADER_WGSL: &str = include_str!("../../../shaders/deferred_resolve.wgsl");

/// Deferred resolve/lighting pass: reads G-buffer + IBL textures, outputs lit pixels.
pub struct ResolvePass {
    pub material_instance: Arc<MaterialInstance>,
    pub mesh: Arc<redlilium_graphics::Mesh>,
    pub uniform_buffer: Arc<redlilium_graphics::Buffer>,
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

        // Create resolve uniform buffer
        let uniform_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<ResolveUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create resolve uniform buffer");

        // Binding layouts
        let resolve_uniform_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
                )
                .with_label("resolve_uniform_bindings"),
        );

        let gbuffer_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(2, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(3, BindingType::Sampler)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_label("gbuffer_bindings"),
        );

        let ibl_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::TextureCube)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::TextureCube)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(2, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(3, BindingType::Sampler)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_label("ibl_bindings"),
        );

        // Compose resolve shader with HDR define if active
        let mut shader_composer =
            ShaderComposer::with_standard_library().expect("Failed to create shader composer");

        let shader_defs: Vec<(&str, ShaderDef)> = if hdr_active {
            log::info!("Compiling resolve shader with HDR_OUTPUT define");
            vec![("HDR_OUTPUT", ShaderDef::Bool(true))]
        } else {
            log::info!("Compiling resolve shader for SDR output");
            vec![]
        };

        let composed_resolve_shader = shader_composer
            .compose(RESOLVE_SHADER_WGSL, &shader_defs)
            .expect("Failed to compose resolve shader");
        log::info!("Resolve shader composed with library imports");

        // Create resolve material
        let resolve_material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        composed_resolve_shader.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        composed_resolve_shader.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(resolve_uniform_layout)
                    .with_binding_layout(gbuffer_layout)
                    .with_binding_layout(ibl_layout)
                    .with_color_format(surface_format)
                    .with_label("resolve_material"),
            )
            .expect("Failed to create resolve material");

        // Create binding groups
        #[allow(clippy::arc_with_non_send_sync)]
        let resolve_uniform_binding =
            Arc::new(BindingGroup::new().with_buffer(0, uniform_buffer.clone()));

        #[allow(clippy::arc_with_non_send_sync)]
        let gbuffer_binding = Arc::new(
            BindingGroup::new()
                .with_texture(0, gbuffer.albedo.clone())
                .with_texture(1, gbuffer.normal_metallic.clone())
                .with_texture(2, gbuffer.position_roughness.clone())
                .with_sampler(3, gbuffer_sampler),
        );

        #[allow(clippy::arc_with_non_send_sync)]
        let ibl_binding = Arc::new(
            BindingGroup::new()
                .with_texture(0, ibl.irradiance_cubemap.clone())
                .with_texture(1, ibl.prefilter_cubemap.clone())
                .with_texture(2, ibl.brdf_lut.clone())
                .with_sampler(3, ibl.sampler.clone()),
        );

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

        if let Some(vb) = mesh.vertex_buffer(0) {
            let dummy_data: [f32; 3] = [0.0, 0.0, 0.0];
            device
                .write_buffer(vb, 0, bytemuck::cast_slice(&dummy_data))
                .expect("Failed to write resolve mesh vertex buffer");
        }

        log::info!("Resolve pass resources created");

        Self {
            material_instance,
            mesh,
            uniform_buffer,
        }
    }

    /// Update resolve uniform buffer.
    pub fn update_uniforms(
        &self,
        device: &Arc<GraphicsDevice>,
        camera_pos: Vec3,
        width: u32,
        height: u32,
    ) {
        let uniforms = ResolveUniforms {
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            screen_size: [
                width as f32,
                height as f32,
                1.0 / width as f32,
                1.0 / height as f32,
            ],
        };

        device
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms))
            .expect("Failed to write resolve uniform buffer");
    }
}
