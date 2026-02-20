//! Sphere grid instances and G-buffer material for PBR rendering.

use std::sync::Arc;

use redlilium_core::math::{Mat4, Vec3, mat4_from_translation, mat4_to_cols_array_2d};
use redlilium_core::mesh::generators;
use redlilium_core::profiling::{profile_function, profile_scope};
use redlilium_graphics::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, BufferDescriptor, BufferUsage,
    GraphicsDevice, MaterialDescriptor, MaterialInstance, PolygonMode, ShaderComposer,
    ShaderSource, ShaderStage, ShaderStageFlags, TextureFormat,
};

use crate::GRID_SIZE;
use crate::uniforms::{CameraUniforms, SphereInstance};

const GBUFFER_SHADER_GLSL: &str = include_str!("../../../shaders/deferred_gbuffer.glsl");

/// Sphere grid with G-buffer material, instanced mesh, and camera/instance buffers.
pub struct SphereGrid {
    pub material_instance: Arc<MaterialInstance>,
    pub wireframe_material_instance: Arc<MaterialInstance>,
    pub mesh: Arc<redlilium_graphics::Mesh>,
    pub camera_buffer: Arc<redlilium_graphics::Buffer>,
    pub instance_buffer: Arc<redlilium_graphics::Buffer>,
}

impl SphereGrid {
    /// Create the sphere grid: G-buffer material, sphere mesh, camera/instance buffers.
    pub fn create(device: &Arc<GraphicsDevice>) -> Self {
        profile_function!();

        // Generate sphere mesh on CPU
        let sphere_cpu = generators::generate_sphere(0.5, 32, 16);
        let vertex_layout = sphere_cpu.layout().clone();

        // Create camera binding layout
        let camera_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::StorageBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX),
                )
                .with_label("camera_bindings"),
        );

        // Compose G-buffer shader (separate compilation per stage)
        let shader_composer = ShaderComposer::with_standard_library();
        let composed_vs = shader_composer
            .compose(GBUFFER_SHADER_GLSL, ShaderStage::Vertex, &[])
            .expect("Failed to compose G-buffer vertex shader");
        let composed_fs = shader_composer
            .compose(GBUFFER_SHADER_GLSL, ShaderStage::Fragment, &[])
            .expect("Failed to compose G-buffer fragment shader");
        log::info!("G-buffer shader composed with library imports");

        // Build base G-buffer material descriptor (shared between fill and wireframe)
        let base_descriptor = MaterialDescriptor::new()
            .with_shader(ShaderSource::new(
                ShaderStage::Vertex,
                composed_vs.as_bytes().to_vec(),
                "main",
            ))
            .with_shader(ShaderSource::new(
                ShaderStage::Fragment,
                composed_fs.as_bytes().to_vec(),
                "main",
            ))
            .with_binding_layout(camera_binding_layout)
            .with_vertex_layout(vertex_layout)
            .with_color_format(TextureFormat::Rgba8UnormSrgb)
            .with_color_format(TextureFormat::Rgba16Float)
            .with_color_format(TextureFormat::Rgba16Float)
            .with_depth_format(TextureFormat::Depth32Float);

        // Create fill material
        let material = device
            .create_material(&base_descriptor.clone().with_label("gbuffer_material"))
            .expect("Failed to create G-buffer material");

        // Create wireframe material
        let wireframe_material = device
            .create_material(
                &base_descriptor
                    .with_polygon_mode(PolygonMode::Line)
                    .with_label("gbuffer_wireframe_material"),
            )
            .expect("Failed to create wireframe G-buffer material");

        // Create camera uniform buffer
        let camera_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<CameraUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create camera buffer");

        // Create instance buffer with default instances
        let instances = Self::create_instances_default();
        let instance_data = bytemuck::cast_slice(&instances);
        let instance_buffer = device
            .create_buffer(&BufferDescriptor::new(
                instance_data.len() as u64,
                BufferUsage::STORAGE | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create instance buffer");
        device
            .write_buffer(&instance_buffer, 0, instance_data)
            .expect("Failed to write instance buffer");

        // Create material instance with camera bindings
        #[allow(clippy::arc_with_non_send_sync)]
        let camera_binding_group = Arc::new(
            BindingGroup::new()
                .with_buffer(0, camera_buffer.clone())
                .with_buffer(1, instance_buffer.clone()),
        );

        let material_instance = Arc::new(
            MaterialInstance::new(material).with_binding_group(camera_binding_group.clone()),
        );
        let wireframe_material_instance = Arc::new(
            MaterialInstance::new(wireframe_material).with_binding_group(camera_binding_group),
        );

        // Create GPU mesh
        let mesh = device
            .create_mesh_from_cpu(&sphere_cpu)
            .expect("Failed to create sphere mesh");

        Self {
            material_instance,
            wireframe_material_instance,
            mesh,
            camera_buffer,
            instance_buffer,
        }
    }

    /// Write sphere instance data to the GPU buffer.
    pub fn write_instances(&self, device: &Arc<GraphicsDevice>, instances: &[SphereInstance]) {
        profile_scope!("SphereGrid::write_instances");
        let instance_data = bytemuck::cast_slice(instances);
        device
            .write_buffer(&self.instance_buffer, 0, instance_data)
            .expect("Failed to write instance buffer");
    }

    /// Update camera uniform buffer from pre-computed matrices.
    pub fn write_camera_uniforms(
        &self,
        device: &Arc<GraphicsDevice>,
        view: Mat4,
        proj: Mat4,
        camera_pos: Vec3,
    ) {
        profile_scope!("SphereGrid::write_camera_uniforms");
        let view_proj = proj * view;

        let uniforms = CameraUniforms {
            view_proj: mat4_to_cols_array_2d(&view_proj),
            view: mat4_to_cols_array_2d(&view),
            proj: mat4_to_cols_array_2d(&proj),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
        };

        device
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniforms))
            .expect("Failed to write camera uniform buffer");
    }

    fn create_instances_default() -> Vec<SphereInstance> {
        Self::build_instances_default([0.9, 0.1, 0.1, 1.0])
    }

    fn build_instances_default(base_color: [f32; 4]) -> Vec<SphereInstance> {
        let spacing = crate::SPHERE_SPACING;
        let mut instances = Vec::with_capacity(GRID_SIZE * GRID_SIZE);
        let offset = (GRID_SIZE as f32 - 1.0) * spacing / 2.0;

        for row in 0..GRID_SIZE {
            for col in 0..GRID_SIZE {
                let x = col as f32 * spacing - offset;
                let z = row as f32 * spacing - offset;

                let model = mat4_from_translation(Vec3::new(x, 0.0, z));
                let metallic = col as f32 / (GRID_SIZE - 1) as f32;
                let roughness = (row as f32 / (GRID_SIZE - 1) as f32).max(0.05);

                instances.push(SphereInstance {
                    model: mat4_to_cols_array_2d(&model),
                    base_color,
                    metallic_roughness: [metallic, roughness, 0.0, 0.0],
                });
            }
        }

        instances
    }
}
