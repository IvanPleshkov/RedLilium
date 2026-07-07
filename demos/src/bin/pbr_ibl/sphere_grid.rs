//! Sphere grid instances and G-buffer material for PBR rendering.

use std::sync::Arc;

use redlilium_core::math::{Mat4, Vec3, mat4_to_cols_array_2d};
use redlilium_core::mesh::generators;
use redlilium_core::profiling::{profile_function, profile_scope};
use redlilium_graphics::{
    BindingGroupDescriptor, BufferUsage, GraphicsDevice, Material, MaterialDescriptor,
    MaterialInstance, PolygonMode, RenderGraph, RingBuffer, ShaderSource, ShaderStage,
    TextureFormat, TransferConfig, TransferOperation, TransferPass,
};

use crate::uniforms::{CameraUniforms, SPHERE_INSTANCE_STRIDE, SphereInstance};

const GBUFFER_SHADER_SLANG: &str = include_str!("../../../shaders/deferred_gbuffer.slang");

/// Sphere grid with G-buffer material, instanced mesh, and camera/instance rings.
///
/// # Frame-invariant binding groups (issue #40)
///
/// Both material instances' binding groups are created **once** in [`create`]
/// and reused every frame — the demo issues zero `create_binding_group` calls
/// in steady state. The camera set (binding 0) is a dynamic uniform bound at
/// offset 0 and selected per draw via [`camera_offset`](Self::camera_offset).
/// The instance storage set (binding 1) binds the **whole** instance ring at
/// offset 0; each frame's block is addressed by an element index
/// ([`CameraUniforms::instance_base`]) the vertex shader adds to
/// `SV_InstanceID`, so the group never has to be rebuilt for a new ring slot.
pub struct SphereGrid {
    pub material_instance: Arc<MaterialInstance>,
    pub wireframe_material_instance: Arc<MaterialInstance>,
    pub mesh: Arc<redlilium_graphics::Mesh>,

    /// Per-frame camera uniform ring (one slot written per frame). Bound by
    /// range and additionally addressed via a per-draw dynamic offset.
    camera_ring: RingBuffer,
    /// This frame's offset into `camera_ring`.
    camera_offset: u32,
    /// Per-frame instance storage ring (one block written per frame). Bound in
    /// full at offset 0; the current block is selected by `instance_base`.
    instance_ring: RingBuffer,
    /// Number of instances written this frame.
    instance_count: u32,
    /// Element index of this frame's instance block within `instance_ring`
    /// (`alloc.offset / SPHERE_INSTANCE_STRIDE`), folded into the camera uniform.
    instance_base: u32,

    /// Mesh data queued at init, flushed into the first frame's graph.
    pending_uploads: Vec<TransferOperation>,
}

impl SphereGrid {
    /// Create the sphere grid: G-buffer material, sphere mesh, camera/instance rings.
    pub fn create(device: &Arc<GraphicsDevice>) -> Self {
        profile_function!();

        // Generate sphere mesh on CPU
        let sphere_cpu = generators::generate_sphere(0.5, 32, 16);
        let vertex_layout = sphere_cpu.layout().clone();

        // Build base G-buffer material descriptor (shared between fill and wireframe).
        // Binding 0 (camera uniforms) is a per-draw dynamic offset into a ring.
        let base_descriptor = MaterialDescriptor::new()
            .with_shader(ShaderSource::slang(
                ShaderStage::Vertex,
                GBUFFER_SHADER_SLANG.as_bytes().to_vec(),
                "vs_main",
                vec![],
            ))
            .with_shader(ShaderSource::slang(
                ShaderStage::Fragment,
                GBUFFER_SHADER_SLANG.as_bytes().to_vec(),
                "fs_main",
                vec![],
            ))
            .with_vertex_layout(vertex_layout)
            .with_color_format(TextureFormat::Rgba8UnormSrgb)
            .with_color_format(TextureFormat::Rgba16Float)
            .with_color_format(TextureFormat::Rgba16Float)
            .with_depth_format(TextureFormat::Depth32Float)
            .with_dynamic_uniform(0, 0);

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

        // Camera uniform ring + instance storage ring — one slot written per
        // frame so writes never race the GPU reading a previous frame in flight.
        let camera_ring = RingBuffer::new(
            device,
            1 << 16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "sphere_camera_ring",
        )
        .expect("Failed to create camera ring");
        let instance_ring = RingBuffer::new(
            device,
            1 << 20,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
            "sphere_instance_ring",
        )
        .expect("Failed to create instance ring");

        // Frame-invariant material instances: the binding groups are created
        // once here (camera dynamic uniform at offset 0, whole instance ring at
        // offset 0) and reused every frame — see the type-level doc.
        let material_instance =
            Self::make_instance(device, &material, &camera_ring, &instance_ring);
        let wireframe_material_instance =
            Self::make_instance(device, &wireframe_material, &camera_ring, &instance_ring);

        // Create GPU mesh; its data uploads through the frame graph.
        let (mesh, mesh_ops) = device
            .create_mesh_deferred(&sphere_cpu)
            .expect("Failed to create sphere mesh");

        Self {
            material_instance,
            wireframe_material_instance,
            mesh,
            camera_ring,
            camera_offset: 0,
            instance_ring,
            instance_count: 0,
            instance_base: 0,
            pending_uploads: mesh_ops,
        }
    }

    /// Build a frame-invariant material instance: binding 0 is the camera ring
    /// bound at offset 0 (a dynamic uniform selected per draw via
    /// [`camera_offset`](Self::camera_offset)); binding 1 is the **whole**
    /// instance ring bound at offset 0 (this frame's block is selected in-shader
    /// by [`CameraUniforms::instance_base`]). Neither depends on the frame, so
    /// the group is created once and never rebuilt.
    fn make_instance(
        device: &Arc<GraphicsDevice>,
        material: &Arc<Material>,
        camera_ring: &RingBuffer,
        instance_ring: &RingBuffer,
    ) -> Arc<MaterialInstance> {
        let camera_size = std::mem::size_of::<CameraUniforms>() as u64;
        let binding_group = device
            .create_binding_group(
                material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new()
                    .with_buffer_range(0, camera_ring.buffer().clone(), 0, camera_size)
                    .with_buffer_range(
                        1,
                        instance_ring.buffer().clone(),
                        0,
                        instance_ring.capacity(),
                    ),
            )
            .expect("create sphere binding group");
        Arc::new(MaterialInstance::new(material.clone()).with_binding_group(binding_group))
    }

    /// This frame's dynamic offset into the camera ring (for the draw command).
    pub fn camera_offset(&self) -> u32 {
        self.camera_offset
    }

    /// Number of instances written for this frame.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Flush queued mesh uploads into the current frame's render graph.
    pub fn flush_uploads(&mut self, graph: &mut RenderGraph) {
        if self.pending_uploads.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.pending_uploads);
        let mut pass = TransferPass::new("sphere_mesh_uploads".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(ops));
        graph.add_transfer_pass(pass);
    }

    /// Write sphere instance data into the instance ring and record this frame's
    /// block base index. The binding groups are frame-invariant (bound to the
    /// whole ring), so nothing is rebuilt here.
    ///
    /// Call this **before** [`write_camera_uniforms`](Self::write_camera_uniforms):
    /// the base index it records is folded into the camera uniform.
    pub fn write_instances(&mut self, instances: &[SphereInstance]) {
        profile_scope!("SphereGrid::write_instances");

        let count = instances.len();
        let bytes: &[u8] = bytemuck::cast_slice(instances);
        let size = bytes.len().max(1) as u64;
        let alloc = self.instance_ring.allocate(size).unwrap_or_else(|| {
            self.instance_ring.reset();
            self.instance_ring
                .allocate(size)
                .expect("instance ring too small")
        });
        if !bytes.is_empty() {
            self.instance_ring
                .write(&alloc, bytes)
                .expect("Failed to write instance ring");
        }
        self.instance_count = count as u32;
        // The ring's 256-byte alignment is a multiple of the 128-byte stride, so
        // the offset divides exactly into an element index.
        debug_assert_eq!(alloc.offset % SPHERE_INSTANCE_STRIDE, 0);
        self.instance_base = (alloc.offset / SPHERE_INSTANCE_STRIDE) as u32;
    }

    /// Update camera uniforms by writing this frame's slot into the camera ring.
    ///
    /// Folds in the instance base index recorded by the preceding
    /// [`write_instances`](Self::write_instances) call.
    pub fn write_camera_uniforms(&mut self, view: Mat4, proj: Mat4, camera_pos: Vec3) {
        profile_scope!("SphereGrid::write_camera_uniforms");
        let view_proj = proj * view;

        let uniforms = CameraUniforms {
            view_proj: mat4_to_cols_array_2d(&view_proj),
            view: mat4_to_cols_array_2d(&view),
            proj: mat4_to_cols_array_2d(&proj),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            instance_base: self.instance_base,
            _pad: [0; 3],
        };

        let bytes = bytemuck::bytes_of(&uniforms);
        let size = bytes.len() as u64;
        let alloc = self.camera_ring.allocate(size).unwrap_or_else(|| {
            self.camera_ring.reset();
            self.camera_ring
                .allocate(size)
                .expect("camera ring too small")
        });
        self.camera_ring
            .write(&alloc, bytes)
            .expect("Failed to write camera ring");
        self.camera_offset = alloc.offset as u32;
    }
}
