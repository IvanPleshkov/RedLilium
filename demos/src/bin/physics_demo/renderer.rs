//! Physics shape renderer.
//!
//! Renders rapier collider shapes with basic directional lighting.
//! Each shape type (sphere, box) uses instanced drawing with a storage buffer
//! holding per-instance model matrices and colors.

use std::sync::Arc;

use redlilium_core::math::{self, Mat4, Vec3, mat4_to_cols_array_2d};
use redlilium_ecs::physics::physics2d::PhysicsWorld2D;
use redlilium_ecs::physics::physics3d::PhysicsWorld3D;
use redlilium_graphics::{
    BindingGroupDescriptor, BindingLayout, BindingLayoutEntry, BindingType, BufferUsage,
    DrawCommand, GraphicsDevice, GraphicsPass, Material, MaterialDescriptor, MaterialInstance,
    Mesh, RenderGraph, RingBuffer, ShaderSource, ShaderStage, ShaderStageFlags, Texture,
    TextureDescriptor, TextureFormat, TextureUsage, TransferConfig, TransferOperation,
    TransferPass,
};

const MAX_INSTANCES: usize = 4096;
const INSTANCE_STRIDE: usize = std::mem::size_of::<ShapeInstance>();

// ---------------------------------------------------------------------------
// WGSL shader
// ---------------------------------------------------------------------------

const SHAPE_SHADER: &str = r#"
struct CameraUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    // Element index of this draw's first instance within the whole-buffer-bound
    // instance ring; added to instance_index so the binding group stays
    // frame-invariant. Selected per draw via the camera's dynamic offset.
    instance_base: u32,
}

struct ShapeInstance {
    model_0: vec4<f32>,
    model_1: vec4<f32>,
    model_2: vec4<f32>,
    model_3: vec4<f32>,
    color: vec4<f32>,
    // Padding to a 128-byte stride so the CPU can address a batch's block by an
    // exact element index (CameraUniforms.instance_base).
    _pad0: vec4<f32>,
    _pad1: vec4<f32>,
    _pad2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> instances: array<ShapeInstance>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) iid: u32) -> VertexOutput {
    let inst = instances[camera.instance_base + iid];
    let model = mat4x4<f32>(inst.model_0, inst.model_1, inst.model_2, inst.model_3);
    let world_pos = model * vec4(in.position, 1.0);
    let normal_mat = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = normalize(normal_mat * in.normal);
    out.world_pos = world_pos.xyz;
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(camera.light_dir.xyz);
    let n = normalize(in.world_normal);

    let ambient = 0.15;
    let diffuse = max(dot(n, light_dir), 0.0) * 0.7;

    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);
    let half_dir = normalize(light_dir + view_dir);
    let spec = pow(max(dot(n, half_dir), 0.0), 32.0) * 0.3;

    let lighting = ambient + diffuse + spec;
    return vec4(in.color.rgb * lighting, in.color.a);
}
"#;

// ---------------------------------------------------------------------------
// GPU data types
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    /// Element index of this draw's first instance within the instance ring;
    /// added to `instance_index` in the vertex shader.
    instance_base: u32,
    _pad: [u32; 3],
}

/// Per-instance data. Padded to 128 bytes (a divisor of the ring's 256-byte
/// allocation alignment) so each batch's block starts at an offset that is an
/// exact multiple of the stride, making `offset / 128` a valid element index
/// (passed to the shader as [`CameraUniforms::instance_base`]).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeInstance {
    model: [[f32; 4]; 4],
    color: [f32; 4],
    _pad: [[f32; 4]; 3],
}

// ---------------------------------------------------------------------------
// Color palette for dynamic bodies
// ---------------------------------------------------------------------------

const DYNAMIC_COLORS: [[f32; 4]; 8] = [
    [0.85, 0.35, 0.30, 1.0],
    [0.35, 0.75, 0.35, 1.0],
    [0.35, 0.40, 0.85, 1.0],
    [0.85, 0.80, 0.30, 1.0],
    [0.80, 0.35, 0.80, 1.0],
    [0.30, 0.80, 0.80, 1.0],
    [0.90, 0.55, 0.25, 1.0],
    [0.60, 0.35, 0.75, 1.0],
];

const FIXED_COLOR: [f32; 4] = [0.35, 0.35, 0.40, 1.0];
const KINEMATIC_COLOR: [f32; 4] = [0.30, 0.55, 0.85, 1.0];

// ---------------------------------------------------------------------------
// Box mesh generation (unit cube, half-extent 0.5, position+normal+uv layout)
// ---------------------------------------------------------------------------

fn generate_box_cpu() -> redlilium_graphics::CpuMesh {
    use redlilium_graphics::VertexLayout;

    let h: f32 = 0.5;

    // 24 vertices (4 per face), each: [px, py, pz, nx, ny, nz, u, v]
    #[rustfmt::skip]
    let verts: Vec<[f32; 8]> = vec![
        // +X
        [h,-h,-h, 1.0,0.0,0.0, 0.0,1.0], [h, h,-h, 1.0,0.0,0.0, 0.0,0.0],
        [h, h, h, 1.0,0.0,0.0, 1.0,0.0], [h,-h, h, 1.0,0.0,0.0, 1.0,1.0],
        // -X
        [-h,-h, h, -1.0,0.0,0.0, 0.0,1.0], [-h, h, h, -1.0,0.0,0.0, 0.0,0.0],
        [-h, h,-h, -1.0,0.0,0.0, 1.0,0.0], [-h,-h,-h, -1.0,0.0,0.0, 1.0,1.0],
        // +Y
        [-h, h,-h, 0.0,1.0,0.0, 0.0,1.0], [-h, h, h, 0.0,1.0,0.0, 0.0,0.0],
        [ h, h, h, 0.0,1.0,0.0, 1.0,0.0], [ h, h,-h, 0.0,1.0,0.0, 1.0,1.0],
        // -Y
        [-h,-h, h, 0.0,-1.0,0.0, 0.0,1.0], [-h,-h,-h, 0.0,-1.0,0.0, 0.0,0.0],
        [ h,-h,-h, 0.0,-1.0,0.0, 1.0,0.0], [ h,-h, h, 0.0,-1.0,0.0, 1.0,1.0],
        // +Z
        [-h,-h, h, 0.0,0.0,1.0, 0.0,1.0], [ h,-h, h, 0.0,0.0,1.0, 1.0,1.0],
        [ h, h, h, 0.0,0.0,1.0, 1.0,0.0], [-h, h, h, 0.0,0.0,1.0, 0.0,0.0],
        // -Z
        [ h,-h,-h, 0.0,0.0,-1.0, 0.0,1.0], [-h,-h,-h, 0.0,0.0,-1.0, 1.0,1.0],
        [-h, h,-h, 0.0,0.0,-1.0, 1.0,0.0], [ h, h,-h, 0.0,0.0,-1.0, 0.0,0.0],
    ];

    let vertex_data: Vec<u8> = verts
        .iter()
        .flat_map(|v| bytemuck::bytes_of(v).to_vec())
        .collect();

    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
         0, 1, 2,  0, 2, 3,   // +X
         4, 5, 6,  4, 6, 7,   // -X
         8, 9,10,  8,10,11,   // +Y
        12,13,14, 12,14,15,   // -Y
        16,17,18, 16,18,19,   // +Z
        20,21,22, 20,22,23,   // -Z
    ];

    redlilium_graphics::CpuMesh::new(VertexLayout::position_normal_uv())
        .with_vertex_data(0, vertex_data)
        .with_indices_u32(&indices)
        .with_label("unit_box")
}

// ---------------------------------------------------------------------------
// Shape batch — one per shape type (sphere, box)
// ---------------------------------------------------------------------------

struct ShapeBatch {
    mesh: Arc<Mesh>,
    count: u32,
    /// This frame's camera dynamic offset for this batch. The camera slot it
    /// points at carries the batch's `instance_base`, so the shared (frame-
    /// invariant) binding group needs no per-batch rebuild.
    camera_offset: u32,
}

// ---------------------------------------------------------------------------
// PhysicsRenderer
// ---------------------------------------------------------------------------

/// # Frame-invariant binding group (issue #40)
///
/// A single [`MaterialInstance`] with one binding group is created in [`new`]
/// and shared by both shape batches, so the demo issues zero
/// `create_binding_group` calls per frame. Binding 0 is the camera ring bound
/// as a dynamic uniform at offset 0; binding 1 is the **whole** instance ring
/// bound at offset 0. Each frame the two batches write their instances into
/// distinct ring blocks and get their own camera slot (carrying that block's
/// `instance_base`); the per-draw camera dynamic offset selects the slot, and
/// the vertex shader adds `instance_base` to `instance_index`.
pub struct PhysicsRenderer {
    depth_texture: Arc<Texture>,
    /// Per-frame camera uniform ring (one slot written per batch per frame).
    camera_ring: RingBuffer,
    /// Per-frame instance storage ring (bound in full; batches write distinct
    /// blocks selected in-shader by `instance_base`).
    instance_ring: RingBuffer,
    /// Shared, frame-invariant material instance (its binding group is created
    /// once and reused by every draw).
    material_instance: Arc<MaterialInstance>,
    sphere_batch: ShapeBatch,
    box_batch: ShapeBatch,
    /// Mesh uploads queued at init, flushed into the first frame's graph.
    pending_uploads: Vec<TransferOperation>,
}

impl PhysicsRenderer {
    /// Create the renderer with all GPU resources.
    pub fn new(
        device: &Arc<GraphicsDevice>,
        width: u32,
        height: u32,
        surface_format: TextureFormat,
    ) -> Self {
        // Depth texture
        let depth_texture = device
            .create_texture(&TextureDescriptor::new_2d(
                width.max(1),
                height.max(1),
                TextureFormat::Depth32Float,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .expect("depth texture");

        // Camera uniform ring + instance storage ring — one slot written per
        // frame so writes never race the GPU reading a previous frame in flight.
        let camera_ring = RingBuffer::new(
            device,
            1 << 16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "physics_camera_ring",
        )
        .expect("camera ring");
        let instance_ring = RingBuffer::new(
            device,
            (MAX_INSTANCES * INSTANCE_STRIDE * 8) as u64,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
            "physics_instance_ring",
        )
        .expect("instance ring");

        // Binding layout: binding 0 = camera uniform (dynamic — the per-draw
        // offset selects each batch's slot), binding 1 = instance storage.
        let binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::DynamicUniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    // The WGSL shader declares `var<storage, read>` — wgpu
                    // requires the layout access mode to match exactly.
                    BindingLayoutEntry::new(1, BindingType::StorageBufferReadOnly)
                        .with_visibility(ShaderStageFlags::VERTEX),
                )
                .with_label("shape_bindings"),
        );

        // Material (shared pipeline for all shape types)
        let shader_bytes = SHAPE_SHADER.as_bytes().to_vec();
        let vertex_layout = redlilium_graphics::VertexLayout::position_normal_uv();

        let descriptor = MaterialDescriptor::new()
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
            .with_binding_layout(binding_layout.clone())
            .with_vertex_layout(vertex_layout)
            .with_color_format(surface_format)
            .with_depth_format(TextureFormat::Depth32Float);

        let material = device.create_material(&descriptor).expect("shape material");

        // Generate meshes; their data uploads through the frame graph.
        let sphere_cpu = redlilium_core::mesh::generators::generate_sphere(1.0, 16, 8);
        let (sphere_mesh, sphere_ops) = device
            .create_mesh_deferred(&sphere_cpu)
            .expect("sphere mesh");

        let box_cpu = generate_box_cpu();
        let (box_mesh, box_ops) = device.create_mesh_deferred(&box_cpu).expect("box mesh");

        let mut pending_uploads = Vec::new();
        pending_uploads.extend(sphere_ops);
        pending_uploads.extend(box_ops);

        // Single frame-invariant material instance shared by both batches: the
        // binding group is created once (camera dynamic uniform at offset 0,
        // whole instance ring at offset 0) and reused every frame.
        let material_instance = Self::make_instance(
            device,
            binding_layout,
            &material,
            &camera_ring,
            &instance_ring,
        );

        Self {
            depth_texture,
            camera_ring,
            instance_ring,
            material_instance,
            sphere_batch: ShapeBatch {
                mesh: sphere_mesh,
                count: 0,
                camera_offset: 0,
            },
            box_batch: ShapeBatch {
                mesh: box_mesh,
                count: 0,
                camera_offset: 0,
            },
            pending_uploads,
        }
    }

    /// Build the shared, frame-invariant material instance: binding 0 is the
    /// camera ring bound at offset 0 (a dynamic uniform selected per draw),
    /// binding 1 is the **whole** instance ring bound at offset 0 (each draw's
    /// block is selected in-shader by `CameraUniforms.instance_base`).
    fn make_instance(
        device: &Arc<GraphicsDevice>,
        layout: Arc<BindingLayout>,
        material: &Arc<Material>,
        camera_ring: &RingBuffer,
        instance_ring: &RingBuffer,
    ) -> Arc<MaterialInstance> {
        let camera_size = std::mem::size_of::<CameraUniforms>() as u64;
        let bg = device
            .create_binding_group(
                layout,
                BindingGroupDescriptor::new()
                    .with_buffer_range(0, camera_ring.buffer().clone(), 0, camera_size)
                    .with_buffer_range(
                        1,
                        instance_ring.buffer().clone(),
                        0,
                        instance_ring.capacity(),
                    ),
            )
            .expect("create shape binding group");
        #[allow(clippy::arc_with_non_send_sync)]
        Arc::new(MaterialInstance::new(material.clone()).with_binding_group(bg))
    }

    /// Write a camera slot carrying `instance_base` into the camera ring; returns
    /// its offset (used as a batch's per-draw dynamic offset).
    fn write_camera_slot(&mut self, view_proj: Mat4, camera_pos: Vec3, instance_base: u32) -> u32 {
        let uniforms = CameraUniforms {
            view_proj: mat4_to_cols_array_2d(&view_proj),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
            light_dir: [0.4, 0.8, 0.5, 0.0],
            instance_base,
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
            .expect("write camera ring");
        alloc.offset as u32
    }

    /// Write a batch's instances into the instance ring. Returns
    /// `(instance_base, count)`: the block's element index (for the shader) and
    /// the number of instances written. The binding group is frame-invariant, so
    /// nothing is rebuilt here.
    fn write_instances(&mut self, instances: &[ShapeInstance]) -> (u32, u32) {
        let count = instances.len().min(MAX_INSTANCES);
        let size = (count.max(1) * INSTANCE_STRIDE) as u64;
        let alloc = self.instance_ring.allocate(size).unwrap_or_else(|| {
            self.instance_ring.reset();
            self.instance_ring
                .allocate(size)
                .expect("instance ring too small")
        });
        if count > 0 {
            let bytes = bytemuck::cast_slice(&instances[..count]);
            self.instance_ring
                .write(&alloc, bytes)
                .expect("write instance ring");
        }
        // The ring's 256-byte alignment is a multiple of the 128-byte stride, so
        // the offset divides exactly into an element index.
        debug_assert_eq!(alloc.offset % INSTANCE_STRIDE as u64, 0);
        let base = (alloc.offset / INSTANCE_STRIDE as u64) as u32;
        (base, count as u32)
    }

    /// Flush queued mesh uploads into the current frame's render graph.
    pub fn flush_uploads(&mut self, graph: &mut RenderGraph) {
        if self.pending_uploads.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.pending_uploads);
        let mut pass = TransferPass::new("physics_mesh_uploads".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(ops));
        graph.add_transfer_pass(pass);
    }

    /// Recreate the depth texture on resize.
    pub fn resize(&mut self, device: &Arc<GraphicsDevice>, width: u32, height: u32) {
        self.depth_texture = device
            .create_texture(&TextureDescriptor::new_2d(
                width.max(1),
                height.max(1),
                TextureFormat::Depth32Float,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .expect("depth texture resize");
    }

    /// Returns the depth texture for the render pass.
    pub fn depth_texture(&self) -> &Arc<Texture> {
        &self.depth_texture
    }

    /// Build instance data from a 3D physics world and upload to the rings.
    pub fn update_3d(&mut self, physics: &PhysicsWorld3D, view_proj: Mat4, camera_pos: Vec3) {
        use redlilium_ecs::physics::rapier3d::prelude::*;

        let mut sphere_instances = Vec::new();
        let mut box_instances = Vec::new();

        for (col_handle, collider) in physics.colliders.iter() {
            // Determine body type for coloring
            let (color, color_idx) = if let Some(parent) = collider.parent() {
                if let Some(body) = physics.bodies.get(parent) {
                    if body.is_fixed() {
                        (FIXED_COLOR, 0)
                    } else if body.is_kinematic() {
                        (KINEMATIC_COLOR, 0)
                    } else {
                        let idx = col_handle.0.into_raw_parts().0 as usize;
                        (DYNAMIC_COLORS[idx % DYNAMIC_COLORS.len()], idx)
                    }
                } else {
                    (FIXED_COLOR, 0)
                }
            } else {
                (FIXED_COLOR, 0)
            };
            let _ = color_idx;

            let pos = collider.position();
            let t = pos.translation;
            let r = pos.rotation;
            let translation = math::Vec3::new(t.x as f32, t.y as f32, t.z as f32);
            let rotation = math::quat_from_xyzw(r.x as f32, r.y as f32, r.z as f32, r.w as f32);

            let typed = collider.shape().as_typed_shape();
            match typed {
                TypedShape::Ball(ball) => {
                    let r = ball.radius as f32;
                    let model = math::mat4_from_scale_rotation_translation(
                        math::Vec3::new(r, r, r),
                        rotation,
                        translation,
                    );
                    sphere_instances.push(ShapeInstance {
                        model: mat4_to_cols_array_2d(&model),
                        color,
                        ..bytemuck::Zeroable::zeroed()
                    });
                }
                TypedShape::Cuboid(cuboid) => {
                    let he = cuboid.half_extents;
                    let scale =
                        math::Vec3::new(he.x as f32 * 2.0, he.y as f32 * 2.0, he.z as f32 * 2.0);
                    let model =
                        math::mat4_from_scale_rotation_translation(scale, rotation, translation);
                    box_instances.push(ShapeInstance {
                        model: mat4_to_cols_array_2d(&model),
                        color,
                        ..bytemuck::Zeroable::zeroed()
                    });
                }
                TypedShape::Capsule(capsule) => {
                    // Approximate capsule as a stretched sphere
                    let r = capsule.radius as f32;
                    let half_h = capsule.segment.a.distance(capsule.segment.b) as f32 / 2.0;
                    let scale = math::Vec3::new(r, half_h + r, r);
                    let model =
                        math::mat4_from_scale_rotation_translation(scale, rotation, translation);
                    sphere_instances.push(ShapeInstance {
                        model: mat4_to_cols_array_2d(&model),
                        color,
                        ..bytemuck::Zeroable::zeroed()
                    });
                }
                // For trimesh, heightfield, etc. — render as a box using AABB
                _ => {
                    let aabb = collider.shape().compute_local_aabb();
                    let center = aabb.center();
                    let he = aabb.half_extents();
                    let local_center =
                        math::Vec3::new(center.x as f32, center.y as f32, center.z as f32);
                    let scale =
                        math::Vec3::new(he.x as f32 * 2.0, he.y as f32 * 2.0, he.z as f32 * 2.0);
                    let world_center = translation + math::quat_rotate_vec3(rotation, local_center);
                    let model =
                        math::mat4_from_scale_rotation_translation(scale, rotation, world_center);
                    box_instances.push(ShapeInstance {
                        model: mat4_to_cols_array_2d(&model),
                        color,
                        ..bytemuck::Zeroable::zeroed()
                    });
                }
            }
        }

        // Upload instances into the ring and rebuild the batches' bindings.
        // Instances first: each batch's block base is folded into its camera slot.
        let (sphere_base, sphere_count) = self.write_instances(&sphere_instances);
        let (box_base, box_count) = self.write_instances(&box_instances);
        self.sphere_batch.count = sphere_count;
        self.sphere_batch.camera_offset =
            self.write_camera_slot(view_proj, camera_pos, sphere_base);
        self.box_batch.count = box_count;
        self.box_batch.camera_offset = self.write_camera_slot(view_proj, camera_pos, box_base);
    }

    /// Build instance data from a 2D physics world and upload to the rings.
    pub fn update_2d(&mut self, physics: &PhysicsWorld2D, view_proj: Mat4, camera_pos: Vec3) {
        use redlilium_ecs::physics::rapier2d::prelude::*;

        let mut sphere_instances = Vec::new();
        let mut box_instances = Vec::new();
        let z_depth: f32 = 0.4;

        for (col_handle, collider) in physics.colliders.iter() {
            let color = if let Some(parent) = collider.parent() {
                if let Some(body) = physics.bodies.get(parent) {
                    if body.is_fixed() {
                        FIXED_COLOR
                    } else if body.is_kinematic() {
                        KINEMATIC_COLOR
                    } else {
                        let idx = col_handle.0.into_raw_parts().0 as usize;
                        DYNAMIC_COLORS[idx % DYNAMIC_COLORS.len()]
                    }
                } else {
                    FIXED_COLOR
                }
            } else {
                FIXED_COLOR
            };

            let pos = collider.position();
            let t = pos.translation;
            let angle = pos.rotation.angle() as f32;
            let translation = math::Vec3::new(t.x as f32, t.y as f32, 0.0);
            let rotation = math::quat_from_rotation_z(angle);

            let typed = collider.shape().as_typed_shape();
            match typed {
                TypedShape::Ball(ball) => {
                    let r = ball.radius as f32;
                    let scale = math::Vec3::new(r, r, z_depth);
                    let model =
                        math::mat4_from_scale_rotation_translation(scale, rotation, translation);
                    sphere_instances.push(ShapeInstance {
                        model: mat4_to_cols_array_2d(&model),
                        color,
                        ..bytemuck::Zeroable::zeroed()
                    });
                }
                TypedShape::Cuboid(cuboid) => {
                    let he = cuboid.half_extents;
                    let scale = math::Vec3::new(he.x as f32 * 2.0, he.y as f32 * 2.0, z_depth);
                    let model =
                        math::mat4_from_scale_rotation_translation(scale, rotation, translation);
                    box_instances.push(ShapeInstance {
                        model: mat4_to_cols_array_2d(&model),
                        color,
                        ..bytemuck::Zeroable::zeroed()
                    });
                }
                TypedShape::Capsule(capsule) => {
                    let r = capsule.radius as f32;
                    let half_h = capsule.segment.a.distance(capsule.segment.b) as f32 / 2.0;
                    let scale = math::Vec3::new(r, half_h + r, z_depth);
                    let model =
                        math::mat4_from_scale_rotation_translation(scale, rotation, translation);
                    sphere_instances.push(ShapeInstance {
                        model: mat4_to_cols_array_2d(&model),
                        color,
                        ..bytemuck::Zeroable::zeroed()
                    });
                }
                _ => {
                    let aabb = collider.shape().compute_local_aabb();
                    let center = aabb.center();
                    let he = aabb.half_extents();
                    let local_center = math::Vec3::new(center.x as f32, center.y as f32, 0.0);
                    let scale = math::Vec3::new(he.x as f32 * 2.0, he.y as f32 * 2.0, z_depth);
                    let world_center = translation + math::quat_rotate_vec3(rotation, local_center);
                    let model =
                        math::mat4_from_scale_rotation_translation(scale, rotation, world_center);
                    box_instances.push(ShapeInstance {
                        model: mat4_to_cols_array_2d(&model),
                        color,
                        ..bytemuck::Zeroable::zeroed()
                    });
                }
            }
        }

        // Instances first: each batch's block base is folded into its camera slot.
        let (sphere_base, sphere_count) = self.write_instances(&sphere_instances);
        let (box_base, box_count) = self.write_instances(&box_instances);
        self.sphere_batch.count = sphere_count;
        self.sphere_batch.camera_offset =
            self.write_camera_slot(view_proj, camera_pos, sphere_base);
        self.box_batch.count = box_count;
        self.box_batch.camera_offset = self.write_camera_slot(view_proj, camera_pos, box_base);
    }

    /// Add draw commands for all shape batches to the graphics pass. Both draws
    /// share the frame-invariant binding group; the per-batch camera dynamic
    /// offset selects the slot carrying that batch's `instance_base`.
    pub fn add_draws(&self, pass: &mut GraphicsPass) {
        for batch in [&self.sphere_batch, &self.box_batch] {
            if batch.count > 0 {
                pass.add_draw_command(
                    DrawCommand::new(batch.mesh.clone(), self.material_instance.clone())
                        .with_instance_count(batch.count)
                        .with_dynamic_offsets(vec![vec![batch.camera_offset]]),
                );
            }
        }
    }
}
