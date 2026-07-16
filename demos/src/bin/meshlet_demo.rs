//! # Meshlet Demo (#111)
//!
//! Meshlet rendering through `VK_EXT_mesh_shader`: a grid of spheres, each
//! pre-partitioned at generation time into 32 meshlets (5×5-vertex patches of
//! a UV sphere — deliberately no meshoptimizer, the partition falls out of
//! the grid), drawn with a task → mesh → fragment pipeline:
//!
//! - the **task stage** culls whole meshlets before rasterization — bounding
//!   sphere vs. the cull camera's frustum, plus a backface cone test — and
//!   forwards survivors through the payload
//! - the **mesh stage** fetches the surviving meshlets' vertices/triangles
//!   from storage buffers (there is no vertex input at all) and emits them,
//!   colored by meshlet index so the partition is visible
//!
//! Press **SPACE** to freeze the cull camera: orbiting on reveals the
//! meshlets the frozen camera culled (holes where backfacing/off-frustum
//! meshlets used to be) — the proof the task stage is actually culling.
//!
//! Requires `DeviceCapabilities::mesh_shading`; renders a blank screen with a
//! notice on devices without it.

use std::sync::Arc;

use redlilium_app::{App, AppArgs, AppContext, AppHandler, DefaultAppArgs, DrawContext};
use redlilium_core::math::{
    DepthConvention, Mat4, Vec3, look_at_rh, mat4_to_cols_array_2d, perspective_rh_reversed,
};
use redlilium_graphics::{
    BindingGroupDescriptor, BufferDescriptor, BufferUsage, ColorAttachment, DepthStencilAttachment,
    FrameSchedule, GraphicsPass, MaterialDescriptor, MaterialInstance, RenderTargetConfig,
    RingBuffer, ShaderSource, ShaderStage, TextureDescriptor, TextureFormat, TextureUsage,
    TransferConfig, TransferOperation, TransferPass,
};
use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

const MESHLET_SHADER_SLANG: &str = include_str!("../../shaders/meshlet.slang");

/// Threads per task workgroup — must match TASK_GROUP_SIZE in the shader.
const TASK_GROUP_SIZE: u32 = 32;

// === GPU Data (layouts match meshlet.slang) ===

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMeshletVertex {
    /// xyz = object-space position.
    position: [f32; 4],
    /// xyz = object-space normal.
    normal: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMeshlet {
    vertex_offset: u32,
    vertex_count: u32,
    triangle_offset: u32,
    triangle_count: u32,
    /// xyz = center, w = radius (object space).
    sphere: [f32; 4],
    /// xyz = axis, w = cutoff (object space; cutoff 1 disables the test).
    cone: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstance {
    /// xyz = world translation, w = uniform scale.
    position_scale: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    view_proj: [[f32; 4]; 4],
    cull_planes: [[f32; 4]; 6],
    cull_camera_pos: [f32; 4],
    light_dir: [f32; 4],
    meshlet_count: u32,
    instance_count: u32,
    _pad: [u32; 2],
}

// === Sphere → Meshlets Generation ===

/// UV-sphere resolution. With 4×4-quad patches this yields exactly
/// (32/4)×(16/4) = 32 meshlets of 5×5 = 25 vertices / 32 triangles each —
/// comfortably under the shader's 64-vertex / 124-triangle output limits.
const SECTORS: usize = 32;
const STACKS: usize = 16;
const PATCH: usize = 4;

struct SphereMeshlets {
    vertices: Vec<GpuMeshletVertex>,
    vertex_indices: Vec<u32>,
    /// One packed triangle per u32: local indices i0 | i1<<8 | i2<<16.
    triangles: Vec<u32>,
    meshlets: Vec<GpuMeshlet>,
}

fn generate_sphere_meshlets() -> SphereMeshlets {
    // Shared vertex grid (seam column duplicated so patches never wrap).
    let mut vertices = Vec::with_capacity((STACKS + 1) * (SECTORS + 1));
    for r in 0..=STACKS {
        let theta = std::f32::consts::PI * r as f32 / STACKS as f32;
        for c in 0..=SECTORS {
            let phi = std::f32::consts::TAU * c as f32 / SECTORS as f32;
            let p = Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            );
            vertices.push(GpuMeshletVertex {
                position: [p.x, p.y, p.z, 1.0],
                normal: [p.x, p.y, p.z, 0.0],
            });
        }
    }
    let global = |r: usize, c: usize| (r * (SECTORS + 1) + c) as u32;
    let pos = |i: u32| {
        let p = vertices[i as usize].position;
        Vec3::new(p[0], p[1], p[2])
    };

    let mut vertex_indices = Vec::new();
    let mut triangles = Vec::new();
    let mut meshlets = Vec::new();

    for pj in 0..STACKS / PATCH {
        for pi in 0..SECTORS / PATCH {
            let vertex_offset = vertex_indices.len() as u32;
            let triangle_offset = triangles.len() as u32;

            // 5×5 local vertex grid → global indices.
            let side = PATCH + 1;
            for lr in 0..side {
                for lc in 0..side {
                    vertex_indices.push(global(pj * PATCH + lr, pi * PATCH + lc));
                }
            }
            let local_global = |l: usize| vertex_indices[vertex_offset as usize + l];

            // Two triangles per quad, wound so the face normal points outward
            // (checked against the centroid direction — the sphere is centered
            // at the origin, so no manual winding reasoning can go stale).
            let mut push_tri = |a: usize, b: usize, c: usize| {
                let (pa, pb, pc) = (
                    pos(local_global(a)),
                    pos(local_global(b)),
                    pos(local_global(c)),
                );
                let normal = (pb - pa).cross(&(pc - pa));
                let outward = pa + pb + pc;
                let (b, c) = if normal.dot(&outward) < 0.0 {
                    (c, b)
                } else {
                    (b, c)
                };
                triangles.push((a as u32) | ((b as u32) << 8) | ((c as u32) << 16));
            };
            for lr in 0..PATCH {
                for lc in 0..PATCH {
                    let l = lr * side + lc;
                    push_tri(l, l + side, l + 1);
                    push_tri(l + 1, l + side, l + side + 1);
                }
            }

            let vertex_count = (side * side) as u32;
            let triangle_count = triangles.len() as u32 - triangle_offset;

            // Bounding sphere over the patch vertices.
            let patch_positions: Vec<Vec3> = (0..vertex_count)
                .map(|l| pos(local_global(l as usize)))
                .collect();
            let center = patch_positions.iter().sum::<Vec3>() / vertex_count as f32;
            let radius = patch_positions
                .iter()
                .map(|p| (p - center).norm())
                .fold(0.0f32, f32::max);

            // Backface cone (meshoptimizer convention): axis is the
            // area-weighted average triangle normal; cutoff comes from the
            // worst-case spread. Degenerate (pole) triangles contribute
            // nothing. mindot <= 0 → the meshlet can never be fully
            // backfacing → cutoff 1 disables the test.
            let tri_normals: Vec<Vec3> = (0..triangle_count)
                .map(|t| {
                    let packed = triangles[(triangle_offset + t) as usize];
                    let (a, b, c) = (
                        (packed & 0xFF) as usize,
                        ((packed >> 8) & 0xFF) as usize,
                        ((packed >> 16) & 0xFF) as usize,
                    );
                    let (pa, pb, pc) = (
                        pos(local_global(a)),
                        pos(local_global(b)),
                        pos(local_global(c)),
                    );
                    (pb - pa).cross(&(pc - pa))
                })
                .collect();
            let axis_sum: Vec3 = tri_normals.iter().sum();
            let (axis, cutoff) = if axis_sum.norm() < 1e-6 {
                (Vec3::new(0.0, 1.0, 0.0), 1.0)
            } else {
                let axis = axis_sum.normalize();
                let mindot = tri_normals
                    .iter()
                    .filter(|n| n.norm() > 1e-6)
                    .map(|n| axis.dot(&n.normalize()))
                    .fold(1.0f32, f32::min);
                let cutoff = if mindot <= 0.0 {
                    1.0
                } else {
                    (1.0 - mindot * mindot).sqrt()
                };
                (axis, cutoff)
            };

            meshlets.push(GpuMeshlet {
                vertex_offset,
                vertex_count,
                triangle_offset,
                triangle_count,
                sphere: [center.x, center.y, center.z, radius],
                cone: [axis.x, axis.y, axis.z, cutoff],
            });
        }
    }

    SphereMeshlets {
        vertices,
        vertex_indices,
        triangles,
        meshlets,
    }
}

// === Scene ===

const GRID: usize = 8;

fn build_instances() -> Vec<GpuInstance> {
    const PALETTE: [[f32; 4]; 4] = [
        [0.90, 0.90, 0.95, 1.0],
        [0.95, 0.55, 0.40, 1.0],
        [0.45, 0.75, 0.95, 1.0],
        [0.95, 0.85, 0.45, 1.0],
    ];
    let mut instances = Vec::with_capacity(GRID * GRID);
    for j in 0..GRID {
        for i in 0..GRID {
            let x = (i as f32 - (GRID - 1) as f32 / 2.0) * 3.0;
            let z = (j as f32 - (GRID - 1) as f32 / 2.0) * 3.0;
            instances.push(GpuInstance {
                position_scale: [x, 0.0, z, 1.0],
                color: PALETTE[(i + j) % PALETTE.len()],
            });
        }
    }
    instances
}

/// World-space frustum planes from a view-projection matrix
/// (Gribb–Hartmann; the engine's projections use [0,1] depth, so the two z
/// bounds are row 2 alone and row 3 − row 2). Convention-independent: under
/// reversed-Z (ADR-038) the same two planes swap near/far roles, and all six
/// are tested uniformly. Inside when `dot(n, p) + d >= 0`.
fn frustum_planes(vp: &Mat4) -> [[f32; 4]; 6] {
    let row = |i: usize| vp.row(i).transpose();
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map(|p| {
        let n = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        [p.x / n, p.y / n, p.z / n, p.w / n]
    })
}

// === Demo Application ===

struct MeshletDemo {
    material_instance: Option<Arc<MaterialInstance>>,
    uniform_ring: Option<RingBuffer>,
    uniform_offset: u32,
    depth_texture: Option<Arc<redlilium_graphics::Texture>>,
    pending_uploads: Vec<TransferOperation>,
    meshlet_count: u32,
    instance_count: u32,
    /// Whether SPACE froze the cull camera; the capture itself happens in
    /// `on_update`, where the real aspect ratio is available.
    freeze_culling: bool,
    /// Frozen cull camera (view-projection + position).
    frozen_cull: Option<(Mat4, Vec3)>,
    supported: bool,
    time: f32,
}

impl MeshletDemo {
    fn new() -> Self {
        Self {
            material_instance: None,
            uniform_ring: None,
            uniform_offset: 0,
            depth_texture: None,
            pending_uploads: Vec::new(),
            meshlet_count: 0,
            instance_count: 0,
            freeze_culling: false,
            frozen_cull: None,
            supported: false,
            time: 0.0,
        }
    }

    fn create_gpu_resources(&mut self, ctx: &mut AppContext) {
        let device = ctx.device();

        let sphere = generate_sphere_meshlets();
        let instances = build_instances();
        self.meshlet_count = sphere.meshlets.len() as u32;
        self.instance_count = instances.len() as u32;
        log::info!(
            "Generated {} meshlets/sphere ({} verts, {} tris) × {} instances",
            self.meshlet_count,
            sphere.vertices.len(),
            sphere.triangles.len(),
            self.instance_count,
        );

        // Static storage buffers, uploaded once through the frame graph.
        let mut storage = |label: &str, bytes: &[u8]| {
            let buffer = device
                .create_buffer(
                    &BufferDescriptor::new(
                        bytes.len() as u64,
                        BufferUsage::STORAGE | BufferUsage::COPY_DST,
                    )
                    .with_label(label),
                )
                .unwrap_or_else(|e| panic!("create {label}: {e:?}"));
            self.pending_uploads.push(TransferOperation::write_buffer(
                buffer.clone(),
                0,
                bytes.to_vec().into(),
            ));
            buffer
        };
        let vertex_buffer = storage("meshlet vertices", bytemuck::cast_slice(&sphere.vertices));
        let vertex_index_buffer = storage(
            "meshlet vertex indices",
            bytemuck::cast_slice(&sphere.vertex_indices),
        );
        let triangle_buffer = storage("meshlet triangles", bytemuck::cast_slice(&sphere.triangles));
        let meshlet_buffer = storage("meshlets", bytemuck::cast_slice(&sphere.meshlets));
        let instance_buffer = storage("meshlet instances", bytemuck::cast_slice(&instances));

        // Task + mesh + fragment material from the Slang source. Binding
        // layouts come from reflection (baked on the default Slang-off
        // build); binding 0 becomes a per-draw dynamic offset into the ring.
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Task,
                        MESHLET_SHADER_SLANG.as_bytes().to_vec(),
                        "ts_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Mesh,
                        MESHLET_SHADER_SLANG.as_bytes().to_vec(),
                        "ms_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        MESHLET_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_color_format(ctx.surface_format())
                    .with_depth_format(TextureFormat::Depth32Float)
                    .with_dynamic_uniform(0, 0)
                    .with_label("meshlet material"),
            )
            .expect("create meshlet material");

        let uniform_ring = RingBuffer::new(
            device,
            1 << 16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "meshlet scene ring",
        )
        .expect("create uniform ring");

        let layout = material.binding_layouts()[0].clone();
        let binding_group = device
            .create_binding_group(
                layout,
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        uniform_ring.buffer().clone(),
                        0,
                        size_of::<SceneUniforms>() as u64,
                    )
                    .with_buffer(1, vertex_buffer)
                    .with_buffer(2, vertex_index_buffer)
                    .with_buffer(3, triangle_buffer)
                    .with_buffer(4, meshlet_buffer)
                    .with_buffer(5, instance_buffer)
                    .with_label("meshlet binding group"),
            )
            .expect("create binding group");

        self.uniform_ring = Some(uniform_ring);
        self.material_instance = Some(Arc::new(
            MaterialInstance::new(material).with_binding_group(binding_group),
        ));
        self.create_depth_texture(ctx);

        log::info!("Meshlet GPU resources created");
    }

    fn create_depth_texture(&mut self, ctx: &AppContext) {
        let depth = ctx
            .device()
            .create_texture(
                &TextureDescriptor::new_2d(
                    ctx.width(),
                    ctx.height(),
                    TextureFormat::Depth32Float,
                    TextureUsage::RENDER_ATTACHMENT,
                )
                .with_label("meshlet depth"),
            )
            .expect("create depth texture");
        self.depth_texture = Some(depth);
    }

    /// This frame's render camera: a slow orbit around the sphere grid.
    fn render_camera(&self, aspect: f32) -> (Mat4, Vec3) {
        let angle = self.time * 0.3;
        let eye = Vec3::new(angle.cos() * 18.0, 9.0, angle.sin() * 18.0);
        let view = look_at_rh(&eye, &Vec3::new(0.0, 0.0, 0.0), &Vec3::new(0.0, 1.0, 0.0));
        let proj = perspective_rh_reversed(60.0_f32.to_radians(), aspect, 0.1, 200.0);
        (proj * view, eye)
    }

    fn update_uniforms(&mut self, ctx: &AppContext) {
        let (view_proj, eye) = self.render_camera(ctx.aspect_ratio());
        // Culling follows the render camera unless frozen (SPACE).
        let (cull_vp, cull_pos) = self.frozen_cull.unwrap_or((view_proj, eye));

        let light_dir = Vec3::new(-0.4, -1.0, -0.25).normalize();
        let uniforms = SceneUniforms {
            view_proj: mat4_to_cols_array_2d(&view_proj),
            cull_planes: frustum_planes(&cull_vp),
            cull_camera_pos: [cull_pos.x, cull_pos.y, cull_pos.z, 1.0],
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
            meshlet_count: self.meshlet_count,
            instance_count: self.instance_count,
            _pad: [0; 2],
        };

        if let Some(ring) = &mut self.uniform_ring {
            let bytes = bytemuck::bytes_of(&uniforms);
            let size = bytes.len() as u64;
            let alloc = ring.allocate(size).unwrap_or_else(|| {
                ring.reset();
                ring.allocate(size).expect("uniform ring too small")
            });
            ring.write(&alloc, bytes).expect("write scene uniforms");
            self.uniform_offset = alloc.offset as u32;
        }
    }
}

impl AppHandler for MeshletDemo {
    fn on_init(&mut self, ctx: &mut AppContext) {
        log::info!("Initializing Meshlet Demo (#111)");

        self.supported = ctx.device().capabilities().mesh_shading;
        if !self.supported {
            log::warn!(
                "This device does not support mesh shading \
                 (DeviceCapabilities::mesh_shading is false) — the demo renders a blank \
                 screen. Mesh shading needs the Vulkan backend on a GPU with \
                 VK_EXT_mesh_shader."
            );
            return;
        }
        log::info!(
            "Mesh shading supported on {} — SPACE freezes the cull camera",
            ctx.device().name()
        );

        self.create_gpu_resources(ctx);
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        if self.supported {
            self.create_depth_texture(ctx);
        }
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        if !self.supported
            || event.state != ElementState::Pressed
            || event.repeat
            || event.physical_key != PhysicalKey::Code(KeyCode::Space)
        {
            return;
        }
        self.freeze_culling = !self.freeze_culling;
        if self.freeze_culling {
            log::info!("Cull camera FROZEN — orbit on to see culled meshlets");
        } else {
            log::info!("Cull camera unfrozen");
        }
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        if !self.supported {
            return true;
        }
        self.time += 1.0 / 60.0;
        if !self.freeze_culling {
            self.frozen_cull = None;
        } else if self.frozen_cull.is_none() {
            self.frozen_cull = Some(self.render_camera(ctx.aspect_ratio()));
        }
        self.update_uniforms(ctx);
        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        let mut graph = ctx.acquire_graph();

        if !self.supported {
            let mut pass = GraphicsPass::new("clear".into());
            pass.set_render_targets(
                RenderTargetConfig::new().with_color(
                    ColorAttachment::from_surface(ctx.swapchain_texture())
                        .with_clear_color(0.08, 0.02, 0.02, 1.0),
                ),
            );
            graph.add_graphics_pass(pass);
            return ctx.render(graph);
        }

        // First frame: geometry uploads.
        let mut upload_handle = None;
        if !self.pending_uploads.is_empty() {
            let ops = std::mem::take(&mut self.pending_uploads);
            let mut transfer_pass = TransferPass::new("meshlet uploads".into());
            transfer_pass.set_transfer_config(TransferConfig::new().with_operations(ops));
            upload_handle = Some(graph.add_transfer_pass(transfer_pass));
            log::info!("Meshlet buffers queued for upload");
        }

        // Mesh-shading pass: one task workgroup per TASK_GROUP_SIZE meshlets,
        // one row of workgroups per instance.
        let mut render_pass = GraphicsPass::new("meshlets".into());
        render_pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(
                    ColorAttachment::from_surface(ctx.swapchain_texture())
                        .with_clear_color(0.05, 0.06, 0.09, 1.0),
                )
                .with_depth_stencil(
                    DepthStencilAttachment::from_texture(
                        self.depth_texture.as_ref().unwrap().clone(),
                    )
                    .with_clear_depth(DepthConvention::default().clear_depth()),
                ),
        );
        render_pass.add_mesh_tasks_command(
            redlilium_graphics::MeshTasksDrawCommand::new(
                Arc::clone(self.material_instance.as_ref().unwrap()),
                [
                    self.meshlet_count.div_ceil(TASK_GROUP_SIZE),
                    self.instance_count,
                    1,
                ],
            )
            .with_dynamic_offsets(vec![vec![self.uniform_offset]]),
        );
        let render_handle = graph.add_graphics_pass(render_pass);
        if let Some(upload) = upload_handle {
            graph.add_dependency(render_handle, upload);
        }

        ctx.render(graph)
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down Meshlet Demo");
    }
}

// === Entry Point ===

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args =
        DefaultAppArgs::parse().with_title_str("Meshlet Demo (#111) — SPACE freezes culling");
    App::run(MeshletDemo::new(), args);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Mesh shading is native-only (#111); no wasm entry point.
}
