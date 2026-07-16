//! # Ray Query Demo (#110, ADR-032)
//!
//! Hardware-ray-traced scene rendered entirely by inline ray tracing
//! (`VK_KHR_ray_query`) from a fullscreen fragment shader — no rasterized
//! geometry at all:
//!
//! - two BLASes (floor quad, cube) built on the GPU through the frame graph
//! - a TLAS with several cube instances, rebuilt every frame (one cube spins)
//! - primary rays traced per pixel against the TLAS; hit surfaces shaded
//!   with Lambert lighting plus **ray-traced hard shadows** (a second ray
//!   query toward the light)
//!
//! Requires `DeviceCapabilities::ray_query` (Vulkan backend with
//! `VK_KHR_acceleration_structure` + `VK_KHR_ray_query`); prints a notice and
//! exits cleanly on devices without it.

use std::sync::Arc;

use redlilium_app::{App, AppArgs, AppContext, AppHandler, DefaultAppArgs, DrawContext};
use redlilium_core::math::{
    Mat4, Vec3, look_at_rh, mat4_from_scale_rotation_translation, mat4_to_cols_array_2d,
    perspective_rh, quat_from_rotation_y,
};
use redlilium_core::mesh::{CpuMesh, IndexFormat, VertexLayout};
use redlilium_graphics::{
    AccelerationStructureBuildPass, BindingGroupDescriptor, BindingLayout, Blas, BlasDescriptor,
    BlasTriangles, BufferDescriptor, BufferUsage, ColorAttachment, DrawCommand, FrameSchedule,
    GraphicsPass, MaterialDescriptor, MaterialInstance, Mesh, QueuePreference, RenderTargetConfig,
    RingBuffer, ShaderSource, ShaderStage, Tlas, TlasDescriptor, TlasInstance, TransferConfig,
    TransferOperation, TransferPass,
};

// === Shader (WGSL: naga compiles ray queries to SPIR-V on the Vulkan backend) ===

const RAY_SHADER_WGSL: &str = r#"
struct Camera {
    inv_view_proj: mat4x4<f32>,
    // xyz = camera position
    origin: vec4<f32>,
    // xyz = normalized direction the light travels
    light_dir: vec4<f32>,
    // per-mesh (vertex_base, index_base) pairs, indexed by instance_custom_index:
    // x/y = mesh 0, z/w = mesh 1 (bases in floats / u32s respectively)
    mesh_table: vec4<u32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var tlas: acceleration_structure;
@group(0) @binding(2) var<storage, read> positions: array<f32>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.clip = vec4<f32>(pos.xy, 0.0, 1.0);
    out.ndc = pos.xy;
    return out;
}

fn fetch_position(base: u32, index: u32) -> vec3<f32> {
    let p = base + index * 3u;
    return vec3<f32>(positions[p], positions[p + 1u], positions[p + 2u]);
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(vec3<f32>(0.35, 0.30, 0.28), vec3<f32>(0.35, 0.55, 0.85), t);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Reconstruct the primary ray from the inverse view-projection ([0,1]
    // depth: z = 1 is the far plane).
    let far = camera.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - camera.origin.xyz);

    var rq: ray_query;
    rayQueryInitialize(&rq, tlas,
        RayDesc(0x0u, 0xFFu, 0.001, 1000.0, camera.origin.xyz, dir));
    while (rayQueryProceed(&rq)) {}
    let hit = rayQueryGetCommittedIntersection(&rq);

    if (hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE) {
        return vec4<f32>(sky(dir), 1.0);
    }

    // Fetch the hit triangle and derive its world-space geometric normal.
    let mesh = min(hit.instance_custom_index, 1u);
    var vertex_base = camera.mesh_table.x;
    var index_base = camera.mesh_table.y;
    if (mesh == 1u) {
        vertex_base = camera.mesh_table.z;
        index_base = camera.mesh_table.w;
    }
    let tri = index_base + hit.primitive_index * 3u;
    let p0 = fetch_position(vertex_base, indices[tri]);
    let p1 = fetch_position(vertex_base, indices[tri + 1u]);
    let p2 = fetch_position(vertex_base, indices[tri + 2u]);
    let w0 = hit.object_to_world * vec4<f32>(p0, 1.0);
    let w1 = hit.object_to_world * vec4<f32>(p1, 1.0);
    let w2 = hit.object_to_world * vec4<f32>(p2, 1.0);
    var normal = normalize(cross(w1 - w0, w2 - w0));
    if (dot(normal, dir) > 0.0) {
        normal = -normal;
    }

    let hit_point = camera.origin.xyz + dir * hit.t;
    let to_light = -camera.light_dir.xyz;

    // Shadow ray: any hit toward the light means shadow. 0x4 =
    // terminate-on-first-hit; geometry is opaque via BLAS flags.
    var shadow_rq: ray_query;
    rayQueryInitialize(&shadow_rq, tlas,
        RayDesc(0x4u, 0xFFu, 0.01, 1000.0, hit_point + normal * 0.005, to_light));
    while (rayQueryProceed(&shadow_rq)) {}
    let shadow_hit = rayQueryGetCommittedIntersection(&shadow_rq);
    var shadow = 1.0;
    if (shadow_hit.kind != RAY_QUERY_INTERSECTION_NONE) {
        shadow = 0.15;
    }

    // Per-instance albedo from the custom index.
    var albedo = vec3<f32>(0.75, 0.75, 0.78);
    switch hit.instance_id % 4u {
        case 1u: { albedo = vec3<f32>(0.85, 0.30, 0.25); }
        case 2u: { albedo = vec3<f32>(0.25, 0.65, 0.90); }
        case 3u: { albedo = vec3<f32>(0.95, 0.75, 0.20); }
        default: {}
    }

    let diffuse = max(dot(normal, to_light), 0.0) * shadow;
    let ambient = 0.22;
    // Linear output: the surface is an sRGB format, the hardware encodes.
    let color = albedo * (ambient + diffuse * 0.9);
    return vec4<f32>(color, 1.0);
}
"#;

// === Uniform Data ===

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniforms {
    inv_view_proj: [[f32; 4]; 4],
    origin: [f32; 4],
    light_dir: [f32; 4],
    mesh_table: [u32; 4],
}

// === Scene Geometry ===

/// Floor quad on y = 0: 4 vertices, 2 triangles.
const FLOOR_POSITIONS: [f32; 12] = [
    -12.0, 0.0, -12.0, //
    12.0, 0.0, -12.0, //
    12.0, 0.0, 12.0, //
    -12.0, 0.0, 12.0,
];
const FLOOR_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// Unit cube centered at the origin: 8 vertices, 12 triangles.
const CUBE_POSITIONS: [f32; 24] = [
    -0.5, -0.5, -0.5, //
    0.5, -0.5, -0.5, //
    0.5, 0.5, -0.5, //
    -0.5, 0.5, -0.5, //
    -0.5, -0.5, 0.5, //
    0.5, -0.5, 0.5, //
    0.5, 0.5, 0.5, //
    -0.5, 0.5, 0.5,
];
const CUBE_INDICES: [u32; 36] = [
    0, 2, 1, 0, 3, 2, // -Z
    4, 5, 6, 4, 6, 7, // +Z
    0, 1, 5, 0, 5, 4, // -Y
    3, 6, 2, 3, 7, 6, // +Y
    0, 4, 7, 0, 7, 3, // -X
    1, 2, 6, 1, 6, 5, // +X
];

/// Row-major 3×4 (`VkTransformMatrixKHR`) from a column-major `Mat4`.
fn instance_transform(m: &Mat4) -> [[f32; 4]; 3] {
    let c = mat4_to_cols_array_2d(m);
    [
        [c[0][0], c[1][0], c[2][0], c[3][0]],
        [c[0][1], c[1][1], c[2][1], c[3][1]],
        [c[0][2], c[1][2], c[2][2], c[3][2]],
    ]
}

// === Demo Application ===

struct RayQueryDemo {
    material_instance: Option<Arc<MaterialInstance>>,
    fullscreen_mesh: Option<Arc<Mesh>>,
    uniform_ring: Option<RingBuffer>,
    uniform_offset: u32,
    floor_blas: Option<Arc<Blas>>,
    cube_blas: Option<Arc<Blas>>,
    tlas: Option<Arc<Tlas>>,
    /// Geometry uploads queued at init, flushed into the first frame's graph.
    pending_uploads: Vec<TransferOperation>,
    /// BLASes build once, on the first drawn frame.
    blases_built: bool,
    /// Whether the device supports ray queries; when false the demo idles.
    supported: bool,
    time: f32,
}

impl RayQueryDemo {
    fn new() -> Self {
        Self {
            material_instance: None,
            fullscreen_mesh: None,
            uniform_ring: None,
            uniform_offset: 0,
            floor_blas: None,
            cube_blas: None,
            tlas: None,
            pending_uploads: Vec::new(),
            blases_built: false,
            supported: false,
            time: 0.0,
        }
    }

    fn create_gpu_resources(&mut self, ctx: &mut AppContext) {
        let device = ctx.device();

        // --- Geometry buffers: concatenated positions/indices, shared by the
        // BLAS builds (AS input) and the shader (storage fetch for normals).
        let mut positions: Vec<f32> = Vec::new();
        positions.extend_from_slice(&FLOOR_POSITIONS);
        positions.extend_from_slice(&CUBE_POSITIONS);
        let mut indices: Vec<u32> = Vec::new();
        indices.extend_from_slice(&FLOOR_INDICES);
        indices.extend_from_slice(&CUBE_INDICES);

        let geometry_usage = BufferUsage::ACCELERATION_STRUCTURE_INPUT
            | BufferUsage::STORAGE
            | BufferUsage::COPY_DST;
        let position_buffer = device
            .create_buffer(
                &BufferDescriptor::new((positions.len() * 4) as u64, geometry_usage)
                    .with_label("rq positions"),
            )
            .expect("create position buffer");
        let index_buffer = device
            .create_buffer(
                &BufferDescriptor::new((indices.len() * 4) as u64, geometry_usage)
                    .with_label("rq indices"),
            )
            .expect("create index buffer");
        self.pending_uploads.push(TransferOperation::write_buffer(
            position_buffer.clone(),
            0,
            bytemuck::cast_slice(&positions).to_vec().into(),
        ));
        self.pending_uploads.push(TransferOperation::write_buffer(
            index_buffer.clone(),
            0,
            bytemuck::cast_slice(&indices).to_vec().into(),
        ));

        // --- BLASes: floor (geometry range at offset 0) and cube (offsets
        // past the floor's 4 vertices / 6 indices).
        let floor_blas = device
            .create_blas(
                &BlasDescriptor::new(vec![BlasTriangles::indexed(
                    position_buffer.clone(),
                    4,
                    index_buffer.clone(),
                    IndexFormat::Uint32,
                    2,
                )])
                .with_label("floor blas")
                .with_compaction(),
            )
            .expect("create floor BLAS");
        let cube_geometry = BlasTriangles {
            vertex_buffer: position_buffer.clone(),
            vertex_offset: (FLOOR_POSITIONS.len() * 4) as u64,
            vertex_stride: 12,
            vertex_count: 8,
            index_buffer: Some(index_buffer.clone()),
            index_offset: (FLOOR_INDICES.len() * 4) as u64,
            index_format: IndexFormat::Uint32,
            triangle_count: 12,
            opaque: true,
        };
        let cube_blas = device
            .create_blas(
                &BlasDescriptor::new(vec![cube_geometry])
                    .with_label("cube blas")
                    .with_compaction(),
            )
            .expect("create cube BLAS");

        // --- TLAS for the floor + up to 8 cubes.
        let tlas = device
            .create_tlas(&TlasDescriptor::new(8).with_label("scene tlas"))
            .expect("create TLAS");

        // --- Fullscreen triangle mesh (position-only, covers clip space).
        let fullscreen = CpuMesh::new(VertexLayout::position_only())
            .with_vertex_data(
                0,
                bytemuck::cast_slice(&[
                    -1.0f32, -1.0, 0.0, //
                    3.0, -1.0, 0.0, //
                    -1.0, 3.0, 0.0,
                ])
                .to_vec(),
            )
            .with_label("fullscreen triangle");
        let (mesh, mesh_ops) = device
            .create_mesh_deferred(&fullscreen)
            .expect("create fullscreen mesh");
        self.pending_uploads.extend(mesh_ops);
        self.fullscreen_mesh = Some(mesh);

        // --- Material: WGSL ray-query shader with an explicit binding layout
        // (WGSL sources have no reflection).
        let binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    redlilium_graphics::BindingLayoutEntry::new(
                        0,
                        redlilium_graphics::BindingType::DynamicUniformBuffer,
                    )
                    .with_visibility(redlilium_graphics::ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    redlilium_graphics::BindingLayoutEntry::new(
                        1,
                        redlilium_graphics::BindingType::AccelerationStructure,
                    )
                    .with_visibility(redlilium_graphics::ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    redlilium_graphics::BindingLayoutEntry::new(
                        2,
                        redlilium_graphics::BindingType::StorageBufferReadOnly,
                    )
                    .with_visibility(redlilium_graphics::ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    redlilium_graphics::BindingLayoutEntry::new(
                        3,
                        redlilium_graphics::BindingType::StorageBufferReadOnly,
                    )
                    .with_visibility(redlilium_graphics::ShaderStageFlags::FRAGMENT),
                )
                .with_label("rq bindings"),
        );
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        RAY_SHADER_WGSL.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        RAY_SHADER_WGSL.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_vertex_layout(VertexLayout::position_only())
                    .with_binding_layout(binding_layout.clone())
                    .with_color_format(ctx.surface_format())
                    .with_label("rq material"),
            )
            .expect("create ray-query material");

        let uniform_ring = RingBuffer::new(
            device,
            1 << 16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "rq camera ring",
        )
        .expect("create uniform ring");

        let binding_group = device
            .create_binding_group(
                binding_layout,
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        uniform_ring.buffer().clone(),
                        0,
                        size_of::<CameraUniforms>() as u64,
                    )
                    .with_acceleration_structure(1, tlas.clone())
                    .with_buffer(2, position_buffer.clone())
                    .with_buffer(3, index_buffer.clone())
                    .with_label("rq binding group"),
            )
            .expect("create binding group");
        self.uniform_ring = Some(uniform_ring);
        self.material_instance = Some(Arc::new(
            MaterialInstance::new(material).with_binding_group(binding_group),
        ));

        self.floor_blas = Some(floor_blas);
        self.cube_blas = Some(cube_blas);
        self.tlas = Some(tlas);

        log::info!("Ray-query GPU resources created");
    }

    /// This frame's TLAS instances: static floor and cubes, one spinning.
    fn build_instances(&self) -> Vec<TlasInstance> {
        let floor = self.floor_blas.as_ref().unwrap();
        let cube = self.cube_blas.as_ref().unwrap();

        let cube_at = |position: Vec3, scale: f32, angle: f32| {
            mat4_from_scale_rotation_translation(
                Vec3::new(scale, scale, scale),
                quat_from_rotation_y(angle),
                position,
            )
        };

        vec![
            TlasInstance::new(Arc::clone(floor)).with_custom_index(0),
            TlasInstance::new(Arc::clone(cube))
                .with_custom_index(1)
                .with_transform(instance_transform(&cube_at(
                    Vec3::new(0.0, 0.9, 0.0),
                    1.8,
                    self.time,
                ))),
            TlasInstance::new(Arc::clone(cube))
                .with_custom_index(1)
                .with_transform(instance_transform(&cube_at(
                    Vec3::new(-2.6, 0.5, 1.2),
                    1.0,
                    0.6,
                ))),
            TlasInstance::new(Arc::clone(cube))
                .with_custom_index(1)
                .with_transform(instance_transform(&cube_at(
                    Vec3::new(2.4, 0.35, -1.0),
                    0.7,
                    -0.4,
                ))),
        ]
    }

    fn update_camera(&mut self, ctx: &AppContext) {
        let aspect = ctx.aspect_ratio();
        let proj = perspective_rh(60.0_f32.to_radians(), aspect, 0.1, 100.0);
        let eye = Vec3::new(5.0, 3.5, 6.5);
        let view = look_at_rh(&eye, &Vec3::new(0.0, 0.7, 0.0), &Vec3::new(0.0, 1.0, 0.0));
        let inv_view_proj = (proj * view)
            .try_inverse()
            .expect("view-projection must be invertible");

        let light_dir = Vec3::new(-0.45, -1.0, -0.3).normalize();
        let uniforms = CameraUniforms {
            inv_view_proj: mat4_to_cols_array_2d(&inv_view_proj),
            origin: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
            mesh_table: [
                0,
                0,
                FLOOR_POSITIONS.len() as u32,
                FLOOR_INDICES.len() as u32,
            ],
        };

        if let Some(ring) = &mut self.uniform_ring {
            let bytes = bytemuck::bytes_of(&uniforms);
            let size = bytes.len() as u64;
            let alloc = ring.allocate(size).unwrap_or_else(|| {
                ring.reset();
                ring.allocate(size).expect("uniform ring too small")
            });
            ring.write(&alloc, bytes).expect("write camera uniforms");
            self.uniform_offset = alloc.offset as u32;
        }
    }
}

impl AppHandler for RayQueryDemo {
    fn on_init(&mut self, ctx: &mut AppContext) {
        log::info!("Initializing Ray Query Demo (#110)");

        self.supported = ctx.device().capabilities().ray_query;
        if !self.supported {
            log::warn!(
                "This device does not support ray queries \
                 (DeviceCapabilities::ray_query is false) — the demo renders a blank screen. \
                 Ray queries need the Vulkan backend on a GPU with \
                 VK_KHR_acceleration_structure + VK_KHR_ray_query."
            );
            return;
        }
        log::info!(
            "Ray queries supported on {} — building acceleration structures",
            ctx.device().name()
        );

        self.create_gpu_resources(ctx);
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        if !self.supported {
            return true;
        }
        self.time += 1.0 / 60.0;
        self.update_camera(ctx);
        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        let mut graph = ctx.acquire_graph();

        if !self.supported {
            // Clear-only frame on unsupported devices.
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

        // 1) First frame: upload geometry and build both BLASes on the
        // **async compute queue** (#110 phase 3 — AS builds are legal on any
        // compute queue; falls back to graphics on devices without one). The
        // main graph's TLAS build reads the BLAS backings, so the cross-queue
        // build→build hazard is resolved automatically by the trackers' timeline
        // waits (no manual dependency across graphs).
        if !self.blases_built {
            let mut async_graph = ctx.acquire_graph();
            async_graph.set_queue_preference(QueuePreference::AsyncCompute);

            let ops = std::mem::take(&mut self.pending_uploads);
            let mut transfer_pass = TransferPass::new("geometry uploads".into());
            transfer_pass.set_transfer_config(TransferConfig::new().with_operations(ops));
            let upload = async_graph.add_transfer_pass(transfer_pass);

            let mut pass = AccelerationStructureBuildPass::new("blas build".into());
            pass.add_blas_build(Arc::clone(self.floor_blas.as_ref().unwrap()));
            pass.add_blas_build(Arc::clone(self.cube_blas.as_ref().unwrap()));
            let build = async_graph.add_acceleration_structure_build_pass(pass);
            async_graph.add_dependency(build, upload);

            ctx.submit(async_graph);
            self.blases_built = true;
            log::info!("BLAS builds queued on the async compute queue (compaction enabled)");
        }

        // 2) Every frame: write this frame's instances and rebuild the TLAS.
        let tlas = Arc::clone(self.tlas.as_ref().unwrap());
        tlas.write_instances(&self.build_instances())
            .expect("write TLAS instances");
        let tlas_handle = {
            let mut pass = AccelerationStructureBuildPass::new("tlas build".into());
            pass.add_tlas_build(Arc::clone(&tlas));
            graph.add_acceleration_structure_build_pass(pass)
        };

        // Drive transparent BLAS compaction: a few frames after the build the
        // compacted size comes back and this adds the compaction copy; the
        // structure is swapped for the smaller one with no visible change.
        graph.flush_acceleration_structure_compaction(ctx.device());

        // 3) Fullscreen ray-traced pass into the swapchain.
        let mut render_pass = GraphicsPass::new("ray trace".into());
        render_pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_surface(ctx.swapchain_texture())
                    .with_clear_color(0.0, 0.0, 0.0, 1.0),
            ),
        );
        render_pass.add_draw_command(
            DrawCommand::new(
                Arc::clone(self.fullscreen_mesh.as_ref().unwrap()),
                Arc::clone(self.material_instance.as_ref().unwrap()),
            )
            .with_dynamic_offsets(vec![vec![self.uniform_offset]]),
        );
        let render_handle = graph.add_graphics_pass(render_pass);
        graph.add_dependency(render_handle, tlas_handle);

        ctx.render(graph)
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down Ray Query Demo");
    }
}

// === Entry Point ===

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args = DefaultAppArgs::parse().with_title_str("Ray Query Demo (#110)");
    App::run(RayQueryDemo::new(), args);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Ray queries are native-only (#110); no wasm entry point.
}
