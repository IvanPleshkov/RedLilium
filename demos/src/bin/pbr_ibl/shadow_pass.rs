//! Ray-traced shadow-mask pass (#110 phase 2).
//!
//! Produces a screen-space **visibility mask** (R8Unorm, `r` = key-light
//! visibility) that the deferred resolve pass multiplies into its direct
//! lighting term. For every G-buffer pixel with geometry the pass reconstructs
//! the world position + normal, offsets along the normal, and casts a small
//! cone of `terminate-on-first-hit` ray queries toward the sun (modelled as an
//! area light), averaging them into a **soft** penumbra — hard binary shadows
//! read badly layered over IBL.
//!
//! The scene BLAS is built **straight from the rendered sphere mesh** — its
//! vertex/index buffers opt into `ACCELERATION_STRUCTURE_INPUT` (see
//! [`SphereGrid::create`](crate::sphere_grid::SphereGrid::create)) — so there
//! is no duplicate geometry. The TLAS is rebuilt each frame from the current
//! sphere transforms (they move when the grid spacing changes).
//!
//! The tracing shader is **WGSL with an explicit binding layout**: the Slang
//! reflection path has no acceleration-structure binding type, so a ray query
//! can only be expressed through WGSL today. The resolve shader stays Slang and
//! just samples the resulting mask as an ordinary texture.
//!
//! Ray queries are Vulkan-only (`DeviceCapabilities::ray_query`). On any other
//! device [`ShadowPass`] holds no ray-tracing resources and [`record`] simply
//! clears the mask to white (everything lit), so the resolve shader's shadow
//! term becomes a no-op.
//!
//! [`record`]: ShadowPass::record

use std::sync::Arc;

use redlilium_core::mesh::{CpuMesh, IndexFormat, VertexLayout};
use redlilium_graphics::{
    AccelerationStructureBuildPass, BindingGroupDescriptor, BindingLayout, BindingLayoutEntry,
    BindingType, Blas, BlasDescriptor, BlasTriangles, BufferUsage, ColorAttachment, CpuSampler,
    DrawCommand, GraphicsDevice, GraphicsPass, Material, MaterialDescriptor, MaterialInstance,
    Mesh, RenderGraph, RenderTargetConfig, RingBuffer, Sampler, ShaderSource, ShaderStage,
    ShaderStageFlags, Texture, TextureDescriptor, TextureFormat, TextureUsage, Tlas,
    TlasDescriptor, TlasInstance, TransferConfig, TransferOperation, TransferPass,
};

use crate::gbuffer::GBuffer;
use crate::uniforms::{SUN_DIR_TO_LIGHT, ShadowUniforms, SphereInstance};

/// WGSL ray-query shadow-mask shader. naga compiles ray queries to SPIR-V
/// (`SPV_KHR_ray_query`) on the Vulkan backend.
const SHADOW_SHADER_WGSL: &str = r#"
struct Shadow {
    // xyz = normalized direction toward the key light
    light_dir: vec4<f32>,
}

@group(0) @binding(0) var<uniform> shadow: Shadow;
@group(0) @binding(1) var tlas: acceleration_structure;
@group(0) @binding(2) var gbuffer_position: texture_2d<f32>;
@group(0) @binding(3) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(4) var gbuffer_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.clip = vec4<f32>(pos.xy, 0.0, 1.0);
    // Match the resolve pass's fullscreen-triangle UV mapping (y flipped) so
    // the same G-buffer texels are sampled.
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}

// Soft-shadow controls. SAMPLES rays are cast in a cone toward the light
// (an area light of angular radius CONE_RADIUS) and averaged, so the penumbra
// is soft instead of a hard binary edge — which reads far better layered over
// IBL than a crisp ray-traced shadow.
const SAMPLES: i32 = 16;
const CONE_RADIUS: f32 = 0.07;

// Cheap per-pixel hash for rotating the sample pattern, so the soft edge
// dithers instead of banding.
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let position_sample = textureSampleLevel(gbuffer_position, gbuffer_sampler, in.uv, 0.0);
    let normal_sample = textureSampleLevel(gbuffer_normal, gbuffer_sampler, in.uv, 0.0);

    // Background pixels: the G-buffer normal target is cleared to 0.5 (decodes
    // to a zero-length vector). Skip the ray and report "lit".
    let n_raw = normal_sample.xyz * 2.0 - 1.0;
    if (dot(n_raw, n_raw) < 0.01) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }

    let normal = normalize(n_raw);
    let world_pos = position_sample.xyz;
    let to_light = normalize(shadow.light_dir.xyz);

    // Orthonormal basis around the light direction to spread cone samples in.
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(to_light.y) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, to_light));
    let bitangent = cross(to_light, tangent);

    // Offset along the geometric normal to escape self-intersection with the
    // faceted sphere BLAS. 0x4 = terminate-on-first-hit; geometry is opaque.
    let origin = world_pos + normal * 0.02;
    let rot = hash12(in.clip.xy) * 6.2831853;

    var lit = 0.0;
    for (var i = 0; i < SAMPLES; i = i + 1) {
        // Vogel-disk sample within the cone, rotated per pixel.
        let fi = f32(i);
        let r = sqrt((fi + 0.5) / f32(SAMPLES)) * CONE_RADIUS;
        let theta = fi * 2.399963229 + rot;
        let disk = vec2<f32>(cos(theta), sin(theta)) * r;
        let dir = normalize(to_light + tangent * disk.x + bitangent * disk.y);

        var rq: ray_query;
        rayQueryInitialize(&rq, tlas, RayDesc(0x4u, 0xFFu, 0.02, 1000.0, origin, dir));
        while (rayQueryProceed(&rq)) {}
        if (rayQueryGetCommittedIntersection(&rq).kind == RAY_QUERY_INTERSECTION_NONE) {
            lit = lit + 1.0;
        }
    }

    let visibility = lit / f32(SAMPLES);
    return vec4<f32>(visibility, 0.0, 0.0, 1.0);
}
"#;

/// Ray-tracing resources for the shadow pass; absent on devices without ray
/// queries.
struct ShadowRt {
    blas: Arc<Blas>,
    tlas: Arc<Tlas>,
    /// Stable material + explicit binding layout (WGSL has no reflection).
    material: Arc<Material>,
    binding_layout: Arc<BindingLayout>,
    /// Kept alive so the binding group can be rebuilt on resize.
    sampler: Arc<Sampler>,
    /// Fullscreen triangle covering the screen.
    fullscreen_mesh: Arc<Mesh>,
    /// Constant light-direction uniform: written once, reused every frame.
    uniform_ring: RingBuffer,
    uniform_offset: u32,
    /// Rebuilt on resize — it references the (resized) G-buffer textures.
    material_instance: Arc<MaterialInstance>,
    /// Fullscreen-mesh upload, drained into the first frame's graph.
    pending_uploads: Vec<TransferOperation>,
    /// Whether the BLAS build has been recorded (once; geometry is static).
    blas_built: bool,
    /// This frame's per-sphere model matrices (column-major), one TLAS instance
    /// each.
    instance_models: Vec<[[f32; 4]; 4]>,
}

/// Ray-traced shadow-mask pass.
pub struct ShadowPass {
    /// Screen-sized visibility mask (R8Unorm); sampled by the resolve pass.
    mask: Arc<Texture>,
    /// Ray-tracing resources; `None` without ray-query support.
    rt: Option<ShadowRt>,
}

impl ShadowPass {
    /// Create the shadow pass. `supported` gates the ray-tracing path — pass
    /// `device.capabilities().ray_query`. `sphere_mesh` must have been created
    /// with acceleration-structure-input usage (see
    /// [`SphereGrid::create`](crate::sphere_grid::SphereGrid::create)).
    pub fn create(
        device: &Arc<GraphicsDevice>,
        gbuffer: &GBuffer,
        sphere_mesh: &Arc<Mesh>,
        width: u32,
        height: u32,
        supported: bool,
    ) -> Self {
        let mask = Self::create_mask(device, width, height);

        if !supported {
            return Self { mask, rt: None };
        }

        // --- BLAS over the rendered sphere mesh (position_normal_uv: 32-byte
        // stride, position at offset 0). The build only reads positions, so the
        // interleaved normal/uv in the stride are simply skipped.
        let vertex_buffer = sphere_mesh
            .vertex_buffer(0)
            .expect("sphere mesh has a vertex buffer")
            .clone();
        let index_buffer = sphere_mesh
            .index_buffer()
            .expect("sphere mesh is indexed")
            .clone();
        let geometry = BlasTriangles {
            vertex_buffer,
            vertex_offset: 0,
            vertex_stride: 32,
            vertex_count: sphere_mesh.vertex_count(),
            index_buffer: Some(index_buffer),
            index_offset: 0,
            index_format: IndexFormat::Uint32,
            triangle_count: sphere_mesh.index_count() / 3,
            opaque: true,
        };
        let blas = device
            .create_blas(
                &BlasDescriptor::new(vec![geometry])
                    .with_label("sphere shadow blas")
                    .with_compaction(),
            )
            .expect("create sphere BLAS");

        let tlas = device
            .create_tlas(
                &TlasDescriptor::new(crate::GRID_SIZE as u32 * crate::GRID_SIZE as u32)
                    .with_label("shadow tlas"),
            )
            .expect("create shadow TLAS");

        // --- Fullscreen triangle (position-only), uploaded through the graph.
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
            .with_label("shadow fullscreen triangle");
        let (fullscreen_mesh, pending_uploads) = device
            .create_mesh_deferred(&fullscreen)
            .expect("create shadow fullscreen mesh");

        // --- WGSL ray-query material with an explicit binding layout.
        let binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::DynamicUniformBuffer)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::AccelerationStructure)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(2, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(3, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(4, BindingType::Sampler)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_label("shadow bindings"),
        );
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        SHADOW_SHADER_WGSL.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        SHADOW_SHADER_WGSL.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_vertex_layout(VertexLayout::position_only())
                    .with_binding_layout(binding_layout.clone())
                    .with_color_format(TextureFormat::R8Unorm)
                    .with_label("shadow material"),
            )
            .expect("create shadow material");

        let sampler = device
            .create_sampler_from_cpu(&CpuSampler::nearest().with_name("shadow_gbuffer_sampler"))
            .expect("create shadow sampler");

        // Constant light-direction uniform: written once at offset 0.
        let mut uniform_ring = RingBuffer::new(
            device,
            1 << 12,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "shadow_uniform_ring",
        )
        .expect("create shadow uniform ring");
        let uniforms = ShadowUniforms {
            light_dir: [
                SUN_DIR_TO_LIGHT[0],
                SUN_DIR_TO_LIGHT[1],
                SUN_DIR_TO_LIGHT[2],
                0.0,
            ],
        };
        let bytes = bytemuck::bytes_of(&uniforms);
        let alloc = uniform_ring
            .allocate(bytes.len() as u64)
            .expect("shadow uniform ring slot");
        uniform_ring
            .write(&alloc, bytes)
            .expect("write shadow uniforms");
        let uniform_offset = alloc.offset as u32;

        let material_instance = Self::make_instance(
            device,
            &material,
            &binding_layout,
            &uniform_ring,
            &tlas,
            gbuffer,
            &sampler,
        );

        Self {
            mask,
            rt: Some(ShadowRt {
                blas,
                tlas,
                material,
                binding_layout,
                sampler,
                fullscreen_mesh,
                uniform_ring,
                uniform_offset,
                material_instance,
                pending_uploads,
                blas_built: false,
                instance_models: Vec::new(),
            }),
        }
    }

    /// The visibility mask texture (bind it into the resolve pass).
    pub fn mask(&self) -> &Arc<Texture> {
        &self.mask
    }

    /// Recreate the screen-sized mask and rebind the (resized) G-buffer
    /// textures. The ray-tracing geometry (BLAS/TLAS/material) is preserved.
    pub fn resize(
        &mut self,
        device: &Arc<GraphicsDevice>,
        gbuffer: &GBuffer,
        width: u32,
        height: u32,
    ) {
        self.mask = Self::create_mask(device, width, height);
        if let Some(rt) = &mut self.rt {
            rt.material_instance = Self::make_instance(
                device,
                &rt.material,
                &rt.binding_layout,
                &rt.uniform_ring,
                &rt.tlas,
                gbuffer,
                &rt.sampler,
            );
        }
    }

    /// Record this frame's per-sphere model matrices for the TLAS rebuild. A
    /// no-op without ray-query support.
    pub fn set_instances(&mut self, instances: &[SphereInstance]) {
        if let Some(rt) = &mut self.rt {
            rt.instance_models.clear();
            rt.instance_models.extend(instances.iter().map(|i| i.model));
        }
    }

    /// Add this frame's shadow work to the graph: a one-time BLAS build, the
    /// per-frame TLAS rebuild, and the fullscreen ray-traced mask pass. Without
    /// ray-query support it clears the mask to white (everything lit).
    pub fn record(&mut self, graph: &mut RenderGraph, device: &Arc<GraphicsDevice>) {
        let mask = self.mask.clone();
        let Some(rt) = &mut self.rt else {
            let mut pass = GraphicsPass::new("shadow_mask_clear".into());
            pass.set_render_targets(RenderTargetConfig::new().with_color(
                ColorAttachment::from_texture(mask).with_clear_color(1.0, 1.0, 1.0, 1.0),
            ));
            graph.add_graphics_pass(pass);
            return;
        };

        // First frame: upload the fullscreen mesh and build the BLAS. The build
        // reads the sphere mesh buffers; the graph auto-orders it after their
        // upload transfer (added earlier this frame by SphereGrid).
        if !rt.blas_built {
            let ops = std::mem::take(&mut rt.pending_uploads);
            if !ops.is_empty() {
                let mut pass = TransferPass::new("shadow_geometry_upload".into());
                pass.set_transfer_config(TransferConfig::new().with_operations(ops));
                graph.add_transfer_pass(pass);
            }
            let mut build = AccelerationStructureBuildPass::new("sphere blas build".into());
            build.add_blas_build(Arc::clone(&rt.blas));
            graph.add_acceleration_structure_build_pass(build);
            rt.blas_built = true;
        }

        // Every frame: rebuild the TLAS from the current sphere transforms.
        let instances: Vec<TlasInstance> = rt
            .instance_models
            .iter()
            .map(|m| TlasInstance::new(Arc::clone(&rt.blas)).with_transform(to_tlas_transform(m)))
            .collect();
        rt.tlas
            .write_instances(&instances)
            .expect("write shadow TLAS instances");
        let tlas_handle = {
            let mut pass = AccelerationStructureBuildPass::new("shadow tlas build".into());
            pass.add_tlas_build(Arc::clone(&rt.tlas));
            graph.add_acceleration_structure_build_pass(pass)
        };

        // Drive transparent BLAS compaction (#110 phase 3).
        graph.flush_acceleration_structure_compaction(device);

        // Fullscreen ray-traced mask pass. It samples the G-buffer (written by
        // the gbuffer pass) and reads the TLAS, so both dependencies are
        // resolved automatically / via the explicit TLAS edge below.
        let mut pass = GraphicsPass::new("shadow_mask".into());
        pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_texture(mask).with_clear_color(1.0, 1.0, 1.0, 1.0),
            ),
        );
        pass.add_draw_command(
            DrawCommand::new(
                Arc::clone(&rt.fullscreen_mesh),
                Arc::clone(&rt.material_instance),
            )
            .with_dynamic_offsets(vec![vec![rt.uniform_offset]]),
        );
        let shadow_handle = graph.add_graphics_pass(pass);
        graph.add_dependency(shadow_handle, tlas_handle);
    }

    /// Create the screen-sized R8Unorm visibility mask.
    fn create_mask(device: &Arc<GraphicsDevice>, width: u32, height: u32) -> Arc<Texture> {
        device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::R8Unorm,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("shadow_mask"),
            )
            .expect("create shadow mask texture")
    }

    /// Build the shadow binding group (uniform + TLAS + G-buffer + sampler).
    #[allow(clippy::too_many_arguments)]
    fn make_instance(
        device: &Arc<GraphicsDevice>,
        material: &Arc<Material>,
        binding_layout: &Arc<BindingLayout>,
        uniform_ring: &RingBuffer,
        tlas: &Arc<Tlas>,
        gbuffer: &GBuffer,
        sampler: &Arc<Sampler>,
    ) -> Arc<MaterialInstance> {
        let binding_group = device
            .create_binding_group(
                binding_layout.clone(),
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        uniform_ring.buffer().clone(),
                        0,
                        std::mem::size_of::<ShadowUniforms>() as u64,
                    )
                    .with_acceleration_structure(1, tlas.clone())
                    .with_texture(2, gbuffer.position_roughness.clone())
                    .with_texture(3, gbuffer.normal_metallic.clone())
                    .with_sampler(4, sampler.clone())
                    .with_label("shadow binding group"),
            )
            .expect("create shadow binding group");
        Arc::new(MaterialInstance::new(material.clone()).with_binding_group(binding_group))
    }
}

/// Row-major 3×4 `VkTransformMatrixKHR` from a column-major `Mat4` array.
fn to_tlas_transform(c: &[[f32; 4]; 4]) -> [[f32; 4]; 3] {
    [
        [c[0][0], c[1][0], c[2][0], c[3][0]],
        [c[0][1], c[1][1], c[2][1], c[3][1]],
        [c[0][2], c[1][2], c[2][2], c[3][2]],
    ]
}
