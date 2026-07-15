//! # Bindless Demo (#117)
//!
//! The bindless texture heap in action: a grid of quads where **every
//! instance picks its texture by a plain integer** — one draw call, one
//! material, no per-texture descriptor sets anywhere. 48 procedurally
//! generated textures (checkers, rings, stripes, dots) plus two samplers
//! (linear / nearest, alternating per instance) live in the device-owned
//! update-after-bind heap; the fragment shader samples
//! `bindless_textures[NonUniformResourceIndex(index)]`.
//!
//! Every ~1.5 seconds one quad's texture is **churned**: a freshly generated
//! texture registers into the heap, the instance data switches to the new
//! slot through the frame graph, and the old slot unregisters — exercising
//! the fence-deferred slot recycling while frames are in flight.
//!
//! Requires `DeviceCapabilities::bindless`; renders a blank screen with a
//! notice on devices without it.

use std::sync::Arc;

use redlilium_app::{App, AppArgs, AppContext, AppHandler, DefaultAppArgs, DrawContext};
use redlilium_core::mesh::generators;
use redlilium_graphics::{
    BindingGroupDescriptor, BindingLayout, BindingLayoutEntry, BindingType, BufferDescriptor,
    BufferUsage, ColorAttachment, DrawCommand, FilterMode, FrameSchedule, GraphicsPass,
    MaterialDescriptor, MaterialInstance, Mesh, RenderTargetConfig, RingBuffer, SamplerDescriptor,
    ShaderSource, ShaderStage, ShaderStageFlags, TextureDescriptor, TextureFormat, TextureUsage,
    TransferConfig, TransferOperation, TransferPass,
};

const BINDLESS_SHADER_SLANG: &str = include_str!("../../shaders/bindless.slang");

const TEX_SIZE: u32 = 64;
const GRID_COLS: usize = 8;
const GRID_ROWS: usize = 6;
const INSTANCE_COUNT: usize = GRID_COLS * GRID_ROWS;
/// Frames between texture churns (register new / retire old slot).
const CHURN_PERIOD: u64 = 90;

// === GPU Data (layouts match bindless.slang) ===

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstanceData {
    /// xy = NDC center, z = NDC half-height (x is aspect-corrected in the
    /// shader), w unused.
    center_size: [f32; 4],
    texture_index: u32,
    sampler_index: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    /// x = time in seconds, y = aspect ratio, zw unused.
    time_aspect: [f32; 4],
}

// === Procedural Textures ===

/// A small palette + pattern family keyed by index, so every generated
/// texture is visually distinct and the churned ones are obviously new.
fn generate_pattern(index: u32) -> Vec<u8> {
    let h = index.wrapping_mul(2654435761);
    let base = [
        64 + (h & 0x7F) as u8,
        64 + ((h >> 8) & 0x7F) as u8,
        64 + ((h >> 16) & 0x7F) as u8,
    ];
    let accent = [255 - base[0], 255 - base[1], 255 - base[2]];

    let mut data = Vec::with_capacity((TEX_SIZE * TEX_SIZE * 4) as usize);
    for y in 0..TEX_SIZE {
        for x in 0..TEX_SIZE {
            let on = match index % 4 {
                // Checkerboard.
                0 => ((x / 8) + (y / 8)) % 2 == 0,
                // Concentric rings.
                1 => {
                    let dx = x as f32 - TEX_SIZE as f32 / 2.0;
                    let dy = y as f32 - TEX_SIZE as f32 / 2.0;
                    (((dx * dx + dy * dy).sqrt() / 6.0) as u32).is_multiple_of(2)
                }
                // Diagonal stripes.
                2 => ((x + y) / 8) % 2 == 0,
                // Dots.
                _ => (x % 16 < 8) && (y % 16 < 8),
            };
            let c = if on { accent } else { base };
            data.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    data
}

// === Demo Application ===

struct BindlessDemo {
    material_instance: Option<Arc<MaterialInstance>>,
    quad_mesh: Option<Arc<Mesh>>,
    uniform_ring: Option<RingBuffer>,
    uniform_offset: u32,
    instance_buffer: Option<Arc<redlilium_graphics::Buffer>>,
    /// Heap slot of each instance's texture (index = instance).
    texture_slots: Vec<u32>,
    pending_uploads: Vec<TransferOperation>,
    /// Which instance the next churn replaces, and how many happened.
    churn_cursor: usize,
    churn_count: u32,
    frame: u64,
    supported: bool,
    time: f32,
}

impl BindlessDemo {
    fn new() -> Self {
        Self {
            material_instance: None,
            quad_mesh: None,
            uniform_ring: None,
            uniform_offset: 0,
            instance_buffer: None,
            texture_slots: Vec::new(),
            pending_uploads: Vec::new(),
            churn_cursor: 0,
            churn_count: 0,
            frame: 0,
            supported: false,
            time: 0.0,
        }
    }

    /// Create a procedural texture, queue its upload, register it in the
    /// heap, and return its slot.
    fn make_registered_texture(&mut self, ctx: &AppContext, pattern: u32, label: &str) -> u32 {
        let device = ctx.device();
        let texture = device
            .create_texture(
                &TextureDescriptor::new_2d(
                    TEX_SIZE,
                    TEX_SIZE,
                    TextureFormat::Rgba8UnormSrgb,
                    TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                )
                .with_label(label),
            )
            .expect("create bindless texture");
        self.pending_uploads.push(
            TransferOperation::upload_texture_data(
                device,
                texture.clone(),
                &generate_pattern(pattern),
            )
            .expect("stage texture upload"),
        );
        device
            .bindless_register_texture(&texture)
            .expect("register bindless texture")
    }

    fn create_gpu_resources(&mut self, ctx: &mut AppContext) {
        // 48 unique textures, one per instance, all in the heap.
        for i in 0..INSTANCE_COUNT {
            let slot = self.make_registered_texture(ctx, i as u32, &format!("bindless tex {i}"));
            self.texture_slots.push(slot);
        }

        let device = ctx.device();

        // Two samplers, alternating per instance: linear vs nearest.
        let mut sampler_slots = Vec::new();
        for (label, filter) in [
            ("linear", FilterMode::Linear),
            ("nearest", FilterMode::Nearest),
        ] {
            let sampler = device
                .create_sampler(&SamplerDescriptor {
                    label: Some(format!("bindless {label}")),
                    mag_filter: filter,
                    min_filter: filter,
                    ..Default::default()
                })
                .expect("create sampler");
            sampler_slots.push(
                device
                    .bindless_register_sampler(&sampler)
                    .expect("register bindless sampler"),
            );
        }

        // Instance grid (NDC): GRID_COLS × GRID_ROWS quads.
        let instances: Vec<GpuInstanceData> = (0..INSTANCE_COUNT)
            .map(|i| {
                let col = i % GRID_COLS;
                let row = i / GRID_COLS;
                let x = -0.9 + 1.8 * (col as f32 + 0.5) / GRID_COLS as f32;
                let y = -0.9 + 1.8 * (row as f32 + 0.5) / GRID_ROWS as f32;
                GpuInstanceData {
                    center_size: [x, y, 0.115, 0.0],
                    texture_index: self.texture_slots[i],
                    sampler_index: sampler_slots[i % sampler_slots.len()],
                    _pad: [0; 2],
                }
            })
            .collect();
        let instance_buffer = device
            .create_buffer(
                &BufferDescriptor::new(
                    (instances.len() * size_of::<GpuInstanceData>()) as u64,
                    BufferUsage::STORAGE | BufferUsage::COPY_DST,
                )
                .with_label("bindless instances"),
            )
            .expect("create instance buffer");
        self.pending_uploads.push(TransferOperation::write_buffer(
            instance_buffer.clone(),
            0,
            bytemuck::cast_slice(&instances).to_vec().into(),
        ));

        // Material: explicit binding layouts (reflection cannot express the
        // heap's runtime arrays) — set 0 = data, set 1 = the device's heap
        // layout Arc, so material and heap group share it by pointer.
        let data_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::DynamicUniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX | ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::StorageBufferReadOnly)
                        .with_visibility(ShaderStageFlags::VERTEX),
                )
                .with_label("bindless demo data"),
        );
        let heap_layout = device.bindless_heap_layout().expect("bindless heap layout");

        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        BINDLESS_SHADER_SLANG.as_bytes().to_vec(),
                        "vs_main",
                        vec![],
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        BINDLESS_SHADER_SLANG.as_bytes().to_vec(),
                        "fs_main",
                        vec![],
                    ))
                    .with_vertex_layout(redlilium_core::mesh::VertexLayout::position_uv())
                    .with_binding_layout(data_layout.clone())
                    .with_binding_layout(heap_layout)
                    .with_color_format(ctx.surface_format())
                    .with_label("bindless material"),
            )
            .expect("create bindless material");

        let uniform_ring = RingBuffer::new(
            device,
            1 << 16,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "bindless scene ring",
        )
        .expect("create uniform ring");

        let data_group = device
            .create_binding_group(
                data_layout,
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        uniform_ring.buffer().clone(),
                        0,
                        size_of::<SceneUniforms>() as u64,
                    )
                    .with_buffer(1, instance_buffer.clone())
                    .with_label("bindless demo data"),
            )
            .expect("create data group");
        let heap_group = device.bindless_heap_group().expect("bindless heap group");

        self.material_instance = Some(Arc::new(
            MaterialInstance::new(material)
                .with_binding_group(data_group)
                .with_binding_group(heap_group),
        ));
        self.uniform_ring = Some(uniform_ring);
        self.instance_buffer = Some(instance_buffer);

        // Fullscreen-independent unit quad; instances place and scale it.
        let (mesh, mesh_ops) = device
            .create_mesh_deferred(&generators::generate_quad(1.0, 1.0))
            .expect("create quad mesh");
        self.pending_uploads.extend(mesh_ops);
        self.quad_mesh = Some(mesh);

        log::info!(
            "Bindless GPU resources created: {INSTANCE_COUNT} textures + 2 samplers in the heap"
        );
    }

    /// Replace one instance's texture: register a fresh one, point the
    /// instance at the new slot through the frame graph, retire the old
    /// slot (recycled after the in-flight window — this is the churn the
    /// deferred free exists for).
    fn churn_texture(&mut self, ctx: &AppContext) {
        let instance = self.churn_cursor;
        self.churn_cursor = (self.churn_cursor + 1) % INSTANCE_COUNT;
        self.churn_count += 1;

        let pattern = INSTANCE_COUNT as u32 + self.churn_count;
        let new_slot = self.make_registered_texture(
            ctx,
            pattern,
            &format!("bindless churn {}", self.churn_count),
        );

        let old_slot = std::mem::replace(&mut self.texture_slots[instance], new_slot);
        let offset = (instance * size_of::<GpuInstanceData>()
            + std::mem::offset_of!(GpuInstanceData, texture_index)) as u64;
        self.pending_uploads.push(TransferOperation::write_buffer(
            self.instance_buffer.as_ref().unwrap().clone(),
            offset,
            new_slot.to_le_bytes().to_vec().into(),
        ));

        ctx.device()
            .bindless_unregister_texture(old_slot)
            .expect("unregister bindless texture");
    }

    fn update_uniforms(&mut self, ctx: &AppContext) {
        let uniforms = SceneUniforms {
            time_aspect: [self.time, ctx.aspect_ratio(), 0.0, 0.0],
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

impl AppHandler for BindlessDemo {
    fn on_init(&mut self, ctx: &mut AppContext) {
        log::info!("Initializing Bindless Demo (#117)");

        self.supported = ctx.device().capabilities().bindless;
        if !self.supported {
            log::warn!(
                "This device does not support the bindless heap \
                 (DeviceCapabilities::bindless is false) — the demo renders a blank screen. \
                 Bindless needs the Vulkan backend with the descriptor-indexing feature bits."
            );
            return;
        }
        log::info!(
            "Bindless supported on {} — one texture churns every {CHURN_PERIOD} frames",
            ctx.device().name()
        );

        self.create_gpu_resources(ctx);
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        if !self.supported {
            return true;
        }
        self.time += 1.0 / 60.0;
        self.frame += 1;
        if self.frame.is_multiple_of(CHURN_PERIOD) {
            self.churn_texture(ctx);
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

        // Uploads: initial geometry/textures on the first frame, churned
        // textures + instance updates afterwards.
        let mut upload_handle = None;
        if !self.pending_uploads.is_empty() {
            let ops = std::mem::take(&mut self.pending_uploads);
            let mut transfer_pass = TransferPass::new("bindless uploads".into());
            transfer_pass.set_transfer_config(TransferConfig::new().with_operations(ops));
            upload_handle = Some(graph.add_transfer_pass(transfer_pass));
        }

        let mut render_pass = GraphicsPass::new("bindless grid".into());
        render_pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_surface(ctx.swapchain_texture())
                    .with_clear_color(0.05, 0.06, 0.09, 1.0),
            ),
        );
        render_pass.add_draw_command(
            DrawCommand::new(
                Arc::clone(self.quad_mesh.as_ref().unwrap()),
                Arc::clone(self.material_instance.as_ref().unwrap()),
            )
            .with_instance_count(INSTANCE_COUNT as u32)
            .with_dynamic_offsets(vec![vec![self.uniform_offset]]),
        );
        let render_handle = graph.add_graphics_pass(render_pass);
        if let Some(upload) = upload_handle {
            graph.add_dependency(render_handle, upload);
        }

        ctx.render(graph)
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down Bindless Demo");
    }
}

// === Entry Point ===

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args = DefaultAppArgs::parse().with_title_str("Bindless Demo (#117)");
    App::run(BindlessDemo::new(), args);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // The bindless heap is native-only (#117); no wasm entry point.
}
