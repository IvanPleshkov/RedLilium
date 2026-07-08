//! Egui renderer for RedLilium graphics.
//!
//! This module handles GPU resources and rendering for egui.

use std::collections::HashMap;
use std::sync::Arc;

use egui::epaint::{ImageDelta, Primitive, Vertex};
use egui::{ClippedPrimitive, TextureId, TexturesDelta};

use crate::GraphicsDevice;
use crate::graph::{
    ColorAttachment, DrawCommand, GraphicsPass, LoadOp, RenderGraph, RenderTarget,
    RenderTargetConfig, TransferConfig, TransferOperation, TransferPass,
};
use crate::materials::{
    BindingGroupDescriptor, Material, MaterialDescriptor, MaterialInstance, ShaderSource,
    ShaderStage,
};
use crate::mesh::{
    IndexFormat, Mesh, PrimitiveTopology, VertexAttribute, VertexAttributeFormat,
    VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
};
use crate::resources::{RingBuffer, Sampler, Texture};
use crate::shader::EGUI_SHADER_SOURCE;
use crate::types::{
    AddressMode, BufferUsage, FilterMode, SamplerDescriptor, TextureDescriptor, TextureFormat,
    TextureUsage,
};

/// Egui vertex data matching egui's Vertex structure.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EguiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl From<&Vertex> for EguiVertex {
    fn from(v: &Vertex) -> Self {
        Self {
            pos: [v.pos.x, v.pos.y],
            uv: [v.uv.x, v.uv.y],
            color: [
                v.color.r() as f32 / 255.0,
                v.color.g() as f32 / 255.0,
                v.color.b() as f32 / 255.0,
                v.color.a() as f32 / 255.0,
            ],
        }
    }
}

/// Uniform buffer data for egui rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EguiUniforms {
    pub screen_size: [f32; 2],
    pub _padding: [f32; 2],
}

/// CPU-side texture data for handling partial updates.
#[allow(dead_code)]
struct TextureData {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Capacity of the per-frame uniform ring (bytes).
const UNIFORM_RING_CAPACITY: u64 = 1 << 16;

/// Capacity of the per-frame vertex ring (bytes).
const VERTEX_RING_CAPACITY: u64 = 1 << 22;

/// Capacity of the per-frame index ring (bytes).
const INDEX_RING_CAPACITY: u64 = 1 << 21;

/// Manages GPU resources for egui rendering.
pub struct EguiRenderer {
    device: Arc<GraphicsDevice>,
    material: Arc<Material>,
    vertex_layout: Arc<VertexLayout>,
    /// Per-frame uniform ring (screen-size uniform; bound with a dynamic offset).
    uniform_ring: RingBuffer,
    /// Per-frame vertex ring; all egui geometry lives here, addressed via offsets.
    vertex_ring: RingBuffer,
    /// Per-frame index ring; addressed via offsets.
    index_ring: RingBuffer,
    sampler: Arc<Sampler>,
    textures: HashMap<TextureId, Arc<Texture>>,
    /// Eagerly-created group-0 binding (the screen-size uniform, bound at
    /// offset 0 with a per-draw dynamic offset). Built once against the
    /// stable uniform ring buffer and reused every frame.
    uniform_binding: Option<Arc<crate::materials::BindingGroup>>,
    /// Eagerly-created group-1 (texture + sampler) bindings, one per egui
    /// texture id. Built when a texture first appears and reused every frame;
    /// dropped/rebuilt when its texture is replaced or freed.
    texture_bindings: HashMap<TextureId, Arc<crate::materials::BindingGroup>>,
    /// CPU-side texture data for partial update support.
    texture_data: HashMap<TextureId, TextureData>,
    /// Counter for generating unique user texture IDs.
    next_user_texture_id: u64,
    /// Scratch buffer reused for per-primitive vertex conversion.
    scratch_vertices: Vec<EguiVertex>,
    /// Pending atlas uploads, drained into the frame graph by
    /// [`flush_uploads`](Self::flush_uploads). Texture data is uploaded through
    /// the render graph, never synchronously.
    pending_uploads: Vec<TransferOperation>,
}

impl EguiRenderer {
    /// Create a new egui renderer.
    pub fn new(device: Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        // Create vertex layout for egui vertices
        let vertex_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(
                    std::mem::size_of::<EguiVertex>() as u32
                ))
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::Position,
                    format: VertexAttributeFormat::Float2,
                    offset: 0,
                    buffer_index: 0,
                })
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::TexCoord0,
                    format: VertexAttributeFormat::Float2,
                    offset: 8,
                    buffer_index: 0,
                })
                .with_attribute(VertexAttribute {
                    semantic: VertexAttributeSemantic::Color,
                    format: VertexAttributeFormat::Float4,
                    offset: 16,
                    buffer_index: 0,
                })
                .with_label("egui_vertex_layout"),
        );

        // Pass surface-type defines so the shader handles color space correctly
        let mut defines = Vec::new();
        if surface_format.is_hdr() {
            defines.push(("HDR_OUTPUT".to_string(), String::new()));
        } else if surface_format.is_srgb() {
            defines.push(("SRGB_FRAMEBUFFER".to_string(), String::new()));
        }

        // Create material using Slang shader source
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Vertex,
                        EGUI_SHADER_SOURCE.as_bytes().to_vec(),
                        "vs_main",
                        defines.clone(),
                    ))
                    .with_shader(ShaderSource::slang(
                        ShaderStage::Fragment,
                        EGUI_SHADER_SOURCE.as_bytes().to_vec(),
                        "fs_main",
                        defines,
                    ))
                    .with_vertex_layout(vertex_layout.clone())
                    .with_blend_state(crate::materials::BlendState::premultiplied_alpha())
                    .with_color_format(surface_format)
                    // Binding (0, 0) is the screen-size uniform, supplied per-draw
                    // as a dynamic offset into `uniform_ring`.
                    .with_dynamic_uniform(0, 0)
                    .with_label("egui_material"),
            )
            .expect("Failed to create egui material");

        // Per-frame rings. egui geometry and uniforms are written into a fresh
        // ring slot each frame, so they never race a frame still in flight.
        let uniform_ring = RingBuffer::new(
            &device,
            UNIFORM_RING_CAPACITY,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "egui_uniform_ring",
        )
        .expect("Failed to create egui uniform ring");

        let vertex_ring = RingBuffer::new(
            &device,
            VERTEX_RING_CAPACITY,
            BufferUsage::VERTEX | BufferUsage::COPY_DST,
            "egui_vertex_ring",
        )
        .expect("Failed to create egui vertex ring");

        let index_ring = RingBuffer::new(
            &device,
            INDEX_RING_CAPACITY,
            BufferUsage::INDEX | BufferUsage::COPY_DST,
            "egui_index_ring",
        )
        .expect("Failed to create egui index ring");

        // Create sampler
        let sampler = device
            .create_sampler(&SamplerDescriptor {
                label: Some("egui_sampler".into()),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Nearest,
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                ..Default::default()
            })
            .expect("Failed to create egui sampler");

        Self {
            device,
            material,
            vertex_layout,
            uniform_ring,
            vertex_ring,
            index_ring,
            sampler,
            textures: HashMap::new(),
            uniform_binding: None,
            texture_bindings: HashMap::new(),
            texture_data: HashMap::new(),
            next_user_texture_id: 0,
            scratch_vertices: Vec::new(),
            pending_uploads: Vec::new(),
        }
    }

    /// Flush queued atlas uploads into the current frame's render graph.
    ///
    /// Call once per frame while building the frame graph (before or alongside
    /// the egui pass). The graph compiler orders the upload before the egui draw
    /// that samples the atlas, so updates take effect the same frame.
    pub fn flush_uploads(&mut self, graph: &mut RenderGraph) {
        if self.pending_uploads.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.pending_uploads);
        let mut pass = TransferPass::new("egui_texture_uploads".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(ops));
        graph.add_transfer_pass(pass);
    }

    /// Process texture updates from egui.
    pub fn update_textures(&mut self, textures_delta: &TexturesDelta) {
        // Free textures that are no longer needed
        for id in &textures_delta.free {
            self.textures.remove(id);
            self.texture_data.remove(id);
            // Drop the cached binding group so it can't outlive its texture.
            self.texture_bindings.remove(id);
        }

        // Set or update textures
        for (id, delta) in &textures_delta.set {
            // A replaced/updated texture may swap the underlying `Arc<Texture>`;
            // invalidate the cached binding so `create_graphics_pass` rebuilds
            // it against the current texture.
            self.texture_bindings.remove(id);
            self.set_texture(*id, delta);
        }
    }

    /// Set or update a texture.
    fn set_texture(&mut self, id: TextureId, delta: &ImageDelta) {
        let region_width = delta.image.width() as u32;
        let region_height = delta.image.height() as u32;

        // Convert image data to RGBA8
        let new_pixels: Vec<u8> = match &delta.image {
            egui::ImageData::Color(image) => {
                image.pixels.iter().flat_map(|c| c.to_array()).collect()
            }
        };

        if let Some(pos) = delta.pos {
            // Partial update - update the CPU-side data and re-upload
            if let Some(data) = self.texture_data.get_mut(&id) {
                let start_x = pos[0] as u32;
                let start_y = pos[1] as u32;

                // Copy the new pixels into the correct region of the stored data
                for y in 0..region_height {
                    for x in 0..region_width {
                        let src_idx = ((y * region_width + x) * 4) as usize;
                        let dst_x = start_x + x;
                        let dst_y = start_y + y;
                        let dst_idx = ((dst_y * data.width + dst_x) * 4) as usize;

                        if dst_idx + 4 <= data.pixels.len() && src_idx + 4 <= new_pixels.len() {
                            data.pixels[dst_idx..dst_idx + 4]
                                .copy_from_slice(&new_pixels[src_idx..src_idx + 4]);
                        }
                    }
                }

                // Re-upload the full texture through the frame graph.
                if let Some(texture) = self.textures.get(&id).cloned() {
                    let op =
                        TransferOperation::upload_texture_data(&self.device, texture, &data.pixels)
                            .expect("Failed to stage egui texture upload");
                    self.pending_uploads.push(op);
                }
                return;
            }
            // If we don't have the texture data, fall through to create a new texture
            // This shouldn't happen in normal operation
            log::warn!(
                "Partial texture update for unknown texture {:?}, creating new",
                id
            );
        }

        // Full update - create or recreate texture
        let texture = self
            .device
            .create_texture(
                &TextureDescriptor::new_2d(
                    region_width,
                    region_height,
                    TextureFormat::Rgba8UnormSrgb,
                    TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
                )
                .with_label(format!("egui_texture_{:?}", id)),
            )
            .expect("Failed to create egui texture");

        // Upload pixel data through the frame graph.
        let op =
            TransferOperation::upload_texture_data(&self.device, Arc::clone(&texture), &new_pixels)
                .expect("Failed to stage egui texture upload");
        self.pending_uploads.push(op);

        // Store CPU-side data for future partial updates
        self.texture_data.insert(
            id,
            TextureData {
                width: region_width,
                height: region_height,
                pixels: new_pixels,
            },
        );

        self.textures.insert(id, texture);
    }

    /// Register a user-managed texture with egui.
    ///
    /// This allows external textures (such as render targets, offscreen buffers,
    /// or any GPU texture) to be displayed in egui UI elements like `ui.image()`.
    ///
    /// # Arguments
    ///
    /// * `texture` - The GPU texture to register
    ///
    /// # Returns
    ///
    /// A `TextureId` that can be used with egui's image widgets.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let texture_id = renderer.register_user_texture(my_render_target);
    /// // In egui update:
    /// ui.image(egui::load::SizedTexture::new(texture_id, [256.0, 256.0]));
    /// ```
    pub fn register_user_texture(&mut self, texture: Arc<Texture>) -> TextureId {
        let id = TextureId::User(self.next_user_texture_id);
        self.next_user_texture_id += 1;
        self.textures.insert(id, texture);
        id
    }

    /// Update a previously registered user texture.
    ///
    /// This is useful when the underlying texture has been recreated (e.g., on resize).
    ///
    /// # Arguments
    ///
    /// * `id` - The texture ID returned from `register_user_texture`
    /// * `texture` - The new GPU texture
    pub fn update_user_texture(&mut self, id: TextureId, texture: Arc<Texture>) {
        if matches!(id, TextureId::User(_)) {
            // The binding group is eagerly built once per texture id and cached
            // (#40). When the underlying texture is swapped for a new `Arc`
            // (e.g. the scene target recreated on resize), the cached binding
            // still points at the old texture — invalidate it so the next draw
            // rebuilds against the current one. Same-`Arc` re-registration (the
            // common per-frame call) leaves the cache intact, keeping the
            // zero-per-draw descriptor win.
            let changed = self
                .textures
                .get(&id)
                .is_none_or(|existing| !Arc::ptr_eq(existing, &texture));
            if changed {
                self.texture_bindings.remove(&id);
            }
            self.textures.insert(id, texture);
        } else {
            log::warn!("Attempted to update non-user texture {:?}", id);
        }
    }

    /// Unregister a user-managed texture.
    ///
    /// The texture will no longer be available for rendering in egui.
    ///
    /// # Arguments
    ///
    /// * `id` - The texture ID returned from `register_user_texture`
    pub fn unregister_user_texture(&mut self, id: TextureId) {
        if matches!(id, TextureId::User(_)) {
            self.textures.remove(&id);
        } else {
            log::warn!("Attempted to unregister non-user texture {:?}", id);
        }
    }

    /// Push one primitive's geometry into the vertex/index rings.
    ///
    /// Writes `verts` and `indices` into fresh ring slots and returns their
    /// byte offsets `(vertex_offset, index_offset)`. Resets the relevant ring
    /// (recycling its space) if the current frame's geometry would overflow it.
    fn push_geometry(
        vring: &mut RingBuffer,
        iring: &mut RingBuffer,
        verts: &[EguiVertex],
        indices: &[u32],
    ) -> (u64, u64) {
        let vbytes: &[u8] = bytemuck::cast_slice(verts);
        let v_alloc = vring.allocate(vbytes.len() as u64).unwrap_or_else(|| {
            vring.reset();
            vring
                .allocate(vbytes.len() as u64)
                .expect("egui vertex ring too small")
        });
        vring
            .write(&v_alloc, vbytes)
            .expect("Failed to write egui vertex ring");

        let ibytes: &[u8] = bytemuck::cast_slice(indices);
        let i_alloc = iring.allocate(ibytes.len() as u64).unwrap_or_else(|| {
            iring.reset();
            iring
                .allocate(ibytes.len() as u64)
                .expect("egui index ring too small")
        });
        iring
            .write(&i_alloc, ibytes)
            .expect("Failed to write egui index ring");

        (v_alloc.offset, i_alloc.offset)
    }

    /// Create a graphics pass for rendering egui primitives.
    ///
    /// All per-frame geometry and the screen-size uniform are written into
    /// per-frame ring buffers, addressed via per-draw offsets — race-free across
    /// frames in flight.
    ///
    /// # Arguments
    ///
    /// * `primitives` - The tessellated egui primitives to render
    /// * `render_target` - The render target to render to (surface or texture)
    /// * `screen_width` - Screen width in physical pixels
    /// * `screen_height` - Screen height in physical pixels
    /// * `pixels_per_point` - DPI scale factor for converting points to pixels
    pub fn create_graphics_pass(
        &mut self,
        primitives: &[ClippedPrimitive],
        render_target: &RenderTarget,
        screen_width: u32,
        screen_height: u32,
        pixels_per_point: f32,
    ) -> GraphicsPass {
        let mut pass = GraphicsPass::new("egui".into());

        // Set render target (draw on top of existing content)
        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(ColorAttachment::new(render_target.clone()).with_load_op(LoadOp::Load)),
        );

        // Write this frame's screen-size uniform once into the uniform ring.
        // egui outputs vertices in POINTS, so the shader needs the size in points.
        let uniforms = EguiUniforms {
            screen_size: [
                screen_width as f32 / pixels_per_point,
                screen_height as f32 / pixels_per_point,
            ],
            _padding: [0.0, 0.0],
        };
        let uniform_size = std::mem::size_of::<EguiUniforms>() as u64;
        let uniform_offset = {
            let bytes = bytemuck::bytes_of(&uniforms);
            let alloc = self.uniform_ring.allocate(uniform_size).unwrap_or_else(|| {
                self.uniform_ring.reset();
                self.uniform_ring
                    .allocate(uniform_size)
                    .expect("egui uniform ring too small")
            });
            self.uniform_ring
                .write(&alloc, bytes)
                .expect("Failed to write egui uniform ring");
            alloc.offset
        };

        // Uniform binding: bind the whole element range at offset 0; each draw
        // selects this frame's slot via a dynamic offset. The ring buffer is
        // stable, so this group is built once and reused every frame.
        let uniform_binding = match &self.uniform_binding {
            Some(g) => g.clone(),
            None => {
                let layout = self.material.binding_layouts()[0].clone();
                match self.device.create_binding_group(
                    layout,
                    BindingGroupDescriptor::new().with_buffer_range(
                        0,
                        self.uniform_ring.buffer().clone(),
                        0,
                        uniform_size,
                    ),
                ) {
                    Ok(group) => {
                        self.uniform_binding = Some(group.clone());
                        group
                    }
                    Err(e) => {
                        log::error!("egui: failed to create uniform binding group: {e}");
                        return pass;
                    }
                }
            }
        };

        // Process each primitive
        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in primitives
        {
            match primitive {
                Primitive::Mesh(mesh) => {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }

                    // Get texture for this mesh
                    let texture = match self.textures.get(&mesh.texture_id) {
                        Some(t) => t.clone(),
                        None => {
                            log::warn!("Missing texture {:?}", mesh.texture_id);
                            continue;
                        }
                    };

                    let vertex_count = mesh.vertices.len() as u32;
                    let index_count = mesh.indices.len() as u32;

                    // Convert vertices into the reusable scratch buffer.
                    self.scratch_vertices.clear();
                    self.scratch_vertices
                        .extend(mesh.vertices.iter().map(EguiVertex::from));

                    // Write geometry into the per-frame rings, capturing offsets.
                    let (v_off, i_off) = Self::push_geometry(
                        &mut self.vertex_ring,
                        &mut self.index_ring,
                        &self.scratch_vertices,
                        &mesh.indices,
                    );

                    // Construct a Mesh referencing the shared ring buffers, with
                    // this draw's data addressed via byte offsets.
                    let gpu_mesh = Arc::new(
                        Mesh::new(
                            Arc::clone(&self.device),
                            self.vertex_layout.clone(),
                            PrimitiveTopology::TriangleList,
                            vec![self.vertex_ring.buffer().clone()],
                            vertex_count,
                            Some(self.index_ring.buffer().clone()),
                            Some(IndexFormat::Uint32),
                            index_count,
                            Some("egui_mesh".into()),
                        )
                        .with_buffer_offsets(vec![v_off], i_off),
                    );

                    // Texture binding group (group 1): built once per texture id
                    // and reused; rebuilt when the texture is replaced/freed.
                    let texture_binding = match self.texture_bindings.get(&mesh.texture_id) {
                        Some(g) => g.clone(),
                        None => {
                            let layout = self.material.binding_layouts()[1].clone();
                            match self.device.create_binding_group(
                                layout,
                                BindingGroupDescriptor::new()
                                    .with_texture(0, texture)
                                    .with_sampler(1, self.sampler.clone()),
                            ) {
                                Ok(group) => {
                                    self.texture_bindings.insert(mesh.texture_id, group.clone());
                                    group
                                }
                                Err(e) => {
                                    log::warn!("egui: failed to create texture binding group: {e}");
                                    continue;
                                }
                            }
                        }
                    };

                    // Create material instance
                    let material_instance = Arc::new(
                        MaterialInstance::new(self.material.clone())
                            .with_binding_group(uniform_binding.clone())
                            .with_binding_group(texture_binding),
                    );

                    // Calculate scissor rect - clip_rect is in points, but scissor needs physical pixels
                    let clip_min_x = (clip_rect.min.x * pixels_per_point).round() as i32;
                    let clip_min_y = (clip_rect.min.y * pixels_per_point).round() as i32;
                    let clip_max_x = (clip_rect.max.x * pixels_per_point).round() as i32;
                    let clip_max_y = (clip_rect.max.y * pixels_per_point).round() as i32;

                    let scissor_x = clip_min_x.max(0);
                    let scissor_y = clip_min_y.max(0);
                    let scissor_width = (clip_max_x - clip_min_x).max(0) as u32;
                    let scissor_height = (clip_max_y - clip_min_y).max(0) as u32;

                    // Clamp to screen bounds
                    let scissor_width =
                        scissor_width.min(screen_width.saturating_sub(scissor_x as u32));
                    let scissor_height =
                        scissor_height.min(screen_height.saturating_sub(scissor_y as u32));

                    if scissor_width > 0 && scissor_height > 0 {
                        pass.add_draw_command(
                            DrawCommand::new(gpu_mesh, material_instance)
                                .with_dynamic_offsets(vec![vec![uniform_offset as u32]])
                                .with_scissor_rect(crate::types::ScissorRect {
                                    x: scissor_x,
                                    y: scissor_y,
                                    width: scissor_width,
                                    height: scissor_height,
                                }),
                        );
                    }
                }
                Primitive::Callback(_) => {
                    // Custom rendering callbacks are not supported yet
                    log::warn!("Egui render callbacks are not supported");
                }
            }
        }

        pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;

    fn make_texture(device: &Arc<GraphicsDevice>) -> Arc<Texture> {
        device
            .create_texture(
                &TextureDescriptor::new_2d(
                    64,
                    64,
                    TextureFormat::Bgra8UnormSrgb,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("test_user_texture"),
            )
            .expect("create test texture")
    }

    /// Populate the eager binding cache for `id` the same way the draw path does
    /// (`create_graphics_pass`): group 1 = the user texture + the shared sampler.
    fn cache_binding(renderer: &mut EguiRenderer, id: TextureId, texture: Arc<Texture>) {
        let layout = renderer.material.binding_layouts()[1].clone();
        let group = renderer
            .device
            .create_binding_group(
                layout,
                BindingGroupDescriptor::new()
                    .with_texture(0, texture)
                    .with_sampler(1, renderer.sampler.clone()),
            )
            .expect("create user-texture binding group");
        renderer.texture_bindings.insert(id, group);
    }

    /// Regression (#40 eager binding, exposed via the editor SceneView going
    /// gray): swapping a user texture's underlying `Arc` (scene target recreated
    /// on resize) must drop the cached binding so the next draw rebuilds against
    /// the new texture — otherwise egui samples the stale (old) texture.
    #[test]
    fn update_user_texture_invalidates_binding_when_arc_swaps() {
        let device = GraphicsInstance::new().unwrap().create_device().unwrap();
        let mut renderer = EguiRenderer::new(device.clone(), TextureFormat::Bgra8UnormSrgb);

        let tex_a = make_texture(&device);
        let id = renderer.register_user_texture(tex_a.clone());
        cache_binding(&mut renderer, id, tex_a);
        assert!(renderer.texture_bindings.contains_key(&id));

        // Resize path: a brand-new texture Arc replaces the old one.
        let tex_b = make_texture(&device);
        renderer.update_user_texture(id, tex_b);

        assert!(
            !renderer.texture_bindings.contains_key(&id),
            "stale binding must be dropped when the user texture is swapped"
        );
    }

    /// The common per-frame re-registration passes the *same* `Arc`; the cached
    /// binding must survive so #40's zero-per-draw descriptor win is preserved.
    #[test]
    fn update_user_texture_keeps_binding_when_arc_unchanged() {
        let device = GraphicsInstance::new().unwrap().create_device().unwrap();
        let mut renderer = EguiRenderer::new(device.clone(), TextureFormat::Bgra8UnormSrgb);

        let tex = make_texture(&device);
        let id = renderer.register_user_texture(tex.clone());
        cache_binding(&mut renderer, id, tex.clone());

        renderer.update_user_texture(id, tex);

        assert!(
            renderer.texture_bindings.contains_key(&id),
            "binding must be reused when the same texture Arc is re-registered"
        );
    }
}
