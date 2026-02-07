//! Egui renderer for RedLilium graphics.
//!
//! This module handles GPU resources and rendering for egui.

use std::collections::HashMap;
use std::sync::Arc;

use egui::epaint::{ImageDelta, Primitive, Vertex};
use egui::{ClippedPrimitive, TextureId, TexturesDelta};

use crate::GraphicsDevice;
use crate::graph::{ColorAttachment, GraphicsPass, LoadOp, RenderTarget, RenderTargetConfig};
use crate::materials::{
    BindingGroup, BindingLayout, BindingLayoutEntry, BindingType, Material, MaterialDescriptor,
    MaterialInstance, ShaderSource, ShaderStage, ShaderStageFlags,
};
use crate::mesh::{
    IndexFormat, Mesh, PrimitiveTopology, VertexAttribute, VertexAttributeFormat,
    VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
};
use crate::resources::{Buffer, Sampler, Texture};
use crate::shader::{EGUI_SHADER_SOURCE, ShaderComposer};
use crate::types::{
    AddressMode, BufferDescriptor, BufferUsage, FilterMode, SamplerDescriptor, TextureDescriptor,
    TextureFormat, TextureUsage,
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

/// Default initial capacity for egui mesh vertex buffers.
const DEFAULT_VERTEX_CAPACITY: u32 = 1024;

/// Default initial capacity for egui mesh index buffers.
const DEFAULT_INDEX_CAPACITY: u32 = 3072;

/// Cached GPU buffers for a single egui mesh draw.
struct CachedEguiMesh {
    vertex_buffer: Arc<Buffer>,
    index_buffer: Arc<Buffer>,
    vertex_capacity: u32,
    index_capacity: u32,
}

/// Pool of egui mesh GPU buffers, reused across frames.
///
/// Entries persist across frames to avoid GPU buffer reallocation.
/// The scratch vertex buffer is cleared per-primitive but keeps its capacity.
struct EguiMeshPool {
    entries: Vec<CachedEguiMesh>,
    vertices: Vec<EguiVertex>,
}

/// Manages GPU resources for egui rendering.
pub struct EguiRenderer {
    device: Arc<GraphicsDevice>,
    material: Arc<Material>,
    vertex_layout: Arc<VertexLayout>,
    uniform_buffer: Arc<Buffer>,
    sampler: Arc<Sampler>,
    textures: HashMap<TextureId, Arc<Texture>>,
    /// CPU-side texture data for partial update support.
    texture_data: HashMap<TextureId, TextureData>,
    #[allow(dead_code)]
    uniform_binding_layout: Arc<BindingLayout>,
    #[allow(dead_code)]
    texture_binding_layout: Arc<BindingLayout>,
    /// Counter for generating unique user texture IDs.
    next_user_texture_id: u64,
    /// Pool of GPU mesh buffers reused across frames.
    mesh_pool: EguiMeshPool,
}

impl EguiRenderer {
    /// Create a new egui renderer.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
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

        // Create binding layouts
        let uniform_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::UniformBuffer)
                        .with_visibility(ShaderStageFlags::VERTEX),
                )
                .with_label("egui_uniform_bindings"),
        );

        let texture_binding_layout = Arc::new(
            BindingLayout::new()
                .with_entry(
                    BindingLayoutEntry::new(0, BindingType::Texture)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_entry(
                    BindingLayoutEntry::new(1, BindingType::Sampler)
                        .with_visibility(ShaderStageFlags::FRAGMENT),
                )
                .with_label("egui_texture_bindings"),
        );

        // Compose shader with library imports
        let mut shader_composer =
            ShaderComposer::with_standard_library().expect("Failed to create shader composer");
        let composed_shader = shader_composer
            .compose(EGUI_SHADER_SOURCE, &[])
            .expect("Failed to compose egui shader");

        // Create material
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::new(
                        ShaderStage::Vertex,
                        composed_shader.as_bytes().to_vec(),
                        "vs_main",
                    ))
                    .with_shader(ShaderSource::new(
                        ShaderStage::Fragment,
                        composed_shader.as_bytes().to_vec(),
                        "fs_main",
                    ))
                    .with_binding_layout(uniform_binding_layout.clone())
                    .with_binding_layout(texture_binding_layout.clone())
                    .with_vertex_layout(vertex_layout.clone())
                    .with_blend_state(crate::materials::BlendState::premultiplied_alpha())
                    .with_label("egui_material"),
            )
            .expect("Failed to create egui material");

        // Create uniform buffer
        let uniform_buffer = device
            .create_buffer(&BufferDescriptor::new(
                std::mem::size_of::<EguiUniforms>() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            ))
            .expect("Failed to create egui uniform buffer");

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
            uniform_buffer,
            sampler,
            textures: HashMap::new(),
            texture_data: HashMap::new(),
            uniform_binding_layout,
            texture_binding_layout,
            next_user_texture_id: 0,
            mesh_pool: EguiMeshPool {
                entries: Vec::new(),
                vertices: Vec::new(),
            },
        }
    }

    /// Update screen size uniforms (integer version for resize events).
    pub fn update_screen_size(&self, width: u32, height: u32) {
        self.update_screen_size_f32(width as f32, height as f32);
    }

    /// Update screen size uniforms with float values.
    ///
    /// This is used when the screen size needs to be in logical points rather than
    /// physical pixels (e.g., for egui rendering where vertices are in points).
    pub fn update_screen_size_f32(&self, width: f32, height: f32) {
        let uniforms = EguiUniforms {
            screen_size: [width, height],
            _padding: [0.0, 0.0],
        };
        self.device
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms))
            .expect("Failed to write egui uniform buffer");
    }

    /// Process texture updates from egui.
    pub fn update_textures(&mut self, textures_delta: &TexturesDelta) {
        // Free textures that are no longer needed
        for id in &textures_delta.free {
            self.textures.remove(id);
            self.texture_data.remove(id);
        }

        // Set or update textures
        for (id, delta) in &textures_delta.set {
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

                // Re-upload the full texture
                if let Some(texture) = self.textures.get(&id) {
                    self.device
                        .write_texture(texture, &data.pixels)
                        .expect("Failed to write egui texture");
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

        // Upload pixel data
        self.device
            .write_texture(&texture, &new_pixels)
            .expect("Failed to write egui texture");

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

    /// Get or grow a pooled mesh entry at the given index.
    ///
    /// If the entry doesn't exist or its buffers are too small, (re)allocates
    /// with capacity grown by factor 2 from the current size, or the default
    /// capacity, whichever is larger.
    fn get_or_grow_mesh_entry<'a>(
        device: &Arc<GraphicsDevice>,
        pool: &'a mut EguiMeshPool,
        index: usize,
        vertex_count: u32,
        index_count: u32,
    ) -> &'a mut CachedEguiMesh {
        let needs_realloc = if let Some(entry) = pool.entries.get(index) {
            entry.vertex_capacity < vertex_count || entry.index_capacity < index_count
        } else {
            true
        };

        if needs_realloc {
            let old_vcap = pool.entries.get(index).map_or(0, |e| e.vertex_capacity);
            let old_icap = pool.entries.get(index).map_or(0, |e| e.index_capacity);

            let new_vcap = vertex_count
                .max(old_vcap.saturating_mul(2))
                .max(DEFAULT_VERTEX_CAPACITY);
            let new_icap = index_count
                .max(old_icap.saturating_mul(2))
                .max(DEFAULT_INDEX_CAPACITY);

            let vertex_stride = std::mem::size_of::<EguiVertex>() as u64;
            let vertex_buffer = device
                .create_buffer(
                    &BufferDescriptor::new(
                        new_vcap as u64 * vertex_stride,
                        BufferUsage::VERTEX | BufferUsage::COPY_DST,
                    )
                    .with_label("egui_pool_vertices"),
                )
                .expect("Failed to create egui pooled vertex buffer");

            let index_buffer = device
                .create_buffer(
                    &BufferDescriptor::new(
                        new_icap as u64 * std::mem::size_of::<u32>() as u64,
                        BufferUsage::INDEX | BufferUsage::COPY_DST,
                    )
                    .with_label("egui_pool_indices"),
                )
                .expect("Failed to create egui pooled index buffer");

            let entry = CachedEguiMesh {
                vertex_buffer,
                index_buffer,
                vertex_capacity: new_vcap,
                index_capacity: new_icap,
            };

            if index < pool.entries.len() {
                pool.entries[index] = entry;
            } else {
                pool.entries.push(entry);
            }
        }

        &mut pool.entries[index]
    }

    /// Create a graphics pass for rendering egui primitives.
    ///
    /// Reuses pooled GPU buffers across frames, only reallocating when the
    /// existing buffers are too small (growing by factor 2).
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

        // Create uniform binding group
        #[allow(clippy::arc_with_non_send_sync)]
        let uniform_binding =
            Arc::new(BindingGroup::new().with_buffer(0, self.uniform_buffer.clone()));

        let pool = &mut self.mesh_pool;
        let mut mesh_index = 0;

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

                    // Convert vertices into reusable scratch buffer
                    pool.vertices.clear();
                    pool.vertices
                        .extend(mesh.vertices.iter().map(EguiVertex::from));

                    // Get or grow pooled GPU buffers for this draw
                    Self::get_or_grow_mesh_entry(
                        &self.device,
                        pool,
                        mesh_index,
                        vertex_count,
                        index_count,
                    );
                    let entry = &pool.entries[mesh_index];

                    // Clone buffer Arcs for upload and mesh construction
                    let vb = entry.vertex_buffer.clone();
                    let ib = entry.index_buffer.clone();

                    // Upload vertex data to pooled buffer
                    self.device
                        .write_buffer(&vb, 0, bytemuck::cast_slice(&pool.vertices))
                        .expect("Failed to write egui vertex buffer");

                    // Upload index data to pooled buffer
                    self.device
                        .write_buffer(&ib, 0, bytemuck::cast_slice(&mesh.indices))
                        .expect("Failed to write egui index buffer");

                    // Construct a Mesh referencing the pooled buffers with actual counts
                    let gpu_mesh = Arc::new(Mesh::new(
                        Arc::clone(&self.device),
                        self.vertex_layout.clone(),
                        PrimitiveTopology::TriangleList,
                        vec![vb],
                        vertex_count,
                        Some(ib),
                        Some(IndexFormat::Uint32),
                        index_count,
                        Some("egui_mesh".into()),
                    ));

                    mesh_index += 1;

                    // Create texture binding group
                    #[allow(clippy::arc_with_non_send_sync)]
                    let texture_binding = Arc::new(
                        BindingGroup::new()
                            .with_texture(0, texture)
                            .with_sampler(1, self.sampler.clone()),
                    );

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
                        pass.add_draw_with_scissor(
                            gpu_mesh,
                            material_instance,
                            crate::types::ScissorRect {
                                x: scissor_x,
                                y: scissor_y,
                                width: scissor_width,
                                height: scissor_height,
                            },
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
