//! Scene view state for rendering ECS entities into the editor's SceneView panel.
//!
//! Reads Camera, GlobalTransform, MeshRenderer, and Visibility
//! from the ECS World and builds a forward rendering pass targeting the
//! swapchain with viewport/scissor matching the egui panel rect.
//!
//! Also maintains an R32Uint entity-index texture for GPU-based object picking.

use std::sync::{Arc, Mutex};

use redlilium_core::material::CpuMaterial;
use redlilium_core::math::mat4_to_cols_array_2d;
use redlilium_ecs::{
    Camera, CameraTarget, Entity, GlobalTransform, MeshRenderer, PrimitiveMaterial, RenderPassType,
    Visibility, World, shaders,
};
use redlilium_graphics::{
    Buffer, BufferDescriptor, BufferTextureCopyRegion, BufferTextureLayout, BufferUsage,
    ColorAttachment, DepthStencilAttachment, DrawCommand, GraphicsDevice, GraphicsPass, LoadOp,
    Material, RenderTarget, RenderTargetConfig, RingBuffer, ScissorRect, StoreOp,
    TextureCopyLocation, TextureDescriptor, TextureFormat, TextureOrigin, TextureUsage,
    TransferConfig, TransferOperation, TransferPass, Viewport,
};

/// Manages GPU resources and rendering for the editor's SceneView panel.
pub struct SceneViewState {
    device: Arc<GraphicsDevice>,
    /// Format of the scene color target (swapchain format), for (re)creation. The
    /// scene's off-screen color+depth now live in a [`CameraTarget`] on the
    /// camera (the (ii) render-to-texture model); `ensure_camera_target` manages
    /// them. Separate from the window-sized `depth_texture` used by the
    /// entity-index/picking pass (decoupled — that pass clears its own depth).
    color_format: TextureFormat,
    depth_texture: Arc<redlilium_graphics::Texture>,
    opaque_material: Arc<Material>,
    cpu_material: Arc<CpuMaterial>,
    viewport: Option<Viewport>,
    scissor: Option<ScissorRect>,
    last_size: (u32, u32),
    /// Current size of the scene color/depth targets (panel physical size).
    scene_target_size: (u32, u32),

    // --- Picking ---
    entity_index_material: Arc<Material>,
    entity_index_texture: Arc<redlilium_graphics::Texture>,
    readback_buffer: Arc<Buffer>,
    /// Pixel coordinates (physical) of a pending pick request, resolved next frame.
    pending_pick: Option<[u32; 2]>,
    /// Single-pixel pick result bytes, filled by the frame pipeline after the GPU
    /// readback completes (one or more frames later). Polled by `resolve_pick`.
    pick_result: Arc<Mutex<Vec<u8>>>,

    // --- Rect selection readback ---
    pending_rect_pick: Option<[u32; 4]>,
    rect_readback_buffer: Arc<Buffer>,
    /// Rect pick result bytes, filled by the frame pipeline post-readback.
    rect_result: Arc<Mutex<Vec<u8>>>,
    /// Dimensions [w, h] and padded bytes_per_row of the last rect readback.
    rect_pick_layout: [u32; 3],

    // --- Per-draw dynamic-uniform rings ---
    /// Buffer of the scene forward `FrameRing` — now an ECS resource (filled by
    /// fill_transform_rings via the world, pushed to by the ForwardRender system
    /// later). Held here so the per-primitive material can bind it (group 0
    /// transform + group 1 material props, same buffer, different dyn offsets).
    /// Set by `set_frame_ring_buffer` after the resource is created.
    frame_ring_buffer: Option<Arc<Buffer>>,
    /// Ring of `EntityIndexUniforms` (picking pass), filled per frame. Stays
    /// editor-side with picking.
    entity_index_ring: RingBuffer,
    /// Per-entity entity-index (picking) ring offset, recorded by
    /// [`fill_picking_rings`]. The forward-pass offsets now live in the
    /// ForwardRender system (computed inline per draw).
    picking_offsets: std::collections::HashMap<u32, u32>,
    /// Pending mesh-data uploads (from [`create_entity_resources`]), flushed into
    /// the frame graph by [`flush_uploads`](Self::flush_uploads).
    pending_uploads: Vec<TransferOperation>,
}

impl SceneViewState {
    /// Create scene view resources.
    pub fn new(device: Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        let opaque_material = shaders::create_opaque_color_material(
            &device,
            surface_format,
            TextureFormat::Depth32Float,
        );
        let cpu_material = shaders::create_opaque_color_cpu_material();

        let entity_index_material =
            shaders::create_entity_index_material(&device, TextureFormat::Depth32Float);

        let depth_texture = Self::create_depth_texture(&device, 256, 256);
        let entity_index_texture = Self::create_entity_index_texture(&device, 256, 256);

        let readback_buffer = device
            .create_buffer(&BufferDescriptor::new(
                4,
                BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            ))
            .expect("Failed to create picking readback buffer");

        // Default rect readback buffer: 256×256 × 4 bytes with 256-byte row alignment.
        let default_rect_size = 256u64 * 256 * 4;
        let rect_readback_buffer = device
            .create_buffer(&BufferDescriptor::new(
                default_rect_size,
                BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            ))
            .expect("Failed to create rect readback buffer");

        // Persistent per-entity transform rings (filled each frame, monotonic
        // with wrap; sized generously so a region is reused only many frames
        // later — after its fence — avoiding frames-in-flight races).
        let entity_index_ring = RingBuffer::new(
            &device,
            1 << 20,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "scene_view_transform_ei",
        )
        .expect("Failed to create entity-index transform ring");

        Self {
            device,
            color_format: surface_format,
            depth_texture,
            opaque_material,
            cpu_material,
            viewport: None,
            scissor: None,
            last_size: (256, 256),
            scene_target_size: (256, 256),
            entity_index_material,
            entity_index_texture,
            readback_buffer,
            pending_pick: None,
            pick_result: Arc::new(Mutex::new(Vec::new())),
            pending_rect_pick: None,
            rect_readback_buffer,
            rect_result: Arc::new(Mutex::new(Vec::new())),
            rect_pick_layout: [0; 3],
            frame_ring_buffer: None,
            entity_index_ring,
            picking_offsets: std::collections::HashMap::new(),
            pending_uploads: Vec::new(),
        }
    }

    /// Flush queued mesh uploads into the current frame's render graph.
    pub fn flush_uploads(&mut self, graph: &mut redlilium_graphics::RenderGraph) {
        if self.pending_uploads.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.pending_uploads);
        let mut pass = TransferPass::new("scene_view_mesh_uploads".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(ops));
        graph.add_transfer_pass(pass);
    }

    /// Create GPU resources for a renderable entity with picking support.
    ///
    /// Returns `(per_entity_buffers, primitive_material, gpu_mesh)`. The caller
    /// assembles these into a [`Primitive`](redlilium_ecs::Primitive) +
    /// [`MeshRenderer`](redlilium_ecs::MeshRenderer).
    /// Build the per-entity material (the mesh is now an asset, requested
    /// separately via `MeshManager`). Group 0 (transform) and group 1 (material
    /// props) bind shared rings as dynamic uniforms; per-draw offsets are filled
    /// each frame by `fill_transform_rings`.
    pub fn create_entity_material(&mut self) -> PrimitiveMaterial {
        let frame_ring_buffer = self
            .frame_ring_buffer
            .as_ref()
            .expect("frame ring buffer set before entities are created");
        shaders::create_opaque_color_primitive_material_ring(
            &self.opaque_material,
            Some(&self.entity_index_material),
            &self.cpu_material,
            frame_ring_buffer, // group 0 (transform)
            Some(self.entity_index_ring.buffer()),
            frame_ring_buffer, // group 1 (material props) — same ring
        )
    }

    /// Fill the picking (entity-index) ring for this frame and record each
    /// entity's offset. Editor-side (picking is editor-only); the forward fill
    /// lives in `fill_transform_rings` (heading for the ForwardRender system).
    pub fn fill_picking_rings(&mut self, world: &World) {
        self.picking_offsets.clear();

        let Ok(cameras) = world.read_all::<Camera>() else {
            return;
        };
        let vp = match cameras.iter().next() {
            Some((_, camera)) => mat4_to_cols_array_2d(&camera.view_projection()),
            None => return,
        };
        drop(cameras);

        let Ok(renderers) = world.read::<MeshRenderer>() else {
            return;
        };
        let Ok(globals) = world.read::<GlobalTransform>() else {
            return;
        };

        for (idx, _renderer) in renderers.iter() {
            let model = globals
                .get(idx)
                .map(|g| mat4_to_cols_array_2d(&g.0))
                .unwrap_or_else(|| mat4_to_cols_array_2d(&redlilium_core::math::Mat4::identity()));
            let ei = shaders::EntityIndexUniforms {
                view_projection: vp,
                model,
                entity_index: idx,
                _padding: [0; 3],
            };
            let offset = Self::ring_push(&mut self.entity_index_ring, bytemuck::bytes_of(&ei));
            self.picking_offsets.insert(idx, offset);
        }
    }

    /// Allocate a slot in `ring` (wrapping when full) and write `data`,
    /// returning its byte offset.
    fn ring_push(ring: &mut RingBuffer, data: &[u8]) -> u32 {
        let size = data.len() as u64;
        let alloc = ring.allocate(size).unwrap_or_else(|| {
            // Wrap: the start of the ring was written many frames ago, so its
            // GPU work has completed (ring sized well above frames-in-flight).
            ring.reset();
            ring.allocate(size)
                .expect("transform ring too small for one element")
        });
        let _ = ring.write(&alloc, data);
        alloc.offset as u32
    }

    /// Update the viewport and scissor from an egui panel rect.
    pub fn set_viewport(&mut self, rect: egui::Rect, pixels_per_point: f32) {
        let x = rect.min.x * pixels_per_point;
        let y = rect.min.y * pixels_per_point;
        let w = rect.width() * pixels_per_point;
        let h = rect.height() * pixels_per_point;

        self.viewport = Some(Viewport::new(x, y, w, h));
        self.scissor = Some(ScissorRect::new(x as i32, y as i32, w as u32, h as u32));
    }

    /// Recreate the depth and entity-index textures if the window size changed.
    pub fn resize_if_needed(&mut self, width: u32, height: u32) -> bool {
        let width = width.max(1);
        let height = height.max(1);

        if (width, height) == self.last_size {
            return false;
        }

        self.depth_texture = Self::create_depth_texture(&self.device, width, height);
        self.entity_index_texture = Self::create_entity_index_texture(&self.device, width, height);
        self.last_size = (width, height);
        true
    }

    /// Ensure the `camera` entity has a [`CameraTarget`] sized to the SceneView
    /// panel (the scene renders into it; egui shows it 1:1). Recreates the
    /// color+depth on size change. Returns the color texture so the caller can
    /// (re)register it with egui.
    pub fn ensure_camera_target(
        &mut self,
        world: &mut World,
        camera: Entity,
        width: u32,
        height: u32,
    ) -> Arc<redlilium_graphics::Texture> {
        let width = width.max(1);
        let height = height.max(1);
        let stale = (width, height) != self.scene_target_size
            || world.get::<CameraTarget>(camera).is_none();
        if stale {
            let color = Self::create_color_texture(&self.device, width, height, self.color_format);
            let depth = Self::create_depth_texture(&self.device, width, height);
            let _ = world.insert(
                camera,
                CameraTarget::new(color.clone(), depth, [0.055, 0.063, 0.078, 1.0]),
            );
            self.scene_target_size = (width, height);
            color
        } else {
            world
                .get::<CameraTarget>(camera)
                .map(|t| t.color.clone())
                .expect("CameraTarget present (checked above)")
        }
    }

    /// Build a graphics pass that renders entity indices to the entity-index
    /// texture (R32Uint). Uses the same depth buffer as the scene pass.
    pub fn build_entity_index_pass(&self, world: &World) -> Option<GraphicsPass> {
        let renderers = world.read::<MeshRenderer>().ok()?;
        let visibilities = world.read::<Visibility>().ok()?;

        let mut pass = GraphicsPass::new("entity_index".into());

        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(
                    ColorAttachment::new(RenderTarget::from_texture(
                        self.entity_index_texture.clone(),
                    ))
                    .with_load_op(LoadOp::clear_color(0.0, 0.0, 0.0, 0.0))
                    .with_store_op(StoreOp::Store),
                )
                .with_depth_stencil(
                    DepthStencilAttachment::from_texture(self.depth_texture.clone())
                        .with_clear_depth(1.0),
                ),
        );

        if let Some(viewport) = &self.viewport {
            pass.set_viewport(*viewport);
        }
        if let Some(scissor) = &self.scissor {
            pass.set_scissor_rect(*scissor);
        }

        let mut draw_count = 0u32;
        for (entity_idx, renderer) in renderers.iter() {
            if let Some(vis) = visibilities.get(entity_idx)
                && !vis.is_visible()
            {
                continue;
            }
            let ei_off = self.picking_offsets.get(&entity_idx).copied().unwrap_or(0);
            for primitive in &renderer.primitives {
                // Skip primitives whose mesh hasn't finished loading.
                let Some(mesh) = primitive.mesh() else {
                    continue;
                };
                if let Some(instance) = primitive.material.pass(RenderPassType::EntityIndex) {
                    pass.add_draw_command(
                        DrawCommand::new(mesh, Arc::clone(instance))
                            .with_dynamic_offsets(vec![vec![ei_off]]),
                    );
                    draw_count += 1;
                }
            }
        }

        if draw_count == 0 {
            log::warn!("Entity index pass has 0 draws — no EntityIndex pass on any material");
        }

        Some(pass)
    }

    /// Build a transfer pass that copies a single pixel from the entity-index
    /// texture into the readback buffer.
    pub fn build_pick_readback(&self, px: u32, py: u32) -> TransferPass {
        let (w, h) = self.last_size;
        let px = px.min(w.saturating_sub(1));
        let py = py.min(h.saturating_sub(1));

        let region = BufferTextureCopyRegion::new(
            BufferTextureLayout::packed(),
            TextureCopyLocation::new(0, TextureOrigin::new(px, py, 0)),
            redlilium_graphics::Extent3d {
                width: 1,
                height: 1,
                depth: 1,
            },
        );

        // Clear any stale result so `resolve_pick` only sees this readback.
        if let Ok(mut g) = self.pick_result.lock() {
            g.clear();
        }

        let mut pass = TransferPass::new("pick_readback".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(vec![
            // 1. Copy the picked pixel from the entity-index texture into the
            //    host-visible readback buffer.
            TransferOperation::readback_texture(
                self.entity_index_texture.clone(),
                self.readback_buffer.clone(),
                vec![region],
            ),
            // 2. After the fence, the frame pipeline copies the buffer into
            //    `pick_result` for `resolve_pick` to poll.
            TransferOperation::readback_buffer(
                self.readback_buffer.clone(),
                0..4,
                self.pick_result.clone(),
            ),
        ]));
        pass
    }

    /// Request a pick at the given physical pixel coordinates.
    ///
    /// The result will be available next frame via [`resolve_pick`].
    pub fn request_pick(&mut self, px: u32, py: u32) {
        log::info!(
            "Pick requested at pixel ({px}, {py}), texture size = {:?}",
            self.last_size
        );
        self.pending_pick = Some([px, py]);
    }

    /// Take the pending pick coordinates (consumed once to build the readback pass).
    pub fn take_pending_pick(&mut self) -> Option<[u32; 2]> {
        self.pending_pick.take()
    }

    /// Poll the pick result, filled asynchronously by the frame pipeline once
    /// the GPU readback completes.
    ///
    /// Returns `Some(entity_index)` if an entity was hit, `None` if empty space
    /// or while still waiting for the GPU. The result is consumed once read.
    pub fn resolve_pick(&mut self) -> Option<u32> {
        let data = {
            let mut guard = self.pick_result.lock().ok()?;
            if guard.len() < 4 {
                return None; // not ready yet
            }
            std::mem::take(&mut *guard)
        };
        let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        if value == 0 {
            None // cleared background — no entity
        } else {
            Some(value - 1) // shader wrote entity_index + 1
        }
    }

    // ---- Rect selection readback ----

    /// Request a rect readback at the given physical-pixel rectangle.
    ///
    /// Resizes the readback buffer if needed. The result will be available
    /// after 2 frames via [`resolve_rect_pick`].
    pub fn request_rect_pick(&mut self, x: u32, y: u32, w: u32, h: u32) {
        let (tex_w, tex_h) = self.last_size;
        let x = x.min(tex_w.saturating_sub(1));
        let y = y.min(tex_h.saturating_sub(1));
        let w = w.min(tex_w - x).max(1);
        let h = h.min(tex_h - y).max(1);

        // Padded bytes_per_row (aligned to 256 for GPU transfer requirements).
        let bytes_per_row = (w * 4).div_ceil(256) * 256;
        let required_size = bytes_per_row as u64 * h as u64;

        if required_size > self.rect_readback_buffer.size() {
            self.rect_readback_buffer = self
                .device
                .create_buffer(&BufferDescriptor::new(
                    required_size,
                    BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                ))
                .expect("Failed to resize rect readback buffer");
        }

        self.pending_rect_pick = Some([x, y, w, h]);
    }

    /// Take the pending rect pick coordinates (consumed to build the readback pass).
    pub fn take_pending_rect_pick(&mut self) -> Option<[u32; 4]> {
        self.pending_rect_pick.take()
    }

    /// Build a transfer pass that copies a rectangular region from the
    /// entity-index texture into the rect readback buffer.
    pub fn build_rect_readback(&self, x: u32, y: u32, w: u32, h: u32) -> TransferPass {
        let bytes_per_row = (w * 4).div_ceil(256) * 256;

        let region = BufferTextureCopyRegion::new(
            BufferTextureLayout::new(0, Some(bytes_per_row), Some(h)),
            TextureCopyLocation::new(0, TextureOrigin::new(x, y, 0)),
            redlilium_graphics::Extent3d {
                width: w,
                height: h,
                depth: 1,
            },
        );

        let total_bytes = (bytes_per_row * h) as usize;

        if let Ok(mut g) = self.rect_result.lock() {
            g.clear();
        }

        let mut pass = TransferPass::new("rect_pick_readback".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(vec![
            TransferOperation::readback_texture(
                self.entity_index_texture.clone(),
                self.rect_readback_buffer.clone(),
                vec![region],
            ),
            TransferOperation::readback_buffer(
                self.rect_readback_buffer.clone(),
                0..total_bytes,
                self.rect_result.clone(),
            ),
        ]));
        pass
    }

    /// Record the layout [w, h, bytes_per_row] of the in-flight rect readback,
    /// used by [`resolve_rect_pick`] to decode the result bytes.
    pub fn set_rect_layout(&mut self, w: u32, h: u32) {
        let bytes_per_row = (w * 4).div_ceil(256) * 256;
        self.rect_pick_layout = [w, h, bytes_per_row];
    }

    /// Read rect pick results from the readback buffer.
    ///
    /// Returns `Some(entity_indices)` with unique entity indices found in the
    /// rectangle, or `None` if still waiting for the GPU.
    pub fn resolve_rect_pick(&mut self) -> Option<Vec<u32>> {
        let data = {
            let mut guard = self.rect_result.lock().ok()?;
            if guard.is_empty() {
                return None; // not ready yet
            }
            std::mem::take(&mut *guard)
        };

        let [w, h, bytes_per_row] = self.rect_pick_layout;

        let mut unique = std::collections::HashSet::new();
        let pixel_bytes = (w * 4) as usize;
        let row_stride = bytes_per_row as usize;

        for row in 0..h as usize {
            let row_start = row * row_stride;
            let row_end = row_start + pixel_bytes;
            if row_end > data.len() {
                break;
            }
            for pixel in data[row_start..row_end].chunks_exact(4) {
                let value = u32::from_le_bytes([pixel[0], pixel[1], pixel[2], pixel[3]]);
                if value != 0 {
                    unique.insert(value - 1); // shader wrote entity_index + 1
                }
            }
        }

        Some(unique.into_iter().collect())
    }

    /// Clear the viewport (e.g. when the SceneView tab is not visible).
    pub fn clear_viewport(&mut self) {
        self.viewport = None;
        self.scissor = None;
    }

    /// Whether the viewport has been set (i.e. the SceneView tab is visible).
    pub fn has_viewport(&self) -> bool {
        self.viewport.is_some()
    }

    /// Get the viewport aspect ratio, or 1.0 if no viewport is set.
    pub fn aspect_ratio(&self) -> f32 {
        if let Some(vp) = &self.viewport {
            vp.width / vp.height.max(1.0)
        } else {
            let (w, h) = self.last_size;
            w as f32 / h.max(1) as f32
        }
    }

    /// The graphics device used by this scene view.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    /// Set the forward `FrameRing`'s buffer (the resource it wraps), so
    /// per-primitive materials can bind it. Call once, before creating entities.
    pub fn set_frame_ring_buffer(&mut self, buffer: Arc<Buffer>) {
        self.frame_ring_buffer = Some(buffer);
    }

    /// Get the GPU material for the opaque color shader.
    pub fn opaque_material(&self) -> &Arc<Material> {
        &self.opaque_material
    }

    /// Get the CPU material definition for the opaque color shader.
    pub fn cpu_material(&self) -> &Arc<CpuMaterial> {
        &self.cpu_material
    }

    /// Get the GPU material for the entity-index picking shader.
    pub fn entity_index_material(&self) -> &Arc<Material> {
        &self.entity_index_material
    }

    fn create_color_texture(
        device: &Arc<GraphicsDevice>,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Arc<redlilium_graphics::Texture> {
        device
            .create_texture(
                // RENDER_ATTACHMENT to draw the scene into it; TEXTURE_BINDING so
                // egui can sample it as a user texture in the SceneView panel.
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    format,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("scene_view_color"),
            )
            .expect("Failed to create scene view color texture")
    }

    fn create_depth_texture(
        device: &Arc<GraphicsDevice>,
        width: u32,
        height: u32,
    ) -> Arc<redlilium_graphics::Texture> {
        device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::Depth32Float,
                    TextureUsage::RENDER_ATTACHMENT,
                )
                .with_label("scene_view_depth"),
            )
            .expect("Failed to create scene view depth texture")
    }

    fn create_entity_index_texture(
        device: &Arc<GraphicsDevice>,
        width: u32,
        height: u32,
    ) -> Arc<redlilium_graphics::Texture> {
        device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::R32Uint,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
                )
                .with_label("scene_view_entity_index"),
            )
            .expect("Failed to create scene view entity index texture")
    }
}
