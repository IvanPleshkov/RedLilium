//! Scene view state for rendering ECS entities into the editor's SceneView panel.
//!
//! Reads Camera, GlobalTransform, RenderMesh, RenderMaterial, and Visibility
//! from the ECS World and builds a forward rendering pass targeting the
//! swapchain with viewport/scissor matching the egui panel rect.
//!
//! Also maintains an R32Uint entity-index texture for GPU-based object picking.

use std::sync::Arc;

use redlilium_core::material::CpuMaterial;
use redlilium_ecs::{
    PerEntityBuffers, RenderMaterial, RenderMesh, RenderPassType, Visibility, World, shaders,
};
use redlilium_graphics::{
    Buffer, BufferDescriptor, BufferTextureCopyRegion, BufferTextureLayout, BufferUsage,
    ColorAttachment, DepthStencilAttachment, GraphicsDevice, GraphicsPass, LoadOp, Material,
    RenderTarget, RenderTargetConfig, ScissorRect, StoreOp, SurfaceTexture, TextureCopyLocation,
    TextureDescriptor, TextureFormat, TextureOrigin, TextureUsage, TransferConfig,
    TransferOperation, TransferPass, Viewport,
};

/// Manages GPU resources and rendering for the editor's SceneView panel.
pub struct SceneViewState {
    device: Arc<GraphicsDevice>,
    depth_texture: Arc<redlilium_graphics::Texture>,
    opaque_material: Arc<Material>,
    cpu_material: Arc<CpuMaterial>,
    viewport: Option<Viewport>,
    scissor: Option<ScissorRect>,
    last_size: (u32, u32),

    // --- Picking ---
    entity_index_material: Arc<Material>,
    entity_index_texture: Arc<redlilium_graphics::Texture>,
    readback_buffer: Arc<Buffer>,
    /// Pixel coordinates (physical) of a pending pick request, resolved next frame.
    pending_pick: Option<[u32; 2]>,
    /// Frames remaining until the readback buffer is ready to read.
    /// Set to 2 when a readback is submitted (GPU needs at least one full
    /// frame to finish the transfer). Decremented each frame; read at 0.
    pick_frames_remaining: u32,

    // --- Rect selection readback ---
    pending_rect_pick: Option<[u32; 4]>,
    rect_readback_buffer: Arc<Buffer>,
    rect_pick_frames_remaining: u32,
    /// Dimensions [w, h] and padded bytes_per_row of the in-flight rect readback.
    rect_pick_layout: [u32; 3],
}

impl SceneViewState {
    /// Create scene view resources.
    pub fn new(device: Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        let (opaque_material, _layout) = shaders::create_opaque_color_material(
            &device,
            surface_format,
            TextureFormat::Depth32Float,
        );
        let cpu_material = shaders::create_opaque_color_cpu_material();

        let (entity_index_material, _ei_layout) =
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

        Self {
            device,
            depth_texture,
            opaque_material,
            cpu_material,
            viewport: None,
            scissor: None,
            last_size: (256, 256),
            entity_index_material,
            entity_index_texture,
            readback_buffer,
            pending_pick: None,
            pick_frames_remaining: 0,
            pending_rect_pick: None,
            rect_readback_buffer,
            rect_pick_frames_remaining: 0,
            rect_pick_layout: [0; 3],
        }
    }

    /// Create GPU resources for a renderable entity with picking support.
    ///
    /// Returns `(per_entity_buffers, render_material, gpu_mesh)`.
    pub fn create_entity_resources(
        &self,
        cpu_mesh: &redlilium_core::mesh::CpuMesh,
    ) -> (
        PerEntityBuffers,
        RenderMaterial,
        Arc<redlilium_graphics::Mesh>,
    ) {
        let (per_entity, render_material, _bundle) = shaders::create_opaque_color_entity_full(
            &self.device,
            &self.opaque_material,
            &self.entity_index_material,
            &self.cpu_material,
        );

        let gpu_mesh = self
            .device
            .create_mesh_from_cpu(cpu_mesh)
            .expect("Failed to create entity GPU mesh");

        (per_entity, render_material, gpu_mesh)
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

    /// Build a graphics pass that renders ECS entities to the swapchain
    /// at the SceneView panel's viewport.
    pub fn build_scene_pass(
        &self,
        world: &World,
        swapchain: &SurfaceTexture,
    ) -> Option<GraphicsPass> {
        let meshes = world.read::<RenderMesh>().ok()?;
        let materials = world.read::<RenderMaterial>().ok()?;
        let visibilities = world.read::<Visibility>().ok()?;

        let mut pass = GraphicsPass::new("scene_view".into());

        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(
                    ColorAttachment::from_surface(swapchain)
                        .with_clear_color(0.055, 0.063, 0.078, 1.0),
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

        for (entity_idx, render_mesh) in meshes.iter() {
            let Some(render_material) = materials.get(entity_idx) else {
                continue;
            };
            if let Some(vis) = visibilities.get(entity_idx)
                && !vis.is_visible()
            {
                continue;
            }
            if let Some(instance) = render_material.pass(RenderPassType::Forward) {
                pass.add_draw(render_mesh.mesh.clone(), Arc::clone(instance));
            }
        }

        Some(pass)
    }

    /// Build a graphics pass that renders entity indices to the entity-index
    /// texture (R32Uint). Uses the same depth buffer as the scene pass.
    pub fn build_entity_index_pass(&self, world: &World) -> Option<GraphicsPass> {
        let meshes = world.read::<RenderMesh>().ok()?;
        let materials = world.read::<RenderMaterial>().ok()?;
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
        for (entity_idx, render_mesh) in meshes.iter() {
            let Some(render_material) = materials.get(entity_idx) else {
                continue;
            };
            if let Some(vis) = visibilities.get(entity_idx)
                && !vis.is_visible()
            {
                continue;
            }
            if let Some(instance) = render_material.pass(RenderPassType::EntityIndex) {
                pass.add_draw(render_mesh.mesh.clone(), Arc::clone(instance));
                draw_count += 1;
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

        let mut pass = TransferPass::new("pick_readback".into());
        pass.set_transfer_config(TransferConfig::new().with_operation(
            TransferOperation::readback_texture(
                self.entity_index_texture.clone(),
                self.readback_buffer.clone(),
                vec![region],
            ),
        ));
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

    /// Mark that a readback was submitted. The result will be ready after
    /// `pick_frames_remaining` frames have elapsed (typically 2).
    pub fn set_pick_in_flight(&mut self) {
        self.pick_frames_remaining = 2;
    }

    /// Read the pick result from the readback buffer (call after GPU has finished).
    ///
    /// Returns `Some(entity_index)` if an entity was hit, `None` if empty space.
    /// Returns `None` while still waiting for the GPU to finish the readback.
    pub fn resolve_pick(&mut self) -> Option<u32> {
        if self.pick_frames_remaining == 0 {
            return None;
        }
        self.pick_frames_remaining -= 1;
        if self.pick_frames_remaining > 0 {
            return None; // still waiting for GPU
        }

        let data = self.device.read_buffer(&self.readback_buffer, 0, 4);
        if data.len() < 4 {
            log::warn!(
                "Pick readback: buffer returned {} bytes (expected 4)",
                data.len()
            );
            return None;
        }
        let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        log::info!(
            "Pick readback: raw value = {value}, decoded = {}",
            if value == 0 {
                "None (background)".to_string()
            } else {
                format!("Some({})", value - 1)
            }
        );
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

        let mut pass = TransferPass::new("rect_pick_readback".into());
        pass.set_transfer_config(TransferConfig::new().with_operation(
            TransferOperation::readback_texture(
                self.entity_index_texture.clone(),
                self.rect_readback_buffer.clone(),
                vec![region],
            ),
        ));
        pass
    }

    /// Mark that a rect readback was submitted with the given dimensions.
    pub fn set_rect_pick_in_flight(&mut self, w: u32, h: u32) {
        let bytes_per_row = (w * 4).div_ceil(256) * 256;
        self.rect_pick_frames_remaining = 2;
        self.rect_pick_layout = [w, h, bytes_per_row];
    }

    /// Read rect pick results from the readback buffer.
    ///
    /// Returns `Some(entity_indices)` with unique entity indices found in the
    /// rectangle, or `None` if still waiting for the GPU.
    pub fn resolve_rect_pick(&mut self) -> Option<Vec<u32>> {
        if self.rect_pick_frames_remaining == 0 {
            return None;
        }
        self.rect_pick_frames_remaining -= 1;
        if self.rect_pick_frames_remaining > 0 {
            return None;
        }

        let [w, h, bytes_per_row] = self.rect_pick_layout;
        let total_bytes = bytes_per_row as u64 * h as u64;
        let data = self
            .device
            .read_buffer(&self.rect_readback_buffer, 0, total_bytes);

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
