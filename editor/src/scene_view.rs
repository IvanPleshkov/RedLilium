//! Scene view state for rendering ECS entities into the editor's SceneView panel.
//!
//! Reads Camera, GlobalTransform, MeshRenderer, and Visibility
//! from the ECS World and builds a forward rendering pass targeting the
//! swapchain with viewport/scissor matching the egui panel rect.
//!
//! Also maintains an R32Uint entity-index texture for GPU-based object picking.

use std::sync::{Arc, Mutex};

use redlilium_core::math::mat4_to_cols_array_2d;
use redlilium_ecs::ui::Selection;
use redlilium_ecs::{
    Camera, CameraTarget, Entity, GlobalTransform, MeshRenderer, Visibility, World, shaders,
};
use redlilium_graphics::{
    BindingGroup, BindingGroupDescriptor, Buffer, BufferDescriptor, BufferTextureCopyRegion,
    BufferTextureLayout, BufferUsage, ColorAttachment, DepthConvention, DepthStencilAttachment,
    DrawCommand, GraphicsDevice, GraphicsPass, LoadOp, Material, MaterialInstance, MeshDescriptor,
    RenderTarget, RenderTargetConfig, RingBuffer, ScissorRect, StoreOp, TextureCopyLocation,
    TextureDescriptor, TextureFormat, TextureOrigin, TextureUsage, TransferConfig,
    TransferOperation, TransferPass, VertexAttributeSemantic, VertexBufferLayout, VertexLayout,
    Viewport,
};

use crate::selection_outline::{SelectionOutlineUniforms, create_selection_outline_material};

/// Selection outline color (editor orange).
const SELECTION_OUTLINE_COLOR: [f32; 4] = [1.0, 0.6, 0.0, 1.0];
/// Selection outline width in pixels.
const SELECTION_OUTLINE_THICKNESS: f32 = 2.0;

/// Byte offset of the pick-depth texel inside the pick readback buffer (the
/// entity index occupies bytes 0..4; texture-copy destinations keep a safe
/// 256-byte alignment).
const PICK_DEPTH_OFFSET: u64 = 256;

/// Resolved GPU pick: the hit entity index (`None` = empty space) and the
/// exact world-space surface point under the cursor, reconstructed from the
/// picking pass's depth output. `world_point` is `None` when the pick hit no
/// rendered geometry (cleared reversed-Z background).
pub struct PickHit {
    pub entity: Option<u32>,
    pub world_point: Option<redlilium_core::math::Vec3>,
}

/// Inputs to reconstruct a pick's world-space point from its depth readback:
/// the picked pixel, the view-projection the picking pass rendered with, and
/// the viewport mapping NDC to pick-texture pixels at that time.
struct PickSnapshot {
    px: u32,
    py: u32,
    view_projection: redlilium_core::math::Mat4,
    /// (x, y, w, h) of the viewport in pick-texture pixels.
    viewport: [f32; 4],
}

impl PickSnapshot {
    /// Unproject the picked texel at the given [0,1] reversed-Z depth through
    /// the inverse view-projection. NDC y is up while pixel y grows down;
    /// sampling at the texel center.
    fn unproject(&self, depth: f32) -> Option<redlilium_core::math::Vec3> {
        let [vx, vy, vw, vh] = self.viewport;
        if vw <= 0.0 || vh <= 0.0 {
            return None;
        }
        let ndc_x = 2.0 * ((self.px as f32 + 0.5 - vx) / vw) - 1.0;
        let ndc_y = 1.0 - 2.0 * ((self.py as f32 + 0.5 - vy) / vh);
        let inv = self.view_projection.try_inverse()?;
        let world = inv * redlilium_core::math::Vec4::new(ndc_x, ndc_y, depth, 1.0);
        if world.w.abs() < 1e-9 {
            return None;
        }
        Some(redlilium_core::math::Vec3::new(
            world.x / world.w,
            world.y / world.w,
            world.z / world.w,
        ))
    }
}

/// Manages GPU resources and rendering for the editor's SceneView panel.
pub struct SceneViewState {
    device: Arc<GraphicsDevice>,
    /// Format of the scene color target (the swapchain format), fed into the
    /// camera's [`CameraOutput`](redlilium_ecs::CameraOutput) spec so the
    /// derived target matches the surface's color space (ADR-029). Separate
    /// from the window-sized `depth_texture` used by the entity-index/picking
    /// pass (decoupled — that pass clears its own depth).
    color_format: TextureFormat,
    depth_texture: Arc<redlilium_graphics::Texture>,
    viewport: Option<Viewport>,
    scissor: Option<ScissorRect>,
    last_size: (u32, u32),

    // --- Picking ---
    /// Picking materials + their ring binding groups, one per distinct mesh
    /// vertex layout, created on first use. The pipeline's vertex-input state
    /// comes from the material's layout, so a single fixed-layout material
    /// misreads any mesh with a different stride (a generated sphere used to
    /// pick as a ~2x vertex soup). The group binds the entity-index ring at
    /// offset 0, the per-draw dynamic offset selecting the entity's slot.
    entity_index_materials:
        std::collections::HashMap<VertexLayout, (Arc<Material>, Arc<BindingGroup>)>,
    entity_index_texture: Arc<redlilium_graphics::Texture>,
    /// R32Float pick-depth target, the entity-index pass's third MRT output
    /// (raw reversed-Z `frag_coord.z`). Read back with the pick to
    /// reconstruct the exact world-space hit point.
    pick_depth_texture: Arc<redlilium_graphics::Texture>,
    /// View-projection the last `fill_picking_rings` rendered with —
    /// snapshotted per pick for the depth unprojection.
    picking_view_projection: Option<redlilium_core::math::Mat4>,
    /// Unprojection inputs captured when the pick readback was built.
    pick_snapshot: Option<PickSnapshot>,
    readback_buffer: Arc<Buffer>,
    /// Pixel coordinates (physical) of a pending pick request, resolved next frame.
    pending_pick: Option<[u32; 2]>,
    /// Single-pixel pick result bytes, filled by the frame pipeline after the GPU
    /// readback completes (one or more frames later). Polled by `resolve_pick`.
    pick_result: Arc<Mutex<Vec<u8>>>,
    /// Frames a pending pick has been held waiting on `readback_buffer`'s prior
    /// async map to resolve (#33). Only a diagnostic — the map callback always
    /// clears the flag, so this can't wedge; a high value flags a stuck readback.
    pick_wait_frames: u32,

    // --- Selection outline ---
    /// R8 selection mask, the entity-index pass's second target (window-sized,
    /// like `entity_index_texture`). Read by the outline pass.
    selection_mask_texture: Arc<redlilium_graphics::Texture>,
    outline_material: Arc<Material>,
    /// Outline group (group 0): the params ring at binding 0 (per-draw dynamic
    /// offset) plus the mask texture at binding 1. Rebuilt on mask resize.
    outline_group: Arc<BindingGroup>,
    /// Ring of `SelectionOutlineUniforms`, one slot per frame.
    outline_ring: RingBuffer,
    /// Fullscreen dummy triangle (3 vertices, positions from `SV_VertexID`).
    outline_mesh: Arc<redlilium_graphics::Mesh>,
    /// Whether `fill_picking_rings` flagged any selected renderer this frame;
    /// gates the outline pass.
    has_selection: bool,

    // --- Rect selection readback ---
    pending_rect_pick: Option<[u32; 4]>,
    rect_readback_buffer: Arc<Buffer>,
    /// Rect pick result bytes, filled by the frame pipeline post-readback.
    rect_result: Arc<Mutex<Vec<u8>>>,
    /// Dimensions [w, h] and padded bytes_per_row of the last rect readback.
    rect_pick_layout: [u32; 3],
    /// As [`pick_wait_frames`](Self::pick_wait_frames) but for the rect readback.
    rect_wait_frames: u32,

    // --- Per-draw dynamic-uniform rings ---
    /// Buffer of the scene forward `FrameRing` — now an ECS resource (filled by
    /// fill_transform_rings via the world, pushed to by the SceneDrawer
    /// later). Held here so the per-primitive material can bind it (group 0
    /// transform + group 1 material props, same buffer, different dyn offsets).
    /// Set by `set_frame_ring_buffer` after the resource is created.
    frame_ring_buffer: Option<Arc<Buffer>>,
    /// Ring of `EntityIndexUniforms` (picking pass), filled per frame. Stays
    /// editor-side with picking.
    entity_index_ring: RingBuffer,
    /// Per-entity entity-index (picking) ring offset, recorded by
    /// [`fill_picking_rings`]. The forward-pass offsets now live in the
    /// SceneDrawer (computed inline per draw).
    picking_offsets: std::collections::HashMap<u32, u32>,
    /// Pending mesh-data uploads (from [`create_entity_resources`]), flushed into
    /// the frame graph by [`flush_uploads`](Self::flush_uploads).
    pending_uploads: Vec<TransferOperation>,
}

impl SceneViewState {
    /// Create scene view resources.
    pub fn new(device: Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        let depth_texture = Self::create_depth_texture(&device, 256, 256);
        let entity_index_texture = Self::create_entity_index_texture(&device, 256, 256);
        let selection_mask_texture = Self::create_selection_mask_texture(&device, 256, 256);
        let pick_depth_texture = Self::create_pick_depth_texture(&device, 256, 256);

        // Holds the picked entity-index texel at 0..4 and the pick-depth
        // texel at PICK_DEPTH_OFFSET.
        let readback_buffer = device
            .create_buffer(&BufferDescriptor::new(
                PICK_DEPTH_OFFSET + 4,
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

        // Selection outline: fullscreen contour material over the mask the
        // entity-index pass writes; the params ring holds one slot per frame.
        let outline_material = create_selection_outline_material(&device, surface_format);
        let outline_ring = RingBuffer::new(
            &device,
            16 << 10,
            BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            "scene_view_outline_params",
        )
        .expect("Failed to create selection outline params ring");
        let outline_group = Self::create_outline_group(
            &device,
            &outline_material,
            &outline_ring,
            &selection_mask_texture,
        );

        // The fullscreen triangle is generated from SV_VertexID; the mesh only
        // supplies a vertex count (plus a minimal dummy buffer so every
        // backend has something to bind) — same shape as the present blit.
        let outline_layout = Arc::new(
            VertexLayout::new()
                .with_buffer(VertexBufferLayout::new(4))
                .with_label("selection_outline_layout"),
        );
        let outline_mesh = device
            .create_mesh(
                &MeshDescriptor::new(outline_layout)
                    .with_vertex_count(3)
                    .with_label("selection_outline_triangle"),
            )
            .expect("Failed to create selection outline mesh");
        let mut pending_uploads = Vec::new();
        if let Some(vb) = outline_mesh.vertex_buffer(0) {
            let dummy: [f32; 3] = [0.0; 3];
            pending_uploads.push(TransferOperation::write_buffer(
                vb.clone(),
                0,
                Arc::from(bytemuck::cast_slice(&dummy)),
            ));
        }

        Self {
            device,
            color_format: surface_format,
            depth_texture,
            viewport: None,
            scissor: None,
            last_size: (256, 256),
            entity_index_materials: std::collections::HashMap::new(),
            entity_index_texture,
            pick_depth_texture,
            picking_view_projection: None,
            pick_snapshot: None,
            selection_mask_texture,
            outline_material,
            outline_group,
            outline_ring,
            outline_mesh,
            has_selection: false,
            readback_buffer,
            pending_pick: None,
            pick_result: Arc::new(Mutex::new(Vec::new())),
            pick_wait_frames: 0,
            pending_rect_pick: None,
            rect_readback_buffer,
            rect_result: Arc::new(Mutex::new(Vec::new())),
            rect_pick_layout: [0; 3],
            rect_wait_frames: 0,
            frame_ring_buffer: None,
            entity_index_ring,
            picking_offsets: std::collections::HashMap::new(),
            pending_uploads,
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

    /// Fill the picking (entity-index) ring for this frame and record each
    /// entity's offset. Editor-side (picking is editor-only); the forward fill
    /// lives in `fill_transform_rings` (now the SceneDrawer).
    pub fn fill_picking_rings(&mut self, world: &World) {
        self.picking_offsets.clear();
        self.has_selection = false;
        self.picking_view_projection = None;

        let Ok(cameras) = world.read_all::<Camera>() else {
            return;
        };
        let vp_mat = match cameras.iter().next() {
            Some((_, camera)) => camera.view_projection(),
            None => return,
        };
        drop(cameras);
        self.picking_view_projection = Some(vp_mat);
        let vp = mat4_to_cols_array_2d(&vp_mat);

        let Ok(renderers) = world.read::<MeshRenderer>() else {
            return;
        };
        let Ok(globals) = world.read::<GlobalTransform>() else {
            return;
        };

        // Selected entity indices, flagged into the mask target (the outline).
        let selected: std::collections::HashSet<u32> = if world.has_resource::<Selection>() {
            world
                .resource::<Selection>()
                .entities()
                .iter()
                .map(|e| e.index())
                .collect()
        } else {
            std::collections::HashSet::new()
        };

        for (idx, _renderer) in renderers.iter() {
            let model = globals
                .get(idx)
                .map(|g| mat4_to_cols_array_2d(&g.0))
                .unwrap_or_else(|| mat4_to_cols_array_2d(&redlilium_core::math::Mat4::identity()));
            let is_selected = selected.contains(&idx);
            self.has_selection |= is_selected;
            let ei = shaders::EntityIndexUniforms {
                view_projection: vp,
                model,
                entity_index: idx,
                selected: u32::from(is_selected),
                _padding: [0; 2],
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

    /// Recreate the depth, entity-index, and selection-mask textures if the
    /// window size changed.
    pub fn resize_if_needed(&mut self, width: u32, height: u32) -> bool {
        let width = width.max(1);
        let height = height.max(1);

        if (width, height) == self.last_size {
            return false;
        }

        self.depth_texture = Self::create_depth_texture(&self.device, width, height);
        self.entity_index_texture = Self::create_entity_index_texture(&self.device, width, height);
        self.selection_mask_texture =
            Self::create_selection_mask_texture(&self.device, width, height);
        self.pick_depth_texture = Self::create_pick_depth_texture(&self.device, width, height);
        // The outline group binds the mask texture — rebuild against the new one.
        self.outline_group = Self::create_outline_group(
            &self.device,
            &self.outline_material,
            &self.outline_ring,
            &self.selection_mask_texture,
        );
        self.last_size = (width, height);
        true
    }

    /// Keep the `camera` entity's [`CameraOutput`] spec (ADR-029) in sync
    /// with the SceneView panel: an `Offscreen` target fixed to the panel's
    /// physical size, in the surface's color space (egui shows it 1:1). The
    /// GPU textures themselves are derived by the `EnsureCameraTargets`
    /// system inside the Render schedule — no host-side texture management.
    pub fn sync_camera_output(&self, world: &mut World, camera: Entity, width: u32, height: u32) {
        use redlilium_ecs::{CameraOutput, OutputFormat, SizePolicy};
        let desired = CameraOutput::offscreen(SizePolicy::Fixed(width.max(1), height.max(1)), None)
            .with_clear_color([0.055, 0.063, 0.078, 1.0])
            .with_format(OutputFormat::matching_surface(self.color_format));
        let stale = world
            .get::<CameraOutput>(camera)
            .is_none_or(|current| *current != desired);
        if stale {
            let _ = world.insert(camera, desired);
        }
    }

    /// The picking material + ring binding group for `layout`, created on
    /// first use. Returns `None` for layouts the picking shader cannot
    /// consume: shader location 0 maps to the layout's first attribute
    /// (ADR-019), so it must be the position — skinned layouts that lead
    /// with texcoords are not pickable (pre-existing limitation).
    fn entity_index_material_for(
        &mut self,
        layout: &Arc<VertexLayout>,
    ) -> Option<(Arc<Material>, Arc<BindingGroup>)> {
        if layout.attributes.first().map(|a| a.semantic) != Some(VertexAttributeSemantic::Position)
        {
            return None;
        }
        if let Some(entry) = self.entity_index_materials.get(layout.as_ref()) {
            return Some(entry.clone());
        }
        let material = shaders::create_entity_index_material(
            &self.device,
            layout,
            TextureFormat::Depth32Float,
        );
        let group = self
            .device
            .create_binding_group(
                material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new().with_buffer_range(
                    0,
                    self.entity_index_ring.buffer().clone(),
                    0,
                    std::mem::size_of::<shaders::EntityIndexUniforms>() as u64,
                ),
            )
            .expect("Failed to create entity-index binding group");
        let entry = (material, group);
        self.entity_index_materials
            .insert(layout.as_ref().clone(), entry.clone());
        Some(entry)
    }

    /// Build a graphics pass that renders entity indices to the entity-index
    /// texture (R32Uint). Uses the same depth buffer as the scene pass.
    pub fn build_entity_index_pass(&mut self, world: &World) -> Option<GraphicsPass> {
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
                // Second target: the selection mask the outline pass reads.
                .with_color(
                    ColorAttachment::new(RenderTarget::from_texture(
                        self.selection_mask_texture.clone(),
                    ))
                    .with_load_op(LoadOp::clear_color(0.0, 0.0, 0.0, 0.0))
                    .with_store_op(StoreOp::Store),
                )
                // Third target: raw reversed-Z depth for the pick's
                // world-point readback (clear 0 = background).
                .with_color(
                    ColorAttachment::new(RenderTarget::from_texture(
                        self.pick_depth_texture.clone(),
                    ))
                    .with_load_op(LoadOp::clear_color(0.0, 0.0, 0.0, 0.0))
                    .with_store_op(StoreOp::Store),
                )
                .with_depth_stencil(
                    // Reversed-Z clear (ADR-038) — must match the editor
                    // camera's reversed projection and the picking material's
                    // GreaterEqual compare.
                    DepthStencilAttachment::from_texture(self.depth_texture.clone())
                        .with_clear_depth(DepthConvention::default().clear_depth()),
                ),
        );

        if let Some(viewport) = &self.viewport {
            pass.set_viewport(*viewport);
        }
        if let Some(scissor) = &self.scissor {
            pass.set_scissor_rect(*scissor);
        }

        // The picking pipeline is specialized per mesh vertex layout (see
        // `entity_index_material_for`); group 0 binds the shared entity-index
        // ring, the per-draw offset selecting the entity's slot.
        for (entity_idx, renderer) in renderers.iter() {
            if let Some(vis) = visibilities.get(entity_idx)
                && !vis.is_visible()
            {
                continue;
            }
            let ei_off = self.picking_offsets.get(&entity_idx).copied().unwrap_or(0);
            for primitive in &renderer.primitives {
                // Skip primitives whose mesh hasn't finished loading (0 draws while
                // meshes stream in is expected, not an error).
                let Some(mesh) = primitive.mesh() else {
                    continue;
                };
                let Some((material, group)) = self.entity_index_material_for(mesh.layout()) else {
                    continue;
                };
                let instance = Arc::new(MaterialInstance::new(material).with_binding_group(group));
                pass.add_draw_command(
                    DrawCommand::new(mesh, instance).with_dynamic_offsets(vec![vec![ei_off]]),
                );
            }
        }

        Some(pass)
    }

    /// Build the selection-outline pass: a fullscreen triangle over the scene
    /// camera's color target drawing a contour around the selection mask the
    /// entity-index pass wrote this frame. Returns `None` when no selected
    /// renderer was flagged (by [`fill_picking_rings`](Self::fill_picking_rings))
    /// or the viewport/camera target is missing.
    ///
    /// Graph ordering is the caller's job: after the entity-index pass (mask
    /// producer) and the CameraTarget's last writer, before the egui overlay
    /// that samples the target.
    pub fn build_selection_outline_pass(&mut self, world: &World) -> Option<GraphicsPass> {
        if !self.has_selection {
            return None;
        }
        // Panel origin inside the window-sized mask. Headless has no panel —
        // the mask and the target are the same size, offset zero.
        let (offset_x, offset_y) = self.viewport.as_ref().map_or((0.0, 0.0), |vp| (vp.x, vp.y));
        // The scene camera's color target (panel-sized; the mask is
        // window-sized, offset by the panel origin).
        let color = world
            .read_all::<CameraTarget>()
            .ok()
            .and_then(|t| t.iter().next().map(|(_, target)| target.color.clone()))?;

        let (mask_w, mask_h) = self.last_size;
        let params = SelectionOutlineUniforms {
            mask_offset: [offset_x, offset_y],
            mask_size: [mask_w as f32, mask_h as f32],
            color: SELECTION_OUTLINE_COLOR,
            thickness: SELECTION_OUTLINE_THICKNESS,
            _padding: [0.0; 3],
        };
        let offset = Self::ring_push(&mut self.outline_ring, bytemuck::bytes_of(&params));

        let mut pass = GraphicsPass::new("selection_outline".into());
        // Load the scene contents; only outline pixels are written (the
        // fragment shader discards everything else).
        pass.set_render_targets(
            RenderTargetConfig::new()
                .with_color(ColorAttachment::new(RenderTarget::from_texture(color))),
        );
        let instance = Arc::new(
            MaterialInstance::new(Arc::clone(&self.outline_material))
                .with_binding_group(Arc::clone(&self.outline_group)),
        );
        pass.add_draw_command(
            DrawCommand::new(self.outline_mesh.clone(), instance)
                .with_dynamic_offsets(vec![vec![offset]]),
        );
        Some(pass)
    }

    /// Build a transfer pass that copies the picked pixel from the
    /// entity-index texture (bytes 0..4) and the pick-depth texture (at
    /// [`PICK_DEPTH_OFFSET`]) into the readback buffer, and snapshot the
    /// unprojection inputs for [`resolve_pick`](Self::resolve_pick).
    pub fn build_pick_readback(&mut self, px: u32, py: u32) -> TransferPass {
        let (w, h) = self.last_size;
        let px = px.min(w.saturating_sub(1));
        let py = py.min(h.saturating_sub(1));

        self.pick_snapshot = self.picking_view_projection.map(|vp| PickSnapshot {
            px,
            py,
            view_projection: vp,
            viewport: self
                .viewport
                .as_ref()
                .map_or([0.0, 0.0, w as f32, h as f32], |v| {
                    [v.x, v.y, v.width, v.height]
                }),
        });

        let origin = TextureCopyLocation::new(0, TextureOrigin::new(px, py, 0));
        let one_texel = redlilium_graphics::Extent3d {
            width: 1,
            height: 1,
            depth: 1,
        };
        let index_region =
            BufferTextureCopyRegion::new(BufferTextureLayout::packed(), origin, one_texel);
        let depth_region = BufferTextureCopyRegion::new(
            BufferTextureLayout::new(PICK_DEPTH_OFFSET, None, None),
            origin,
            one_texel,
        );

        // Clear any stale result so `resolve_pick` only sees this readback.
        if let Ok(mut g) = self.pick_result.lock() {
            g.clear();
        }

        let mut pass = TransferPass::new("pick_readback".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(vec![
            // 1. Copy the picked texels from the entity-index and pick-depth
            //    textures into the host-visible readback buffer.
            TransferOperation::readback_texture(
                self.entity_index_texture.clone(),
                self.readback_buffer.clone(),
                vec![index_region],
            ),
            TransferOperation::readback_texture(
                self.pick_depth_texture.clone(),
                self.readback_buffer.clone(),
                vec![depth_region],
            ),
            // 2. After the fence, the frame pipeline copies the buffer into
            //    `pick_result` for `resolve_pick` to poll.
            TransferOperation::readback_buffer(
                self.readback_buffer.clone(),
                0..(PICK_DEPTH_OFFSET as usize + 4),
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
    ///
    /// Retry guard (#33): if `readback_buffer`'s previous async readback map is
    /// still in flight, DON'T start a new pick this frame — issuing a
    /// `TextureToBuffer` into a still-mapped buffer is the write-while-mapped
    /// hazard, and it would clobber the in-flight result. The pending pick is
    /// kept and retried next frame; the map callback clears the flag
    /// unconditionally, so this cannot wedge (the counter is only a diagnostic).
    pub fn take_pending_pick(&mut self) -> Option<[u32; 2]> {
        if self.pending_pick.is_none() {
            self.pick_wait_frames = 0;
            return None;
        }
        if self.readback_buffer.is_map_pending() {
            self.pick_wait_frames += 1;
            if self.pick_wait_frames == 60 {
                log::warn!(
                    "pick held {} frames waiting on the readback map to resolve — possible \
                     stuck readback",
                    self.pick_wait_frames
                );
            }
            return None;
        }
        self.pick_wait_frames = 0;
        self.pending_pick.take()
    }

    /// Poll the pick result, filled asynchronously by the frame pipeline once
    /// the GPU readback completes.
    ///
    /// The outer `Option` is readiness (`None` while the GPU readback is in
    /// flight); the [`PickHit`] carries the hit entity (`None` for empty
    /// space) and the depth-derived world-space surface point. A miss is a
    /// completed result, not a pending one: remote picks must answer it. The
    /// result is consumed once read.
    pub fn resolve_pick(&mut self) -> Option<PickHit> {
        let data = {
            let mut guard = self.pick_result.lock().ok()?;
            if guard.len() < PICK_DEPTH_OFFSET as usize + 4 {
                return None; // not ready yet
            }
            std::mem::take(&mut *guard)
        };
        let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let d = PICK_DEPTH_OFFSET as usize;
        let depth = f32::from_le_bytes([data[d], data[d + 1], data[d + 2], data[d + 3]]);

        let snapshot = self.pick_snapshot.take();
        // Depth 0 is the cleared reversed-Z background — no surface there.
        let world_point = if depth > 0.0 {
            snapshot.and_then(|s| s.unproject(depth))
        } else {
            None
        };
        let entity = if value == 0 {
            None // cleared background — no entity
        } else {
            Some(value - 1) // shader wrote entity_index + 1
        };
        Some(PickHit {
            entity,
            world_point,
        })
    }

    // ---- Rect selection readback ----

    /// Request a rect readback at the given physical-pixel rectangle.
    ///
    /// The result will be available after 2 frames via [`resolve_rect_pick`].
    /// Any buffer resize happens in [`take_pending_rect_pick`], AFTER the
    /// map-pending guard — reallocating here could replace `rect_readback_buffer`
    /// while a prior async map is still in flight, orphaning its pending flag and
    /// racing the old callback against the new readback (#33).
    pub fn request_rect_pick(&mut self, x: u32, y: u32, w: u32, h: u32) {
        let (tex_w, tex_h) = self.last_size;
        let x = x.min(tex_w.saturating_sub(1));
        let y = y.min(tex_h.saturating_sub(1));
        let w = w.min(tex_w - x).max(1);
        let h = h.min(tex_h - y).max(1);

        self.pending_rect_pick = Some([x, y, w, h]);
    }

    /// Take the pending rect pick coordinates (consumed to build the readback pass).
    ///
    /// Same retry guard as [`take_pending_pick`](Self::take_pending_pick): while
    /// `rect_readback_buffer`'s prior async map is in flight, keep the pending
    /// pick and retry next frame. Once the buffer is free, resize it here if the
    /// rectangle needs more room — safe only because no map is outstanding.
    pub fn take_pending_rect_pick(&mut self) -> Option<[u32; 4]> {
        let Some([_, _, w, h]) = self.pending_rect_pick else {
            self.rect_wait_frames = 0;
            return None;
        };
        if self.rect_readback_buffer.is_map_pending() {
            self.rect_wait_frames += 1;
            if self.rect_wait_frames == 60 {
                log::warn!(
                    "rect pick held {} frames waiting on the readback map to resolve — possible \
                     stuck readback",
                    self.rect_wait_frames
                );
            }
            return None;
        }
        self.rect_wait_frames = 0;

        // Buffer is free (no map in flight): safe to grow it for this rectangle.
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

    /// The scene color format (the surface format the view was created with).
    pub fn color_format(&self) -> TextureFormat {
        self.color_format
    }

    /// Set the forward `FrameRing`'s buffer (the resource it wraps), so
    /// per-primitive materials can bind it. Call once, before creating entities.
    pub fn set_frame_ring_buffer(&mut self, buffer: Arc<Buffer>) {
        self.frame_ring_buffer = Some(buffer);
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
                    // TEXTURE_BINDING: the deferred path reconstructs world
                    // position from the depth buffer (resolve/SSAO/TAA/MB).
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
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

    fn create_pick_depth_texture(
        device: &Arc<GraphicsDevice>,
        width: u32,
        height: u32,
    ) -> Arc<redlilium_graphics::Texture> {
        device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::R32Float,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
                )
                .with_label("scene_view_pick_depth"),
            )
            .expect("Failed to create scene view pick depth texture")
    }

    fn create_selection_mask_texture(
        device: &Arc<GraphicsDevice>,
        width: u32,
        height: u32,
    ) -> Arc<redlilium_graphics::Texture> {
        device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::R8Unorm,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("scene_view_selection_mask"),
            )
            .expect("Failed to create scene view selection mask texture")
    }

    fn create_outline_group(
        device: &Arc<GraphicsDevice>,
        material: &Arc<Material>,
        ring: &RingBuffer,
        mask: &Arc<redlilium_graphics::Texture>,
    ) -> Arc<BindingGroup> {
        device
            .create_binding_group(
                material.binding_layouts()[0].clone(),
                BindingGroupDescriptor::new()
                    .with_buffer_range(
                        0,
                        ring.buffer().clone(),
                        0,
                        std::mem::size_of::<SelectionOutlineUniforms>() as u64,
                    )
                    .with_texture(1, mask.clone()),
            )
            .expect("Failed to create selection outline binding group")
    }
}
