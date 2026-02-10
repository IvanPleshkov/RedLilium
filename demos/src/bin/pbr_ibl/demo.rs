//! PBR IBL Demo application.

use std::sync::{Arc, RwLock};

use redlilium_app::{AppContext, AppHandler, DrawContext};
use redlilium_core::profiling::{
    profile_function, profile_memory_stats, profile_message, profile_scope,
};
use redlilium_graphics::{
    BufferUsage, ColorAttachment, DepthStencilAttachment, FrameSchedule, GraphicsPass, LoadOp,
    RenderTarget, RenderTargetConfig, RingAllocation, TransferPass, egui::EguiController,
};
use winit::event::KeyEvent;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::camera::OrbitCamera;
use crate::ecs_scene::EcsScene;
use crate::gbuffer::GBuffer;
use crate::ibl_textures::IblTextures;
use crate::resolve_pass::ResolvePass;
use crate::skybox_pass::SkyboxPass;
use crate::sphere_grid::SphereGrid;
use crate::ui::PbrUi;
use crate::uniforms::{CameraUniforms, ResolveUniforms, SkyboxUniforms};
use crate::{GRID_SIZE, PREFILTER_SIZE};

/// Per-frame uniform allocation offsets from the ring buffer.
#[derive(Default)]
#[allow(dead_code)]
struct FrameUniformAllocations {
    camera: Option<RingAllocation>,
    skybox: Option<RingAllocation>,
    resolve: Option<RingAllocation>,
}

/// The main PBR IBL demo application.
pub struct PbrIblDemo {
    camera: OrbitCamera,
    mouse_pressed: bool,
    last_mouse_x: f64,
    last_mouse_y: f64,
    shift_pressed: bool,
    needs_instance_update: bool,
    hdr_active: bool,
    frame_allocations: FrameUniformAllocations,

    // ECS scene
    ecs_scene: Option<EcsScene>,

    // Subsystems (populated in on_init)
    gbuffer: Option<GBuffer>,
    ibl: Option<IblTextures>,
    spheres: Option<SphereGrid>,
    skybox: Option<SkyboxPass>,
    resolve: Option<ResolvePass>,

    // UI
    egui_controller: Option<EguiController>,
    egui_ui: Arc<RwLock<PbrUi>>,
}

impl PbrIblDemo {
    pub fn new() -> Self {
        Self {
            camera: OrbitCamera::new(),
            mouse_pressed: false,
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            shift_pressed: false,
            needs_instance_update: false,
            hdr_active: false,
            frame_allocations: FrameUniformAllocations::default(),
            ecs_scene: None,
            gbuffer: None,
            ibl: None,
            spheres: None,
            skybox: None,
            resolve: None,
            egui_controller: None,
            egui_ui: Arc::new(RwLock::new(PbrUi::new())),
        }
    }
}

impl Default for PbrIblDemo {
    fn default() -> Self {
        Self::new()
    }
}

impl AppHandler for PbrIblDemo {
    fn on_init(&mut self, ctx: &mut AppContext) {
        profile_function!();
        profile_message!("PBR Demo: Initializing");

        log::info!("Initializing Deferred PBR IBL Demo");
        log::info!(
            "Grid: {}x{} spheres with varying metallic/roughness",
            GRID_SIZE,
            GRID_SIZE
        );
        log::info!("Deferred rendering with G-buffer + IBL resolve pass");
        log::info!(
            "Surface format: {:?}, HDR: {}",
            ctx.surface_format(),
            ctx.hdr_active()
        );
        log::info!("Controls:");
        log::info!("  - Left mouse drag: Rotate camera");
        log::info!("  - Scroll: Zoom");
        log::info!("  - H: Toggle UI visibility");

        self.hdr_active = ctx.hdr_active();
        let device = ctx.device();

        // Create ECS scene
        let ecs_scene = EcsScene::new(ctx.aspect_ratio());
        self.ecs_scene = Some(ecs_scene);

        // Create subsystems
        let ibl = IblTextures::create(device);
        let mut gbuffer = GBuffer::create(device, ctx.width(), ctx.height());
        let spheres = SphereGrid::create(device);
        let skybox = SkyboxPass::create(device, &ibl, ctx.surface_format());
        let resolve = ResolvePass::create(
            device,
            &gbuffer,
            &ibl,
            ctx.surface_format(),
            self.hdr_active,
        );

        // Initialize per-frame ring buffer
        ctx.pipeline_mut()
            .create_ring_buffers(
                4 * 1024,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                "per_frame_uniforms",
            )
            .expect("Failed to create per-frame ring buffers");
        log::info!(
            "Created per-frame ring buffers: {} frames x {} bytes",
            ctx.pipeline().frames_in_flight(),
            ctx.pipeline().ring_buffer_capacity().unwrap_or(0)
        );

        // Initialize egui controller
        let mut egui_controller = EguiController::new(
            ctx.device().clone(),
            self.egui_ui.clone(),
            ctx.width(),
            ctx.height(),
            ctx.scale_factor(),
            ctx.surface_format(),
        );

        // Register G-buffer textures with egui for UI preview
        gbuffer.register_with_egui(&mut egui_controller);

        // Pass texture IDs to UI
        if let Ok(mut ui) = self.egui_ui.write() {
            ui.set_gbuffer_texture_ids(
                gbuffer.albedo_egui_id,
                gbuffer.normal_egui_id,
                gbuffer.position_egui_id,
            );
        }

        self.egui_controller = Some(egui_controller);
        self.ibl = Some(ibl);
        self.gbuffer = Some(gbuffer);
        self.spheres = Some(spheres);
        self.skybox = Some(skybox);
        self.resolve = Some(resolve);
    }

    fn on_resize(&mut self, ctx: &mut AppContext) {
        // Recreate G-buffer at new size
        let mut gbuffer = GBuffer::create(ctx.device(), ctx.width(), ctx.height());

        // Carry over egui IDs from old gbuffer
        if let Some(old_gbuffer) = &self.gbuffer {
            gbuffer.albedo_egui_id = old_gbuffer.albedo_egui_id;
            gbuffer.normal_egui_id = old_gbuffer.normal_egui_id;
            gbuffer.position_egui_id = old_gbuffer.position_egui_id;
        }

        if let Some(egui) = &mut self.egui_controller {
            gbuffer.update_egui_textures(egui);
            egui.on_resize(ctx.width(), ctx.height());
        }

        // Recreate resolve pass with new G-buffer textures
        if let Some(ibl) = &self.ibl {
            self.resolve = Some(ResolvePass::create(
                ctx.device(),
                &gbuffer,
                ibl,
                ctx.surface_format(),
                self.hdr_active,
            ));
        }

        self.gbuffer = Some(gbuffer);

        // Update camera projection for new aspect ratio
        if let Some(scene) = &mut self.ecs_scene {
            scene.update_camera_projection(ctx.aspect_ratio());
        }
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        profile_scope!("on_update");

        // Process UI state changes
        if let Ok(mut ui) = self.egui_ui.write() {
            if ui.take_state_changed() {
                let state = ui.state().clone();

                if let Some(skybox) = &mut self.skybox {
                    skybox.mip_level = state.skybox_mip_level;
                }

                self.camera.distance = state.camera_distance;
                self.needs_instance_update = true;
            }

            // Sync camera distance back to UI
            ui.set_camera_distance(self.camera.distance);

            // Auto-rotate if enabled
            if ui.state().auto_rotate && !self.mouse_pressed {
                self.camera.rotate(ctx.delta_time() * 0.15, 0.0);
            }
        }

        // Update ECS scene from orbit camera and run systems
        if let Some(scene) = &mut self.ecs_scene {
            // Sync orbit camera → ECS camera entity transform
            scene.update_camera_transform(self.camera.position(), self.camera.target);

            // Update sphere properties if UI changed
            if self.needs_instance_update
                && let Ok(ui) = self.egui_ui.read()
            {
                let state = ui.state();
                let base_color = [
                    state.base_color[0],
                    state.base_color[1],
                    state.base_color[2],
                    1.0,
                ];
                scene.update_spheres(base_color, state.sphere_spacing);
            }

            // Run ECS systems (UpdateGlobalTransforms → UpdateCameraMatrices)
            scene.run_systems();

            // Extract camera data from ECS
            let (view, proj) = scene.camera_matrices();
            let camera_pos = scene.camera_position();

            // Write GPU buffers from ECS data
            if let Some(spheres) = &self.spheres {
                spheres.write_camera_uniforms(ctx.device(), view, proj, camera_pos);

                // Build instance data from ECS entities
                let instances = scene.build_sphere_instances();
                spheres.write_instances(ctx.device(), &instances);
            }
            if let Some(skybox) = &self.skybox {
                skybox.update_uniforms(ctx.device(), view, proj, camera_pos);
            }
            if let Some(resolve) = &self.resolve {
                resolve.update_uniforms(ctx.device(), camera_pos, ctx.width(), ctx.height());
            }
        }

        self.needs_instance_update = false;

        true
    }

    fn on_draw(&mut self, mut ctx: DrawContext) -> FrameSchedule {
        profile_scope!("on_draw");

        // Allocate per-frame uniforms from ring buffer
        if ctx.has_ring_buffer() {
            self.frame_allocations = FrameUniformAllocations {
                camera: ctx.allocate(std::mem::size_of::<CameraUniforms>() as u64),
                skybox: ctx.allocate(std::mem::size_of::<SkyboxUniforms>() as u64),
                resolve: ctx.allocate(std::mem::size_of::<ResolveUniforms>() as u64),
            };

            if let Some(ring) = ctx.ring_buffer()
                && ctx.frame_number() < 3
            {
                log::debug!(
                    "Ring buffer slot {}: allocated {} bytes, {} remaining",
                    ctx.frame_slot(),
                    ring.used(),
                    ring.remaining()
                );
            }
        }

        let mut graph = ctx.acquire_graph();

        // Upload IBL textures on first frame
        if let Some(ibl) = &mut self.ibl
            && let Some(transfer_config) = ibl.take_transfer_config()
        {
            let mut transfer_pass = TransferPass::new("ibl_upload".into());
            transfer_pass.set_transfer_config(transfer_config);
            graph.add_transfer_pass(transfer_pass);
            log::info!("IBL textures uploaded via transfer pass");
        }

        // === Pass 1: G-Buffer Pass ===
        let mut gbuffer_pass = GraphicsPass::new("gbuffer".into());

        if let Some(gbuffer) = &self.gbuffer {
            gbuffer_pass.set_render_targets(
                RenderTargetConfig::new()
                    .with_color(
                        ColorAttachment::from_texture(gbuffer.albedo.clone())
                            .with_clear_color(0.0, 0.0, 0.0, 0.0),
                    )
                    .with_color(
                        ColorAttachment::from_texture(gbuffer.normal_metallic.clone())
                            .with_clear_color(0.5, 0.5, 0.5, 0.0),
                    )
                    .with_color(
                        ColorAttachment::from_texture(gbuffer.position_roughness.clone())
                            .with_clear_color(0.0, 0.0, 0.0, 0.0),
                    )
                    .with_depth_stencil(
                        DepthStencilAttachment::from_texture(gbuffer.depth.clone())
                            .with_clear_depth(1.0),
                    ),
            );
        }

        if let Some(spheres) = &self.spheres {
            let show_wireframe = self
                .egui_ui
                .read()
                .map(|ui| ui.state().show_wireframe)
                .unwrap_or(false);
            let material = if show_wireframe {
                spheres.wireframe_material_instance.clone()
            } else {
                spheres.material_instance.clone()
            };
            gbuffer_pass.add_draw_instanced(
                spheres.mesh.clone(),
                material,
                (GRID_SIZE * GRID_SIZE) as u32,
            );
        }

        graph.add_graphics_pass(gbuffer_pass);

        // === Pass 2: Skybox Pass ===
        let mut skybox_pass = GraphicsPass::new("skybox".into());

        skybox_pass.set_render_targets(
            RenderTargetConfig::new().with_color(
                ColorAttachment::from_surface(ctx.swapchain_texture())
                    .with_clear_color(0.02, 0.02, 0.03, 1.0),
            ),
        );

        if let Some(skybox) = &self.skybox {
            skybox_pass.add_draw(skybox.mesh.clone(), skybox.material_instance.clone());
        }

        graph.add_graphics_pass(skybox_pass);

        // === Pass 3: Resolve/Lighting Pass ===
        let mut resolve_pass = GraphicsPass::new("resolve".into());

        resolve_pass.set_render_targets(RenderTargetConfig::new().with_color(
            ColorAttachment::from_surface(ctx.swapchain_texture()).with_load_op(LoadOp::Load),
        ));

        if let Some(resolve) = &self.resolve {
            resolve_pass.add_draw(resolve.mesh.clone(), resolve.material_instance.clone());
        }

        let resolve_handle = graph.add_graphics_pass(resolve_pass);

        // === Pass 4: Egui Pass ===
        if let Some(egui) = &mut self.egui_controller {
            let width = ctx.width();
            let height = ctx.height();
            let elapsed = ctx.elapsed_time() as f64;
            let render_target = RenderTarget::from_surface(ctx.swapchain_texture());

            egui.begin_frame(elapsed);
            if let Some(egui_pass) = egui.end_frame(&render_target, width, height) {
                let egui_handle = graph.add_graphics_pass(egui_pass);
                graph.add_dependency(egui_handle, resolve_handle);
            }
        }

        let _handle = ctx.submit("main", graph, &[]);

        profile_memory_stats!();

        ctx.finish(&[])
    }

    fn on_mouse_move(&mut self, _ctx: &mut AppContext, x: f64, y: f64) {
        let egui_wants_pointer = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_move(x, y)
        } else {
            false
        };

        if self.mouse_pressed && !egui_wants_pointer {
            let dx = (x - self.last_mouse_x) as f32 * 0.005;
            let dy = (y - self.last_mouse_y) as f32 * 0.005;
            self.camera.rotate(-dx, -dy);
        }
        self.last_mouse_x = x;
        self.last_mouse_y = y;
    }

    fn on_mouse_button(
        &mut self,
        _ctx: &mut AppContext,
        button: winit::event::MouseButton,
        pressed: bool,
    ) {
        let egui_wants_pointer = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_button(button, pressed)
        } else {
            false
        };

        if button == winit::event::MouseButton::Left && !egui_wants_pointer {
            self.mouse_pressed = pressed;
        }
    }

    fn on_mouse_scroll(&mut self, _ctx: &mut AppContext, _dx: f32, dy: f32) {
        let egui_wants_pointer = if let Some(egui) = &mut self.egui_controller {
            egui.on_mouse_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, dy))
        } else {
            false
        };

        if !egui_wants_pointer {
            self.camera.zoom(dy * 0.5);
        }
    }

    fn on_key(&mut self, _ctx: &mut AppContext, event: &KeyEvent) {
        if let Some(egui) = &mut self.egui_controller {
            egui.on_key(event);
        }

        // Track shift key state
        if let PhysicalKey::Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) = event.physical_key {
            self.shift_pressed = event.state.is_pressed();
            return;
        }

        if !event.state.is_pressed() {
            return;
        }

        // H key toggles UI visibility
        if let PhysicalKey::Code(KeyCode::KeyH) = event.physical_key {
            if let Ok(mut ui) = self.egui_ui.write() {
                ui.toggle_visibility();
            }
            return;
        }

        // Shift + digit: change skybox MIP level
        if self.shift_pressed {
            let mip_levels = (PREFILTER_SIZE as f32).log2().floor() as u32 + 1;
            let max_mip = mip_levels as f32 - 1.0;

            let new_mip = match event.physical_key {
                PhysicalKey::Code(KeyCode::Digit0) => Some(0.0),
                PhysicalKey::Code(KeyCode::Digit1) => Some(1.0_f32.min(max_mip)),
                PhysicalKey::Code(KeyCode::Digit2) => Some(2.0_f32.min(max_mip)),
                PhysicalKey::Code(KeyCode::Digit3) => Some(3.0_f32.min(max_mip)),
                PhysicalKey::Code(KeyCode::Digit4) => Some(4.0_f32.min(max_mip)),
                PhysicalKey::Code(KeyCode::Digit5) => Some(5.0_f32.min(max_mip)),
                PhysicalKey::Code(KeyCode::Digit6) => Some(6.0_f32.min(max_mip)),
                PhysicalKey::Code(KeyCode::Digit7) => Some(7.0_f32.min(max_mip)),
                _ => None,
            };

            if let Some(mip) = new_mip
                && let Some(skybox) = &mut self.skybox
            {
                skybox.mip_level = mip;
                log::info!("Skybox MIP level: {}", mip);
            }
        }
    }

    fn on_shutdown(&mut self, _ctx: &mut AppContext) {
        log::info!("Shutting down PBR IBL Demo");
    }
}
