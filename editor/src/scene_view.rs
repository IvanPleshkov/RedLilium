//! Scene view state for rendering ECS entities into the editor's SceneView panel.
//!
//! Reads Camera, GlobalTransform, RenderMesh, RenderMaterial, and Visibility
//! from the ECS World and builds a forward rendering pass targeting the
//! swapchain with viewport/scissor matching the egui panel rect.

use std::sync::Arc;

use redlilium_ecs::{
    Entity, MaterialBundle, RenderMaterial, RenderMesh, RenderPassType, Visibility, World, shaders,
};
use redlilium_graphics::{
    Buffer, ColorAttachment, DepthStencilAttachment, GraphicsDevice, GraphicsPass, Material,
    RenderTargetConfig, ScissorRect, SurfaceTexture, TextureDescriptor, TextureFormat,
    TextureUsage, Viewport,
};

/// Manages GPU resources and rendering for the editor's SceneView panel.
pub struct SceneViewState {
    device: Arc<GraphicsDevice>,
    depth_texture: Arc<redlilium_graphics::Texture>,
    opaque_material: Arc<Material>,
    viewport: Option<Viewport>,
    scissor: Option<ScissorRect>,
    last_size: (u32, u32),
}

impl SceneViewState {
    /// Create scene view resources.
    pub fn new(device: Arc<GraphicsDevice>, surface_format: TextureFormat) -> Self {
        let (opaque_material, _layout) = shaders::create_opaque_color_material(
            &device,
            surface_format,
            TextureFormat::Depth32Float,
        );

        let depth_texture = Self::create_depth_texture(&device, 256, 256);

        Self {
            device,
            depth_texture,
            opaque_material,
            viewport: None,
            scissor: None,
            last_size: (256, 256),
        }
    }

    /// Create GPU resources for a renderable entity: uniform buffer + MaterialBundle.
    ///
    /// Returns `(uniform_buffer, gpu_mesh, material_bundle)`.
    pub fn create_entity_resources(
        &self,
        cpu_mesh: &redlilium_core::mesh::CpuMesh,
    ) -> (
        Arc<Buffer>,
        Arc<redlilium_graphics::Mesh>,
        Arc<MaterialBundle>,
    ) {
        let (uniform_buffer, bundle) =
            shaders::create_opaque_color_entity(&self.device, &self.opaque_material);

        let gpu_mesh = self
            .device
            .create_mesh_from_cpu(cpu_mesh)
            .expect("Failed to create entity GPU mesh");

        (uniform_buffer, gpu_mesh, bundle)
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

    /// Recreate the depth texture if the window size changed.
    pub fn resize_if_needed(&mut self, width: u32, height: u32) -> bool {
        let width = width.max(1);
        let height = height.max(1);

        if (width, height) == self.last_size {
            return false;
        }

        self.depth_texture = Self::create_depth_texture(&self.device, width, height);
        self.last_size = (width, height);
        true
    }

    /// Update per-entity uniform buffers with current camera VP and model matrices.
    pub fn update_uniforms(&self, world: &World, entity_buffers: &[(Entity, Arc<Buffer>)]) {
        shaders::update_opaque_color_uniforms(&self.device, world, entity_buffers);
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
}
