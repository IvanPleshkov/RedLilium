//! G-Buffer resources for deferred rendering.

use std::sync::Arc;

use redlilium_core::profiling::profile_scope;
use redlilium_graphics::{
    GraphicsDevice, TextureDescriptor, TextureFormat, TextureUsage,
    egui::{EguiController, egui},
};

/// G-Buffer textures for deferred rendering (depth + 3 MRT color targets).
pub struct GBuffer {
    pub depth: Arc<redlilium_graphics::Texture>,
    pub albedo: Arc<redlilium_graphics::Texture>,
    pub normal_metallic: Arc<redlilium_graphics::Texture>,
    pub position_roughness: Arc<redlilium_graphics::Texture>,
    // Egui preview texture IDs
    pub albedo_egui_id: Option<egui::TextureId>,
    pub normal_egui_id: Option<egui::TextureId>,
    pub position_egui_id: Option<egui::TextureId>,
}

impl GBuffer {
    /// Create G-Buffer textures for the given dimensions.
    pub fn create(device: &Arc<GraphicsDevice>, width: u32, height: u32) -> Self {
        profile_scope!("GBuffer::create");

        let depth = device
            .create_texture(&TextureDescriptor::new_2d(
                width,
                height,
                TextureFormat::Depth32Float,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .expect("Failed to create depth texture");

        // RT0: Albedo (sRGB for correct color)
        let albedo = device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::Rgba8UnormSrgb,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("gbuffer_albedo"),
            )
            .expect("Failed to create G-buffer albedo");

        // RT1: Normal (RGB) + Metallic (A) - high precision for normals
        let normal_metallic = device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::Rgba16Float,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("gbuffer_normal_metallic"),
            )
            .expect("Failed to create G-buffer normal/metallic");

        // RT2: Position (RGB) + Roughness (A) - high precision for world positions
        let position_roughness = device
            .create_texture(
                &TextureDescriptor::new_2d(
                    width,
                    height,
                    TextureFormat::Rgba16Float,
                    TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
                )
                .with_label("gbuffer_position_roughness"),
            )
            .expect("Failed to create G-buffer position/roughness");

        Self {
            depth,
            albedo,
            normal_metallic,
            position_roughness,
            albedo_egui_id: None,
            normal_egui_id: None,
            position_egui_id: None,
        }
    }

    /// Register G-Buffer textures with egui for UI preview.
    pub fn register_with_egui(&mut self, egui: &mut EguiController) {
        self.albedo_egui_id = Some(egui.register_user_texture(self.albedo.clone()));
        self.normal_egui_id = Some(egui.register_user_texture(self.normal_metallic.clone()));
        self.position_egui_id = Some(egui.register_user_texture(self.position_roughness.clone()));
    }

    /// Update egui texture bindings after resize.
    pub fn update_egui_textures(&self, egui: &mut EguiController) {
        if let Some(id) = self.albedo_egui_id {
            egui.update_user_texture(id, self.albedo.clone());
        }
        if let Some(id) = self.normal_egui_id {
            egui.update_user_texture(id, self.normal_metallic.clone());
        }
        if let Some(id) = self.position_egui_id {
            egui.update_user_texture(id, self.position_roughness.clone());
        }
    }
}
