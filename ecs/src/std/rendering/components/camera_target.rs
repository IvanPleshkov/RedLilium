//! Camera target component.

use std::sync::Arc;

use redlilium_graphics::Texture;

/// Render target for a camera entity.
///
/// Specifies which textures the camera renders to. Attach this to an entity
/// that already has a [`Camera`](crate::Camera) component. The forward render
/// system will create a graphics pass for each camera that has a `CameraTarget`.
///
/// The color and depth textures must be created with `TextureUsage::RENDER_ATTACHMENT`.
#[derive(Debug, Clone, crate::Component)]
#[skip_serialization]
pub struct CameraTarget {
    /// Color texture to render to.
    pub color: Arc<Texture>,
    /// Depth texture for depth testing.
    pub depth: Arc<Texture>,
    /// Clear color (RGBA) applied at the start of the render pass.
    pub clear_color: [f32; 4],
}

impl CameraTarget {
    /// Create a new camera target with the given textures and clear color.
    pub fn new(color: Arc<Texture>, depth: Arc<Texture>, clear_color: [f32; 4]) -> Self {
        Self {
            color,
            depth,
            clear_color,
        }
    }
}
