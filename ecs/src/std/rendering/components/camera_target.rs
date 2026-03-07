//! Camera target and per-entity buffer components.

use std::sync::Arc;

use redlilium_graphics::{Buffer, Texture};

/// Per-entity GPU uniform buffers for transform data (VP + model matrix).
///
/// Holds the forward-pass uniform buffer and an optional entity-index pass
/// buffer. The [`UpdatePerEntityUniforms`](super::super::UpdatePerEntityUniforms)
/// system writes camera and transform data into these buffers each frame.
#[derive(Debug, Clone, crate::Component)]
#[skip_serialization]
pub struct PerEntityBuffers {
    /// Forward pass uniform buffer (VP + model matrices).
    pub forward_buffer: Arc<Buffer>,
    /// Entity-index pass uniform buffer (VP + model + entity index), if present.
    pub entity_index_buffer: Option<Arc<Buffer>>,
}

impl PerEntityBuffers {
    /// Create per-entity buffers with forward pass only.
    pub fn new(forward_buffer: Arc<Buffer>) -> Self {
        Self {
            forward_buffer,
            entity_index_buffer: None,
        }
    }

    /// Create per-entity buffers with forward and entity-index passes.
    pub fn with_entity_index(
        forward_buffer: Arc<Buffer>,
        entity_index_buffer: Arc<Buffer>,
    ) -> Self {
        Self {
            forward_buffer,
            entity_index_buffer: Some(entity_index_buffer),
        }
    }
}

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
