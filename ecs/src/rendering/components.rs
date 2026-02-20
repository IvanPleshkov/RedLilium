use std::sync::Arc;

use redlilium_graphics::{MaterialInstance, Mesh, Texture};

/// GPU mesh component.
///
/// Wraps an `Arc<Mesh>` (GPU-uploaded mesh) so it can be attached to entities.
/// Entities with both `RenderMesh` and [`RenderMaterial`] are collected by the
/// forward render system and drawn each frame.
#[derive(Debug, Clone, crate::Component)]
#[require(crate::Transform, crate::GlobalTransform, crate::Visibility)]
#[skip_serialization]
pub struct RenderMesh(pub Arc<Mesh>);

impl RenderMesh {
    /// Create a new render mesh component.
    pub fn new(mesh: Arc<Mesh>) -> Self {
        Self(mesh)
    }

    /// Get the inner GPU mesh.
    pub fn mesh(&self) -> &Arc<Mesh> {
        &self.0
    }
}

/// GPU material instance component.
///
/// Wraps an `Arc<MaterialInstance>` containing bound shader resources.
/// Attach alongside [`RenderMesh`] to make an entity renderable.
#[derive(Debug, Clone, crate::Component)]
#[skip_serialization]
pub struct RenderMaterial(pub Arc<MaterialInstance>);

impl RenderMaterial {
    /// Create a new render material component.
    pub fn new(material: Arc<MaterialInstance>) -> Self {
        Self(material)
    }

    /// Get the inner material instance.
    pub fn material(&self) -> &Arc<MaterialInstance> {
        &self.0
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
