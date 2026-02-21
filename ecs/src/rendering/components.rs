use std::sync::Arc;

use redlilium_graphics::{MaterialInstance, Mesh, Texture};

/// GPU mesh component.
///
/// Wraps an `Arc<Mesh>` (GPU-uploaded mesh) so it can be attached to entities.
/// Entities with both `RenderMesh` and [`RenderMaterial`] are collected by the
/// forward render system and drawn each frame.
#[derive(Debug, Clone)]
pub struct RenderMesh(pub Arc<Mesh>);

impl crate::Component for RenderMesh {
    const NAME: &'static str = "RenderMesh";

    fn inspect_ui(&self, ui: &mut crate::egui::Ui) -> Option<Self> {
        ui.horizontal(|ui| {
            ui.label("mesh");
            match self.0.label() {
                Some(label) => ui.label(format!("Mesh: {label}")),
                None => ui.weak("Mesh (unnamed)"),
            };
        });
        None
    }

    fn collect_entities(&self, _collector: &mut Vec<crate::Entity>) {}

    fn remap_entities(&mut self, _map: &mut dyn FnMut(crate::Entity) -> crate::Entity) {}

    fn register_required(world: &mut crate::World) {
        world.register_required::<Self, crate::Transform>();
        world.register_required::<Self, crate::GlobalTransform>();
        world.register_required::<Self, crate::Visibility>();
    }

    fn serialize_component(
        &self,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<crate::serialize::Value, crate::serialize::SerializeError> {
        let mesh_name = {
            let world = ctx.world();
            if !world.has_resource::<super::MeshManager>() {
                return Err(crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "MeshManager resource not found".into(),
                });
            }
            let manager = world.resource::<super::MeshManager>();
            manager
                .find_name(&self.0)
                .or_else(|| self.0.label())
                .ok_or_else(|| crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "mesh has no registered name and no label".into(),
                })?
                .to_owned()
        };
        ctx.begin_struct(Self::NAME)?;
        ctx.write_serde("0", &mesh_name)?;
        ctx.end_struct()
    }

    fn deserialize_component(
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.begin_struct(Self::NAME)?;
        let mesh_name: String = ctx.read_serde("0")?;
        let world = ctx.world();
        if !world.has_resource::<super::MeshManager>() {
            return Err(crate::serialize::DeserializeError::FormatError(
                "MeshManager resource not found".into(),
            ));
        }
        let manager = world.resource::<super::MeshManager>();
        let mesh = manager.get_mesh(&mesh_name).ok_or_else(|| {
            crate::serialize::DeserializeError::FormatError(format!(
                "mesh '{mesh_name}' not found in MeshManager"
            ))
        })?;
        let mesh = Arc::clone(mesh);
        drop(manager);
        ctx.end_struct()?;
        Ok(Self(mesh))
    }
}

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
