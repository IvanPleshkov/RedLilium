//! GPU mesh component.

use std::sync::Arc;

use redlilium_graphics::Mesh;

/// GPU mesh component.
///
/// Wraps an `Arc<Mesh>` (GPU-uploaded mesh) so it can be attached to entities.
/// Entities with both `RenderMesh` and [`RenderMaterial`](super::RenderMaterial)
/// are collected by the forward render system and drawn each frame.
#[derive(Debug, Clone)]
pub struct RenderMesh {
    /// The GPU mesh handle.
    pub mesh: Arc<Mesh>,
    /// Cached local-space AABB (computed from CPU mesh data at creation time).
    pub aabb: Option<redlilium_core::math::Aabb>,
}

impl crate::Component for RenderMesh {
    const NAME: &'static str = "RenderMesh";

    fn inspect_ui(
        &self,
        ui: &mut crate::egui::Ui,
        _world: &crate::World,
        _entity: crate::Entity,
    ) -> crate::InspectResult {
        ui.horizontal(|ui| {
            ui.label("mesh");
            match self.mesh.label() {
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

    fn aabb(&self, _world: &crate::World) -> Option<redlilium_core::math::Aabb> {
        self.aabb
    }

    fn serialize_component(
        &self,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<crate::serialize::Value, crate::serialize::SerializeError> {
        let mesh_name = {
            let world = ctx.world();
            if !world.has_resource::<super::super::MeshManager>() {
                return Err(crate::serialize::SerializeError::FieldError {
                    field: "mesh".to_owned(),
                    message: "MeshManager resource not found".into(),
                });
            }
            let manager = world.resource::<super::super::MeshManager>();
            manager
                .find_name(&self.mesh)
                .or_else(|| self.mesh.label())
                .ok_or_else(|| crate::serialize::SerializeError::FieldError {
                    field: "mesh".to_owned(),
                    message: "mesh has no registered name and no label".into(),
                })?
                .to_owned()
        };
        ctx.begin_struct(Self::NAME)?;
        ctx.write_serde("mesh", &mesh_name)?;
        ctx.end_struct()
    }

    fn deserialize_component(
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.begin_struct(Self::NAME)?;
        let mesh_name: String = ctx.read_serde("mesh")?;
        let (mesh, aabb) = {
            let world = ctx.world();
            if !world.has_resource::<super::super::MeshManager>() {
                return Err(crate::serialize::DeserializeError::FormatError(
                    "MeshManager resource not found".into(),
                ));
            }
            let manager = world.resource::<super::super::MeshManager>();
            let mesh = manager.get_mesh(&mesh_name).ok_or_else(|| {
                crate::serialize::DeserializeError::FormatError(format!(
                    "mesh '{mesh_name}' not found in MeshManager"
                ))
            })?;
            let mesh = Arc::clone(mesh);
            let aabb = manager.get_aabb_by_mesh(&mesh);
            (mesh, aabb)
        };
        ctx.end_struct()?;
        Ok(Self { mesh, aabb })
    }
}

impl RenderMesh {
    /// Create a new render mesh component from a GPU mesh (no AABB).
    pub fn new(mesh: Arc<Mesh>) -> Self {
        Self { mesh, aabb: None }
    }

    /// Create a new render mesh component with a precomputed local-space AABB.
    pub fn with_aabb(mesh: Arc<Mesh>, aabb: redlilium_core::math::Aabb) -> Self {
        Self {
            mesh,
            aabb: Some(aabb),
        }
    }

    /// Get the inner GPU mesh.
    pub fn mesh(&self) -> &Arc<Mesh> {
        &self.mesh
    }
}
