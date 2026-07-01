//! Mesh renderer component.
//!
//! A [`MeshRenderer`] holds an ordered list of [`Primitive`]s — each a
//! `(mesh, material instance)` pair — that share the entity's single transform.
//! This mirrors the glTF model (one mesh = many primitives, each with its own
//! material) and keeps the entity count low: one renderable entity instead of one
//! entity per primitive.
//!
//! Both the mesh and the material instance bind through the **asset system**:
//! each is a serialized `Source` plus a demand-to-load handle that resolves to the
//! shared GPU resource once it loads (`docs/MATERIAL_ASSETS.md`). The material is a
//! [`MaterialInstanceSource`] resolving to a [`ResolvedInstance`].

use redlilium_core::math::Aabb;
use redlilium_graphics::Mesh;
use std::sync::Arc;

use crate::serialize::Value;
use crate::std::rendering::loaders::{MaterialInstanceSource, MeshSource};
use crate::std::rendering::{InstanceHandle, MeshHandle, ResolvedInstance};

// ---------------------------------------------------------------------------
// Primitive — one (mesh, material instance) pair
// ---------------------------------------------------------------------------

/// A single renderable primitive: a mesh drawn with a material instance, both
/// bound by asset source. All primitives of a [`MeshRenderer`] share the entity's
/// transform.
///
/// Both resolve **asynchronously**: each `*_source` is the serialized identity and
/// each `*_handle` resolves to the shared GPU resource once it loads. Use
/// [`mesh`](Self::mesh) / [`material`](Self::material) / [`aabb`](Self::aabb),
/// which return `None` until then.
#[derive(Debug, Clone)]
pub struct Primitive {
    /// The mesh's asset identity (serialized).
    pub source: MeshSource,
    /// Demand-to-load handle resolving to the shared `Arc<Mesh>` (not serialized).
    pub handle: MeshHandle,
    /// The material instance's asset identity (serialized).
    pub material_source: MaterialInstanceSource,
    /// Demand-to-load handle resolving to the [`ResolvedInstance`] (not serialized).
    pub material_handle: InstanceHandle,
}

impl Primitive {
    /// Create a primitive from a mesh source + handle and a material-instance
    /// source + handle (both from their managers' `request`).
    pub fn new(
        source: MeshSource,
        handle: MeshHandle,
        material_source: MaterialInstanceSource,
        material_handle: InstanceHandle,
    ) -> Self {
        Self {
            source,
            handle,
            material_source,
            material_handle,
        }
    }

    /// The resolved GPU mesh, if it has finished loading.
    pub fn mesh(&self) -> Option<Arc<Mesh>> {
        self.handle.get()
    }

    /// The resolved material instance, if it has finished loading.
    pub fn material(&self) -> Option<Arc<ResolvedInstance>> {
        self.material_handle.get()
    }

    /// The mesh's local-space AABB, once loaded (carried on the `Mesh`).
    pub fn aabb(&self) -> Option<Aabb> {
        self.handle.get().and_then(|m| m.aabb())
    }
}

// ---------------------------------------------------------------------------
// MeshRenderer — the component
// ---------------------------------------------------------------------------

/// Renderable component: an ordered list of [`Primitive`]s sharing one transform.
#[derive(Debug, Clone, Default)]
pub struct MeshRenderer {
    /// The primitives to draw, in order.
    pub primitives: Vec<Primitive>,
}

impl MeshRenderer {
    /// Create an empty mesh renderer.
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
        }
    }

    /// Create a single-primitive mesh renderer.
    pub fn single(primitive: Primitive) -> Self {
        Self {
            primitives: vec![primitive],
        }
    }

    /// Create from a list of primitives.
    pub fn from_primitives(primitives: Vec<Primitive>) -> Self {
        Self { primitives }
    }

    /// Append a primitive.
    pub fn with_primitive(mut self, primitive: Primitive) -> Self {
        self.primitives.push(primitive);
        self
    }
}

// NOTE: This is a manual `Component` impl rather than `#[derive(Component)]`.
// The derive is all-or-nothing and can't express a custom `aabb()` (union over
// primitives), a per-primitive inspector, and manager-based async (de)serialization
// (meshes + material instances bound by asset source). This matches how GPU
// components have always been written in this engine.
impl crate::Component for MeshRenderer {
    const NAME: &'static str = "MeshRenderer";

    fn inspect_ui(
        &self,
        ui: &mut crate::egui::Ui,
        world: &crate::World,
        entity: crate::Entity,
    ) -> crate::InspectResult {
        super::super::material_inspector::inspect_mesh_renderer_ui(world, entity, ui)
    }

    fn collect_entities(&self, _collector: &mut Vec<crate::Entity>) {}

    fn remap_entities(&mut self, _map: &mut dyn FnMut(crate::Entity) -> crate::Entity) {}

    fn register_required(world: &mut crate::World) {
        world.register_required::<Self, crate::Transform>();
        world.register_required::<Self, crate::GlobalTransform>();
        world.register_required::<Self, crate::Visibility>();
    }

    fn aabb(&self, _world: &crate::World) -> Option<Aabb> {
        self.primitives
            .iter()
            .filter_map(|p| p.aabb())
            .reduce(|acc, a| acc.union(&a))
    }

    fn serialize_component(
        &self,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<Value, crate::serialize::SerializeError> {
        let mut prims: Vec<Value> = Vec::with_capacity(self.primitives.len());
        for primitive in &self.primitives {
            // Both the mesh and the material instance are bound by asset source
            // (serialized as RON), not by name.
            let mesh = ron::to_string(&primitive.source).map_err(|e| {
                crate::serialize::SerializeError::FieldError {
                    field: "mesh".to_owned(),
                    message: format!("serialize mesh source: {e}"),
                }
            })?;
            let material = ron::to_string(&primitive.material_source).map_err(|e| {
                crate::serialize::SerializeError::FieldError {
                    field: "material".to_owned(),
                    message: format!("serialize material source: {e}"),
                }
            })?;
            prims.push(Value::Map(vec![
                ("mesh".to_owned(), Value::String(mesh)),
                ("material".to_owned(), Value::String(material)),
            ]));
        }

        ctx.begin_struct(Self::NAME)?;
        ctx.write_field("primitives", Value::List(prims))?;
        ctx.end_struct()
    }

    fn deserialize_component(
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.begin_struct(Self::NAME)?;
        let prims_val = ctx.read_field("primitives")?;
        let prim_list = match prims_val {
            Value::List(list) => list,
            _ => {
                return Err(crate::serialize::DeserializeError::TypeMismatch {
                    field: "primitives".to_owned(),
                    expected: "List".into(),
                    found: format!("{prims_val:?}"),
                });
            }
        };

        let mut primitives = Vec::with_capacity(prim_list.len());
        for prim_val in prim_list {
            let fields = match prim_val {
                Value::Map(fields) => fields,
                _ => {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "expected Map for primitive".into(),
                    ));
                }
            };

            let mesh_src = match map_get(&fields, "mesh") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "missing 'mesh' source in primitive".into(),
                    ));
                }
            };
            let material_src = match map_get(&fields, "material") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "missing 'material' source in primitive".into(),
                    ));
                }
            };

            let source: MeshSource = ron::from_str(&mesh_src).map_err(|e| {
                crate::serialize::DeserializeError::FormatError(format!("mesh source: {e}"))
            })?;
            let material_source: MaterialInstanceSource =
                ron::from_str(&material_src).map_err(|e| {
                    crate::serialize::DeserializeError::FormatError(format!(
                        "material source: {e}"
                    ))
                })?;

            // Request both from their managers (async load + shared resolution).
            let (handle, material_handle) = {
                let world = ctx.world();
                if !world.has_resource::<super::super::MeshManager>() {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "MeshManager resource not found".into(),
                    ));
                }
                if !world.has_resource::<super::super::MaterialInstanceManager>() {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "MaterialInstanceManager resource not found".into(),
                    ));
                }
                let handle = world
                    .resource_mut::<super::super::MeshManager>()
                    .request(source.clone());
                let material_handle = world
                    .resource_mut::<super::super::MaterialInstanceManager>()
                    .request(material_source.clone());
                (handle, material_handle)
            };

            primitives.push(Primitive::new(source, handle, material_source, material_handle));
        }

        ctx.end_struct()?;
        Ok(Self { primitives })
    }
}

fn map_get<'a>(map: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    map.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}
