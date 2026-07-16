//! Mesh renderer component.
//!
//! A [`MeshRenderer`] holds an ordered list of [`Primitive`]s — each a
//! `(mesh, material instance)` pair — that share the entity's single transform.
//! This mirrors the glTF model (one mesh = many primitives, each with its own
//! material) and keeps the entity count low: one renderable entity instead of one
//! entity per primitive.
//!
//! Both fields are [`AssetRef`]s: plain data carrying the serialized source plus
//! the cached resolved `Arc`. The `MeshLoad` system resolves them against the
//! managers (demand-driven — constructing a primitive requests nothing), and
//! re-resolves on hot reload; reads are a field access.

use redlilium_assets::AssetRef;
use redlilium_core::math::Aabb;
use redlilium_graphics::Mesh;
use std::sync::Arc;

use crate::serialize::Value;
use crate::std::rendering::ResolvedInstance;
use crate::std::rendering::loaders::{MaterialInstanceSource, MeshSource};

// ---------------------------------------------------------------------------
// Primitive — one (mesh, material instance) pair
// ---------------------------------------------------------------------------

/// A single renderable primitive: a mesh drawn with a material instance, both
/// bound by asset reference. All primitives of a [`MeshRenderer`] share the
/// entity's transform.
///
/// Both refs resolve **asynchronously** (filled by the `MeshLoad` sync system);
/// [`mesh`](Self::mesh) / [`material`](Self::material) / [`aabb`](Self::aabb)
/// return `None` until then.
#[derive(Debug, Clone)]
pub struct Primitive {
    /// The mesh reference (source is serialized; resolution is runtime-only).
    pub mesh: AssetRef<MeshSource>,
    /// The material-instance reference.
    pub material: AssetRef<MaterialInstanceSource>,
}

impl Primitive {
    /// Create a primitive from a mesh source and a material-instance source.
    /// Both resolve asynchronously once the sync system sees the component.
    pub fn new(mesh: MeshSource, material: MaterialInstanceSource) -> Self {
        Self {
            mesh: AssetRef::new(mesh),
            material: AssetRef::new(material),
        }
    }

    /// The resolved GPU mesh, if it has finished loading.
    pub fn mesh(&self) -> Option<Arc<Mesh>> {
        self.mesh.get().cloned()
    }

    /// The resolved material instance, if it has finished loading.
    pub fn material(&self) -> Option<Arc<ResolvedInstance>> {
        self.material.get().cloned()
    }

    /// The mesh's local-space AABB, once loaded (carried on the `Mesh`).
    pub fn aabb(&self) -> Option<Aabb> {
        self.mesh.get().and_then(|m| m.aabb())
    }
}

// ---------------------------------------------------------------------------
// MeshRenderer — the component
// ---------------------------------------------------------------------------

/// Renderable component: an ordered list of [`Primitive`]s sharing one transform.
#[derive(Debug, Clone)]
pub struct MeshRenderer {
    /// The primitives to draw, in order.
    pub primitives: Vec<Primitive>,
    /// Whether this entity is drawn into shadow-casting depth passes
    /// (`RenderPhase::ShadowCaster`, ADR-035/#129). On by default; turn off
    /// for objects that should not occlude light (fake geometry, effects).
    pub casts_shadows: bool,
}

impl Default for MeshRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshRenderer {
    /// Create an empty mesh renderer.
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
            casts_shadows: true,
        }
    }

    /// Create a single-primitive mesh renderer.
    pub fn single(primitive: Primitive) -> Self {
        Self {
            primitives: vec![primitive],
            casts_shadows: true,
        }
    }

    /// Create from a list of primitives.
    pub fn from_primitives(primitives: Vec<Primitive>) -> Self {
        Self {
            primitives,
            casts_shadows: true,
        }
    }

    /// Append a primitive.
    pub fn with_primitive(mut self, primitive: Primitive) -> Self {
        self.primitives.push(primitive);
        self
    }

    /// Set whether the entity casts shadows.
    pub fn with_casts_shadows(mut self, casts_shadows: bool) -> Self {
        self.casts_shadows = casts_shadows;
        self
    }
}

// NOTE: This is a manual `Component` impl rather than `#[derive(Component)]`.
// The derive is all-or-nothing and can't express a custom `aabb()` (union over
// primitives), a per-primitive inspector, and source-only serialization of the
// `AssetRef` fields. This matches how GPU components have always been written in
// this engine.
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

    fn visit_asset_refs(&self, f: &mut dyn FnMut(&dyn std::any::Any)) {
        for primitive in &self.primitives {
            f(&primitive.mesh);
            f(&primitive.material);
        }
    }

    fn visit_asset_refs_mut(&mut self, f: &mut dyn FnMut(&mut dyn std::any::Any)) {
        for primitive in &mut self.primitives {
            f(&mut primitive.mesh);
            f(&mut primitive.material);
        }
    }

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
            // Only the sources are serialized (as structured values, #5);
            // resolution is runtime state.
            let mesh = crate::serialize::value::to_value(primitive.mesh.source()).map_err(|e| {
                crate::serialize::SerializeError::FieldError {
                    field: "mesh".to_owned(),
                    message: format!("serialize mesh source: {e}"),
                }
            })?;
            let material =
                crate::serialize::value::to_value(primitive.material.source()).map_err(|e| {
                    crate::serialize::SerializeError::FieldError {
                        field: "material".to_owned(),
                        message: format!("serialize material source: {e}"),
                    }
                })?;
            prims.push(Value::Map(vec![
                ("mesh".to_owned(), mesh),
                ("material".to_owned(), material),
            ]));
        }

        ctx.begin_struct(Self::NAME)?;
        ctx.write_field("primitives", Value::List(prims))?;
        ctx.write_field("casts_shadows", Value::Bool(self.casts_shadows))?;
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

            let mesh: MeshSource = deserialize_source(&fields, "mesh")?;
            let material: MaterialInstanceSource = deserialize_source(&fields, "material")?;

            // No manager access here: the refs start unresolved and the sync
            // system requests + resolves them once it sees the component.
            primitives.push(Primitive::new(mesh, material));
        }

        // Absent in scenes written before #129 — default to casting.
        let casts_shadows = match ctx.read_field("casts_shadows") {
            Ok(Value::Bool(b)) => b,
            _ => true,
        };

        ctx.end_struct()?;
        Ok(Self {
            primitives,
            casts_shadows,
        })
    }
}

fn map_get<'a>(map: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    map.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// Deserialize a primitive source (`MeshSource` / `MaterialInstanceSource`)
/// from a field: structured value (current format, #5) or the legacy
/// RON-encoded-into-a-string form written by older prefabs.
fn deserialize_source<T: serde::de::DeserializeOwned>(
    fields: &[(String, Value)],
    key: &str,
) -> Result<T, crate::serialize::DeserializeError> {
    match map_get(fields, key) {
        Some(Value::String(s)) => ron::from_str(s).map_err(|e| {
            crate::serialize::DeserializeError::FormatError(format!("{key} source (legacy): {e}"))
        }),
        Some(value) => crate::serialize::value::from_value(value.clone()).map_err(|e| {
            crate::serialize::DeserializeError::FormatError(format!("{key} source: {e}"))
        }),
        None => Err(crate::serialize::DeserializeError::FormatError(format!(
            "missing '{key}' source in primitive"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;
    use crate::component::Component;
    use crate::serialize::{DeserializeContext, SerializeContext};
    use redlilium_assets::Guid;

    fn sample_renderer() -> MeshRenderer {
        MeshRenderer::single(Primitive::new(
            MeshSource::File(Guid::stable("meshes/cube.rmesh")),
            MaterialInstanceSource {
                guid: Guid::stable("materials/default.matinst"),
            },
        ))
    }

    fn serialize(renderer: &MeshRenderer) -> Value {
        let world = World::new();
        let mut ctx = SerializeContext::new(&world);
        renderer.serialize_component(&mut ctx).unwrap()
    }

    fn deserialize(value: &Value) -> MeshRenderer {
        let mut world = World::new();
        let mut ctx = DeserializeContext::new(&mut world);
        ctx.load_data(value).unwrap();
        MeshRenderer::deserialize_component(&mut ctx).unwrap()
    }

    /// Sources serialize as structured values, not RON-encoded strings (#5).
    #[test]
    fn sources_serialize_structured() {
        let value = serialize(&sample_renderer());
        let Value::Map(fields) = &value else {
            panic!("expected Map");
        };
        let Some(Value::List(prims)) = map_get(fields, "primitives") else {
            panic!("expected primitives list");
        };
        let Value::Map(prim) = &prims[0] else {
            panic!("expected primitive map");
        };
        for key in ["mesh", "material"] {
            assert!(
                !matches!(map_get(prim, key), Some(Value::String(_))),
                "'{key}' must be a structured value, not a RON string"
            );
        }
    }

    #[test]
    fn sources_roundtrip() {
        let renderer = sample_renderer();
        let restored = deserialize(&serialize(&renderer));
        assert_eq!(
            restored.primitives[0].mesh.source(),
            renderer.primitives[0].mesh.source()
        );
        assert_eq!(
            restored.primitives[0].material.source(),
            renderer.primitives[0].material.source()
        );
    }

    /// `casts_shadows` roundtrips; scenes written before #129 (no field)
    /// default to casting.
    #[test]
    fn casts_shadows_roundtrips_and_defaults_on() {
        let renderer = sample_renderer().with_casts_shadows(false);
        let restored = deserialize(&serialize(&renderer));
        assert!(!restored.casts_shadows, "explicit false survives roundtrip");

        // A pre-#129 payload: primitives only, no casts_shadows field.
        let Value::Map(mut fields) = serialize(&sample_renderer()) else {
            panic!("expected Map");
        };
        fields.retain(|(k, _)| k != "casts_shadows");
        let restored = deserialize(&Value::Map(fields));
        assert!(restored.casts_shadows, "legacy scenes default to casting");
    }

    /// Prefabs written before #5 encode sources as RON strings — they must
    /// still load.
    #[test]
    fn legacy_string_sources_still_parse() {
        let renderer = sample_renderer();
        let mesh_ron = ron::to_string(renderer.primitives[0].mesh.source()).unwrap();
        let material_ron = ron::to_string(renderer.primitives[0].material.source()).unwrap();
        let legacy = Value::Map(vec![(
            "primitives".to_owned(),
            Value::List(vec![Value::Map(vec![
                ("mesh".to_owned(), Value::String(mesh_ron)),
                ("material".to_owned(), Value::String(material_ron)),
            ])]),
        )]);

        let restored = deserialize(&legacy);
        assert_eq!(
            restored.primitives[0].mesh.source(),
            renderer.primitives[0].mesh.source()
        );
        assert_eq!(
            restored.primitives[0].material.source(),
            renderer.primitives[0].material.source()
        );
    }
}
