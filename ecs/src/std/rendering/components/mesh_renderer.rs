//! Mesh renderer component.
//!
//! A [`MeshRenderer`] holds an ordered list of [`Primitive`]s — each a
//! `(mesh, material)` pair — that share the entity's single transform
//! (`PerEntityBuffers`). This mirrors the glTF model (one mesh = many
//! primitives, each with its own material) and keeps the entity count low:
//! one renderable entity instead of one entity per primitive.

use std::sync::Arc;

use redlilium_core::material::{CpuMaterialInstance, MaterialValue};
use redlilium_core::math::Aabb;
use redlilium_graphics::{Buffer, MaterialInstance, Mesh};

use super::material_bundle::{
    MaterialBundle, RenderPassType, deserialize_material_value, serialize_material_value,
};
use crate::serialize::Value;
use crate::std::rendering::MeshHandle;
use crate::std::rendering::loaders::MeshSource;

// ---------------------------------------------------------------------------
// PrimitiveMaterial — the per-primitive material instance
// ---------------------------------------------------------------------------

/// A single primitive's material instance (was the standalone `RenderMaterial`
/// component). Wraps a per-pass [`MaterialBundle`], optional CPU data for the
/// inspector / serialization, and the GPU buffer holding packed material
/// property uniforms.
#[derive(Debug, Clone)]
pub struct PrimitiveMaterial {
    /// The GPU material bundle (one instance per render pass).
    bundle: Arc<MaterialBundle>,
    /// CPU-side material data for inspector and serialization (optional).
    cpu_instance: Option<Arc<CpuMaterialInstance>>,
    /// Pass type → material name mapping, for bundle recreation.
    pass_materials: Option<Vec<(RenderPassType, String)>>,
    /// GPU buffer holding packed material property uniforms (group 1, binding 0).
    material_uniform_buffer: Option<Arc<Buffer>>,
}

impl PrimitiveMaterial {
    /// Create from a material bundle with no CPU data.
    pub fn new(bundle: Arc<MaterialBundle>) -> Self {
        Self {
            bundle,
            cpu_instance: None,
            pass_materials: None,
            material_uniform_buffer: None,
        }
    }

    /// Create with CPU-side data for inspector editing and serialization.
    pub fn with_cpu_data(
        bundle: Arc<MaterialBundle>,
        cpu_instance: Arc<CpuMaterialInstance>,
        pass_materials: Vec<(RenderPassType, String)>,
    ) -> Self {
        Self {
            bundle,
            cpu_instance: Some(cpu_instance),
            pass_materials: Some(pass_materials),
            material_uniform_buffer: None,
        }
    }

    /// Attach the GPU buffer for material property uniforms.
    pub fn with_material_uniform_buffer(mut self, buffer: Arc<Buffer>) -> Self {
        self.material_uniform_buffer = Some(buffer);
        self
    }

    /// Get the inner material bundle.
    pub fn bundle(&self) -> &Arc<MaterialBundle> {
        &self.bundle
    }

    /// Get the material instance for a specific render pass.
    pub fn pass(&self, pass_type: RenderPassType) -> Option<&Arc<MaterialInstance>> {
        self.bundle.get(pass_type)
    }

    /// Get the CPU-side material instance, if present.
    pub fn cpu_instance(&self) -> Option<&Arc<CpuMaterialInstance>> {
        self.cpu_instance.as_ref()
    }

    /// Get the pass type → material name mapping, if present.
    pub fn pass_materials(&self) -> Option<&[(RenderPassType, String)]> {
        self.pass_materials.as_deref()
    }

    /// Get the material uniform buffer, if any.
    pub fn material_uniform_buffer(&self) -> Option<&Arc<Buffer>> {
        self.material_uniform_buffer.as_ref()
    }

    /// Replace all material property values (marks the owning [`MeshRenderer`]
    /// dirty when mutated through a `Mut`, so `SyncMaterialUniforms` re-uploads).
    pub fn set_values(&mut self, values: Vec<MaterialValue>) {
        if let Some(cpu_inst) = &mut self.cpu_instance {
            Arc::make_mut(cpu_inst).values = values;
        }
    }

    /// Mutable reference to the CPU material values.
    pub fn values_mut(&mut self) -> Option<&mut Vec<MaterialValue>> {
        self.cpu_instance
            .as_mut()
            .map(|arc| &mut Arc::make_mut(arc).values)
    }

    /// Replace the bundle and optionally the CPU instance (full rebuild, e.g.
    /// after a texture binding change).
    pub fn set_bundle(
        &mut self,
        bundle: Arc<MaterialBundle>,
        cpu_instance: Option<Arc<CpuMaterialInstance>>,
        pass_materials: Option<Vec<(RenderPassType, String)>>,
    ) {
        self.bundle = bundle;
        if cpu_instance.is_some() {
            self.cpu_instance = cpu_instance;
        }
        if pass_materials.is_some() {
            self.pass_materials = pass_materials;
        }
    }
}

// ---------------------------------------------------------------------------
// Primitive — one (mesh, material) pair
// ---------------------------------------------------------------------------

/// A single renderable primitive: a mesh (bound by asset source) drawn with a
/// material. All primitives of a [`MeshRenderer`] share the entity's transform.
///
/// The mesh resolves **asynchronously**: `source` is the serialized identity and
/// `handle` (from [`MeshManager::request`](super::super::MeshManager::request))
/// resolves to the `Arc<Mesh>` once it loads. Use [`mesh`](Self::mesh) /
/// [`aabb`](Self::aabb), which return `None` until then.
#[derive(Debug, Clone)]
pub struct Primitive {
    /// The mesh's asset identity (serialized).
    pub source: MeshSource,
    /// Demand-to-load handle resolving to the shared `Arc<Mesh>` (not serialized).
    pub handle: MeshHandle,
    /// The material instance for this primitive.
    pub material: PrimitiveMaterial,
}

impl Primitive {
    /// Create a primitive from a mesh source + its request handle and a material.
    pub fn new(source: MeshSource, handle: MeshHandle, material: PrimitiveMaterial) -> Self {
        Self {
            source,
            handle,
            material,
        }
    }

    /// The resolved GPU mesh, if it has finished loading.
    pub fn mesh(&self) -> Option<Arc<Mesh>> {
        self.handle.get()
    }

    /// The mesh's local-space AABB, once loaded (carried on the `Mesh`).
    pub fn aabb(&self) -> Option<Aabb> {
        self.handle.get().and_then(|m| m.aabb())
    }
}

// ---------------------------------------------------------------------------
// MeshRenderer — the component
// ---------------------------------------------------------------------------

/// Renderable component: an ordered list of [`Primitive`]s sharing one
/// transform. Replaces the former `RenderMesh` + `RenderMaterial` pair.
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
// The derive is all-or-nothing and can't express three things `MeshRenderer`
// needs: a custom `aabb()` (union over primitives), a per-primitive material
// inspector that rebuilds GPU bundles, and manager-based serialization (meshes
// by name, materials rebuilt via `MaterialManager`). This matches how GPU
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
            // The mesh is bound by asset source (serialized as RON), not by name.
            let source = ron::to_string(&primitive.source).map_err(|e| {
                crate::serialize::SerializeError::FieldError {
                    field: "mesh".to_owned(),
                    message: format!("serialize mesh source: {e}"),
                }
            })?;
            let material = serialize_primitive_material(&primitive.material, ctx)?;
            prims.push(Value::Map(vec![
                ("mesh".to_owned(), Value::String(source)),
                ("material".to_owned(), material),
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

            let mesh_name = match map_get(&fields, "mesh") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "missing 'mesh' name in primitive".into(),
                    ));
                }
            };
            let material_val = map_get(&fields, "material").cloned().ok_or_else(|| {
                crate::serialize::DeserializeError::FormatError(
                    "missing 'material' in primitive".into(),
                )
            })?;

            // The "mesh" field is the RON-encoded asset source; request it from
            // the MeshManager (which loads it asynchronously and shares the Arc).
            let source: MeshSource = ron::from_str(&mesh_name).map_err(|e| {
                crate::serialize::DeserializeError::FormatError(format!("mesh source: {e}"))
            })?;
            let handle = {
                let world = ctx.world();
                if !world.has_resource::<super::super::MeshManager>() {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "MeshManager resource not found".into(),
                    ));
                }
                world
                    .resource_mut::<super::super::MeshManager>()
                    .request(source.clone())
            };

            let material = deserialize_primitive_material(&material_val, ctx)?;
            primitives.push(Primitive::new(source, handle, material));
        }

        ctx.end_struct()?;
        Ok(Self { primitives })
    }
}

// ---------------------------------------------------------------------------
// Material (de)serialization helpers — operate on a Value map directly so they
// can be nested inside the primitive list.
// ---------------------------------------------------------------------------

fn map_get<'a>(map: &'a [(String, Value)], key: &str) -> Option<&'a Value> {
    map.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn serialize_primitive_material(
    mat: &PrimitiveMaterial,
    ctx: &crate::serialize::SerializeContext<'_>,
) -> Result<Value, crate::serialize::SerializeError> {
    let cpu_inst =
        mat.cpu_instance()
            .ok_or_else(|| crate::serialize::SerializeError::FieldError {
                field: "material".to_owned(),
                message: "no cpu_instance on primitive material".into(),
            })?;
    let pass_materials =
        mat.pass_materials()
            .ok_or_else(|| crate::serialize::SerializeError::FieldError {
                field: "material".to_owned(),
                message: "no pass_materials on primitive material".into(),
            })?;

    let values = {
        let world = ctx.world();
        if !world.has_resource::<super::super::TextureManager>() {
            return Err(crate::serialize::SerializeError::FieldError {
                field: "material".to_owned(),
                message: "TextureManager resource not found".into(),
            });
        }
        let tex_manager = world.resource::<super::super::TextureManager>();
        cpu_inst
            .values
            .iter()
            .map(|v| serialize_material_value(v, &tex_manager))
            .collect::<Result<Vec<_>, _>>()?
    };

    let passes_map: Vec<(String, Value)> = pass_materials
        .iter()
        .map(|(pass, name)| (pass.as_str().to_owned(), Value::String(name.clone())))
        .collect();

    let name_val = match &cpu_inst.name {
        Some(name) => Value::String(name.clone()),
        None => Value::Null,
    };

    Ok(Value::Map(vec![
        ("passes".to_owned(), Value::Map(passes_map)),
        ("name".to_owned(), name_val),
        ("values".to_owned(), Value::List(values)),
    ]))
}

fn deserialize_primitive_material(
    value: &Value,
    ctx: &mut crate::serialize::DeserializeContext<'_>,
) -> Result<PrimitiveMaterial, crate::serialize::DeserializeError> {
    let fields = match value {
        Value::Map(fields) => fields,
        _ => {
            return Err(crate::serialize::DeserializeError::FormatError(
                "expected Map for material".into(),
            ));
        }
    };

    // Passes map: { "Forward": "pbr", ... }
    let pass_entries = match map_get(fields, "passes") {
        Some(Value::Map(entries)) => entries,
        _ => {
            return Err(crate::serialize::DeserializeError::FormatError(
                "missing or invalid 'passes' map in material".into(),
            ));
        }
    };
    let mut pass_materials: Vec<(RenderPassType, String)> = Vec::new();
    for (key, val) in pass_entries {
        let pass_type = RenderPassType::parse(key).ok_or_else(|| {
            crate::serialize::DeserializeError::FormatError(format!(
                "unknown render pass type: '{key}'"
            ))
        })?;
        let mat_name = match val {
            Value::String(s) => s.clone(),
            _ => {
                return Err(crate::serialize::DeserializeError::TypeMismatch {
                    field: key.clone(),
                    expected: "String".into(),
                    found: format!("{val:?}"),
                });
            }
        };
        pass_materials.push((pass_type, mat_name));
    }

    let instance_name: Option<String> = match map_get(fields, "name") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Null) | None => None,
        Some(other) => {
            return Err(crate::serialize::DeserializeError::TypeMismatch {
                field: "name".to_owned(),
                expected: "String or Null".into(),
                found: format!("{other:?}"),
            });
        }
    };

    let value_list = match map_get(fields, "values") {
        Some(Value::List(list)) => list.clone(),
        _ => {
            return Err(crate::serialize::DeserializeError::FormatError(
                "missing or invalid 'values' list in material".into(),
            ));
        }
    };
    let values: Vec<MaterialValue> = value_list
        .into_iter()
        .map(deserialize_material_value)
        .collect::<Result<_, _>>()?;

    let first_mat_name = pass_materials
        .first()
        .map(|(_, n)| n.clone())
        .ok_or_else(|| {
            crate::serialize::DeserializeError::FormatError("passes map is empty".into())
        })?;

    let (bundle, cpu_instance) = {
        let world = ctx.world_mut();
        if !world.has_resource::<super::super::MaterialManager>() {
            return Err(crate::serialize::DeserializeError::FormatError(
                "MaterialManager resource not found".into(),
            ));
        }
        if !world.has_resource::<super::super::TextureManager>() {
            return Err(crate::serialize::DeserializeError::FormatError(
                "TextureManager resource not found".into(),
            ));
        }

        let cpu_material = {
            let mat_manager = world.resource::<super::super::MaterialManager>();
            let cpu = mat_manager
                .get_cpu_material(&first_mat_name)
                .ok_or_else(|| {
                    crate::serialize::DeserializeError::FormatError(format!(
                        "material '{first_mat_name}' not found in MaterialManager"
                    ))
                })?;
            Arc::clone(cpu)
        };

        let mut cpu_instance = CpuMaterialInstance::new(cpu_material);
        cpu_instance.name = instance_name;
        cpu_instance.values = values;

        let pass_refs: Vec<(RenderPassType, &str)> = pass_materials
            .iter()
            .map(|(p, n)| (*p, n.as_str()))
            .collect();

        let mut mat_manager = world.resource_mut::<super::super::MaterialManager>();
        let mut tex_manager = world.resource_mut::<super::super::TextureManager>();
        let bundle = mat_manager
            .create_bundle(&cpu_instance, &pass_refs, &mut tex_manager)
            .map_err(|e| {
                crate::serialize::DeserializeError::FormatError(format!(
                    "failed to create material bundle: {e}"
                ))
            })?;
        (bundle, Arc::new(cpu_instance))
    };

    Ok(PrimitiveMaterial::with_cpu_data(
        bundle,
        cpu_instance,
        pass_materials,
    ))
}
