use std::collections::HashMap;
use std::sync::Arc;

use redlilium_core::material::{CpuMaterialInstance, MaterialValue, TextureRef, TextureSource};
use redlilium_graphics::{BindingGroup, Buffer, MaterialInstance, Mesh, Texture};

use crate::serialize::Value;

/// Identifies which render pass a material instance is intended for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPassType {
    /// Main forward color pass.
    Forward,
    /// Depth-only prepass (for early-Z or screen-space effects).
    DepthPrepass,
    /// Shadow map pass.
    Shadow,
    /// Deferred G-buffer pass.
    Deferred,
    /// Entity index pass — renders entity ID to an R32Uint target for picking.
    EntityIndex,
}

impl RenderPassType {
    /// All known render pass types.
    pub const ALL: &[Self] = &[
        Self::Forward,
        Self::DepthPrepass,
        Self::Shadow,
        Self::Deferred,
        Self::EntityIndex,
    ];

    /// Serialize to a string key.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Forward => "Forward",
            Self::DepthPrepass => "DepthPrepass",
            Self::Shadow => "Shadow",
            Self::Deferred => "Deferred",
            Self::EntityIndex => "EntityIndex",
        }
    }

    /// Parse from a string key.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Forward" => Some(Self::Forward),
            "DepthPrepass" => Some(Self::DepthPrepass),
            "Shadow" => Some(Self::Shadow),
            "Deferred" => Some(Self::Deferred),
            "EntityIndex" => Some(Self::EntityIndex),
            _ => None,
        }
    }
}

/// A collection of [`MaterialInstance`]s for different render passes.
///
/// All instances in a bundle share the same binding groups (textures, uniform
/// buffers) but reference different GPU [`Material`](redlilium_graphics::Material)
/// pipelines — one per pass type. This allows a single logical material to be
/// used across forward, shadow, depth-prepass, and deferred passes without
/// duplicating GPU resources.
///
/// # Example
///
/// ```ignore
/// let bundle = MaterialBundle::new()
///     .with_pass(RenderPassType::Forward, forward_instance)
///     .with_pass(RenderPassType::Shadow, shadow_instance)
///     .with_label("pbr_metal");
/// ```
#[derive(Debug, Clone)]
pub struct MaterialBundle {
    /// Material instance per pass type.
    passes: HashMap<RenderPassType, Arc<MaterialInstance>>,
    /// Shared binding groups referenced by each instance.
    shared_bindings: Vec<Arc<BindingGroup>>,
    /// Optional debug label.
    label: Option<String>,
}

impl MaterialBundle {
    /// Create an empty material bundle.
    pub fn new() -> Self {
        Self {
            passes: HashMap::new(),
            shared_bindings: Vec::new(),
            label: None,
        }
    }

    /// Add a material instance for a specific render pass.
    pub fn with_pass(mut self, pass_type: RenderPassType, instance: Arc<MaterialInstance>) -> Self {
        self.passes.insert(pass_type, instance);
        self
    }

    /// Set the shared binding groups.
    pub fn with_shared_bindings(mut self, bindings: Vec<Arc<BindingGroup>>) -> Self {
        self.shared_bindings = bindings;
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the material instance for a specific pass type.
    pub fn get(&self, pass_type: RenderPassType) -> Option<&Arc<MaterialInstance>> {
        self.passes.get(&pass_type)
    }

    /// Get all pass entries.
    pub fn passes(&self) -> &HashMap<RenderPassType, Arc<MaterialInstance>> {
        &self.passes
    }

    /// Get the shared binding groups.
    pub fn shared_bindings(&self) -> &[Arc<BindingGroup>] {
        &self.shared_bindings
    }

    /// Get the bundle label, if set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl Default for MaterialBundle {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU mesh component.
///
/// Wraps an `Arc<Mesh>` (GPU-uploaded mesh) so it can be attached to entities.
/// Entities with both `RenderMesh` and [`RenderMaterial`] are collected by the
/// forward render system and drawn each frame.
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
            if !world.has_resource::<super::MeshManager>() {
                return Err(crate::serialize::SerializeError::FieldError {
                    field: "mesh".to_owned(),
                    message: "MeshManager resource not found".into(),
                });
            }
            let manager = world.resource::<super::MeshManager>();
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

/// Per-entity GPU uniform buffers for transform data (VP + model matrix).
///
/// Holds the forward-pass uniform buffer and an optional entity-index pass
/// buffer. The [`UpdatePerEntityUniforms`](super::UpdatePerEntityUniforms)
/// system writes camera and transform data into these buffers each frame.
///
/// This replaces the previous pattern of storing `Vec<(Entity, Arc<Buffer>)>`
/// lists outside the ECS.
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

/// GPU material component.
///
/// Wraps an `Arc<MaterialBundle>` containing material instances for each
/// render pass type (forward, shadow, depth-prepass, deferred). All instances
/// in the bundle share the same bindings but use different shader pipelines.
///
/// Attach alongside [`RenderMesh`] to make an entity renderable.
///
/// Optionally holds a [`CpuMaterialInstance`] for inspector editing and
/// serialization. When present, the component inspector displays editable
/// material properties (color pickers, sliders, etc.).
#[derive(Debug, Clone)]
pub struct RenderMaterial {
    /// The GPU material bundle.
    bundle: Arc<MaterialBundle>,
    /// CPU-side material data for inspector and serialization (optional).
    cpu_instance: Option<Arc<CpuMaterialInstance>>,
    /// Pass type → material name mapping for bundle recreation.
    pass_materials: Option<Vec<(RenderPassType, String)>>,
    /// The GPU buffer holding packed material property uniforms (binding 0).
    material_uniform_buffer: Option<Arc<Buffer>>,
    /// Whether CPU-side values have been modified since the last GPU upload.
    dirty: bool,
}

impl crate::Component for RenderMaterial {
    const NAME: &'static str = "RenderMaterial";

    fn inspect_ui(
        &self,
        ui: &mut crate::egui::Ui,
        world: &crate::World,
        entity: crate::Entity,
    ) -> crate::InspectResult {
        #[cfg(feature = "inspector")]
        {
            super::material_inspector::inspect_material_ui(world, entity, ui)
        }
        #[cfg(not(feature = "inspector"))]
        {
            let _ = (world, entity);
            ui.horizontal(|ui| {
                ui.label("material");
                match self.bundle().label() {
                    Some(label) => ui.label(format!("Material: {label}")),
                    None => ui.weak("Material (unnamed)"),
                };
            });

            // Show CPU-side material properties (read-only)
            if let Some(cpu_inst) = &self.cpu_instance {
                let cpu_mat = &cpu_inst.material;
                for (i, binding_def) in cpu_mat.bindings.iter().enumerate() {
                    if i >= cpu_inst.values.len() {
                        break;
                    }
                    show_material_value_readonly(ui, &binding_def.name, &cpu_inst.values[i]);
                }
            }

            None
        }
    }

    fn collect_entities(&self, _collector: &mut Vec<crate::Entity>) {}

    fn remap_entities(&mut self, _map: &mut dyn FnMut(crate::Entity) -> crate::Entity) {}

    fn register_required(_world: &mut crate::World) {}

    fn serialize_component(
        &self,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<Value, crate::serialize::SerializeError> {
        let (passes_map, instance_name, values) = {
            let world = ctx.world();
            if !world.has_resource::<super::MaterialManager>() {
                return Err(crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "MaterialManager resource not found".into(),
                });
            }
            if !world.has_resource::<super::TextureManager>() {
                return Err(crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "TextureManager resource not found".into(),
                });
            }

            let mat_manager = world.resource::<super::MaterialManager>();
            let tex_manager = world.resource::<super::TextureManager>();

            let cpu_bundle = mat_manager.get_cpu_bundle(&self.bundle).ok_or_else(|| {
                crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "material bundle not tracked in MaterialManager".into(),
                }
            })?;

            // Serialize pass → material-name map
            let passes_map: Vec<(String, Value)> = cpu_bundle
                .pass_materials
                .iter()
                .map(|(pass, name)| (pass.as_str().to_owned(), Value::String(name.clone())))
                .collect();

            let values: Vec<Value> = cpu_bundle
                .cpu_instance
                .values
                .iter()
                .map(|v| serialize_material_value(v, &tex_manager))
                .collect::<Result<_, _>>()?;

            let instance_name = cpu_bundle.cpu_instance.name.clone();

            (passes_map, instance_name, values)
        };

        ctx.begin_struct(Self::NAME)?;
        ctx.write_field("passes", Value::Map(passes_map))?;
        if let Some(name) = &instance_name {
            ctx.write_serde("name", name)?;
        } else {
            ctx.write_field("name", Value::Null)?;
        }
        ctx.write_field("values", Value::List(values))?;
        ctx.end_struct()
    }

    fn deserialize_component(
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.begin_struct(Self::NAME)?;

        // Read passes map: { "Forward": "pbr", "Shadow": "pbr_shadow", ... }
        let passes_val = ctx.read_field("passes")?;
        let pass_entries = match passes_val {
            Value::Map(entries) => entries,
            // Backward compat: single "material" field → Forward only
            _ => {
                return Err(crate::serialize::DeserializeError::TypeMismatch {
                    field: "passes".to_owned(),
                    expected: "Map".into(),
                    found: format!("{passes_val:?}"),
                });
            }
        };
        let mut pass_materials: Vec<(RenderPassType, String)> = Vec::new();
        for (key, val) in pass_entries {
            let pass_type = RenderPassType::parse(&key).ok_or_else(|| {
                crate::serialize::DeserializeError::FormatError(format!(
                    "unknown render pass type: '{key}'"
                ))
            })?;
            let mat_name = match val {
                Value::String(s) => s,
                _ => {
                    return Err(crate::serialize::DeserializeError::TypeMismatch {
                        field: key,
                        expected: "String".into(),
                        found: format!("{val:?}"),
                    });
                }
            };
            pass_materials.push((pass_type, mat_name));
        }

        let instance_name: Option<String> = {
            let val = ctx.read_field("name")?;
            match val {
                Value::Null => None,
                Value::String(s) => Some(s),
                _ => {
                    return Err(crate::serialize::DeserializeError::TypeMismatch {
                        field: "name".to_owned(),
                        expected: "String or Null".into(),
                        found: format!("{val:?}"),
                    });
                }
            }
        };
        let values_val = ctx.read_field("values")?;
        let value_list = match values_val {
            Value::List(list) => list,
            _ => {
                return Err(crate::serialize::DeserializeError::TypeMismatch {
                    field: "values".to_owned(),
                    expected: "List".into(),
                    found: format!("{values_val:?}"),
                });
            }
        };

        let values: Vec<MaterialValue> = value_list
            .into_iter()
            .map(deserialize_material_value)
            .collect::<Result<_, _>>()?;

        // Need the CPU material to build the instance. Use the first pass's material name.
        let first_mat_name = pass_materials
            .first()
            .map(|(_, n)| n.clone())
            .ok_or_else(|| {
                crate::serialize::DeserializeError::FormatError("passes map is empty".into())
            })?;

        let (bundle, cpu_instance) = {
            let world = ctx.world_mut();
            if !world.has_resource::<super::MaterialManager>() {
                return Err(crate::serialize::DeserializeError::FormatError(
                    "MaterialManager resource not found".into(),
                ));
            }
            if !world.has_resource::<super::TextureManager>() {
                return Err(crate::serialize::DeserializeError::FormatError(
                    "TextureManager resource not found".into(),
                ));
            }

            let cpu_material = {
                let mat_manager = world.resource::<super::MaterialManager>();
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

            let mut mat_manager = world.resource_mut::<super::MaterialManager>();
            let mut tex_manager = world.resource_mut::<super::TextureManager>();
            let bundle = mat_manager
                .create_bundle(&cpu_instance, &pass_refs, &mut tex_manager)
                .map_err(|e| {
                    crate::serialize::DeserializeError::FormatError(format!(
                        "failed to create material bundle: {e}"
                    ))
                })?;
            (bundle, Arc::new(cpu_instance))
        };

        ctx.end_struct()?;
        Ok(Self::with_cpu_data(bundle, cpu_instance, pass_materials))
    }

    fn post_deserialize(entity: crate::Entity, world: &mut crate::World) {
        // The deserialized RenderMaterial has a bundle created by
        // MaterialManager::create_bundle which only has material bindings at
        // group 0. The opaque_color shader expects per-entity transforms at
        // group 0 and material props at group 1. Rebuild with the correct
        // layout and create PerEntityBuffers.
        let Some(render_mat) = world.get::<Self>(entity) else {
            return;
        };
        let Some(cpu_instance) = render_mat.cpu_instance.clone() else {
            return;
        };
        let Some(pass_materials) = render_mat.pass_materials.clone() else {
            return;
        };

        if !world.has_resource::<super::MaterialManager>() {
            return;
        }

        let first_mat_name = pass_materials
            .first()
            .map(|(_, n)| n.as_str())
            .unwrap_or("");

        // Look up the entity-index GPU material for picking support
        let mat_manager = world.resource::<super::MaterialManager>();
        let Some(forward_gpu) = mat_manager.get_material(first_mat_name).cloned() else {
            return;
        };
        let ei_gpu = mat_manager.get_material("entity_index").cloned();
        let device = mat_manager.device().clone();
        drop(mat_manager);

        // Create the proper two-group bundle using the shader factory
        let cpu_material = cpu_instance.material.clone();

        let (per_entity, mut new_render_mat, bundle) = if let Some(ei_material) = &ei_gpu {
            super::shaders::create_opaque_color_entity_full(
                &device,
                &forward_gpu,
                ei_material,
                &cpu_material,
            )
        } else {
            // Fallback: forward-only (no picking)
            let (fwd_buf, bundle) =
                super::shaders::create_opaque_color_entity(&device, &forward_gpu);
            let per_entity = PerEntityBuffers::new(fwd_buf);
            let rm = Self::with_cpu_data(
                Arc::clone(&bundle),
                Arc::new(CpuMaterialInstance::new(cpu_material)),
                pass_materials.clone(),
            );
            (per_entity, rm, bundle)
        };

        // Apply the deserialized values
        new_render_mat.set_values(cpu_instance.values.clone());

        // Write deserialized values to the material props GPU buffer
        if let Some(buf) = new_render_mat.material_uniform_buffer() {
            let bytes = super::pack_uniform_bytes(&cpu_instance.material, &cpu_instance.values);
            if !bytes.is_empty() {
                let _ = device.write_buffer(buf, 0, &bytes);
            }
            // Mark as synced so the sync system doesn't overwrite with stale data
            new_render_mat.mark_synced();
        }

        // Register the new bundle for serialization
        {
            let mut mat_manager = world.resource_mut::<super::MaterialManager>();
            mat_manager.register_bundle(&bundle, Arc::clone(&cpu_instance), pass_materials);
        }

        // Replace the component and add PerEntityBuffers
        let _ = world.insert(entity, new_render_mat);
        let _ = world.insert(entity, per_entity);
    }
}

impl RenderMaterial {
    /// Create a new render material component from a material bundle (no CPU data).
    pub fn new(bundle: Arc<MaterialBundle>) -> Self {
        Self {
            bundle,
            cpu_instance: None,
            pass_materials: None,
            material_uniform_buffer: None,
            dirty: false,
        }
    }

    /// Create a render material with CPU-side data for inspector editing.
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
            dirty: false,
        }
    }

    /// Set the GPU buffer for material property uniforms (binding 0).
    pub fn with_material_uniform_buffer(mut self, buffer: Arc<Buffer>) -> Self {
        self.material_uniform_buffer = Some(buffer);
        self
    }

    // --- Immutable accessors ---

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

    // --- Tick-based mutation ---

    /// Replace all material property values. Marks the component dirty so the
    /// [`SyncMaterialUniforms`](super::SyncMaterialUniforms) system will
    /// re-upload the uniform buffer on the next frame.
    pub fn set_values(&mut self, values: Vec<MaterialValue>) {
        if let Some(cpu_inst) = &mut self.cpu_instance {
            Arc::make_mut(cpu_inst).values = values;
            self.dirty = true;
        }
    }

    /// Get a mutable reference to the CPU material values.
    /// Marks the component dirty so the sync system will re-upload.
    pub fn values_mut(&mut self) -> Option<&mut Vec<MaterialValue>> {
        self.dirty = true;
        self.cpu_instance
            .as_mut()
            .map(|arc| &mut Arc::make_mut(arc).values)
    }

    /// Whether CPU values have been modified since last GPU sync.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark as synced after GPU upload. Called by `SyncMaterialUniforms`.
    pub(crate) fn mark_synced(&mut self) {
        self.dirty = false;
    }

    // --- Bundle replacement (for texture changes that need full rebuild) ---

    /// Replace the bundle and optionally the CPU instance (full rebuild).
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
        // After a full rebuild, GPU is already up-to-date
        self.dirty = false;
    }
}

// ---------------------------------------------------------------------------
// Material value serialization helpers
// ---------------------------------------------------------------------------

fn serialize_material_value(
    value: &MaterialValue,
    tex_manager: &super::TextureManager,
) -> Result<Value, crate::serialize::SerializeError> {
    match value {
        MaterialValue::Float(v) => Ok(Value::Map(vec![
            ("t".to_owned(), Value::String("f".to_owned())),
            ("v".to_owned(), Value::F32(*v)),
        ])),
        MaterialValue::Vec3(v) => Ok(Value::Map(vec![
            ("t".to_owned(), Value::String("v3".to_owned())),
            (
                "v".to_owned(),
                Value::List(v.iter().map(|f| Value::F32(*f)).collect()),
            ),
        ])),
        MaterialValue::Vec4(v) => Ok(Value::Map(vec![
            ("t".to_owned(), Value::String("v4".to_owned())),
            (
                "v".to_owned(),
                Value::List(v.iter().map(|f| Value::F32(*f)).collect()),
            ),
        ])),
        MaterialValue::Texture(tex_ref) => {
            let texture_name = match &tex_ref.texture {
                TextureSource::Named(name) => name.clone(),
                TextureSource::Cpu(cpu_tex) => cpu_tex
                    .name
                    .clone()
                    .unwrap_or_else(|| "<unnamed>".to_owned()),
            };

            let sampler_val = if let Some(sampler) = &tex_ref.sampler {
                Value::String(
                    sampler
                        .name
                        .clone()
                        .unwrap_or_else(|| "<unnamed>".to_owned()),
                )
            } else {
                Value::Null
            };

            let _ = tex_manager; // used for future texture name resolution

            Ok(Value::Map(vec![
                ("t".to_owned(), Value::String("tex".to_owned())),
                ("texture".to_owned(), Value::String(texture_name)),
                ("sampler".to_owned(), sampler_val),
                ("tex_coord".to_owned(), Value::U64(tex_ref.tex_coord as u64)),
            ]))
        }
    }
}

fn deserialize_material_value(
    value: Value,
) -> Result<MaterialValue, crate::serialize::DeserializeError> {
    let fields = match value {
        Value::Map(fields) => fields,
        _ => {
            return Err(crate::serialize::DeserializeError::FormatError(
                "expected Map for material value".into(),
            ));
        }
    };

    let mut map: std::collections::HashMap<String, Value> = fields.into_iter().collect();
    let type_tag = match map.remove("t") {
        Some(Value::String(s)) => s,
        _ => {
            return Err(crate::serialize::DeserializeError::FormatError(
                "missing or invalid 't' field in material value".into(),
            ));
        }
    };

    match type_tag.as_str() {
        "f" => {
            let v = extract_f32(&map, "v")?;
            Ok(MaterialValue::Float(v))
        }
        "v3" => {
            let list = extract_f32_list(&map, "v", 3)?;
            Ok(MaterialValue::Vec3([list[0], list[1], list[2]]))
        }
        "v4" => {
            let list = extract_f32_list(&map, "v", 4)?;
            Ok(MaterialValue::Vec4([list[0], list[1], list[2], list[3]]))
        }
        "tex" => {
            let texture_name = match map.get("texture") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "missing 'texture' field in texture value".into(),
                    ));
                }
            };

            let sampler = match map.get("sampler") {
                Some(Value::String(s)) => {
                    let mut cpu_sampler = redlilium_core::sampler::CpuSampler::linear();
                    cpu_sampler.name = Some(s.clone());
                    Some(Arc::new(cpu_sampler))
                }
                Some(Value::Null) | None => None,
                _ => {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "invalid 'sampler' field in texture value".into(),
                    ));
                }
            };

            let tex_coord = match map.get("tex_coord") {
                Some(Value::U64(n)) => *n as u32,
                _ => 0,
            };

            Ok(MaterialValue::Texture(TextureRef {
                texture: TextureSource::Named(texture_name),
                sampler,
                tex_coord,
            }))
        }
        _ => Err(crate::serialize::DeserializeError::FormatError(format!(
            "unknown material value type tag: '{type_tag}'"
        ))),
    }
}

fn extract_f32(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Result<f32, crate::serialize::DeserializeError> {
    match map.get(key) {
        Some(Value::F32(v)) => Ok(*v),
        Some(Value::F64(v)) => Ok(*v as f32),
        Some(Value::I64(v)) => Ok(*v as f32),
        Some(Value::U64(v)) => Ok(*v as f32),
        _ => Err(crate::serialize::DeserializeError::FormatError(format!(
            "expected numeric value for '{key}'"
        ))),
    }
}

fn extract_f32_list(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
    expected_len: usize,
) -> Result<Vec<f32>, crate::serialize::DeserializeError> {
    match map.get(key) {
        Some(Value::List(list)) => {
            if list.len() != expected_len {
                return Err(crate::serialize::DeserializeError::FormatError(format!(
                    "expected {expected_len} elements for '{key}', got {}",
                    list.len()
                )));
            }
            list.iter()
                .map(|v| match v {
                    Value::F32(f) => Ok(*f),
                    Value::F64(f) => Ok(*f as f32),
                    Value::I64(i) => Ok(*i as f32),
                    Value::U64(u) => Ok(*u as f32),
                    _ => Err(crate::serialize::DeserializeError::FormatError(
                        "expected numeric in list".into(),
                    )),
                })
                .collect()
        }
        _ => Err(crate::serialize::DeserializeError::FormatError(format!(
            "expected List for '{key}'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Read-only material value display (non-inspector builds)
// ---------------------------------------------------------------------------

/// Display a single material property value as a read-only label row.
#[cfg(not(feature = "inspector"))]
fn show_material_value_readonly(ui: &mut crate::egui::Ui, name: &str, value: &MaterialValue) {
    ui.horizontal(|ui| {
        ui.label(name);
        match value {
            MaterialValue::Float(v) => {
                ui.weak(format!("{v:.3}"));
            }
            MaterialValue::Vec3(v) => {
                ui.weak(format!("[{:.3}, {:.3}, {:.3}]", v[0], v[1], v[2]));
            }
            MaterialValue::Vec4(v) => {
                ui.weak(format!(
                    "[{:.3}, {:.3}, {:.3}, {:.3}]",
                    v[0], v[1], v[2], v[3]
                ));
            }
            MaterialValue::Texture(tex_ref) => {
                let tex_name = match &tex_ref.texture {
                    TextureSource::Named(n) => n.as_str(),
                    TextureSource::Cpu(cpu_tex) => cpu_tex.name.as_deref().unwrap_or("<embedded>"),
                };
                ui.weak(tex_name);
            }
        }
    });
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
