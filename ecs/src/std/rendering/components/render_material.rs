//! GPU material component.

use std::sync::Arc;

use redlilium_core::material::{CpuMaterialInstance, MaterialValue};
use redlilium_graphics::{Buffer, MaterialInstance};

use super::material_bundle::{
    MaterialBundle, RenderPassType, deserialize_material_value, serialize_material_value,
};
use crate::serialize::Value;

/// GPU material component.
///
/// Wraps an `Arc<MaterialBundle>` containing material instances for each
/// render pass type (forward, shadow, depth-prepass, deferred). All instances
/// in the bundle share the same bindings but use different shader pipelines.
///
/// Attach alongside [`RenderMesh`](super::RenderMesh) to make an entity renderable.
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
            super::super::material_inspector::inspect_material_ui(world, entity, ui)
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
        let cpu_inst = self.cpu_instance.as_ref().ok_or_else(|| {
            crate::serialize::SerializeError::FieldError {
                field: "0".to_owned(),
                message: "no cpu_instance on RenderMaterial".into(),
            }
        })?;
        let pass_materials = self.pass_materials.as_ref().ok_or_else(|| {
            crate::serialize::SerializeError::FieldError {
                field: "0".to_owned(),
                message: "no pass_materials on RenderMaterial".into(),
            }
        })?;

        let values = {
            let world = ctx.world();
            if !world.has_resource::<super::super::TextureManager>() {
                return Err(crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
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

        ctx.begin_struct(Self::NAME)?;
        ctx.write_field("passes", Value::Map(passes_map))?;
        if let Some(name) = &cpu_inst.name {
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

        ctx.end_struct()?;
        let mut mat = Self::with_cpu_data(bundle, cpu_instance, pass_materials);
        mat.dirty = true;
        Ok(mat)
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
    /// [`SyncMaterialUniforms`](super::super::SyncMaterialUniforms) system will
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
