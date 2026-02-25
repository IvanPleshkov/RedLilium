//! Custom inspector UI for the [`RenderMaterial`] component.
//!
//! Displays material binding values (colors, sliders, texture names) in the
//! component inspector panel, similar to Unity/Unreal material editors.
//! Edits produce undoable actions that rebuild the GPU material bundle.

use std::sync::Arc;

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::material::{
    AlphaMode, CpuMaterial, CpuMaterialInstance, MaterialValue, MaterialValueType, TextureRef,
    TextureSource,
};

use super::components::RenderPassType;
use super::resources::{MaterialManager, TextureManager};
use crate::rendering::RenderMaterial;
use crate::{Entity, InspectResult, World};

// ---------------------------------------------------------------------------
// Custom inspect function
// ---------------------------------------------------------------------------

/// Custom inspector for [`RenderMaterial`] that reads the component's
/// [`CpuMaterialInstance`] to display and edit material properties.
pub(crate) fn inspect_material_ui(
    world: &World,
    entity: Entity,
    ui: &mut egui::Ui,
) -> InspectResult {
    let comp = world.get::<RenderMaterial>(entity)?;

    // Show label
    ui.horizontal(|ui| {
        ui.label("material");
        match comp.bundle.label() {
            Some(label) => ui.label(label),
            None => ui.weak("(unnamed)"),
        };
    });

    // Need CPU data for property editing
    let cpu_instance = comp.cpu_instance.as_ref()?;
    let pass_materials = comp.pass_materials.as_ref()?;

    let cpu_instance = Arc::clone(cpu_instance);
    let pass_materials = pass_materials.clone();
    let cpu_mat = Arc::clone(&cpu_instance.material);

    // Show pipeline state (read-only, collapsed)
    show_pipeline_state(ui, &cpu_mat);

    ui.add_space(4.0);

    // Show editable properties
    let mut new_values = cpu_instance.values.clone();
    let mut any_changed = false;

    for (i, binding_def) in cpu_mat.bindings.iter().enumerate() {
        if i >= new_values.len() {
            break;
        }
        if show_material_value(
            ui,
            &binding_def.name,
            binding_def.value_type,
            &mut new_values[i],
        ) {
            any_changed = true;
        }
    }

    if !any_changed {
        return None;
    }

    Some(vec![Box::new(SetMaterialValuesAction {
        entity,
        old_values: cpu_instance.values.clone(),
        new_values,
        cpu_material: cpu_mat,
        instance_name: cpu_instance.name.clone(),
        pass_materials,
    })])
}

// ---------------------------------------------------------------------------
// Pipeline state display (read-only)
// ---------------------------------------------------------------------------

fn show_pipeline_state(ui: &mut egui::Ui, cpu_mat: &CpuMaterial) {
    egui::CollapsingHeader::new("Pipeline State")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("alpha_mode");
                match cpu_mat.alpha_mode {
                    AlphaMode::Opaque => {
                        ui.label("Opaque");
                    }
                    AlphaMode::Mask { cutoff } => {
                        ui.label(format!("Mask (cutoff: {cutoff:.2})"));
                    }
                    AlphaMode::Blend => {
                        ui.label("Blend");
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("double_sided");
                ui.label(if cpu_mat.double_sided { "Yes" } else { "No" });
            });

            ui.horizontal(|ui| {
                ui.label("polygon_mode");
                ui.label(format!("{:?}", cpu_mat.polygon_mode));
            });
        });
}

// ---------------------------------------------------------------------------
// Material value widgets
// ---------------------------------------------------------------------------

/// Renders the appropriate widget for a material value. Returns `true` if changed.
fn show_material_value(
    ui: &mut egui::Ui,
    name: &str,
    _value_type: MaterialValueType,
    value: &mut MaterialValue,
) -> bool {
    match value {
        MaterialValue::Float(v) => show_float_value(ui, name, v),
        MaterialValue::Vec3(v) => show_vec3_value(ui, name, v),
        MaterialValue::Vec4(v) => show_vec4_value(ui, name, v),
        MaterialValue::Texture(tex_ref) => {
            show_texture_value(ui, name, tex_ref);
            false
        }
    }
}

/// Float value: slider for known PBR properties, drag value otherwise.
fn show_float_value(ui: &mut egui::Ui, name: &str, value: &mut f32) -> bool {
    ui.horizontal(|ui| {
        ui.label(name);
        match name {
            "metallic" | "roughness" | "occlusion_strength" => {
                ui.add(egui::Slider::new(value, 0.0..=1.0)).changed()
            }
            "normal_scale" => ui
                .add(egui::DragValue::new(value).speed(0.01).range(0.0..=10.0))
                .changed(),
            _ => ui.add(egui::DragValue::new(value).speed(0.01)).changed(),
        }
    })
    .inner
}

/// Vec3 value: color picker if name suggests a color, 3x drag values otherwise.
fn show_vec3_value(ui: &mut egui::Ui, name: &str, value: &mut [f32; 3]) -> bool {
    let is_color = name.contains("color") || name.contains("emissive");
    ui.horizontal(|ui| {
        ui.label(name);
        if is_color {
            ui.color_edit_button_rgb(value).changed()
        } else {
            let x = ui
                .add(
                    egui::DragValue::new(&mut value[0])
                        .speed(0.01)
                        .prefix("x: "),
                )
                .changed();
            let y = ui
                .add(
                    egui::DragValue::new(&mut value[1])
                        .speed(0.01)
                        .prefix("y: "),
                )
                .changed();
            let z = ui
                .add(
                    egui::DragValue::new(&mut value[2])
                        .speed(0.01)
                        .prefix("z: "),
                )
                .changed();
            x || y || z
        }
    })
    .inner
}

/// Vec4 value: color picker if name suggests a color, 4x drag values otherwise.
fn show_vec4_value(ui: &mut egui::Ui, name: &str, value: &mut [f32; 4]) -> bool {
    let is_color = name.contains("color");
    ui.horizontal(|ui| {
        ui.label(name);
        if is_color {
            ui.color_edit_button_rgba_unmultiplied(value).changed()
        } else {
            let x = ui
                .add(
                    egui::DragValue::new(&mut value[0])
                        .speed(0.01)
                        .prefix("x: "),
                )
                .changed();
            let y = ui
                .add(
                    egui::DragValue::new(&mut value[1])
                        .speed(0.01)
                        .prefix("y: "),
                )
                .changed();
            let z = ui
                .add(
                    egui::DragValue::new(&mut value[2])
                        .speed(0.01)
                        .prefix("z: "),
                )
                .changed();
            let w = ui
                .add(
                    egui::DragValue::new(&mut value[3])
                        .speed(0.01)
                        .prefix("w: "),
                )
                .changed();
            x || y || z || w
        }
    })
    .inner
}

/// Texture value: read-only label showing the texture name.
fn show_texture_value(ui: &mut egui::Ui, name: &str, tex_ref: &TextureRef) {
    ui.horizontal(|ui| {
        ui.label(name);
        let tex_name = match &tex_ref.texture {
            TextureSource::Named(n) => n.as_str(),
            TextureSource::Cpu(cpu_tex) => cpu_tex.name.as_deref().unwrap_or("<embedded>"),
        };
        ui.weak(tex_name);
    });
}

// ---------------------------------------------------------------------------
// SetMaterialValuesAction — undoable material property edit
// ---------------------------------------------------------------------------

/// Reversible action that updates material property values and rebuilds the
/// GPU bundle via [`MaterialManager::create_bundle`].
struct SetMaterialValuesAction {
    entity: Entity,
    old_values: Vec<MaterialValue>,
    new_values: Vec<MaterialValue>,
    cpu_material: Arc<CpuMaterial>,
    instance_name: Option<String>,
    pass_materials: Vec<(RenderPassType, String)>,
}

impl std::fmt::Debug for SetMaterialValuesAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetMaterialValuesAction")
            .field("entity", &self.entity)
            .finish()
    }
}

impl SetMaterialValuesAction {
    /// Rebuild the GPU material bundle from the given values and set it on the entity.
    fn rebuild_and_set(&self, world: &mut World, values: &[MaterialValue]) -> EditActionResult {
        if !world.is_alive(self.entity) {
            return Err(EditActionError::TargetNotFound("entity despawned".into()));
        }

        let mut cpu_instance = CpuMaterialInstance::new(Arc::clone(&self.cpu_material));
        cpu_instance.name = self.instance_name.clone();
        cpu_instance.values = values.to_vec();

        let pass_refs: Vec<(RenderPassType, &str)> = self
            .pass_materials
            .iter()
            .map(|(p, n)| (*p, n.as_str()))
            .collect();

        let bundle = {
            let mut mat_manager = world.resource_mut::<MaterialManager>();
            let mut tex_manager = world.resource_mut::<TextureManager>();
            mat_manager
                .create_bundle(&cpu_instance, &pass_refs, &mut tex_manager)
                .map_err(|e| EditActionError::Custom(format!("material rebuild failed: {e}")))?
        };

        let new_comp = RenderMaterial::with_cpu_data(
            bundle,
            Arc::new(cpu_instance),
            self.pass_materials.clone(),
        );
        let _ = world.insert(self.entity, new_comp);
        Ok(())
    }
}

impl EditAction<World> for SetMaterialValuesAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let values = self.new_values.clone();
        self.rebuild_and_set(world, &values)
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        let values = self.old_values.clone();
        self.rebuild_and_set(world, &values)
    }

    fn description(&self) -> &str {
        "Edit material"
    }

    fn merge(&mut self, other: Box<dyn EditAction<World>>) -> Option<Box<dyn EditAction<World>>> {
        if let Some(other) = other.as_any().downcast_ref::<Self>()
            && self.entity == other.entity
        {
            self.new_values = other.new_values.clone();
            return None; // consumed
        }
        Some(other)
    }
}
