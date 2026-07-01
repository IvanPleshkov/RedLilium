//! Inspecting/editing asset settings via the unified [`ComponentField`] field
//! widgets. The per-kind dispatch lives here (next to the loaders), so the editor
//! stays generic — it just calls [`inspect_asset_settings`].

use redlilium_core::mesh::VertexLayout;

use crate::ComponentField;
use crate::serialize::{DeserializeContext, DeserializeError, SerializeContext, SerializeError};

// VertexLayout is a `ComponentField` — editable as a field (here in the asset
// inspector, and in components/prefabs). The orphan rule allows this: the trait
// is local to ecs. Serialization delegates to serde (the layout is entity-free,
// so it needs no World).
impl ComponentField for VertexLayout {
    fn inspect_field(&self, _name: &str, ui: &mut egui::Ui) -> Option<Self> {
        let mut layout = self.clone();
        let mut changed = false;

        // Editable label (reuses the String field widget).
        let label = layout.label.clone().unwrap_or_default();
        if let Some(new_label) = label.inspect_field("label", ui) {
            layout.label = (!new_label.is_empty()).then_some(new_label);
            changed = true;
        }

        // Buffers + attributes shown read-only for now (rich list editing later).
        ui.separator();
        ui.label(format!("buffers ({})", layout.buffers.len()));
        for (i, buffer) in layout.buffers.iter().enumerate() {
            ui.monospace(format!(
                "  [{i}] stride {} · {:?}",
                buffer.stride, buffer.step_mode
            ));
        }
        ui.label(format!("attributes ({})", layout.attributes.len()));
        for attr in &layout.attributes {
            ui.monospace(format!(
                "  {:?} · {:?} @ {} (buffer {})",
                attr.semantic, attr.format, attr.offset, attr.buffer_index
            ));
        }

        changed.then_some(layout)
    }

    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_serde(name, self)
    }

    fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Self, DeserializeError> {
        ctx.read_serde(name)
    }
}

/// Render the editable view of an asset's per-kind `settings` (RON), returning
/// the new settings if the user edited them. Dispatches by `kind`; the editor
/// calls this without knowing any asset kinds.
///
/// Asset settings are entity-free, so (de)serialization here is plain serde (no
/// World) — distinct from `ComponentField`'s World-aware context used for
/// prefabs.
pub fn inspect_asset_settings(
    kind: &str,
    settings: Option<&str>,
    ui: &mut egui::Ui,
    world: &crate::World,
) -> Option<String> {
    match kind {
        "vertex_layout" => {
            let Some(text) = settings else {
                ui.weak("(no parameters in record)");
                return None;
            };
            let layout: VertexLayout = match ron::from_str(text) {
                Ok(l) => l,
                Err(e) => {
                    ui.colored_label(egui::Color32::RED, format!("invalid settings RON: {e}"));
                    return None;
                }
            };
            let edited = layout.inspect_field("vertex_layout", ui)?;
            ron::to_string(&edited).ok()
        }
        _ => {
            // Material / material-instance editing lives behind the rendering
            // feature (it needs the shading registry + asset DB); `Some` means the
            // kind was handled there.
            if let Some(result) = inspect_render_asset(kind, settings, ui, world) {
                return result;
            }
            match settings {
                Some(s) => {
                    ui.label("settings");
                    ui.monospace(s);
                }
                None => {
                    ui.weak("No editable data for this asset kind.");
                }
            }
            None
        }
    }
}

#[cfg(not(feature = "rendering"))]
fn inspect_render_asset(
    _kind: &str,
    _settings: Option<&str>,
    _ui: &mut egui::Ui,
    _world: &crate::World,
) -> Option<Option<String>> {
    None
}

#[cfg(feature = "rendering")]
use render_assets::inspect_render_asset;

/// The default file extension + record `settings` for a newly-created asset of a
/// given kind — used by the editor's "New" menu. Kind-knowledge stays here (next
/// to the loaders); the editor just writes the file + record.
#[cfg(feature = "rendering")]
pub use render_assets::{NewAssetSpec, new_asset_spec};

/// Material / material-instance settings editing — schema-aware via the
/// [`ShadingRegistry`](super::ShadingRegistry): each shading-model property slot
/// is shown with the right widget (a color picker for `*_color`), the current
/// value taken from the record (an instance also inherits its parent material's
/// values), and edits written back as a property/override.
#[cfg(feature = "rendering")]
mod render_assets {
    use redlilium_assets::{AssetDb, Guid};
    use redlilium_core::mesh::VertexLayout;

    use super::super::{MaterialData, MaterialInstanceData, PropValue, ShadingRegistry};
    use crate::World;

    /// Default extension + record `settings` for a newly-created asset (the "New"
    /// menu). See [`new_asset_spec`].
    pub struct NewAssetSpec {
        pub extension: &'static str,
        pub settings: Option<String>,
    }

    /// The default spec for creating an asset of `kind`. A material instance needs
    /// a `parent_material` guid (there is no valid instance without a parent), so
    /// creation of one returns `None` if the caller can't supply a parent.
    pub fn new_asset_spec(kind: &str, parent_material: Option<Guid>) -> Option<NewAssetSpec> {
        match kind {
            "vertex_layout" => {
                let layout = (*VertexLayout::position_normal()).clone();
                Some(NewAssetSpec {
                    extension: "vlayout",
                    settings: ron::to_string(&layout).ok(),
                })
            }
            "material" => {
                let data = MaterialData {
                    shading_model: "opaque".to_owned(),
                    properties: Vec::new(),
                };
                Some(NewAssetSpec {
                    extension: "material",
                    settings: ron::to_string(&data).ok(),
                })
            }
            "material_instance" => {
                let data = MaterialInstanceData {
                    parent: parent_material?,
                    overrides: Vec::new(),
                };
                Some(NewAssetSpec {
                    extension: "matinst",
                    settings: ron::to_string(&data).ok(),
                })
            }
            _ => None,
        }
    }

    /// Dispatch the render-asset kinds. `Some(result)` = handled.
    pub(super) fn inspect_render_asset(
        kind: &str,
        settings: Option<&str>,
        ui: &mut egui::Ui,
        world: &World,
    ) -> Option<Option<String>> {
        match kind {
            "material" => Some(inspect_material(settings, ui, world)),
            "material_instance" => Some(inspect_material_instance(settings, ui, world)),
            _ => None,
        }
    }

    fn inspect_material(settings: Option<&str>, ui: &mut egui::Ui, world: &World) -> Option<String> {
        let text = settings?;
        let mut data: MaterialData = match ron::from_str(text) {
            Ok(d) => d,
            Err(e) => {
                ui.colored_label(egui::Color32::RED, format!("invalid material RON: {e}"));
                return None;
            }
        };
        ui.horizontal(|ui| {
            ui.label("shading_model");
            ui.monospace(&data.shading_model);
        });
        ui.separator();

        let schema = world
            .resource::<ShadingRegistry>()
            .get(&data.shading_model)
            .map(|m| m.schema.clone());
        let Some(schema) = schema else {
            ui.weak(format!("unknown shading model '{}'", data.shading_model));
            return None;
        };

        let mut changed = false;
        for slot in &schema {
            let mut val = find_prop(&data.properties, &slot.name).unwrap_or_else(|| slot.default.clone());
            if inspect_prop(ui, &slot.name, &mut val) {
                set_prop(&mut data.properties, &slot.name, val);
                changed = true;
            }
        }
        changed.then(|| ron::to_string(&data).ok()).flatten()
    }

    fn inspect_material_instance(
        settings: Option<&str>,
        ui: &mut egui::Ui,
        world: &World,
    ) -> Option<String> {
        let text = settings?;
        let mut data: MaterialInstanceData = match ron::from_str(text) {
            Ok(d) => d,
            Err(e) => {
                ui.colored_label(egui::Color32::RED, format!("invalid instance RON: {e}"));
                return None;
            }
        };
        ui.horizontal(|ui| {
            ui.label("parent");
            ui.monospace(format!("{:?}", data.parent));
        });
        ui.separator();

        // Resolve the parent material → the shading-model schema + the parent's
        // property values (the instance inherits any it doesn't override).
        let (schema, parent_props) = resolve_parent(world, &data.parent);
        let Some(schema) = schema else {
            ui.weak("parent material not resolved");
            return None;
        };

        let mut changed = false;
        for slot in &schema {
            let mut val = find_prop(&data.overrides, &slot.name)
                .or_else(|| find_prop(&parent_props, &slot.name))
                .unwrap_or_else(|| slot.default.clone());
            if inspect_prop(ui, &slot.name, &mut val) {
                set_prop(&mut data.overrides, &slot.name, val);
                changed = true;
            }
        }
        changed.then(|| ron::to_string(&data).ok()).flatten()
    }

    /// Resolve a parent material guid → (schema, its property values).
    fn resolve_parent(
        world: &World,
        parent: &Guid,
    ) -> (Option<Vec<super::super::PropDef>>, Vec<(String, PropValue)>) {
        let text = {
            let db = world.resource::<AssetDb>();
            match db.record(parent).and_then(|r| r.settings.clone()) {
                Some(t) => t,
                None => return (None, Vec::new()),
            }
        };
        let Ok(mat) = ron::from_str::<MaterialData>(&text) else {
            return (None, Vec::new());
        };
        let schema = world
            .resource::<ShadingRegistry>()
            .get(&mat.shading_model)
            .map(|m| m.schema.clone());
        (schema, mat.properties)
    }

    fn find_prop(list: &[(String, PropValue)], name: &str) -> Option<PropValue> {
        list.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone())
    }

    fn set_prop(list: &mut Vec<(String, PropValue)>, name: &str, value: PropValue) {
        match list.iter_mut().find(|(n, _)| n == name) {
            Some(entry) => entry.1 = value,
            None => list.push((name.to_owned(), value)),
        }
    }

    /// One property widget: a color picker for `*_color` / `emissive`, else numeric
    /// drag values. Returns `true` if edited.
    fn inspect_prop(ui: &mut egui::Ui, name: &str, value: &mut PropValue) -> bool {
        let is_color = name.contains("color") || name.contains("emissive");
        ui.horizontal(|ui| {
            ui.label(name);
            match value {
                PropValue::Float(v) => ui.add(egui::DragValue::new(v).speed(0.01)).changed(),
                PropValue::Vec3(v) => {
                    if is_color {
                        ui.color_edit_button_rgb(v).changed()
                    } else {
                        drag_n(ui, v)
                    }
                }
                PropValue::Vec4(v) => {
                    if is_color {
                        ui.color_edit_button_rgba_unmultiplied(v).changed()
                    } else {
                        drag_n(ui, v)
                    }
                }
            }
        })
        .inner
    }

    fn drag_n(ui: &mut egui::Ui, values: &mut [f32]) -> bool {
        let mut changed = false;
        for v in values.iter_mut() {
            changed |= ui.add(egui::DragValue::new(v).speed(0.01)).changed();
        }
        changed
    }
}
