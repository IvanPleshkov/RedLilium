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
