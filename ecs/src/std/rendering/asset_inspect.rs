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
    fn inspect_field(
        &self,
        _name: &str,
        ui: &mut egui::Ui,
        ctx: &crate::FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut layout = self.clone();
        let mut changed = false;

        // Editable label (reuses the String field widget).
        let label = layout.label.clone().unwrap_or_default();
        if let Some(new_label) = label.inspect_field("label", ui, ctx) {
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
            // Asset settings have no owning entity; the ctx carries a
            // dangling one (entity-free fields never look at it).
            let ctx = &crate::FieldInspectCtx {
                world,
                entity: crate::Entity::DANGLING,
            };
            let edited = layout.inspect_field("vertex_layout", ui, ctx)?;
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

/// The record kind a named reference of a `record_kind` asset accepts — what
/// its inspector drop target allows (e.g. a mesh's `"layout"` reference takes a
/// vertex layout). `None` = the reference is not editable by drop.
pub fn reference_accepted_kind(record_kind: &str, role: &str) -> Option<&'static str> {
    match (record_kind, role) {
        ("mesh", "layout") => Some("vertex_layout"),
        _ => None,
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

    use super::super::{
        MaterialData, MaterialInstanceData, PropValue, ShadingRegistry, StorageBufferSource,
        TextureSettings, TextureSource,
    };
    use crate::World;

    /// Default extension, record `settings`, and initial file bytes for a
    /// newly-created asset (the "New" menu). See [`new_asset_spec`].
    pub struct NewAssetSpec {
        pub extension: &'static str,
        pub settings: Option<String>,
        /// Initial file content. Most kinds keep their data in the record's
        /// `settings` and start as an empty file; kinds whose content lives
        /// in the file itself (scenes) must start valid, not zero-byte —
        /// their loader would reject an empty file.
        pub content: Vec<u8>,
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
                    content: Vec::new(),
                })
            }
            "material" => {
                let data = MaterialData {
                    shading_model: "opaque".to_owned(),
                    properties: Vec::new(),
                    features: Vec::new(),
                };
                Some(NewAssetSpec {
                    extension: "material",
                    settings: ron::to_string(&data).ok(),
                    content: Vec::new(),
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
                    content: Vec::new(),
                })
            }
            #[cfg(feature = "serialize-ron")]
            "scene" => Some(NewAssetSpec {
                extension: "scene",
                settings: None,
                content: crate::std::scene::empty_scene_ron().into_bytes(),
            }),
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
            "texture" => Some(inspect_texture(settings, ui)),
            _ => None,
        }
    }

    /// Texture import + sampling settings ([`TextureSettings`]). Absent record
    /// settings edit as the defaults; any change persists the full struct.
    fn inspect_texture(settings: Option<&str>, ui: &mut egui::Ui) -> Option<String> {
        use redlilium_core::sampler::{AddressMode, FilterMode};

        let mut data = settings
            .and_then(|s| ron::from_str::<TextureSettings>(s).ok())
            .unwrap_or_default();
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("srgb");
            changed |= ui.checkbox(&mut data.srgb, "").changed();
        });
        ui.horizontal(|ui| {
            ui.label("filter");
            for (mode, label) in [
                (FilterMode::Linear, "linear"),
                (FilterMode::Nearest, "nearest"),
            ] {
                changed |= ui.selectable_value(&mut data.filter, mode, label).changed();
            }
        });
        ui.horizontal(|ui| {
            ui.label("address");
            for (mode, label) in [
                (AddressMode::Repeat, "repeat"),
                (AddressMode::ClampToEdge, "clamp"),
                (AddressMode::MirrorRepeat, "mirror"),
            ] {
                changed |= ui
                    .selectable_value(&mut data.address, mode, label)
                    .changed();
            }
        });
        ui.horizontal(|ui| {
            ui.label("anisotropy");
            changed |= ui
                .add(egui::DragValue::new(&mut data.anisotropy).range(1..=16))
                .changed();
        });

        changed.then(|| ron::to_string(&data).ok()).flatten()
    }

    fn inspect_material(
        settings: Option<&str>,
        ui: &mut egui::Ui,
        world: &World,
    ) -> Option<String> {
        let text = settings?;
        let mut data: MaterialData = match ron::from_str(text) {
            Ok(d) => d,
            Err(e) => {
                ui.colored_label(egui::Color32::RED, format!("invalid material RON: {e}"));
                return None;
            }
        };
        // The shading model is the material's one "shader" knob (the model owns
        // the shader + the property schema — `docs/MATERIAL_ASSETS.md` Decision
        // 3), so it's selected from the registry, never a raw shader reference.
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("shading_model");
            egui::ComboBox::from_id_salt("shading_model")
                .selected_text(&data.shading_model)
                .show_ui(ui, |ui| {
                    for id in world.resource::<ShadingRegistry>().ids() {
                        if ui.selectable_label(data.shading_model == id, id).clicked()
                            && data.shading_model != id
                        {
                            data.shading_model = id.to_owned();
                            changed = true;
                        }
                    }
                });
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

        for slot in &schema {
            let mut val =
                find_prop(&data.properties, &slot.name).unwrap_or_else(|| slot.default.clone());
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
                // A texture slot: inline color editing for a solid value, and a
                // drop target either way — dropping a texture asset from the
                // browser sets the slot to its file source.
                PropValue::Texture(source) => {
                    let mut changed = false;
                    let label = match source {
                        TextureSource::Solid(rgba) => {
                            let mut srgba = egui::Color32::from_rgba_unmultiplied(
                                rgba[0], rgba[1], rgba[2], rgba[3],
                            );
                            if ui.color_edit_button_srgba(&mut srgba).changed() {
                                *rgba = srgba.to_srgba_unmultiplied();
                                changed = true;
                            }
                            "solid".to_owned()
                        }
                        TextureSource::SolidArray(_, layers) => format!("solid array ×{layers}"),
                        TextureSource::File(guid) => format!("{guid:?}"),
                        TextureSource::Virtual(guid) => format!("virtual {guid:?}"),
                    };
                    if let Some(guid) = super::super::asset_drop_target(
                        ui,
                        &label,
                        false,
                        <TextureSource as redlilium_assets::AssetRefSource>::KIND,
                    ) {
                        *source = TextureSource::File(guid);
                        changed = true;
                    }
                    changed
                }
                // A texture-array slot (`Texture2DArray` in the shader): a drop
                // target for an array texture asset. The default is a solid
                // white array; dropping a texture file sets the file source.
                PropValue::TextureArray(source) => {
                    let label = match source {
                        TextureSource::SolidArray(_, layers) => format!("solid array ×{layers}"),
                        TextureSource::Solid(_) => "solid".to_owned(),
                        TextureSource::File(guid) => format!("{guid:?}"),
                        TextureSource::Virtual(guid) => format!("virtual {guid:?}"),
                    };
                    if let Some(guid) = super::super::asset_drop_target(
                        ui,
                        &label,
                        false,
                        <TextureSource as redlilium_assets::AssetRefSource>::KIND,
                    ) {
                        *source = TextureSource::File(guid);
                        true
                    } else {
                        false
                    }
                }
                // A storage-buffer slot: read-only in the inspector (its bytes
                // are authored data or an externally published buffer, not a
                // scalar the property editor drags).
                PropValue::StorageBuffer(source) => {
                    let label = match source {
                        StorageBufferSource::Inline(bytes) => {
                            format!("storage buffer: inline {} bytes", bytes.len())
                        }
                        StorageBufferSource::Ref(guid) => format!("storage buffer: ref {guid:?}"),
                    };
                    ui.label(label);
                    false
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
