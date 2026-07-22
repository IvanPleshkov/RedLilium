//! Unified field-level inspection and serialization trait.
//!
//! The [`ComponentField`] trait provides a single extension point for types
//! used as fields inside `#[derive(Component)]` structs. Implementing it gives
//! a type automatic support for:
//!
//! - Inspector UI (via [`Inspect`](crate::inspect::Inspect))
//! - Serialization (via [`SerializeField`](crate::serialize::SerializeField))
//! - Deserialization (via [`DeserializeField`](crate::serialize::DeserializeField))
//!
//! The derive macro dispatches through wrapper types that prefer
//! `ComponentField` inherent methods over the fallback traits.
//!
//! # Adding a custom field type
//!
//! ```ignore
//! impl ComponentField for MyColor {
//!     fn inspect_field(&self, name: &str, ui: &mut egui::Ui, _ctx: &FieldInspectCtx<'_>) -> Option<Self> {
//!         let mut value = *self;
//!         let changed = ui.horizontal(|ui| {
//!             ui.label(name);
//!             // custom color picker widget; return true if edited
//!             false
//!         }).inner;
//!         changed.then_some(value)
//!     }
//!
//!     fn serialize_field(
//!         &self,
//!         name: &str,
//!         ctx: &mut SerializeContext<'_>,
//!     ) -> Result<(), SerializeError> {
//!         ctx.write_serde(name, self)
//!     }
//!
//!     fn deserialize_field(
//!         name: &str,
//!         ctx: &mut DeserializeContext<'_>,
//!     ) -> Result<Self, DeserializeError> {
//!         ctx.read_serde(name)
//!     }
//! }
//! ```

use std::sync::Arc;

use redlilium_core::math::{Mat4, Quat, Vec2, Vec3, Vec4};

use crate::serialize::{DeserializeContext, DeserializeError, SerializeContext, SerializeError};

/// Context handed to [`ComponentField::inspect_field`]: read-only world
/// access plus the entity owning the inspected component. This is what lets
/// field widgets query world state — the entity picker lists live entities;
/// future asset pickers can browse the asset DB the same way (#73).
pub struct FieldInspectCtx<'a> {
    /// The world the inspected component lives in (read-only).
    pub world: &'a crate::World,
    /// The entity owning the inspected component.
    pub entity: crate::Entity,
}

/// Unified field-level inspection + serialization trait.
///
/// Implement this for any type you want to use as a field inside a
/// `#[derive(Component)]` struct with full inspector and serialization
/// support. Types that implement `ComponentField` are preferred over
/// the fallback traits ([`InspectFallback`](crate::inspect::InspectFallback),
/// [`SerializeFieldFallback`](crate::serialize::SerializeFieldFallback),
/// [`DeserializeFieldFallback`](crate::serialize::DeserializeFieldFallback)).
pub trait ComponentField: Send + Sync + 'static {
    /// Render an inspector UI widget for this field value.
    ///
    /// Takes an immutable reference and returns `Some(new_value)` if the
    /// user edited the value in the UI, or `None` if unchanged. `ctx` gives
    /// world-aware widgets (entity pickers, asset browsers) read access to
    /// the world; plain value widgets ignore it.
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self>
    where
        Self: Sized;

    /// Serialize this field value into the context.
    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError>;

    /// Deserialize a field value from the context.
    fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Self, DeserializeError>
    where
        Self: Sized;

    /// Visit the asset references carried by this field (read-only), passing
    /// each as `&dyn Any` (a concrete `AssetRef<S>`). Default: no refs. An
    /// `AssetRef` field yields itself; container/compound fields forward to
    /// their elements. Drives the generic asset-sync system.
    fn visit_asset_refs(&self, _f: &mut dyn FnMut(&dyn std::any::Any)) {}

    /// Visit the asset references carried by this field (mutably) — the write
    /// half of [`visit_asset_refs`](Self::visit_asset_refs), used to apply a
    /// re-resolution. Default: no refs.
    fn visit_asset_refs_mut(&mut self, _f: &mut dyn FnMut(&mut dyn std::any::Any)) {}
}

// ---------------------------------------------------------------------------
// Primitive types
// ---------------------------------------------------------------------------

impl ComponentField for f32 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut value = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut value).speed(0.01))
                    .changed()
            })
            .inner;
        changed.then_some(value)
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

impl ComponentField for f64 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut value = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut value).speed(0.01))
                    .changed()
            })
            .inner;
        changed.then_some(value)
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

impl ComponentField for bool {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut value = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                ui.checkbox(&mut value, "").changed()
            })
            .inner;
        changed.then_some(value)
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

impl ComponentField for u8 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut v = *self as i32;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut v).range(0..=255))
                    .changed()
            })
            .inner;
        changed.then_some(v as u8)
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

impl ComponentField for u32 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut v = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut v)).changed()
            })
            .inner;
        changed.then_some(v)
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

impl ComponentField for i32 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut value = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut value)).changed()
            })
            .inner;
        changed.then_some(value)
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

impl ComponentField for u64 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        // Native u64 backing: the old cast through i64 silently rewrote
        // values above i64::MAX on edit. DragValue still goes through f64
        // internally, so drags on values above 2^53 remain imprecise.
        let mut v = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut v)).changed()
            })
            .inner;
        changed.then_some(v)
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

impl ComponentField for usize {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut v = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                ui.add(egui::DragValue::new(&mut v)).changed()
            })
            .inner;
        changed.then_some(v)
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

impl ComponentField for String {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut value = self.clone();
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                ui.text_edit_singleline(&mut value).changed()
            })
            .inner;
        changed.then_some(value)
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

// ---------------------------------------------------------------------------
// Entity types
// ---------------------------------------------------------------------------

/// Display label for an entity in the picker: `Name (index@tick)` when a
/// [`Name`](crate::Name) is present, bare `index@tick` otherwise.
fn entity_label(world: &crate::World, entity: crate::Entity) -> String {
    if entity == crate::Entity::DANGLING {
        return "dangling".to_owned();
    }
    if !world.is_alive(entity) {
        return format!("dead ({}@{})", entity.index(), entity.spawn_tick());
    }
    match world.get::<crate::Name>(entity) {
        Some(name) if !name.0.is_empty() => {
            format!("{} ({}@{})", name.0, entity.index(), entity.spawn_tick())
        }
        _ => format!("{}@{}", entity.index(), entity.spawn_tick()),
    }
}

/// Entity dropdown shared by the `Entity`-family field impls. Lists live,
/// non-editor entities (editor-owned entities like the camera must never be
/// referenced from scene data — the reference would dangle on load).
/// Returns `Some(selection)` when the user picks an item; the inner `None`
/// is the "no entity" choice (offered only when `allow_none`).
fn entity_picker(
    current: Option<crate::Entity>,
    none_label: &str,
    allow_none: bool,
    name: &str,
    ui: &mut egui::Ui,
    ctx: &FieldInspectCtx<'_>,
) -> Option<Option<crate::Entity>> {
    let mut picked = None;
    let selected = match current {
        Some(e) => entity_label(ctx.world, e),
        None => none_label.to_owned(),
    };
    egui::ComboBox::from_id_salt(ui.id().with((name, "entity_picker")))
        .selected_text(selected)
        .show_ui(ui, |ui| {
            if allow_none && ui.selectable_label(current.is_none(), none_label).clicked() {
                picked = Some(None);
            }
            for e in ctx.world.iter_entities() {
                let flags = ctx.world.get_entity_flags(e);
                if flags & (crate::Entity::EDITOR | crate::Entity::INHERITED_EDITOR) != 0 {
                    continue;
                }
                let label = entity_label(ctx.world, e);
                if ui.selectable_label(current == Some(e), label).clicked() {
                    picked = Some(Some(e));
                }
            }
        });
    picked
}

/// Jump button next to an entity-reference field: selects the referenced
/// entity (through the editor's `SelectAction`), so a reference seen in
/// the inspector is one click away from the gizmo. No-op without an
/// action queue or with a dead/dangling target.
fn entity_jump_button(
    current: Option<crate::Entity>,
    ui: &mut egui::Ui,
    ctx: &FieldInspectCtx<'_>,
) {
    let Some(target) = current.filter(|&e| ctx.world.is_alive(e)) else {
        return;
    };
    if ui
        .small_button("→")
        .on_hover_text("Select the referenced entity")
        .clicked()
    {
        crate::ui::request_select(ctx.world, vec![target]);
    }
}

impl ComponentField for crate::Entity {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let current = (*self != crate::Entity::DANGLING).then_some(*self);
        ui.horizontal(|ui| {
            ui.label(name);
            let picked = entity_picker(current, "dangling", false, name, ui, ctx);
            entity_jump_button(current, ui, ctx);
            picked
        })
        .inner
        .map(|picked| picked.unwrap_or(crate::Entity::DANGLING))
    }

    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_entity(name, self)
    }

    fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Self, DeserializeError> {
        ctx.read_entity(name)
    }
}

impl ComponentField for Vec<crate::Entity> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut edited: Option<Self> = None;
        ui.vertical(|ui| {
            ui.label(format!("{name} [{}]", self.len()));
            for (i, entity) in self.iter().enumerate() {
                let element = format!("{name}[{i}]");
                ui.horizontal(|ui| {
                    ui.label(format!("[{i}]"));
                    let current = (*entity != crate::Entity::DANGLING).then_some(*entity);
                    if let Some(picked) =
                        entity_picker(current, "dangling", false, &element, ui, ctx)
                    {
                        let mut v = self.clone();
                        v[i] = picked.unwrap_or(crate::Entity::DANGLING);
                        edited = Some(v);
                    }
                    entity_jump_button(current, ui, ctx);
                    if ui.small_button("✕").clicked() {
                        let mut v = self.clone();
                        v.remove(i);
                        edited = Some(v);
                    }
                });
            }
            if ui.small_button("+").clicked() {
                let mut v = self.clone();
                v.push(crate::Entity::DANGLING);
                edited = Some(v);
            }
        });
        edited
    }

    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_entity_list(name, self)
    }

    fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Self, DeserializeError> {
        ctx.read_entity_list(name)
    }
}

impl ComponentField for Option<crate::Entity> {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.horizontal(|ui| {
            ui.label(name);
            let picked = entity_picker(*self, "None", true, name, ui, ctx);
            entity_jump_button(*self, ui, ctx);
            picked
        })
        .inner
    }

    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_optional_entity(name, self)
    }

    fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Self, DeserializeError> {
        ctx.read_optional_entity(name)
    }
}

// ---------------------------------------------------------------------------
// Math types
// ---------------------------------------------------------------------------

impl ComponentField for Vec2 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut v = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                let x = ui
                    .add(egui::DragValue::new(&mut v.x).speed(0.01).prefix("x: "))
                    .changed();
                let y = ui
                    .add(egui::DragValue::new(&mut v.y).speed(0.01).prefix("y: "))
                    .changed();
                x || y
            })
            .inner;
        changed.then_some(v)
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

impl ComponentField for Vec3 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut v = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                let x = ui
                    .add(egui::DragValue::new(&mut v.x).speed(0.01).prefix("x: "))
                    .changed();
                let y = ui
                    .add(egui::DragValue::new(&mut v.y).speed(0.01).prefix("y: "))
                    .changed();
                let z = ui
                    .add(egui::DragValue::new(&mut v.z).speed(0.01).prefix("z: "))
                    .changed();
                x || y || z
            })
            .inner;
        changed.then_some(v)
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

impl ComponentField for Vec4 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        let mut v = *self;
        let changed = ui
            .horizontal(|ui| {
                ui.label(name);
                let x = ui
                    .add(egui::DragValue::new(&mut v.x).speed(0.01).prefix("x: "))
                    .changed();
                let y = ui
                    .add(egui::DragValue::new(&mut v.y).speed(0.01).prefix("y: "))
                    .changed();
                let z = ui
                    .add(egui::DragValue::new(&mut v.z).speed(0.01).prefix("z: "))
                    .changed();
                let w = ui
                    .add(egui::DragValue::new(&mut v.w).speed(0.01).prefix("w: "))
                    .changed();
                x || y || z || w
            })
            .inner;
        changed.then_some(v)
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

impl ComponentField for Quat {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.label(format!(
                "[{:.3}, {:.3}, {:.3}, {:.3}]",
                self.coords.x, self.coords.y, self.coords.z, self.coords.w
            ));
        });
        None // read-only display
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

impl ComponentField for Mat4 {
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.label("(matrix)");
        });
        None // read-only display
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

// ---------------------------------------------------------------------------
// Arc<T> — deduplicating serialization, opaque inspection
// ---------------------------------------------------------------------------

impl<T: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + 'static> ComponentField
    for Arc<T>
{
    fn inspect_field(
        &self,
        name: &str,
        ui: &mut egui::Ui,
        _ctx: &FieldInspectCtx<'_>,
    ) -> Option<Self> {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.weak(format!("({})", std::any::type_name::<Self>()));
        });
        None // opaque, read-only
    }

    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_arc(name, self)
    }

    fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Self, DeserializeError> {
        ctx.read_arc(name)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;
    use crate::serialize::value::Value;

    fn round_trip_serde<T: ComponentField + std::fmt::Debug + PartialEq>(
        value: T,
        field_name: &str,
    ) -> T {
        let world = World::new();
        let mut ctx = SerializeContext::new(&world);
        ctx.begin_struct("Test").unwrap();
        value.serialize_field(field_name, &mut ctx).unwrap();
        let serialized = ctx.end_struct().unwrap();

        let mut world = World::new();
        let mut dctx = DeserializeContext::new(&mut world);
        dctx.load_data(&serialized).unwrap();
        dctx.begin_struct("Test").unwrap();
        let result = T::deserialize_field(field_name, &mut dctx).unwrap();
        dctx.end_struct().unwrap();
        result
    }

    #[test]
    fn round_trip_f32() {
        assert_eq!(round_trip_serde(1.5f32, "x"), 1.5f32);
    }

    #[test]
    fn round_trip_bool() {
        assert!(round_trip_serde(true, "flag"));
    }

    #[test]
    fn round_trip_string() {
        assert_eq!(
            round_trip_serde("hello".to_string(), "name"),
            "hello".to_string()
        );
    }

    #[test]
    fn round_trip_vec3() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(round_trip_serde(v, "pos"), v);
    }

    // Hygiene: a derived Component with fields literally named `ui` / `ctx`
    // (the generated function parameters) must still compile and round-trip —
    // the macro uses `__ui` / `__ctx` internally to avoid being shadowed.
    #[derive(Clone, Debug, PartialEq, crate::Component)]
    struct HygieneProbe {
        ui: f32,
        ctx: f32,
    }

    #[test]
    fn derive_handles_fields_named_ui_and_ctx() {
        use crate::Component;
        let probe = HygieneProbe { ui: 1.5, ctx: 2.5 };

        let world = World::new();
        let mut sctx = SerializeContext::new(&world);
        let serialized = probe.serialize_component(&mut sctx).unwrap();

        let mut world = World::new();
        let mut dctx = DeserializeContext::new(&mut world);
        dctx.load_data(&serialized).unwrap();
        let back = HygieneProbe::deserialize_component(&mut dctx).unwrap();

        assert_eq!(probe, back);
    }

    #[test]
    fn serialize_entity() {
        let entity = crate::Entity::new(42, 100);
        let world = World::new();
        let mut ctx = SerializeContext::new(&world);
        ctx.begin_struct("Test").unwrap();
        entity.serialize_field("e", &mut ctx).unwrap();
        let result = ctx.end_struct().unwrap();
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields[0].1,
                    Value::Entity {
                        index: 42,
                        spawn_tick: 100,
                    }
                );
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn serialize_entity_list() {
        let entities = vec![crate::Entity::new(1, 0), crate::Entity::new(2, 0)];
        let world = World::new();
        let mut ctx = SerializeContext::new(&world);
        ctx.begin_struct("Test").unwrap();
        entities.serialize_field("children", &mut ctx).unwrap();
        let result = ctx.end_struct().unwrap();
        match result {
            Value::Map(fields) => {
                assert!(matches!(&fields[0].1, Value::List(items) if items.len() == 2));
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn serialize_optional_entity() {
        let some = Some(crate::Entity::new(5, 0));
        let none: Option<crate::Entity> = None;
        let world = World::new();

        let mut ctx = SerializeContext::new(&world);
        ctx.begin_struct("Test").unwrap();
        some.serialize_field("parent", &mut ctx).unwrap();
        let result = ctx.end_struct().unwrap();
        match result {
            Value::Map(fields) => {
                assert!(matches!(&fields[0].1, Value::Entity { index: 5, .. }));
            }
            _ => panic!("expected Map"),
        }

        let mut ctx = SerializeContext::new(&world);
        ctx.begin_struct("Test").unwrap();
        none.serialize_field("parent", &mut ctx).unwrap();
        let result = ctx.end_struct().unwrap();
        match result {
            Value::Map(fields) => {
                assert_eq!(fields[0].1, Value::Null);
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn serialize_arc_dedup() {
        let shared = Arc::new("hello".to_string());
        let world = World::new();
        let mut ctx = SerializeContext::new(&world);
        ctx.begin_struct("Test").unwrap();
        shared.serialize_field("a", &mut ctx).unwrap();
        shared.serialize_field("b", &mut ctx).unwrap();
        let result = ctx.end_struct().unwrap();
        match result {
            Value::Map(fields) => {
                assert!(matches!(&fields[0].1, Value::ArcValue { id: 0, .. }));
                assert_eq!(fields[1].1, Value::ArcRef(0));
            }
            _ => panic!("expected Map"),
        }
    }
}
