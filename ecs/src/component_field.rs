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
//!     fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
//!         ui.horizontal(|ui| {
//!             ui.label(name);
//!             // custom color picker widget
//!         });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui);

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
}

// ---------------------------------------------------------------------------
// Primitive types
// ---------------------------------------------------------------------------

impl ComponentField for f32 {
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(egui::DragValue::new(self).speed(0.01));
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(egui::DragValue::new(self).speed(0.01));
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.checkbox(self, "");
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        let mut v = *self as i32;
        ui.horizontal(|ui| {
            ui.label(name);
            if ui
                .add(egui::DragValue::new(&mut v).range(0..=255))
                .changed()
            {
                *self = v as u8;
            }
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        let mut v = *self as i64;
        ui.horizontal(|ui| {
            ui.label(name);
            if ui
                .add(egui::DragValue::new(&mut v).range(0..=u32::MAX as i64))
                .changed()
            {
                *self = v as u32;
            }
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(egui::DragValue::new(self));
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        let mut v = *self as i64;
        ui.horizontal(|ui| {
            ui.label(name);
            if ui
                .add(egui::DragValue::new(&mut v).range(0..=i64::MAX))
                .changed()
            {
                *self = v as u64;
            }
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        let mut v = *self as i64;
        ui.horizontal(|ui| {
            ui.label(name);
            if ui
                .add(egui::DragValue::new(&mut v).range(0..=i64::MAX))
                .changed()
            {
                *self = v as usize;
            }
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.text_edit_singleline(self);
        });
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

impl ComponentField for crate::Entity {
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.label(format!("Entity({}@{})", self.index(), self.spawn_tick()));
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.label(format!("[{} entities]", self.len()));
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            match self {
                Some(e) => ui.label(format!("Entity({}@{})", e.index(), e.spawn_tick())),
                None => ui.weak("None"),
            };
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(egui::DragValue::new(&mut self.x).speed(0.01).prefix("x: "));
            ui.add(egui::DragValue::new(&mut self.y).speed(0.01).prefix("y: "));
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(egui::DragValue::new(&mut self.x).speed(0.01).prefix("x: "));
            ui.add(egui::DragValue::new(&mut self.y).speed(0.01).prefix("y: "));
            ui.add(egui::DragValue::new(&mut self.z).speed(0.01).prefix("z: "));
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.add(egui::DragValue::new(&mut self.x).speed(0.01).prefix("x: "));
            ui.add(egui::DragValue::new(&mut self.y).speed(0.01).prefix("y: "));
            ui.add(egui::DragValue::new(&mut self.z).speed(0.01).prefix("z: "));
            ui.add(egui::DragValue::new(&mut self.w).speed(0.01).prefix("w: "));
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.label(format!(
                "[{:.3}, {:.3}, {:.3}, {:.3}]",
                self.coords.x, self.coords.y, self.coords.z, self.coords.w
            ));
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.label("(matrix)");
        });
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
    fn inspect_field(&mut self, name: &str, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(name);
            ui.weak(format!("({})", std::any::type_name::<Self>()));
        });
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
