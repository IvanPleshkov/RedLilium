//! Field-level serialization and deserialization wrappers.
//!
//! Uses the same method-resolution-priority trick as
//! [`Inspect`](crate::inspect::Inspect) and
//! [`EntityRef`](crate::map_entities::EntityRef): inherent methods on
//! wrapper types for known types (Entity, Arc) take precedence over
//! blanket fallback trait impls.
//!
//! The `#[derive(Component)]` macro generates `serialize_component` by
//! wrapping each field in `SerializeField(&self.field).serialize_field(name, ctx)`.

use std::marker::PhantomData;
use std::sync::Arc;

use super::context::{DeserializeContext, SerializeContext};
use super::error::{DeserializeError, SerializeError};
use crate::Entity;

// ---------------------------------------------------------------------------
// Serialize
// ---------------------------------------------------------------------------

/// Wrapper for serializing a single component field.
///
/// Inherent `serialize_field` methods for known types (Entity, Arc)
/// take priority over the [`SerializeFieldFallback`] blanket trait impl.
pub struct SerializeField<'a, T: ?Sized>(pub &'a T);

/// Fallback trait for serializing fields of types that implement
/// [`serde::Serialize`].
///
/// The blanket impl converts the value to a [`Value`](super::Value)
/// via serde. Rust's method resolution ensures this is only used when
/// no inherent `serialize_field` method exists.
pub trait SerializeFieldFallback {
    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError>;
}

impl<T: serde::Serialize + 'static> SerializeFieldFallback for SerializeField<'_, T> {
    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_serde(name, self.0)
    }
}

// --- Entity inherent impls ---

impl SerializeField<'_, Entity> {
    pub fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_entity(name, self.0)
    }
}

impl SerializeField<'_, Vec<Entity>> {
    pub fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_entity_list(name, self.0)
    }
}

impl SerializeField<'_, Option<Entity>> {
    pub fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_optional_entity(name, self.0)
    }
}

// --- Arc inherent impl (generic, takes priority over fallback for Arc types) ---

impl<T: serde::Serialize + Send + Sync + 'static> SerializeField<'_, Arc<T>> {
    pub fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        ctx.write_arc(name, self.0)
    }
}

// ---------------------------------------------------------------------------
// Deserialize
// ---------------------------------------------------------------------------

/// Wrapper for deserializing a single component field.
///
/// Inherent `deserialize_field` methods for known types (Entity, Arc)
/// take priority over the [`DeserializeFieldFallback`] blanket trait impl.
pub struct DeserializeField<T>(pub PhantomData<T>);

/// Fallback trait for deserializing fields of types that implement
/// [`serde::de::DeserializeOwned`].
///
/// The blanket impl converts from a [`Value`](super::Value) via serde.
pub trait DeserializeFieldFallback<T> {
    fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<T, DeserializeError>;
}

impl<T: serde::de::DeserializeOwned + 'static> DeserializeFieldFallback<T> for DeserializeField<T> {
    fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<T, DeserializeError> {
        ctx.read_serde(name)
    }
}

// --- Entity inherent impls ---

impl DeserializeField<Entity> {
    pub fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Entity, DeserializeError> {
        ctx.read_entity(name)
    }
}

impl DeserializeField<Vec<Entity>> {
    pub fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Vec<Entity>, DeserializeError> {
        ctx.read_entity_list(name)
    }
}

impl DeserializeField<Option<Entity>> {
    pub fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Option<Entity>, DeserializeError> {
        ctx.read_optional_entity(name)
    }
}

// --- Arc inherent impl ---

impl<T: serde::de::DeserializeOwned + Send + Sync + 'static> DeserializeField<Arc<T>> {
    pub fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Arc<T>, DeserializeError> {
        ctx.read_arc(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;
    use crate::serialize::context::SerializeContext;
    use crate::serialize::value::Value;

    #[test]
    fn serialize_f32_via_fallback() {
        let world = World::new();
        let mut ctx = SerializeContext::new(&world);
        ctx.begin_struct("Test").unwrap();

        use super::SerializeFieldFallback as _;
        SerializeField(&1.5f32)
            .serialize_field("x", &mut ctx)
            .unwrap();

        let result = ctx.end_struct().unwrap();
        match result {
            Value::Map(fields) => {
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].0, "x");
                assert_eq!(fields[0].1, Value::F32(1.5));
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn serialize_entity_via_inherent() {
        let world = World::new();
        let mut ctx = SerializeContext::new(&world);
        ctx.begin_struct("Test").unwrap();

        // Entity inherent method takes priority - no fallback import needed
        let entity = Entity::new(42, 100);
        SerializeField(&entity)
            .serialize_field("e", &mut ctx)
            .unwrap();

        let result = ctx.end_struct().unwrap();
        match result {
            Value::Map(fields) => {
                assert_eq!(
                    fields[0].1,
                    Value::Entity {
                        index: 42,
                        spawn_tick: 100
                    }
                );
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn serialize_arc_dedup() {
        let world = World::new();
        let mut ctx = SerializeContext::new(&world);
        ctx.begin_struct("Test").unwrap();

        let shared = Arc::new("hello".to_string());
        SerializeField(&shared)
            .serialize_field("a", &mut ctx)
            .unwrap();
        SerializeField(&shared)
            .serialize_field("b", &mut ctx)
            .unwrap();

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
