//! Serialization and deserialization contexts.
//!
//! [`SerializeContext`] tracks Arc deduplication and provides access to
//! the [`World`](crate::World) for custom serialization.
//! [`DeserializeContext`] provides field access and entity remapping.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use super::error::{DeserializeError, SerializeError};
use super::value::{self, Value};
use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// SerializeContext
// ---------------------------------------------------------------------------

/// Context for serializing component fields.
///
/// Provides field accumulation, Arc deduplication by pointer identity,
/// and access to the [`World`] for resource-based custom serialization.
pub struct SerializeContext<'w> {
    world: &'w World,
    fields: Vec<(String, Value)>,
    arc_ids: HashMap<usize, u32>,
    next_arc_id: u32,
}

impl<'w> SerializeContext<'w> {
    /// Create a new context for serialization.
    pub fn new(world: &'w World) -> Self {
        Self {
            world,
            fields: Vec::new(),
            arc_ids: HashMap::new(),
            next_arc_id: 0,
        }
    }

    /// Get a reference to the world (for resource access in custom impls).
    pub fn world(&self) -> &World {
        self.world
    }

    /// Begin serializing a struct component.
    pub fn begin_struct(&mut self, _name: &str) -> Result<(), SerializeError> {
        self.fields.clear();
        Ok(())
    }

    /// Write a pre-built Value for a field.
    pub fn write_field(&mut self, name: &str, value: Value) -> Result<(), SerializeError> {
        self.fields.push((name.to_owned(), value));
        Ok(())
    }

    /// Write a serde-serializable value as a field.
    pub fn write_serde<T: serde::Serialize>(
        &mut self,
        name: &str,
        val: &T,
    ) -> Result<(), SerializeError> {
        let value = value::to_value(val).map_err(|e| SerializeError::FieldError {
            field: name.to_owned(),
            message: e.to_string(),
        })?;
        self.write_field(name, value)
    }

    /// Write an entity reference field.
    pub fn write_entity(&mut self, name: &str, entity: &Entity) -> Result<(), SerializeError> {
        self.write_field(
            name,
            Value::Entity {
                index: entity.index(),
                spawn_tick: entity.spawn_tick(),
            },
        )
    }

    /// Write a list of entity references.
    pub fn write_entity_list(
        &mut self,
        name: &str,
        entities: &[Entity],
    ) -> Result<(), SerializeError> {
        let values = entities
            .iter()
            .map(|e| Value::Entity {
                index: e.index(),
                spawn_tick: e.spawn_tick(),
            })
            .collect();
        self.write_field(name, Value::List(values))
    }

    /// Write an optional entity reference.
    pub fn write_optional_entity(
        &mut self,
        name: &str,
        entity: &Option<Entity>,
    ) -> Result<(), SerializeError> {
        let value = match entity {
            Some(e) => Value::Entity {
                index: e.index(),
                spawn_tick: e.spawn_tick(),
            },
            None => Value::Null,
        };
        self.write_field(name, value)
    }

    /// Write an Arc-wrapped value with deduplication.
    ///
    /// The first time an Arc is seen (by pointer identity), its data is
    /// serialized inline as [`Value::ArcValue`]. Subsequent occurrences
    /// emit [`Value::ArcRef`] with the same ID.
    pub fn write_arc<T: serde::Serialize + Send + Sync + 'static>(
        &mut self,
        name: &str,
        arc: &Arc<T>,
    ) -> Result<(), SerializeError> {
        let ptr = Arc::as_ptr(arc) as usize;
        if let Some(&id) = self.arc_ids.get(&ptr) {
            self.write_field(name, Value::ArcRef(id))
        } else {
            let id = self.next_arc_id;
            self.next_arc_id += 1;
            self.arc_ids.insert(ptr, id);
            let inner = value::to_value(arc.as_ref()).map_err(|e| SerializeError::FieldError {
                field: name.to_owned(),
                message: e.to_string(),
            })?;
            self.write_field(
                name,
                Value::ArcValue {
                    id,
                    inner: Box::new(inner),
                },
            )
        }
    }

    /// Finish struct serialization and return the accumulated fields as a Value.
    pub fn end_struct(&mut self) -> Result<Value, SerializeError> {
        Ok(Value::Map(std::mem::take(&mut self.fields)))
    }
}

// ---------------------------------------------------------------------------
// DeserializeContext
// ---------------------------------------------------------------------------

/// What to do with a serialized entity reference that is not in the remap
/// table (its target was not part of the serialized set).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum UnmappedEntityPolicy {
    /// Fail deserialization (safe default for contexts without a remap
    /// table, e.g. single-component import).
    #[default]
    Error,
    /// Keep the original handle. Correct for same-world snapshots (undo):
    /// a still-alive target stays linked, a dead one stays dangling.
    Keep,
    /// Resolve to [`Entity::DANGLING`]. Correct for prefab assets loaded
    /// into a foreign world, where the original handle could collide with
    /// an unrelated live entity — a reference that was dangling at export
    /// stays dangling after instantiation.
    Dangling,
}

/// Context for deserializing component fields.
///
/// Provides field-by-field access, Arc deduplication cache, entity
/// remapping, and access to the [`World`] for custom deserialization.
pub struct DeserializeContext<'w> {
    world: &'w mut World,
    fields: HashMap<String, Value>,
    arc_cache: HashMap<u32, Arc<dyn Any + Send + Sync>>,
    /// Raw (still-typed-as-`Value`) inner of every `ArcValue` seen during a
    /// pre-scan, keyed by dedup id. Lets an `ArcRef` be resolved even when the
    /// component that carried the original `ArcValue` was skipped (unknown type),
    /// by deserializing the inner at the (known-typed) `ArcRef` site instead.
    arc_raw: HashMap<u32, Value>,
    entity_map: HashMap<(u32, u64), Entity>,
    /// How to resolve references missing from `entity_map`.
    unmapped_policy: UnmappedEntityPolicy,
    /// Name of the component currently being deserialized, used to enrich
    /// [`DeserializeError::MissingField`] / `TypeMismatch` diagnostics.
    current_component: String,
}

impl<'w> DeserializeContext<'w> {
    /// Create a new context for deserialization.
    pub fn new(world: &'w mut World) -> Self {
        Self {
            world,
            fields: HashMap::new(),
            arc_cache: HashMap::new(),
            arc_raw: HashMap::new(),
            entity_map: HashMap::new(),
            unmapped_policy: UnmappedEntityPolicy::default(),
            current_component: String::new(),
        }
    }

    /// Sets how references missing from the remap table are resolved.
    pub fn set_unmapped_policy(&mut self, policy: UnmappedEntityPolicy) {
        self.unmapped_policy = policy;
    }

    /// Sets the name of the component about to be deserialized, used to enrich
    /// error diagnostics. Cleared after the component is processed.
    pub fn set_current_component(&mut self, name: &str) {
        self.current_component = name.to_owned();
    }

    /// Pre-scans a serialized component [`Value`] tree, recording the raw inner
    /// of every [`Value::ArcValue`] by id. Call this for *all* components of a
    /// prefab (including those whose type is unknown/skipped) before
    /// deserializing any of them, so a shared Arc first introduced on a skipped
    /// component can still be resolved by a later [`Value::ArcRef`].
    pub fn prescan_arc_values(&mut self, value: &Value) {
        match value {
            Value::ArcValue { id, inner } => {
                self.arc_raw.entry(*id).or_insert_with(|| (**inner).clone());
                self.prescan_arc_values(inner);
            }
            Value::List(items) => {
                for v in items {
                    self.prescan_arc_values(v);
                }
            }
            Value::Map(entries) => {
                for (_, v) in entries {
                    self.prescan_arc_values(v);
                }
            }
            _ => {}
        }
    }

    /// Get a reference to the world (for resource access in custom impls).
    pub fn world(&self) -> &World {
        self.world
    }

    /// Get a mutable reference to the world (for resource access in custom impls).
    pub fn world_mut(&mut self) -> &mut World {
        self.world
    }

    /// Set the entity remapping table.
    pub fn set_entity_map(&mut self, map: HashMap<(u32, u64), Entity>) {
        self.entity_map = map;
    }

    /// Load serialized component data into the context.
    ///
    /// Call this before [`begin_struct`](Self::begin_struct) / `T::deserialize_component(ctx)`.
    /// The `deserialize_prefab` method calls this automatically.
    pub fn load_data(&mut self, data: &Value) -> Result<(), DeserializeError> {
        self.fields.clear();
        match data {
            Value::Map(entries) => {
                for (k, v) in entries {
                    self.fields.insert(k.clone(), v.clone());
                }
                Ok(())
            }
            _ => Err(DeserializeError::FormatError(
                "expected Map value for struct".into(),
            )),
        }
    }

    /// Begin deserializing a struct component.
    ///
    /// Fields should already be loaded via [`load_data`](Self::load_data).
    pub fn begin_struct(&mut self, _name: &str) -> Result<(), DeserializeError> {
        Ok(())
    }

    /// Read a raw Value for a field.
    pub fn read_field(&mut self, name: &str) -> Result<Value, DeserializeError> {
        self.fields
            .remove(name)
            .ok_or_else(|| DeserializeError::MissingField {
                field: name.to_owned(),
                component: self.current_component.clone(),
            })
    }

    /// Read a serde-deserializable value from a field.
    pub fn read_serde<T: serde::de::DeserializeOwned>(
        &mut self,
        name: &str,
    ) -> Result<T, DeserializeError> {
        let val = self.read_field(name)?;
        value::from_value(val).map_err(|e| DeserializeError::FormatError(e.to_string()))
    }

    /// Resolves a serialized entity reference through the remap table,
    /// falling back to the configured [`UnmappedEntityPolicy`] on a miss.
    fn resolve_entity(
        &self,
        field: &str,
        index: u32,
        spawn_tick: u64,
    ) -> Result<Entity, DeserializeError> {
        if let Some(&mapped) = self.entity_map.get(&(index, spawn_tick)) {
            return Ok(mapped);
        }
        match self.unmapped_policy {
            // Never fabricate a handle by default — it could silently alias an
            // unrelated or dead entity in the destination world.
            UnmappedEntityPolicy::Error => Err(DeserializeError::UnmappedEntityReference {
                field: field.to_owned(),
                index,
                spawn_tick,
            }),
            UnmappedEntityPolicy::Keep => Ok(Entity::new(index, spawn_tick)),
            UnmappedEntityPolicy::Dangling => Ok(Entity::DANGLING),
        }
    }

    /// Read an entity reference, applying the remap table.
    pub fn read_entity(&mut self, name: &str) -> Result<Entity, DeserializeError> {
        let val = self.read_field(name)?;
        match val {
            Value::Entity { index, spawn_tick } => self.resolve_entity(name, index, spawn_tick),
            _ => Err(DeserializeError::TypeMismatch {
                field: name.to_owned(),
                expected: "Entity".into(),
                found: format!("{val:?}"),
            }),
        }
    }

    /// Read a list of entity references, applying the remap table.
    pub fn read_entity_list(&mut self, name: &str) -> Result<Vec<Entity>, DeserializeError> {
        let val = self.read_field(name)?;
        match val {
            Value::List(items) => items
                .into_iter()
                .map(|v| match v {
                    Value::Entity { index, spawn_tick } => {
                        self.resolve_entity(name, index, spawn_tick)
                    }
                    _ => Err(DeserializeError::TypeMismatch {
                        field: name.to_owned(),
                        expected: "Entity".into(),
                        found: format!("{v:?}"),
                    }),
                })
                .collect(),
            _ => Err(DeserializeError::TypeMismatch {
                field: name.to_owned(),
                expected: "List of Entity".into(),
                found: format!("{val:?}"),
            }),
        }
    }

    /// Read an optional entity reference, applying the remap table.
    ///
    /// Under [`UnmappedEntityPolicy::Error`] an unmapped reference resolves to
    /// `None` (the field is optional — dropping the link is the lenient
    /// counterpart of the bare-`Entity` error); the other policies resolve it
    /// like [`read_entity`](Self::read_entity) does.
    pub fn read_optional_entity(&mut self, name: &str) -> Result<Option<Entity>, DeserializeError> {
        let val = self.read_field(name)?;
        match val {
            Value::Null => Ok(None),
            Value::Entity { index, spawn_tick } => {
                if self.unmapped_policy == UnmappedEntityPolicy::Error {
                    Ok(self.entity_map.get(&(index, spawn_tick)).copied())
                } else {
                    self.resolve_entity(name, index, spawn_tick).map(Some)
                }
            }
            _ => Err(DeserializeError::TypeMismatch {
                field: name.to_owned(),
                expected: "Entity or Null".into(),
                found: format!("{val:?}"),
            }),
        }
    }

    /// Read an Arc-wrapped value with deduplication support.
    ///
    /// [`Value::ArcValue`] deserializes the inner data, wraps in Arc,
    /// and caches for reuse. [`Value::ArcRef`] returns a clone of the
    /// cached Arc.
    pub fn read_arc<T: serde::de::DeserializeOwned + Send + Sync + 'static>(
        &mut self,
        name: &str,
    ) -> Result<Arc<T>, DeserializeError> {
        let val = self.read_field(name)?;
        match val {
            Value::ArcValue { id, inner } => {
                let data: T = value::from_value(*inner).map_err(|e| {
                    DeserializeError::FormatError(format!(
                        "failed to deserialize Arc inner for field '{name}': {e}"
                    ))
                })?;
                let arc = Arc::new(data);
                self.arc_cache.insert(id, arc.clone());
                Ok(arc)
            }
            Value::ArcRef(id) => {
                if let Some(any_arc) = self.arc_cache.get(&id) {
                    let typed = any_arc.clone().downcast::<T>().map_err(|_| {
                        DeserializeError::TypeMismatch {
                            field: name.to_owned(),
                            expected: std::any::type_name::<T>().to_owned(),
                            found: "Arc of different type".into(),
                        }
                    })?;
                    return Ok(typed);
                }
                // Not yet materialized as a typed Arc — the original `ArcValue`
                // was on a skipped/earlier component. Deserialize the raw inner
                // recorded by `prescan_arc_values` as `T` here (we know the type
                // at this site), then cache it so further `ArcRef`s share it.
                let inner = self
                    .arc_raw
                    .get(&id)
                    .ok_or(DeserializeError::InvalidArcRef { id })?
                    .clone();
                let data: T = value::from_value(inner).map_err(|e| {
                    DeserializeError::FormatError(format!(
                        "failed to deserialize shared Arc inner for field '{name}': {e}"
                    ))
                })?;
                let arc = Arc::new(data);
                self.arc_cache.insert(id, arc.clone());
                Ok(arc)
            }
            _ => {
                // Fallback: try to deserialize as a plain value and wrap in Arc
                let data: T = value::from_value(val)
                    .map_err(|e| DeserializeError::FormatError(e.to_string()))?;
                Ok(Arc::new(data))
            }
        }
    }

    /// Finish struct deserialization.
    pub fn end_struct(&mut self) -> Result<(), DeserializeError> {
        self.fields.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc_ref_resolves_when_value_was_on_skipped_component() {
        // The original `ArcValue` lives on a component we never deserialize
        // (unknown type → skipped). A later registered component references it
        // via `ArcRef`. After pre-scanning, the ref must still resolve.
        let mut world = World::new();
        let mut ctx = DeserializeContext::new(&mut world);

        let skipped = Value::Map(vec![(
            "shared".to_string(),
            Value::ArcValue {
                id: 7,
                inner: Box::new(Value::U64(42)),
            },
        )]);
        ctx.prescan_arc_values(&skipped);

        let registered = Value::Map(vec![("shared".to_string(), Value::ArcRef(7))]);
        ctx.load_data(&registered).unwrap();
        let arc: Arc<u64> = ctx.read_arc("shared").unwrap();
        assert_eq!(*arc, 42);
    }

    #[test]
    fn two_arc_refs_to_skipped_value_share_one_arc() {
        let mut world = World::new();
        let mut ctx = DeserializeContext::new(&mut world);

        ctx.prescan_arc_values(&Value::ArcValue {
            id: 3,
            inner: Box::new(Value::U64(100)),
        });

        ctx.load_data(&Value::Map(vec![("a".to_string(), Value::ArcRef(3))]))
            .unwrap();
        let first: Arc<u64> = ctx.read_arc("a").unwrap();

        ctx.load_data(&Value::Map(vec![("b".to_string(), Value::ArcRef(3))]))
            .unwrap();
        let second: Arc<u64> = ctx.read_arc("b").unwrap();

        assert!(Arc::ptr_eq(&first, &second), "ArcRefs must share one Arc");
    }

    #[test]
    fn unknown_arc_ref_still_errors() {
        let mut world = World::new();
        let mut ctx = DeserializeContext::new(&mut world);
        ctx.load_data(&Value::Map(vec![("x".to_string(), Value::ArcRef(99))]))
            .unwrap();
        let res: Result<Arc<u64>, _> = ctx.read_arc("x");
        assert!(matches!(
            res,
            Err(DeserializeError::InvalidArcRef { id: 99 })
        ));
    }
}
