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

/// Context for deserializing component fields.
///
/// Provides field-by-field access, Arc deduplication cache, entity
/// remapping, and access to the [`World`] for custom deserialization.
pub struct DeserializeContext<'w> {
    world: &'w mut World,
    fields: HashMap<String, Value>,
    arc_cache: HashMap<u32, Arc<dyn Any + Send + Sync>>,
    entity_map: HashMap<(u32, u64), Entity>,
}

impl<'w> DeserializeContext<'w> {
    /// Create a new context for deserialization.
    pub fn new(world: &'w mut World) -> Self {
        Self {
            world,
            fields: HashMap::new(),
            arc_cache: HashMap::new(),
            entity_map: HashMap::new(),
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
                component: String::new(),
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

    /// Read an entity reference, applying the remap table.
    pub fn read_entity(&mut self, name: &str) -> Result<Entity, DeserializeError> {
        let val = self.read_field(name)?;
        match val {
            Value::Entity { index, spawn_tick } => Ok(self
                .entity_map
                .get(&(index, spawn_tick))
                .copied()
                .unwrap_or(Entity::new(index, spawn_tick))),
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
                    Value::Entity { index, spawn_tick } => Ok(self
                        .entity_map
                        .get(&(index, spawn_tick))
                        .copied()
                        .unwrap_or(Entity::new(index, spawn_tick))),
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
    pub fn read_optional_entity(&mut self, name: &str) -> Result<Option<Entity>, DeserializeError> {
        let val = self.read_field(name)?;
        match val {
            Value::Null => Ok(None),
            Value::Entity { index, spawn_tick } => Ok(Some(
                self.entity_map
                    .get(&(index, spawn_tick))
                    .copied()
                    .unwrap_or(Entity::new(index, spawn_tick)),
            )),
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
                let any_arc = self
                    .arc_cache
                    .get(&id)
                    .ok_or(DeserializeError::InvalidArcRef { id })?;
                let typed = any_arc.clone().downcast::<T>().map_err(|_| {
                    DeserializeError::TypeMismatch {
                        field: name.to_owned(),
                        expected: std::any::type_name::<T>().to_owned(),
                        found: "Arc of different type".into(),
                    }
                })?;
                Ok(typed)
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
