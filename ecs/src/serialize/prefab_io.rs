//! Serialized prefab data structures for file I/O.
//!
//! A [`SerializedPrefab`] is the on-disk representation of an entity
//! tree. It can be encoded to RON or bincode via the [`format`](super::format)
//! module.

use serde::{Deserialize, Serialize};

use super::value::Value;

/// A fully serialized prefab (entity tree), suitable for file I/O.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedPrefab {
    /// Serialized entities in BFS order. Index 0 is the root.
    pub entities: Vec<SerializedEntity>,
}

/// A single entity's serialized component data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedEntity {
    /// The original entity index (for entity remapping during deserialization).
    pub entity_index: u32,
    /// The original entity spawn tick (for entity remapping during deserialization).
    pub entity_spawn_tick: u64,
    /// All serializable components on this entity.
    pub components: Vec<SerializedComponent>,
}

/// A single component's serialized data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedComponent {
    /// The component type name (matches [`Component::NAME`](crate::Component::NAME)).
    pub type_name: String,
    /// The serialized field data.
    pub data: Value,
}
