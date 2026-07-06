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
    /// Per-entity flag bits (disabled, static, editor, etc.).
    pub entity_flags: u32,
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

/// A whole-world snapshot: every alive entity plus the opted-in snapshot
/// resources.
///
/// Produced by [`World::serialize_world`](crate::World::serialize_world) and
/// restored by [`World::deserialize_world_into`](crate::World::deserialize_world_into).
/// This is the single snapshot mechanism shared by Play mode, scene save/load,
/// and warm-restart reload. Unlike [`SerializedPrefab`] (a subtree), it carries
/// no BFS root — entity references remap across the whole set, and a reference
/// to an entity that is not in the snapshot resolves to
/// [`Entity::DANGLING`](crate::Entity::DANGLING).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedWorld {
    /// All alive entities' serialized components, in
    /// [`World::iter_entities`](crate::World::iter_entities) order.
    pub entities: Vec<SerializedEntity>,
    /// Opted-in snapshot resources (see
    /// [`SnapshotResource`](crate::SnapshotResource)).
    pub resources: Vec<SerializedResource>,
}

/// A single resource's serialized data (mirrors [`SerializedComponent`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedResource {
    /// The resource type name (matches
    /// [`SnapshotResource::NAME`](crate::SnapshotResource::NAME)).
    pub type_name: String,
    /// The serialized resource data.
    pub data: Value,
}
