//! Serialized prefab data structures for file I/O.
//!
//! A [`SerializedPrefab`] is the on-disk representation of an entity
//! tree. It can be encoded to RON or bincode via the [`format`](super::format)
//! module.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::value::Value;

/// Schema hash for a component (blake3 hex digest).
///
/// Used to detect field reordering or schema changes during deserialization.
/// Empty string means no schema validation for that component.
pub type SchemaHash = String;

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
    /// The schema version of the component when serialized.
    ///
    /// Used during deserialization to select and execute the appropriate migration
    /// when the current component schema differs from what was saved.
    /// Defaults to 1 for backward compatibility with pre-Phase-4 snapshots.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// The serialized field data.
    pub data: Value,
}

/// Snapshot metadata: engine fingerprint, timestamp, and per-component schema hashes.
///
/// Stores schema hashes to detect field reordering or type changes during restore.
/// Phase 5 feature: enables safe reload by rejecting snapshots with mismatched schemas
/// (silent corruption via positional codec desyncs is prevented).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Engine generation ID (fingerprint) when this snapshot was captured.
    pub engine_fingerprint: String,
    /// Timestamp when the snapshot was captured (seconds since Unix epoch).
    pub timestamp: u64,
    /// Per-component schema hashes: component name → blake3 hash of (field_names, field_types).
    /// Empty if component doesn't track schema; validated on deserialize if present.
    pub component_schemas: HashMap<String, SchemaHash>,
}

impl SnapshotMetadata {
    /// Create new snapshot metadata with current timestamp.
    pub fn new(engine_fingerprint: String) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            engine_fingerprint,
            timestamp,
            component_schemas: HashMap::new(),
        }
    }
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
    /// Snapshot metadata: engine fingerprint, timestamp, and component schema hashes.
    /// Phase 5: enables schema validation to prevent silent corruption on reload.
    #[serde(default)]
    pub metadata: SnapshotMetadata,
}

/// A single resource's serialized data (mirrors [`SerializedComponent`]).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializedResource {
    /// The resource type name (matches
    /// [`SnapshotResource::NAME`](crate::SnapshotResource::NAME)).
    pub type_name: String,
    /// The schema version of the resource when serialized.
    ///
    /// Used during deserialization to select and execute the appropriate migration
    /// when the current resource schema differs from what was saved.
    /// Defaults to 1 for backward compatibility with pre-Phase-4 snapshots.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// The serialized resource data.
    pub data: Value,
}

fn default_schema_version() -> u32 {
    1
}
