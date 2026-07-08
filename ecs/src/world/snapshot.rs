//! Opt-in snapshot-resource registry.
//!
//! Whole-world snapshots ([`World::serialize_world`] /
//! [`World::deserialize_world_into`]) capture entities unconditionally, but a
//! resource crosses the boundary only if it opts in via
//! [`SnapshotResource`](crate::SnapshotResource) and is registered with
//! [`World::register_snapshot_resource`]. This module holds the type-erased
//! capture/restore hooks that let the whole-world path enumerate exactly the
//! marked resources.

use crate::resource::SnapshotResource;
use crate::serialize::value;
use crate::serialize::{DeserializeError, SerializeError, SerializedResource};

use super::World;

/// Type-erased capture/restore hooks for one registered snapshot resource.
pub(crate) struct SnapshotResourceEntry {
    /// Reads the resource (if present) and serializes it. Returns `Ok(None)`
    /// when the resource type is registered but not currently in the world.
    capture: fn(&World) -> Result<Option<SerializedResource>, SerializeError>,
    /// Deserializes the resource and inserts (or replaces) it in the world.
    restore: fn(&mut World, &value::Value) -> Result<(), DeserializeError>,
}

/// Monomorphized capture hook for `T` (stored as a `fn` pointer).
fn capture_snapshot_resource<T: SnapshotResource>(
    world: &World,
) -> Result<Option<SerializedResource>, SerializeError> {
    if !world.has_resource::<T>() {
        return Ok(None);
    }
    let res = world.resource::<T>();
    let data = value::to_value(&*res)?;
    Ok(Some(SerializedResource {
        type_name: T::NAME.to_owned(),
        schema_version: T::SCHEMA_VERSION,
        data,
    }))
}

/// Monomorphized restore hook for `T` (stored as a `fn` pointer).
fn restore_snapshot_resource<T: SnapshotResource>(
    world: &mut World,
    data: &value::Value,
) -> Result<(), DeserializeError> {
    let res: T = value::from_value(data.clone())?;
    world.insert_resource(res);
    Ok(())
}

impl World {
    /// Registers a resource type for whole-world snapshots.
    ///
    /// After this call, [`serialize_world`](World::serialize_world) captures
    /// the resource (when present) and [`deserialize_world_into`](World::deserialize_world_into)
    /// restores it. Idempotent: registering the same type twice is a no-op.
    ///
    /// Registration only installs the capture/restore hooks — it does not
    /// insert the resource. A registered-but-absent resource is simply skipped
    /// at capture time.
    pub fn register_snapshot_resource<T: SnapshotResource>(&mut self) {
        self.snapshot_resources
            .entry(T::NAME)
            .or_insert_with(|| SnapshotResourceEntry {
                capture: capture_snapshot_resource::<T>,
                restore: restore_snapshot_resource::<T>,
            });
    }

    /// Captures every registered snapshot resource that is currently present.
    pub(crate) fn capture_snapshot_resources(
        &self,
    ) -> Result<Vec<SerializedResource>, SerializeError> {
        let mut out = Vec::new();
        for entry in self.snapshot_resources.values() {
            if let Some(res) = (entry.capture)(self)? {
                out.push(res);
            }
        }
        Ok(out)
    }

    /// Restores captured snapshot resources into this world.
    ///
    /// A serialized resource whose type is not registered here is skipped
    /// silently (mirrors the unknown-component handling in the entity path):
    /// a snapshot taken with a plugin loaded can be restored into a world that
    /// no longer has that plugin.
    pub(crate) fn restore_snapshot_resources(
        &mut self,
        resources: &[SerializedResource],
    ) -> Result<(), DeserializeError> {
        for res in resources {
            // Copy the `fn` pointer out first so the immutable borrow of the
            // registry ends before the `&mut World` call.
            let restore = self
                .snapshot_resources
                .get(res.type_name.as_str())
                .map(|e| e.restore);
            if let Some(restore) = restore {
                restore(self, &res.data)?;
            }
        }
        Ok(())
    }
}
