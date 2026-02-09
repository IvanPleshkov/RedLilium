/// Per-component RwLock layer for multi-threaded system execution.
///
/// In multi-threaded mode, multiple systems may run concurrently on different
/// threads. `WorldLocks` provides an external locking layer that systems acquire
/// in TypeId-sorted order to prevent deadlocks.
///
/// In single-threaded mode, this module is unused — no locking overhead.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod inner {
    use std::any::TypeId;
    use std::collections::HashMap;
    use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

    use crate::access_set::AccessInfo;

    /// A lock guard for either a read or write lock on a component storage.
    ///
    /// The guard is held purely for its RAII drop behavior (releasing the lock).
    #[allow(dead_code)]
    pub enum LockGuard<'a> {
        Read(RwLockReadGuard<'a, ()>),
        Write(RwLockWriteGuard<'a, ()>),
    }

    /// External per-component RwLock layer.
    ///
    /// Created by the multi-threaded runner before executing systems.
    /// Each registered component and resource type gets its own RwLock.
    pub struct WorldLocks {
        locks: HashMap<TypeId, RwLock<()>>,
    }

    impl WorldLocks {
        /// Creates a new WorldLocks with the given set of TypeIds.
        ///
        /// Call with the union of all component and resource TypeIds
        /// registered in the World.
        pub fn new(type_ids: impl IntoIterator<Item = TypeId>) -> Self {
            let locks = type_ids
                .into_iter()
                .map(|id| (id, RwLock::new(())))
                .collect();
            Self { locks }
        }

        /// Acquires locks in TypeId-sorted order, preventing deadlocks.
        ///
        /// For each `AccessInfo`:
        /// - `is_write == true` → exclusive (write) lock
        /// - `is_write == false` → shared (read) lock
        ///
        /// Types not present in the lock map are skipped (e.g. unregistered
        /// Optional types).
        ///
        /// Returns a list of guards that release on drop.
        pub fn acquire_sorted(&self, infos: &[AccessInfo]) -> Vec<LockGuard<'_>> {
            // Sort by TypeId to ensure consistent acquisition order
            let mut sorted: Vec<AccessInfo> = infos.to_vec();
            sorted.sort_by_key(|info| info.type_id);

            // Deduplicate by TypeId, preferring write over read
            sorted.dedup_by(|a, b| {
                if a.type_id == b.type_id {
                    b.is_write = b.is_write || a.is_write;
                    true
                } else {
                    false
                }
            });

            let mut guards = Vec::with_capacity(sorted.len());
            for info in &sorted {
                if let Some(lock) = self.locks.get(&info.type_id) {
                    if info.is_write {
                        guards.push(LockGuard::Write(lock.write().unwrap()));
                    } else {
                        guards.push(LockGuard::Read(lock.read().unwrap()));
                    }
                }
                // Skip types not in the map (unregistered Optional types)
            }
            guards
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use inner::WorldLocks;
