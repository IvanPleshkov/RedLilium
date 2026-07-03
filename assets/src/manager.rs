//! The shared manager building blocks.
//!
//! Every asset manager is the **single source of truth** for its resident
//! `Arc`s and the single requester towards the [`AssetProcessor`] (which does
//! not dedup by design). They all share the same skeleton:
//!
//! - [`ResidentCache`] — the published resident `Arc`s + the failure latch +
//!   the generation counter (bumped on every change; `Arc` pointer identity is
//!   the version, the generation is the cheap "anything changed?" gate).
//! - [`AssetManager`] — a complete manager for the common case: a guid-keyed,
//!   dependency-free loader (`request → poll → publish`, failure latching,
//!   hot-reload invalidation). Managers with dependencies or multi-phase
//!   resolution embed a [`ResidentCache`] (and often an [`AssetManager`] for
//!   their data phase) and keep their drive logic explicit.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use crate::db::AssetDb;
use crate::handle::AssetHandle;
use crate::loader::AssetLoader;
use crate::processor::AssetProcessor;
use crate::source::Guid;

// ---------------------------------------------------------------------------
// ResidentCache
// ---------------------------------------------------------------------------

/// The resident half of an asset manager: published `Arc`s, the failure latch
/// (so a broken asset isn't re-requested every frame by demand-driven sync),
/// and the generation counter.
pub struct ResidentCache<K, T> {
    resident: HashMap<K, Arc<T>>,
    failed: HashSet<K>,
    generation: u64,
}

impl<K: Eq + Hash, T> Default for ResidentCache<K, T> {
    fn default() -> Self {
        Self {
            resident: HashMap::new(),
            failed: HashSet::new(),
            generation: 0,
        }
    }
}

impl<K: Eq + Hash, T> ResidentCache<K, T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// The resident value for `key`, if published.
    pub fn get(&self, key: &K) -> Option<&Arc<T>> {
        self.resident.get(key)
    }

    /// Whether `key` is latched as failed.
    pub fn is_failed(&self, key: &K) -> bool {
        self.failed.contains(key)
    }

    /// Publish (or republish — hot reload) the resident value for `key`.
    pub fn publish(&mut self, key: K, value: Arc<T>) {
        self.resident.insert(key, value);
        self.generation += 1;
    }

    /// Latch `key` as failed (cleared by [`invalidate`](Self::invalidate)).
    pub fn fail(&mut self, key: K) {
        self.failed.insert(key);
    }

    /// Drop the resident/failed state for `key` so it reloads (hot reload).
    /// Consumers keep serving the old `Arc` until the new one is published.
    ///
    /// Always bumps the generation — even when nothing was resident (e.g. the
    /// key was still loading or failed). Invalidation clears the manager's
    /// in-flight expectations, and consumers gate their demand-scan on the
    /// generation: without the bump, an invalidate landing mid-load would
    /// never be re-requested (gated scans skip unchanged components).
    pub fn invalidate(&mut self, key: &K) {
        self.resident.remove(key);
        self.failed.remove(key);
        self.generation += 1;
    }

    /// Iterate the resident entries (e.g. for pull-validation passes).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &Arc<T>)> {
        self.resident.iter()
    }

    /// Bumped on every change of the manager's state that consumers must react
    /// to: publish (load / reload) and invalidate.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

// ---------------------------------------------------------------------------
// AssetManager — the standard single-phase manager
// ---------------------------------------------------------------------------

/// A complete manager for the common loader shape: guid-keyed source
/// (`L::Source: From<Guid>`), no dependencies (`L::Deps = ()`), single async
/// pipeline. Handles request/poll/publish, failure latching, and hot-reload
/// invalidation. Used directly as an ECS resource (e.g. the shader manager) or
/// embedded as the data phase of a multi-phase manager.
pub struct AssetManager<L: AssetLoader<Deps = ()>>
where
    L::Source: From<Guid>,
{
    cache: ResidentCache<Guid, L::Asset>,
    pending: HashMap<Guid, AssetHandle<L::Asset>>,
}

impl<L: AssetLoader<Deps = ()>> Default for AssetManager<L>
where
    L::Source: From<Guid>,
{
    fn default() -> Self {
        Self {
            cache: ResidentCache::new(),
            pending: HashMap::new(),
        }
    }
}

impl<L: AssetLoader<Deps = ()>> AssetManager<L>
where
    L::Source: From<Guid>,
{
    pub fn new() -> Self {
        Self::default()
    }

    /// The resident asset for `guid`, requesting it once if not yet seen.
    /// `None` while loading (or after a failure) — call again next frame to
    /// advance. On a resident hit this is a plain map lookup.
    pub fn get_or_request(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        guid: Guid,
    ) -> Option<Arc<L::Asset>> {
        if let Some(asset) = self.cache.get(&guid) {
            return Some(asset.clone());
        }
        if self.cache.is_failed(&guid) {
            return None;
        }

        match self.pending.get(&guid).map(|h| h.get()) {
            // Not requested yet → request below.
            None => {}
            // Requested, still loading.
            Some(None) => return None,
            // Delivered: publish as resident.
            Some(Some(Ok(asset))) => {
                self.pending.remove(&guid);
                self.cache.publish(guid, asset.clone());
                return Some(asset);
            }
            // Failed: latch so we don't re-request every frame.
            Some(Some(Err(e))) => {
                log::warn!("{} {guid:?} failed to load: {e}", L::NAME);
                self.pending.remove(&guid);
                self.cache.fail(guid);
                return None;
            }
        }

        let handle = processor.request::<L>(db, L::Source::from(guid), ());
        self.pending.insert(guid, handle);
        None
    }

    /// The resident asset for `guid` if loaded — no request side effect.
    pub fn get(&self, guid: Guid) -> Option<&Arc<L::Asset>> {
        self.cache.get(&guid)
    }

    /// Whether `guid` is latched as failed.
    pub fn is_failed(&self, guid: Guid) -> bool {
        self.cache.is_failed(&guid)
    }

    /// Drop all state for `guid` so the next `get_or_request` reloads it (hot
    /// reload). Consumers keep serving the old `Arc` until the new one lands,
    /// then re-resolve by pointer identity.
    pub fn invalidate(&mut self, guid: Guid) {
        self.cache.invalidate(&guid);
        self.pending.remove(&guid);
    }

    /// Bumped whenever the resident set changes (load / reload / invalidate).
    pub fn generation(&self) -> u64 {
        self.cache.generation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Invalidation must bump the generation even when nothing was resident
    /// (mid-load / failed): consumers gate their demand-scan on the generation,
    /// and an unbumped invalidate would wedge the reload forever (nobody would
    /// re-request). Regression test for the rapid-edit hot-reload wedge.
    #[test]
    fn invalidate_always_bumps_generation() {
        let mut cache: ResidentCache<u32, String> = ResidentCache::new();

        // Not resident at all (e.g. a load is still in flight).
        let g0 = cache.generation();
        cache.invalidate(&1);
        assert!(cache.generation() > g0, "mid-load invalidate must bump");

        // Failed (latched).
        cache.fail(2);
        let g1 = cache.generation();
        cache.invalidate(&2);
        assert!(cache.generation() > g1, "failed-latch invalidate must bump");
        assert!(!cache.is_failed(&2), "invalidate clears the failure latch");

        // Resident.
        cache.publish(3, Arc::new("x".into()));
        let g2 = cache.generation();
        cache.invalidate(&3);
        assert!(cache.generation() > g2, "resident invalidate must bump");
        assert!(cache.get(&3).is_none());
    }
}
