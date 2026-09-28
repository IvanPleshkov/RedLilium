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
use std::sync::{Arc, Weak};

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
    resident: HashMap<K, Cached<T>>,
    failed: HashSet<K>,
    generation: u64,
}

enum Cached<T> {
    Retained(Arc<T>),
    Released(Weak<T>),
}

impl<T> Cached<T> {
    fn get(&self) -> Option<Arc<T>> {
        match self {
            Self::Retained(value) => Some(value.clone()),
            Self::Released(value) => value.upgrade(),
        }
    }

    fn is_unused(&self) -> bool {
        match self {
            Self::Retained(_) => false,
            Self::Released(value) => value.strong_count() == 0,
        }
    }
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
    pub fn get(&self, key: &K) -> Option<Arc<T>> {
        self.resident.get(key).and_then(Cached::get)
    }

    /// Stop retaining this version, but keep its identity while another owner
    /// holds it. Does not invalidate it or clear a failure latch. Unlike
    /// `invalidate`, a later lookup can still upgrade the weak reference.
    pub fn release(&mut self, key: &K) {
        if let Some(entry) = self.resident.get_mut(key)
            && let Cached::Retained(value) = entry
        {
            *entry = Cached::Released(Arc::downgrade(value));
            self.generation += 1;
        }
    }

    /// Remove cache-only strong entries and expired weak entries. Returns the
    /// removed keys so composite managers can retire associated metadata.
    /// Live released entries and failure latches remain. Run explicitly at a
    /// scene/preview boundary; this is not automatic eviction on every frame.
    pub fn collect_unused(&mut self) -> Vec<K>
    where
        K: Clone,
    {
        // Equal assets may be published under multiple keys. Count all strong
        // owners inside this cache so aliases do not keep each other alive.
        let mut retained = HashMap::new();
        for entry in self.resident.values() {
            if let Cached::Retained(value) = entry {
                *retained.entry(Arc::as_ptr(value)).or_insert(0usize) += 1;
            }
        }
        for entry in self.resident.values_mut() {
            if let Cached::Retained(value) = entry
                && Arc::strong_count(value) == retained[&Arc::as_ptr(value)]
            {
                *retained.get_mut(&Arc::as_ptr(value)).unwrap() -= 1;
                *entry = Cached::Released(Arc::downgrade(value));
            }
        }
        let mut removed = Vec::new();
        self.resident.retain(|key, value| {
            if value.is_unused() {
                removed.push(key.clone());
                false
            } else {
                true
            }
        });
        if !removed.is_empty() {
            self.generation += 1;
        }
        removed
    }

    /// Whether `key` is latched as failed.
    pub fn is_failed(&self, key: &K) -> bool {
        self.failed.contains(key)
    }

    /// Publish (or republish — hot reload) the resident value for `key`.
    pub fn publish(&mut self, key: K, value: Arc<T>) {
        self.failed.remove(&key);
        self.resident.insert(key, Cached::Retained(value));
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
    pub fn iter(&self) -> impl Iterator<Item = (&K, Arc<T>)> {
        self.resident
            .iter()
            .filter_map(|(key, value)| value.get().map(|value| (key, value)))
    }

    /// Bump the generation without touching resident state: consumers that
    /// gate their demand-scan on the generation will rescan and resolve fresh
    /// references against the *existing* residents. For flows that introduce
    /// new reference holders while the manager is in steady state (scene
    /// instantiation) — unlike [`invalidate`](Self::invalidate), nothing
    /// reloads.
    pub fn bump_generation(&mut self) {
        self.generation += 1;
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
    /// advance. A resident hit clones its retained Arc or upgrades its Weak.
    pub fn get_or_request(
        &mut self,
        processor: &mut AssetProcessor,
        db: &AssetDb,
        guid: Guid,
    ) -> Option<Arc<L::Asset>> {
        if let Some(asset) = self.cache.get(&guid) {
            return Some(asset);
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

    /// Seed a resident asset directly, bypassing the load pipeline — the
    /// programmatic-publish integration point (virtual assets, mirroring
    /// `MeshManager::insert_external` / `TextureManager::publish_virtual`).
    /// Replaces any pending load or resident entry and bumps the generation,
    /// so consumers re-resolve to the new value.
    pub fn publish(&mut self, guid: Guid, asset: Arc<L::Asset>) {
        self.pending.remove(&guid);
        self.cache.publish(guid, asset);
    }

    /// The resident asset for `guid` if loaded — no request side effect.
    pub fn get(&self, guid: Guid) -> Option<Arc<L::Asset>> {
        self.cache.get(&guid)
    }

    /// Release cache ownership and cancel this manager's pending request.
    /// Existing consumers keep the version alive and subsequent requests reuse
    /// it through a weak reference. Failure latches require invalidation.
    pub fn release(&mut self, guid: Guid) {
        self.cache.release(&guid);
        if self.pending.remove(&guid).is_some() {
            self.cache.bump_generation();
        }
    }

    /// Collect unreferenced assets and dead weak entries; return entry count.
    /// Pending requests and failure latches are not discarded.
    pub fn collect_unused(&mut self) -> usize {
        self.cache.collect_unused().len()
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

    #[test]
    fn released_assets_reuse_identity_and_dead_weak_entries_are_collected() {
        let mut cache = ResidentCache::new();
        let value = Arc::new(vec![1u8; 1024]);
        cache.publish(1, value.clone());
        cache.release(&1);
        assert_eq!(Arc::strong_count(&value), 1);
        assert!(Arc::ptr_eq(&cache.get(&1).unwrap(), &value));
        assert!(cache.collect_unused().is_empty());
        let second_owner = cache.get(&1).unwrap();
        drop(value);
        assert!(Arc::ptr_eq(&cache.get(&1).unwrap(), &second_owner));
        drop(second_owner);
        assert!(cache.get(&1).is_none());
        assert_eq!(
            cache.resident.len(),
            1,
            "dead weak allocation remains tracked"
        );
        assert_eq!(cache.collect_unused(), vec![1]);
        assert!(
            cache.resident.is_empty(),
            "including the weak allocation owner"
        );
        assert!(cache.collect_unused().is_empty());
    }

    #[test]
    fn collect_unused_handles_aliases_without_evicting_external_owners() {
        let mut cache = ResidentCache::new();
        let value = Arc::new(42);
        let weak = Arc::downgrade(&value);
        cache.publish(1, value.clone());
        cache.publish(2, value.clone());
        cache.publish(3, value.clone());
        cache.release(&3);
        assert!(cache.collect_unused().is_empty());
        drop(value);
        let mut removed = cache.collect_unused();
        removed.sort();
        assert_eq!(removed, vec![1, 2, 3]);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn invalidation_and_republication_never_resurrect_released_versions() {
        let mut cache = ResidentCache::new();
        let old = Arc::new(1);
        cache.publish(1, old.clone());
        cache.release(&1);
        cache.invalidate(&1);
        assert!(cache.get(&1).is_none());
        let new = Arc::new(2);
        cache.publish(1, new.clone());
        cache.release(&1);
        assert!(Arc::ptr_eq(&cache.get(&1).unwrap(), &new));
        assert!(!Arc::ptr_eq(&cache.get(&1).unwrap(), &old));
        cache.publish(1, old.clone());
        assert!(Arc::ptr_eq(&cache.get(&1).unwrap(), &old));
        cache.fail(2);
        cache.release(&2);
        cache.collect_unused();
        assert!(
            cache.is_failed(&2),
            "collection must not cause failure retry loops"
        );
    }

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
