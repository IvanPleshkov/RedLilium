use std::any::TypeId;

use smallvec::SmallVec;

use crate::entity::Entity;
use crate::query::access::{AccessInfo, AccessKind};
use crate::query::{AddedFilter, ChangedFilter, ContainsChecker, RemovedFilter};
use crate::sparse_set::{LockGuard, Ref, RefMut};

use super::{World, WorldError};

/// How long a single lock acquisition may block before it is declared a
/// deadlock and panics (#17). Frame-driven systems hold locks for
/// microseconds-to-milliseconds; ten seconds of blocking means a
/// cross-system lock-order (ABBA) deadlock, which would otherwise hang the
/// process silently. (A debugger pause can trip this — resume-and-rerun.)
pub(crate) const LOCK_ACQUIRE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Try-acquire loop with a watchdog deadline (#17).
///
/// The fast path takes no clock reading: `Instant::now` is touched only
/// after a failed try. On wasm (no std clock, single-threaded) contention
/// cannot occur, so the slow path is unreachable there.
fn acquire_watched<G>(
    info: &AccessInfo,
    timeout: std::time::Duration,
    mut try_acquire: impl FnMut() -> Option<G>,
) -> G {
    if let Some(guard) = try_acquire() {
        return guard;
    }
    let deadline = std::time::Instant::now() + timeout;
    loop {
        std::thread::sleep(std::time::Duration::from_micros(100));
        if let Some(guard) = try_acquire() {
            return guard;
        }
        if std::time::Instant::now() >= deadline {
            let mode = if info.is_write { "write" } else { "read" };
            panic!(
                "ECS lock acquisition timed out after {timeout:?}: {mode} lock on {} `{}` \
                 is still held elsewhere — likely a cross-system lock-order (ABBA) deadlock \
                 from nested ctx.lock calls (#17). Combine the accesses into a single \
                 lock/query call, or order the conflicting systems explicitly.",
                info.kind.storage_class().noun(),
                info.type_name,
            );
        }
    }
}

impl World {
    // ---- Query access (runtime borrow-checked, take &self) ----

    /// Gets shared read access to all components of type T.
    ///
    /// Returns a guard that dereferences to [`SparseSetInner<T>`](crate::SparseSetInner),
    /// allowing iteration and lookup.
    ///
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if `T` has never been registered or inserted.
    ///
    /// # Panics
    ///
    /// Panics if T is exclusively borrowed by a [`write`](World::write) call.
    pub fn read<T: 'static>(&self) -> Result<Ref<'_, T>, WorldError> {
        let storage =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(Ref::new(storage, self.entities()))
    }

    /// Gets shared read access to all components of type T, returning `None`
    /// if the type has never been registered.
    ///
    /// Non-panicking variant of [`read`](World::read). Used by `OptionalRead<T>`.
    pub fn try_read<T: 'static>(&self) -> Option<Ref<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(Ref::new(storage, self.entities()))
    }

    /// Gets exclusive write access to all components of type T.
    ///
    /// Returns a guard that dereferences to [`SparseSetInner<T>`](crate::SparseSetInner),
    /// allowing iteration, lookup, and mutation.
    ///
    /// Takes `&mut self`: unlocked readers ([`get`](World::get) and the bare
    /// filter constructors) hand out references with no lock guard, so every
    /// public mutation path must be exclusive for the borrow checker to keep
    /// them apart. To hold write access to several storages at once, use
    /// [`query`](World::query).
    ///
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if `T` has never been registered or inserted.
    pub fn write<T: 'static>(&mut self) -> Result<RefMut<'_, T>, WorldError> {
        self.write_storage()
    }

    /// Gets exclusive write access to all components of type T, returning `None`
    /// if the type has never been registered.
    ///
    /// Non-panicking variant of [`write`](World::write).
    pub fn try_write<T: 'static>(&mut self) -> Option<RefMut<'_, T>> {
        self.try_write_storage()
    }

    /// `&self` counterpart of [`write`](World::write) for crate internals.
    ///
    /// The returned guard holds the storage's write lock, which makes it safe
    /// against other *locking* accessors from any thread — but not against
    /// the unlocked readers ([`get`](World::get), bare filters), which is why
    /// the public API requires `&mut self`. Callers here are the locking
    /// `fetch` paths and in-crate systems, which never use unlocked reads
    /// concurrently.
    ///
    /// # Panics
    ///
    /// Panics if T is borrowed by any read or write guard.
    pub(crate) fn write_storage<T: 'static>(&self) -> Result<RefMut<'_, T>, WorldError> {
        self.write_storage_at(self.current_tick())
    }

    /// Like [`write_storage`](World::write_storage) but stamps writes with an
    /// explicit tick (a system run's `this_run`).
    pub(crate) fn write_storage_at<T: 'static>(
        &self,
        tick: u64,
    ) -> Result<RefMut<'_, T>, WorldError> {
        let storage =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(RefMut::new(storage, self.entities(), tick))
    }

    /// `&self` counterpart of [`try_write`](World::try_write) for crate
    /// internals (see [`write_storage`](World::write_storage)). Used by
    /// `OptionalWrite<T>`.
    pub(crate) fn try_write_storage<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        self.try_write_storage_at(self.current_tick())
    }

    /// Like [`try_write_storage`](World::try_write_storage) with an explicit
    /// write-stamp tick.
    pub(crate) fn try_write_storage_at<T: 'static>(&self, tick: u64) -> Option<RefMut<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new(storage, self.entities(), tick))
    }

    // ---- ReadAll access (includes static entities) ----

    /// Gets shared read access including static entities.
    ///
    /// Like [`read`](World::read), but only excludes disabled and hidden-in-play entities —
    /// static and editor entities are included. Use this in systems that need to
    /// observe engine infrastructure (e.g., transforms, asset loading, hotreload).
    pub fn read_all<T: 'static>(&self) -> Result<Ref<'_, T>, WorldError> {
        let storage =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(Ref::new_with_mask(
            storage,
            self.entities(),
            Entity::INFRASTRUCTURE_QUERY_EXCLUDE_MASK,
        ))
    }

    /// Gets shared read access including static entities, returning `None`
    /// if the type has never been registered.
    pub fn try_read_all<T: 'static>(&self) -> Option<Ref<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(Ref::new_with_mask(
            storage,
            self.entities(),
            Entity::INFRASTRUCTURE_QUERY_EXCLUDE_MASK,
        ))
    }

    // ---- WriteAll access (includes static and editor entities) ----

    /// Gets exclusive write access including static and editor entities.
    ///
    /// Like [`write`](World::write), but only excludes disabled entities —
    /// both static and editor entities are included. Takes `&mut self` for
    /// the same reason as [`write`](World::write).
    pub fn write_all<T: 'static>(&mut self) -> Result<RefMut<'_, T>, WorldError> {
        self.write_all_storage()
    }

    /// Gets exclusive write access including static and editor entities,
    /// returning `None` if the type has never been registered.
    pub fn try_write_all<T: 'static>(&mut self) -> Option<RefMut<'_, T>> {
        self.try_write_all_storage()
    }

    /// `&self` counterpart of [`write_all`](World::write_all) for crate
    /// internals (see [`write_storage`](World::write_storage)).
    pub(crate) fn write_all_storage<T: 'static>(&self) -> Result<RefMut<'_, T>, WorldError> {
        self.write_all_storage_at(self.current_tick())
    }

    /// Like [`write_all_storage`](World::write_all_storage) with an explicit
    /// write-stamp tick.
    pub(crate) fn write_all_storage_at<T: 'static>(
        &self,
        tick: u64,
    ) -> Result<RefMut<'_, T>, WorldError> {
        let storage =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(RefMut::new_with_mask(
            storage,
            self.entities(),
            Entity::INFRASTRUCTURE_QUERY_EXCLUDE_MASK,
            tick,
        ))
    }

    /// `&self` counterpart of [`try_write_all`](World::try_write_all) for
    /// crate internals (see [`write_storage`](World::write_storage)).
    pub(crate) fn try_write_all_storage<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new_with_mask(
            storage,
            self.entities(),
            Entity::INFRASTRUCTURE_QUERY_EXCLUDE_MASK,
            self.current_tick(),
        ))
    }

    // ---- Multi-storage access ----

    /// Acquires locks for the given access set and returns a guard holding
    /// the locked data — the world-owner counterpart of
    /// [`SystemContext::query`](crate::SystemContext::query).
    ///
    /// Use this to work with several storages at once (including multiple
    /// writes), which the single-storage `&mut self` accessors like
    /// [`write`](World::write) cannot express:
    ///
    /// ```ignore
    /// let mut q = world.query::<(Write<Position>, Read<Velocity>)>();
    /// let (positions, velocities) = q.items_mut();
    /// ```
    ///
    /// Takes `&mut self` so the guard cannot coexist with the unlocked
    /// readers ([`get`](World::get) and the bare filter constructors).
    ///
    /// # Panics
    ///
    /// Panics on an aliasing-unsafe access set (e.g. `(Write<T>, Read<T>)`)
    /// and if the set contains `MainThreadRes`/`MainThreadResMut`.
    pub fn query<A: crate::query::AccessSet>(&mut self) -> crate::query::QueryGuard<'_, A> {
        if A::needs_main_thread() {
            panic!("World::query does not support main-thread resources");
        }
        let infos = A::access_infos();
        let ticks = crate::query::FetchTicks::frame(self);
        let guards = self.acquire_sorted(&infos);
        let items = A::fetch_unlocked(self, ticks);
        crate::query::QueryGuard::new(guards, items)
    }

    // ---- Unlocked access (for use when locks are held externally) ----

    /// Gets shared read access without acquiring a lock.
    ///
    /// The caller must ensure the read lock is already held externally.
    pub(crate) fn read_unlocked<T: 'static>(&self) -> Result<Ref<'_, T>, WorldError> {
        let lock =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        let storage = unsafe { &*lock.data_ptr() };
        Ok(Ref::new_unlocked(storage, self.entities()))
    }

    /// Gets exclusive write access without acquiring a lock.
    ///
    /// The caller must ensure the write lock is already held externally.
    pub(crate) fn write_unlocked<T: 'static>(
        &self,
        tick: u64,
    ) -> Result<RefMut<'_, T>, WorldError> {
        let lock =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(RefMut::new_unlocked(lock.data_ptr(), self.entities(), tick))
    }

    /// Gets optional shared read access without acquiring a lock.
    pub(crate) fn try_read_unlocked<T: 'static>(&self) -> Option<Ref<'_, T>> {
        let lock = self.components.get(&TypeId::of::<T>())?;
        let storage = unsafe { &*lock.data_ptr() };
        Some(Ref::new_unlocked(storage, self.entities()))
    }

    /// Gets optional exclusive write access without acquiring a lock.
    pub(crate) fn try_write_unlocked<T: 'static>(&self, tick: u64) -> Option<RefMut<'_, T>> {
        let lock = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new_unlocked(lock.data_ptr(), self.entities(), tick))
    }

    /// Gets shared read access including static entities, without acquiring a lock.
    pub(crate) fn read_all_unlocked<T: 'static>(&self) -> Result<Ref<'_, T>, WorldError> {
        let lock =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        let storage = unsafe { &*lock.data_ptr() };
        Ok(Ref::new_unlocked_with_mask(
            storage,
            self.entities(),
            Entity::INFRASTRUCTURE_QUERY_EXCLUDE_MASK,
        ))
    }

    /// Gets exclusive write access including static and editor entities,
    /// without acquiring a lock.
    pub(crate) fn write_all_unlocked<T: 'static>(
        &self,
        tick: u64,
    ) -> Result<RefMut<'_, T>, WorldError> {
        let lock =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(RefMut::new_unlocked_with_mask(
            lock.data_ptr(),
            self.entities(),
            Entity::INFRASTRUCTURE_QUERY_EXCLUDE_MASK,
            tick,
        ))
    }

    /// Acquires component AND resource locks in storage-sorted order.
    ///
    /// Used by `LockRequest` during system execution. Sorted acquisition
    /// prevents deadlocks *within one call*: because the order is global and
    /// consistent across all systems, two systems whose accesses are declared
    /// in a single `lock`/`query` call can never take them in opposite
    /// orders. Nested `ctx.lock` calls break that guarantee (each call sorts
    /// only its own set), which is why acquisition runs under a watchdog: a
    /// lock still unavailable after [`LOCK_ACQUIRE_TIMEOUT`] is treated as a
    /// cross-system lock-order (ABBA) deadlock and panics with the storage
    /// name instead of hanging silently (#17).
    pub(crate) fn acquire_sorted(&self, infos: &[AccessInfo]) -> SmallVec<[LockGuard<'_>; 8]> {
        self.acquire_sorted_with_timeout(infos, LOCK_ACQUIRE_TIMEOUT)
    }

    /// [`acquire_sorted`](Self::acquire_sorted) with an explicit watchdog
    /// timeout (tests use a short one).
    pub(crate) fn acquire_sorted_with_timeout(
        &self,
        infos: &[AccessInfo],
        timeout: std::time::Duration,
    ) -> SmallVec<[LockGuard<'_>; 8]> {
        // Reject aliasing-unsafe access sets (e.g. `(Write<T>, Write<T>)`)
        // before any data is fetched unlocked, otherwise the per-element
        // fetch would hand out two references to the same storage (UB).
        crate::query::access::validate_no_aliasing_conflict(infos);

        let sorted = crate::query::access::normalize_access_infos(infos);

        sorted
            .iter()
            .filter_map(|info| match info.kind {
                AccessKind::Component | AccessKind::ComponentFilter => {
                    let lock = self.components.get(&info.type_id)?;
                    Some(if info.is_write {
                        LockGuard::Write(acquire_watched(info, timeout, || lock.try_write()))
                    } else {
                        LockGuard::Read(acquire_watched(info, timeout, || lock.try_read()))
                    })
                }
                AccessKind::Resource => {
                    // Unregistered resource → no lock to take (the fetch
                    // will panic with its own message).
                    if !self.resources.contains_dyn(info.type_id) {
                        return None;
                    }
                    Some(if info.is_write {
                        LockGuard::ResourceWrite(acquire_watched(info, timeout, || {
                            self.resources.try_write_guard_dyn(info.type_id).flatten()
                        }))
                    } else {
                        LockGuard::ResourceRead(acquire_watched(info, timeout, || {
                            self.resources.try_read_guard_dyn(info.type_id).flatten()
                        }))
                    })
                }
                // Main-thread resources are single-threaded (no lock); pure
                // filter markers borrow no storage. RawWorld is a diagnostic
                // marker (#54), never part of a lock request.
                AccessKind::MainThreadResource | AccessKind::Filter | AccessKind::RawWorld => None,
            })
            .collect()
    }

    /// Returns the human-readable type name for a component TypeId, if registered.
    pub(crate) fn component_type_name(&self, type_id: TypeId) -> Option<&'static str> {
        let lock = self.components.get(&type_id)?;
        let storage = unsafe { &*lock.data_ptr() };
        Some(storage.type_name())
    }

    /// Returns whether a component type has been registered.
    pub fn is_component_registered<T: 'static>(&self) -> bool {
        self.components.contains_key(&TypeId::of::<T>())
    }

    /// Returns the TypeIds of all registered component types.
    pub fn component_type_ids(&self) -> impl Iterator<Item = TypeId> + '_ {
        self.components.keys().copied()
    }

    /// Returns the [`QualifiedTypeId`](crate::QualifiedTypeId) of every
    /// registered component type — the source-qualified parallel to
    /// [`component_type_ids`](World::component_type_ids), consulted by #45 and
    /// the downcast guard. The source is resolved from the registration map
    /// (`type_sources`, `HOST` in single-process builds), which every
    /// `register_component` path populates — so no storage lock (and no
    /// unlocked read of component data) is needed.
    pub fn component_qualified_type_ids(
        &self,
    ) -> impl Iterator<Item = crate::QualifiedTypeId> + '_ {
        self.components.keys().filter_map(|type_id| {
            self.resolved_source_by_id(*type_id)
                .map(|source| crate::QualifiedTypeId::new(*type_id, source))
        })
    }

    // ---- Filters ----

    /// Creates a `With<T>` filter that checks for component presence.
    ///
    /// Returns a [`ContainsChecker`] that does not borrow component data.
    /// If T has never been registered, the filter matches nothing.
    pub fn with<T: 'static>(&self) -> ContainsChecker<'_> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .map(|lock| unsafe { &*lock.data_ptr() });
        ContainsChecker::with(storage)
    }

    /// Creates a `Without<T>` filter that checks for component absence.
    ///
    /// Returns a [`ContainsChecker`] that does not borrow component data.
    /// If T has never been registered, the filter matches everything.
    pub fn without<T: 'static>(&self) -> ContainsChecker<'_> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .map(|lock| unsafe { &*lock.data_ptr() });
        ContainsChecker::without(storage)
    }

    /// Creates a filter matching entities whose component T was changed
    /// since (strictly after) `since_tick`.
    ///
    /// Does not borrow component data. If T has never been registered,
    /// the filter matches nothing.
    pub fn changed<T: 'static>(&self, since_tick: u64) -> ChangedFilter<'_> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .map(|lock| unsafe { &*lock.data_ptr() });
        ChangedFilter::new(storage, since_tick)
    }

    /// Creates a filter matching entities whose component T was added
    /// since (strictly after) `since_tick`.
    ///
    /// Does not borrow component data. If T has never been registered,
    /// the filter matches nothing.
    pub fn added<T: 'static>(&self, since_tick: u64) -> AddedFilter<'_> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .map(|lock| unsafe { &*lock.data_ptr() });
        AddedFilter::new(storage, since_tick)
    }

    /// Creates a filter matching entities whose component T was removed
    /// since (strictly after) `since_tick`.
    ///
    /// Does not borrow component data. If T has never been registered,
    /// the filter matches nothing.
    ///
    /// Removal records are accumulated across frames. Call
    /// [`clear_removed_tracking`](World::clear_removed_tracking) to reset them.
    pub fn removed<T: 'static>(&self, since_tick: u64) -> RemovedFilter<'_> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .map(|lock| unsafe { &*lock.data_ptr() });
        RemovedFilter::new(storage, since_tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Pos(#[allow(dead_code)] f32);

    /// The #17 watchdog: a lock held by another thread past the timeout is a
    /// panic naming the storage, not a silent hang. (Cross-system ABBA
    /// deadlocks reduce to exactly this: some system never releases the lock
    /// this one is waiting for.)
    #[test]
    #[should_panic(expected = "ECS lock acquisition timed out")]
    fn acquisition_watchdog_panics_instead_of_hanging() {
        let mut world = World::new();
        world.register_component::<Pos>();

        let infos = [AccessInfo::component::<Pos>(true)];
        std::thread::scope(|scope| {
            let world = &world;
            scope.spawn(move || {
                let _guard = world.acquire_sorted(&infos);
                std::thread::sleep(std::time::Duration::from_millis(500));
            });
            // Give the holder thread time to take the write lock, then time out.
            std::thread::sleep(std::time::Duration::from_millis(100));
            let _hung =
                world.acquire_sorted_with_timeout(&infos, std::time::Duration::from_millis(50));
        });
    }

    /// The watchdog's retry loop must still succeed when the contended lock
    /// is released before the deadline.
    #[test]
    fn acquisition_watchdog_recovers_after_release() {
        let mut world = World::new();
        world.register_component::<Pos>();

        let infos = [AccessInfo::component::<Pos>(true)];
        std::thread::scope(|scope| {
            let world = &world;
            scope.spawn(move || {
                let _guard = world.acquire_sorted(&infos);
                std::thread::sleep(std::time::Duration::from_millis(100));
            });
            std::thread::sleep(std::time::Duration::from_millis(30));
            let guards =
                world.acquire_sorted_with_timeout(&infos, std::time::Duration::from_secs(5));
            assert_eq!(guards.len(), 1);
        });
    }
}
