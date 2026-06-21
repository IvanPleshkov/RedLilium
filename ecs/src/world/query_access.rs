use std::any::TypeId;

use smallvec::SmallVec;

use crate::entity::Entity;
use crate::query::access::{AccessInfo, AccessKind};
use crate::query::{AddedFilter, ChangedFilter, ContainsChecker, RemovedFilter};
use crate::sparse_set::{LockGuard, Ref, RefMut};

use super::{World, WorldError};

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
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if `T` has never been registered or inserted.
    ///
    /// # Panics
    ///
    /// Panics if T is borrowed by any [`read`](World::read) or [`write`](World::write) call.
    pub fn write<T: 'static>(&self) -> Result<RefMut<'_, T>, WorldError> {
        let storage =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(RefMut::new(storage, self.entities(), self.tick))
    }

    /// Gets exclusive write access to all components of type T, returning `None`
    /// if the type has never been registered.
    ///
    /// Non-panicking variant of [`write`](World::write). Used by `OptionalWrite<T>`.
    pub fn try_write<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new(storage, self.entities(), self.tick))
    }

    // ---- ReadAll access (includes static entities) ----

    /// Gets shared read access including static entities.
    ///
    /// Like [`read`](World::read), but only excludes disabled entities —
    /// static entities are included. Use this in systems that need to
    /// observe all active entities (e.g., rendering, physics broadphase).
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
            Entity::DISABLED,
        ))
    }

    /// Gets shared read access including static entities, returning `None`
    /// if the type has never been registered.
    pub fn try_read_all<T: 'static>(&self) -> Option<Ref<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(Ref::new_with_mask(
            storage,
            self.entities(),
            Entity::DISABLED,
        ))
    }

    // ---- WriteAll access (includes static and editor entities) ----

    /// Gets exclusive write access including static and editor entities.
    ///
    /// Like [`write`](World::write), but only excludes disabled entities —
    /// both static and editor entities are included.
    pub fn write_all<T: 'static>(&self) -> Result<RefMut<'_, T>, WorldError> {
        let storage =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(RefMut::new_with_mask(
            storage,
            self.entities(),
            Entity::DISABLED,
            self.tick,
        ))
    }

    /// Gets exclusive write access including static and editor entities,
    /// returning `None` if the type has never been registered.
    pub fn try_write_all<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new_with_mask(
            storage,
            self.entities(),
            Entity::DISABLED,
            self.tick,
        ))
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
    pub(crate) fn write_unlocked<T: 'static>(&self) -> Result<RefMut<'_, T>, WorldError> {
        let lock =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(RefMut::new_unlocked(
            lock.data_ptr(),
            self.entities(),
            self.tick,
        ))
    }

    /// Gets optional shared read access without acquiring a lock.
    pub(crate) fn try_read_unlocked<T: 'static>(&self) -> Option<Ref<'_, T>> {
        let lock = self.components.get(&TypeId::of::<T>())?;
        let storage = unsafe { &*lock.data_ptr() };
        Some(Ref::new_unlocked(storage, self.entities()))
    }

    /// Gets optional exclusive write access without acquiring a lock.
    pub(crate) fn try_write_unlocked<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        let lock = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new_unlocked(
            lock.data_ptr(),
            self.entities(),
            self.tick,
        ))
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
            Entity::DISABLED,
        ))
    }

    /// Gets exclusive write access including static and editor entities,
    /// without acquiring a lock.
    pub(crate) fn write_all_unlocked<T: 'static>(&self) -> Result<RefMut<'_, T>, WorldError> {
        let lock =
            self.components
                .get(&TypeId::of::<T>())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        Ok(RefMut::new_unlocked_with_mask(
            lock.data_ptr(),
            self.entities(),
            Entity::DISABLED,
            self.tick,
        ))
    }

    /// Acquires component locks in TypeId-sorted order.
    ///
    /// Used by `LockRequest` during system execution. Sorted acquisition
    /// prevents deadlocks when multiple systems run concurrently.
    ///
    /// Resources are NOT included — they lock themselves via their own
    /// `Arc<RwLock<T>>` when accessed.
    pub(crate) fn acquire_sorted(&self, infos: &[AccessInfo]) -> SmallVec<[LockGuard<'_>; 8]> {
        // Reject aliasing-unsafe access sets (e.g. `(Write<T>, Write<T>)`)
        // before any data is fetched unlocked, otherwise the per-element
        // fetch would hand out two references to the same storage (UB).
        crate::query::access::validate_no_aliasing_conflict(infos);

        let sorted = crate::query::access::normalize_access_infos(infos);

        // Acquire every lock up-front in the normalized (TypeId-then-kind) order.
        // Because the order is global and consistent across all systems, two
        // systems that touch the same set of components/resources can never take
        // them in opposite orders, so there is no lock-ordering deadlock — and
        // resources block here instead of panicking on contention via `try_*`.
        sorted
            .iter()
            .filter_map(|info| match info.kind {
                AccessKind::Component | AccessKind::ComponentFilter => {
                    let lock = self.components.get(&info.type_id)?;
                    Some(if info.is_write {
                        LockGuard::Write(lock.write())
                    } else {
                        LockGuard::Read(lock.read())
                    })
                }
                AccessKind::Resource => {
                    if info.is_write {
                        self.resource_write_guard_dyn(info.type_id)
                            .map(LockGuard::ResourceWrite)
                    } else {
                        self.resource_read_guard_dyn(info.type_id)
                            .map(LockGuard::ResourceRead)
                    }
                }
                // Main-thread resources are single-threaded (no lock); pure
                // filter markers borrow no storage.
                AccessKind::MainThreadResource | AccessKind::Filter => None,
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
