use std::any::Any;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Typed sparse set storing components of type T.
///
/// Uses a sparse array (entity index → dense index) and a dense array
/// (contiguous component data + entity mapping) for O(1) insert/remove/get
/// and cache-friendly iteration.
///
/// ## Change Detection
///
/// Each component tracks the tick when it was added and last changed.
/// Use [`insert_with_tick`](SparseSetInner::insert_with_tick) and
/// [`get_mut_tracked`](SparseSetInner::get_mut_tracked) to enable tracking.
/// Query with [`changed_since`](SparseSetInner::changed_since) and
/// [`added_since`](SparseSetInner::added_since).
pub struct SparseSetInner<T: 'static> {
    /// Sparse array: `entity_index -> dense_index`. `None` means the entity
    /// does not have this component.
    sparse: Vec<Option<u32>>,
    /// Dense array of component values (contiguous for iteration).
    dense: Vec<T>,
    /// Entity indices corresponding to each dense element.
    entities: Vec<u32>,
    /// Tick when each component was added (parallel to dense).
    ticks_added: Vec<u64>,
    /// Tick when each component was last changed (parallel to dense).
    ticks_changed: Vec<u64>,
}

impl<T: 'static> SparseSetInner<T> {
    /// Creates a new empty sparse set.
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
            entities: Vec::new(),
            ticks_added: Vec::new(),
            ticks_changed: Vec::new(),
        }
    }

    /// Inserts a component for the given entity index.
    /// If the entity already has this component, the value is replaced.
    /// Uses tick 0 for change tracking (untracked).
    pub fn insert(&mut self, entity_index: u32, value: T) {
        self.insert_with_tick(entity_index, value, 0);
    }

    /// Inserts a component with change tracking at the given tick.
    ///
    /// If the entity already has this component, the value is replaced
    /// and `ticks_changed` is updated. If it's a new insertion,
    /// both `ticks_added` and `ticks_changed` are set to `tick`.
    pub fn insert_with_tick(&mut self, entity_index: u32, value: T, tick: u64) {
        let idx = entity_index as usize;

        // Grow sparse array if needed
        if idx >= self.sparse.len() {
            self.sparse.resize(idx + 1, None);
        }

        if let Some(dense_idx) = self.sparse[idx] {
            // Replace existing value
            let di = dense_idx as usize;
            self.dense[di] = value;
            self.ticks_changed[di] = tick;
        } else {
            // Insert new value
            let dense_idx = self.dense.len() as u32;
            self.sparse[idx] = Some(dense_idx);
            self.dense.push(value);
            self.entities.push(entity_index);
            self.ticks_added.push(tick);
            self.ticks_changed.push(tick);
        }
    }

    /// Removes a component for the given entity index.
    /// Returns the removed value, or `None` if the entity did not have this component.
    pub fn remove(&mut self, entity_index: u32) -> Option<T> {
        let idx = entity_index as usize;
        if idx >= self.sparse.len() {
            return None;
        }

        let dense_idx = self.sparse[idx]?;
        self.sparse[idx] = None;

        let last_dense = self.dense.len() - 1;
        let dense_idx = dense_idx as usize;

        if dense_idx != last_dense {
            // Swap-remove: move last element into the removed slot
            let swapped_entity = self.entities[last_dense];
            self.sparse[swapped_entity as usize] = Some(dense_idx as u32);
            self.entities[dense_idx] = swapped_entity;
            self.ticks_added[dense_idx] = self.ticks_added[last_dense];
            self.ticks_changed[dense_idx] = self.ticks_changed[last_dense];
        }

        self.entities.pop();
        self.ticks_added.pop();
        self.ticks_changed.pop();
        Some(self.dense.swap_remove(dense_idx))
    }

    /// Returns a reference to the component for the given entity index.
    pub fn get(&self, entity_index: u32) -> Option<&T> {
        let idx = entity_index as usize;
        let dense_idx = *self.sparse.get(idx)?.as_ref()? as usize;
        Some(&self.dense[dense_idx])
    }

    /// Returns a mutable reference to the component for the given entity index.
    pub fn get_mut(&mut self, entity_index: u32) -> Option<&mut T> {
        let idx = entity_index as usize;
        let dense_idx = *self.sparse.get(idx)?.as_ref()? as usize;
        Some(&mut self.dense[dense_idx])
    }

    /// Returns whether the entity has this component.
    pub fn contains(&self, entity_index: u32) -> bool {
        let idx = entity_index as usize;
        idx < self.sparse.len() && self.sparse[idx].is_some()
    }

    /// Returns the number of components stored.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Returns whether this sparse set is empty.
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Reserves capacity for at least `additional` more components.
    ///
    /// Pre-grows dense arrays to avoid repeated allocations during
    /// batch insertions.
    pub fn reserve(&mut self, additional: usize) {
        self.dense.reserve(additional);
        self.entities.reserve(additional);
        self.ticks_added.reserve(additional);
        self.ticks_changed.reserve(additional);
    }

    /// Iterates over `(entity_index, &component)` pairs in dense order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.entities.iter().copied().zip(self.dense.iter())
    }

    /// Iterates over `(entity_index, &mut component)` pairs in dense order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        self.entities.iter().copied().zip(self.dense.iter_mut())
    }

    /// Returns a slice of entity indices in dense order.
    pub fn entities(&self) -> &[u32] {
        &self.entities
    }

    /// Returns a mutable pointer to the component for the given entity index.
    ///
    /// # Safety
    ///
    /// - `this` must be a valid, properly aligned pointer to an initialized
    ///   `SparseSetInner<T>`.
    /// - The caller must have exclusive access to the storage (e.g., write lock held).
    /// - The caller must ensure no other mutable reference to the same dense
    ///   slot exists.
    pub(crate) unsafe fn get_ptr_mut(this: *mut Self, entity_index: u32) -> Option<*mut T> {
        // SAFETY: caller guarantees `this` is valid and exclusively accessed.
        unsafe {
            let set = &mut *this;
            let idx = entity_index as usize;
            let dense_idx = *set.sparse.get(idx)?.as_ref()? as usize;
            Some(set.dense.as_mut_ptr().add(dense_idx))
        }
    }

    // ---- Change detection ----

    /// Returns a mutable reference and marks the component as changed at `tick`.
    pub fn get_mut_tracked(&mut self, entity_index: u32, tick: u64) -> Option<&mut T> {
        let idx = entity_index as usize;
        let dense_idx = *self.sparse.get(idx)?.as_ref()? as usize;
        self.ticks_changed[dense_idx] = tick;
        Some(&mut self.dense[dense_idx])
    }

    /// Iterates with mutation, marking all accessed components as changed at `tick`.
    pub fn iter_mut_tracked(&mut self, tick: u64) -> impl Iterator<Item = (u32, &mut T)> {
        // Mark all as changed
        for tc in self.ticks_changed.iter_mut() {
            *tc = tick;
        }
        self.entities.iter().copied().zip(self.dense.iter_mut())
    }

    /// Returns true if the component was changed since (strictly after) `since_tick`.
    pub fn changed_since(&self, entity_index: u32, since_tick: u64) -> bool {
        let idx = entity_index as usize;
        if let Some(Some(dense_idx)) = self.sparse.get(idx) {
            self.ticks_changed[*dense_idx as usize] > since_tick
        } else {
            false
        }
    }

    /// Returns true if the component was added since (strictly after) `since_tick`.
    pub fn added_since(&self, entity_index: u32, since_tick: u64) -> bool {
        let idx = entity_index as usize;
        if let Some(Some(dense_idx)) = self.sparse.get(idx) {
            self.ticks_added[*dense_idx as usize] > since_tick
        } else {
            false
        }
    }
}

impl<T: 'static> Default for SparseSetInner<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Type-erased operation function signatures
type RemoveFn = fn(&mut dyn Any, u32) -> bool;
type ContainsFn = fn(&dyn Any, u32) -> bool;
type ChangedSinceFn = fn(&dyn Any, u32, u64) -> bool;
type AddedSinceFn = fn(&dyn Any, u32, u64) -> bool;

/// A lock guard for either a read or write lock on a storage.
///
/// The guard is held purely for its RAII drop behavior (releasing the lock).
#[allow(dead_code)]
pub(crate) enum LockGuard<'a> {
    Read(RwLockReadGuard<'a, ()>),
    Write(RwLockWriteGuard<'a, ()>),
}

/// A type-erased sparse set that stores components of a single type.
///
/// Provides per-storage RwLock synchronization for thread-safe access.
/// Used internally by [`World`](crate::World).
pub(crate) struct ComponentStorage {
    inner: Box<dyn Any + Send + Sync>,
    /// Per-storage lock for thread-safe borrow management.
    lock: RwLock<()>,
    /// Human-readable type name for error messages.
    type_name: &'static str,
    /// Type-erased remove operation for despawn.
    remove_fn: RemoveFn,
    /// Type-erased contains check.
    contains_fn: ContainsFn,
    /// Type-erased changed_since check.
    changed_since_fn: ChangedSinceFn,
    /// Type-erased added_since check.
    added_since_fn: AddedSinceFn,
    /// Records of (entity_index, tick) for recently removed components.
    /// Cleared by [`World::clear_removed_tracking`](crate::World::clear_removed_tracking).
    removed_ticks: Vec<(u32, u64)>,
}

impl ComponentStorage {
    /// Creates a new component storage for type `T`.
    pub fn new<T: Send + Sync + 'static>() -> Self {
        Self {
            inner: Box::new(SparseSetInner::<T>::new()),
            lock: RwLock::new(()),
            type_name: std::any::type_name::<T>(),
            remove_fn: |any, entity_index| {
                let set = any.downcast_mut::<SparseSetInner<T>>().unwrap();
                set.remove(entity_index).is_some()
            },
            contains_fn: |any, entity_index| {
                let set = any.downcast_ref::<SparseSetInner<T>>().unwrap();
                set.contains(entity_index)
            },
            changed_since_fn: |any, entity_index, since_tick| {
                let set = any.downcast_ref::<SparseSetInner<T>>().unwrap();
                set.changed_since(entity_index, since_tick)
            },
            added_since_fn: |any, entity_index, since_tick| {
                let set = any.downcast_ref::<SparseSetInner<T>>().unwrap();
                set.added_since(entity_index, since_tick)
            },
            removed_ticks: Vec::new(),
        }
    }

    /// Downcasts to the typed sparse set.
    pub fn typed<T: 'static>(&self) -> &SparseSetInner<T> {
        self.inner.downcast_ref::<SparseSetInner<T>>().unwrap()
    }

    /// Downcasts to the typed sparse set (mutable).
    pub fn typed_mut<T: 'static>(&mut self) -> &mut SparseSetInner<T> {
        self.inner.downcast_mut::<SparseSetInner<T>>().unwrap()
    }

    /// Acquires a shared read lock. Panics immediately if a write lock is held.
    ///
    /// Uses `try_read` for instant conflict detection (panic, not deadlock).
    /// Used by the direct World API (`world.read()`).
    pub(crate) fn lock_read(&self) -> RwLockReadGuard<'_, ()> {
        self.lock.try_read().unwrap_or_else(|_| {
            panic!(
                "Cannot borrow `{}` immutably: already borrowed mutably",
                self.type_name
            )
        })
    }

    /// Acquires an exclusive write lock. Panics immediately if any lock is held.
    ///
    /// Uses `try_write` for instant conflict detection (panic, not deadlock).
    /// Used by the direct World API (`world.write()`).
    pub(crate) fn lock_write(&self) -> RwLockWriteGuard<'_, ()> {
        self.lock.try_write().unwrap_or_else(|_| {
            panic!(
                "Cannot borrow `{}` mutably: already borrowed",
                self.type_name
            )
        })
    }

    /// Returns the human-readable type name of the stored component.
    pub(crate) fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// Returns a reference to the underlying RwLock.
    ///
    /// Used by `acquire_sorted` for blocking lock acquisition in system execution.
    pub(crate) fn rw_lock(&self) -> &RwLock<()> {
        &self.lock
    }

    /// Removes a component by entity index (type-erased). Returns true if removed.
    pub fn remove_untyped(&mut self, entity_index: u32) -> bool {
        (self.remove_fn)(self.inner.as_mut(), entity_index)
    }

    /// Checks if the entity has this component (type-erased).
    pub fn contains_untyped(&self, entity_index: u32) -> bool {
        (self.contains_fn)(self.inner.as_ref(), entity_index)
    }

    /// Checks if the component was changed since `since_tick` (type-erased).
    pub fn changed_since_untyped(&self, entity_index: u32, since_tick: u64) -> bool {
        (self.changed_since_fn)(self.inner.as_ref(), entity_index, since_tick)
    }

    /// Checks if the component was added since `since_tick` (type-erased).
    pub fn added_since_untyped(&self, entity_index: u32, since_tick: u64) -> bool {
        (self.added_since_fn)(self.inner.as_ref(), entity_index, since_tick)
    }

    /// Records that a component was removed from the given entity at the given tick.
    pub fn record_removal(&mut self, entity_index: u32, tick: u64) {
        self.removed_ticks.push((entity_index, tick));
    }

    /// Checks if a component was removed from the entity since (strictly after) `since_tick`.
    pub fn removed_since_untyped(&self, entity_index: u32, since_tick: u64) -> bool {
        self.removed_ticks
            .iter()
            .any(|&(e, t)| e == entity_index && t > since_tick)
    }

    /// Returns an iterator over `(entity_index, tick)` pairs for all removals
    /// that happened since (strictly after) `since_tick`.
    pub fn removed_entities_since(&self, since_tick: u64) -> impl Iterator<Item = u32> + '_ {
        self.removed_ticks
            .iter()
            .filter(move |&&(_, t)| t > since_tick)
            .map(|&(e, _)| e)
    }

    /// Clears all removal tracking records.
    pub fn clear_removed(&mut self) {
        self.removed_ticks.clear();
    }
}

/// Shared read access to a component storage.
///
/// Automatically releases the lock when dropped.
/// Dereferences to [`SparseSetInner<T>`] for accessing component data.
pub struct Ref<'a, T: 'static> {
    inner: &'a SparseSetInner<T>,
    _guard: Option<RwLockReadGuard<'a, ()>>,
}

impl<'a, T: 'static> Ref<'a, T> {
    /// Creates a new shared borrow guard, acquiring the storage's read lock.
    pub(crate) fn new(storage: &'a ComponentStorage) -> Self {
        let guard = storage.lock_read();
        Self {
            inner: storage.typed::<T>(),
            _guard: Some(guard),
        }
    }

    /// Creates a shared borrow without acquiring a lock.
    ///
    /// The caller must ensure the lock is already held externally
    /// (e.g. via `acquire_sorted`).
    pub(crate) fn new_unlocked(storage: &'a ComponentStorage) -> Self {
        Self {
            inner: storage.typed::<T>(),
            _guard: None,
        }
    }

    /// Returns a reference to the underlying storage with the storage lifetime.
    ///
    /// Unlike `Deref` (which ties the result to the borrow of `Ref`), this
    /// returns a reference with the original `'a` lifetime of the storage.
    pub(crate) fn storage(&self) -> &'a SparseSetInner<T> {
        self.inner
    }
}

impl<T: 'static> Deref for Ref<'_, T> {
    type Target = SparseSetInner<T>;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

// Ref holds either an RwLockReadGuard (auto-released on drop) or nothing.
// No manual Drop needed — the Option<RwLockReadGuard> handles it.

// SAFETY: Ref only provides shared (&) access to the inner data.
// The RwLock guarantees no exclusive access exists when the guard is held.
// When unlocked, the caller guarantees external lock management.
unsafe impl<T: Send + Sync + 'static> Send for Ref<'_, T> {}
unsafe impl<T: Send + Sync + 'static> Sync for Ref<'_, T> {}

/// Exclusive write access to a component storage.
///
/// Automatically releases the lock when dropped.
/// Dereferences to [`SparseSetInner<T>`] for accessing and modifying component data.
pub struct RefMut<'a, T: 'static> {
    inner: *mut SparseSetInner<T>,
    _guard: Option<RwLockWriteGuard<'a, ()>>,
    _marker: PhantomData<&'a mut SparseSetInner<T>>,
}

impl<'a, T: 'static> RefMut<'a, T> {
    /// Creates a new exclusive borrow guard, acquiring the storage's write lock.
    pub(crate) fn new(storage: &'a ComponentStorage) -> Self {
        let guard = storage.lock_write();
        // SAFETY: lock_write() guarantees exclusive access. We cast away
        // the shared reference to get a mutable pointer, which is safe because
        // the write lock ensures no other references exist.
        let inner = storage.typed::<T>() as *const SparseSetInner<T> as *mut SparseSetInner<T>;
        Self {
            inner,
            _guard: Some(guard),
            _marker: PhantomData,
        }
    }

    /// Creates an exclusive borrow without acquiring a lock.
    ///
    /// The caller must ensure the write lock is already held externally
    /// (e.g. via `acquire_sorted`).
    pub(crate) fn new_unlocked(storage: &'a ComponentStorage) -> Self {
        let inner = storage.typed::<T>() as *const SparseSetInner<T> as *mut SparseSetInner<T>;
        Self {
            inner,
            _guard: None,
            _marker: PhantomData,
        }
    }

    /// Returns the raw pointer to the underlying storage.
    ///
    /// Used by [`QueryItem`](crate::QueryItem) to access components without
    /// requiring `&mut self`, enabling per-entity mutable access in iterators.
    pub(crate) fn storage_ptr(&self) -> *mut SparseSetInner<T> {
        self.inner
    }
}

impl<T: 'static> Deref for RefMut<'_, T> {
    type Target = SparseSetInner<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: We have exclusive access guaranteed by the write lock.
        unsafe { &*self.inner }
    }
}

impl<T: 'static> DerefMut for RefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: We have exclusive access guaranteed by the write lock.
        unsafe { &mut *self.inner }
    }
}

// RefMut holds either an RwLockWriteGuard (auto-released on drop) or nothing.
// No manual Drop needed — the Option<RwLockWriteGuard> handles it.

// SAFETY: RefMut has exclusive access to the inner data.
// The RwLock ensures no other access exists when the guard is held.
unsafe impl<T: Send + Sync + 'static> Send for RefMut<'_, T> {}
unsafe impl<T: Send + Sync + 'static> Sync for RefMut<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42);
        assert_eq!(set.get(5), Some(&42));
    }

    #[test]
    fn insert_replace() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42);
        set.insert(5, 99);
        assert_eq!(set.get(5), Some(&99));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn remove_returns_value() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42);
        assert_eq!(set.remove(5), Some(42));
        assert_eq!(set.get(5), None);
    }

    #[test]
    fn remove_nonexistent() {
        let mut set = SparseSetInner::<u32>::new();
        assert_eq!(set.remove(5), None);
    }

    #[test]
    fn contains() {
        let mut set = SparseSetInner::<u32>::new();
        assert!(!set.contains(5));
        set.insert(5, 42);
        assert!(set.contains(5));
        set.remove(5);
        assert!(!set.contains(5));
    }

    #[test]
    fn iteration() {
        let mut set = SparseSetInner::<&str>::new();
        set.insert(1, "a");
        set.insert(5, "b");
        set.insert(3, "c");

        let mut items: Vec<_> = set.iter().collect();
        items.sort_by_key(|(idx, _)| *idx);
        assert_eq!(items, vec![(1, &"a"), (3, &"c"), (5, &"b")]);
    }

    #[test]
    fn swap_remove_correctness() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(0, 10);
        set.insert(1, 20);
        set.insert(2, 30);

        // Remove middle element (dense index 1), last element (entity 2) swaps in
        set.remove(1);

        assert_eq!(set.get(0), Some(&10));
        assert_eq!(set.get(1), None);
        assert_eq!(set.get(2), Some(&30));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn lock_shared_multiple() {
        let storage = ComponentStorage::new::<u32>();
        let _a = storage.lock_read();
        let _b = storage.lock_read();
        // Both locks succeed
    }

    #[test]
    fn lock_exclusive_alone() {
        let storage = ComponentStorage::new::<u32>();
        let _guard = storage.lock_write();
    }

    #[test]
    #[should_panic(expected = "Cannot borrow `u32` mutably: already borrowed")]
    fn lock_exclusive_conflicts_shared() {
        let storage = ComponentStorage::new::<u32>();
        let _r = storage.lock_read();
        let _w = storage.lock_write(); // Should panic
    }

    #[test]
    #[should_panic(expected = "Cannot borrow `u32` immutably: already borrowed mutably")]
    fn lock_shared_conflicts_exclusive() {
        let storage = ComponentStorage::new::<u32>();
        let _w = storage.lock_write();
        let _r = storage.lock_read(); // Should panic
    }

    #[test]
    fn lock_released_on_drop() {
        let storage = ComponentStorage::new::<u32>();
        {
            let _guard = Ref::<u32>::new(&storage);
        }
        // After Ref is dropped, exclusive lock should succeed
        let _guard = RefMut::<u32>::new(&storage);
    }

    #[test]
    fn ref_mut_allows_mutation() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.typed_mut::<u32>().insert(0, 42);
        {
            let mut guard = RefMut::<u32>::new(&storage);
            guard.insert(0, 99);
        }
        assert_eq!(storage.typed::<u32>().get(0), Some(&99));
    }

    #[test]
    fn remove_untyped_works() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.typed_mut::<u32>().insert(5, 42);
        assert!(storage.contains_untyped(5));
        assert!(storage.remove_untyped(5));
        assert!(!storage.contains_untyped(5));
    }

    // ---- Change detection tests ----

    #[test]
    fn insert_with_tick_tracks_added() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(5, 42, 10);
        assert!(set.added_since(5, 0));
        assert!(set.added_since(5, 9));
        assert!(!set.added_since(5, 10)); // not strictly after
        assert!(!set.added_since(5, 11));
    }

    #[test]
    fn insert_with_tick_tracks_changed() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(5, 42, 10);
        assert!(set.changed_since(5, 0));
        assert!(set.changed_since(5, 9));
        assert!(!set.changed_since(5, 10));
    }

    #[test]
    fn replace_updates_changed_tick() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(5, 42, 10);
        set.insert_with_tick(5, 99, 20); // replace

        // Added tick stays at 10
        assert!(set.added_since(5, 9));
        assert!(!set.added_since(5, 10));

        // Changed tick is now 20
        assert!(set.changed_since(5, 19));
        assert!(!set.changed_since(5, 20));
    }

    #[test]
    fn get_mut_tracked_marks_changed() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(5, 42, 10);

        *set.get_mut_tracked(5, 25).unwrap() = 99;
        assert_eq!(set.get(5), Some(&99));
        assert!(set.changed_since(5, 24));
        assert!(!set.changed_since(5, 25));
    }

    #[test]
    fn iter_mut_tracked_marks_all_changed() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(0, 10, 1);
        set.insert_with_tick(1, 20, 2);
        set.insert_with_tick(2, 30, 3);

        for (_, val) in set.iter_mut_tracked(50) {
            *val += 1;
        }

        assert!(set.changed_since(0, 49));
        assert!(set.changed_since(1, 49));
        assert!(set.changed_since(2, 49));
        assert!(!set.changed_since(0, 50));
    }

    #[test]
    fn untracked_insert_uses_tick_zero() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42); // tick = 0
        assert!(!set.added_since(5, 0));
        assert!(!set.changed_since(5, 0));
    }

    #[test]
    fn remove_maintains_tick_arrays() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(0, 10, 1);
        set.insert_with_tick(1, 20, 5);
        set.insert_with_tick(2, 30, 10);

        // Remove entity 0 — entity 2 swaps into its slot
        set.remove(0);

        // Entity 2 should retain its ticks
        assert!(set.added_since(2, 9));
        assert!(!set.added_since(2, 10));

        // Entity 1 should retain its ticks
        assert!(set.added_since(1, 4));
        assert!(!set.added_since(1, 5));
    }

    #[test]
    fn changed_since_nonexistent_entity() {
        let set = SparseSetInner::<u32>::new();
        assert!(!set.changed_since(999, 0));
        assert!(!set.added_since(999, 0));
    }

    #[test]
    fn storage_changed_since_untyped() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.typed_mut::<u32>().insert_with_tick(5, 42, 10);

        assert!(storage.changed_since_untyped(5, 9));
        assert!(!storage.changed_since_untyped(5, 10));
        assert!(storage.added_since_untyped(5, 9));
        assert!(!storage.added_since_untyped(5, 10));
    }
}
