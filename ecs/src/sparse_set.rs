use std::any::Any;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicI32, Ordering};

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

/// A type-erased sparse set that stores components of a single type.
///
/// Provides runtime borrow checking to prevent simultaneous shared and
/// exclusive access. Thread-safe via atomic borrow tracking.
/// Used internally by [`World`](crate::World).
pub(crate) struct ComponentStorage {
    inner: Box<dyn Any + Send + Sync>,
    /// Borrow state: 0 = free, positive = N shared readers, -1 = exclusive writer.
    borrow_state: AtomicI32,
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
}

// SAFETY: ComponentStorage uses AtomicI32 for borrow tracking and
// Box<dyn Any + Send + Sync> for the inner data. All access is
// protected by the atomic borrow protocol.
unsafe impl Send for ComponentStorage {}
unsafe impl Sync for ComponentStorage {}

impl ComponentStorage {
    /// Creates a new component storage for type `T`.
    pub fn new<T: Send + Sync + 'static>() -> Self {
        Self {
            inner: Box::new(SparseSetInner::<T>::new()),
            borrow_state: AtomicI32::new(0),
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

    /// Acquires a shared borrow. Panics if exclusively borrowed.
    pub fn borrow(&self) {
        let prev = self.borrow_state.fetch_add(1, Ordering::Acquire);
        if prev < 0 {
            // Undo the increment before panicking
            self.borrow_state.fetch_sub(1, Ordering::Release);
            panic!(
                "Cannot borrow `{}` immutably: already borrowed mutably",
                self.type_name
            );
        }
    }

    /// Releases a shared borrow.
    pub fn release_borrow(&self) {
        let prev = self.borrow_state.fetch_sub(1, Ordering::Release);
        debug_assert!(prev > 0, "release_borrow called without matching borrow");
    }

    /// Acquires an exclusive borrow. Panics if any borrow is active.
    pub fn borrow_mut(&self) {
        match self
            .borrow_state
            .compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => {}
            Err(state) => {
                if state > 0 {
                    panic!(
                        "Cannot borrow `{}` mutably: already borrowed immutably ({} readers)",
                        self.type_name, state
                    );
                } else {
                    panic!(
                        "Cannot borrow `{}` mutably: already borrowed mutably",
                        self.type_name
                    );
                }
            }
        }
    }

    /// Releases an exclusive borrow.
    pub fn release_borrow_mut(&self) {
        let prev = self.borrow_state.swap(0, Ordering::Release);
        debug_assert_eq!(
            prev, -1,
            "release_borrow_mut called without matching borrow_mut"
        );
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
}

/// Shared read access to a component storage.
///
/// Automatically releases the shared borrow when dropped.
/// Dereferences to [`SparseSetInner<T>`] for accessing component data.
pub struct Ref<'a, T: 'static> {
    inner: &'a SparseSetInner<T>,
    storage: &'a ComponentStorage,
}

impl<'a, T: 'static> Ref<'a, T> {
    /// Creates a new shared borrow guard.
    pub(crate) fn new(storage: &'a ComponentStorage) -> Self {
        storage.borrow();
        Self {
            inner: storage.typed::<T>(),
            storage,
        }
    }
}

impl<T: 'static> Deref for Ref<'_, T> {
    type Target = SparseSetInner<T>;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<T: 'static> Drop for Ref<'_, T> {
    fn drop(&mut self) {
        self.storage.release_borrow();
    }
}

// SAFETY: Ref only provides shared (&) access to the inner data.
// The atomic borrow tracking ensures no exclusive access exists.
unsafe impl<T: Send + Sync + 'static> Send for Ref<'_, T> {}
unsafe impl<T: Send + Sync + 'static> Sync for Ref<'_, T> {}

/// Exclusive write access to a component storage.
///
/// Automatically releases the exclusive borrow when dropped.
/// Dereferences to [`SparseSetInner<T>`] for accessing and modifying component data.
pub struct RefMut<'a, T: 'static> {
    inner: *mut SparseSetInner<T>,
    storage: &'a ComponentStorage,
    _marker: PhantomData<&'a mut SparseSetInner<T>>,
}

impl<'a, T: 'static> RefMut<'a, T> {
    /// Creates a new exclusive borrow guard.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `storage` contains a `SparseSetInner<T>`
    /// and that no other references to the inner data exist. This is enforced
    /// by the runtime borrow checking in `ComponentStorage::borrow_mut()`.
    pub(crate) fn new(storage: &'a ComponentStorage) -> Self {
        storage.borrow_mut();
        // SAFETY: borrow_mut() guarantees exclusive access. We cast away
        // the shared reference to get a mutable pointer, which is safe because
        // the borrow tracking ensures no other references exist.
        let inner = storage.typed::<T>() as *const SparseSetInner<T> as *mut SparseSetInner<T>;
        Self {
            inner,
            storage,
            _marker: PhantomData,
        }
    }
}

impl<T: 'static> Deref for RefMut<'_, T> {
    type Target = SparseSetInner<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: We have exclusive access guaranteed by borrow tracking.
        unsafe { &*self.inner }
    }
}

impl<T: 'static> DerefMut for RefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: We have exclusive access guaranteed by borrow tracking.
        unsafe { &mut *self.inner }
    }
}

impl<T: 'static> Drop for RefMut<'_, T> {
    fn drop(&mut self) {
        self.storage.release_borrow_mut();
    }
}

// SAFETY: RefMut has exclusive access to the inner data.
// The atomic borrow tracking ensures no other access exists.
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
    fn borrow_shared_multiple() {
        let storage = ComponentStorage::new::<u32>();
        storage.borrow();
        storage.borrow();
        // Both borrows succeed
        storage.release_borrow();
        storage.release_borrow();
    }

    #[test]
    fn borrow_exclusive_alone() {
        let storage = ComponentStorage::new::<u32>();
        storage.borrow_mut();
        storage.release_borrow_mut();
    }

    #[test]
    #[should_panic(expected = "Cannot borrow `u32` mutably: already borrowed immutably")]
    fn borrow_exclusive_conflicts_shared() {
        let storage = ComponentStorage::new::<u32>();
        storage.borrow();
        storage.borrow_mut(); // Should panic
    }

    #[test]
    #[should_panic(expected = "Cannot borrow `u32` immutably: already borrowed mutably")]
    fn borrow_shared_conflicts_exclusive() {
        let storage = ComponentStorage::new::<u32>();
        storage.borrow_mut();
        storage.borrow(); // Should panic
    }

    #[test]
    fn borrow_released_on_drop() {
        let storage = ComponentStorage::new::<u32>();
        {
            let _guard = Ref::<u32>::new(&storage);
        }
        // After Ref is dropped, exclusive borrow should succeed
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
