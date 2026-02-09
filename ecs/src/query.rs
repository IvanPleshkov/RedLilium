use std::marker::PhantomData;

use crate::sparse_set::{ComponentStorage, Ref, RefMut};

/// Shared read access to all components of type T.
///
/// This is a type alias for [`Ref<T>`], returned by [`World::read`](crate::World::read).
/// Dereferences to [`SparseSetInner<T>`](crate::SparseSetInner) for iteration and lookup.
pub type Read<'a, T> = Ref<'a, T>;

/// Exclusive write access to all components of type T.
///
/// This is a type alias for [`RefMut<T>`], returned by [`World::write`](crate::World::write).
/// Dereferences to [`SparseSetInner<T>`](crate::SparseSetInner) for iteration, lookup, and mutation.
pub type Write<'a, T> = RefMut<'a, T>;

/// Marker type for filtering entities that have component T,
/// without borrowing the component data.
///
/// Used with [`World::with`](crate::World::with) to create a [`ContainsChecker`].
pub struct With<T: 'static>(PhantomData<T>);

/// Marker type for filtering entities that do NOT have component T.
///
/// Used with [`World::without`](crate::World::without) to create a [`ContainsChecker`].
pub struct Without<T: 'static>(PhantomData<T>);

/// A lightweight filter for checking whether entities have a specific component.
///
/// Created by [`World::with`](crate::World::with) and [`World::without`](crate::World::without).
/// Does not borrow component data — only checks existence in the sparse array.
///
/// # Example
///
/// ```ignore
/// let positions = world.write::<Position>();
/// let frozen = world.without::<Frozen>();
///
/// for (entity_idx, pos) in positions.iter_mut() {
///     if frozen.matches(entity_idx) {
///         pos.x += 1.0;
///     }
/// }
/// ```
pub struct ContainsChecker<'a> {
    storage: Option<&'a ComponentStorage>,
    inverted: bool,
}

impl<'a> ContainsChecker<'a> {
    /// Creates a `With`-style filter (matches entities that have the component).
    pub(crate) fn with(storage: Option<&'a ComponentStorage>) -> Self {
        Self {
            storage,
            inverted: false,
        }
    }

    /// Creates a `Without`-style filter (matches entities that lack the component).
    pub(crate) fn without(storage: Option<&'a ComponentStorage>) -> Self {
        Self {
            storage,
            inverted: true,
        }
    }

    /// Returns true if the entity passes this filter.
    pub fn matches(&self, entity_index: u32) -> bool {
        let has = self
            .storage
            .is_some_and(|s| s.contains_untyped(entity_index));
        if self.inverted { !has } else { has }
    }
}

/// Filter for entities whose component was changed since a given tick.
///
/// Created by [`World::changed`](crate::World::changed). Works like
/// [`ContainsChecker`] — use in iteration with `matches()`.
///
/// # Example
///
/// ```ignore
/// let transforms = world.read::<Transform>();
/// let changed = world.changed::<Transform>(last_tick);
/// for (idx, t) in transforms.iter() {
///     if changed.matches(idx) {
///         // transform was modified since last_tick
///     }
/// }
/// ```
pub struct ChangedFilter<'a> {
    storage: Option<&'a ComponentStorage>,
    since_tick: u64,
}

impl<'a> ChangedFilter<'a> {
    /// Creates a new changed filter.
    pub(crate) fn new(storage: Option<&'a ComponentStorage>, since_tick: u64) -> Self {
        Self {
            storage,
            since_tick,
        }
    }

    /// Returns true if the entity's component was changed since `since_tick`.
    pub fn matches(&self, entity_index: u32) -> bool {
        self.storage
            .is_some_and(|s| s.changed_since_untyped(entity_index, self.since_tick))
    }
}

/// Filter for entities whose component was added since a given tick.
///
/// Created by [`World::added`](crate::World::added). Works like
/// [`ContainsChecker`] — use in iteration with `matches()`.
pub struct AddedFilter<'a> {
    storage: Option<&'a ComponentStorage>,
    since_tick: u64,
}

impl<'a> AddedFilter<'a> {
    /// Creates a new added filter.
    pub(crate) fn new(storage: Option<&'a ComponentStorage>, since_tick: u64) -> Self {
        Self {
            storage,
            since_tick,
        }
    }

    /// Returns true if the entity's component was added since `since_tick`.
    pub fn matches(&self, entity_index: u32) -> bool {
        self.storage
            .is_some_and(|s| s.added_since_untyped(entity_index, self.since_tick))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_filter_matches() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.typed_mut::<u32>().insert(5, 42);
        let checker = ContainsChecker::with(Some(&storage));
        assert!(checker.matches(5));
    }

    #[test]
    fn with_filter_rejects() {
        let storage = ComponentStorage::new::<u32>();
        let checker = ContainsChecker::with(Some(&storage));
        assert!(!checker.matches(5));
    }

    #[test]
    fn without_filter_matches() {
        let storage = ComponentStorage::new::<u32>();
        let checker = ContainsChecker::without(Some(&storage));
        assert!(checker.matches(5)); // Entity 5 does NOT have the component
    }

    #[test]
    fn without_filter_rejects() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.typed_mut::<u32>().insert(5, 42);
        let checker = ContainsChecker::without(Some(&storage));
        assert!(!checker.matches(5)); // Entity 5 HAS the component
    }

    #[test]
    fn missing_storage_with_matches_nothing() {
        let checker = ContainsChecker::with(None);
        assert!(!checker.matches(0));
        assert!(!checker.matches(100));
    }

    #[test]
    fn missing_storage_without_matches_everything() {
        let checker = ContainsChecker::without(None);
        assert!(checker.matches(0));
        assert!(checker.matches(100));
    }
}
