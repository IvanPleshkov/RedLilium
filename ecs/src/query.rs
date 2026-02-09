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
