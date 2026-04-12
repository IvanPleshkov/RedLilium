use std::marker::PhantomData;

use crate::sparse_set::ComponentStorage;

/// Trait for filter types that can check entity membership.
///
/// Implemented by [`ContainsChecker`], [`ChangedFilter`], [`AddedFilter`],
/// [`RemovedFilter`], [`OrFilter`], and [`AnyFilter`].
pub trait Filter {
    /// Returns `true` if the entity passes this filter.
    fn matches(&self, entity_index: u32) -> bool;
}

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

impl Filter for ContainsChecker<'_> {
    fn matches(&self, entity_index: u32) -> bool {
        self.matches(entity_index)
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

impl Filter for ChangedFilter<'_> {
    fn matches(&self, entity_index: u32) -> bool {
        self.matches(entity_index)
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

impl Filter for AddedFilter<'_> {
    fn matches(&self, entity_index: u32) -> bool {
        self.matches(entity_index)
    }
}

/// Filter for entities whose component was removed since a given tick.
///
/// Created by [`World::removed`](crate::World::removed). Works like
/// [`ContainsChecker`] — use in iteration with `matches()`.
///
/// Removal tracking records are cleared by
/// [`World::clear_removed_tracking`](crate::World::clear_removed_tracking).
///
/// # Example
///
/// ```ignore
/// let positions = world.read::<Position>();
/// let removed_health = world.removed::<Health>(last_tick);
/// for (idx, pos) in positions.iter() {
///     if removed_health.matches(idx) {
///         // entity had Health removed since last_tick
///     }
/// }
/// ```
pub struct RemovedFilter<'a> {
    storage: Option<&'a ComponentStorage>,
    since_tick: u64,
}

impl<'a> RemovedFilter<'a> {
    /// Creates a new removed filter.
    pub(crate) fn new(storage: Option<&'a ComponentStorage>, since_tick: u64) -> Self {
        Self {
            storage,
            since_tick,
        }
    }

    /// Returns true if the entity's component was removed since `since_tick`.
    pub fn matches(&self, entity_index: u32) -> bool {
        self.storage
            .is_some_and(|s| s.removed_since_untyped(entity_index, self.since_tick))
    }

    /// Returns an iterator over entity indices whose component was removed
    /// since `since_tick`.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.storage
            .into_iter()
            .flat_map(move |s| s.removed_entities_since(self.since_tick))
    }
}

impl Filter for RemovedFilter<'_> {
    fn matches(&self, entity_index: u32) -> bool {
        self.matches(entity_index)
    }
}

/// Composite filter that matches if **either** sub-filter matches (logical OR).
///
/// Created by [`Or<A, B>`](crate::Or) access element.
///
/// # Example
///
/// ```ignore
/// let (positions, can_move) = ctx.query::<(Read<Pos>, Or<With<Flying>, With<Swimming>>)>();
/// for (idx, pos) in positions.iter() {
///     if can_move.matches(idx) {
///         // entity has Flying OR Swimming
///     }
/// }
/// ```
pub struct OrFilter<A, B> {
    a: A,
    b: B,
}

impl<A: Filter, B: Filter> OrFilter<A, B> {
    pub(crate) fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Filter, B: Filter> Filter for OrFilter<A, B> {
    fn matches(&self, entity_index: u32) -> bool {
        self.a.matches(entity_index) || self.b.matches(entity_index)
    }
}

/// Composite filter that matches if **any** sub-filter in the tuple matches.
///
/// Created by [`Any<(A, B, ...)>`](crate::Any) access element.
/// Supports tuples of 2-8 filter elements.
///
/// # Example
///
/// ```ignore
/// let (positions, movable) = ctx.query::<(Read<Pos>, Any<(With<Flying>, With<Swimming>, With<Walking>)>)>();
/// for (idx, pos) in positions.iter() {
///     if movable.matches(idx) {
///         // entity has Flying OR Swimming OR Walking
///     }
/// }
/// ```
pub struct AnyFilter<T> {
    filters: T,
}

impl<T> AnyFilter<T> {
    pub(crate) fn new(filters: T) -> Self {
        Self { filters }
    }
}

macro_rules! impl_any_filter {
    ($($idx:tt $T:ident),+) => {
        impl<$($T: Filter),+> Filter for AnyFilter<($($T,)+)> {
            fn matches(&self, entity_index: u32) -> bool {
                $(self.filters.$idx.matches(entity_index))||+
            }
        }
    };
}

impl_any_filter!(0 A, 1 B);
impl_any_filter!(0 A, 1 B, 2 C);
impl_any_filter!(0 A, 1 B, 2 C, 3 D);
impl_any_filter!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_any_filter!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_any_filter!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_any_filter!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);

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

    #[test]
    fn removed_filter_matches() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.typed_mut::<u32>().insert(5, 42);
        storage.typed_mut::<u32>().remove(5);
        storage.record_removal(5, 10);

        let filter = RemovedFilter::new(Some(&storage), 0);
        assert!(filter.matches(5));
    }

    #[test]
    fn removed_filter_respects_tick() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.record_removal(5, 10);

        // since_tick 9 → tick 10 > 9, so matches
        assert!(RemovedFilter::new(Some(&storage), 9).matches(5));
        // since_tick 10 → tick 10 is NOT strictly after 10
        assert!(!RemovedFilter::new(Some(&storage), 10).matches(5));
        // since_tick 11 → no match
        assert!(!RemovedFilter::new(Some(&storage), 11).matches(5));
    }

    #[test]
    fn removed_filter_rejects_non_removed() {
        let storage = ComponentStorage::new::<u32>();
        let filter = RemovedFilter::new(Some(&storage), 0);
        assert!(!filter.matches(5));
    }

    #[test]
    fn removed_filter_missing_storage_matches_nothing() {
        let filter = RemovedFilter::new(None, 0);
        assert!(!filter.matches(0));
        assert!(!filter.matches(100));
    }

    #[test]
    fn removed_filter_iter() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.record_removal(3, 10);
        storage.record_removal(7, 15);
        storage.record_removal(5, 5);

        let filter = RemovedFilter::new(Some(&storage), 9);
        let mut entities: Vec<u32> = filter.iter().collect();
        entities.sort();
        assert_eq!(entities, vec![3, 7]);
    }

    #[test]
    fn removed_filter_iter_empty_on_none() {
        let filter = RemovedFilter::new(None, 0);
        assert_eq!(filter.iter().count(), 0);
    }

    #[test]
    fn clear_removed_resets_tracking() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.record_removal(5, 10);
        assert!(RemovedFilter::new(Some(&storage), 0).matches(5));

        storage.clear_removed();
        assert!(!RemovedFilter::new(Some(&storage), 0).matches(5));
    }

    // ---- OrFilter tests ----

    #[test]
    fn or_filter_matches_first() {
        let mut s1 = ComponentStorage::new::<u32>();
        s1.typed_mut::<u32>().insert(5, 42);
        let s2 = ComponentStorage::new::<u64>();

        let a = ContainsChecker::with(Some(&s1));
        let b = ContainsChecker::with(Some(&s2));
        let or = OrFilter::new(a, b);
        assert!(or.matches(5)); // has u32
    }

    #[test]
    fn or_filter_matches_second() {
        let s1 = ComponentStorage::new::<u32>();
        let mut s2 = ComponentStorage::new::<u64>();
        s2.typed_mut::<u64>().insert(5, 99);

        let a = ContainsChecker::with(Some(&s1));
        let b = ContainsChecker::with(Some(&s2));
        let or = OrFilter::new(a, b);
        assert!(or.matches(5)); // has u64
    }

    #[test]
    fn or_filter_matches_both() {
        let mut s1 = ComponentStorage::new::<u32>();
        s1.typed_mut::<u32>().insert(5, 42);
        let mut s2 = ComponentStorage::new::<u64>();
        s2.typed_mut::<u64>().insert(5, 99);

        let a = ContainsChecker::with(Some(&s1));
        let b = ContainsChecker::with(Some(&s2));
        let or = OrFilter::new(a, b);
        assert!(or.matches(5)); // has both
    }

    #[test]
    fn or_filter_rejects_neither() {
        let s1 = ComponentStorage::new::<u32>();
        let s2 = ComponentStorage::new::<u64>();

        let a = ContainsChecker::with(Some(&s1));
        let b = ContainsChecker::with(Some(&s2));
        let or = OrFilter::new(a, b);
        assert!(!or.matches(5)); // has neither
    }

    #[test]
    fn or_filter_via_filter_trait() {
        let mut s1 = ComponentStorage::new::<u32>();
        s1.typed_mut::<u32>().insert(5, 42);
        let s2 = ComponentStorage::new::<u64>();

        let a = ContainsChecker::with(Some(&s1));
        let b = ContainsChecker::with(Some(&s2));
        let or = OrFilter::new(a, b);
        // Use via Filter trait
        let f: &dyn Filter = &or;
        assert!(f.matches(5));
    }

    // ---- AnyFilter tests ----

    #[test]
    fn any_filter_matches_one_of_three() {
        let s1 = ComponentStorage::new::<u32>();
        let mut s2 = ComponentStorage::new::<u64>();
        s2.typed_mut::<u64>().insert(5, 99);
        let s3 = ComponentStorage::new::<f32>();

        let a = ContainsChecker::with(Some(&s1));
        let b = ContainsChecker::with(Some(&s2));
        let c = ContainsChecker::with(Some(&s3));
        let any = AnyFilter::new((a, b, c));
        assert!(any.matches(5)); // has u64
    }

    #[test]
    fn any_filter_rejects_none() {
        let s1 = ComponentStorage::new::<u32>();
        let s2 = ComponentStorage::new::<u64>();
        let s3 = ComponentStorage::new::<f32>();

        let a = ContainsChecker::with(Some(&s1));
        let b = ContainsChecker::with(Some(&s2));
        let c = ContainsChecker::with(Some(&s3));
        let any = AnyFilter::new((a, b, c));
        assert!(!any.matches(5)); // has none
    }
}
