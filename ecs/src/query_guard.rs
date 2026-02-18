use std::cell::Cell;
use std::ops::{Deref, DerefMut};

use fixedbitset::FixedBitSet;

use crate::access_set::AccessSet;
use crate::resource::{ResourceRef, ResourceRefMut};
use crate::sparse_set::{LockGuard, Ref, RefMut, SparseSetInner};
use crate::system_context::LockTracking;

/// A guard holding component/resource locks and their fetched data.
///
/// Created by [`SystemContext::query()`](crate::SystemContext::query).
/// Locks are acquired in TypeId-sorted order (same as `lock().execute()`)
/// to prevent deadlocks. Locks are held until the guard is dropped.
///
/// Unlike `lock().execute()`, which runs a synchronous closure with the
/// locked data, `QueryGuard` lets you access the data directly without
/// a closure — enabling normal control flow, `?` operators, and multiple
/// statements with the locks held.
///
/// Access fetched data via the public `items` field:
///
/// ```ignore
/// // Read-only:
/// let q = ctx.query::<(Read<Position>, Read<Velocity>)>();
/// let (positions, velocities) = &q.items;
///
/// // With writes:
/// let mut q = ctx.query::<(Write<Position>, Read<Velocity>)>();
/// let (positions, velocities) = &mut q.items;
/// for (idx, pos) in positions.iter_mut() {
///     if let Some(vel) = velocities.get(idx) {
///         pos.x += vel.x;
///     }
/// }
/// // locks released when `q` goes out of scope
/// ```
///
/// # Limitations
///
/// Main-thread resources ([`MainThreadRes`](crate::MainThreadRes),
/// [`MainThreadResMut`](crate::MainThreadResMut)) are not supported.
/// Use `lock().execute()` for those.
pub struct QueryGuard<'a, A: AccessSet> {
    _guards: Vec<LockGuard<'a>>,
    /// The fetched component/resource data. Destructure this to access
    /// individual storages.
    pub items: A::Item<'a>,
    /// Deadlock tracking — unregisters held locks when this guard is dropped.
    /// `None` when created outside of a SystemContext (e.g. in tests).
    _tracking: Option<LockTracking<'a>>,
}

impl<'a, A: AccessSet> QueryGuard<'a, A> {
    #[cfg(test)]
    pub(crate) fn new(guards: Vec<LockGuard<'a>>, items: A::Item<'a>) -> Self {
        Self {
            _guards: guards,
            items,
            _tracking: None,
        }
    }

    pub(crate) fn new_tracked(
        guards: Vec<LockGuard<'a>>,
        items: A::Item<'a>,
        tracking: LockTracking<'a>,
    ) -> Self {
        Self {
            _guards: guards,
            items,
            _tracking: Some(tracking),
        }
    }
}

impl<'a, A: AccessSet> QueryGuard<'a, A>
where
    A::Item<'a>: QueryItem,
{
    /// Iterates over matching entities in parallel, calling `f` for each.
    ///
    /// Splits the entity list into batches and processes them on separate
    /// threads via [`std::thread::scope`]. On WASM, falls back to
    /// sequential iteration.
    ///
    /// The closure receives `(entity_index, item)` for each matching
    /// entity. Since it is called from multiple threads, it must be `Fn`
    /// (not `FnMut`). Use atomics, `Mutex`, or thread-local accumulators
    /// for shared mutable state.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let q = ctx.query::<(Write<Position>, Read<Velocity>)>();
    /// q.par_for_each(|entity_idx, (pos, vel)| {
    ///     pos.x += vel.x;
    /// });
    /// ```
    pub fn par_for_each<F>(&self, f: F)
    where
        A::Item<'a>: Sync,
        F: Fn(u32, <A::Item<'a> as QueryItem>::Item) + Sync,
    {
        self.par_for_each_with(crate::par_for_each::ParConfig::default(), f);
    }

    /// Like [`par_for_each`](Self::par_for_each), but with explicit
    /// parallelism configuration.
    pub fn par_for_each_with<F>(&self, config: crate::par_for_each::ParConfig, f: F)
    where
        A::Item<'a>: Sync,
        F: Fn(u32, <A::Item<'a> as QueryItem>::Item) + Sync,
    {
        if let Some(intersected) = self.items.query_intersected_entities() {
            crate::par_for_each::par_for_each_entities(&self.items, &intersected, &config, &f);
        } else {
            crate::par_for_each::par_for_each_entities(
                &self.items,
                self.items.query_entities(),
                &config,
                &f,
            );
        }
    }
}

impl<'a, A: AccessSet> IntoIterator for QueryGuard<'a, A>
where
    A::Item<'a>: QueryItem,
{
    type Item = (u32, <A::Item<'a> as QueryItem>::Item);
    type IntoIter = QueryIter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        QueryIter::from(self)
    }
}

// ---- QueryItem trait and iterator ----

/// Trait for types that provide per-entity access in a joined query.
///
/// Implemented for [`Ref`] (shared component access) and [`RefMut`]
/// (exclusive component access), and for tuples of these types.
///
/// This trait enables [`QueryIter`] to perform inner joins across
/// multiple component storages.
pub trait QueryItem {
    /// The per-entity reference type (e.g., `&T` or `&mut T`).
    type Item;

    /// Number of entities in this storage.
    fn query_count(&self) -> usize;

    /// Entity indices in dense order.
    fn query_entities(&self) -> &[u32];

    /// Fetch the item for a given entity index.
    ///
    /// # Safety
    ///
    /// For mutable items, the caller must ensure each `entity_index` is
    /// accessed at most once (no aliasing mutable references).
    unsafe fn query_get(&self, entity_index: u32) -> Option<Self::Item>;

    /// Returns the component membership bitset, if available.
    ///
    /// Returns `Some` for component storages (`Ref`/`RefMut`), `None` for
    /// resources (which are singletons and always match).
    fn query_membership(&self) -> Option<&FixedBitSet> {
        None
    }

    /// Returns the per-entity flags slice, if available.
    ///
    /// Returns `Some` for component storages, `None` for resources.
    fn query_flags(&self) -> Option<&[u32]> {
        None
    }

    /// Pre-computes the set of entity indices that match all component storages.
    ///
    /// Returns `Some(entities)` when there are 2+ component memberships to
    /// intersect, `None` otherwise (single component or resources only).
    /// The returned vec contains only entities present in all storages and
    /// not disabled.
    fn query_intersected_entities(&self) -> Option<Vec<u32>> {
        None
    }
}

impl<'w, T: 'static> QueryItem for Ref<'w, T> {
    type Item = &'w T;

    fn query_count(&self) -> usize {
        self.len()
    }

    fn query_entities(&self) -> &[u32] {
        self.entities()
    }

    unsafe fn query_get(&self, entity_index: u32) -> Option<&'w T> {
        if self.is_entity_disabled(entity_index) {
            return None;
        }
        self.storage().get(entity_index)
    }

    fn query_membership(&self) -> Option<&FixedBitSet> {
        Some(self.storage().membership())
    }

    fn query_flags(&self) -> Option<&[u32]> {
        Some(self.entity_flags())
    }
}

impl<'w, T: 'static> QueryItem for RefMut<'w, T> {
    type Item = &'w mut T;

    fn query_count(&self) -> usize {
        self.len()
    }

    fn query_entities(&self) -> &[u32] {
        self.entities()
    }

    unsafe fn query_get(&self, entity_index: u32) -> Option<&'w mut T> {
        if self.is_entity_disabled(entity_index) {
            return None;
        }
        // SAFETY: write lock is held (via QueryGuard._guards), and the caller
        // (QueryIter) visits each entity_index at most once, so no aliasing
        // mutable references are created.
        unsafe { SparseSetInner::get_ptr_mut(self.storage_ptr(), entity_index).map(|p| &mut *p) }
    }

    fn query_membership(&self) -> Option<&FixedBitSet> {
        // SAFETY: write lock guarantees exclusive access.
        Some(unsafe { &*self.storage_ptr() }.membership())
    }

    fn query_flags(&self) -> Option<&[u32]> {
        Some(self.entity_flags())
    }
}

impl<'w, T: 'static> QueryItem for ResourceRef<'w, T> {
    type Item = &'w T;

    fn query_count(&self) -> usize {
        // Resources are singletons — never the smallest set, so they
        // never drive iteration.
        usize::MAX
    }

    fn query_entities(&self) -> &[u32] {
        // Never selected (count is MAX), but must return a valid slice.
        &[]
    }

    unsafe fn query_get(&self, _entity_index: u32) -> Option<&'w T> {
        // SAFETY: the RwLockReadGuard inside ResourceRef keeps the data
        // valid for 'w. Multiple shared references are safe (read-only).
        unsafe {
            let ptr: *const T = &**self;
            Some(&*ptr)
        }
    }
}

impl<'w, T: 'static> QueryItem for ResourceRefMut<'w, T> {
    type Item = ResMutRef<'w, T>;

    fn query_count(&self) -> usize {
        usize::MAX
    }

    fn query_entities(&self) -> &[u32] {
        &[]
    }

    unsafe fn query_get(&self, _entity_index: u32) -> Option<ResMutRef<'w, T>> {
        assert!(
            !self.borrowed.get(),
            "ResMut<{}> already borrowed mutably by a previous iterator item. \
             Drop the previous item before calling next().",
            std::any::type_name::<T>()
        );
        self.borrowed.set(true);
        // SAFETY: the RwLockWriteGuard inside ResourceRefMut keeps exclusive
        // access for 'w. The borrow flag ensures only one ResMutRef exists
        // at a time, preventing aliasing &mut T.
        // The flag pointer is valid because ResourceRefMut (owning the Cell)
        // lives inside QueryGuard which outlives every ResMutRef.
        let flag = &self.borrowed as *const Cell<bool>;
        unsafe {
            Some(ResMutRef {
                ptr: &mut *self.as_ptr_mut(),
                flag,
            })
        }
    }
}

/// RAII guard for a mutable resource reference during iteration.
///
/// Returned by [`QueryIter::next()`] when the query includes
/// [`ResMut<T>`](crate::ResMut). Dereferences to `&mut T`.
///
/// Clears the borrow flag on [`ResourceRefMut`] when dropped, allowing
/// the next iteration step to create a new mutable reference. Attempting
/// to hold two `ResMutRef`s from the same resource simultaneously (e.g.
/// by calling `iter.next()` while a previous item is still alive) will
/// panic at runtime, similar to [`RefCell`](std::cell::RefCell).
pub struct ResMutRef<'w, T: 'static> {
    ptr: &'w mut T,
    flag: *const Cell<bool>,
}

impl<T: 'static> Deref for ResMutRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.ptr
    }
}

impl<T: 'static> DerefMut for ResMutRef<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.ptr
    }
}

impl<T: 'static> Drop for ResMutRef<'_, T> {
    fn drop(&mut self) {
        // SAFETY: the flag pointer is valid — it points to a Cell<bool>
        // owned by ResourceRefMut inside the QueryGuard, which outlives
        // every ResMutRef yielded by the iterator.
        unsafe { &*self.flag }.set(false);
    }
}

macro_rules! impl_query_item {
    ($($idx:tt $T:ident),+) => {
        impl<$($T: QueryItem),+> QueryItem for ($($T,)+) {
            type Item = ($($T::Item,)+);

            fn query_count(&self) -> usize {
                let mut min = usize::MAX;
                $(
                    min = min.min(self.$idx.query_count());
                )+
                min
            }

            fn query_entities(&self) -> &[u32] {
                let mut _min_count = usize::MAX;
                let mut min_entities: &[u32] = &[];
                $(
                    let count = self.$idx.query_count();
                    if count < _min_count {
                        _min_count = count;
                        min_entities = self.$idx.query_entities();
                    }
                )+
                min_entities
            }

            unsafe fn query_get(&self, entity_index: u32) -> Option<Self::Item> {
                // SAFETY: delegates to each element's query_get with the same
                // entity_index. The caller guarantees unique access per index.
                unsafe {
                    Some(($( self.$idx.query_get(entity_index)?, )+))
                }
            }

            fn query_intersected_entities(&self) -> Option<Vec<u32>> {
                // Collect all component membership bitsets (skip resources which return None).
                let mut bitsets: Vec<&FixedBitSet> = Vec::new();
                $(
                    if let Some(bs) = self.$idx.query_membership() {
                        bitsets.push(bs);
                    }
                )+
                // Need at least 2 component bitsets to benefit from intersection.
                if bitsets.len() < 2 {
                    return None;
                }
                // Sort by population count so we clone the smallest.
                bitsets.sort_by_key(|bs| bs.count_ones(..));
                // Clone the smallest and intersect with all others.
                let mut result = bitsets[0].clone();
                for bs in &bitsets[1..] {
                    result.intersect_with(bs);
                }
                // Filter out disabled entities via flag bits.
                let flags = None $(.or(self.$idx.query_flags()))+;
                if let Some(flags) = flags {
                    Some(result.ones().filter(|&i| {
                        i >= flags.len() || flags[i] & crate::entity::Entity::DISABLED == 0
                    }).map(|i| i as u32).collect())
                } else {
                    Some(result.ones().map(|i| i as u32).collect())
                }
            }
        }
    };
}

impl_query_item!(0 A);
impl_query_item!(0 A, 1 B);
impl_query_item!(0 A, 1 B, 2 C);
impl_query_item!(0 A, 1 B, 2 C, 3 D);
impl_query_item!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_query_item!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_query_item!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_query_item!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);

/// Iterator over entities and their components from a [`QueryGuard`].
///
/// Created via [`From<QueryGuard>`] or [`QueryGuard::into_iter()`].
/// Performs an inner join: iterates over the smallest component storage
/// and yields only entities present in all queried storages.
///
/// Owns the underlying [`QueryGuard`], keeping locks held for the
/// iterator's lifetime. Use [`into_guard`](QueryIter::into_guard) to
/// recover the guard after (partial) iteration.
///
/// ```ignore
/// let q = ctx.query::<(Write<Position>, Read<Velocity>)>();
/// for (entity_idx, (pos, vel)) in q {
///     pos.x += vel.x;
/// }
/// ```
pub struct QueryIter<'a, A: AccessSet> {
    guard: QueryGuard<'a, A>,
    /// Pre-computed matching entity indices (from bitset intersection),
    /// or `None` to fall back to the smallest-set iteration path.
    intersected: Option<Vec<u32>>,
    idx: usize,
}

impl<'a, A: AccessSet> QueryIter<'a, A> {
    /// Converts this iterator back into the underlying [`QueryGuard`],
    /// releasing the iterator state while keeping the locks held.
    pub fn into_guard(self) -> QueryGuard<'a, A> {
        self.guard
    }
}

impl<'a, A: AccessSet> From<QueryGuard<'a, A>> for QueryIter<'a, A>
where
    A::Item<'a>: QueryItem,
{
    fn from(guard: QueryGuard<'a, A>) -> Self {
        let intersected = guard.items.query_intersected_entities();
        Self {
            guard,
            intersected,
            idx: 0,
        }
    }
}

impl<'a, A: AccessSet> Iterator for QueryIter<'a, A>
where
    A::Item<'a>: QueryItem,
{
    type Item = (u32, <A::Item<'a> as QueryItem>::Item);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref entities) = self.intersected {
            // Bitset-accelerated path: every entity is guaranteed to match.
            while self.idx < entities.len() {
                let entity_idx = entities[self.idx];
                self.idx += 1;
                // SAFETY: bitset intersection guarantees the entity has all
                // components and is not disabled. Each entity is visited once.
                if let Some(item) = unsafe { self.guard.items.query_get(entity_idx) } {
                    return Some((entity_idx, item));
                }
            }
        } else {
            // Fallback: walk the smallest set and probe other storages.
            let entities = self.guard.items.query_entities();
            while self.idx < entities.len() {
                let entity_idx = entities[self.idx];
                self.idx += 1;
                // SAFETY: the iterator visits each entity exactly once
                // (monotonically increasing idx), so no aliasing mutable
                // references are created across calls to next().
                if let Some(item) = unsafe { self.guard.items.query_get(entity_idx) } {
                    return Some((entity_idx, item));
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let entities = if let Some(ref intersected) = self.intersected {
            intersected.as_slice()
        } else {
            self.guard.items.query_entities()
        };
        let remaining = entities.len().saturating_sub(self.idx);
        (0, Some(remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_set::{AccessSet, Read, Res, ResMut, Write};
    use crate::world::World;

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }

    /// Helper: constructs a QueryGuard directly from a World (same logic as
    /// `SystemContext::query()` but callable from sync tests).
    fn query<'w, A: AccessSet>(world: &'w World) -> QueryGuard<'w, A> {
        let guards = world.acquire_sorted(&A::access_infos());
        let items = A::fetch_unlocked(world);
        QueryGuard::new(guards, items)
    }

    #[test]
    fn query_reads_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        let q = query::<(Read<Position>,)>(&world);
        let (positions,) = &q.items;
        assert_eq!(positions.len(), 1);
        assert_eq!(positions.iter().next().unwrap().1.x, 42.0);
    }

    #[test]
    fn query_writes_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();

        {
            let mut q = query::<(Write<Position>,)>(&world);
            let (positions,) = &mut q.items;
            for (_, pos) in positions.iter_mut() {
                pos.x = 99.0;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn query_multiple_accesses() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();
        world.insert(e, Velocity { x: 5.0 }).unwrap();

        {
            let mut q = query::<(Write<Position>, Read<Velocity>)>(&world);
            let (positions, velocities) = &mut q.items;
            for (idx, pos) in positions.iter_mut() {
                if let Some(vel) = velocities.get(idx) {
                    pos.x += vel.x;
                }
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[test]
    fn query_with_resources() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.insert_resource(2.0f32);
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();

        {
            let mut q = query::<(Write<Position>, Res<f32>)>(&world);
            let (positions, factor) = &mut q.items;
            let f = **factor;
            for (_, pos) in positions.iter_mut() {
                pos.x *= f;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 20.0);
    }

    #[test]
    fn query_locks_released_on_drop() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        // First query: read
        {
            let q = query::<(Read<Position>,)>(&world);
            let (positions,) = &q.items;
            assert_eq!(positions.len(), 1);
        }
        // Guard dropped — now we can acquire a write lock
        {
            let mut q = query::<(Write<Position>,)>(&world);
            let (positions,) = &mut q.items;
            for (_, pos) in positions.iter_mut() {
                pos.x = 42.0;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 42.0);
    }

    #[test]
    fn query_returns_value_from_get() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        let q = query::<(Read<Position>,)>(&world);
        let (positions,) = &q.items;
        let sum: f32 = positions.iter().map(|(_, p)| p.x).sum();
        assert_eq!(sum, 42.0);
    }

    // ---- QueryIter tests ----

    #[test]
    fn iter_read_only() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();

        let q = query::<(Read<Position>,)>(&world);
        let mut sum = 0.0;
        for (_, (pos,)) in q {
            sum += pos.x;
        }
        assert_eq!(sum, 3.0);
    }

    #[test]
    fn iter_write_mutates() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();

        {
            let q = query::<(Write<Position>,)>(&world);
            for (_, (pos,)) in q {
                pos.x = 99.0;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn iter_join_two_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        // Entity with both components
        let e1 = world.spawn();
        world.insert(e1, Position { x: 10.0 }).unwrap();
        world.insert(e1, Velocity { x: 5.0 }).unwrap();

        // Entity with only Position
        let e2 = world.spawn();
        world.insert(e2, Position { x: 20.0 }).unwrap();

        {
            let q = query::<(Write<Position>, Read<Velocity>)>(&world);
            let mut count = 0;
            for (_, (pos, vel)) in q {
                pos.x += vel.x;
                count += 1;
            }
            // Only e1 has both components
            assert_eq!(count, 1);
        }

        assert_eq!(world.get::<Position>(e1).unwrap().x, 15.0);
        // e2 unchanged (didn't have Velocity)
        assert_eq!(world.get::<Position>(e2).unwrap().x, 20.0);
    }

    #[test]
    fn iter_uses_smallest_set() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        // 3 entities with Position, 1 with Velocity
        for i in 0..3 {
            let e = world.spawn();
            world.insert(e, Position { x: i as f32 }).unwrap();
        }
        let e_vel = world.spawn();
        world.insert(e_vel, Position { x: 100.0 }).unwrap();
        world.insert(e_vel, Velocity { x: 1.0 }).unwrap();

        let q = query::<(Read<Position>, Read<Velocity>)>(&world);
        let results: Vec<_> = QueryIter::from(q).map(|(idx, (p, _))| (idx, p.x)).collect();
        // Should only find the entity that has both
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 100.0);
    }

    #[test]
    fn iter_empty_when_no_matches() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();
        // No Velocity on any entity

        let q = query::<(Read<Position>, Read<Velocity>)>(&world);
        assert_eq!(QueryIter::from(q).count(), 0);
    }

    #[test]
    fn iter_into_iterator() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        let q = query::<(Read<Position>,)>(&world);
        let mut found = false;
        for (_, (pos,)) in q {
            assert_eq!(pos.x, 42.0);
            found = true;
        }
        assert!(found);
    }

    #[test]
    fn iter_multiple_writes() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Velocity { x: 10.0 }).unwrap();
        let e2 = world.spawn();
        world.insert(e2, Position { x: 2.0 }).unwrap();
        world.insert(e2, Velocity { x: 20.0 }).unwrap();

        {
            let q = query::<(Write<Position>, Write<Velocity>)>(&world);
            for (_, (pos, vel)) in q {
                pos.x += 100.0;
                vel.x += 100.0;
            }
        }

        assert_eq!(world.get::<Position>(e1).unwrap().x, 101.0);
        assert_eq!(world.get::<Velocity>(e1).unwrap().x, 110.0);
        assert_eq!(world.get::<Position>(e2).unwrap().x, 102.0);
        assert_eq!(world.get::<Velocity>(e2).unwrap().x, 120.0);
    }

    #[test]
    fn iter_with_resource() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        world.insert_resource(2.0f32); // speed multiplier

        let e1 = world.spawn();
        world.insert(e1, Position { x: 0.0 }).unwrap();
        world.insert(e1, Velocity { x: 3.0 }).unwrap();
        let e2 = world.spawn();
        world.insert(e2, Position { x: 0.0 }).unwrap();
        world.insert(e2, Velocity { x: 5.0 }).unwrap();

        {
            let q = query::<(Write<Position>, Read<Velocity>, Res<f32>)>(&world);
            for (_, (pos, vel, factor)) in q {
                pos.x += vel.x * *factor;
            }
        }

        assert_eq!(world.get::<Position>(e1).unwrap().x, 6.0);
        assert_eq!(world.get::<Position>(e2).unwrap().x, 10.0);
    }

    #[test]
    fn iter_into_guard_recovers_locks() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        let q = query::<(Write<Position>,)>(&world);
        let mut iter = QueryIter::from(q);

        // Partially consume the iterator
        let (_, (pos,)) = iter.next().unwrap();
        pos.x = 42.0;

        // Recover the guard — locks still held
        let guard = iter.into_guard();
        let (positions,) = &guard.items;
        assert_eq!(positions.get(e.index()).unwrap().x, 42.0);
    }

    #[test]
    fn iter_with_res_mut() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.insert_resource(0.0f32); // accumulator

        let e1 = world.spawn();
        world.insert(e1, Position { x: 3.0 }).unwrap();
        let e2 = world.spawn();
        world.insert(e2, Position { x: 7.0 }).unwrap();

        {
            let q = query::<(Read<Position>, ResMut<f32>)>(&world);
            for (_, (pos, mut acc)) in q {
                *acc += pos.x;
            }
        }

        let acc = world.resource::<f32>();
        assert_eq!(*acc, 10.0);
    }

    #[test]
    #[should_panic(expected = "already borrowed mutably by a previous iterator item")]
    fn iter_res_mut_detects_aliasing() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.insert_resource(0.0f32);

        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        let e2 = world.spawn();
        world.insert(e2, Position { x: 2.0 }).unwrap();

        let q = query::<(Read<Position>, ResMut<f32>)>(&world);
        let mut iter = QueryIter::from(q);
        let _a = iter.next().unwrap(); // holds ResMutRef
        let _b = iter.next().unwrap(); // panics: _a still alive
    }

    // ---- Bitset intersection tests ----

    #[test]
    fn iter_bitset_intersection_two_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        // 100 entities with Position only, 5 with both
        for i in 0..100 {
            let e = world.spawn();
            world.insert(e, Position { x: i as f32 }).unwrap();
        }
        for _ in 0..5 {
            let e = world.spawn();
            world.insert(e, Position { x: 999.0 }).unwrap();
            world.insert(e, Velocity { x: 1.0 }).unwrap();
        }

        let q = query::<(Read<Position>, Read<Velocity>)>(&world);
        let results: Vec<_> = QueryIter::from(q).collect();
        assert_eq!(results.len(), 5);
        for (_, (pos, _)) in &results {
            assert_eq!(pos.x, 999.0);
        }
    }

    #[test]
    fn iter_single_component_uses_fallback() {
        let mut world = World::new();
        world.register_component::<Position>();

        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        let q = query::<(Read<Position>,)>(&world);
        let iter = QueryIter::from(q);
        // Single component should NOT use intersection (no benefit)
        assert!(iter.intersected.is_none());
        let results: Vec<_> = iter.collect();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn iter_bitset_intersection_with_write() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        let e1 = world.spawn();
        world.insert(e1, Position { x: 10.0 }).unwrap();
        world.insert(e1, Velocity { x: 5.0 }).unwrap();

        let e2 = world.spawn();
        world.insert(e2, Position { x: 20.0 }).unwrap();

        {
            let q = query::<(Write<Position>, Read<Velocity>)>(&world);
            for (_, (pos, vel)) in q {
                pos.x += vel.x;
            }
        }

        assert_eq!(world.get::<Position>(e1).unwrap().x, 15.0);
        assert_eq!(world.get::<Position>(e2).unwrap().x, 20.0); // unchanged
    }

    #[test]
    fn iter_bitset_intersection_with_resource() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        world.insert_resource(2.0f32);

        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();
        world.insert(e, Velocity { x: 3.0 }).unwrap();

        // Only entity with only Position
        let e2 = world.spawn();
        world.insert(e2, Position { x: 100.0 }).unwrap();

        {
            let q = query::<(Write<Position>, Read<Velocity>, Res<f32>)>(&world);
            let iter = QueryIter::from(q);
            // 2 component bitsets (Position + Velocity), so intersection is used
            assert!(iter.intersected.is_some());
            for (_, (pos, vel, factor)) in iter {
                pos.x += vel.x * *factor;
            }
        }

        assert_eq!(world.get::<Position>(e).unwrap().x, 6.0);
        assert_eq!(world.get::<Position>(e2).unwrap().x, 100.0); // unchanged
    }

    // ---- par_for_each tests ----

    #[test]
    fn par_for_each_single_component_write() {
        let mut world = World::new();
        world.register_component::<Position>();
        for i in 0..1000 {
            let e = world.spawn();
            world.insert(e, Position { x: i as f32 }).unwrap();
        }

        {
            let q = query::<(Write<Position>,)>(&world);
            q.par_for_each(|_entity, (pos,)| {
                pos.x += 1.0;
            });
        }

        let q2 = query::<(Read<Position>,)>(&world);
        let mut count = 0;
        for (_, (pos,)) in q2 {
            assert!(pos.x >= 1.0);
            count += 1;
        }
        assert_eq!(count, 1000);
    }

    #[test]
    fn par_for_each_two_component_join() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        for i in 0..500 {
            let e = world.spawn();
            world.insert(e, Position { x: 0.0 }).unwrap();
            world
                .insert(
                    e,
                    Velocity {
                        x: (i as f32) * 0.1,
                    },
                )
                .unwrap();
        }
        // Entities without Velocity
        for _ in 0..500 {
            let e = world.spawn();
            world.insert(e, Position { x: -1.0 }).unwrap();
        }

        {
            let q = query::<(Write<Position>, Read<Velocity>)>(&world);
            q.par_for_each(|_entity, (pos, vel)| {
                pos.x += vel.x;
            });
        }

        let q2 = query::<(Read<Position>,)>(&world);
        for (_, (pos,)) in q2 {
            assert!(pos.x >= -1.0);
        }
    }

    #[test]
    fn par_for_each_with_resource() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.insert_resource(2.0f32);

        for _ in 0..100 {
            let e = world.spawn();
            world.insert(e, Position { x: 1.0 }).unwrap();
        }

        {
            let q = query::<(Write<Position>, Res<f32>)>(&world);
            q.par_for_each(|_entity, (pos, factor)| {
                pos.x *= *factor;
            });
        }

        let q2 = query::<(Read<Position>,)>(&world);
        for (_, (pos,)) in q2 {
            assert_eq!(pos.x, 2.0);
        }
    }

    #[test]
    fn par_for_each_accumulation_with_atomic() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let mut world = World::new();
        world.register_component::<Position>();
        for _ in 0..1000 {
            let e = world.spawn();
            world.insert(e, Position { x: 1.0 }).unwrap();
        }

        let counter = AtomicU32::new(0);
        let q = query::<(Read<Position>,)>(&world);
        q.par_for_each(|_entity, (_pos,)| {
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1000);
    }

    #[test]
    fn par_for_each_empty_set() {
        let mut world = World::new();
        world.register_component::<Position>();

        let q = query::<(Read<Position>,)>(&world);
        q.par_for_each(|_entity, (_pos,)| {
            panic!("should not be called");
        });
    }

    #[test]
    fn par_for_each_small_set() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let mut world = World::new();
        world.register_component::<Position>();
        for _ in 0..10 {
            let e = world.spawn();
            world.insert(e, Position { x: 1.0 }).unwrap();
        }

        let counter = AtomicU32::new(0);
        let q = query::<(Read<Position>,)>(&world);
        q.par_for_each(|_entity, (_pos,)| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }
}
