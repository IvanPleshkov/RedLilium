use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use fixedbitset::FixedBitSet;
use smallvec::SmallVec;

use crate::entity::Entities;
use crate::query::AccessSet;
use crate::resource::{ResourceRef, ResourceRefMut};
use crate::sparse_set::{LockGuard, Mut, Ref, RefMut, SparseSetInner};
use crate::system::context::LockTracking;

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
/// Access fetched data via [`items`](Self::items) / [`items_mut`](Self::items_mut):
///
/// ```ignore
/// // Read-only:
/// let q = ctx.query::<(Read<Position>, Read<Velocity>)>();
/// let (positions, velocities) = q.items();
///
/// // With writes:
/// let mut q = ctx.query::<(Write<Position>, Read<Velocity>)>();
/// let (positions, velocities) = q.items_mut();
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
    _guards: SmallVec<[LockGuard<'a>; 8]>,
    /// The fetched component/resource data. Private: the unlocked `Ref`s /
    /// `RefMut`s in here are only kept alive by `_guards`, so moving them out
    /// of the guard (possible through a public field) would dangle once the
    /// guard drops. Access goes through [`items`](Self::items) /
    /// [`items_mut`](Self::items_mut), which tie the borrow to the guard.
    items: A::Item<'a>,
    /// Deadlock tracking — unregisters held locks when this guard is dropped.
    /// `None` when created outside of a SystemContext (e.g. in tests).
    _tracking: Option<LockTracking<'a>>,
}

impl<'a, A: AccessSet> QueryGuard<'a, A> {
    /// Creates a guard without deadlock tracking.
    ///
    /// Used by [`World::query`](crate::World::query) (the world owner is
    /// exclusive, so same-system lock tracking does not apply) and by tests.
    pub(crate) fn new(guards: SmallVec<[LockGuard<'a>; 8]>, items: A::Item<'a>) -> Self {
        Self {
            _guards: guards,
            items,
            _tracking: None,
        }
    }

    pub(crate) fn new_tracked(
        guards: SmallVec<[LockGuard<'a>; 8]>,
        items: A::Item<'a>,
        tracking: LockTracking<'a>,
    ) -> Self {
        Self {
            _guards: guards,
            items,
            _tracking: Some(tracking),
        }
    }

    /// Returns the fetched data, borrowed for the guard's lifetime.
    ///
    /// Destructure the tuple to access individual storages.
    pub fn items(&self) -> &A::Item<'a> {
        &self.items
    }

    /// Mutable variant of [`items`](Self::items).
    pub fn items_mut(&mut self) -> &mut A::Item<'a> {
        &mut self.items
    }
}

impl<'a, A: AccessSet> QueryGuard<'a, A>
where
    A::Item<'a>: QueryItem,
{
    /// Iterates over matching entities (inner join across all queried
    /// storages), read-only variant.
    ///
    /// Only available when every element of the access set is read-only
    /// ([`ReadOnlyQueryItem`]); multiple concurrent `iter()` calls on the
    /// same guard are then harmless. Yielded items borrow the guard, so
    /// they cannot outlive it.
    pub fn iter(&self) -> QueryIter<'_, 'a, A>
    where
        A::Item<'a>: ReadOnlyQueryItem,
    {
        QueryIter::new(&self.items)
    }

    /// Iterates over matching entities (inner join across all queried
    /// storages), yielding mutable items for `Write`/`ResMut` elements.
    ///
    /// Takes `&mut self` so only one iterator (and its items) can exist at
    /// a time — yielded items borrow the guard and cannot outlive it:
    ///
    /// ```ignore
    /// let mut q = ctx.query::<(Write<Position>, Read<Velocity>)>();
    /// for (entity_idx, (pos, vel)) in q.iter_mut() {
    ///     pos.x += vel.x;
    /// }
    /// ```
    ///
    /// Dropping the guard while collected items are alive is rejected at
    /// compile time (the items borrow the guard, which keeps its locks
    /// held):
    ///
    /// ```compile_fail
    /// use redlilium_ecs::{World, Write};
    ///
    /// struct Position { x: f32 }
    ///
    /// let mut world = World::new();
    /// world.register_component::<Position>();
    /// let e = world.spawn();
    /// world.insert(e, Position { x: 1.0 }).unwrap();
    ///
    /// let mut q = world.query::<(Write<Position>,)>();
    /// let items: Vec<_> = q.iter_mut().collect();
    /// drop(q); // ERROR: `q` is still borrowed by `items`
    /// let _ = &items;
    /// ```
    pub fn iter_mut(&mut self) -> QueryIter<'_, 'a, A> {
        QueryIter::new(&self.items)
    }

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
    /// Takes `&mut self` for the same reason as [`iter_mut`]
    /// (Self::iter_mut): items handed to `f` are tied to this borrow and
    /// cannot alias a later iteration.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut q = ctx.query::<(Write<Position>, Read<Velocity>)>();
    /// q.par_for_each(|entity_idx, (pos, vel)| {
    ///     pos.x += vel.x;
    /// });
    /// ```
    pub fn par_for_each<'g, F>(&'g mut self, f: F)
    where
        A::Item<'a>: Sync,
        F: Fn(u32, <A::Item<'a> as QueryItem>::Item<'g>) + Sync,
    {
        self.par_for_each_with(crate::system::par_for_each::ParConfig::default(), f);
    }

    /// Like [`par_for_each`](Self::par_for_each), but with explicit
    /// parallelism configuration.
    pub fn par_for_each_with<'g, F>(
        &'g mut self,
        config: crate::system::par_for_each::ParConfig,
        f: F,
    ) where
        A::Item<'a>: Sync,
        F: Fn(u32, <A::Item<'a> as QueryItem>::Item<'g>) + Sync,
    {
        if let Some(intersected) = self.items.query_intersected_entities() {
            crate::system::par_for_each::par_for_each_entities(
                &self.items,
                &intersected,
                &config,
                &f,
            );
        } else {
            crate::system::par_for_each::par_for_each_entities(
                &self.items,
                self.items.query_entities(),
                &config,
                &f,
            );
        }
    }
}

impl<'g, 'a, A: AccessSet> IntoIterator for &'g QueryGuard<'a, A>
where
    A::Item<'a>: ReadOnlyQueryItem,
{
    type Item = (u32, <A::Item<'a> as QueryItem>::Item<'g>);
    type IntoIter = QueryIter<'g, 'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'g, 'a, A: AccessSet> IntoIterator for &'g mut QueryGuard<'a, A>
where
    A::Item<'a>: QueryItem,
{
    type Item = (u32, <A::Item<'a> as QueryItem>::Item<'g>);
    type IntoIter = QueryIter<'g, 'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
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
    /// The per-entity reference type at lifetime `'q` (e.g., `&'q T` or
    /// [`Mut<'q, T>`]). The lifetime is chosen by the `query_get` caller;
    /// safe wrappers tie it to a borrow of the owning [`QueryGuard`].
    type Item<'q>;

    /// Number of entities in this storage.
    fn query_count(&self) -> usize;

    /// Entity indices in dense order.
    fn query_entities(&self) -> &[u32];

    /// Fetch the item for a given entity index.
    ///
    /// # Safety
    ///
    /// The caller chooses `'q` and must guarantee:
    /// - the lock/borrow backing `self` stays held for all of `'q` (safe
    ///   wrappers achieve this by tying `'q` to a borrow of the
    ///   [`QueryGuard`] that owns the locks);
    /// - for mutable items, each `entity_index` is fetched at most once
    ///   while previous items are alive (no aliasing mutable references).
    unsafe fn query_get<'q>(&self, entity_index: u32) -> Option<Self::Item<'q>>;

    /// Returns the component membership bitset, if available.
    ///
    /// Returns `Some` for component storages (`Ref`/`RefMut`), `None` for
    /// resources (which are singletons and always match).
    fn query_membership(&self) -> Option<&FixedBitSet> {
        None
    }

    /// Returns the entities reference, if available.
    ///
    /// Returns `Some` for component storages, `None` for resources.
    #[doc(hidden)]
    fn query_entities_ref(&self) -> Option<&Entities> {
        None
    }

    /// Returns the exclude mask used by this query item, if available.
    ///
    /// Returns `Some` for component storages (`Ref`/`RefMut`), `None` for
    /// resources. Used by `query_intersected_entities()` to combine masks
    /// from all elements in a tuple query.
    fn query_exclude_mask(&self) -> Option<u32> {
        None
    }

    /// Pre-computes the set of entity indices that match all component storages.
    ///
    /// Returns `Some(entities)` when there are 2+ component memberships to
    /// intersect, `None` otherwise (single component or resources only).
    /// The returned vec contains only entities present in all storages and
    /// not excluded (disabled/static).
    fn query_intersected_entities(&self) -> Option<Vec<u32>> {
        None
    }
}

impl<'w, T: 'static> QueryItem for Ref<'w, T> {
    type Item<'q> = &'q T;

    fn query_count(&self) -> usize {
        self.len()
    }

    fn query_entities(&self) -> &[u32] {
        self.entities()
    }

    unsafe fn query_get<'q>(&self, entity_index: u32) -> Option<&'q T> {
        if self.is_entity_excluded(entity_index) {
            return None;
        }
        // SAFETY: the caller guarantees the read lock stays held for 'q,
        // so extending the storage borrow to 'q cannot dangle.
        self.storage()
            .get(entity_index)
            .map(|v| unsafe { &*(v as *const T) })
    }

    fn query_membership(&self) -> Option<&FixedBitSet> {
        Some(self.storage().membership())
    }

    fn query_entities_ref(&self) -> Option<&Entities> {
        Some(self.entities_ref())
    }

    fn query_exclude_mask(&self) -> Option<u32> {
        Some(self.exclude_mask())
    }
}

impl<'w, T: 'static> QueryItem for RefMut<'w, T> {
    type Item<'q> = Mut<'q, T>;

    fn query_count(&self) -> usize {
        self.len()
    }

    fn query_entities(&self) -> &[u32] {
        self.entities()
    }

    unsafe fn query_get<'q>(&self, entity_index: u32) -> Option<Mut<'q, T>> {
        if self.is_entity_excluded(entity_index) {
            return None;
        }
        // SAFETY: the caller guarantees the write lock stays held for 'q and
        // that each entity_index is fetched at most once, so no aliasing
        // mutable references are created.
        unsafe {
            SparseSetInner::get_ptr_mut_with_tick(self.storage_ptr(), entity_index)
                .map(|(val_ptr, tick_ptr)| Mut::from_raw(val_ptr, tick_ptr, self.query_tick()))
        }
    }

    fn query_membership(&self) -> Option<&FixedBitSet> {
        // SAFETY: write lock guarantees exclusive access.
        Some(unsafe { &*self.storage_ptr() }.membership())
    }

    fn query_entities_ref(&self) -> Option<&Entities> {
        Some(self.entities_ref())
    }

    fn query_exclude_mask(&self) -> Option<u32> {
        Some(self.exclude_mask())
    }
}

impl<'w, T: 'static> QueryItem for ResourceRef<'w, T> {
    type Item<'q> = &'q T;

    fn query_count(&self) -> usize {
        // Resources are singletons — never the smallest set, so they
        // never drive iteration.
        usize::MAX
    }

    fn query_entities(&self) -> &[u32] {
        // Never selected (count is MAX), but must return a valid slice.
        &[]
    }

    unsafe fn query_get<'q>(&self, _entity_index: u32) -> Option<&'q T> {
        // SAFETY: the caller guarantees the RwLockReadGuard inside
        // ResourceRef stays held for 'q. Multiple shared references are
        // safe (read-only).
        unsafe {
            let ptr: *const T = &**self;
            Some(&*ptr)
        }
    }
}

impl<'w, T: 'static> QueryItem for ResourceRefMut<'w, T> {
    type Item<'q> = ResMutRef<'q, T>;

    fn query_count(&self) -> usize {
        usize::MAX
    }

    fn query_entities(&self) -> &[u32] {
        &[]
    }

    unsafe fn query_get<'q>(&self, _entity_index: u32) -> Option<ResMutRef<'q, T>> {
        assert!(
            !self.borrowed.load(Ordering::Relaxed),
            "ResMut<{}> already borrowed mutably by a previous iterator item. \
             Drop the previous item before calling next().",
            std::any::type_name::<T>()
        );
        self.borrowed.store(true, Ordering::Relaxed);
        // SAFETY: the RwLockWriteGuard inside ResourceRefMut keeps exclusive
        // access for 'w. The borrow flag ensures only one ResMutRef exists
        // at a time, preventing aliasing &mut T. The flag is shared via Arc,
        // so it stays valid even if the guard (and this ResourceRefMut) is
        // moved or dropped while the ResMutRef is still alive.
        unsafe {
            Some(ResMutRef {
                ptr: &mut *self.as_ptr_mut(),
                flag: Arc::clone(&self.borrowed),
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
    /// Shared with the originating [`ResourceRefMut`]; the `Arc` keeps the
    /// flag alive independently of guard moves and drop order.
    flag: Arc<AtomicBool>,
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
        self.flag.store(false, Ordering::Relaxed);
    }
}

/// Marker for query items that only ever hand out shared references.
///
/// Implemented for [`Ref`], [`ResourceRef`], and tuples composed entirely
/// of read-only items.
///
/// # Safety
///
/// Implementors must guarantee that `query_get` never produces mutable
/// access. [`QueryGuard::iter`] relies on this: it allows multiple
/// concurrent iterators from a shared guard borrow, which would alias
/// mutable references for a non-read-only item.
pub unsafe trait ReadOnlyQueryItem: QueryItem {}

// SAFETY: Ref/ResourceRef items are `&T` — read-only.
unsafe impl<T: 'static> ReadOnlyQueryItem for Ref<'_, T> {}
// SAFETY: see above.
unsafe impl<T: 'static> ReadOnlyQueryItem for ResourceRef<'_, T> {}

macro_rules! impl_query_item {
    ($($idx:tt $T:ident),+) => {
        // SAFETY: a tuple is read-only iff every element is.
        unsafe impl<$($T: ReadOnlyQueryItem),+> ReadOnlyQueryItem for ($($T,)+) {}

        impl<$($T: QueryItem),+> QueryItem for ($($T,)+) {
            type Item<'q> = ($($T::Item<'q>,)+);

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

            unsafe fn query_get<'q>(&self, entity_index: u32) -> Option<Self::Item<'q>> {
                // SAFETY: delegates to each element's query_get with the same
                // entity_index and 'q. The caller upholds the contract.
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
                // Combine exclude masks from all elements (OR = most restrictive).
                let mask = 0u32 $(| self.$idx.query_exclude_mask().unwrap_or(0))+;
                // Filter out excluded entities via flag bits.
                let entities = None $(.or(self.$idx.query_entities_ref()))+;
                if let Some(entities) = entities {
                    Some(result.ones().filter(|&i| {
                        i >= entities.slots_len() || entities.get_flags(i as u32) & mask == 0
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
/// Created via [`QueryGuard::iter`] / [`QueryGuard::iter_mut`] (or
/// `IntoIterator` on `&guard` / `&mut guard`). Performs an inner join:
/// iterates over the smallest component storage and yields only entities
/// present in all queried storages.
///
/// Borrows the [`QueryGuard`], and yielded items carry that borrow's
/// lifetime — the guard (and thus its locks) cannot be dropped while the
/// iterator or any yielded item is alive. `collect()`-ing items is safe:
/// the collection keeps the guard borrowed.
///
/// ```ignore
/// let mut q = ctx.query::<(Write<Position>, Read<Velocity>)>();
/// for (entity_idx, (pos, vel)) in q.iter_mut() {
///     pos.x += vel.x;
/// }
/// ```
pub struct QueryIter<'g, 'a, A: AccessSet> {
    /// Borrow of the guard's fetched data. The guard outlives 'g, keeping
    /// the locks held for as long as this iterator or its items exist.
    items: &'g A::Item<'a>,
    /// Pre-computed matching entity indices (from bitset intersection),
    /// or `None` to fall back to the smallest-set iteration path.
    intersected: Option<Vec<u32>>,
    idx: usize,
}

impl<'g, 'a, A: AccessSet> QueryIter<'g, 'a, A>
where
    A::Item<'a>: QueryItem,
{
    fn new(items: &'g A::Item<'a>) -> Self {
        let intersected = items.query_intersected_entities();
        Self {
            items,
            intersected,
            idx: 0,
        }
    }
}

impl<'g, 'a, A: AccessSet> Iterator for QueryIter<'g, 'a, A>
where
    A::Item<'a>: QueryItem,
{
    type Item = (u32, <A::Item<'a> as QueryItem>::Item<'g>);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref entities) = self.intersected {
            // Bitset-accelerated path: every entity is guaranteed to match.
            while self.idx < entities.len() {
                let entity_idx = entities[self.idx];
                self.idx += 1;
                // SAFETY: the guard is borrowed for 'g, so its locks stay
                // held for 'g. Bitset intersection guarantees the entity has
                // all components and is not disabled. Each entity is visited
                // once (monotonically increasing idx).
                if let Some(item) = unsafe { self.items.query_get(entity_idx) } {
                    return Some((entity_idx, item));
                }
            }
        } else {
            // Fallback: walk the smallest set and probe other storages.
            let entities = self.items.query_entities();
            while self.idx < entities.len() {
                let entity_idx = entities[self.idx];
                self.idx += 1;
                // SAFETY: the guard is borrowed for 'g, so its locks stay
                // held for 'g. The iterator visits each entity exactly once
                // (monotonically increasing idx), so no aliasing mutable
                // references are created across calls to next().
                if let Some(item) = unsafe { self.items.query_get(entity_idx) } {
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
            self.items.query_entities()
        };
        let remaining = entities.len().saturating_sub(self.idx);
        (0, Some(remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{AccessSet, Read, Res, ResMut, Write};
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
        let ticks = crate::query::FetchTicks::frame(world);
        let guards = world.acquire_sorted(&A::access_infos());
        let items = A::fetch_unlocked(world, ticks);
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
            for (_, mut pos) in positions.iter_mut() {
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
            for (idx, mut pos) in positions.iter_mut() {
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
            for (_, mut pos) in positions.iter_mut() {
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
            for (_, mut pos) in positions.iter_mut() {
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
        for (_, (pos,)) in q.iter() {
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
            let mut q = query::<(Write<Position>,)>(&world);
            for (_, (mut pos,)) in q.iter_mut() {
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
            let mut q = query::<(Write<Position>, Read<Velocity>)>(&world);
            let mut count = 0;
            for (_, (mut pos, vel)) in q.iter_mut() {
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
        let results: Vec<_> = q.iter().map(|(idx, (p, _))| (idx, p.x)).collect();
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
        assert_eq!(q.iter().count(), 0);
    }

    #[test]
    fn iter_into_iterator() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        let mut q = query::<(Read<Position>,)>(&world);
        let mut found = false;
        // IntoIterator on &guard (read-only)...
        for (_, (pos,)) in &q {
            assert_eq!(pos.x, 42.0);
            found = true;
        }
        assert!(found);
        // ...and on &mut guard.
        let mut found_mut = false;
        for (_, (pos,)) in &mut q {
            assert_eq!(pos.x, 42.0);
            found_mut = true;
        }
        assert!(found_mut);
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
            let mut q = query::<(Write<Position>, Write<Velocity>)>(&world);
            for (_, (mut pos, mut vel)) in q.iter_mut() {
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
            let mut q = query::<(Write<Position>, Read<Velocity>, Res<f32>)>(&world);
            for (_, (mut pos, vel, factor)) in q.iter_mut() {
                pos.x += vel.x * *factor;
            }
        }

        assert_eq!(world.get::<Position>(e1).unwrap().x, 6.0);
        assert_eq!(world.get::<Position>(e2).unwrap().x, 10.0);
    }

    #[test]
    fn iter_partial_then_guard_still_usable() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        let mut q = query::<(Write<Position>,)>(&world);

        // Partially consume an iterator, then drop it (locks stay held by
        // the guard).
        let mut iter = q.iter_mut();
        let (_, (mut pos,)) = iter.next().unwrap();
        pos.x = 42.0;
        drop(iter);

        // The guard is usable again after the iterator borrow ends.
        let (positions,) = q.items();
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
            let mut q = query::<(Read<Position>, ResMut<f32>)>(&world);
            for (_, (pos, mut acc)) in q.iter_mut() {
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

        let mut q = query::<(Read<Position>, ResMut<f32>)>(&world);
        let mut iter = q.iter_mut();
        let _a = iter.next().unwrap(); // holds ResMutRef
        let _b = iter.next().unwrap(); // panics: _a still alive
    }

    #[test]
    fn iter_collected_items_stay_valid() {
        let mut world = World::new();
        world.register_component::<Position>();
        for i in 0..4 {
            let e = world.spawn();
            world.insert(e, Position { x: i as f32 }).unwrap();
        }

        let mut q = query::<(Write<Position>,)>(&world);
        // Collecting mutable items is safe: they borrow `q`, so the guard
        // (and its locks) cannot be dropped while they are alive.
        let mut items: Vec<_> = q.iter_mut().collect();
        for (_, (pos,)) in &mut items {
            pos.x += 10.0;
        }
        drop(items);

        let sum: f32 = q.items().0.iter().map(|(_, p)| p.x).sum();
        assert_eq!(sum, 0.0 + 1.0 + 2.0 + 3.0 + 40.0);
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
        let results: Vec<_> = q.iter().collect();
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
        let iter = q.iter();
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
            let mut q = query::<(Write<Position>, Read<Velocity>)>(&world);
            for (_, (mut pos, vel)) in q.iter_mut() {
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
            let mut q = query::<(Write<Position>, Read<Velocity>, Res<f32>)>(&world);
            let iter = q.iter_mut();
            // 2 component bitsets (Position + Velocity), so intersection is used
            assert!(iter.intersected.is_some());
            for (_, (mut pos, vel, factor)) in iter {
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
            let mut q = query::<(Write<Position>,)>(&world);
            q.par_for_each(|_entity, (mut pos,)| {
                pos.x += 1.0;
            });
        }

        let q2 = query::<(Read<Position>,)>(&world);
        let mut count = 0;
        for (_, (pos,)) in q2.iter() {
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
            let mut q = query::<(Write<Position>, Read<Velocity>)>(&world);
            q.par_for_each(|_entity, (mut pos, vel)| {
                pos.x += vel.x;
            });
        }

        let q2 = query::<(Read<Position>,)>(&world);
        for (_, (pos,)) in q2.iter() {
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
            let mut q = query::<(Write<Position>, Res<f32>)>(&world);
            q.par_for_each(|_entity, (mut pos, factor)| {
                pos.x *= *factor;
            });
        }

        let q2 = query::<(Read<Position>,)>(&world);
        for (_, (pos,)) in q2.iter() {
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
        let mut q = query::<(Read<Position>,)>(&world);
        q.par_for_each(|_entity, (_pos,)| {
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1000);
    }

    #[test]
    fn par_for_each_empty_set() {
        let mut world = World::new();
        world.register_component::<Position>();

        let mut q = query::<(Read<Position>,)>(&world);
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
        let mut q = query::<(Read<Position>,)>(&world);
        q.par_for_each(|_entity, (_pos,)| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }
}
