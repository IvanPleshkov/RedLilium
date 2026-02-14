use std::any::TypeId;
use std::marker::PhantomData;

use crate::query::{AddedFilter, ContainsChecker, RemovedFilter, With, Without};
use crate::resource::{ResourceRef, ResourceRefMut};
use crate::sparse_set::{Ref, RefMut};
use crate::world::World;

/// Metadata about a single component/resource access request.
#[derive(Debug, Clone, Copy)]
pub struct AccessInfo {
    pub type_id: TypeId,
    pub is_write: bool,
}

/// Normalizes access infos: sorts by TypeId and deduplicates, upgrading
/// to write if any duplicate requests write access.
pub(crate) fn normalize_access_infos(infos: &[AccessInfo]) -> Vec<AccessInfo> {
    let mut sorted = infos.to_vec();
    sorted.sort_by_key(|info| info.type_id);
    sorted.dedup_by(|a, b| {
        if a.type_id == b.type_id {
            b.is_write = b.is_write || a.is_write;
            true
        } else {
            false
        }
    });
    sorted
}

/// Trait for a single access element (Read, Write, Res, etc.).
///
/// Each element knows its TypeId, whether it's a write, and how to
/// fetch its data from a World.
pub trait AccessElement {
    /// The type received by the execute closure for this element.
    type Item<'w>;

    /// Returns the access metadata for this element.
    fn access_info() -> AccessInfo;

    /// Fetches this element's data from the world, acquiring per-storage locks.
    fn fetch(world: &World) -> Self::Item<'_>;

    /// Fetches this element's data without acquiring locks.
    ///
    /// The caller must ensure that the appropriate locks are already held
    /// externally (e.g. via `World::acquire_sorted`).
    fn fetch_unlocked(world: &World) -> Self::Item<'_>;

    /// Whether this element requires main-thread access.
    ///
    /// Returns `true` for [`MainThreadRes`] and [`MainThreadResMut`].
    /// When any element in an access set returns `true`, the entire
    /// `execute()` closure is dispatched to the main thread.
    fn needs_main_thread() -> bool {
        false
    }
}

/// Trait for a set of access elements (tuples of Read/Write/Res/etc.).
///
/// Implemented for tuples up to 8 elements via macro.
/// Provides sorted access metadata and batch fetching.
pub trait AccessSet {
    /// The tuple of items received by the execute closure.
    type Item<'w>;

    /// Returns access metadata for all elements.
    fn access_infos() -> Vec<AccessInfo>;

    /// Fetches all elements from the world, acquiring per-storage locks.
    fn fetch(world: &World) -> Self::Item<'_>;

    /// Fetches all elements without acquiring locks.
    ///
    /// The caller must ensure locks are already held externally.
    fn fetch_unlocked(world: &World) -> Self::Item<'_>;

    /// Returns `true` if any element in the set requires main-thread access.
    fn needs_main_thread() -> bool {
        false
    }
}

// ---- Marker types ----

/// Shared read access to component type `T`.
///
/// In the execute closure, yields `Ref<'_, T>` (deref to `SparseSetInner<T>`).
///
/// # Panics
///
/// Panics if `T` has never been registered.
pub struct Read<T: 'static>(PhantomData<T>);

/// Exclusive write access to component type `T`.
///
/// In the execute closure, yields `RefMut<'_, T>` (deref to `SparseSetInner<T>`).
///
/// # Panics
///
/// Panics if `T` has never been registered.
pub struct Write<T: 'static>(PhantomData<T>);

/// Optional shared read access to component type `T`.
///
/// In the execute closure, yields `Option<Ref<'_, T>>`.
/// Returns `None` if the type has never been registered (no panic).
pub struct OptionalRead<T: 'static>(PhantomData<T>);

/// Optional exclusive write access to component type `T`.
///
/// In the execute closure, yields `Option<RefMut<'_, T>>`.
/// Returns `None` if the type has never been registered (no panic).
pub struct OptionalWrite<T: 'static>(PhantomData<T>);

/// Shared read access to a resource of type `T`.
///
/// In the execute closure, yields `ResourceRef<'_, T>`.
///
/// # Panics
///
/// Panics if the resource does not exist.
pub struct Res<T: 'static>(PhantomData<T>);

/// Exclusive write access to a resource of type `T`.
///
/// In the execute closure, yields `ResourceRefMut<'_, T>`.
///
/// # Panics
///
/// Panics if the resource does not exist.
pub struct ResMut<T: 'static>(PhantomData<T>);

/// Shared read access to a main-thread resource of type `T`.
///
/// `T` does **not** need to be `Send + Sync`. The scheduler transparently
/// dispatches the `execute()` closure to the main thread when this type
/// is in the access set.
///
/// In the execute closure, yields `&T`.
///
/// # Panics
///
/// Panics if the main-thread resource does not exist.
pub struct MainThreadRes<T: 'static>(PhantomData<T>);

/// Exclusive write access to a main-thread resource of type `T`.
///
/// `T` does **not** need to be `Send + Sync`. The scheduler transparently
/// dispatches the `execute()` closure to the main thread when this type
/// is in the access set.
///
/// In the execute closure, yields `&mut T`.
///
/// # Panics
///
/// Panics if the main-thread resource does not exist.
pub struct MainThreadResMut<T: 'static>(PhantomData<T>);

/// Filter for entities whose component `T` was added this tick.
///
/// In the execute closure, yields [`AddedFilter`](crate::AddedFilter).
/// Use `filter.matches(entity_index)` to check individual entities.
///
/// # Panics
///
/// Panics if `T` has never been registered. Use [`MaybeAdded`] for
/// a non-panicking variant.
pub struct Added<T: 'static>(PhantomData<T>);

/// Filter for entities whose component `T` was removed this tick.
///
/// In the execute closure, yields [`RemovedFilter`](crate::RemovedFilter).
/// Use `filter.matches(entity_index)` or `filter.iter()` to query.
///
/// # Panics
///
/// Panics if `T` has never been registered. Use [`MaybeRemoved`] for
/// a non-panicking variant.
pub struct Removed<T: 'static>(PhantomData<T>);

/// Optional filter for entities whose component `T` was added this tick.
///
/// In the execute closure, yields [`AddedFilter`](crate::AddedFilter).
/// If `T` has never been registered, the filter matches nothing (no panic).
pub struct MaybeAdded<T: 'static>(PhantomData<T>);

/// Optional filter for entities whose component `T` was removed this tick.
///
/// In the execute closure, yields [`RemovedFilter`](crate::RemovedFilter).
/// If `T` has never been registered, the filter matches nothing (no panic).
pub struct MaybeRemoved<T: 'static>(PhantomData<T>);

// ---- AccessElement implementations ----

impl<T: 'static> AccessElement for Read<T> {
    type Item<'w> = Ref<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<T>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        world
            .read::<T>()
            .expect("Component not registered for Read<T> access")
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        world
            .read_unlocked::<T>()
            .expect("Component not registered for Read<T> access")
    }
}

impl<T: 'static> AccessElement for Write<T> {
    type Item<'w> = RefMut<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<T>(),
            is_write: true,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        world
            .write::<T>()
            .expect("Component not registered for Write<T> access")
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        world
            .write_unlocked::<T>()
            .expect("Component not registered for Write<T> access")
    }
}

impl<T: 'static> AccessElement for OptionalRead<T> {
    type Item<'w> = Option<Ref<'w, T>>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<T>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        world.try_read::<T>()
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        world.try_read_unlocked::<T>()
    }
}

impl<T: 'static> AccessElement for OptionalWrite<T> {
    type Item<'w> = Option<RefMut<'w, T>>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<T>(),
            is_write: true,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        world.try_write::<T>()
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        world.try_write_unlocked::<T>()
    }
}

impl<T: 'static> AccessElement for Res<T> {
    type Item<'w> = ResourceRef<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<T>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        world.resource::<T>()
    }

    /// Resources self-lock via `Arc<RwLock<T>>`, so fetch_unlocked
    /// behaves identically to fetch.
    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        world.resource::<T>()
    }
}

impl<T: 'static> AccessElement for ResMut<T> {
    type Item<'w> = ResourceRefMut<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<T>(),
            is_write: true,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        world.resource_mut::<T>()
    }

    /// Resources self-lock via `Arc<RwLock<T>>`, so fetch_unlocked
    /// behaves identically to fetch.
    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        world.resource_mut::<T>()
    }
}

impl<T: 'static> AccessElement for MainThreadRes<T> {
    type Item<'w> = &'w T;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<T>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        // SAFETY: only called from main thread via dispatcher
        unsafe { world.main_thread_resource::<T>() }
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        // Same as fetch — main-thread resources have no locks
        unsafe { world.main_thread_resource::<T>() }
    }

    fn needs_main_thread() -> bool {
        true
    }
}

impl<T: 'static> AccessElement for MainThreadResMut<T> {
    type Item<'w> = &'w mut T;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<T>(),
            is_write: true,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        // SAFETY: only called from main thread via dispatcher
        unsafe { world.main_thread_resource_mut::<T>() }
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        // Same as fetch — main-thread resources have no locks
        unsafe { world.main_thread_resource_mut::<T>() }
    }

    fn needs_main_thread() -> bool {
        true
    }
}

impl<T: 'static> AccessElement for Added<T> {
    type Item<'w> = AddedFilter<'w>;

    fn access_info() -> AccessInfo {
        // Use the marker type's own TypeId so it doesn't collide with
        // component storage and no lock is acquired.
        AccessInfo {
            type_id: TypeId::of::<Added<T>>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        assert!(
            world.is_component_registered::<T>(),
            "Component `{}` not registered for Added<T> filter",
            std::any::type_name::<T>()
        );
        let since_tick = world.current_tick().saturating_sub(1);
        world.added::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        // Filters don't hold locks — same as fetch
        Self::fetch(world)
    }
}

impl<T: 'static> AccessElement for Removed<T> {
    type Item<'w> = RemovedFilter<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<Removed<T>>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        assert!(
            world.is_component_registered::<T>(),
            "Component `{}` not registered for Removed<T> filter",
            std::any::type_name::<T>()
        );
        let since_tick = world.current_tick().saturating_sub(1);
        world.removed::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        Self::fetch(world)
    }
}

impl<T: 'static> AccessElement for MaybeAdded<T> {
    type Item<'w> = AddedFilter<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<MaybeAdded<T>>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        let since_tick = world.current_tick().saturating_sub(1);
        world.added::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        Self::fetch(world)
    }
}

impl<T: 'static> AccessElement for MaybeRemoved<T> {
    type Item<'w> = RemovedFilter<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<MaybeRemoved<T>>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        let since_tick = world.current_tick().saturating_sub(1);
        world.removed::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        Self::fetch(world)
    }
}

impl<T: 'static> AccessElement for With<T> {
    type Item<'w> = ContainsChecker<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<With<T>>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        world.with::<T>()
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        // Filters don't hold locks — same as fetch
        world.with::<T>()
    }
}

impl<T: 'static> AccessElement for Without<T> {
    type Item<'w> = ContainsChecker<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo {
            type_id: TypeId::of::<Without<T>>(),
            is_write: false,
        }
    }

    fn fetch(world: &World) -> Self::Item<'_> {
        world.without::<T>()
    }

    fn fetch_unlocked(world: &World) -> Self::Item<'_> {
        // Filters don't hold locks — same as fetch
        world.without::<T>()
    }
}

// ---- Tuple AccessSet implementations ----

// Empty tuple (no access)
impl AccessSet for () {
    type Item<'w> = ();

    fn access_infos() -> Vec<AccessInfo> {
        Vec::new()
    }

    fn fetch(_world: &World) -> Self::Item<'_> {}

    fn fetch_unlocked(_world: &World) -> Self::Item<'_> {}

    fn needs_main_thread() -> bool {
        false
    }
}

macro_rules! impl_access_set {
    ($($idx:tt $T:ident),+) => {
        impl<$($T: AccessElement),+> AccessSet for ($($T,)+) {
            type Item<'w> = ($($T::Item<'w>,)+);

            fn access_infos() -> Vec<AccessInfo> {
                vec![$($T::access_info()),+]
            }

            fn fetch(world: &World) -> Self::Item<'_> {
                ($($T::fetch(world),)+)
            }

            fn fetch_unlocked(world: &World) -> Self::Item<'_> {
                ($($T::fetch_unlocked(world),)+)
            }

            fn needs_main_thread() -> bool {
                $($T::needs_main_thread())||+
            }
        }
    };
}

impl_access_set!(0 A);
impl_access_set!(0 A, 1 B);
impl_access_set!(0 A, 1 B, 2 C);
impl_access_set!(0 A, 1 B, 2 C, 3 D);
impl_access_set!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_access_set!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_access_set!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_access_set!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);

#[cfg(test)]
mod tests {
    use super::*;

    struct Position {
        x: f32,
    }
    struct Velocity {
        _x: f32,
    }

    #[test]
    fn read_access_info() {
        let info = <Read<Position>>::access_info();
        assert_eq!(info.type_id, TypeId::of::<Position>());
        assert!(!info.is_write);
    }

    #[test]
    fn write_access_info() {
        let info = <Write<Position>>::access_info();
        assert_eq!(info.type_id, TypeId::of::<Position>());
        assert!(info.is_write);
    }

    #[test]
    fn tuple_access_infos() {
        let infos = <(Read<Position>, Write<Velocity>)>::access_infos();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].type_id, TypeId::of::<Position>());
        assert!(!infos[0].is_write);
        assert_eq!(infos[1].type_id, TypeId::of::<Velocity>());
        assert!(infos[1].is_write);
    }

    #[test]
    fn empty_tuple() {
        let infos = <()>::access_infos();
        assert!(infos.is_empty());
    }

    #[test]
    fn fetch_reads_from_world() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();
        world.insert(e, Velocity { _x: 5.0 }).unwrap();

        let (positions, velocities) = <(Read<Position>, Read<Velocity>)>::fetch(&world);
        assert_eq!(positions.len(), 1);
        assert_eq!(velocities.len(), 1);
    }

    #[test]
    fn fetch_write_from_world() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();

        let (mut positions,) = <(Write<Position>,)>::fetch(&world);
        for (_, pos) in positions.iter_mut() {
            pos.x = 99.0;
        }
        drop(positions);

        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn optional_read_returns_none_for_unregistered() {
        let world = World::new();
        let (opt,) = <(OptionalRead<Position>,)>::fetch(&world);
        assert!(opt.is_none());
    }

    #[test]
    fn optional_read_returns_some_for_registered() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        let (opt,) = <(OptionalRead<Position>,)>::fetch(&world);
        assert!(opt.is_some());
        assert_eq!(opt.unwrap().len(), 1);
    }

    #[test]
    fn res_fetch_from_world() {
        let mut world = World::new();
        world.insert_resource(1.5f64);

        let (dt,) = <(Res<f64>,)>::fetch(&world);
        assert_eq!(*dt, 1.5);
    }

    // ---- Added/Removed filter tests ----

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    #[test]
    fn added_filter_access_info_uses_marker_type() {
        let info = <Added<Position>>::access_info();
        // TypeId should be Added<Position>, not Position itself
        assert_ne!(info.type_id, TypeId::of::<Position>());
        assert_eq!(info.type_id, TypeId::of::<Added<Position>>());
        assert!(!info.is_write);
    }

    #[test]
    fn removed_filter_access_info_uses_marker_type() {
        let info = <Removed<Position>>::access_info();
        assert_ne!(info.type_id, TypeId::of::<Position>());
        assert_eq!(info.type_id, TypeId::of::<Removed<Position>>());
        assert!(!info.is_write);
    }

    #[test]
    fn added_filter_detects_addition() {
        let mut world = World::new();
        world.register_component::<Health>();

        world.advance_tick(); // tick = 1
        let e = world.spawn();
        world.insert_tracked(e, Health(100)).unwrap();

        let (filter,) = <(Added<Health>,)>::fetch(&world);
        assert!(filter.matches(e.index()));
    }

    #[test]
    fn added_filter_does_not_match_old() {
        let mut world = World::new();
        world.register_component::<Health>();

        let e = world.spawn();
        world.insert_tracked(e, Health(100)).unwrap(); // tick 0

        world.advance_tick(); // tick = 1
        world.advance_tick(); // tick = 2

        // since_tick = 2 - 1 = 1, component was added at tick 0, so 0 > 1 is false
        let (filter,) = <(Added<Health>,)>::fetch(&world);
        assert!(!filter.matches(e.index()));
    }

    #[test]
    fn removed_filter_detects_removal() {
        let mut world = World::new();
        world.register_component::<Health>();

        let e = world.spawn();
        world.insert(e, Health(100)).unwrap();

        world.advance_tick(); // tick = 1
        world.remove::<Health>(e); // removed at tick 1

        let (filter,) = <(Removed<Health>,)>::fetch(&world);
        assert!(filter.matches(e.index()));
    }

    #[test]
    fn removed_filter_iter_in_tuple() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Health(100)).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();
        world.insert(e2, Health(200)).unwrap();

        world.advance_tick(); // tick = 1
        world.remove::<Health>(e1); // removed at tick 1

        let (positions, removed) = <(Read<Position>, Removed<Health>)>::fetch(&world);
        let affected: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| removed.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        assert_eq!(affected, vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "not registered for Added")]
    fn added_panics_for_unregistered() {
        let world = World::new();
        let _ = <(Added<Health>,)>::fetch(&world);
    }

    #[test]
    #[should_panic(expected = "not registered for Removed")]
    fn removed_panics_for_unregistered() {
        let world = World::new();
        let _ = <(Removed<Health>,)>::fetch(&world);
    }

    #[test]
    fn maybe_added_no_panic_for_unregistered() {
        let world = World::new();
        let (filter,) = <(MaybeAdded<Health>,)>::fetch(&world);
        assert!(!filter.matches(0));
    }

    #[test]
    fn maybe_removed_no_panic_for_unregistered() {
        let world = World::new();
        let (filter,) = <(MaybeRemoved<Health>,)>::fetch(&world);
        assert!(!filter.matches(0));
    }

    #[test]
    fn maybe_added_works_when_registered() {
        let mut world = World::new();
        world.register_component::<Health>();

        world.advance_tick(); // tick = 1
        let e = world.spawn();
        world.insert_tracked(e, Health(50)).unwrap();

        let (filter,) = <(MaybeAdded<Health>,)>::fetch(&world);
        assert!(filter.matches(e.index()));
    }

    #[test]
    fn maybe_removed_works_when_registered() {
        let mut world = World::new();
        world.register_component::<Health>();

        let e = world.spawn();
        world.insert(e, Health(50)).unwrap();

        world.advance_tick(); // tick = 1
        world.remove::<Health>(e);

        let (filter,) = <(MaybeRemoved<Health>,)>::fetch(&world);
        assert!(filter.matches(e.index()));
    }

    // ---- With/Without filter tests ----

    #[derive(Debug, PartialEq)]
    struct Frozen;

    #[test]
    fn with_access_info_uses_marker_type() {
        let info = <With<Position>>::access_info();
        assert_ne!(info.type_id, TypeId::of::<Position>());
        assert_eq!(info.type_id, TypeId::of::<With<Position>>());
        assert!(!info.is_write);
    }

    #[test]
    fn without_access_info_uses_marker_type() {
        let info = <Without<Position>>::access_info();
        assert_ne!(info.type_id, TypeId::of::<Position>());
        assert_eq!(info.type_id, TypeId::of::<Without<Position>>());
        assert!(!info.is_write);
    }

    #[test]
    fn with_filter_in_tuple() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Frozen>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Frozen).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();

        let (positions, has_frozen) = <(Read<Position>, With<Frozen>)>::fetch(&world);
        let matched: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| has_frozen.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        assert_eq!(matched, vec![1.0]);
    }

    #[test]
    fn without_filter_in_tuple() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Frozen>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Frozen).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();

        let (positions, not_frozen) = <(Read<Position>, Without<Frozen>)>::fetch(&world);
        let matched: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| not_frozen.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        assert_eq!(matched, vec![2.0]);
    }

    #[test]
    fn without_unregistered_matches_everything() {
        let mut world = World::new();
        world.register_component::<Position>();

        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        // Frozen never registered — Without<Frozen> matches all entities
        let (positions, not_frozen) = <(Read<Position>, Without<Frozen>)>::fetch(&world);
        let count = positions
            .iter()
            .filter(|(idx, _)| not_frozen.matches(*idx))
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn with_unregistered_matches_nothing() {
        let mut world = World::new();
        world.register_component::<Position>();

        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        // Frozen never registered — With<Frozen> matches no entities
        let (positions, has_frozen) = <(Read<Position>, With<Frozen>)>::fetch(&world);
        let count = positions
            .iter()
            .filter(|(idx, _)| has_frozen.matches(*idx))
            .count();
        assert_eq!(count, 0);
    }
}
