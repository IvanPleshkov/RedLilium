use std::any::TypeId;
use std::marker::PhantomData;

use crate::resource::{ResourceRef, ResourceRefMut};
use crate::sparse_set::{Ref, RefMut};
use crate::world::World;

/// Metadata about a single component/resource access request.
#[derive(Debug, Clone, Copy)]
pub struct AccessInfo {
    pub type_id: TypeId,
    pub is_write: bool,
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

    /// Fetches this element's data from the world.
    fn fetch(world: &World) -> Self::Item<'_>;
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

    /// Fetches all elements from the world.
    fn fetch(world: &World) -> Self::Item<'_>;
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
        world.read::<T>()
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
        world.write::<T>()
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
}

impl<T: Send + Sync + 'static> AccessElement for Res<T> {
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
}

impl<T: Send + Sync + 'static> AccessElement for ResMut<T> {
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
}

// ---- Tuple AccessSet implementations ----

// Empty tuple (no access)
impl AccessSet for () {
    type Item<'w> = ();

    fn access_infos() -> Vec<AccessInfo> {
        Vec::new()
    }

    fn fetch(_world: &World) -> Self::Item<'_> {}
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
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 });
        world.insert(e, Velocity { _x: 5.0 });

        let (positions, velocities) = <(Read<Position>, Read<Velocity>)>::fetch(&world);
        assert_eq!(positions.len(), 1);
        assert_eq!(velocities.len(), 1);
    }

    #[test]
    fn fetch_write_from_world() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 });

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
        let e = world.spawn();
        world.insert(e, Position { x: 1.0 });

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
}
