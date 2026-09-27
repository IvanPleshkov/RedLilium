//! Short-lived storage views returned by QueryGuard.

use super::{AddedFilter, AnyFilter, ChangedFilter, ContainsChecker, OrFilter, RemovedFilter};
use crate::{Ref, RefMut, ResourceRef, ResourceRefMut};

/// Reborrows fetched query data without moving it out of its lock owner.
///
/// The returned views borrow `self`, rather than inheriting the original
/// world's lifetime. Components retain their filtering and change tracking;
/// resources become ordinary references. Options and tuples preserve their
/// shape. No locks, allocations, or reference-count updates are needed.
///
/// Custom access sets can implement this trait on their fetched item types
/// to support [`QueryGuard::items`](crate::QueryGuard::items).
///
/// # Safety
///
/// Returned views must keep every access to externally locked data tied to
/// the borrow of `self`, never expose the original world's longer lifetime,
/// and never move out an owning or unlocked item. Shared views must not grant
/// exclusive access. This is an unsafe contract because an associated view
/// type could otherwise hide an older lifetime from the borrow checker.
///
/// ```compile_fail,E0200
/// use redlilium_ecs::QueryBorrow;
/// struct Custom;
/// impl QueryBorrow for Custom {
///     type Read<'q> = ();
///     type Write<'q> = ();
///     fn borrow_read(&self) {}
///     fn borrow_write(&mut self) {}
/// }
/// ```
pub unsafe trait QueryBorrow {
    type Read<'q>
    where
        Self: 'q;
    type Write<'q>
    where
        Self: 'q;

    fn borrow_read(&self) -> Self::Read<'_>;
    fn borrow_write(&mut self) -> Self::Write<'_>;
}

// SAFETY: reborrow shortens the storage lifetime to &self and remains read-only.
unsafe impl<T: 'static> QueryBorrow for Ref<'_, T> {
    type Read<'q>
        = Ref<'q, T>
    where
        Self: 'q;
    type Write<'q>
        = Ref<'q, T>
    where
        Self: 'q;
    fn borrow_read(&self) -> Self::Read<'_> {
        self.reborrow()
    }
    fn borrow_write(&mut self) -> Self::Write<'_> {
        self.reborrow()
    }
}
// SAFETY: shared/exclusive views respectively borrow self shared/exclusively.
unsafe impl<T: 'static> QueryBorrow for RefMut<'_, T> {
    type Read<'q>
        = Ref<'q, T>
    where
        Self: 'q;
    type Write<'q>
        = RefMut<'q, T>
    where
        Self: 'q;
    fn borrow_read(&self) -> Self::Read<'_> {
        self.reborrow()
    }
    fn borrow_write(&mut self) -> Self::Write<'_> {
        self.reborrow_mut()
    }
}
// SAFETY: Deref ties the shared resource reference to the borrow of self.
unsafe impl<T: 'static> QueryBorrow for ResourceRef<'_, T> {
    type Read<'q>
        = &'q T
    where
        Self: 'q;
    type Write<'q>
        = &'q T
    where
        Self: 'q;
    fn borrow_read(&self) -> Self::Read<'_> {
        self
    }
    fn borrow_write(&mut self) -> Self::Write<'_> {
        self
    }
}
// SAFETY: Deref/DerefMut return references tied to the matching self borrow.
unsafe impl<T: 'static> QueryBorrow for ResourceRefMut<'_, T> {
    type Read<'q>
        = &'q T
    where
        Self: 'q;
    type Write<'q>
        = &'q mut T
    where
        Self: 'q;
    fn borrow_read(&self) -> Self::Read<'_> {
        self
    }
    fn borrow_write(&mut self) -> Self::Write<'_> {
        self
    }
}
// SAFETY: map reborrows the element through its QueryBorrow contract; no take.
unsafe impl<T: QueryBorrow> QueryBorrow for Option<T> {
    type Read<'q>
        = Option<T::Read<'q>>
    where
        Self: 'q;
    type Write<'q>
        = Option<T::Write<'q>>
    where
        Self: 'q;
    fn borrow_read(&self) -> Self::Read<'_> {
        self.as_ref().map(QueryBorrow::borrow_read)
    }
    fn borrow_write(&mut self) -> Self::Write<'_> {
        self.as_mut().map(QueryBorrow::borrow_write)
    }
}

// Filters are shared metadata views, including in a mutable query.
macro_rules! borrow_filter {
    ($($filter:ident),+) => { $(
        // SAFETY: only shared references to filters are returned, tied to self.
        unsafe impl QueryBorrow for $filter<'_> {
            type Read<'q> = &'q Self where Self: 'q;
            type Write<'q> = &'q Self where Self: 'q;
            fn borrow_read(&self) -> Self::Read<'_> { self }
            fn borrow_write(&mut self) -> Self::Write<'_> { self }
        }
    )+ };
}
borrow_filter!(ContainsChecker, AddedFilter, ChangedFilter, RemovedFilter);
// SAFETY: filter internals stay private behind a shared borrow of self.
unsafe impl<A, B> QueryBorrow for OrFilter<A, B> {
    type Read<'q>
        = &'q Self
    where
        Self: 'q;
    type Write<'q>
        = &'q Self
    where
        Self: 'q;
    fn borrow_read(&self) -> Self::Read<'_> {
        self
    }
    fn borrow_write(&mut self) -> Self::Write<'_> {
        self
    }
}
// SAFETY: filter internals stay private behind a shared borrow of self.
unsafe impl<T> QueryBorrow for AnyFilter<T> {
    type Read<'q>
        = &'q Self
    where
        Self: 'q;
    type Write<'q>
        = &'q Self
    where
        Self: 'q;
    fn borrow_read(&self) -> Self::Read<'_> {
        self
    }
    fn borrow_write(&mut self) -> Self::Write<'_> {
        self
    }
}
// SAFETY: unit contains no data access.
unsafe impl QueryBorrow for () {
    type Read<'q> = ();
    type Write<'q> = ();
    fn borrow_read(&self) {}
    fn borrow_write(&mut self) {}
}
macro_rules! borrow_tuple {
    ($($i:tt $T:ident),+) => {
        // SAFETY: delegates to each element with disjoint tuple-field borrows.
        unsafe impl<$($T: QueryBorrow),+> QueryBorrow for ($($T,)+) {
            type Read<'q> = ($($T::Read<'q>,)+) where Self: 'q;
            type Write<'q> = ($($T::Write<'q>,)+) where Self: 'q;
            fn borrow_read(&self) -> Self::Read<'_> { ($(self.$i.borrow_read(),)+) }
            fn borrow_write(&mut self) -> Self::Write<'_> { ($(self.$i.borrow_write(),)+) }
        }
    };
}
borrow_tuple!(0 A);
borrow_tuple!(0 A, 1 B);
borrow_tuple!(0 A, 1 B, 2 C);
borrow_tuple!(0 A, 1 B, 2 C, 3 D);
borrow_tuple!(0 A, 1 B, 2 C, 3 D, 4 E);
borrow_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
borrow_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
borrow_tuple!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);

#[cfg(test)]
mod tests {
    use crate::{
        Any, Entity, Filter, OptionalRead, OptionalWrite, Or, Read, ReadAll, Res, ResMut, With,
        Without, World, Write, WriteAll,
    };

    #[test]
    fn optional_views_preserve_storage_and_change_tracking() {
        let mut world = World::new();
        world.register_component::<u32>();
        let a = world.spawn();
        let b = world.spawn();
        world.insert(a, 1_u32).unwrap();
        world.insert(b, 2_u32).unwrap();
        let before = world.current_tick();
        world.advance_tick();
        let mut q = world.query::<(OptionalWrite<u32>, OptionalRead<u64>)>();
        {
            let (mut present, missing) = q.items_mut();
            assert!(missing.is_none());
            let mut view = present.take().unwrap();
            let _unchanged = view.get_mut(b.index()).unwrap();
            drop(_unchanged);
            *view.get_mut(a.index()).unwrap() += 10;
        }
        let (values, missing) = q.items();
        let values = values.unwrap();
        assert!(missing.is_none());
        assert_eq!(values.get(a.index()), Some(&11));
        assert!(values.changed_since(a.index(), before));
        assert!(!values.changed_since(b.index(), before));
    }

    #[test]
    fn shared_and_mutable_views_preserve_exclusion_masks() {
        let mut world = World::new();
        world.register_component::<u32>();
        let e = world.spawn();
        world.insert(e, 1_u32).unwrap();
        world.set_entity_flags(e, Entity::STATIC);
        {
            let mut q = world.query::<(Write<u32>,)>();
            assert!(!q.items().0.contains(e.index()));
            assert!(q.items_mut().0.get_mut(e.index()).is_none());
        }
        {
            let mut q = world.query::<(WriteAll<u32>,)>();
            assert!(q.items().0.contains(e.index()));
            *q.items_mut().0.get_mut(e.index()).unwrap() += 1;
        }
        assert!(!world.query::<(Read<u32>,)>().items().0.contains(e.index()));
        assert_eq!(
            world.query::<(ReadAll<u32>,)>().items().0.get(e.index()),
            Some(&2)
        );
    }

    #[test]
    fn resources_and_nested_filters_can_be_reborrowed() {
        let mut world = World::new();
        world.register_component::<u32>();
        let e = world.spawn();
        world.insert(e, 1_u32).unwrap();
        world.insert_resource(2_u64);
        world.insert_resource(3_i64);
        type Access = (
            ResMut<u64>,
            Res<i64>,
            Or<With<u32>, Without<u8>>,
            Any<(With<u32>, Without<u16>)>,
        );
        let mut q = world.query::<Access>();
        {
            let (value, source, either, any) = q.items_mut();
            *value += *source as u64;
            fn matches(filter: impl Filter, entity: u32) -> bool {
                filter.matches(entity)
            }
            assert!(matches(either, e.index()));
            assert!(matches(any, e.index()));
        }
        let (a, b, _, _) = q.items();
        let (c, _, _, _) = q.items();
        assert_eq!((*a, *b, *c), (5, 3, 5));
    }

    #[test]
    fn storage_and_iterator_access_can_alternate() {
        let mut world = World::new();
        world.register_component::<u32>();
        let e = world.spawn();
        world.insert(e, 1_u32).unwrap();
        world.insert_resource(0_u64);
        let mut q = world.query::<(Write<u32>, ResMut<u64>)>();
        {
            let (mut values, total) = q.items_mut();
            *values.get_mut(e.index()).unwrap() += 1;
            *total = 2;
        }
        for (_, (mut value, mut total)) in q.iter_mut() {
            *value += 1;
            *total += 1;
        }
        let (values, total) = q.items();
        assert_eq!(values.get(e.index()), Some(&3));
        assert_eq!(*total, 3);
    }
}
