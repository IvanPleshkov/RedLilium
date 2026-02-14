use std::marker::PhantomData;

use crate::access_set::{AccessElement, AccessSet};
use crate::query_guard::QueryItem;
use crate::system::System;
use crate::system_context::SystemContext;

/// A system built from a plain function.
///
/// Wraps a function whose parameters are automatically extracted from the
/// world based on an [`AccessSet`]. Created via [`IntoSystem::into_system`]
/// or [`SystemsContainer::add_fn_raw`](crate::SystemsContainer::add_fn_raw).
///
/// The function receives the access set's items as a tuple. All locks are
/// acquired before the function runs and released after it returns.
///
/// Function systems are **synchronous** — for multi-phase async systems,
/// compute spawning, or deferred commands, use the struct-based [`System`]
/// trait directly.
///
/// # Example
///
/// ```ignore
/// use redlilium_ecs::*;
///
/// fn movement(
///     (mut positions, velocities): (RefMut<Position>, Ref<Velocity>),
/// ) {
///     for (idx, pos) in positions.iter_mut() {
///         if let Some(vel) = velocities.get(idx) {
///             pos.x += vel.x;
///         }
///     }
/// }
///
/// let mut container = SystemsContainer::new();
/// container.add_fn_raw::<(Write<Position>, Read<Velocity>), _>(movement);
/// ```
pub struct FunctionSystem<F, Marker> {
    func: F,
    _marker: PhantomData<fn() -> Marker>,
}

/// Converts a function into a [`System`].
///
/// Two flavors:
///
/// - **Zero parameters**: `IntoSystem<fn()>` — for functions taking no arguments.
/// - **With access**: `IntoSystem<A>` where `A: AccessSet` — the function
///   receives `A::Item` (a tuple of component/resource refs).
///
/// # Supported access types
///
/// | Access type | Function receives |
/// |-------------|------------------|
/// | [`Read<T>`](crate::Read) | [`Ref<T>`](crate::Ref) |
/// | [`Write<T>`](crate::Write) | [`RefMut<T>`](crate::RefMut) |
/// | [`OptionalRead<T>`](crate::OptionalRead) | `Option<Ref<T>>` |
/// | [`OptionalWrite<T>`](crate::OptionalWrite) | `Option<RefMut<T>>` |
/// | [`Res<T>`](crate::Res) | [`ResourceRef<T>`](crate::ResourceRef) |
/// | [`ResMut<T>`](crate::ResMut) | [`ResourceRefMut<T>`](crate::ResourceRefMut) |
pub trait IntoSystem<Marker> {
    /// The system type produced.
    type System: System;

    /// Converts this function into a system.
    fn into_system(self) -> Self::System;
}

// ---- Zero-parameter function system ----

impl<F> System for FunctionSystem<F, fn()>
where
    F: Fn() + Send + Sync + 'static,
{
    type Result = ();
    async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
        (self.func)();
    }
}

impl<F> IntoSystem<fn()> for F
where
    F: Fn() + Send + Sync + 'static,
{
    type System = FunctionSystem<F, fn()>;

    fn into_system(self) -> Self::System {
        FunctionSystem {
            func: self,
            _marker: PhantomData,
        }
    }
}

// ---- AccessSet-parameterized function system ----

impl<A, F> System for FunctionSystem<F, A>
where
    A: AccessSet + Send + Sync + 'static,
    F: for<'a> Fn(A::Item<'a>) + Send + Sync + 'static,
{
    type Result = ();
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<A>().execute(|items| (self.func)(items)).await;
    }
}

impl<A, F> IntoSystem<A> for F
where
    A: AccessSet + Send + Sync + 'static,
    F: for<'a> Fn(A::Item<'a>) + Send + Sync + 'static,
{
    type System = FunctionSystem<F, A>;

    fn into_system(self) -> Self::System {
        FunctionSystem {
            func: self,
            _marker: PhantomData,
        }
    }
}

// ---- Per-entity for-each function system ----

/// Bridges an [`AccessSet`] to per-entity iteration for [`ForEachSystem`].
///
/// Maps each `AccessElement`'s locked storage to a per-entity item type
/// (e.g., `Ref<T>` → `&T`, `RefMut<T>` → `&mut T`). Implemented for
/// tuples of access elements whose items implement [`QueryItem`].
pub trait ForEachAccess: AccessSet {
    /// The per-entity item type (e.g., `(&mut Position, &Velocity)`).
    type EachItem<'w>;

    /// Iterate over matching entities, calling `f` for each one.
    ///
    /// Performs an inner join: iterates the smallest component storage
    /// and yields only entities present in all queried storages.
    fn run_for_each<'w>(items: &Self::Item<'w>, f: impl FnMut(Self::EachItem<'w>));
}

macro_rules! impl_for_each_access {
    ($($idx:tt $T:ident),+) => {
        impl<$($T: AccessElement),+> ForEachAccess for ($($T,)+)
        where
            $(for<'w> $T::Item<'w>: QueryItem,)+
        {
            type EachItem<'w> = ($(<$T::Item<'w> as QueryItem>::Item,)+);

            fn run_for_each<'w>(items: &Self::Item<'w>, mut f: impl FnMut(Self::EachItem<'w>)) {
                // Find the smallest storage to drive iteration.
                let mut _min_count = usize::MAX;
                let mut min_entities: &[u32] = &[];
                $(
                    let count = items.$idx.query_count();
                    if count < _min_count {
                        _min_count = count;
                        min_entities = items.$idx.query_entities();
                    }
                )+
                // Inner join: for each entity in the smallest set,
                // try to fetch from all storages.
                for &entity in min_entities {
                    // SAFETY: each entity_index is visited exactly once
                    // (dense entity arrays have no duplicates).
                    let item = unsafe { ($( match items.$idx.query_get(entity) {
                        Some(v) => v,
                        None => continue,
                    }, )+) };
                    f(item);
                }
            }
        }
    };
}

impl_for_each_access!(0 A);
impl_for_each_access!(0 A, 1 B);
impl_for_each_access!(0 A, 1 B, 2 C);
impl_for_each_access!(0 A, 1 B, 2 C, 3 D);
impl_for_each_access!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_for_each_access!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_for_each_access!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_for_each_access!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);

/// Marker type for per-entity function systems.
///
/// Used as the `Marker` parameter in [`IntoSystem<ForEach<A>>`] to
/// distinguish per-entity functions from storage-level functions.
pub struct ForEach<A>(PhantomData<A>);

/// A system that calls a function once per matching entity.
///
/// Created via [`for_each()`], [`IntoSystem::<ForEach<A>>::into_system()`],
/// or [`SystemsContainer::add_fn()`](crate::SystemsContainer::add_fn).
///
/// The function receives per-entity references (e.g., `&mut Position`)
/// rather than whole storages. The framework handles locking and
/// inner-join iteration automatically.
///
/// # Example
///
/// ```ignore
/// use redlilium_ecs::*;
///
/// fn movement((pos, vel): (&mut Position, &Velocity)) {
///     pos.x += vel.x;
/// }
///
/// let mut container = SystemsContainer::new();
/// container.add_fn::<(Write<Position>, Read<Velocity>), _>(movement);
/// ```
pub struct ForEachSystem<F, A> {
    func: F,
    _marker: PhantomData<fn() -> A>,
}

impl<A, F> System for ForEachSystem<F, A>
where
    A: ForEachAccess + Send + Sync + 'static,
    F: for<'a> Fn(A::EachItem<'a>) + Send + Sync + 'static,
{
    type Result = ();
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<A>()
            .execute(|items| {
                A::run_for_each(&items, |item| {
                    (self.func)(item);
                });
            })
            .await;
    }
}

impl<A, F> IntoSystem<ForEach<A>> for F
where
    A: ForEachAccess + Send + Sync + 'static,
    F: for<'a> Fn(A::EachItem<'a>) + Send + Sync + 'static,
{
    type System = ForEachSystem<F, A>;

    fn into_system(self) -> Self::System {
        ForEachSystem {
            func: self,
            _marker: PhantomData,
        }
    }
}

/// Creates a [`ForEachSystem`] that calls `func` once per matching entity.
///
/// The access set `A` determines which components/resources are locked.
/// The function receives per-entity references as a tuple.
///
/// # Example
///
/// ```ignore
/// use redlilium_ecs::*;
///
/// fn movement((pos, vel): (&mut Position, &Velocity)) {
///     pos.x += vel.x;
/// }
///
/// let sys = for_each::<(Write<Position>, Read<Velocity>), _>(movement);
/// ```
pub fn for_each<A, F>(func: F) -> ForEachSystem<F, A>
where
    A: ForEachAccess + Send + Sync + 'static,
    F: for<'a> Fn(A::EachItem<'a>) + Send + Sync + 'static,
{
    ForEachSystem {
        func,
        _marker: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_set::{Read, Write};
    use crate::compute::ComputePool;
    use crate::io_runtime::IoRuntime;
    use crate::system::run_system_blocking;
    use crate::world::World;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }

    #[test]
    fn zero_param_function_system() {
        let flag = std::sync::Arc::new(AtomicBool::new(false));
        let flag2 = flag.clone();
        let sys = IntoSystem::<fn()>::into_system(move || {
            flag2.store(true, Ordering::Relaxed);
        });

        let world = World::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn single_read_param() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();

        fn check_positions((positions,): (crate::Ref<Position>,)) {
            assert_eq!(positions.len(), 1);
        }

        let sys = IntoSystem::<(Read<Position>,)>::into_system(check_positions);
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
    }

    #[test]
    fn single_write_param() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();

        fn set_positions((mut positions,): (crate::RefMut<Position>,)) {
            for (_, pos) in positions.iter_mut() {
                pos.x = 99.0;
            }
        }

        let sys = IntoSystem::<(Write<Position>,)>::into_system(set_positions);
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn multi_param_movement() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();
        world.insert(e, Velocity { x: 5.0 }).unwrap();

        fn movement((mut positions, velocities): (crate::RefMut<Position>, crate::Ref<Velocity>)) {
            for (idx, pos) in positions.iter_mut() {
                if let Some(vel) = velocities.get(idx) {
                    pos.x += vel.x;
                }
            }
        }

        let sys = IntoSystem::<(Write<Position>, Read<Velocity>)>::into_system(movement);
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[test]
    fn resource_param() {
        let mut world = World::new();
        world.insert_resource(1.5f64);

        fn read_dt((dt,): (crate::ResourceRef<f64>,)) {
            assert_eq!(*dt, 1.5);
        }

        let sys = IntoSystem::<(crate::access_set::Res<f64>,)>::into_system(read_dt);
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
    }

    #[test]
    fn optional_param_returns_none() {
        let world = World::new();

        fn check_optional((opt,): (Option<crate::Ref<Position>>,)) {
            assert!(opt.is_none());
        }

        let sys =
            IntoSystem::<(crate::access_set::OptionalRead<Position>,)>::into_system(check_optional);
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
    }

    #[test]
    fn optional_param_returns_some() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        fn check_optional((opt,): (Option<crate::Ref<Position>>,)) {
            assert!(opt.is_some());
            assert_eq!(opt.unwrap().len(), 1);
        }

        let sys =
            IntoSystem::<(crate::access_set::OptionalRead<Position>,)>::into_system(check_optional);
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
    }

    #[test]
    fn closure_as_system() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();

        let sys = IntoSystem::<(Write<Position>,)>::into_system(
            |(mut positions,): (crate::RefMut<Position>,)| {
                for (_, pos) in positions.iter_mut() {
                    pos.x = 77.0;
                }
            },
        );

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert_eq!(world.get::<Position>(e).unwrap().x, 77.0);
    }

    #[test]
    fn fn_system_in_container() {
        fn noop() {}

        let mut container = crate::SystemsContainer::new();
        container.add_fn_raw::<fn(), _>(noop);
        assert_eq!(container.system_count(), 1);
    }

    #[test]
    fn add_fn_raw_with_access() {
        fn gravity((mut velocities,): (crate::RefMut<Velocity>,)) {
            for (_, vel) in velocities.iter_mut() {
                vel.x -= 9.81;
            }
        }

        let mut world = World::new();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Velocity { x: 0.0 }).unwrap();

        let mut container = crate::SystemsContainer::new();
        container.add_fn_raw::<(Write<Velocity>,), _>(gravity);
        assert_eq!(container.system_count(), 1);
    }

    // ---- ForEachSystem tests ----

    #[test]
    fn for_each_single_read() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();

        let sum = std::sync::Arc::new(std::sync::Mutex::new(0.0f32));
        let sum2 = sum.clone();
        let sys = for_each::<(Read<Position>,), _>(move |(pos,): (&Position,)| {
            *sum2.lock().unwrap() += pos.x;
        });

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert_eq!(*sum.lock().unwrap(), 3.0);
    }

    #[test]
    fn for_each_single_write() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();

        let sys = for_each::<(Write<Position>,), _>(|(pos,): (&mut Position,)| {
            pos.x = 99.0;
        });

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn for_each_two_component_join() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        let e1 = world.spawn();
        world.insert(e1, Position { x: 10.0 }).unwrap();
        world.insert(e1, Velocity { x: 5.0 }).unwrap();

        // e2 has only Position — should be skipped
        let e2 = world.spawn();
        world.insert(e2, Position { x: 20.0 }).unwrap();

        fn movement((pos, vel): (&mut Position, &Velocity)) {
            pos.x += vel.x;
        }

        let sys = for_each::<(Write<Position>, Read<Velocity>), _>(movement);
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);

        assert_eq!(world.get::<Position>(e1).unwrap().x, 15.0);
        assert_eq!(world.get::<Position>(e2).unwrap().x, 20.0); // unchanged
    }

    #[test]
    fn for_each_with_resource() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        world.insert_resource(2.0f32); // speed multiplier

        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();
        world.insert(e, Velocity { x: 3.0 }).unwrap();

        let sys = for_each::<(Write<Position>, Read<Velocity>, crate::access_set::Res<f32>), _>(
            |(pos, vel, factor): (&mut Position, &Velocity, &f32)| {
                pos.x += vel.x * *factor;
            },
        );

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert_eq!(world.get::<Position>(e).unwrap().x, 6.0);
    }

    #[test]
    fn for_each_with_res_mut() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.insert_resource(0.0f32); // accumulator

        let e1 = world.spawn();
        world.insert(e1, Position { x: 3.0 }).unwrap();
        let e2 = world.spawn();
        world.insert(e2, Position { x: 7.0 }).unwrap();

        let sys = for_each::<(Read<Position>, crate::access_set::ResMut<f32>), _>(
            |(pos, mut acc): (&Position, crate::query_guard::ResMutRef<f32>)| {
                *acc += pos.x;
            },
        );

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        let acc = world.resource::<f32>();
        assert_eq!(*acc, 10.0);
    }

    #[test]
    fn for_each_closure() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();

        let sys = for_each::<(Write<Position>,), _>(|(pos,): (&mut Position,)| {
            pos.x = 77.0;
        });

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert_eq!(world.get::<Position>(e).unwrap().x, 77.0);
    }

    #[test]
    fn for_each_into_system() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();

        fn set_pos((pos,): (&mut Position,)) {
            pos.x = 42.0;
        }

        let sys = IntoSystem::<ForEach<(Write<Position>,)>>::into_system(set_pos);
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert_eq!(world.get::<Position>(e).unwrap().x, 42.0);
    }

    #[test]
    fn for_each_in_container() {
        fn movement((pos, vel): (&mut Position, &Velocity)) {
            pos.x += vel.x;
        }

        let mut container = crate::SystemsContainer::new();
        container.add_fn::<(Write<Position>, Read<Velocity>), _>(movement);
        assert_eq!(container.system_count(), 1);
    }
}
