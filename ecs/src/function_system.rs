use std::marker::PhantomData;

use crate::access_set::AccessSet;
use crate::system::System;
use crate::system_context::SystemContext;

/// A system built from a plain function.
///
/// Wraps a function whose parameters are automatically extracted from the
/// world based on an [`AccessSet`]. Created via [`IntoSystem::into_system`]
/// or [`SystemsContainer::add_fn`](crate::SystemsContainer::add_fn).
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
/// container.add_fn::<(Write<Position>, Read<Velocity>), _>(movement);
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
        container.add_fn::<fn(), _>(noop);
        assert_eq!(container.system_count(), 1);
    }

    #[test]
    fn add_fn_with_access() {
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
        container.add_fn::<(Write<Velocity>,), _>(gravity);
        assert_eq!(container.system_count(), 1);
    }
}
