use std::any::Any;

use crate::command_collector::CommandCollector;
use crate::compute::ComputePool;
use crate::system_context::SystemContext;
use crate::world::World;

/// A system that processes entities and components in the world.
///
/// Systems receive a [`SystemContext`] that provides typed component access
/// via the lock-execute pattern, compute task spawning, and deferred commands.
///
/// # Component access
///
/// Use [`SystemContext::lock()`] with a tuple of access types to borrow
/// components. The execute closure is synchronous, ensuring locks are
/// released deterministically when the closure returns.
///
/// # Example — simple system
///
/// ```ignore
/// struct MovementSystem;
///
/// impl System for MovementSystem {
///     type Result = ();
///     fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
///         ctx.lock::<(Write<Position>, Read<Velocity>)>()
///             .execute(|(mut positions, velocities)| {
///                 for (idx, pos) in positions.iter_mut() {
///                     if let Some(vel) = velocities.get(idx) {
///                         pos.x += vel.x;
///                     }
///                 }
///             });
///     }
/// }
/// ```
///
/// # Example — system with compute task
///
/// ```ignore
/// struct PathfindSystem;
///
/// impl System for PathfindSystem {
///     type Result = ();
///     fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
///         // Phase 1: extract data (lock released after execute)
///         let graph = ctx.lock::<(Read<NavMesh>,)>()
///             .execute(|(nav,)| {
///                 nav.iter().next().map(|(_, n)| n.clone())
///             });
///
///         // Phase 2: offload heavy computation
///         if let Some(graph) = graph {
///             let mut handle = ctx.compute().spawn(Priority::Low, |_cctx| async move {
///                 compute_paths(graph)
///             });
///             let paths = ctx.compute().block_on(&mut handle);
///
///             // Phase 3: apply results via deferred command
///             if let Some(paths) = paths {
///                 ctx.commands(move |world| {
///                     // apply paths to agents
///                 });
///             }
///         }
///     }
/// }
/// ```
pub trait System: Send + Sync + 'static {
    /// The value returned by this system after execution.
    ///
    /// Downstream systems that depend on this system (via dependency edges)
    /// can read the result through [`SystemContext::system_result::<Self>()`].
    type Result: Send + Sync + 'static;

    /// Execute the system with the given context.
    ///
    /// Use [`SystemContext::lock()`] to borrow components.
    /// Locks are automatically released when the execute closure returns.
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Self::Result;
}

/// Object-safe version of [`System`] used internally for type erasure.
///
/// A blanket implementation converts any `System` into a `DynSystem`.
pub(crate) trait DynSystem: Send + Sync {
    fn run_boxed<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Box<dyn Any + Send + Sync>;
}

impl<S: System> DynSystem for S {
    fn run_boxed<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Box<dyn Any + Send + Sync> {
        let result = self.run(ctx);
        Box::new(result) as Box<dyn Any + Send + Sync>
    }
}

/// A system that receives exclusive `&mut World` access.
///
/// Unlike [`System`], which accesses components through the lock-execute
/// pattern on `&SystemContext`, exclusive systems get direct mutable access
/// to the entire world. This enables immediate structural changes (spawn,
/// despawn, insert, remove) without deferring to commands.
///
/// Exclusive systems act as **barriers** in the scheduler — no other system
/// runs concurrently with an exclusive system. Pending deferred commands from
/// predecessor regular systems are applied before the exclusive system runs.
///
/// # When to use
///
/// Use `ExclusiveSystem` when you need:
/// - Direct `&mut World` access (e.g. scene loading, batch operations)
/// - Immediate structural changes visible within the same frame
/// - Operations that don't fit the lock-execute pattern
///
/// For most systems, prefer the regular [`System`] trait which enables
/// parallel execution.
///
/// # Example
///
/// ```ignore
/// struct SceneLoader;
///
/// impl ExclusiveSystem for SceneLoader {
///     type Result = ();
///     fn run(&mut self, world: &mut World) {
///         let e = world.spawn();
///         world.insert(e, Transform::IDENTITY).unwrap();
///     }
/// }
/// ```
pub trait ExclusiveSystem: Send + Sync + 'static {
    /// The value returned by this system after execution.
    ///
    /// Downstream systems that depend on this system (via dependency edges)
    /// can read the result through
    /// [`SystemContext::exclusive_system_result::<Self>()`].
    type Result: Send + Sync + 'static;

    /// Execute the system with exclusive world access.
    fn run(&mut self, world: &mut World) -> Self::Result;
}

/// Object-safe version of [`ExclusiveSystem`] used internally for type erasure.
pub(crate) trait DynExclusiveSystem: Send + Sync {
    fn run_boxed(&mut self, world: &mut World) -> Box<dyn Any + Send + Sync>;
}

impl<S: ExclusiveSystem> DynExclusiveSystem for S {
    fn run_boxed(&mut self, world: &mut World) -> Box<dyn Any + Send + Sync> {
        let result = self.run(world);
        Box::new(result) as Box<dyn Any + Send + Sync>
    }
}

/// An exclusive system built from a closure.
///
/// Wraps a `FnMut(&mut World)` as an [`ExclusiveSystem`]. Created via
/// [`SystemsContainer::add_exclusive_fn`](crate::SystemsContainer::add_exclusive_fn).
///
/// # Example
///
/// ```ignore
/// let mut container = SystemsContainer::new();
/// container.add_exclusive_fn(|world: &mut World| {
///     world.insert_resource(42u32);
/// });
/// ```
pub struct ExclusiveFunctionSystem<F> {
    func: F,
}

impl<F> ExclusiveFunctionSystem<F> {
    /// Creates a new exclusive function system from a closure.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> ExclusiveSystem for ExclusiveFunctionSystem<F>
where
    F: FnMut(&mut World) + Send + Sync + 'static,
{
    type Result = ();
    fn run(&mut self, world: &mut World) {
        (self.func)(world);
    }
}

/// Runs a system synchronously, returning its result.
///
/// Creates a single-threaded [`SystemContext`], calls [`System::run`],
/// and returns the result. Useful for tests and one-off system
/// invocations outside a runner.
///
/// Note: deferred commands are not applied. Use a runner for full support.
pub fn run_system_blocking<S: System>(
    system: &S,
    world: &World,
    compute: &ComputePool,
    io: &crate::io_runtime::IoRuntime,
) -> S::Result {
    let commands = CommandCollector::new();
    let ctx = SystemContext::new(world, compute, io, &commands);
    system.run(&ctx)
}

/// Runs an exclusive system synchronously, returning its result.
///
/// The system receives `&mut World` directly. Useful for tests and
/// one-off invocations outside a runner.
pub fn run_exclusive_system_blocking<S: ExclusiveSystem>(
    system: &mut S,
    world: &mut World,
) -> S::Result {
    system.run(world)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_set::{Read, Write};
    use crate::io_runtime::IoRuntime;

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }

    struct EmptySystem;
    impl System for EmptySystem {
        type Result = ();
        fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {}
    }

    struct MovementSystem;
    impl System for MovementSystem {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
            ctx.lock::<(Write<Position>, Read<Velocity>)>().execute(
                |(mut positions, velocities)| {
                    for (idx, pos) in positions.iter_mut() {
                        if let Some(vel) = velocities.get(idx) {
                            pos.x += vel.x;
                        }
                    }
                },
            );
        }
    }

    #[test]
    fn empty_system_runs() {
        let world = World::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&EmptySystem, &world, &compute, &io);
    }

    #[test]
    fn system_runs_blocking() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct FlagSystem(std::sync::Arc<AtomicBool>);
        impl System for FlagSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
                self.0.store(true, Ordering::Relaxed);
            }
        }

        let flag = std::sync::Arc::new(AtomicBool::new(false));
        let sys = FlagSystem(flag.clone());
        let world = World::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn movement_system_updates_positions() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();
        world.insert(e, Velocity { x: 5.0 }).unwrap();

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&MovementSystem, &world, &compute, &io);

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    // ---- ExclusiveSystem tests ----

    struct SpawnExclusiveSystem;
    impl ExclusiveSystem for SpawnExclusiveSystem {
        type Result = ();
        fn run(&mut self, world: &mut World) {
            let e = world.spawn();
            world.insert(e, Position { x: 99.0 }).unwrap();
        }
    }

    #[test]
    fn exclusive_system_runs() {
        let mut world = World::new();
        world.register_component::<Position>();

        let mut sys = SpawnExclusiveSystem;
        run_exclusive_system_blocking(&mut sys, &mut world);

        assert_eq!(world.entity_count(), 1);
        let e = world.iter_entities().next().unwrap();
        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn exclusive_function_system_runs() {
        let mut world = World::new();
        world.register_component::<Position>();

        let mut sys = ExclusiveFunctionSystem {
            func: |world: &mut World| {
                let e = world.spawn();
                world.insert(e, Position { x: 42.0 }).unwrap();
            },
        };
        run_exclusive_system_blocking(&mut sys, &mut world);

        assert_eq!(world.entity_count(), 1);
        let e = world.iter_entities().next().unwrap();
        assert_eq!(world.get::<Position>(e).unwrap().x, 42.0);
    }

    #[test]
    fn exclusive_system_with_result() {
        struct CountSystem;
        impl ExclusiveSystem for CountSystem {
            type Result = u32;
            fn run(&mut self, world: &mut World) -> u32 {
                world.entity_count()
            }
        }

        let mut world = World::new();
        world.spawn();
        world.spawn();

        let mut sys = CountSystem;
        let count = run_exclusive_system_blocking(&mut sys, &mut world);
        assert_eq!(count, 2);
    }

    #[test]
    fn run_blocking_drives_compute() {
        use crate::Priority;

        struct ComputeSystem(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl System for ComputeSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
                let slot = self.0.clone();
                let mut handle = ctx.compute().spawn(Priority::Low, |_ctx| async { 99u32 });
                let result = ctx.compute().block_on(&mut handle);
                *slot.lock().unwrap() = result;
            }
        }

        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        let sys = ComputeSystem(result.clone());
        let world = World::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io);
        assert_eq!(*result.lock().unwrap(), Some(99));
    }
}
