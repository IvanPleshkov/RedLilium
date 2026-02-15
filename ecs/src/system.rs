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
