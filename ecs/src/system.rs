use std::any::Any;
use std::future::Future;
use std::pin::Pin;

use crate::command_collector::CommandCollector;
use crate::compute::{ComputePool, noop_waker};
use crate::system_context::SystemContext;
use crate::world::World;

/// A type-erased system future.
///
/// This is a boxed future returned by [`DynSystem::run_boxed`].
/// The result is type-erased as `Box<dyn Any + Send + Sync>` so the runner
/// can store it in [`SystemResultsStore`](crate::system_results_store::SystemResultsStore).
pub(crate) type SystemFuture<'a> =
    Pin<Box<dyn Future<Output = Box<dyn Any + Send + Sync>> + Send + 'a>>;

/// A system that processes entities and components in the world.
///
/// All systems are async. Systems receive a [`SystemContext`] that provides
/// typed component access via the lock-execute pattern, compute task spawning,
/// and deferred commands.
///
/// # Component access
///
/// Use [`SystemContext::lock()`] with a tuple of access types to borrow
/// components. The execute closure is synchronous, preventing guards from
/// being held across `.await` points at compile time.
///
/// # Example — simple system
///
/// ```ignore
/// struct MovementSystem;
///
/// impl System for MovementSystem {
///     async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
///         ctx.lock::<(Write<Position>, Read<Velocity>)>()
///             .execute(|(mut positions, velocities)| {
///                 for (idx, pos) in positions.iter_mut() {
///                     if let Some(vel) = velocities.get(idx) {
///                         pos.x += vel.x;
///                     }
///                 }
///             }).await;
///     }
/// }
/// ```
///
/// # Example — two-phase async system
///
/// ```ignore
/// struct PathfindSystem;
///
/// impl System for PathfindSystem {
///     async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
///         // Phase 1: extract data (lock released after execute)
///         let graph = ctx.lock::<(Read<NavMesh>,)>()
///             .execute(|(nav,)| {
///                 nav.iter().next().map(|(_, n)| n.clone())
///             }).await;
///
///         // Safe to .await — no locks held
///         let mut handle = ctx.compute().spawn(Priority::Low, |_cctx| async move {
///             compute_paths(graph)
///         });
///         let paths = (&mut handle).await;
///
///         // Phase 2: apply results via deferred command
///         if let Some(paths) = paths {
///             ctx.commands(move |world| {
///                 // apply paths to agents
///             });
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
    /// Locks are automatically released when the execute closure returns,
    /// making it safe to `.await` between lock-execute calls.
    fn run<'a>(
        &'a self,
        ctx: &'a SystemContext<'a>,
    ) -> impl Future<Output = Self::Result> + Send + 'a;
}

/// Object-safe version of [`System`] used internally for type erasure.
///
/// A blanket implementation converts any `System` into a `DynSystem`
/// by wrapping the future in `Box::pin`.
pub(crate) trait DynSystem: Send + Sync {
    fn run_boxed<'a>(&'a self, ctx: &'a SystemContext<'a>) -> SystemFuture<'a>;
}

impl<S: System> DynSystem for S {
    fn run_boxed<'a>(&'a self, ctx: &'a SystemContext<'a>) -> SystemFuture<'a> {
        Box::pin(async move {
            let result = self.run(ctx).await;
            Box::new(result) as Box<dyn Any + Send + Sync>
        })
    }
}

/// Runs a system synchronously to completion, returning its result.
///
/// Creates a single-threaded [`SystemContext`], calls [`System::run`], and
/// polls the returned future to completion. Drives the [`ComputePool`]
/// between polls so spawned compute tasks make progress.
///
/// Useful for tests and one-off system invocations outside a runner.
pub fn run_system_blocking<S: System>(
    system: &S,
    world: &World,
    compute: &ComputePool,
    io: &crate::io_runtime::IoRuntime,
) -> S::Result {
    let commands = CommandCollector::new();
    let ctx = SystemContext::new(world, compute, io, &commands);
    let future = system.run(&ctx);
    let mut future = core::pin::pin!(future);
    let waker = noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(result) => break result,
            std::task::Poll::Pending => {
                compute.tick_all();
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::yield_now();
            }
        }
    }

    // Apply deferred commands
    // Safety: we need &mut World but only have &World.
    // This function should be called with an owned/mutable world.
    // For now, commands are discarded — callers should use the runner for full support.
    // TODO: Reconsider this API or require &mut World.
}

/// Polls a type-erased system future to completion, driving compute tasks between polls.
///
/// Returns the boxed result produced by the system.
pub(crate) fn poll_system_future_to_completion(
    future: SystemFuture<'_>,
    compute: &ComputePool,
) -> Box<dyn Any + Send + Sync> {
    let mut future = future;
    // Safety: future is on the stack and won't be moved after this point.
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    let waker = noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(result) => break result,
            std::task::Poll::Pending => {
                compute.tick_all();
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::yield_now();
            }
        }
    }
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
        async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {}
    }

    struct MovementSystem;
    impl System for MovementSystem {
        type Result = ();
        async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
            ctx.lock::<(Write<Position>, Read<Velocity>)>()
                .execute(|(mut positions, velocities)| {
                    for (idx, pos) in positions.iter_mut() {
                        if let Some(vel) = velocities.get(idx) {
                            pos.x += vel.x;
                        }
                    }
                })
                .await;
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
            async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
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
            async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
                let slot = self.0.clone();
                let mut handle = ctx.compute().spawn(Priority::Low, |_ctx| async { 99u32 });
                let result = (&mut handle).await;
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
