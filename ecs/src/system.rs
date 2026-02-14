use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::command_collector::CommandCollector;
use crate::compute::{ComputePool, noop_waker};
use crate::system_context::SystemContext;
use crate::world::World;

/// A type-erased system future.
///
/// This is a boxed future returned by [`System::run`](crate::System::run).
/// Each system invocation produces a `SystemFuture` that the runner polls
/// to completion.
pub(crate) type SystemFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

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
    /// Execute the system with the given context.
    ///
    /// Use [`SystemContext::lock()`] to borrow components.
    /// Locks are automatically released when the execute closure returns,
    /// making it safe to `.await` between lock-execute calls.
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> impl Future<Output = ()> + Send + 'a;
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
        Box::pin(self.run(ctx))
    }
}

/// Runs a system synchronously to completion.
///
/// Creates a single-threaded [`SystemContext`], calls [`System::run`], and
/// polls the returned future to completion. Drives the [`ComputePool`]
/// between polls so spawned compute tasks make progress.
///
/// Useful for tests and one-off system invocations outside a runner.
pub fn run_system_blocking(
    system: &impl System,
    world: &World,
    compute: &ComputePool,
    io: &crate::io_runtime::IoRuntime,
) {
    let commands = CommandCollector::new();
    let ctx = SystemContext::new(world, compute, io, &commands);
    let future = Box::pin(system.run(&ctx));
    poll_system_future_to_completion(future, compute);

    // Apply deferred commands
    // Safety: we need &mut World but only have &World.
    // This function should be called with an owned/mutable world.
    // For now, commands are discarded — callers should use the runner for full support.
    // TODO: Reconsider this API or require &mut World.
}

/// Polls a system future to completion, driving compute tasks between polls.
pub(crate) fn poll_system_future_to_completion(future: SystemFuture<'_>, compute: &ComputePool) {
    let mut future = future;
    // Safety: future is on the stack and won't be moved after this point.
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    let waker = noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(()) => break,
            std::task::Poll::Pending => {
                compute.tick_all();
                #[cfg(not(target_arch = "wasm32"))]
                std::thread::yield_now();
            }
        }
    }
}

/// Typed wrapper for system output, stored as a World resource.
///
/// Keyed by the system type `S` for uniqueness — multiple systems can
/// produce different `T` values without collision.
///
/// # Example
///
/// ```ignore
/// // Producer system stores result as resource:
/// ctx.commands(|world| {
///     world.insert_resource(SystemResult::<PhysicsSystem, PhysicsResult>::new(result));
/// });
///
/// // Consumer system reads the result:
/// ctx.lock::<(Res<SystemResult<PhysicsSystem, PhysicsResult>>,)>()
///     .execute(|(result,)| {
///         // use result.value
///     }).await;
/// ```
pub struct SystemResult<S: 'static, T: Send + Sync + 'static> {
    /// The system's output value.
    pub value: T,
    _marker: PhantomData<fn() -> S>,
}

impl<S: 'static, T: Send + Sync + 'static> SystemResult<S, T> {
    /// Creates a new system result.
    pub fn new(value: T) -> Self {
        Self {
            value,
            _marker: PhantomData,
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
        async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {}
    }

    struct MovementSystem;
    impl System for MovementSystem {
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
