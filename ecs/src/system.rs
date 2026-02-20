use std::any::Any;
use std::fmt;

use crate::command_collector::CommandCollector;
use crate::compute::ComputePool;
use crate::system_context::SystemContext;
use crate::world::World;

/// Error type returned by system execution.
///
/// Covers explicit failures from system logic and panics caught by the
/// runner via [`std::panic::catch_unwind`].
#[derive(Debug)]
pub enum SystemError {
    /// The system panicked during execution.
    ///
    /// Contains the panic message if it could be extracted from the payload.
    Panicked(String),
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SystemError::Panicked(msg) => write!(f, "system panicked: {msg}"),
        }
    }
}

impl std::error::Error for SystemError {}

/// Extracts a human-readable message from a panic payload.
///
/// Handles the two common payload types: `&str` and `String`.
pub fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

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
///     fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
///         ctx.lock::<(Write<Position>, Read<Velocity>)>()
///             .execute(|(mut positions, velocities)| {
///                 for (idx, pos) in positions.iter_mut() {
///                     if let Some(vel) = velocities.get(idx) {
///                         pos.x += vel.x;
///                     }
///                 }
///             });
///         Ok(())
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
///     fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
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
///         Ok(())
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
    ///
    /// Returns `Ok(result)` on success or `Err(SystemError)` on failure.
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError>;

    /// Receive the previous tick's result so allocated memory can be reused.
    ///
    /// Called by the runner before [`run()`](System::run) each tick.
    /// The default implementation drops the value. Override this to store
    /// the previous result internally (e.g. in a `Mutex<Option<Vec<…>>>`)
    /// and reuse its allocation in the next [`run()`](System::run).
    #[allow(unused_variables)]
    fn reuse_result(&self, prev: Self::Result) {}
}

/// Object-safe version of [`System`] used internally for type erasure.
///
/// A blanket implementation converts any `System` into a `DynSystem`.
pub(crate) trait DynSystem: Send + Sync {
    fn run_boxed<'a>(
        &'a self,
        ctx: &'a SystemContext<'a>,
    ) -> Result<Box<dyn Any + Send + Sync>, SystemError>;

    fn reuse_result_boxed(&self, prev: Box<dyn Any + Send + Sync>);
}

impl<S: System> DynSystem for S {
    fn run_boxed<'a>(
        &'a self,
        ctx: &'a SystemContext<'a>,
    ) -> Result<Box<dyn Any + Send + Sync>, SystemError> {
        let result = self.run(ctx)?;
        Ok(Box::new(result) as Box<dyn Any + Send + Sync>)
    }

    fn reuse_result_boxed(&self, prev: Box<dyn Any + Send + Sync>) {
        if let Ok(typed) = prev.downcast::<S::Result>() {
            self.reuse_result(*typed);
        }
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
///     fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
///         let e = world.spawn();
///         world.insert(e, Transform::IDENTITY).unwrap();
///         Ok(())
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
    ///
    /// Returns `Ok(result)` on success or `Err(SystemError)` on failure.
    fn run(&mut self, world: &mut World) -> Result<Self::Result, SystemError>;

    /// Receive the previous tick's result so allocated memory can be reused.
    ///
    /// Called by the runner before [`run()`](ExclusiveSystem::run) each tick.
    /// The default implementation drops the value.
    #[allow(unused_variables)]
    fn reuse_result(&mut self, prev: Self::Result) {}
}

/// Object-safe version of [`ExclusiveSystem`] used internally for type erasure.
pub(crate) trait DynExclusiveSystem: Send + Sync {
    fn run_boxed(&mut self, world: &mut World) -> Result<Box<dyn Any + Send + Sync>, SystemError>;
    fn reuse_result_boxed(&mut self, prev: Box<dyn Any + Send + Sync>);
}

impl<S: ExclusiveSystem> DynExclusiveSystem for S {
    fn run_boxed(&mut self, world: &mut World) -> Result<Box<dyn Any + Send + Sync>, SystemError> {
        let result = self.run(world)?;
        Ok(Box::new(result) as Box<dyn Any + Send + Sync>)
    }

    fn reuse_result_boxed(&mut self, prev: Box<dyn Any + Send + Sync>) {
        if let Ok(typed) = prev.downcast::<S::Result>() {
            self.reuse_result(*typed);
        }
    }
}

/// A system that receives exclusive `&World` (immutable) access.
///
/// Like [`ExclusiveSystem`], read-only exclusive systems act as **barriers** —
/// no other system runs concurrently. However, they only receive a shared
/// reference to the world, which makes them safe to use in read-only
/// [`SystemsContainer`](crate::SystemsContainer)s.
///
/// Because the world reference is immutable, `run` takes `&self` (not
/// `&mut self`), so the runner only needs a read-lock on the system.
///
/// # When to use
///
/// Use `ReadOnlyExclusiveSystem` when you need to inspect the entire world
/// without the lock-execute pattern — for example, serialization, validation,
/// or building an editor snapshot.
///
/// # Example
///
/// ```ignore
/// struct WorldValidator;
///
/// impl ReadOnlyExclusiveSystem for WorldValidator {
///     type Result = ();
///     fn run(&self, world: &World) -> Result<(), SystemError> {
///         // inspect entities, resources, etc.
///         Ok(())
///     }
/// }
/// ```
pub trait ReadOnlyExclusiveSystem: Send + Sync + 'static {
    /// The value returned by this system after execution.
    type Result: Send + Sync + 'static;

    /// Execute the system with immutable world access.
    fn run(&self, world: &World) -> Result<Self::Result, SystemError>;

    /// Receive the previous tick's result so allocated memory can be reused.
    ///
    /// Called by the runner before [`run()`](Self::run) each tick.
    /// The default implementation drops the value.
    #[allow(unused_variables)]
    fn reuse_result(&self, prev: Self::Result) {}
}

/// Object-safe version of [`ReadOnlyExclusiveSystem`] used internally for type erasure.
pub(crate) trait DynReadOnlyExclusiveSystem: Send + Sync {
    fn run_boxed(&self, world: &World) -> Result<Box<dyn Any + Send + Sync>, SystemError>;
    fn reuse_result_boxed(&self, prev: Box<dyn Any + Send + Sync>);
}

impl<S: ReadOnlyExclusiveSystem> DynReadOnlyExclusiveSystem for S {
    fn run_boxed(&self, world: &World) -> Result<Box<dyn Any + Send + Sync>, SystemError> {
        let result = self.run(world)?;
        Ok(Box::new(result) as Box<dyn Any + Send + Sync>)
    }

    fn reuse_result_boxed(&self, prev: Box<dyn Any + Send + Sync>) {
        if let Ok(typed) = prev.downcast::<S::Result>() {
            self.reuse_result(*typed);
        }
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
    fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
        (self.func)(world);
        Ok(())
    }
}

/// A read-only exclusive system built from a closure.
///
/// Wraps a `Fn(&World)` as a [`ReadOnlyExclusiveSystem`]. Created via
/// [`SystemsContainer::add_read_only_exclusive_fn`](crate::SystemsContainer::add_read_only_exclusive_fn).
pub struct ReadOnlyExclusiveFunctionSystem<F> {
    func: F,
}

impl<F> ReadOnlyExclusiveFunctionSystem<F> {
    /// Creates a new read-only exclusive function system from a closure.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> ReadOnlyExclusiveSystem for ReadOnlyExclusiveFunctionSystem<F>
where
    F: Fn(&World) + Send + Sync + 'static,
{
    type Result = ();
    fn run(&self, world: &World) -> Result<(), SystemError> {
        (self.func)(world);
        Ok(())
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
) -> Result<S::Result, SystemError> {
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
) -> Result<S::Result, SystemError> {
    system.run(world)
}

/// Runs a system once, applying deferred commands and flushing observers.
///
/// Unlike [`run_system_blocking`], this function fully processes side effects:
/// 1. Runs the system, collecting deferred commands
/// 2. Applies all deferred commands to the world
/// 3. Flushes pending observers (may cascade)
///
/// Useful for one-shot gameplay actions, event handlers, and testing.
pub fn run_system_once<S: System>(
    system: &S,
    world: &mut World,
    compute: &ComputePool,
    io: &crate::io_runtime::IoRuntime,
) -> Result<S::Result, SystemError> {
    let commands = CommandCollector::new();
    let result = {
        let ctx = SystemContext::new(world, compute, io, &commands);
        system.run(&ctx)?
    };
    for cmd in commands.drain() {
        cmd(world);
    }
    world.flush_observers();
    Ok(result)
}

/// Runs an exclusive system once, flushing observers afterward.
///
/// Unlike [`run_exclusive_system_blocking`], this function flushes pending
/// observers after the system completes. Since exclusive systems receive
/// `&mut World` directly, any mutations they perform may trigger observers.
pub fn run_exclusive_system_once<S: ExclusiveSystem>(
    system: &mut S,
    world: &mut World,
) -> Result<S::Result, SystemError> {
    let result = system.run(world)?;
    world.flush_observers();
    Ok(result)
}

/// Runs a read-only exclusive system synchronously, returning its result.
///
/// The system receives `&World` directly. Useful for tests and
/// one-off invocations outside a runner.
pub fn run_read_only_exclusive_system_blocking<S: ReadOnlyExclusiveSystem>(
    system: &S,
    world: &World,
) -> Result<S::Result, SystemError> {
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
        fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            Ok(())
        }
    }

    struct MovementSystem;
    impl System for MovementSystem {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            ctx.lock::<(Write<Position>, Read<Velocity>)>().execute(
                |(mut positions, velocities)| {
                    for (idx, pos) in positions.iter_mut() {
                        if let Some(vel) = velocities.get(idx) {
                            pos.x += vel.x;
                        }
                    }
                },
            );
            Ok(())
        }
    }

    #[test]
    fn empty_system_runs() {
        let world = World::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&EmptySystem, &world, &compute, &io).unwrap();
    }

    #[test]
    fn system_runs_blocking() {
        use std::sync::atomic::{AtomicBool, Ordering};

        struct FlagSystem(std::sync::Arc<AtomicBool>);
        impl System for FlagSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.store(true, Ordering::Relaxed);
                Ok(())
            }
        }

        let flag = std::sync::Arc::new(AtomicBool::new(false));
        let sys = FlagSystem(flag.clone());
        let world = World::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io).unwrap();
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
        run_system_blocking(&MovementSystem, &world, &compute, &io).unwrap();

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    // ---- ExclusiveSystem tests ----

    struct SpawnExclusiveSystem;
    impl ExclusiveSystem for SpawnExclusiveSystem {
        type Result = ();
        fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
            let e = world.spawn();
            world.insert(e, Position { x: 99.0 }).unwrap();
            Ok(())
        }
    }

    #[test]
    fn exclusive_system_runs() {
        let mut world = World::new();
        world.register_component::<Position>();

        let mut sys = SpawnExclusiveSystem;
        run_exclusive_system_blocking(&mut sys, &mut world).unwrap();

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
        run_exclusive_system_blocking(&mut sys, &mut world).unwrap();

        assert_eq!(world.entity_count(), 1);
        let e = world.iter_entities().next().unwrap();
        assert_eq!(world.get::<Position>(e).unwrap().x, 42.0);
    }

    #[test]
    fn exclusive_system_with_result() {
        struct CountSystem;
        impl ExclusiveSystem for CountSystem {
            type Result = u32;
            fn run(&mut self, world: &mut World) -> Result<u32, SystemError> {
                Ok(world.entity_count())
            }
        }

        let mut world = World::new();
        world.spawn();
        world.spawn();

        let mut sys = CountSystem;
        let count = run_exclusive_system_blocking(&mut sys, &mut world).unwrap();
        assert_eq!(count, 2);
    }

    // ---- One-shot system tests ----

    #[test]
    fn run_system_once_applies_commands() {
        struct SpawnViaCommands;
        impl System for SpawnViaCommands {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                ctx.commands(|world| {
                    let e = world.spawn();
                    world.insert(e, Position { x: 42.0 }).unwrap();
                });
                Ok(())
            }
        }

        let mut world = World::new();
        world.register_component::<Position>();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        run_system_once(&SpawnViaCommands, &mut world, &compute, &io).unwrap();

        assert_eq!(world.entity_count(), 1);
        let e = world.iter_entities().next().unwrap();
        assert_eq!(world.get::<Position>(e).unwrap().x, 42.0);
    }

    #[test]
    fn run_system_once_flushes_observers() {
        use std::sync::atomic::{AtomicU32, Ordering};

        struct InsertViaCommands;
        impl System for InsertViaCommands {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                ctx.commands(|world| {
                    let e = world.spawn();
                    world.insert(e, Position { x: 1.0 }).unwrap();
                });
                Ok(())
            }
        }

        let counter = std::sync::Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Position>();
        world.observe_add::<Position>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();

        run_system_once(&InsertViaCommands, &mut world, &compute, &io).unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn run_exclusive_system_once_flushes_observers() {
        use std::sync::atomic::{AtomicU32, Ordering};

        struct InsertExclusive;
        impl ExclusiveSystem for InsertExclusive {
            type Result = ();
            fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
                let e = world.spawn();
                world.insert(e, Position { x: 1.0 }).unwrap();
                Ok(())
            }
        }

        let counter = std::sync::Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut world = World::new();
        world.register_component::<Position>();
        world.observe_add::<Position>(move |_world, _entity| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        run_exclusive_system_once(&mut InsertExclusive, &mut world).unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn run_blocking_drives_compute() {
        use crate::Priority;

        struct ComputeSystem(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl System for ComputeSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                let slot = self.0.clone();
                let mut handle = ctx.compute().spawn(Priority::Low, |_ctx| async { 99u32 });
                let result = ctx.compute().block_on(&mut handle);
                *slot.lock().unwrap() = result;
                Ok(())
            }
        }

        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        let sys = ComputeSystem(result.clone());
        let world = World::new();
        let compute = ComputePool::new(IoRuntime::new());
        let io = IoRuntime::new();
        run_system_blocking(&sys, &world, &compute, &io).unwrap();
        assert_eq!(*result.lock().unwrap(), Some(99));
    }

    // ---- ReadOnlyExclusiveSystem tests ----

    struct CountEntities;
    impl ReadOnlyExclusiveSystem for CountEntities {
        type Result = u32;
        fn run(&self, world: &World) -> Result<u32, SystemError> {
            Ok(world.entity_count())
        }
    }

    #[test]
    fn read_only_exclusive_system_runs() {
        let mut world = World::new();
        world.spawn();
        world.spawn();

        let sys = CountEntities;
        let count = run_read_only_exclusive_system_blocking(&sys, &world).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn read_only_exclusive_function_system_runs() {
        let mut world = World::new();
        world.insert_resource(42u32);

        let sys = ReadOnlyExclusiveFunctionSystem::new(|world: &World| {
            assert_eq!(*world.resource::<u32>(), 42);
        });
        run_read_only_exclusive_system_blocking(&sys, &world).unwrap();
    }
}
