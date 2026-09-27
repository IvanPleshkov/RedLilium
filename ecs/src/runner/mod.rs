pub(crate) mod single;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod multi;

pub use single::EcsRunnerSingleThread;

#[cfg(not(target_arch = "wasm32"))]
pub use multi::EcsRunnerMultiThread;

use std::time::Duration;

use crate::compute::ComputePool;
use crate::compute::IoRuntime;
use crate::system::SystemsContainer;
use crate::world::World;

/// Error returned when graceful shutdown exceeds the time budget.
#[derive(Debug)]
pub enum ShutdownError {
    /// Shutdown timed out with tasks still pending.
    Timeout {
        /// Number of compute tasks still running.
        remaining_tasks: usize,
    },
}

impl std::fmt::Display for ShutdownError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShutdownError::Timeout { remaining_tasks } => {
                write!(
                    f,
                    "Shutdown timed out with {remaining_tasks} tasks remaining"
                )
            }
        }
    }
}

impl std::error::Error for ShutdownError {}

/// ECS system executor.
///
/// Dispatches to either single-threaded or multi-threaded execution.
///
/// - [`SingleThread`](EcsRunner::SingleThread): cooperative async executor,
///   zero locking overhead. Works everywhere including WASM.
/// - [`MultiThread`](EcsRunner::MultiThread): thread pool executor with
///   per-component RwLock synchronization. Not available on WASM.
///
/// # Example
///
/// ```ignore
/// let mut runner = EcsRunner::single_thread();
/// runner.run(&mut world, &container);
/// ```
pub enum EcsRunner {
    /// Cooperative single-threaded executor.
    SingleThread(EcsRunnerSingleThread),
    /// Multi-threaded executor with per-component locking.
    #[cfg(not(target_arch = "wasm32"))]
    MultiThread(EcsRunnerMultiThread),
}

impl EcsRunner {
    /// Creates a single-threaded runner.
    pub fn single_thread() -> Self {
        Self::SingleThread(EcsRunnerSingleThread::new())
    }

    /// Sequential systems with parallel queries on an explicit shared pool.
    pub fn single_thread_with_executor(executor: crate::ParallelExecutor) -> Self {
        Self::SingleThread(EcsRunnerSingleThread::with_executor(executor))
    }

    /// Creates a runner with this many background workers, excluding the
    /// coordinating thread. Zero is treated as one.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn multi_thread(num_threads: usize) -> Self {
        Self::MultiThread(EcsRunnerMultiThread::new(num_threads))
    }

    /// Creates a runner sharing an explicit CPU worker pool.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn multi_thread_with_executor(executor: crate::ParallelExecutor) -> Self {
        Self::MultiThread(EcsRunnerMultiThread::with_executor(executor))
    }

    /// Stop system/query workers before unloading a game image. Standalone
    /// world executors must also be shut down or have their last handle dropped
    /// while their guest code is still mapped.
    pub fn shutdown_workers(&self) -> Result<(), crate::ExecutorBusy> {
        match self {
            Self::SingleThread(runner) => runner.executor().shutdown_workers(),
            #[cfg(not(target_arch = "wasm32"))]
            Self::MultiThread(runner) => runner.executor().shutdown_workers(),
        }
    }

    /// Prepare this runner for game-module unload. The caller must stop game
    /// producers and quiesce compute first, and keep them stopped until reload
    /// finishes. Joins workers/TLS and drops cached system results while their
    /// guest vtables and destructors are still mapped. Other runners sharing
    /// the pool must also release their caches before unloading their guests.
    pub fn prepare_reload(&self) -> Result<(), crate::ExecutorBusy> {
        self.shutdown_workers()?;
        match self {
            Self::SingleThread(runner) => runner.clear_cached_results(),
            #[cfg(not(target_arch = "wasm32"))]
            Self::MultiThread(runner) => runner.clear_cached_results(),
        }
        Ok(())
    }

    /// Creates a multi-threaded runner using available parallelism.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn multi_thread_default() -> Self {
        Self::MultiThread(EcsRunnerMultiThread::with_default_threads())
    }

    /// Runs all systems in the container, respecting dependency ordering.
    ///
    /// Systems with no dependencies start immediately. As each system
    /// completes, its dependents become eligible to start.
    ///
    /// All systems always run to completion. Deferred commands are applied
    /// after every system has finished.
    /// Command panics are collected per flush in
    /// [`SystemError::DeferredEffectsFailed`](crate::SystemError::DeferredEffectsFailed).
    /// Later commands and systems continue; partial mutations are retained.
    pub fn run(
        &self,
        world: &mut World,
        systems: &SystemsContainer,
    ) -> Vec<crate::system::SystemError> {
        match self {
            Self::SingleThread(runner) => runner.run(world, systems),
            #[cfg(not(target_arch = "wasm32"))]
            Self::MultiThread(runner) => runner.run(world, systems),
        }
    }

    /// Runs all systems with diagnostics collection controlled by `diagnostics`.
    ///
    /// Returns a [`RunResult`](crate::system::diagnostics::RunResult) containing any
    /// system errors and an optional [`RunReport`](crate::system::diagnostics::RunReport)
    /// with ambiguity and timing information.
    pub fn run_with(
        &self,
        world: &mut World,
        systems: &SystemsContainer,
        diagnostics: &crate::system::diagnostics::RunDiagnostics,
    ) -> crate::system::diagnostics::RunResult {
        match self {
            Self::SingleThread(runner) => runner.run_with(world, systems, diagnostics),
            #[cfg(not(target_arch = "wasm32"))]
            Self::MultiThread(runner) => runner.run_with(world, systems, diagnostics),
        }
    }

    /// Returns a reference to the compute pool owned by this runner.
    pub fn compute(&self) -> &ComputePool {
        match self {
            Self::SingleThread(runner) => runner.compute(),
            #[cfg(not(target_arch = "wasm32"))]
            Self::MultiThread(runner) => runner.compute(),
        }
    }

    /// Returns a reference to the IO runtime owned by this runner.
    pub fn io(&self) -> &IoRuntime {
        match self {
            Self::SingleThread(runner) => runner.io(),
            #[cfg(not(target_arch = "wasm32"))]
            Self::MultiThread(runner) => runner.io(),
        }
    }

    /// Gracefully shuts down the runner, completing pending compute tasks.
    ///
    /// Ticks the compute pool until all tasks are drained or the time
    /// budget is exceeded.
    pub fn graceful_shutdown(&self, time_budget: Duration) -> Result<(), ShutdownError> {
        match self {
            Self::SingleThread(runner) => runner.graceful_shutdown(time_budget),
            #[cfg(not(target_arch = "wasm32"))]
            Self::MultiThread(runner) => runner.graceful_shutdown(time_budget),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{Read, Write};
    use crate::system::System;
    use crate::system::SystemContext;

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }

    struct MovementSystem;
    impl System for MovementSystem {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), crate::system::SystemError> {
            ctx.lock::<(Write<Position>, Read<Velocity>)>().execute(
                |(mut positions, velocities)| {
                    for (idx, mut pos) in positions.iter_mut() {
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
    fn single_thread_runner() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();
        world.insert(e, Velocity { x: 5.0 }).unwrap();

        let mut container = SystemsContainer::new();
        container.add(MovementSystem);

        let runner = EcsRunner::single_thread();
        runner.run(&mut world, &container);

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn multi_thread_runner() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();
        world.insert(e, Velocity { x: 5.0 }).unwrap();

        let mut container = SystemsContainer::new();
        container.add(MovementSystem);

        let runner = EcsRunner::multi_thread(2);
        runner.run(&mut world, &container);

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[test]
    fn graceful_shutdown_succeeds() {
        let runner = EcsRunner::single_thread();
        assert!(runner.graceful_shutdown(Duration::from_secs(1)).is_ok());
    }
}
