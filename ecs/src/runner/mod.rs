mod result_cache;
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
    /// IO tasks or workers did not finish.
    Io(crate::IoShutdownError),
    /// Shutdown timed out with compute tasks still pending.
    Timeout {
        /// Number of compute tasks still running.
        remaining_tasks: usize,
    },
}

impl std::fmt::Display for ShutdownError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShutdownError::Io(error) => write!(f, "{error}"),
            ShutdownError::Timeout { remaining_tasks } => {
                write!(
                    f,
                    "compute quiescence timeout with {remaining_tasks} active tasks"
                )
            }
        }
    }
}

impl std::error::Error for ShutdownError {}

/// Failure to establish the complete module-unload barrier.
#[derive(Debug)]
pub enum ReloadError {
    Shutdown(ShutdownError),
    Workers(crate::ExecutorBusy),
}
impl std::fmt::Display for ReloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shutdown(error) => write!(f, "{error}"),
            Self::Workers(error) => write!(f, "{error}"),
        }
    }
}
impl std::error::Error for ReloadError {}

// IO can spawn compute and compute can spawn IO. Recheck both domains after
// each drain; external producers must remain stopped throughout this operation.
fn drain_tasks(
    compute: &ComputePool,
    io: &IoRuntime,
    timeout: Duration,
) -> Result<(), ShutdownError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let start = std::time::Instant::now();
        loop {
            compute
                .quiesce(timeout.saturating_sub(start.elapsed()))
                .map_err(|error| ShutdownError::Timeout {
                    remaining_tasks: error.remaining_tasks,
                })?;
            io.quiesce(timeout.saturating_sub(start.elapsed()))
                .map_err(ShutdownError::Io)?;
            if compute.active_count() == 0 && io.active_count() == 0 {
                return Ok(());
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = timeout;
        while compute.active_count() > 0 {
            if compute.tick_all() == 0 {
                return Err(ShutdownError::Timeout {
                    remaining_tasks: compute.active_count(),
                });
            }
        }
        io.quiesce(Duration::ZERO).map_err(ShutdownError::Io)
    }
}

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

    /// Drain compute and IO, join their workers/TLS and release cached results.
    /// Stop external producers and release other guest-owned results/worlds first;
    /// keep producers stopped until reload finishes. The IO domain belongs to this
    /// runner; independent host IO should use its own runtime. A failure forbids
    /// unloading the old module. Retry after unfinished work completes.
    /// CPU polls/destructors are cooperative and cannot be forcibly preempted.
    pub fn prepare_reload(&self, timeout: Duration) -> Result<(), ReloadError> {
        #[cfg(not(target_arch = "wasm32"))]
        let start = std::time::Instant::now();
        let remaining = || {
            #[cfg(not(target_arch = "wasm32"))]
            {
                timeout.saturating_sub(start.elapsed())
            }
            #[cfg(target_arch = "wasm32")]
            {
                timeout
            }
        };
        match self {
            Self::SingleThread(runner) => runner.clear_cached_results(),
            #[cfg(not(target_arch = "wasm32"))]
            Self::MultiThread(runner) => runner.clear_cached_results(),
        }
        loop {
            self.graceful_shutdown(remaining())
                .map_err(ReloadError::Shutdown)?;
            self.shutdown_workers().map_err(ReloadError::Workers)?;
            self.io()
                .shutdown(remaining())
                .map_err(|error| ReloadError::Shutdown(ShutdownError::Io(error)))?;
            // IO thread-local destructors can enqueue compute; CPU TLS can in
            // turn start IO. Close the cycle before declaring unload safe.
            if self.compute().active_count() != 0 {
                continue;
            }
            self.shutdown_workers().map_err(ReloadError::Workers)?;
            if self.io().is_stopped() && self.compute().active_count() == 0 {
                return Ok(());
            }
        }
    }

    /// Creates a multi-threaded runner using available parallelism.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn multi_thread_default() -> Self {
        Self::MultiThread(EcsRunnerMultiThread::with_default_threads())
    }

    /// Runs all systems in the container, respecting dependency ordering.
    ///
    /// A container binds to its first admitted world. Another world is rejected
    /// with `SystemError::ScheduleWorldMismatch`, including after reload cleanup.
    /// A new container is rejected with `SystemError::OrphanedScheduleResults`
    /// if this runner retains results from destroyed containers. Clear those
    /// through `prepare_reload` before running replacement schedules.
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

    /// Drains pending compute and IO tasks without stopping their workers.
    ///
    /// Waits until all tasks, including concurrent polls and future
    /// destructors, are drained or the time budget is exceeded. Stop producers
    /// first; request cancellation separately when needed.
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
