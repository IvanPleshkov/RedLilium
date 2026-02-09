use std::time::Duration;

use crate::compute::ComputePool;
use crate::runner_single::{EcsRunnerSingleThread, ShutdownError};
use crate::systems_container::SystemsContainer;
use crate::world::World;

#[cfg(not(target_arch = "wasm32"))]
use crate::runner_multi::EcsRunnerMultiThread;

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
/// runner.run(&mut world, &container, Duration::from_millis(16));
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

    /// Creates a multi-threaded runner with the specified thread count.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn multi_thread(num_threads: usize) -> Self {
        Self::MultiThread(EcsRunnerMultiThread::new(num_threads))
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
    /// Deferred commands are applied after all systems complete.
    ///
    /// # Time budget
    ///
    /// The runner attempts to complete all systems within the time budget.
    /// If exceeded, no new systems are started. Already-running systems
    /// complete normally.
    pub fn run(&self, world: &mut World, systems: &SystemsContainer, time_budget: Duration) {
        match self {
            Self::SingleThread(runner) => runner.run(world, systems, time_budget),
            #[cfg(not(target_arch = "wasm32"))]
            Self::MultiThread(runner) => runner.run(world, systems, time_budget),
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
    use crate::access_set::{Read, Write};
    use crate::system::System;
    use crate::system_context::SystemContext;
    use crate::system_future::SystemFuture;

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }

    struct MovementSystem;
    impl System for MovementSystem {
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> SystemFuture<'a> {
            Box::pin(async move {
                ctx.lock::<(Write<Position>, Read<Velocity>)>()
                    .execute(|(mut positions, velocities)| {
                        for (idx, pos) in positions.iter_mut() {
                            if let Some(vel) = velocities.get(idx) {
                                pos.x += vel.x;
                            }
                        }
                    })
                    .await;
            })
        }
    }

    #[test]
    fn single_thread_runner() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 });
        world.insert(e, Velocity { x: 5.0 });

        let mut container = SystemsContainer::new();
        container.add(MovementSystem);

        let runner = EcsRunner::single_thread();
        runner.run(&mut world, &container, Duration::from_secs(1));

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn multi_thread_runner() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 });
        world.insert(e, Velocity { x: 5.0 });

        let mut container = SystemsContainer::new();
        container.add(MovementSystem);

        let runner = EcsRunner::multi_thread(2);
        runner.run(&mut world, &container, Duration::from_secs(1));

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[test]
    fn graceful_shutdown_succeeds() {
        let runner = EcsRunner::single_thread();
        assert!(runner.graceful_shutdown(Duration::from_secs(1)).is_ok());
    }
}
