use std::pin::Pin;
use std::task::Poll;
use std::time::{Duration, Instant};

use crate::command_collector::CommandCollector;
use crate::compute::{ComputePool, noop_waker};
use crate::system_context::SystemContext;
use crate::systems_container::SystemsContainer;
use crate::world::World;

/// Single-threaded cooperative async executor for ECS systems.
///
/// Polls system futures one at a time in a round-robin fashion,
/// driving the compute pool between polls. No locking overhead.
///
/// Suitable for WASM targets and simple applications.
pub struct EcsRunnerSingleThread {
    compute: ComputePool,
}

impl EcsRunnerSingleThread {
    /// Creates a new single-threaded runner.
    pub fn new() -> Self {
        Self {
            compute: ComputePool::new(),
        }
    }

    /// Returns a reference to the compute pool.
    pub fn compute(&self) -> &ComputePool {
        &self.compute
    }

    /// Runs all systems respecting dependency ordering.
    ///
    /// Systems with no dependencies start immediately. As each system
    /// completes, its dependents become eligible to start. The compute
    /// pool is ticked between system polls to drive background tasks.
    ///
    /// # Time budget
    ///
    /// If the time budget is exceeded, no new systems are started.
    /// Already-running systems are polled to completion, and remaining
    /// compute tasks are ticked until budget + tolerance.
    pub fn run(&self, world: &mut World, systems: &SystemsContainer, time_budget: Duration) {
        let start = Instant::now();
        let n = systems.system_count();
        if n == 0 {
            return;
        }

        let commands = CommandCollector::new();

        // Scope the system execution so ctx and futures are dropped
        // before we need &mut world for command application.
        {
            let ctx = SystemContext::new_single_thread(world, &self.compute, &commands);

            // Track dependency completion
            let mut remaining_deps: Vec<usize> = systems.in_degrees().to_vec();
            let mut completed = vec![false; n];
            let mut futures: Vec<
                Option<Pin<Box<dyn std::future::Future<Output = ()> + Send + '_>>>,
            > = (0..n).map(|_| None).collect();
            let mut completed_count = 0;

            let waker = noop_waker();
            let mut cx = std::task::Context::from_waker(&waker);

            while completed_count < n {
                let mut made_progress = false;

                // Check time budget for starting new systems
                let budget_exceeded = start.elapsed() >= time_budget;

                // Start newly ready systems (if budget not exceeded)
                if !budget_exceeded {
                    for i in 0..n {
                        if remaining_deps[i] == 0 && !completed[i] && futures[i].is_none() {
                            let system = systems.get_system(i);
                            futures[i] = Some(system.run_boxed(&ctx));
                            made_progress = true;
                        }
                    }
                }

                // Poll all running futures
                for i in 0..n {
                    if completed[i] {
                        continue;
                    }
                    if let Some(ref mut future) = futures[i] {
                        match future.as_mut().poll(&mut cx) {
                            Poll::Ready(()) => {
                                futures[i] = None;
                                completed[i] = true;
                                completed_count += 1;
                                made_progress = true;

                                // Decrement dependents
                                for &dep in systems.dependents_of(i) {
                                    remaining_deps[dep] -= 1;
                                }
                            }
                            Poll::Pending => {}
                        }
                    }
                }

                // Tick compute pool
                if self.compute.pending_count() > 0 {
                    self.compute.tick_all();
                    made_progress = true;
                }

                // If no progress and budget exceeded, break (avoid infinite loop)
                if !made_progress {
                    if budget_exceeded {
                        log::warn!(
                            "ECS time budget exceeded with {}/{} systems completed",
                            completed_count,
                            n
                        );
                        break;
                    }
                    // Yield CPU time if nothing to do
                    std::thread::yield_now();
                }
            }
        }

        // Apply deferred commands (ctx and futures dropped, world is free)
        for cmd in commands.drain() {
            cmd(world);
        }
    }

    /// Cancels all pending compute tasks and ticks until drained or timeout.
    pub fn graceful_shutdown(&self, time_budget: Duration) -> Result<(), ShutdownError> {
        let start = Instant::now();
        while self.compute.pending_count() > 0 {
            if start.elapsed() >= time_budget {
                return Err(ShutdownError::Timeout {
                    remaining_tasks: self.compute.pending_count(),
                });
            }
            self.compute.tick_all();
        }
        Ok(())
    }
}

impl Default for EcsRunnerSingleThread {
    fn default() -> Self {
        Self::new()
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_set::{Read, Write};
    use crate::system::System;

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
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
    fn run_empty_container() {
        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        let container = SystemsContainer::new();
        runner.run(&mut world, &container, Duration::from_secs(1));
    }

    #[test]
    fn run_single_system() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();
        world.insert(e, Velocity { x: 5.0 }).unwrap();

        let mut container = SystemsContainer::new();
        container.add(MovementSystem);

        let runner = EcsRunnerSingleThread::new();
        runner.run(&mut world, &container, Duration::from_secs(1));

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[test]
    fn run_with_dependencies() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let order = Arc::new(AtomicU32::new(0));

        struct FirstSystem(Arc<AtomicU32>);
        impl System for FirstSystem {
            async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
                self.0.store(1, Ordering::SeqCst);
            }
        }

        struct SecondSystem(Arc<AtomicU32>);
        impl System for SecondSystem {
            async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
                // Should only run after FirstSystem set value to 1
                assert_eq!(self.0.load(Ordering::SeqCst), 1);
                self.0.store(2, Ordering::SeqCst);
            }
        }

        let mut container = SystemsContainer::new();
        container.add(FirstSystem(order.clone()));
        container.add(SecondSystem(order.clone()));
        container.add_edge::<FirstSystem, SecondSystem>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container, Duration::from_secs(1));

        assert_eq!(order.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn deferred_commands_applied() {
        struct SpawnSystem;
        impl System for SpawnSystem {
            async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
                ctx.commands(|world| {
                    world.insert_resource(42u32);
                });
            }
        }

        let mut container = SystemsContainer::new();
        container.add(SpawnSystem);

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container, Duration::from_secs(1));

        assert_eq!(*world.resource::<u32>(), 42);
    }

    #[test]
    fn graceful_shutdown_empty() {
        let runner = EcsRunnerSingleThread::new();
        assert!(runner.graceful_shutdown(Duration::from_secs(1)).is_ok());
    }
}
