use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::task::Poll;
use std::time::{Duration, Instant};

use crate::command_collector::CommandCollector;
use crate::compute::{ComputePool, noop_waker};
use crate::io_runtime::IoRuntime;
use crate::main_thread_dispatcher::{MainThreadDispatcher, RunnerEvent};
use crate::system_context::SystemContext;
use crate::system_results_store::SystemResultsStore;
use crate::systems_container::SystemsContainer;
use crate::world::World;

use super::ShutdownError;

/// Multi-threaded executor that runs independent systems in parallel.
///
/// Systems are dispatched to OS threads via `std::thread::scope`.
/// Component access is synchronized through per-TypeId RwLocks acquired
/// in sorted order to prevent deadlocks.
pub struct EcsRunnerMultiThread {
    compute: ComputePool,
    io: IoRuntime,
    num_threads: usize,
}

impl EcsRunnerMultiThread {
    /// Creates a new multi-threaded runner with the specified thread count.
    pub fn new(num_threads: usize) -> Self {
        let io = IoRuntime::new();
        Self {
            compute: ComputePool::new(io.clone()),
            io,
            num_threads: num_threads.max(1),
        }
    }

    /// Creates a new multi-threaded runner using available parallelism.
    pub fn with_default_threads() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        Self::new(threads)
    }

    /// Returns the configured thread count.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Returns a reference to the compute pool.
    pub fn compute(&self) -> &ComputePool {
        &self.compute
    }

    /// Returns a reference to the IO runtime.
    pub fn io(&self) -> &IoRuntime {
        &self.io
    }

    /// Runs all systems respecting dependency ordering, with parallel execution.
    ///
    /// Independent systems (no dependency conflicts) run concurrently on
    /// separate threads. Component access is synchronized via TypeId-ordered
    /// RwLocks in the execute() closure.
    ///
    /// All systems always run to completion. Deferred commands are applied
    /// after every system has finished.
    pub fn run(&self, world: &mut World, systems: &SystemsContainer) {
        redlilium_core::profile_scope!("ecs: run (multi-thread)");

        let n = systems.system_count();
        if n == 0 {
            return;
        }

        let commands = CommandCollector::new();
        let results_store = SystemResultsStore::new(n, systems.type_id_to_idx().clone());

        // Scope the system execution so ctx and futures are dropped
        // before we need &mut world for command application.
        {
            // Unified event channel for completions AND main-thread dispatch
            let (event_tx, event_rx) = mpsc::channel::<RunnerEvent>();
            let dispatcher = MainThreadDispatcher::new(event_tx.clone());

            // Atomic dependency counters
            let remaining_deps: Vec<AtomicUsize> = systems
                .in_degrees()
                .iter()
                .map(|&d| AtomicUsize::new(d))
                .collect();

            let mut started = vec![false; n];
            let mut completed_count = 0usize;

            std::thread::scope(|scope| {
                // Inline helper: spawn a system on a scoped thread
                macro_rules! spawn_system {
                    ($i:expr) => {{
                        started[$i] = true;
                        let tx = event_tx.clone();
                        let compute_ref = &self.compute;
                        let io_ref = &self.io;
                        let commands_ref = &commands;
                        let dispatcher_ref = &dispatcher;
                        let results_ref = &results_store;
                        let world_ref: &World = world;
                        let idx = $i;
                        let accessible = systems.accessible_results(idx);
                        let system_name = systems.get_type_name(idx);
                        scope.spawn(move || {
                            redlilium_core::set_thread_name!("ecs: worker");
                            redlilium_core::profile_scope_dynamic!(system_name);

                            let ctx = SystemContext::with_dispatcher(
                                world_ref,
                                compute_ref,
                                io_ref,
                                commands_ref,
                                dispatcher_ref,
                            )
                            .with_system_results(results_ref, accessible);

                            let system = systems.get_system(idx);
                            let guard = system.read().unwrap();
                            let future = guard.run_boxed(&ctx);
                            let result =
                                poll_future_to_completion_with_compute(future, compute_ref);
                            results_ref.store(idx, result);
                            let _ = tx.send(RunnerEvent::SystemCompleted(idx));
                        });
                    }};
                }

                // Start initial ready systems
                for i in 0..n {
                    if remaining_deps[i].load(Ordering::Acquire) == 0 {
                        spawn_system!(i);
                    }
                }

                // Coordination loop on the main thread
                while completed_count < n {
                    // Wait for an event with short timeout (allows compute ticking)
                    match event_rx.recv_timeout(Duration::from_millis(1)) {
                        Ok(RunnerEvent::SystemCompleted(completed_idx)) => {
                            completed_count += 1;

                            // Decrement dependents and start newly ready systems
                            for &dep in systems.dependents_of(completed_idx) {
                                let prev = remaining_deps[dep].fetch_sub(1, Ordering::AcqRel);
                                if prev == 1 && !started[dep] {
                                    spawn_system!(dep);
                                }
                            }
                        }
                        Ok(RunnerEvent::MainThreadRequest(work)) => {
                            redlilium_core::profile_scope!("ecs: main-thread dispatch");
                            work();
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            break;
                        }
                    }

                    // Tick compute pool on main thread (budgeted to avoid blocking
                    // on long-running tasks without yields)
                    self.compute.tick_with_budget(Duration::from_millis(1));
                }

                // Drop the sender so scope join doesn't deadlock
                drop(event_tx);
            });
        }

        // Apply deferred commands (ctx and futures dropped, world is free)
        {
            redlilium_core::profile_scope!("ecs: apply commands");
            for cmd in commands.drain() {
                cmd(world);
            }
        }

        // Opportunistically tick remaining compute tasks (time-budgeted).
        // Tasks that don't complete here persist in the pool and continue
        // making progress on subsequent run() calls.
        if self.compute.pending_count() > 0 {
            redlilium_core::profile_scope!("ecs: compute drain");
            self.compute.tick_with_budget(Duration::from_millis(2));
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

impl Default for EcsRunnerMultiThread {
    fn default() -> Self {
        Self::with_default_threads()
    }
}

/// Polls a future to completion, ticking compute between polls.
///
/// Uses budgeted compute ticking so a long-running compute task spawned
/// by another system doesn't block this worker thread indefinitely.
///
/// Returns the type-erased result produced by the system.
fn poll_future_to_completion_with_compute<'a>(
    future: Pin<
        Box<dyn std::future::Future<Output = Box<dyn std::any::Any + Send + Sync>> + Send + 'a>,
    >,
    compute: &ComputePool,
) -> Box<dyn std::any::Any + Send + Sync> {
    let mut future = future;
    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    let waker = noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(result) => break result,
            Poll::Pending => {
                compute.tick_with_budget(Duration::from_millis(1));
                std::thread::yield_now();
            }
        }
    }
}

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
    fn run_single_system_multi_thread() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 10.0 }).unwrap();
        world.insert(e, Velocity { x: 5.0 }).unwrap();

        let mut container = SystemsContainer::new();
        container.add(MovementSystem);

        let runner = EcsRunnerMultiThread::new(2);
        runner.run(&mut world, &container);

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[test]
    fn run_with_dependencies_multi_thread() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let order = Arc::new(AtomicU32::new(0));

        struct FirstSystem(Arc<AtomicU32>);
        impl System for FirstSystem {
            type Result = ();
            async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
                self.0.store(1, Ordering::SeqCst);
            }
        }

        struct SecondSystem(Arc<AtomicU32>);
        impl System for SecondSystem {
            type Result = ();
            async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
                assert_eq!(self.0.load(Ordering::SeqCst), 1);
                self.0.store(2, Ordering::SeqCst);
            }
        }

        let mut container = SystemsContainer::new();
        container.add(FirstSystem(order.clone()));
        container.add(SecondSystem(order.clone()));
        container.add_edge::<FirstSystem, SecondSystem>().unwrap();

        let runner = EcsRunnerMultiThread::new(2);
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(order.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn deferred_commands_applied_multi_thread() {
        struct SpawnSystem;
        impl System for SpawnSystem {
            type Result = ();
            async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
                ctx.commands(|world| {
                    world.insert_resource(42u32);
                });
            }
        }

        let mut container = SystemsContainer::new();
        container.add(SpawnSystem);

        let runner = EcsRunnerMultiThread::new(2);
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*world.resource::<u32>(), 42);
    }

    #[test]
    fn parallel_independent_systems() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let counter = Arc::new(AtomicU32::new(0));

        struct IncrementA(Arc<AtomicU32>);
        impl System for IncrementA {
            type Result = ();
            async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        struct IncrementB(Arc<AtomicU32>);
        impl System for IncrementB {
            type Result = ();
            async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut container = SystemsContainer::new();
        container.add(IncrementA(counter.clone()));
        container.add(IncrementB(counter.clone()));
        // No edges — they can run in parallel

        let runner = EcsRunnerMultiThread::new(4);
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }
}
