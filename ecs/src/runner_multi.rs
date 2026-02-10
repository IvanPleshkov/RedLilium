/// Multi-threaded executor for ECS systems.
///
/// Uses `std::thread::scope` for scoped threading. Systems that need
/// overlapping component access block on RwLock acquisition. Other systems
/// on other threads continue running.
///
/// Not available on `wasm32` targets.
#[cfg(not(target_arch = "wasm32"))]
mod inner {
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::task::Poll;
    use std::time::{Duration, Instant};

    use crate::command_collector::CommandCollector;
    use crate::compute::{ComputePool, noop_waker};
    use crate::runner_single::ShutdownError;
    use crate::system_context::SystemContext;
    use crate::systems_container::SystemsContainer;
    use crate::world::World;
    use crate::world_locks::WorldLocks;

    /// Multi-threaded executor that runs independent systems in parallel.
    ///
    /// Systems are dispatched to OS threads via `std::thread::scope`.
    /// Component access is synchronized through per-TypeId RwLocks acquired
    /// in sorted order to prevent deadlocks.
    pub struct EcsRunnerMultiThread {
        compute: ComputePool,
        num_threads: usize,
    }

    impl EcsRunnerMultiThread {
        /// Creates a new multi-threaded runner with the specified thread count.
        pub fn new(num_threads: usize) -> Self {
            Self {
                compute: ComputePool::new(),
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

        /// Runs all systems respecting dependency ordering, with parallel execution.
        ///
        /// Independent systems (no dependency conflicts) run concurrently on
        /// separate threads. Component access is synchronized via TypeId-ordered
        /// RwLocks in the execute() closure.
        ///
        /// # Time budget
        ///
        /// If the time budget is exceeded, no new systems are started.
        /// Already-running systems complete on their threads.
        pub fn run(&self, world: &mut World, systems: &SystemsContainer, time_budget: Duration) {
            redlilium_core::profile_scope!("ecs: run (multi-thread)");

            let start = Instant::now();
            let n = systems.system_count();
            if n == 0 {
                return;
            }

            let commands = CommandCollector::new();

            // Scope the system execution so ctx and futures are dropped
            // before we need &mut world for command application.
            {
                // Create per-component locks from all registered types
                let type_ids = world.component_type_ids().chain(world.resource_type_ids());
                let world_locks = WorldLocks::new(type_ids);

                let ctx =
                    SystemContext::new_multi_thread(world, &self.compute, &commands, &world_locks);

                // Atomic dependency counters
                let remaining_deps: Vec<AtomicUsize> = systems
                    .in_degrees()
                    .iter()
                    .map(|&d| AtomicUsize::new(d))
                    .collect();

                let (completion_tx, completion_rx) = mpsc::channel::<usize>();
                let mut started = vec![false; n];
                let mut completed_count = 0usize;
                let mut active_count = 0usize;

                std::thread::scope(|scope| {
                    // Inline helper: spawn a system on a scoped thread
                    macro_rules! spawn_system {
                        ($i:expr) => {{
                            started[$i] = true;
                            active_count += 1;
                            let tx = completion_tx.clone();
                            let ctx_ref = &ctx;
                            let compute_ref = &self.compute;
                            let idx = $i;
                            let system_name = systems.get_type_name(idx);
                            scope.spawn(move || {
                                redlilium_core::set_thread_name!("ecs: worker");
                                redlilium_core::profile_scope_dynamic!(system_name);
                                let system = systems.get_system(idx);
                                let future = system.run_boxed(ctx_ref);
                                poll_future_to_completion_with_compute(future, compute_ref);
                                let _ = tx.send(idx);
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
                        let budget_exceeded = start.elapsed() >= time_budget;

                        // Wait for a completion with short timeout (allows compute ticking)
                        match completion_rx.recv_timeout(Duration::from_millis(1)) {
                            Ok(completed_idx) => {
                                completed_count += 1;
                                active_count -= 1;

                                // Decrement dependents and start newly ready systems
                                for &dep in systems.dependents_of(completed_idx) {
                                    let prev = remaining_deps[dep].fetch_sub(1, Ordering::AcqRel);
                                    if prev == 1 && !started[dep] && !budget_exceeded {
                                        spawn_system!(dep);
                                    }
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                            Err(mpsc::RecvTimeoutError::Disconnected) => {
                                break;
                            }
                        }

                        // Tick compute pool on main thread
                        self.compute.tick_all();

                        if budget_exceeded && active_count == 0 {
                            log::warn!(
                                "ECS time budget exceeded with {}/{} systems completed",
                                completed_count,
                                n
                            );
                            break;
                        }
                    }

                    // Drop the sender so scope join doesn't deadlock
                    drop(completion_tx);
                });
            }

            // Apply deferred commands (ctx and futures dropped, world is free)
            {
                redlilium_core::profile_scope!("ecs: apply commands");
                for cmd in commands.drain() {
                    cmd(world);
                }
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
    fn poll_future_to_completion_with_compute<'a>(
        future: Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>>,
        compute: &ComputePool,
    ) {
        let mut future = future;
        let mut future = unsafe { Pin::new_unchecked(&mut future) };
        let waker = noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(()) => break,
                Poll::Pending => {
                    compute.tick_all();
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
            runner.run(&mut world, &container, Duration::from_secs(1));

            assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
        }

        #[test]
        fn run_with_dependencies_multi_thread() {
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
            runner.run(&mut world, &container, Duration::from_secs(1));

            assert_eq!(order.load(Ordering::SeqCst), 2);
        }

        #[test]
        fn deferred_commands_applied_multi_thread() {
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

            let runner = EcsRunnerMultiThread::new(2);
            let mut world = World::new();
            runner.run(&mut world, &container, Duration::from_secs(1));

            assert_eq!(*world.resource::<u32>(), 42);
        }

        #[test]
        fn parallel_independent_systems() {
            use std::sync::Arc;
            use std::sync::atomic::{AtomicU32, Ordering};

            let counter = Arc::new(AtomicU32::new(0));

            struct IncrementA(Arc<AtomicU32>);
            impl System for IncrementA {
                async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }

            struct IncrementB(Arc<AtomicU32>);
            impl System for IncrementB {
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
            runner.run(&mut world, &container, Duration::from_secs(1));

            assert_eq!(counter.load(Ordering::SeqCst), 2);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use inner::EcsRunnerMultiThread;
