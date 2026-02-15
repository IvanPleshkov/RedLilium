use std::any::Any;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::command_collector::CommandCollector;
use crate::compute::ComputePool;
use crate::io_runtime::IoRuntime;
use crate::main_thread_dispatcher::{MainThreadDispatcher, RunnerEvent};
use crate::system::SystemError;
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
///
/// Exclusive systems act as barriers: when an exclusive system becomes
/// ready, all running parallel systems must complete first, then the
/// exclusive system runs alone with `&mut World`.
pub struct EcsRunnerMultiThread {
    compute: ComputePool,
    io: IoRuntime,
    num_threads: usize,
    prev_results: Mutex<Vec<Option<Box<dyn Any + Send + Sync>>>>,
}

impl EcsRunnerMultiThread {
    /// Creates a new multi-threaded runner with the specified thread count.
    pub fn new(num_threads: usize) -> Self {
        let io = IoRuntime::new();
        Self {
            compute: ComputePool::new(io.clone()),
            io,
            num_threads: num_threads.max(1),
            prev_results: Mutex::new(Vec::new()),
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
    /// Independent regular systems run concurrently on separate threads.
    /// Exclusive systems act as barriers — they run alone with `&mut World`
    /// after all preceding parallel systems have completed. Pending deferred
    /// commands are applied before each exclusive system.
    ///
    /// All systems always run to completion. Remaining deferred commands
    /// are applied after every system has finished.
    pub fn run(&self, world: &mut World, systems: &SystemsContainer) -> Vec<SystemError> {
        redlilium_core::profile_scope!("ecs: run (multi-thread)");

        let n = systems.system_count();
        if n == 0 {
            return Vec::new();
        }

        let mut errors = Vec::new();
        let commands = CommandCollector::new();
        let results_store = SystemResultsStore::new(n, systems.type_id_to_idx().clone());

        let mut remaining_deps: Vec<usize> = systems.in_degrees().to_vec();
        let mut started = vec![false; n];
        let mut completed_count = 0usize;

        // Take previous-tick results (if the system count matches).
        let prev = std::mem::take(&mut *self.prev_results.lock().unwrap());
        let prev = Mutex::new(if prev.len() == n { prev } else { Vec::new() });

        while completed_count < n {
            // Collect ready systems
            let mut exclusive_ready = None;
            let mut regular_ready = Vec::new();

            for i in 0..n {
                if !started[i] && remaining_deps[i] == 0 {
                    if systems.is_exclusive(i) {
                        if exclusive_ready.is_none() {
                            exclusive_ready = Some(i);
                        }
                    } else {
                        regular_ready.push(i);
                    }
                }
            }

            if let Some(exc_idx) = exclusive_ready {
                // Run any ready regular systems first (they may be independent
                // of the exclusive system) to maximize parallelism before the barrier.
                if !regular_ready.is_empty() {
                    errors.extend(self.run_parallel_phase(
                        world,
                        systems,
                        &commands,
                        &results_store,
                        &mut remaining_deps,
                        &mut started,
                        &mut completed_count,
                        &regular_ready,
                        &prev,
                    ));
                }

                // Apply pending deferred commands so the exclusive system
                // sees structural changes from predecessors.
                {
                    redlilium_core::profile_scope!("ecs: apply commands (pre-exclusive)");
                    for cmd in commands.drain() {
                        cmd(world);
                    }
                }

                // Run exclusive system with &mut World
                started[exc_idx] = true;
                {
                    let system_name = systems.get_type_name(exc_idx);
                    redlilium_core::profile_scope_dynamic!(system_name);

                    let prev_result = {
                        let mut prev_guard = prev.lock().unwrap();
                        if exc_idx < prev_guard.len() {
                            prev_guard[exc_idx].take()
                        } else {
                            None
                        }
                    };

                    let system = systems.get_exclusive_system(exc_idx);
                    let mut guard = system.write().unwrap();
                    if let Some(prev_result) = prev_result {
                        guard.reuse_result_boxed(prev_result);
                    }
                    match guard.run_boxed(world) {
                        Ok(result) => results_store.store(exc_idx, result),
                        Err(e) => errors.push(e),
                    }
                }
                completed_count += 1;
                for &dep in systems.dependents_of(exc_idx) {
                    remaining_deps[dep] -= 1;
                }
            } else if !regular_ready.is_empty() {
                // All ready systems are regular — run them in parallel
                errors.extend(self.run_parallel_phase(
                    world,
                    systems,
                    &commands,
                    &results_store,
                    &mut remaining_deps,
                    &mut started,
                    &mut completed_count,
                    &regular_ready,
                    &prev,
                ));
            } else {
                // No ready systems — should not happen with a valid DAG
                break;
            }
        }

        // Apply remaining deferred commands
        {
            redlilium_core::profile_scope!("ecs: apply commands");
            for cmd in commands.drain() {
                cmd(world);
            }
        }

        // Save this tick's results for next tick's reuse.
        *self.prev_results.lock().unwrap() = results_store.into_prev_results();

        // Opportunistically tick remaining compute tasks (time-budgeted).
        if self.compute.pending_count() > 0 {
            redlilium_core::profile_scope!("ecs: compute drain");
            self.compute.tick_with_budget(Duration::from_millis(2));
        }

        errors
    }

    /// Runs a batch of regular systems in parallel using a scoped thread pool.
    ///
    /// Systems that become ready during execution (due to completions) are
    /// also spawned — unless they are exclusive, in which case they are
    /// deferred to the caller.
    #[allow(clippy::too_many_arguments)]
    fn run_parallel_phase(
        &self,
        world: &mut World,
        systems: &SystemsContainer,
        commands: &CommandCollector,
        results_store: &SystemResultsStore,
        remaining_deps: &mut [usize],
        started: &mut [bool],
        completed_count: &mut usize,
        initial_ready: &[usize],
        prev_results: &Mutex<Vec<Option<Box<dyn Any + Send + Sync>>>>,
    ) -> Vec<SystemError> {
        let (event_tx, event_rx) = mpsc::channel::<RunnerEvent>();
        let dispatcher = MainThreadDispatcher::new(event_tx.clone());
        let mut active_count = 0usize;
        let thread_errors = std::sync::Mutex::new(Vec::<SystemError>::new());

        std::thread::scope(|scope| {
            macro_rules! spawn_system {
                ($i:expr) => {{
                    started[$i] = true;
                    active_count += 1;
                    let tx = event_tx.clone();
                    let compute_ref = &self.compute;
                    let io_ref = &self.io;
                    let commands_ref = commands;
                    let dispatcher_ref = &dispatcher;
                    let results_ref = results_store;
                    let prev_ref = prev_results;
                    let world_ref: &World = world;
                    let idx = $i;
                    let accessible = systems.accessible_results(idx);
                    let system_name = systems.get_type_name(idx);
                    let errors_ref = &thread_errors;
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

                        let prev_result = {
                            let mut prev_guard = prev_ref.lock().unwrap();
                            if idx < prev_guard.len() {
                                prev_guard[idx].take()
                            } else {
                                None
                            }
                        };

                        let system = systems.get_system(idx);
                        let guard = system.read().unwrap();
                        if let Some(prev_result) = prev_result {
                            guard.reuse_result_boxed(prev_result);
                        }
                        match guard.run_boxed(&ctx) {
                            Ok(result) => results_ref.store(idx, result),
                            Err(e) => errors_ref.lock().unwrap().push(e),
                        }
                        let _ = tx.send(RunnerEvent::SystemCompleted(idx));
                    });
                }};
            }

            // Start initial ready regular systems
            for &i in initial_ready {
                spawn_system!(i);
            }

            // Coordination loop — runs until all active systems complete.
            // Newly ready regular systems are spawned immediately;
            // exclusive systems are left for the outer loop.
            while active_count > 0 {
                match event_rx.recv_timeout(Duration::from_millis(1)) {
                    Ok(RunnerEvent::SystemCompleted(completed_idx)) => {
                        active_count -= 1;
                        *completed_count += 1;

                        for &dep in systems.dependents_of(completed_idx) {
                            remaining_deps[dep] -= 1;
                            if remaining_deps[dep] == 0
                                && !started[dep]
                                && !systems.is_exclusive(dep)
                            {
                                spawn_system!(dep);
                            }
                            // Exclusive systems deferred to outer loop
                        }
                    }
                    Ok(RunnerEvent::MainThreadRequest(work)) => {
                        redlilium_core::profile_scope!("ecs: main-thread dispatch");
                        work();
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }

                self.compute.tick_with_budget(Duration::from_millis(1));
            }

            drop(event_tx);
        });

        thread_errors.into_inner().unwrap()
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
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), crate::system::SystemError> {
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
            fn run<'a>(
                &'a self,
                _ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                self.0.store(1, Ordering::SeqCst);
                Ok(())
            }
        }

        struct SecondSystem(Arc<AtomicU32>);
        impl System for SecondSystem {
            type Result = ();
            fn run<'a>(
                &'a self,
                _ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                assert_eq!(self.0.load(Ordering::SeqCst), 1);
                self.0.store(2, Ordering::SeqCst);
                Ok(())
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
            fn run<'a>(
                &'a self,
                ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                ctx.commands(|world| {
                    world.insert_resource(42u32);
                });
                Ok(())
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
            fn run<'a>(
                &'a self,
                _ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        struct IncrementB(Arc<AtomicU32>);
        impl System for IncrementB {
            type Result = ();
            fn run<'a>(
                &'a self,
                _ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
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

    // ---- Exclusive system tests ----

    use crate::system::ExclusiveSystem;

    #[test]
    fn exclusive_system_multi_thread() {
        struct ExclSystem;
        impl ExclusiveSystem for ExclSystem {
            type Result = ();
            fn run(&mut self, world: &mut World) -> Result<(), crate::system::SystemError> {
                world.insert_resource(99u32);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add_exclusive(ExclSystem);

        let runner = EcsRunnerMultiThread::new(2);
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*world.resource::<u32>(), 99);
    }

    #[test]
    fn exclusive_sees_commands_multi_thread() {
        struct RegularSystem;
        impl System for RegularSystem {
            type Result = ();
            fn run<'a>(
                &'a self,
                ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                ctx.commands(|world| {
                    world.insert_resource(42u32);
                });
                Ok(())
            }
        }

        struct ExclSystem(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl ExclusiveSystem for ExclSystem {
            type Result = ();
            fn run(&mut self, world: &mut World) -> Result<(), crate::system::SystemError> {
                let val = *world.resource::<u32>();
                *self.0.lock().unwrap() = Some(val);
                Ok(())
            }
        }

        let observed = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut container = SystemsContainer::new();
        container.add(RegularSystem);
        container.add_exclusive(ExclSystem(observed.clone()));
        container.add_edge::<RegularSystem, ExclSystem>().unwrap();

        let runner = EcsRunnerMultiThread::new(2);
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*observed.lock().unwrap(), Some(42));
    }

    #[test]
    fn mixed_chain_multi_thread() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let order = Arc::new(AtomicU32::new(0));

        struct First(Arc<AtomicU32>);
        impl System for First {
            type Result = ();
            fn run<'a>(
                &'a self,
                _ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                self.0.store(1, Ordering::SeqCst);
                Ok(())
            }
        }

        struct Middle(Arc<AtomicU32>);
        impl ExclusiveSystem for Middle {
            type Result = ();
            fn run(&mut self, _world: &mut World) -> Result<(), crate::system::SystemError> {
                assert_eq!(self.0.load(Ordering::SeqCst), 1);
                self.0.store(2, Ordering::SeqCst);
                Ok(())
            }
        }

        struct Last(Arc<AtomicU32>);
        impl System for Last {
            type Result = ();
            fn run<'a>(
                &'a self,
                _ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                assert_eq!(self.0.load(Ordering::SeqCst), 2);
                self.0.store(3, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add(First(order.clone()));
        container.add_exclusive(Middle(order.clone()));
        container.add(Last(order.clone()));
        container.add_edge::<First, Middle>().unwrap();
        container.add_edge::<Middle, Last>().unwrap();

        let runner = EcsRunnerMultiThread::new(4);
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(order.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn exclusive_result_accessible_multi_thread() {
        struct ExclProducer;
        impl ExclusiveSystem for ExclProducer {
            type Result = u32;
            fn run(&mut self, _world: &mut World) -> Result<u32, crate::system::SystemError> {
                Ok(42)
            }
        }

        struct Consumer(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl System for Consumer {
            type Result = ();
            fn run<'a>(
                &'a self,
                ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                let val = *ctx.exclusive_system_result::<ExclProducer>();
                *self.0.lock().unwrap() = Some(val);
                Ok(())
            }
        }

        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut container = SystemsContainer::new();
        container.add_exclusive(ExclProducer);
        container.add(Consumer(result.clone()));
        container.add_edge::<ExclProducer, Consumer>().unwrap();

        let runner = EcsRunnerMultiThread::new(2);
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*result.lock().unwrap(), Some(42));
    }

    #[test]
    fn parallel_systems_before_exclusive_barrier() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let counter = Arc::new(AtomicU32::new(0));

        struct ParallelA(Arc<AtomicU32>);
        impl System for ParallelA {
            type Result = ();
            fn run<'a>(
                &'a self,
                _ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        struct ParallelB(Arc<AtomicU32>);
        impl System for ParallelB {
            type Result = ();
            fn run<'a>(
                &'a self,
                _ctx: &'a SystemContext<'a>,
            ) -> Result<(), crate::system::SystemError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        struct Barrier(Arc<AtomicU32>);
        impl ExclusiveSystem for Barrier {
            type Result = ();
            fn run(&mut self, _world: &mut World) -> Result<(), crate::system::SystemError> {
                // Both parallel systems should have completed
                assert_eq!(self.0.load(Ordering::SeqCst), 2);
                self.0.store(10, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add(ParallelA(counter.clone()));
        container.add(ParallelB(counter.clone()));
        container.add_exclusive(Barrier(counter.clone()));
        container.add_edge::<ParallelA, Barrier>().unwrap();
        container.add_edge::<ParallelB, Barrier>().unwrap();

        let runner = EcsRunnerMultiThread::new(4);
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }
}
