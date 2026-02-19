use std::any::Any;
use std::sync::Mutex;
use std::time::Duration;

use crate::command_collector::CommandCollector;
use crate::compute::ComputePool;
use crate::diagnostics::{
    AccessRecorder, RunDiagnostics, RunReport, RunResult, SystemTiming, TimingReport,
    analyze_ambiguities,
};
use crate::io_runtime::IoRuntime;
use crate::system::{SystemError, panic_payload_to_string};
use crate::system_context::SystemContext;
use crate::system_results_store::SystemResultsStore;
use crate::systems_container::SystemsContainer;
use crate::world::World;

use super::ShutdownError;

/// Single-threaded sequential executor for ECS systems.
///
/// Runs each system to completion in pre-computed topological order.
/// No locking overhead, no dependency tracking at runtime.
///
/// Suitable for WASM targets and simple applications.
pub struct EcsRunnerSingleThread {
    compute: ComputePool,
    io: IoRuntime,
    prev_results: Mutex<Vec<Option<Box<dyn Any + Send + Sync>>>>,
}

impl EcsRunnerSingleThread {
    /// Creates a new single-threaded runner.
    pub fn new() -> Self {
        let io = IoRuntime::new();
        Self {
            compute: ComputePool::new(io.clone()),
            io,
            prev_results: Mutex::new(Vec::new()),
        }
    }

    /// Returns a reference to the compute pool.
    pub fn compute(&self) -> &ComputePool {
        &self.compute
    }

    /// Returns a reference to the IO runtime.
    pub fn io(&self) -> &IoRuntime {
        &self.io
    }

    /// Runs all systems in topological order, one at a time.
    ///
    /// Each system runs to completion before the next one starts.
    /// The compute pool is driven between polls so spawned tasks
    /// make progress.
    pub fn run(&self, world: &mut World, systems: &SystemsContainer) -> Vec<SystemError> {
        self.run_with(world, systems, &RunDiagnostics::default())
            .errors
    }

    /// Runs all systems with optional diagnostics collection.
    ///
    /// Like [`run()`](Self::run), but accepts a [`RunDiagnostics`] config
    /// and returns a [`RunResult`] containing both errors and a diagnostic
    /// report.
    pub fn run_with(
        &self,
        world: &mut World,
        systems: &SystemsContainer,
        diagnostics: &RunDiagnostics,
    ) -> RunResult {
        redlilium_core::profile_scope!("ecs: run (single-thread)");

        let order = systems.single_thread_order();
        if order.is_empty() {
            return RunResult {
                errors: Vec::new(),
                report: RunReport::default(),
            };
        }

        let n = systems.node_count();
        let mut errors = Vec::new();
        let commands = CommandCollector::new();
        let results_store = SystemResultsStore::new(n, systems.type_id_to_idx().clone());

        // Optional diagnostics state
        let recorder = if diagnostics.detect_ambiguities {
            Some(AccessRecorder::new(n))
        } else {
            None
        };
        let mut system_timings: Vec<SystemTiming> = Vec::new();

        #[cfg(not(target_arch = "wasm32"))]
        let run_start = if diagnostics.collect_timings {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Swap reactive trigger buffers (last tick's collecting → readable).
        world.update_triggers();

        // Take previous-tick results (if the system count matches).
        let prev = std::mem::take(&mut *self.prev_results.lock().unwrap());
        let mut prev = if prev.len() == n { prev } else { Vec::new() };

        {
            for &idx in order {
                // Skip virtual barrier nodes — no system to execute.
                if systems.is_virtual(idx) {
                    continue;
                }

                // Check run conditions — skip this system if they fail.
                if !systems.check_conditions(idx, &results_store) {
                    continue;
                }

                // Feed previous result to the system for memory reuse.
                let prev_result = if idx < prev.len() {
                    prev[idx].take()
                } else {
                    None
                };

                if systems.is_exclusive(idx) {
                    // Apply pending deferred commands so the exclusive system
                    // sees structural changes from predecessors.
                    {
                        redlilium_core::profile_scope!("ecs: apply commands (pre-exclusive)");
                        for cmd in commands.drain() {
                            cmd(world);
                        }
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    let sys_start = if diagnostics.collect_timings {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };

                    let system = systems.get_exclusive_system(idx);
                    let mut guard = system.write().unwrap();
                    if let Some(prev_result) = prev_result {
                        guard.reuse_result_boxed(prev_result);
                    }
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        guard.run_boxed(world)
                    })) {
                        Ok(Ok(result)) => results_store.store(idx, result),
                        Ok(Err(e)) => errors.push(e),
                        Err(payload) => {
                            errors.push(SystemError::Panicked(panic_payload_to_string(&*payload)));
                        }
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(start) = sys_start {
                        system_timings.push(SystemTiming {
                            name: systems.get_type_name(idx),
                            duration: start.elapsed(),
                        });
                    }
                } else {
                    let mut ctx = SystemContext::new(world, &self.compute, &self.io, &commands)
                        .with_system_results(&results_store, systems.accessible_results(idx));
                    if let Some(ref rec) = recorder {
                        ctx = ctx.with_access_recorder(rec, idx);
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    let sys_start = if diagnostics.collect_timings {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };

                    let system = systems.get_system(idx);
                    let guard = system.read().unwrap();
                    if let Some(prev_result) = prev_result {
                        guard.reuse_result_boxed(prev_result);
                    }
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        guard.run_boxed(&ctx)
                    })) {
                        Ok(Ok(result)) => results_store.store(idx, result),
                        Ok(Err(e)) => errors.push(e),
                        Err(payload) => {
                            errors.push(SystemError::Panicked(panic_payload_to_string(&*payload)));
                        }
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(start) = sys_start {
                        system_timings.push(SystemTiming {
                            name: systems.get_type_name(idx),
                            duration: start.elapsed(),
                        });
                    }
                }
            }
        }

        // Apply deferred commands (ctx dropped, world is free)
        {
            redlilium_core::profile_scope!("ecs: apply commands");
            for cmd in commands.drain() {
                cmd(world);
            }
        }

        // Flush deferred observers (may cascade)
        {
            redlilium_core::profile_scope!("ecs: flush observers");
            world.flush_observers();
        }

        // Drain remaining compute tasks (one poll per task)
        if self.compute.pending_count() > 0 {
            redlilium_core::profile_scope!("ecs: compute drain");
            self.compute.tick_all();
        }

        // Save this tick's results for next tick's reuse.
        *self.prev_results.lock().unwrap() = results_store.into_prev_results();

        // Build diagnostic report
        let ambiguities = if diagnostics.detect_ambiguities {
            Some(analyze_ambiguities(
                recorder.unwrap().into_records(),
                systems,
                world,
            ))
        } else {
            None
        };

        #[cfg(not(target_arch = "wasm32"))]
        let timings = if let Some(start) = run_start {
            let wall_time = start.elapsed();
            let total_cpu_time = system_timings.iter().map(|t| t.duration).sum();
            Some(TimingReport {
                wall_time,
                total_cpu_time,
                num_threads: 1,
                systems: system_timings,
            })
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let timings = None;

        RunResult {
            errors,
            report: RunReport {
                ambiguities,
                timings,
            },
        }
    }

    /// Cancels all pending compute tasks and ticks until drained or timeout.
    pub fn graceful_shutdown(&self, _time_budget: Duration) -> Result<(), ShutdownError> {
        #[cfg(not(target_arch = "wasm32"))]
        let start = std::time::Instant::now();

        while self.compute.pending_count() > 0 {
            #[cfg(not(target_arch = "wasm32"))]
            if start.elapsed() >= _time_budget {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_set::{Read, Write};
    use crate::system::System;
    use crate::system::SystemError;

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
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
    fn run_empty_container() {
        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        let container = SystemsContainer::new();
        runner.run(&mut world, &container);
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
        runner.run(&mut world, &container);

        assert_eq!(world.get::<Position>(e).unwrap().x, 15.0);
    }

    #[test]
    fn run_with_dependencies() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let order = Arc::new(AtomicU32::new(0));

        struct FirstSystem(Arc<AtomicU32>);
        impl System for FirstSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.store(1, Ordering::SeqCst);
                Ok(())
            }
        }

        struct SecondSystem(Arc<AtomicU32>);
        impl System for SecondSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                // Should only run after FirstSystem set value to 1
                assert_eq!(self.0.load(Ordering::SeqCst), 1);
                self.0.store(2, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add(FirstSystem(order.clone()));
        container.add(SecondSystem(order.clone()));
        container.add_edge::<FirstSystem, SecondSystem>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(order.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn deferred_commands_applied() {
        struct SpawnSystem;
        impl System for SpawnSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                ctx.commands(|world| {
                    world.insert_resource(42u32);
                });
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add(SpawnSystem);

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*world.resource::<u32>(), 42);
    }

    #[test]
    fn graceful_shutdown_empty() {
        let runner = EcsRunnerSingleThread::new();
        assert!(runner.graceful_shutdown(Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn system_result_accessible_by_dependent() {
        struct ProducerSystem;
        impl System for ProducerSystem {
            type Result = u32;
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<u32, SystemError> {
                Ok(42)
            }
        }

        struct ConsumerSystem(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl System for ConsumerSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                let value = *ctx.system_result::<ProducerSystem>();
                *self.0.lock().unwrap() = Some(value);
                Ok(())
            }
        }

        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut container = SystemsContainer::new();
        container.add(ProducerSystem);
        container.add(ConsumerSystem(result.clone()));
        container
            .add_edge::<ProducerSystem, ConsumerSystem>()
            .unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*result.lock().unwrap(), Some(42));
    }

    #[test]
    fn system_result_without_edge_returns_error() {
        struct ProducerSystem;
        impl System for ProducerSystem {
            type Result = u32;
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<u32, SystemError> {
                Ok(42)
            }
        }

        struct ConsumerSystem;
        impl System for ConsumerSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                let _ = ctx.system_result::<ProducerSystem>();
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add(ProducerSystem);
        container.add(ConsumerSystem);
        // No edge — consumer's panic is caught by the runner

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        let errors = runner.run(&mut world, &container);

        assert_eq!(errors.len(), 1);
        assert!(errors[0].to_string().contains("not accessible"));
    }

    #[test]
    fn system_result_transitive_access() {
        struct SystemA;
        impl System for SystemA {
            type Result = String;
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<String, SystemError> {
                Ok("hello".to_string())
            }
        }

        struct SystemB;
        impl System for SystemB {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                Ok(())
            }
        }

        // C depends on B depends on A, so C should be able to read A's result
        struct SystemC(std::sync::Arc<std::sync::Mutex<Option<String>>>);
        impl System for SystemC {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                let value = ctx.system_result::<SystemA>().clone();
                *self.0.lock().unwrap() = Some(value);
                Ok(())
            }
        }

        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add(SystemC(result.clone()));
        container.add_edge::<SystemA, SystemB>().unwrap();
        container.add_edge::<SystemB, SystemC>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*result.lock().unwrap(), Some("hello".to_string()));
    }

    // ---- Exclusive system tests ----

    use crate::system::ExclusiveSystem;

    #[test]
    fn exclusive_system_modifies_world() {
        struct SpawnSystem;
        impl ExclusiveSystem for SpawnSystem {
            type Result = ();
            fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
                world.insert_resource(99u32);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add_exclusive(SpawnSystem);

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*world.resource::<u32>(), 99);
    }

    #[test]
    fn exclusive_sees_commands_from_predecessor() {
        struct RegularSystem;
        impl System for RegularSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                ctx.commands(|world| {
                    world.insert_resource(42u32);
                });
                Ok(())
            }
        }

        struct ExclSystem(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl ExclusiveSystem for ExclSystem {
            type Result = ();
            fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
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

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*observed.lock().unwrap(), Some(42));
    }

    #[test]
    fn mixed_chain_regular_exclusive_regular() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let order = Arc::new(AtomicU32::new(0));

        struct First(Arc<AtomicU32>);
        impl System for First {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.store(1, Ordering::SeqCst);
                Ok(())
            }
        }

        struct Middle(Arc<AtomicU32>);
        impl ExclusiveSystem for Middle {
            type Result = ();
            fn run(&mut self, _world: &mut World) -> Result<(), SystemError> {
                assert_eq!(self.0.load(Ordering::SeqCst), 1);
                self.0.store(2, Ordering::SeqCst);
                Ok(())
            }
        }

        struct Last(Arc<AtomicU32>);
        impl System for Last {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
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

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(order.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn exclusive_system_result_accessible() {
        struct ExclProducer;
        impl ExclusiveSystem for ExclProducer {
            type Result = u32;
            fn run(&mut self, _world: &mut World) -> Result<u32, SystemError> {
                Ok(42)
            }
        }

        struct Consumer(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl System for Consumer {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
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

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*result.lock().unwrap(), Some(42));
    }

    // ---- Run condition tests ----

    use crate::condition::{Condition, ConditionMode};

    struct CondTrueSystem;
    impl System for CondTrueSystem {
        type Result = Condition<()>;
        fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<Condition<()>, SystemError> {
            Ok(Condition::True(()))
        }
    }

    struct CondFalseSystem;
    impl System for CondFalseSystem {
        type Result = Condition<()>;
        fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<Condition<()>, SystemError> {
            Ok(Condition::False)
        }
    }

    #[test]
    fn condition_true_allows_system() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let ran = Arc::new(AtomicBool::new(false));

        struct Target(Arc<AtomicBool>);
        impl System for Target {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add_condition(CondTrueSystem);
        container.add(Target(ran.clone()));
        container.add_edge::<CondTrueSystem, Target>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert!(ran.load(Ordering::SeqCst));
    }

    #[test]
    fn condition_false_skips_system() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let ran = Arc::new(AtomicBool::new(false));

        struct Target(Arc<AtomicBool>);
        impl System for Target {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add_condition(CondFalseSystem);
        container.add(Target(ran.clone()));
        container.add_edge::<CondFalseSystem, Target>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert!(!ran.load(Ordering::SeqCst));
    }

    #[test]
    fn skipped_system_dependents_still_run() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let counter = Arc::new(AtomicU32::new(0));

        struct Gated(Arc<AtomicU32>);
        impl System for Gated {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.fetch_add(10, Ordering::SeqCst);
                Ok(())
            }
        }

        struct Downstream(Arc<AtomicU32>);
        impl System for Downstream {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add_condition(CondFalseSystem);
        container.add(Gated(counter.clone()));
        container.add(Downstream(counter.clone()));
        container.add_edge::<CondFalseSystem, Gated>().unwrap();
        container.add_edge::<Gated, Downstream>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        // Gated was skipped (no +10), but Downstream still ran (+1)
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn multiple_conditions_all_mode() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let ran = Arc::new(AtomicBool::new(false));

        struct Target(Arc<AtomicBool>);
        impl System for Target {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add_condition(CondTrueSystem);
        container.add_condition(CondFalseSystem);
        container.add(Target(ran.clone()));
        container.add_edge::<CondTrueSystem, Target>().unwrap();
        container.add_edge::<CondFalseSystem, Target>().unwrap();
        // Default: All mode — both must be true

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert!(!ran.load(Ordering::SeqCst));
    }

    #[test]
    fn multiple_conditions_any_mode() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let ran = Arc::new(AtomicBool::new(false));

        struct Target(Arc<AtomicBool>);
        impl System for Target {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add_condition(CondTrueSystem);
        container.add_condition(CondFalseSystem);
        container.add(Target(ran.clone()));
        container.add_edge::<CondTrueSystem, Target>().unwrap();
        container.add_edge::<CondFalseSystem, Target>().unwrap();
        container.set_condition_mode::<Target>(ConditionMode::Any);

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert!(ran.load(Ordering::SeqCst));
    }

    #[test]
    fn condition_gates_exclusive_system() {
        struct ExclTarget;
        impl ExclusiveSystem for ExclTarget {
            type Result = ();
            fn run(&mut self, world: &mut World) -> Result<(), SystemError> {
                world.insert_resource(99u32);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add_condition(CondFalseSystem);
        container.add_exclusive(ExclTarget);
        container.add_edge::<CondFalseSystem, ExclTarget>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        // ExclTarget was skipped — resource not inserted
        assert!(!world.has_resource::<u32>());
    }

    #[test]
    fn condition_with_payload_piping() {
        struct ConfigCondition;
        impl System for ConfigCondition {
            type Result = Condition<u32>;
            fn run<'a>(
                &'a self,
                _ctx: &'a SystemContext<'a>,
            ) -> Result<Condition<u32>, SystemError> {
                Ok(Condition::True(42))
            }
        }

        struct Reader(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl System for Reader {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                let cond = ctx.system_result::<ConfigCondition>();
                *self.0.lock().unwrap() = cond.value().copied();
                Ok(())
            }
        }

        let result = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut container = SystemsContainer::new();
        container.add_condition(ConfigCondition);
        container.add(Reader(result.clone()));
        container.add_edge::<ConfigCondition, Reader>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(*result.lock().unwrap(), Some(42));
    }

    // ---- System set tests ----

    #[test]
    fn set_ordering_enforced_single_thread() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        let counter = Arc::new(AtomicU32::new(0));

        struct AlphaSystem(Arc<AtomicU32>);
        impl System for AlphaSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                // Should run first; store 1
                assert_eq!(self.0.load(Ordering::SeqCst), 0);
                self.0.store(1, Ordering::SeqCst);
                Ok(())
            }
        }

        struct BetaSystem(Arc<AtomicU32>);
        impl System for BetaSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                // Should run after AlphaSystem
                assert_eq!(self.0.load(Ordering::SeqCst), 1);
                self.0.store(2, Ordering::SeqCst);
                Ok(())
            }
        }

        struct SetA;
        impl crate::SystemSet for SetA {}
        struct SetB;
        impl crate::SystemSet for SetB {}

        let mut container = SystemsContainer::new();
        container.add(AlphaSystem(counter.clone()));
        container.add(BetaSystem(counter.clone()));
        container.add_to_set::<AlphaSystem, SetA>().unwrap();
        container.add_to_set::<BetaSystem, SetB>().unwrap();
        container.add_set_edge::<SetA, SetB>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);

        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    // ---- Panic protection tests ----

    #[test]
    fn panicking_system_caught_single_thread() {
        struct PanicSystem;
        impl System for PanicSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                panic!("system boom");
            }
        }

        let mut container = SystemsContainer::new();
        container.add(PanicSystem);

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        let errors = runner.run(&mut world, &container);

        assert_eq!(errors.len(), 1);
        assert!(
            errors[0].to_string().contains("system boom"),
            "expected panic message, got: {}",
            errors[0]
        );
    }

    #[test]
    fn panicking_system_does_not_block_others_single_thread() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let ran = Arc::new(AtomicBool::new(false));

        struct PanicSystem;
        impl System for PanicSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                panic!("first panics");
            }
        }

        struct OkSystem(Arc<AtomicBool>);
        impl System for OkSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
                self.0.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let mut container = SystemsContainer::new();
        container.add(PanicSystem);
        container.add(OkSystem(ran.clone()));
        container.add_edge::<PanicSystem, OkSystem>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        let errors = runner.run(&mut world, &container);

        assert_eq!(errors.len(), 1);
        assert!(ran.load(Ordering::SeqCst), "second system should still run");
    }

    #[test]
    fn panicking_exclusive_system_caught_single_thread() {
        struct PanicExcl;
        impl ExclusiveSystem for PanicExcl {
            type Result = ();
            fn run(&mut self, _world: &mut World) -> Result<(), SystemError> {
                panic!("exclusive boom");
            }
        }

        let mut container = SystemsContainer::new();
        container.add_exclusive(PanicExcl);

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        let errors = runner.run(&mut world, &container);

        assert_eq!(errors.len(), 1);
        assert!(errors[0].to_string().contains("exclusive boom"));
    }
}
