use std::time::Duration;

use crate::command_collector::CommandCollector;
use crate::compute::ComputePool;
use crate::io_runtime::IoRuntime;
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
}

impl EcsRunnerSingleThread {
    /// Creates a new single-threaded runner.
    pub fn new() -> Self {
        let io = IoRuntime::new();
        Self {
            compute: ComputePool::new(io.clone()),
            io,
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
    pub fn run(&self, world: &mut World, systems: &SystemsContainer) {
        redlilium_core::profile_scope!("ecs: run (single-thread)");

        let order = systems.single_thread_order();
        if order.is_empty() {
            return;
        }

        let commands = CommandCollector::new();
        let results_store =
            SystemResultsStore::new(systems.system_count(), systems.type_id_to_idx().clone());

        {
            for &idx in order {
                let ctx = SystemContext::new(world, &self.compute, &self.io, &commands)
                    .with_system_results(&results_store, systems.accessible_results(idx));

                let system = systems.get_system(idx);
                let guard = system.read().unwrap();
                let result = guard.run_boxed(&ctx);
                results_store.store(idx, result);
            }
        }

        // Apply deferred commands (ctx dropped, world is free)
        {
            redlilium_core::profile_scope!("ecs: apply commands");
            for cmd in commands.drain() {
                cmd(world);
            }
        }

        // Drain remaining compute tasks (one poll per task)
        if self.compute.pending_count() > 0 {
            redlilium_core::profile_scope!("ecs: compute drain");
            self.compute.tick_all();
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

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }

    struct MovementSystem;
    impl System for MovementSystem {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
            ctx.lock::<(Write<Position>, Read<Velocity>)>().execute(
                |(mut positions, velocities)| {
                    for (idx, pos) in positions.iter_mut() {
                        if let Some(vel) = velocities.get(idx) {
                            pos.x += vel.x;
                        }
                    }
                },
            );
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
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
                self.0.store(1, Ordering::SeqCst);
            }
        }

        struct SecondSystem(Arc<AtomicU32>);
        impl System for SecondSystem {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {
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
        runner.run(&mut world, &container);

        assert_eq!(order.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn deferred_commands_applied() {
        struct SpawnSystem;
        impl System for SpawnSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
                ctx.commands(|world| {
                    world.insert_resource(42u32);
                });
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
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> u32 {
                42
            }
        }

        struct ConsumerSystem(std::sync::Arc<std::sync::Mutex<Option<u32>>>);
        impl System for ConsumerSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
                let value = *ctx.system_result::<ProducerSystem>();
                *self.0.lock().unwrap() = Some(value);
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
    #[should_panic(expected = "not accessible")]
    fn system_result_panics_without_edge() {
        struct ProducerSystem;
        impl System for ProducerSystem {
            type Result = u32;
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> u32 {
                42
            }
        }

        struct ConsumerSystem;
        impl System for ConsumerSystem {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
                let _ = ctx.system_result::<ProducerSystem>();
            }
        }

        let mut container = SystemsContainer::new();
        container.add(ProducerSystem);
        container.add(ConsumerSystem);
        // No edge — should panic

        let runner = EcsRunnerSingleThread::new();
        let mut world = World::new();
        runner.run(&mut world, &container);
    }

    #[test]
    fn system_result_transitive_access() {
        struct SystemA;
        impl System for SystemA {
            type Result = String;
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) -> String {
                "hello".to_string()
            }
        }

        struct SystemB;
        impl System for SystemB {
            type Result = ();
            fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {}
        }

        // C depends on B depends on A, so C should be able to read A's result
        struct SystemC(std::sync::Arc<std::sync::Mutex<Option<String>>>);
        impl System for SystemC {
            type Result = ();
            fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
                let value = ctx.system_result::<SystemA>().clone();
                *self.0.lock().unwrap() = Some(value);
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
}
