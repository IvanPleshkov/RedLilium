use std::any::TypeId;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;

use crate::compute::{ComputePool, noop_waker};
use crate::query_access::QueryAccess;
use crate::system::{StoredSystem, System, SystemRef};
use crate::world::World;

/// A collection of systems with dependency resolution and execution ordering.
///
/// The schedule determines the order in which systems run based on:
/// 1. Explicit ordering constraints ([`after`](SystemRef::after) / [`before`](SystemRef::before))
/// 2. Automatic conflict detection (systems accessing the same component types)
///
/// Systems that do not conflict and have no ordering constraints between them
/// are grouped into the same stage and can run in parallel.
///
/// # Example
///
/// ```ignore
/// let mut schedule = Schedule::new();
///
/// schedule.add(MovementSystem);
/// schedule.add(CollisionSystem)
///     .after::<MovementSystem>();
///
/// schedule.build();
/// schedule.run(&world, &compute);
/// ```
pub struct Schedule {
    /// Registered systems, in registration order.
    systems: Vec<StoredSystem>,
    /// Computed execution order: each inner Vec is a stage of system indices
    /// that can run concurrently.
    execution_order: Vec<Vec<usize>>,
    /// Whether the schedule has been built.
    built: bool,
}

impl Schedule {
    /// Creates a new empty schedule.
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            execution_order: Vec::new(),
            built: false,
        }
    }

    /// Registers a system instance.
    ///
    /// Returns a [`SystemRef`] for declaring ordering constraints.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has already been built, or if a system
    /// with the same type has already been registered.
    pub fn add<S: System>(&mut self, system: S) -> SystemRef<'_> {
        assert!(!self.built, "Cannot add systems after build()");

        let type_id = TypeId::of::<S>();

        if self.systems.iter().any(|s| s.type_id == type_id) {
            panic!(
                "Duplicate system type: {} is already registered",
                std::any::type_name::<S>()
            );
        }

        let access = system.access();
        self.systems.push(StoredSystem {
            system: Box::new(system),
            type_id,
            type_name: std::any::type_name::<S>(),
            access,
            after: Vec::new(),
            before: Vec::new(),
        });

        let stored = self.systems.last_mut().unwrap();
        SystemRef::new(stored)
    }

    /// Resolves dependencies and computes the execution order.
    ///
    /// This must be called after all systems are registered and before
    /// the first call to [`run`](Schedule::run).
    ///
    /// # Panics
    ///
    /// Panics if a dependency cycle is detected, or if an `after`/`before`
    /// constraint references a system type that is not registered.
    pub fn build(&mut self) {
        let n = self.systems.len();
        if n == 0 {
            self.built = true;
            return;
        }

        // Build TypeId → index lookup
        let id_to_idx: std::collections::HashMap<TypeId, usize> = self
            .systems
            .iter()
            .enumerate()
            .map(|(i, s)| (s.type_id, i))
            .collect();

        // Build adjacency list: edges[i] contains systems that must run after system i
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree: Vec<usize> = vec![0; n];

        // Explicit ordering constraints
        for (i, system) in self.systems.iter().enumerate() {
            for &dep_id in &system.after {
                let &dep_idx = id_to_idx.get(&dep_id).unwrap_or_else(|| {
                    panic!(
                        "System '{}' declares after a system type that is not registered (TypeId: {:?})",
                        system.type_name, dep_id
                    )
                });
                // dep_idx must run before i → edge from dep_idx to i
                edges[dep_idx].push(i);
                in_degree[i] += 1;
            }
            for &dep_id in &system.before {
                let &dep_idx = id_to_idx.get(&dep_id).unwrap_or_else(|| {
                    panic!(
                        "System '{}' declares before a system type that is not registered (TypeId: {:?})",
                        system.type_name, dep_id
                    )
                });
                // i must run before dep_idx → edge from i to dep_idx
                edges[i].push(dep_idx);
                in_degree[dep_idx] += 1;
            }
        }

        // Implicit conflict edges (registration order tiebreaker)
        // Only add edge if no explicit ordering exists between the pair
        for i in 0..n {
            for j in (i + 1)..n {
                let has_order = edges[i].contains(&j) || edges[j].contains(&i);
                if self.systems[i]
                    .access
                    .conflicts_with(&self.systems[j].access)
                    && !has_order
                {
                    // Earlier registration runs first
                    edges[i].push(j);
                    in_degree[j] += 1;
                }
            }
        }

        // Kahn's algorithm with stage grouping
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, deg) in in_degree.iter().enumerate() {
            if *deg == 0 {
                queue.push_back(i);
            }
        }

        let mut stages: Vec<Vec<usize>> = Vec::new();
        let mut processed = 0;

        while !queue.is_empty() {
            // All items currently in the queue form one stage
            let stage_size = queue.len();
            let mut stage = Vec::with_capacity(stage_size);

            for _ in 0..stage_size {
                let idx = queue.pop_front().unwrap();
                stage.push(idx);
                processed += 1;

                for &next in &edges[idx] {
                    in_degree[next] -= 1;
                    if in_degree[next] == 0 {
                        queue.push_back(next);
                    }
                }
            }

            stages.push(stage);
        }

        if processed != n {
            // Find systems involved in cycles
            let cycle_systems: Vec<&str> = in_degree
                .iter()
                .enumerate()
                .filter(|(_, deg)| **deg > 0)
                .map(|(i, _)| self.systems[i].type_name)
                .collect();
            panic!(
                "Dependency cycle detected among systems: [{}]",
                cycle_systems.join(", ")
            );
        }

        self.execution_order = stages;
        self.built = true;
    }

    /// Executes all systems sequentially in the computed order.
    ///
    /// Systems within the same stage are run one after another. Each system's
    /// future is polled to completion, with [`ComputePool::tick_all`] called
    /// between polls to drive compute tasks.
    ///
    /// For parallel execution, use [`run_parallel`](Schedule::run_parallel).
    ///
    /// # Panics
    ///
    /// Panics if [`build`](Schedule::build) has not been called.
    pub fn run(&self, world: &World, compute: &ComputePool) {
        assert!(self.built, "Schedule::build() must be called before run()");
        for stage in &self.execution_order {
            let mut futures: Vec<Pin<Box<dyn Future<Output = ()> + Send + '_>>> = stage
                .iter()
                .map(|&idx| {
                    let access = QueryAccess::new(world, compute);
                    self.systems[idx].system.run(access)
                })
                .collect();
            poll_async_futures(&mut futures, compute);
        }
    }

    /// Executes systems in parallel using the provided thread pool.
    ///
    /// Systems within the same stage run in parallel via the pool. Each
    /// system's future is moved to a thread pool thread and polled there.
    /// Stages are executed sequentially.
    ///
    /// # Panics
    ///
    /// Panics if [`build`](Schedule::build) has not been called.
    pub fn run_parallel(
        &self,
        world: &World,
        pool: &crate::thread_pool::ThreadPool,
        compute: &ComputePool,
    ) {
        assert!(
            self.built,
            "Schedule::build() must be called before run_parallel()"
        );
        for stage in &self.execution_order {
            let futures: Vec<Pin<Box<dyn Future<Output = ()> + Send + '_>>> = stage
                .iter()
                .map(|&idx| {
                    let access = QueryAccess::new(world, compute);
                    self.systems[idx].system.run(access)
                })
                .collect();

            if futures.len() <= 1 {
                let mut futures = futures;
                poll_async_futures(&mut futures, compute);
            } else {
                // Move each future to its own thread for parallel execution.
                // Futures are Send, and pool.scope() guarantees all threads
                // join before returning (satisfying the borrow lifetimes).
                pool.scope(|s| {
                    for mut future in futures {
                        s.spawn(move || {
                            let waker = noop_waker();
                            let mut cx = std::task::Context::from_waker(&waker);
                            loop {
                                match future.as_mut().poll(&mut cx) {
                                    std::task::Poll::Ready(()) => break,
                                    std::task::Poll::Pending => {
                                        compute.tick_all();
                                        std::thread::yield_now();
                                    }
                                }
                            }
                        });
                    }
                });
            }
        }
    }

    /// Returns the number of registered systems.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Returns the number of stages (groups of non-conflicting systems).
    ///
    /// Only valid after [`build`](Schedule::build) is called.
    pub fn stage_count(&self) -> usize {
        self.execution_order.len()
    }

    /// Returns the system type names in execution order, grouped by stage.
    ///
    /// Useful for debugging and visualization.
    pub fn execution_stages(&self) -> Vec<Vec<&str>> {
        self.execution_order
            .iter()
            .map(|stage| {
                stage
                    .iter()
                    .map(|&idx| self.systems[idx].type_name)
                    .collect()
            })
            .collect()
    }
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

/// Polls async system futures to completion, driving compute tasks between polls.
fn poll_async_futures(
    futures: &mut Vec<Pin<Box<dyn Future<Output = ()> + Send + '_>>>,
    compute: &ComputePool,
) {
    let waker = noop_waker();
    let mut cx = std::task::Context::from_waker(&waker);

    let mut remaining = futures.len();
    let mut done = vec![false; futures.len()];

    while remaining > 0 {
        compute.tick_all();

        let mut made_progress = false;
        for (i, future) in futures.iter_mut().enumerate() {
            if done[i] {
                continue;
            }
            if future.as_mut().poll(&mut cx).is_ready() {
                done[i] = true;
                remaining -= 1;
                made_progress = true;
            }
        }

        if remaining > 0 && !made_progress {
            std::thread::yield_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::Access;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    // ---- Component marker types for access declarations ----

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }
    struct Health;

    // ---- Test systems ----

    struct CounterSystem(Arc<AtomicU32>);
    impl System for CounterSystem {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.fetch_add(1, Ordering::Relaxed);
            })
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    // Systems for ordering tests
    struct FirstSystem(Arc<Mutex<Vec<&'static str>>>);
    impl System for FirstSystem {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().push("first");
            })
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    struct SecondSystem(Arc<Mutex<Vec<&'static str>>>);
    impl System for SecondSystem {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().push("second");
            })
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    // Systems for conflict detection
    struct WritePosA;
    impl System for WritePosA {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_write::<Position>();
            a
        }
    }

    struct WritePosB;
    impl System for WritePosB {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_write::<Position>();
            a
        }
    }

    struct WriteVelA;
    impl System for WriteVelA {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_write::<Velocity>();
            a
        }
    }

    struct ReadPosA;
    impl System for ReadPosA {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_read::<Position>();
            a
        }
    }

    struct ReadPosB;
    impl System for ReadPosB {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_read::<Position>();
            a
        }
    }

    // Systems for registration order tiebreaker
    struct TiebreakFirst(Arc<Mutex<Vec<&'static str>>>);
    impl System for TiebreakFirst {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().push("first_registered");
            })
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_write::<Position>();
            a
        }
    }

    struct TiebreakSecond(Arc<Mutex<Vec<&'static str>>>);
    impl System for TiebreakSecond {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().push("second_registered");
            })
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_write::<Position>();
            a
        }
    }

    // Systems for cycle/missing tests
    struct CycleA;
    impl System for CycleA {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    struct CycleB;
    impl System for CycleB {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    struct MissingDepSystem;
    impl System for MissingDepSystem {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    struct NonexistentSystem;
    impl System for NonexistentSystem {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    // Systems for diamond dependency
    struct DiamondA(Arc<Mutex<Vec<&'static str>>>);
    impl System for DiamondA {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().push("A");
            })
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    struct DiamondB(Arc<Mutex<Vec<&'static str>>>);
    impl System for DiamondB {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().push("B");
            })
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    struct DiamondC(Arc<Mutex<Vec<&'static str>>>);
    impl System for DiamondC {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().push("C");
            })
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    struct DiamondD(Arc<Mutex<Vec<&'static str>>>);
    impl System for DiamondD {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().push("D");
            })
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    // System for world modification test
    struct MovementSystem;
    impl System for MovementSystem {
        fn run<'a>(
            &'a self,
            access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                access.scope(|world| {
                    let mut positions = world.write::<Position>();
                    let velocities = world.read::<Velocity>();
                    for (idx, pos) in positions.iter_mut() {
                        if let Some(vel) = velocities.get(idx) {
                            pos.x += vel.x;
                        }
                    }
                });
            })
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_write::<Position>();
            a.add_read::<Velocity>();
            a
        }
    }

    // Systems for execution_stages test
    struct PhysicsSystem;
    impl System for PhysicsSystem {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_write::<Position>();
            a
        }
    }

    struct AiSystem;
    impl System for AiSystem {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_write::<Health>();
            a
        }
    }

    struct RenderSystem;
    impl System for RenderSystem {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async {})
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_read::<Position>();
            a
        }
    }

    // ---- Tests ----

    #[test]
    fn single_system_runs() {
        let counter = Arc::new(AtomicU32::new(0));

        let mut schedule = Schedule::new();
        schedule.add(CounterSystem(counter.clone()));
        schedule.build();

        let world = World::new();
        let compute = ComputePool::new();
        schedule.run(&world, &compute);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn sequential_ordering_after() {
        let order = Arc::new(Mutex::new(Vec::new()));

        let mut schedule = Schedule::new();
        schedule.add(FirstSystem(order.clone()));
        schedule
            .add(SecondSystem(order.clone()))
            .after::<FirstSystem>();
        schedule.build();

        let world = World::new();
        let compute = ComputePool::new();
        schedule.run(&world, &compute);

        let result = order.lock().unwrap();
        assert_eq!(*result, vec!["first", "second"]);
    }

    #[test]
    fn sequential_ordering_before() {
        let order = Arc::new(Mutex::new(Vec::new()));

        let mut schedule = Schedule::new();
        schedule.add(SecondSystem(order.clone()));
        schedule
            .add(FirstSystem(order.clone()))
            .before::<SecondSystem>();
        schedule.build();

        let world = World::new();
        let compute = ComputePool::new();
        schedule.run(&world, &compute);

        let result = order.lock().unwrap();
        assert_eq!(*result, vec!["first", "second"]);
    }

    #[test]
    fn conflict_detection_separates_stages() {
        let mut schedule = Schedule::new();
        // Both write Position → must be in different stages
        schedule.add(WritePosA);
        schedule.add(WritePosB);
        schedule.build();

        assert_eq!(schedule.stage_count(), 2);
    }

    #[test]
    fn no_conflict_same_stage() {
        let mut schedule = Schedule::new();
        // Different types → can run in same stage
        schedule.add(WritePosA);
        schedule.add(WriteVelA);
        schedule.build();

        assert_eq!(schedule.stage_count(), 1);
        assert_eq!(schedule.execution_stages()[0].len(), 2);
    }

    #[test]
    fn same_reads_same_stage() {
        let mut schedule = Schedule::new();
        schedule.add(ReadPosA);
        schedule.add(ReadPosB);
        schedule.build();

        assert_eq!(schedule.stage_count(), 1);
    }

    #[test]
    fn registration_order_tiebreaker() {
        let order = Arc::new(Mutex::new(Vec::new()));

        let mut schedule = Schedule::new();
        schedule.add(TiebreakFirst(order.clone()));
        schedule.add(TiebreakSecond(order.clone()));
        schedule.build();

        let world = World::new();
        let compute = ComputePool::new();
        schedule.run(&world, &compute);

        let result = order.lock().unwrap();
        assert_eq!(*result, vec!["first_registered", "second_registered"]);
    }

    #[test]
    #[should_panic(expected = "Dependency cycle detected")]
    fn cycle_detection_panics() {
        let mut schedule = Schedule::new();
        schedule.add(CycleA).after::<CycleB>();
        schedule.add(CycleB).after::<CycleA>();
        schedule.build();
    }

    #[test]
    #[should_panic(expected = "not registered")]
    fn missing_dependency_panics() {
        let mut schedule = Schedule::new();
        schedule.add(MissingDepSystem).after::<NonexistentSystem>();
        schedule.build();
    }

    #[test]
    fn complex_diamond_dependency() {
        // A -> B, A -> C, B -> D, C -> D
        let order = Arc::new(Mutex::new(Vec::new()));

        let mut schedule = Schedule::new();
        schedule.add(DiamondA(order.clone()));
        schedule.add(DiamondB(order.clone())).after::<DiamondA>();
        schedule.add(DiamondC(order.clone())).after::<DiamondA>();
        schedule
            .add(DiamondD(order.clone()))
            .after::<DiamondB>()
            .after::<DiamondC>();
        schedule.build();

        let world = World::new();
        let compute = ComputePool::new();
        schedule.run(&world, &compute);

        let result = order.lock().unwrap();
        assert_eq!(result[0], "A");
        assert_eq!(result[3], "D");
        // B and C can be in any order (same stage)
        assert!(result[1..3].contains(&"B"));
        assert!(result[1..3].contains(&"C"));
    }

    #[test]
    fn systems_modify_world_correctly() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 });
        world.insert(e, Velocity { x: 5.0 });

        let mut schedule = Schedule::new();
        schedule.add(MovementSystem);
        schedule.build();

        let compute = ComputePool::new();
        schedule.run(&world, &compute);
        assert_eq!(world.get::<Position>(e).unwrap().x, 5.0);

        schedule.run(&world, &compute);
        assert_eq!(world.get::<Position>(e).unwrap().x, 10.0);
    }

    #[test]
    fn empty_schedule() {
        let mut schedule = Schedule::new();
        schedule.build();
        let world = World::new();
        let compute = ComputePool::new();
        schedule.run(&world, &compute); // Should not panic
        assert_eq!(schedule.system_count(), 0);
        assert_eq!(schedule.stage_count(), 0);
    }

    #[test]
    fn execution_stages_returns_type_names() {
        let mut schedule = Schedule::new();
        schedule.add(PhysicsSystem);
        schedule.add(AiSystem);
        schedule.add(RenderSystem).after::<PhysicsSystem>();
        schedule.build();

        let stages = schedule.execution_stages();
        // physics and ai can be in same stage (different types)
        let first_stage_has_physics = stages[0].iter().any(|name| name.contains("PhysicsSystem"));
        let first_stage_has_ai = stages[0].iter().any(|name| name.contains("AiSystem"));
        assert!(first_stage_has_physics);
        assert!(first_stage_has_ai);
        // render must be after physics
        let last_stage_has_render = stages
            .last()
            .unwrap()
            .iter()
            .any(|name| name.contains("RenderSystem"));
        assert!(last_stage_has_render);
    }

    #[test]
    #[should_panic(expected = "Duplicate system type")]
    fn duplicate_type_panics() {
        let mut schedule = Schedule::new();
        schedule.add(WritePosA);
        schedule.add(WritePosA);
    }

    // ---- Async system tests ----

    use std::sync::atomic::AtomicBool;

    use crate::Priority;

    struct FlagAsyncSystem(Arc<AtomicBool>);
    impl System for FlagAsyncSystem {
        fn run<'a>(
            &'a self,
            _access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                self.0.store(true, Ordering::Relaxed);
            })
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    #[test]
    fn async_system_runs() {
        let flag = Arc::new(AtomicBool::new(false));

        let mut schedule = Schedule::new();
        schedule.add(FlagAsyncSystem(flag.clone()));
        schedule.build();

        let world = World::new();
        let compute = ComputePool::new();
        schedule.run(&world, &compute);

        assert!(flag.load(Ordering::Relaxed));
    }

    struct ComputeAsyncSystem(Arc<Mutex<Option<u32>>>);
    impl System for ComputeAsyncSystem {
        fn run<'a>(
            &'a self,
            access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            let result_slot = self.0.clone();
            Box::pin(async move {
                let mut handle = access.compute().spawn(Priority::Low, async { 42u32 });
                let result = (&mut handle).await;
                *result_slot.lock().unwrap() = result;
            })
        }
        fn access(&self) -> Access {
            Access::new()
        }
    }

    #[test]
    fn async_system_with_compute() {
        let result = Arc::new(Mutex::new(None));

        let mut schedule = Schedule::new();
        schedule.add(ComputeAsyncSystem(result.clone()));
        schedule.build();

        let world = World::new();
        let compute = ComputePool::new();
        schedule.run(&world, &compute);

        assert_eq!(*result.lock().unwrap(), Some(42));
    }

    struct TwoPhaseSystem;
    impl System for TwoPhaseSystem {
        fn run<'a>(
            &'a self,
            access: QueryAccess<'a>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
            Box::pin(async move {
                // Phase 1: read component data
                let sum = access.scope(|world| {
                    let positions = world.read::<Position>();
                    positions.iter().map(|(_, p)| p.x).sum::<f32>()
                });

                // Spawn compute
                let mut handle = access
                    .compute()
                    .spawn(Priority::Low, async move { sum * 2.0 });
                let result = (&mut handle).await.unwrap();

                // Phase 2: write to resource
                access.scope(|world| {
                    let mut res = world.resource_mut::<f32>();
                    *res = result;
                });
            })
        }
        fn access(&self) -> Access {
            let mut a = Access::new();
            a.add_read::<Position>();
            a.add_resource_write::<f32>();
            a
        }
    }

    #[test]
    fn async_system_two_phase() {
        let mut world = World::new();
        let e1 = world.spawn();
        world.insert(e1, Position { x: 3.0 });
        let e2 = world.spawn();
        world.insert(e2, Position { x: 7.0 });
        world.insert_resource(0.0f32);

        let mut schedule = Schedule::new();
        schedule.add(TwoPhaseSystem);
        schedule.build();

        let compute = ComputePool::new();
        schedule.run(&world, &compute);

        let result = world.resource::<f32>();
        assert_eq!(*result, 20.0); // (3.0 + 7.0) * 2.0
    }

    #[test]
    fn mixed_ordering() {
        let order = Arc::new(Mutex::new(Vec::new()));

        struct SyncFirst(Arc<Mutex<Vec<&'static str>>>);
        impl System for SyncFirst {
            fn run<'a>(
                &'a self,
                _access: QueryAccess<'a>,
            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
                Box::pin(async move {
                    self.0.lock().unwrap().push("first");
                })
            }
            fn access(&self) -> Access {
                Access::new()
            }
        }

        struct AsyncSecond(Arc<Mutex<Vec<&'static str>>>);
        impl System for AsyncSecond {
            fn run<'a>(
                &'a self,
                _access: QueryAccess<'a>,
            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
                Box::pin(async move {
                    self.0.lock().unwrap().push("second");
                })
            }
            fn access(&self) -> Access {
                Access::new()
            }
        }

        let mut schedule = Schedule::new();
        schedule.add(SyncFirst(order.clone()));
        schedule
            .add(AsyncSecond(order.clone()))
            .after::<SyncFirst>();
        schedule.build();

        let world = World::new();
        let compute = ComputePool::new();
        schedule.run(&world, &compute);

        let result = order.lock().unwrap();
        assert_eq!(*result, vec!["first", "second"]);
    }

    #[test]
    fn conflict_detection_separates_stages_async() {
        struct AsyncWritePosA;
        impl System for AsyncWritePosA {
            fn run<'a>(
                &'a self,
                _access: QueryAccess<'a>,
            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
                Box::pin(async {})
            }
            fn access(&self) -> Access {
                let mut a = Access::new();
                a.add_write::<Position>();
                a
            }
        }

        struct AsyncWritePosB;
        impl System for AsyncWritePosB {
            fn run<'a>(
                &'a self,
                _access: QueryAccess<'a>,
            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
                Box::pin(async {})
            }
            fn access(&self) -> Access {
                let mut a = Access::new();
                a.add_write::<Position>();
                a
            }
        }

        let mut schedule = Schedule::new();
        schedule.add(AsyncWritePosA);
        schedule.add(AsyncWritePosB);
        schedule.build();

        assert_eq!(schedule.stage_count(), 2);
    }

    #[test]
    #[should_panic(expected = "Duplicate system type")]
    fn duplicate_type_panics_async() {
        let mut schedule = Schedule::new();
        schedule.add(FlagAsyncSystem(Arc::new(AtomicBool::new(false))));
        schedule.add(FlagAsyncSystem(Arc::new(AtomicBool::new(false))));
    }
}
