use std::collections::VecDeque;

use crate::system::{FnSystem, SystemBuilder};
use crate::world::World;

/// A collection of systems with dependency resolution and execution ordering.
///
/// The schedule determines the order in which systems run based on:
/// 1. Explicit ordering constraints (`after` / `before`)
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
/// schedule.add_system("movement")
///     .writes::<Position>()
///     .reads::<Velocity>()
///     .run(movement_system);
///
/// schedule.add_system("collision")
///     .reads::<Position>()
///     .reads::<Collider>()
///     .after("movement")
///     .run(collision_system);
///
/// schedule.build();
/// schedule.run(&world);
/// ```
pub struct Schedule {
    /// Registered systems, in registration order.
    systems: Vec<FnSystem>,
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

    /// Begins registering a system with the given name.
    ///
    /// Returns a [`SystemBuilder`] for declaring access and ordering constraints.
    ///
    /// # Panics
    ///
    /// Panics if the schedule has already been built.
    pub fn add_system(&mut self, name: &str) -> SystemBuilder<'_> {
        assert!(!self.built, "Cannot add systems after build()");
        SystemBuilder::new(&mut self.systems, name.to_string())
    }

    /// Resolves dependencies and computes the execution order.
    ///
    /// This must be called after all systems are registered and before
    /// the first call to [`run`](Schedule::run).
    ///
    /// # Panics
    ///
    /// Panics if a dependency cycle is detected, or if an `after`/`before`
    /// constraint references a non-existent system.
    pub fn build(&mut self) {
        let n = self.systems.len();
        if n == 0 {
            self.built = true;
            return;
        }

        // Build name→index lookup
        let name_to_idx: std::collections::HashMap<&str, usize> = self
            .systems
            .iter()
            .enumerate()
            .map(|(i, s)| (s.name.as_str(), i))
            .collect();

        // Build adjacency list: edges[i] contains systems that must run after system i
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree: Vec<usize> = vec![0; n];

        // Explicit ordering constraints
        for (i, system) in self.systems.iter().enumerate() {
            for dep_name in &system.after {
                let &dep_idx = name_to_idx.get(dep_name.as_str()).unwrap_or_else(|| {
                    panic!(
                        "System '{}' declares after('{}'), but no system named '{}' exists",
                        system.name, dep_name, dep_name
                    )
                });
                // dep_idx must run before i → edge from dep_idx to i
                edges[dep_idx].push(i);
                in_degree[i] += 1;
            }
            for dep_name in &system.before {
                let &dep_idx = name_to_idx.get(dep_name.as_str()).unwrap_or_else(|| {
                    panic!(
                        "System '{}' declares before('{}'), but no system named '{}' exists",
                        system.name, dep_name, dep_name
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
                .map(|(i, _)| self.systems[i].name.as_str())
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
    /// Systems within the same stage are run one after another.
    /// For parallel execution, use [`run_parallel`](Schedule::run_parallel).
    ///
    /// # Panics
    ///
    /// Panics if [`build`](Schedule::build) has not been called.
    pub fn run(&self, world: &World) {
        assert!(self.built, "Schedule::build() must be called before run()");
        for stage in &self.execution_order {
            for &system_idx in stage {
                self.systems[system_idx].run(world);
            }
        }
    }

    /// Executes systems in parallel using the provided thread pool.
    ///
    /// Systems within the same stage run in parallel via the pool.
    /// Stages are executed sequentially. Single-system stages skip
    /// pool overhead and run inline.
    ///
    /// # Panics
    ///
    /// Panics if [`build`](Schedule::build) has not been called.
    pub fn run_parallel(&self, world: &World, pool: &crate::thread_pool::ThreadPool) {
        assert!(
            self.built,
            "Schedule::build() must be called before run_parallel()"
        );
        for stage in &self.execution_order {
            if stage.len() == 1 {
                self.systems[stage[0]].run(world);
            } else {
                pool.scope(|s| {
                    for &system_idx in stage {
                        let system = &self.systems[system_idx];
                        s.spawn(move || {
                            system.run(world);
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

    /// Returns the system names in execution order, grouped by stage.
    ///
    /// Useful for debugging and visualization.
    pub fn execution_stages(&self) -> Vec<Vec<&str>> {
        self.execution_order
            .iter()
            .map(|stage| {
                stage
                    .iter()
                    .map(|&idx| self.systems[idx].name.as_str())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct Position {
        x: f32,
    }
    struct Velocity {
        x: f32,
    }
    struct Health;

    #[test]
    fn single_system_runs() {
        let counter = std::sync::Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let mut schedule = Schedule::new();
        schedule.add_system("test").run(move |_| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });
        schedule.build();

        let world = World::new();
        schedule.run(&world);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn sequential_ordering_after() {
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

        let o1 = order.clone();
        let o2 = order.clone();

        let mut schedule = Schedule::new();
        schedule.add_system("first").run(move |_| {
            o1.lock().unwrap().push("first");
        });
        schedule.add_system("second").after("first").run(move |_| {
            o2.lock().unwrap().push("second");
        });
        schedule.build();

        let world = World::new();
        schedule.run(&world);

        let result = order.lock().unwrap();
        assert_eq!(*result, vec!["first", "second"]);
    }

    #[test]
    fn sequential_ordering_before() {
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));

        let o1 = order.clone();
        let o2 = order.clone();

        let mut schedule = Schedule::new();
        schedule.add_system("second").run(move |_| {
            o2.lock().unwrap().push("second");
        });
        schedule.add_system("first").before("second").run(move |_| {
            o1.lock().unwrap().push("first");
        });
        schedule.build();

        let world = World::new();
        schedule.run(&world);

        let result = order.lock().unwrap();
        assert_eq!(*result, vec!["first", "second"]);
    }

    #[test]
    fn conflict_detection_separates_stages() {
        let mut schedule = Schedule::new();
        // Both write Position → must be in different stages
        schedule.add_system("a").writes::<Position>().run(|_| {});
        schedule.add_system("b").writes::<Position>().run(|_| {});
        schedule.build();

        assert_eq!(schedule.stage_count(), 2);
    }

    #[test]
    fn no_conflict_same_stage() {
        let mut schedule = Schedule::new();
        // Different types → can run in same stage
        schedule.add_system("a").writes::<Position>().run(|_| {});
        schedule.add_system("b").writes::<Velocity>().run(|_| {});
        schedule.build();

        assert_eq!(schedule.stage_count(), 1);
        assert_eq!(schedule.execution_stages()[0].len(), 2);
    }

    #[test]
    fn same_reads_same_stage() {
        let mut schedule = Schedule::new();
        schedule.add_system("a").reads::<Position>().run(|_| {});
        schedule.add_system("b").reads::<Position>().run(|_| {});
        schedule.build();

        assert_eq!(schedule.stage_count(), 1);
    }

    #[test]
    fn registration_order_tiebreaker() {
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let o1 = order.clone();
        let o2 = order.clone();

        let mut schedule = Schedule::new();
        schedule
            .add_system("first_registered")
            .writes::<Position>()
            .run(move |_| {
                o1.lock().unwrap().push("first_registered");
            });
        schedule
            .add_system("second_registered")
            .writes::<Position>()
            .run(move |_| {
                o2.lock().unwrap().push("second_registered");
            });
        schedule.build();

        let world = World::new();
        schedule.run(&world);

        let result = order.lock().unwrap();
        assert_eq!(*result, vec!["first_registered", "second_registered"]);
    }

    #[test]
    #[should_panic(expected = "Dependency cycle detected")]
    fn cycle_detection_panics() {
        let mut schedule = Schedule::new();
        schedule.add_system("a").after("b").run(|_| {});
        schedule.add_system("b").after("a").run(|_| {});
        schedule.build();
    }

    #[test]
    #[should_panic(expected = "no system named 'nonexistent' exists")]
    fn missing_dependency_panics() {
        let mut schedule = Schedule::new();
        schedule.add_system("a").after("nonexistent").run(|_| {});
        schedule.build();
    }

    #[test]
    fn complex_diamond_dependency() {
        // A -> B, A -> C, B -> D, C -> D
        let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let (o1, o2, o3, o4) = (order.clone(), order.clone(), order.clone(), order.clone());

        let mut schedule = Schedule::new();
        schedule.add_system("A").run(move |_| {
            o1.lock().unwrap().push("A");
        });
        schedule.add_system("B").after("A").run(move |_| {
            o2.lock().unwrap().push("B");
        });
        schedule.add_system("C").after("A").run(move |_| {
            o3.lock().unwrap().push("C");
        });
        schedule
            .add_system("D")
            .after("B")
            .after("C")
            .run(move |_| {
                o4.lock().unwrap().push("D");
            });
        schedule.build();

        let world = World::new();
        schedule.run(&world);

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
        schedule
            .add_system("movement")
            .writes::<Position>()
            .reads::<Velocity>()
            .run(|world| {
                let mut positions = world.write::<Position>();
                let velocities = world.read::<Velocity>();
                for (idx, pos) in positions.iter_mut() {
                    if let Some(vel) = velocities.get(idx) {
                        pos.x += vel.x;
                    }
                }
            });
        schedule.build();

        schedule.run(&world);
        assert_eq!(world.get::<Position>(e).unwrap().x, 5.0);

        schedule.run(&world);
        assert_eq!(world.get::<Position>(e).unwrap().x, 10.0);
    }

    #[test]
    fn empty_schedule() {
        let mut schedule = Schedule::new();
        schedule.build();
        let world = World::new();
        schedule.run(&world); // Should not panic
        assert_eq!(schedule.system_count(), 0);
        assert_eq!(schedule.stage_count(), 0);
    }

    #[test]
    fn execution_stages_returns_names() {
        let mut schedule = Schedule::new();
        schedule
            .add_system("physics")
            .writes::<Position>()
            .run(|_| {});
        schedule.add_system("ai").writes::<Health>().run(|_| {});
        schedule
            .add_system("render")
            .reads::<Position>()
            .after("physics")
            .run(|_| {});
        schedule.build();

        let stages = schedule.execution_stages();
        // physics and ai can be in same stage (different types)
        assert!(stages[0].contains(&"physics"));
        assert!(stages[0].contains(&"ai"));
        // render must be after physics
        assert!(stages.last().unwrap().contains(&"render"));
    }
}
