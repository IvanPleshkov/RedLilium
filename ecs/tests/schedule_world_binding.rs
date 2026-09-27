#![cfg(not(target_arch = "wasm32"))]

use redlilium_ecs::{
    EcsRunner, MaybeAdded, Read, RunDiagnostics, System, SystemContext, SystemError,
    SystemsContainer, World,
};
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

#[derive(Default)]
struct Stats {
    runs: AtomicUsize,
    reuses: AtomicUsize,
    added: AtomicUsize,
}
struct Marker;
struct Count(Arc<Stats>);
impl System for Count {
    type Result = Vec<u8>;
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Vec<u8>, SystemError> {
        self.0.runs.fetch_add(1, Ordering::SeqCst);
        ctx.lock::<(Read<Marker>, MaybeAdded<Marker>)>()
            .execute(|(items, added)| {
                for (idx, _) in items.iter() {
                    if added.matches(idx) {
                        self.0.added.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        Ok(vec![1])
    }
    fn reuse_result(&self, prev: Vec<u8>) {
        assert_eq!(prev, [1]);
        self.0.reuses.fetch_add(1, Ordering::SeqCst);
    }
}
fn world() -> World {
    let mut world = World::new();
    world.register_component::<Marker>();
    world.spawn_with((Marker,)).unwrap();
    world
}
fn schedule(stats: &Arc<Stats>) -> SystemsContainer {
    let mut systems = SystemsContainer::new();
    systems.add(Count(stats.clone()));
    systems
}
fn runners() -> [EcsRunner; 2] {
    [EcsRunner::single_thread(), EcsRunner::multi_thread(1)]
}
fn assert_mismatch(errors: &[SystemError]) {
    assert!(
        matches!(errors, [SystemError::ScheduleWorldMismatch { .. }]),
        "{errors:?}"
    );
    assert!(errors[0].to_string().contains("another world"));
}

#[test]
fn moving_world_preserves_binding_and_rejected_run_does_not_reuse_results_or_switch_executor() {
    for runner in runners() {
        let stats = Arc::new(Stats::default());
        let systems = schedule(&stats);
        let mut first = world();
        assert!(runner.run(&mut first, &systems).is_empty());
        let mut moved = Box::new(first);
        assert!(runner.run(&mut moved, &systems).is_empty());
        let mut second = world();
        let original_executor = second.parallel_executor().clone();
        assert_mismatch(
            &runner
                .run_with(&mut second, &systems, &RunDiagnostics::default())
                .errors,
        );
        assert!(
            second
                .parallel_executor()
                .shares_workers_with(&original_executor)
        );
        assert_eq!(stats.runs.load(Ordering::SeqCst), 2);
        assert_eq!(stats.reuses.load(Ordering::SeqCst), 1);
        assert_eq!(stats.added.load(Ordering::SeqCst), 1);
        assert!(runner.run(&mut moved, &systems).is_empty());
        assert_eq!(stats.reuses.load(Ordering::SeqCst), 2);
        assert_eq!(stats.added.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn separate_schedules_share_runner_without_mixing_world_history() {
    for runner in runners() {
        let stats = Arc::new(Stats::default());
        let a = schedule(&stats);
        let b = schedule(&stats);
        let mut first = world();
        let mut second = world();
        for _ in 0..2 {
            assert!(runner.run(&mut first, &a).is_empty());
            assert!(runner.run(&mut second, &b).is_empty());
        }
        assert_eq!(stats.runs.load(Ordering::SeqCst), 4);
        assert_eq!(stats.added.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn dropping_world_and_clearing_caches_or_changing_runners_does_not_rebind() {
    struct DropResource(Arc<AtomicUsize>);
    impl Drop for DropResource {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    for runner in runners() {
        let stats = Arc::new(Stats::default());
        let systems = schedule(&stats);
        let destroyed = Arc::new(AtomicUsize::new(0));
        let mut first = world();
        first.insert_resource(DropResource(destroyed.clone()));
        assert!(runner.run(&mut first, &systems).is_empty());
        drop(first);
        assert_eq!(destroyed.load(Ordering::SeqCst), 1);
        runner.prepare_reload(Duration::from_secs(2)).unwrap();
        let mut second = world();
        assert_mismatch(&runner.run(&mut second, &systems));
        let other_runner = EcsRunner::single_thread();
        assert_mismatch(&other_runner.run(&mut second, &systems));
        let fresh = schedule(&stats);
        assert!(other_runner.run(&mut second, &fresh).is_empty());
        assert_eq!(stats.runs.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn empty_schedule_is_bound_before_systems_are_added() {
    for runner in runners() {
        let mut systems = SystemsContainer::new();
        let mut first = world();
        assert!(runner.run(&mut first, &systems).is_empty());
        assert_mismatch(&runner.run(&mut world(), &systems));
        let stats = Arc::new(Stats::default());
        systems.add(Count(stats.clone()));
        assert!(runner.run(&mut first, &systems).is_empty());
        assert_eq!(stats.added.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn concurrent_first_admission_binds_to_exactly_one_world() {
    let stats = Arc::new(Stats::default());
    let systems = Arc::new(schedule(&stats));
    let start = Arc::new(Barrier::new(2));
    let threads: Vec<_> = [false, true]
        .into_iter()
        .map(|multi| {
            let systems = systems.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                let runner = if multi {
                    EcsRunner::multi_thread(1)
                } else {
                    EcsRunner::single_thread()
                };
                let mut world = world();
                start.wait();
                runner.run(&mut world, &systems)
            })
        })
        .collect();
    let results: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(results.iter().filter(|errors| errors.is_empty()).count(), 1);
    assert_mismatch(results.iter().find(|errors| !errors.is_empty()).unwrap());
    assert_eq!(stats.runs.load(Ordering::SeqCst), 1);
}

#[test]
fn orphan_rejection_does_not_bind_unused_schedule() {
    for runner in runners() {
        let stats = Arc::new(Stats::default());
        let old = schedule(&stats);
        let mut first = world();
        assert!(runner.run(&mut first, &old).is_empty());
        drop(old);
        let replacement = schedule(&stats);
        assert!(matches!(
            runner.run(&mut first, &replacement).as_slice(),
            [SystemError::OrphanedScheduleResults { .. }]
        ));
        runner.prepare_reload(Duration::from_secs(2)).unwrap();
        assert!(runner.run(&mut world(), &replacement).is_empty());
    }
}

#[test]
fn same_world_and_schedule_can_use_another_runner() {
    let stats = Arc::new(Stats::default());
    let systems = schedule(&stats);
    let mut world = world();
    let runners = runners();
    for runner in &runners {
        assert!(runner.run(&mut world, &systems).is_empty());
    }
    assert_eq!(stats.runs.load(Ordering::SeqCst), 2);
    assert_eq!(stats.added.load(Ordering::SeqCst), 1);
}
