#![cfg(not(target_arch = "wasm32"))]

use redlilium_ecs::{
    EcsRunner, RunDiagnostics, System, SystemContext, SystemError, SystemsContainer, World,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

struct Value(Arc<AtomicUsize>);
impl Drop for Value {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
struct Produce {
    drops: Arc<AtomicUsize>,
    runs: Arc<AtomicUsize>,
}
impl System for Produce {
    type Result = Value;
    fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<Value, SystemError> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        Ok(Value(self.drops.clone()))
    }
}
fn schedule(drops: &Arc<AtomicUsize>, runs: &Arc<AtomicUsize>) -> SystemsContainer {
    let mut systems = SystemsContainer::new();
    systems.add(Produce {
        drops: drops.clone(),
        runs: runs.clone(),
    });
    systems
}
fn runners() -> [EcsRunner; 2] {
    [EcsRunner::single_thread(), EcsRunner::multi_thread(1)]
}
fn orphan_ids(errors: &[SystemError]) -> &[u64] {
    match errors {
        [SystemError::OrphanedScheduleResults { container_ids }] => container_ids,
        _ => panic!("expected orphaned cache error, got {errors:?}"),
    }
}

#[test]
fn rejects_new_schedule_without_running_or_dropping_results_and_recovers_after_reload() {
    for runner in runners() {
        let mut world = World::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let a = schedule(&drops, &runs);
        let b = schedule(&drops, &runs);
        assert!(runner.run(&mut world, &a).is_empty());
        assert!(runner.run(&mut world, &b).is_empty());
        drop(a);
        drop(b);
        let replacement = schedule(&drops, &runs);
        let errors = runner.run(&mut world, &replacement);
        let ids = orphan_ids(&errors);
        assert_eq!(ids.len(), 2);
        assert!(ids[0] < ids[1]);
        assert!(errors[0].to_string().contains("prepare_reload"));
        let report = runner.run_with(&mut world, &replacement, &RunDiagnostics::default());
        assert_eq!(orphan_ids(&report.errors), ids);
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        runner.prepare_reload(Duration::from_secs(2)).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert!(runner.run(&mut world, &replacement).is_empty());
        assert_eq!(runs.load(Ordering::SeqCst), 3);
    }
}

#[test]
fn known_schedules_keep_running_and_reusing_results_without_rescanning_orphans() {
    for runner in runners() {
        let mut world = World::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let a = schedule(&drops, &runs);
        let b = schedule(&drops, &runs);
        assert!(runner.run(&mut world, &a).is_empty());
        assert!(runner.run(&mut world, &b).is_empty());
        drop(a);
        assert!(runner.run(&mut world, &b).is_empty());
        assert_eq!(runs.load(Ordering::SeqCst), 3);
        assert_eq!(drops.load(Ordering::SeqCst), 1); // b's previous result was reused
        let c = schedule(&drops, &runs);
        assert_eq!(orphan_ids(&runner.run(&mut world, &c)).len(), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        runner.prepare_reload(Duration::from_secs(2)).unwrap();
    }
}

#[test]
fn live_containers_and_cleanup_before_replacement_are_valid() {
    for runner in runners() {
        let mut world = World::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let a = schedule(&drops, &runs);
        assert!(runner.run(&mut world, &a).is_empty());
        let moved = Some(a); // moving the owner does not invalidate its marker
        let b = schedule(&drops, &runs);
        assert!(runner.run(&mut world, &b).is_empty());
        drop(moved);
        drop(b);
        runner.prepare_reload(Duration::from_secs(2)).unwrap();
        let c = schedule(&drops, &runs);
        assert!(runner.run(&mut world, &c).is_empty());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn new_empty_schedule_does_not_bypass_orphan_check() {
    for runner in runners() {
        let mut world = World::new();
        let old = schedule(
            &Arc::new(AtomicUsize::new(0)),
            &Arc::new(AtomicUsize::new(0)),
        );
        assert!(runner.run(&mut world, &old).is_empty());
        drop(old);
        assert_eq!(
            orphan_ids(&runner.run(&mut world, &SystemsContainer::new())).len(),
            1
        );
    }
}

#[test]
fn destroyed_containers_without_results_do_not_block_new_schedules() {
    struct Fail;
    impl System for Fail {
        type Result = ();
        fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<(), SystemError> {
            Err(SystemError::Panicked {
                system: "Fail".into(),
                message: "no result".into(),
            })
        }
    }
    for runner in runners() {
        let mut world = World::new();
        assert!(runner.run(&mut world, &SystemsContainer::new()).is_empty());
        let mut failed = SystemsContainer::new();
        failed.add(Fail);
        assert!(matches!(
            runner.run(&mut world, &failed).as_slice(),
            [SystemError::Panicked { .. }]
        ));
        drop(failed);
        let good = schedule(
            &Arc::new(AtomicUsize::new(0)),
            &Arc::new(AtomicUsize::new(0)),
        );
        assert!(runner.run(&mut world, &good).is_empty());
    }
}
