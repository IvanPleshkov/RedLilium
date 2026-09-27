#![cfg(not(target_arch = "wasm32"))]

use redlilium_ecs::{
    EcsRunner, ExecutorBusy, ParallelExecutor, Read, System, SystemContext, SystemError,
    SystemsContainer, World,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, ThreadId};
use std::time::Duration;

#[derive(Default)]
struct Seen {
    ids: Mutex<HashSet<ThreadId>>,
    active: AtomicUsize,
    max: AtomicUsize,
    entities: AtomicUsize,
}
struct Work<const N: usize>(Arc<Seen>);
impl<const N: usize> System for Work<N> {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        let active = self.0.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.0.max.fetch_max(active, Ordering::SeqCst);
        self.0.ids.lock().unwrap().insert(thread::current().id());
        thread::sleep(Duration::from_millis(2));
        let mut query = ctx.query::<(Read<u32>,)>();
        query.par_for_each(|_, _| {
            self.0.ids.lock().unwrap().insert(thread::current().id());
            self.0.entities.fetch_add(1, Ordering::Relaxed);
        });
        self.0.active.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }
}
fn world() -> World {
    let mut world = World::new();
    world.register_component::<u32>();
    for i in 0..513 {
        world.spawn_with((i as u32,)).unwrap();
    }
    world
}

#[test]
fn systems_queries_runners_and_worlds_share_the_same_workers() {
    let executor = ParallelExecutor::new(2);
    let runner = EcsRunner::multi_thread_with_executor(executor.clone());
    let seen = Arc::new(Seen::default());
    let mut systems = SystemsContainer::new();
    systems.add(Work::<0>(seen.clone()));
    systems.add(Work::<1>(seen.clone()));
    systems.add(Work::<2>(seen.clone()));
    systems.add(Work::<3>(seen.clone()));
    systems.add(Work::<4>(seen.clone()));
    systems.add(Work::<5>(seen.clone()));
    let mut first = world();
    for _ in 0..2 {
        assert!(runner.run(&mut first, &systems).is_empty());
    }
    let ids = seen.ids.lock().unwrap().clone();
    assert_eq!(ids.len(), 2);
    assert!(!ids.contains(&thread::current().id()));
    assert_eq!(seen.max.load(Ordering::SeqCst), 2);
    assert!(first.parallel_executor().shares_workers_with(&executor));
    let other_runner = EcsRunner::multi_thread_with_executor(executor.clone());
    let mut second = world();
    // Same system code and workers, but world-local system instances/history.
    let mut second_systems = SystemsContainer::new();
    second_systems.add(Work::<0>(seen.clone()));
    second_systems.add(Work::<1>(seen.clone()));
    second_systems.add(Work::<2>(seen.clone()));
    second_systems.add(Work::<3>(seen.clone()));
    second_systems.add(Work::<4>(seen.clone()));
    second_systems.add(Work::<5>(seen.clone()));
    assert!(other_runner.run(&mut second, &second_systems).is_empty());
    assert_eq!(*seen.ids.lock().unwrap(), ids);
    assert_eq!(seen.entities.load(Ordering::Relaxed), 3 * 6 * 513);
    drop(first);
    drop(second);
    executor.shutdown_workers().unwrap();
}

#[test]
fn concurrent_coordinators_share_the_worker_limit() {
    let executor = ParallelExecutor::new(2);
    let seen = Arc::new(Seen::default());
    let start = Arc::new(std::sync::Barrier::new(2));
    let (tx, rx) = mpsc::channel();
    let mut coordinators = Vec::new();
    for _ in 0..2 {
        let executor = executor.clone();
        let seen = seen.clone();
        let start = start.clone();
        let tx = tx.clone();
        coordinators.push(thread::spawn(move || {
            let runner = EcsRunner::multi_thread_with_executor(executor);
            let mut systems = SystemsContainer::new();
            systems.add(Work::<0>(seen.clone()));
            systems.add(Work::<1>(seen));
            let mut world = world();
            start.wait();
            assert!(runner.run(&mut world, &systems).is_empty());
            tx.send(()).unwrap();
        }));
    }
    for _ in 0..2 {
        rx.recv_timeout(Duration::from_secs(10))
            .expect("shared coordinators stalled");
    }
    for coordinator in coordinators {
        coordinator.join().unwrap();
    }
    assert_eq!(seen.ids.lock().unwrap().len(), 2);
    assert_eq!(seen.max.load(Ordering::SeqCst), 2);
    assert_eq!(seen.entities.load(Ordering::Relaxed), 4 * 513);
    executor.shutdown_workers().unwrap();
}

#[test]
fn one_worker_finishes_nested_queries_and_main_thread_requests() {
    use redlilium_ecs::MainThreadRes;
    struct OnMain;
    impl System for OnMain {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            ctx.lock::<(MainThreadRes<ThreadId>,)>()
                .execute(|(owner,)| {
                    assert_eq!(*owner, thread::current().id());
                });
            let mut outer = ctx.query::<(Read<u32>,)>();
            outer.par_for_each(|entity, _| {
                if entity == 0 {
                    let mut inner = ctx.query::<(Read<u32>,)>();
                    inner.par_for_each(|_, _| {});
                }
            });
            Ok(())
        }
    }
    let mut world = world();
    world.insert_main_thread_resource(thread::current().id());
    let mut systems = SystemsContainer::new();
    systems.add(OnMain);
    let runner = EcsRunner::multi_thread(1);
    assert!(runner.run(&mut world, &systems).is_empty());
}

#[test]
fn shutdown_joins_shared_worker_tls_and_allows_restart() {
    use std::cell::RefCell;
    struct DropCount(Arc<AtomicUsize>);
    impl Drop for DropCount {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    thread_local! { static TLS: RefCell<Option<DropCount>> = const { RefCell::new(None) }; }
    struct Install(Arc<AtomicUsize>);
    impl System for Install {
        type Result = ();
        fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<(), SystemError> {
            TLS.with(|value| *value.borrow_mut() = Some(DropCount(self.0.clone())));
            Ok(())
        }
    }
    let executor = ParallelExecutor::new(1);
    let runner = EcsRunner::multi_thread_with_executor(executor.clone());
    let count = Arc::new(AtomicUsize::new(0));
    let mut systems = SystemsContainer::new();
    systems.add(Install(count.clone()));
    let mut world = World::with_parallel_executor(executor.clone());
    assert!(runner.run(&mut world, &systems).is_empty());
    drop(world);
    assert_eq!(count.load(Ordering::SeqCst), 0); // Runner still owns the shared pool.
    executor.shutdown_workers().unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let mut world = World::new();
    let mut second_systems = SystemsContainer::new();
    second_systems.add(Install(count.clone()));
    assert!(runner.run(&mut world, &second_systems).is_empty());
    runner.shutdown_workers().unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[test]
fn busy_shutdown_and_worker_reentry_return_errors() {
    struct Check(ParallelExecutor);
    struct Empty;
    impl System for Empty {
        type Result = ();
        fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<(), SystemError> {
            Ok(())
        }
    }
    impl System for Check {
        type Result = ();
        fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<(), SystemError> {
            assert_eq!(self.0.shutdown_workers(), Err(ExecutorBusy));
            let inner = EcsRunner::multi_thread_with_executor(self.0.clone());
            let mut systems = SystemsContainer::new();
            systems.add(Empty);
            let errors = inner.run(&mut World::new(), &systems);
            assert!(matches!(
                errors.as_slice(),
                [SystemError::ExecutorUnavailable { .. }]
            ));
            Ok(())
        }
    }
    // Timeout catches scheduler regressions without wedging the test harness.
    let (tx, rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let executor = ParallelExecutor::new(1);
        let runner = EcsRunner::multi_thread_with_executor(executor.clone());
        let mut systems = SystemsContainer::new();
        systems.add(Check(executor));
        assert!(runner.run(&mut World::new(), &systems).is_empty());
        tx.send(()).unwrap();
    });
    rx.recv_timeout(Duration::from_secs(10))
        .expect("executor stalled");
    worker.join().unwrap();
}

#[test]
fn reload_preparation_drops_cached_results_for_both_runners() {
    struct ResultDrop(Arc<AtomicUsize>);
    impl Drop for ResultDrop {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct Produce(Arc<AtomicUsize>);
    impl System for Produce {
        type Result = ResultDrop;
        fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<ResultDrop, SystemError> {
            Ok(ResultDrop(self.0.clone()))
        }
    }
    for runner in [EcsRunner::single_thread(), EcsRunner::multi_thread(1)] {
        let count = Arc::new(AtomicUsize::new(0));
        let mut systems = SystemsContainer::new();
        systems.add(Produce(count.clone()));
        let mut world = World::new();
        assert!(runner.run(&mut world, &systems).is_empty());
        drop(world);
        drop(systems);
        assert_eq!(count.load(Ordering::SeqCst), 0);
        runner
            .prepare_reload(std::time::Duration::from_secs(2))
            .unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn result_reuse_panic_completes_the_job_and_worker_remains_usable() {
    struct Reuse;
    impl System for Reuse {
        type Result = ();
        fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<(), SystemError> {
            Ok(())
        }
        fn reuse_result(&self, _: ()) {
            panic!("reuse failure");
        }
    }
    let (tx, rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let runner = EcsRunner::multi_thread(1);
        let mut world = World::new();
        let mut systems = SystemsContainer::new();
        systems.add(Reuse);
        assert!(runner.run(&mut world, &systems).is_empty());
        let errors = runner.run(&mut world, &systems);
        assert!(
            matches!(errors.as_slice(), [SystemError::Panicked { message, .. }] if message == "reuse failure")
        );
        assert!(runner.run(&mut world, &systems).is_empty());
        tx.send(()).unwrap();
    });
    rx.recv_timeout(Duration::from_secs(10))
        .expect("missing completion after reuse panic");
    worker.join().unwrap();
}

#[test]
fn sequential_runner_attaches_shared_query_executor_without_running_systems_on_it() {
    let executor = ParallelExecutor::new(1);
    let runner = EcsRunner::single_thread_with_executor(executor.clone());
    let seen = Arc::new(Seen::default());
    let mut systems = SystemsContainer::new();
    systems.add(Work::<0>(seen.clone()));
    let mut second_systems = SystemsContainer::new();
    second_systems.add(Work::<0>(seen.clone()));
    for systems in [&systems, &second_systems] {
        let mut world = world();
        assert!(runner.run(&mut world, systems).is_empty());
        assert!(world.parallel_executor().shares_workers_with(&executor));
    }
    assert!(seen.ids.lock().unwrap().contains(&thread::current().id()));
    assert!(seen.ids.lock().unwrap().len() <= 2);
    runner
        .prepare_reload(std::time::Duration::from_secs(2))
        .unwrap();
}
