#![cfg(not(target_arch = "wasm32"))]
use redlilium_core::compute::{ComputeContext, IoRunner};
use redlilium_ecs::{
    ComputePool, EcsRunner, ExecutorBusy, IoRuntime, ParallelExecutor, Priority, System,
    SystemContext, SystemError, SystemsContainer, World,
};
use std::future::poll_fn;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::task::Poll;
use std::time::Duration;

#[test]
fn background_io_and_nested_await_progress_between_frames_on_shared_workers() {
    let executor = ParallelExecutor::new(1);
    let runner = EcsRunner::multi_thread_with_executor(executor.clone());
    let ids = Arc::new(Mutex::new(Vec::new()));
    struct Record(Arc<Mutex<Vec<std::thread::ThreadId>>>);
    impl System for Record {
        type Result = ();
        fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<(), SystemError> {
            self.0.lock().unwrap().push(std::thread::current().id());
            Ok(())
        }
    }
    let mut systems = SystemsContainer::new();
    systems.add(Record(ids.clone()));
    assert!(runner.run(&mut World::new(), &systems).is_empty());
    let worker = ids.lock().unwrap()[0];
    let io = runner
        .compute()
        .spawn(Priority::Low, move |ctx| async move {
            let before = std::thread::current().id();
            let value = ctx
                .io()
                .run(async {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    17
                })
                .await;
            (before, std::thread::current().id(), value)
        });
    assert_eq!(
        io.recv_timeout(Duration::from_secs(10)),
        Some((worker, worker, Some(17)))
    );
    let child = runner.compute().spawn(Priority::Low, |_| async { 23 });
    let parent = runner
        .compute()
        .spawn(Priority::High, move |_| async move { child.await });
    assert_eq!(parent.recv_timeout(Duration::from_secs(10)), Some(Some(23)));
    runner.compute().quiesce(Duration::from_secs(1)).unwrap();
    runner
        .prepare_reload(std::time::Duration::from_secs(2))
        .unwrap();
}

#[test]
fn one_worker_system_can_block_on_io_while_helping_its_compute_pool() {
    struct Waiting;
    impl System for Waiting {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
            let mut handle = ctx.compute().spawn(Priority::Low, |ctx| async move {
                ctx.io()
                    .run(async {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        19
                    })
                    .await
            });
            assert_eq!(ctx.compute().block_on(&mut handle), Some(Some(19)));
            Ok(())
        }
    }
    let (tx, rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let runner = EcsRunner::multi_thread(1);
        let mut systems = SystemsContainer::new();
        systems.add(Waiting);
        assert!(runner.run(&mut World::new(), &systems).is_empty());
        tx.send(()).unwrap();
    });
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    thread.join().unwrap();
}

#[test]
fn idle_tasks_do_not_poll_and_token_clones_wake_cancellation() {
    let executor = ParallelExecutor::new(1);
    let pool = ComputePool::with_executor(IoRuntime::new(), executor.clone());
    let polls = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel();
    let handle = pool.spawn(Priority::Low, {
        let polls = polls.clone();
        move |_| {
            poll_fn(move |_| {
                polls.fetch_add(1, Ordering::SeqCst);
                tx.send(()).unwrap();
                Poll::<()>::Pending
            })
        }
    });
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(rx.recv_timeout(Duration::from_millis(30)).is_err());
    assert_eq!(executor.shutdown_workers(), Err(ExecutorBusy));
    handle.cancellation_token().cancel();
    pool.quiesce(Duration::from_secs(1)).unwrap();
    assert!(handle.is_done());
    assert_eq!(polls.load(Ordering::SeqCst), 2);
    executor.shutdown_workers().unwrap();
}

#[test]
fn shared_compute_sources_do_not_starve_each_other() {
    let executor = ParallelExecutor::new(1);
    let first = ComputePool::with_executor(IoRuntime::new(), executor.clone());
    let second = ComputePool::with_executor(IoRuntime::new(), executor.clone());
    let busy = first.spawn(Priority::Critical, |_| {
        poll_fn(|cx| {
            cx.waker().wake_by_ref();
            Poll::<()>::Pending
        })
    });
    let result = second.spawn(Priority::Low, |_| async { 99 });
    assert_eq!(result.recv_timeout(Duration::from_secs(10)), Some(99));
    busy.cancel();
    first.quiesce(Duration::from_secs(1)).unwrap();
    second.quiesce(Duration::from_secs(1)).unwrap();
    executor.shutdown_workers().unwrap();
}

#[test]
fn stale_wakes_after_pool_drop_are_harmless() {
    let executor = ParallelExecutor::new(1);
    let pool = ComputePool::with_executor(IoRuntime::new(), executor.clone());
    let (tx, rx) = mpsc::channel();
    let handle = pool.spawn(Priority::Low, move |_| {
        poll_fn(move |cx| {
            tx.send(cx.waker().clone()).unwrap();
            Poll::<()>::Pending
        })
    });
    let waker = rx.recv_timeout(Duration::from_secs(10)).unwrap();
    drop(pool);
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while executor.shutdown_workers().is_err() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert!(handle.is_done());
    for _ in 0..100 {
        waker.wake_by_ref();
    }
    executor.shutdown_workers().unwrap();
}

#[test]
fn a_single_self_waking_task_does_not_start_extra_workers() {
    let executor = ParallelExecutor::new(4);
    let pool = ComputePool::with_executor(IoRuntime::new(), executor.clone());
    let handle = pool.spawn(Priority::Low, |_| {
        let mut ids = std::collections::HashSet::new();
        let mut polls = 0;
        poll_fn(move |cx| {
            ids.insert(std::thread::current().id());
            polls += 1;
            if polls == 1000 {
                Poll::Ready(ids.len())
            } else {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
    });
    assert_eq!(handle.recv_timeout(Duration::from_secs(10)), Some(1));
    pool.quiesce(Duration::from_secs(1)).unwrap();
    executor.shutdown_workers().unwrap();
}
