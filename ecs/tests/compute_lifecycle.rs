#![cfg(not(target_arch = "wasm32"))]

use redlilium_ecs::{ComputePool, EcsRunner, IoRuntime, Priority, QuiesceTimeout, ShutdownError};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::task::{Context, Poll};
use std::time::Duration;

struct PendingWithDrop {
    entered: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}
impl Future for PendingWithDrop {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        Poll::Pending
    }
}
impl Drop for PendingWithDrop {
    fn drop(&mut self) {
        self.entered.send(()).unwrap();
        self.release.recv_timeout(Duration::from_secs(10)).unwrap();
    }
}

#[test]
fn shutdown_waits_for_destructors_on_every_tick_path() {
    for multi_thread in [false, true] {
        for path in 0..3 {
            let runner = Arc::new(if multi_thread {
                EcsRunner::multi_thread(1)
            } else {
                EcsRunner::single_thread()
            });
            let (entered_tx, entered_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let handle = runner
                .compute()
                .spawn(Priority::Low, move |_| PendingWithDrop {
                    entered: entered_tx,
                    release: release_rx,
                });
            handle.cancel();
            let polling = runner.clone();
            let thread = std::thread::spawn(move || match path {
                0 => polling.compute().tick(),
                1 => polling.compute().tick_all(),
                _ => polling.compute().tick_with_budget(Duration::from_secs(1)),
            });
            entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            assert_eq!(runner.compute().pending_count(), 0);
            assert_eq!(runner.compute().active_count(), 1);
            assert!(!handle.is_done());
            assert_eq!(
                runner.compute().quiesce(Duration::ZERO),
                Err(QuiesceTimeout { remaining_tasks: 1 })
            );
            assert!(matches!(
                runner.graceful_shutdown(Duration::ZERO),
                Err(ShutdownError::Timeout { remaining_tasks: 1 })
            ));
            release_tx.send(()).unwrap();
            assert_eq!(thread.join().unwrap(), 1);
            assert!(handle.is_done());
            assert!(handle.is_cancelled());
            assert!(runner.compute().quiesce(Duration::ZERO).is_ok());
            assert!(runner.graceful_shutdown(Duration::ZERO).is_ok());
        }
    }
}

#[test]
fn construction_is_counted_and_factory_unwind_releases_its_slot() {
    let pool = Arc::new(ComputePool::new(IoRuntime::new()));
    let producer = pool.clone();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        producer.spawn(Priority::Low, move |_| {
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            async { 7 }
        })
    });
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(pool.pending_count(), 0);
    assert_eq!(
        pool.quiesce(Duration::ZERO),
        Err(QuiesceTimeout { remaining_tasks: 1 })
    );
    release_tx.send(()).unwrap();
    let handle = thread.join().unwrap();
    pool.quiesce(Duration::from_secs(1)).unwrap();
    assert_eq!(handle.try_recv(), Some(7));
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.spawn(Priority::Low, |_| -> std::future::Ready<()> {
            panic!("factory")
        });
    }));
    assert!(panic.is_err());
    assert_eq!(pool.active_count(), 0);
    pool.quiesce(Duration::ZERO).unwrap();
}

struct BadDrop;
impl Future for BadDrop {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        Poll::Pending
    }
}
impl Drop for BadDrop {
    fn drop(&mut self) {
        panic!("destructor failure");
    }
}

#[test]
fn destructor_panic_is_reported_and_does_not_lose_other_tasks() {
    for path in 0..3 {
        let pool = ComputePool::new(IoRuntime::new());
        let bad = pool.spawn(Priority::High, |_| BadDrop);
        bad.cancel();
        let good = pool.spawn(Priority::Low, |_| async { 42 });
        match path {
            0 => {
                pool.tick();
            }
            1 => {
                pool.tick_all();
            }
            _ => {
                pool.tick_with_budget(Duration::from_secs(1));
            }
        }
        pool.quiesce(Duration::from_secs(1)).unwrap();
        assert_eq!(bad.panic_message().as_deref(), Some("destructor failure"));
        assert!(bad.is_done());
        assert_eq!(good.try_recv(), Some(42));
        assert_eq!(pool.active_count(), 0);
    }
}

#[test]
fn concurrent_publication_and_polling_keep_exact_task_accounting() {
    let pool = Arc::new(ComputePool::new(IoRuntime::new()));
    let producers = AtomicUsize::new(4);
    let completed = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let pool = &pool;
            let producers = &producers;
            let completed = &completed;
            scope.spawn(move || {
                for _ in 0..512 {
                    let count = completed.clone();
                    let observer = pool.clone();
                    pool.spawn(Priority::Low, move |_| async move {
                        let active = observer.active_count();
                        assert!(
                            (1..=2048).contains(&active),
                            "invalid active count: {active}"
                        );
                        count.fetch_add(1, Ordering::Relaxed);
                    });
                }
                producers.fetch_sub(1, Ordering::Release);
            });
        }
        for _ in 0..4 {
            let pool = &pool;
            let producers = &producers;
            scope.spawn(move || {
                let deadline = std::time::Instant::now() + Duration::from_secs(10);
                while producers.load(Ordering::Acquire) != 0 || pool.active_count() != 0 {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "task accounting stalled"
                    );
                    pool.tick_all();
                    std::thread::yield_now();
                }
            });
        }
    });
    assert_eq!(completed.load(Ordering::Relaxed), 2048);
    assert_eq!(pool.active_count(), 0);
    pool.quiesce(Duration::ZERO).unwrap();
}

#[test]
fn quiescence_includes_tasks_spawned_from_destructors() {
    struct SpawnOnDrop(Arc<ComputePool>, Arc<AtomicUsize>);
    impl Drop for SpawnOnDrop {
        fn drop(&mut self) {
            let done = self.1.clone();
            self.0.spawn(Priority::Low, move |_| async move {
                done.fetch_add(1, Ordering::Relaxed);
            });
        }
    }
    let pool = Arc::new(ComputePool::new(IoRuntime::new()));
    let done = Arc::new(AtomicUsize::new(0));
    let guard = SpawnOnDrop(pool.clone(), done.clone());
    let handle = pool.spawn(Priority::Low, |_| async move {
        std::future::pending::<()>().await;
        drop(guard);
    });
    handle.cancel();
    pool.quiesce(Duration::from_secs(1)).unwrap();
    assert_eq!(done.load(Ordering::Relaxed), 1);
    assert_eq!(pool.active_count(), 0);
}
