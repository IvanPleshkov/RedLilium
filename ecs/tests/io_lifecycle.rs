#![cfg(not(target_arch = "wasm32"))]

use redlilium_core::compute::IoRunner;
use redlilium_ecs::{EcsRunner, IoRuntime, IoShutdownError};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::task::{Context, Poll};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(5);

#[test]
fn dropped_handles_and_clones_share_lifetime_but_other_runtimes_do_not() {
    let io = IoRuntime::new();
    let other = IoRuntime::new();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    drop(io.clone().run(async move {
        let _ = rx.await;
    }));
    assert_eq!(io.active_count(), 1);
    assert_eq!(
        io.quiesce(Duration::ZERO),
        Err(IoShutdownError::TasksRemaining { remaining_tasks: 1 })
    );
    other.shutdown(Duration::ZERO).unwrap();
    tx.send(()).unwrap();
    io.shutdown(WAIT).unwrap();
    assert_eq!(io.active_count(), 0);
    assert_eq!(io.run(async { 9 }).recv(), Some(9));
    io.shutdown(WAIT).unwrap();
}

struct SlowDrop {
    entered: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}
impl Future for SlowDrop {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        Poll::Ready(())
    }
}
impl Drop for SlowDrop {
    fn drop(&mut self) {
        self.entered.send(()).unwrap();
        self.release.recv_timeout(WAIT).unwrap();
    }
}
#[test]
fn completed_future_is_counted_through_its_destructor() {
    let io = IoRuntime::new();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let handle = io.run(SlowDrop {
        entered: entered_tx,
        release: release_rx,
    });
    entered_rx.recv_timeout(WAIT).unwrap();
    assert_eq!(
        io.shutdown(Duration::ZERO),
        Err(IoShutdownError::TasksRemaining { remaining_tasks: 1 })
    );
    release_tx.send(()).unwrap();
    io.shutdown(WAIT).unwrap();
    assert_eq!(handle.recv(), Some(()));
}

struct BadFuture {
    panic_poll: bool,
}
impl Future for BadFuture {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        if self.panic_poll {
            panic!("IO poll panic");
        }
        Poll::Ready(())
    }
}
impl Drop for BadFuture {
    fn drop(&mut self) {
        panic!("IO drop panic");
    }
}
#[test]
fn poll_and_drop_panics_release_lifetimes_and_leave_runtime_usable() {
    let io = IoRuntime::new();
    for panic_poll in [true, false] {
        let handle = io.run(BadFuture { panic_poll });
        assert_eq!(handle.recv(), if panic_poll { None } else { Some(()) });
        io.quiesce(WAIT).unwrap();
        assert_eq!(io.active_count(), 0);
    }
    assert_eq!(io.run(async { 42 }).recv(), Some(42));
    io.shutdown(WAIT).unwrap();
}

struct TlsDrop {
    entered: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
    count: Arc<AtomicUsize>,
}
impl Drop for TlsDrop {
    fn drop(&mut self) {
        self.entered.send(()).unwrap();
        self.release.recv_timeout(WAIT).unwrap();
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}
thread_local! { static TLS: std::cell::RefCell<Option<TlsDrop>> = const { std::cell::RefCell::new(None) }; }

#[test]
fn worker_tls_timeout_rejects_submissions_until_successful_retry() {
    let io = IoRuntime::new();
    let count = Arc::new(AtomicUsize::new(0));
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let tls = TlsDrop {
        entered: entered_tx,
        release: release_rx,
        count: count.clone(),
    };
    io.run(async move {
        TLS.with(|slot| *slot.borrow_mut() = Some(tls));
    })
    .recv()
    .unwrap();
    io.quiesce(WAIT).unwrap();
    assert_eq!(
        io.shutdown(Duration::from_millis(20)),
        Err(IoShutdownError::WorkersRemaining)
    );
    entered_rx.recv_timeout(WAIT).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert_eq!(io.run(async { 99 }).recv(), None);
    release_tx.send(()).unwrap();
    io.shutdown(WAIT).unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(io.run(async { 8 }).recv(), Some(8));
    io.shutdown(WAIT).unwrap();
}

#[test]
fn detached_blocking_operation_is_joined_even_after_outer_future_finishes() {
    let io = IoRuntime::new();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    io.run(async move {
        drop(tokio::task::spawn_blocking(move || {
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(WAIT).unwrap();
        }));
    })
    .recv()
    .unwrap();
    entered_rx.recv_timeout(WAIT).unwrap();
    io.quiesce(WAIT).unwrap();
    assert_eq!(
        io.shutdown(Duration::from_millis(20)),
        Err(IoShutdownError::WorkersRemaining)
    );
    release_tx.send(()).unwrap();
    io.shutdown(WAIT).unwrap();
}

#[test]
fn reload_waits_for_io_after_compute_cancel_and_can_be_retried() {
    for runner in [EcsRunner::single_thread(), EcsRunner::multi_thread(1)] {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let (entered_tx, entered_rx) = mpsc::channel();
        let io = runner.io().clone();
        let task = runner
            .compute()
            .spawn(redlilium_ecs::Priority::Low, move |_| async move {
                let handle = io.run(async move {
                    let _ = rx.await;
                });
                entered_tx.send(()).unwrap();
                handle.await
            });
        entered_rx.recv_timeout(WAIT).unwrap();
        task.cancel();
        runner.compute().quiesce(WAIT).unwrap();
        let error = runner.prepare_reload(Duration::ZERO).unwrap_err();
        assert!(error.to_string().contains("IO quiescence"), "{error}");
        tx.send(()).unwrap();
        runner.prepare_reload(WAIT).unwrap();
        assert_eq!(runner.io().run(async { 17 }).recv(), Some(17));
        runner.prepare_reload(WAIT).unwrap();
    }
}

#[test]
fn retained_waker_is_safe_after_runtime_shutdown_and_restart() {
    struct SaveWaker(Arc<Mutex<Option<std::task::Waker>>>);
    impl Future for SaveWaker {
        type Output = ();
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            *self.0.lock().unwrap() = Some(cx.waker().clone());
            Poll::Ready(())
        }
    }
    let io = IoRuntime::new();
    let saved = Arc::new(Mutex::new(None));
    io.run(SaveWaker(saved.clone())).recv().unwrap();
    io.shutdown(WAIT).unwrap();
    let waker = saved.lock().unwrap().take().unwrap();
    waker.wake_by_ref();
    assert_eq!(io.run(async { 1 }).recv(), Some(1));
    io.shutdown(WAIT).unwrap();
    drop(io);
    waker.wake();
}

#[test]
fn future_destruction_can_spawn_tracked_child_io() {
    struct SpawnOnDrop(IoRuntime, Option<tokio::sync::oneshot::Receiver<()>>);
    impl Future for SpawnOnDrop {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }
    impl Drop for SpawnOnDrop {
        fn drop(&mut self) {
            let wait = self.1.take().unwrap();
            drop(self.0.run(async move {
                let _ = wait.await;
            }));
        }
    }
    let io = IoRuntime::new();
    let (finish, wait) = tokio::sync::oneshot::channel();
    io.run(SpawnOnDrop(io.clone(), Some(wait))).recv().unwrap();
    assert!(io.quiesce(Duration::ZERO).is_err());
    finish.send(()).unwrap();
    io.shutdown(WAIT).unwrap();
}
