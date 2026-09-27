use redlilium_ecs::{ComputePool, IoRuntime, Priority};
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

#[derive(Default)]
struct Count(AtomicUsize);
impl Wake for Count {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn sleeping_tasks_are_not_repolled_and_wakes_are_coalesced() {
    let pool = ComputePool::new(IoRuntime::new());
    let polls = Arc::new(AtomicUsize::new(0));
    let stored = Arc::new(Mutex::new(None::<Waker>));
    let handle = pool.spawn(Priority::Low, {
        let polls = polls.clone();
        let stored = stored.clone();
        move |_| {
            poll_fn(move |cx| {
                polls.fetch_add(1, Ordering::SeqCst);
                *stored.lock().unwrap() = Some(cx.waker().clone());
                Poll::<()>::Pending
            })
        }
    });
    assert_eq!(pool.tick(), 1);
    for _ in 0..10 {
        assert_eq!(pool.tick_all(), 0);
    }
    let waker = stored.lock().unwrap().clone().unwrap();
    for _ in 0..10 {
        waker.wake_by_ref();
    }
    assert_eq!(pool.tick_all(), 1);
    assert_eq!(pool.tick_all(), 0);
    assert_eq!(polls.load(Ordering::SeqCst), 2);
    handle.cancellation_token().cancel();
    assert_eq!(pool.tick(), 1);
    assert!(handle.is_done());
    assert_eq!(pool.active_count(), 0);
    waker.wake_by_ref(); // stale wake cannot resurrect a completed task
    assert_eq!(pool.tick(), 0);
}

#[test]
fn task_handle_wakes_awaiter_on_result_panic_and_cancellation() {
    for mode in 0..3 {
        let pool = ComputePool::new(IoRuntime::new());
        let mut handle = pool.spawn(Priority::Low, move |_| {
            poll_fn(move |_| match mode {
                0 => Poll::Ready(7),
                1 => panic!("poll failure"),
                _ => Poll::Pending,
            })
        });
        let count = Arc::new(Count::default());
        let waker = Waker::from(count.clone());
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut handle).poll(&mut cx).is_pending());
        assert_eq!(count.0.load(Ordering::SeqCst), 0);
        if mode == 2 {
            handle.cancellation_token().cancel();
        }
        pool.tick();
        assert!(count.0.load(Ordering::SeqCst) > 0);
        assert_eq!(
            Pin::new(&mut handle).poll(&mut cx),
            Poll::Ready(if mode == 0 { Some(7) } else { None })
        );
    }
}

#[test]
fn wake_during_poll_survives_without_concurrent_polling() {
    let pool = Arc::new(ComputePool::new(IoRuntime::new()));
    let polls = Arc::new(AtomicUsize::new(0));
    let can_finish = Arc::new(AtomicBool::new(false));
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let handle = pool.spawn(Priority::Low, {
        let polls = polls.clone();
        let can_finish = can_finish.clone();
        move |_| {
            poll_fn(move |cx| {
                if polls.fetch_add(1, Ordering::SeqCst) == 0 {
                    entered_tx.send(cx.waker().clone()).unwrap();
                    release_rx
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .unwrap();
                    Poll::Pending
                } else {
                    assert!(can_finish.load(Ordering::SeqCst));
                    Poll::Ready(9)
                }
            })
        }
    });
    let tick = {
        let pool = pool.clone();
        std::thread::spawn(move || pool.tick())
    };
    let waker = entered_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap();
    for _ in 0..100 {
        waker.wake_by_ref();
        assert_eq!(pool.tick(), 0);
    }
    assert_eq!(polls.load(Ordering::SeqCst), 1);
    can_finish.store(true, Ordering::SeqCst);
    release_tx.send(()).unwrap();
    tick.join().unwrap();
    assert_eq!(pool.tick(), 1);
    assert_eq!(handle.try_recv(), Some(9));
    assert_eq!(polls.load(Ordering::SeqCst), 2);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn standard_async_runtime_can_await_task_handle_without_polling_it() {
    let pool = ComputePool::new(IoRuntime::new());
    let handle = pool.spawn(Priority::Low, |_| async { 42 });
    let io = IoRuntime::new();
    use redlilium_core::compute::IoRunner;
    let awaited = io.run(async move { handle.await });
    pool.tick();
    assert_eq!(
        awaited.recv_timeout(std::time::Duration::from_secs(10)),
        Some(Some(42))
    );
}
