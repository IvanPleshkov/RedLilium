use redlilium_core::compute::{CancellationToken, ComputeMutex, ComputeRwLock, IoHandle};
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
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
fn io_waits_for_notification_and_releases_cancelled_waiters() {
    let count = Arc::new(Count::default());
    let waker = Waker::from(count.clone());
    let mut cx = Context::from_waker(&waker);
    for send in [false, true] {
        let (sender, mut handle) = IoHandle::<u32>::channel();
        let before = count.0.load(Ordering::SeqCst);
        assert!(Pin::new(&mut handle).poll(&mut cx).is_pending());
        assert_eq!(count.0.load(Ordering::SeqCst), before);
        if send {
            sender.send(7).unwrap();
        } else {
            drop(sender);
        }
        assert_eq!(count.0.load(Ordering::SeqCst), before + 1);
        assert_eq!(
            Pin::new(&mut handle).poll(&mut cx),
            Poll::Ready(if send { Some(7) } else { None })
        );
    }
    let (sender, mut handle) = IoHandle::<u32>::channel();
    let before = Arc::strong_count(&count);
    assert!(Pin::new(&mut handle).poll(&mut cx).is_pending());
    assert_eq!(Arc::strong_count(&count), before + 1);
    drop(handle);
    assert_eq!(Arc::strong_count(&count), before);
    drop(sender);
}
#[test]
fn cancellation_clones_notify_all_live_registrations() {
    let count = Arc::new(Count::default());
    let waker = Waker::from(count.clone());
    let token = CancellationToken::new();
    let first = token.on_cancel(&waker);
    let removed = token.on_cancel(&waker);
    drop(removed);
    let second = token.on_cancel(&waker);
    token.clone().cancel();
    assert_eq!(count.0.load(Ordering::SeqCst), 2);
    token.cancel();
    assert_eq!(count.0.load(Ordering::SeqCst), 2);
    let late = token.on_cancel(&waker);
    assert_eq!(count.0.load(Ordering::SeqCst), 3);
    drop((first, second, late));
}
#[test]
fn mutex_unlock_wakes_waiters_and_dropped_waits_unregister() {
    let count = Arc::new(Count::default());
    let waker = Waker::from(count.clone());
    let mut cx = Context::from_waker(&waker);
    let mutex = ComputeMutex::new(7);
    let guard = mutex.try_lock().unwrap();
    let mut waiting = mutex.lock();
    assert!(Pin::new(&mut waiting).poll(&mut cx).is_pending());
    assert_eq!(count.0.load(Ordering::SeqCst), 0);
    drop(guard);
    assert_eq!(count.0.load(Ordering::SeqCst), 1);
    let guard = match Pin::new(&mut waiting).poll(&mut cx) {
        Poll::Ready(g) => g,
        _ => panic!(),
    };
    let mut abandoned = mutex.lock();
    assert!(Pin::new(&mut abandoned).poll(&mut cx).is_pending());
    drop(abandoned);
    drop(guard);
    assert_eq!(count.0.load(Ordering::SeqCst), 1);
}
#[test]
fn rwlock_unlock_and_writer_cancellation_wake_readers() {
    let lock = ComputeRwLock::new(3);
    let count = Arc::new(Count::default());
    let waker = Waker::from(count.clone());
    let mut cx = Context::from_waker(&waker);
    let held = lock.try_read().unwrap();
    let mut writer = lock.write();
    assert!(Pin::new(&mut writer).poll(&mut cx).is_pending());
    let mut reader = lock.read();
    assert!(Pin::new(&mut reader).poll(&mut cx).is_pending());
    drop(writer); // release writer preference and notify blocked readers
    assert_eq!(count.0.load(Ordering::SeqCst), 1);
    assert!(Pin::new(&mut reader).poll(&mut cx).is_ready());
    drop(held);
    let writer = lock.try_write().unwrap();
    let mut reader = lock.read();
    assert!(Pin::new(&mut reader).poll(&mut cx).is_pending());
    drop(writer);
    assert_eq!(count.0.load(Ordering::SeqCst), 2);
    assert!(Pin::new(&mut reader).poll(&mut cx).is_ready());
}
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn io_publication_racing_registration_never_loses_a_wake() {
    for _ in 0..100 {
        let (sender, mut handle) = IoHandle::channel();
        let count = Arc::new(Count::default());
        let waker = Waker::from(count.clone());
        let thread = std::thread::spawn(move || sender.send(11).unwrap());
        let result = Pin::new(&mut handle).poll(&mut Context::from_waker(&waker));
        thread.join().unwrap();
        if result.is_pending() {
            assert!(count.0.load(Ordering::SeqCst) > 0);
            assert_eq!(
                Pin::new(&mut handle).poll(&mut Context::from_waker(&waker)),
                Poll::Ready(Some(11))
            );
        } else {
            assert_eq!(result, Poll::Ready(Some(11)));
        }
    }
}
