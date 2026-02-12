use std::future::Future;
use std::pin::Pin;
use std::sync::mpsc;
use std::task::{Context, Poll};

/// Handle to an async IO task running on a real async runtime.
///
/// Works with cooperative executors using noop wakers via channel-based
/// polling. The actual IO runs on a real runtime (e.g. tokio) with real
/// wakers — only the result delivery uses the channel.
///
/// # Example
///
/// ```ignore
/// let handle = ctx.io().run(async {
///     reqwest::get("https://api.example.com/data").await?.text().await
/// });
///
/// // Await in a compute task (polls channel each tick)
/// let result = handle.await;
/// ```
pub struct IoHandle<T> {
    receiver: mpsc::Receiver<T>,
}

impl<T> IoHandle<T> {
    /// Creates a new IO handle wrapping the given receiver.
    pub fn new(receiver: mpsc::Receiver<T>) -> Self {
        Self { receiver }
    }

    /// Attempts to retrieve the result without blocking.
    ///
    /// Returns `Some(T)` if the IO task has completed, `None` otherwise.
    /// This consumes the value — subsequent calls return `None`.
    pub fn try_recv(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }

    /// Blocks until the IO task completes and returns the result.
    ///
    /// Returns `None` if the sender was dropped without sending.
    ///
    /// # Warning
    ///
    /// This blocks the calling thread. Prefer `try_recv()` or `.await`
    /// in system loops.
    pub fn recv(self) -> Option<T> {
        self.receiver.recv().ok()
    }
}

impl<T> Future for IoHandle<T> {
    type Output = Option<T>;

    /// Polls the IO task for completion.
    ///
    /// Returns `Poll::Ready(Some(T))` if the task completed,
    /// `Poll::Ready(None)` if the sender was dropped,
    /// `Poll::Pending` if the task is still running.
    ///
    /// Designed for manual polling with a noop waker — the real async
    /// runtime drives the IO; this just checks the channel.
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<T>> {
        match self.receiver.try_recv() {
            Ok(val) => Poll::Ready(Some(val)),
            Err(mpsc::TryRecvError::Empty) => Poll::Pending,
            Err(mpsc::TryRecvError::Disconnected) => Poll::Ready(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::{RawWaker, RawWakerVTable, Waker};

    fn noop_waker() -> Waker {
        fn noop(_: *const ()) {}
        fn clone(p: *const ()) -> RawWaker {
            RawWaker::new(p, &VTABLE)
        }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
        unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
    }

    #[test]
    fn try_recv_empty() {
        let (_tx, rx) = mpsc::channel::<u32>();
        let handle = IoHandle::new(rx);
        assert!(handle.try_recv().is_none());
    }

    #[test]
    fn try_recv_ready() {
        let (tx, rx) = mpsc::channel();
        tx.send(42u32).unwrap();
        let handle = IoHandle::new(rx);
        assert_eq!(handle.try_recv(), Some(42));
    }

    #[test]
    fn recv_blocks() {
        let (tx, rx) = mpsc::channel();
        tx.send(99u32).unwrap();
        let handle = IoHandle::new(rx);
        assert_eq!(handle.recv(), Some(99));
    }

    #[test]
    fn recv_disconnected() {
        let (tx, rx) = mpsc::channel::<u32>();
        drop(tx);
        let handle = IoHandle::new(rx);
        assert_eq!(handle.recv(), None);
    }

    #[test]
    fn future_pending_then_ready() {
        let (tx, rx) = mpsc::channel();
        let mut handle = IoHandle::new(rx);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // Before send: Pending
        assert!(Pin::new(&mut handle).poll(&mut cx).is_pending());

        // Send result
        tx.send(77u32).unwrap();

        // After send: Ready
        match Pin::new(&mut handle).poll(&mut cx) {
            Poll::Ready(Some(77)) => {}
            other => panic!("Expected Ready(Some(77)), got {other:?}"),
        }
    }

    #[test]
    fn future_disconnected() {
        let (tx, rx) = mpsc::channel::<u32>();
        let mut handle = IoHandle::new(rx);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        drop(tx);
        match Pin::new(&mut handle).poll(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("Expected Ready(None), got {other:?}"),
        }
    }
}
