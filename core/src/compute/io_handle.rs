use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, mpsc};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

/// One-shot result sender. Sending or dropping it wakes an awaiting handle.
pub struct IoSender<T> {
    sender: Option<mpsc::Sender<T>>,
    waiter: Arc<Mutex<Option<Waker>>>,
}
impl<T> IoSender<T> {
    pub fn send(mut self, value: T) -> Result<(), mpsc::SendError<T>> {
        self.sender.take().expect("unused sender").send(value)
    }
}
impl<T> Drop for IoSender<T> {
    fn drop(&mut self) {
        drop(self.sender.take());
        let waiter = self.waiter.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(waker) = waiter {
            waker.wake();
        }
    }
}

/// One-shot IO result supporting both synchronous receipt and wake-driven await.
pub struct IoHandle<T> {
    receiver: mpsc::Receiver<T>,
    waiter: Arc<Mutex<Option<Waker>>>,
}
impl<T> IoHandle<T> {
    /// Create a result channel with notification on send and sender destruction.
    pub fn channel() -> (IoSender<T>, Self) {
        let (sender, receiver) = mpsc::channel();
        let waiter = Arc::new(Mutex::new(None));
        (
            IoSender {
                sender: Some(sender),
                waiter: waiter.clone(),
            },
            Self { receiver, waiter },
        )
    }
    pub fn try_recv(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }
    pub fn recv(self) -> Option<T> {
        self.receiver.recv().ok()
    }
    pub fn recv_timeout(&self, timeout: Duration) -> Option<T> {
        self.receiver.recv_timeout(timeout).ok()
    }
    pub fn poll_recv(&self, cx: &mut Context<'_>) -> Poll<Option<T>> {
        let mut waiter = self.waiter.lock().unwrap_or_else(|e| e.into_inner());
        let (result, old) = match self.receiver.try_recv() {
            Ok(value) => (Poll::Ready(Some(value)), waiter.take()),
            Err(mpsc::TryRecvError::Disconnected) => (Poll::Ready(None), waiter.take()),
            Err(mpsc::TryRecvError::Empty) => {
                let old = if waiter.as_ref().is_some_and(|w| w.will_wake(cx.waker())) {
                    None
                } else {
                    waiter.replace(cx.waker().clone())
                };
                (Poll::Pending, old)
            }
        };
        drop(waiter);
        drop(old);
        result
    }
}
impl<T> Future for IoHandle<T> {
    type Output = Option<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.poll_recv(cx)
    }
}
impl<T> Drop for IoHandle<T> {
    fn drop(&mut self) {
        let old = self.waiter.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(old);
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
        let (_tx, handle) = IoHandle::<u32>::channel();
        assert!(handle.try_recv().is_none());
    }

    #[test]
    fn try_recv_ready() {
        let (tx, handle) = IoHandle::channel();
        tx.send(42u32).unwrap();
        assert_eq!(handle.try_recv(), Some(42));
    }

    #[test]
    fn recv_blocks() {
        let (tx, handle) = IoHandle::channel();
        tx.send(99u32).unwrap();
        assert_eq!(handle.recv(), Some(99));
    }

    #[test]
    fn recv_disconnected() {
        let (tx, handle) = IoHandle::<u32>::channel();
        drop(tx);
        assert_eq!(handle.recv(), None);
    }

    #[test]
    fn future_pending_then_ready() {
        let (tx, mut handle) = IoHandle::channel();

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
        let (tx, mut handle) = IoHandle::<u32>::channel();

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        drop(tx);
        match Pin::new(&mut handle).poll(&mut cx) {
            Poll::Ready(None) => {}
            other => panic!("Expected Ready(None), got {other:?}"),
        }
    }
}
