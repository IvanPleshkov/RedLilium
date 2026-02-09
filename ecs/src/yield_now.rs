use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Yields control back to the executor, allowing other tasks to run.
///
/// This is the primary mechanism for cooperative multitasking in
/// async compute tasks. Tasks should yield periodically to avoid
/// blocking the thread pool.
///
/// # Example
///
/// ```ignore
/// pool.spawn(Priority::Low, async move {
///     for chunk in data.chunks(256) {
///         process(chunk);
///         yield_now().await;
///     }
/// });
/// ```
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}

/// Future that yields once then completes.
///
/// Returns `Pending` on the first poll (scheduling a wakeup),
/// then `Ready(())` on the second poll.
pub struct YieldNow {
    yielded: bool,
}

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
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
    fn yields_once_then_ready() {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let mut fut = yield_now();
        let mut pinned = Pin::new(&mut fut);

        // First poll: Pending
        assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);
        // Second poll: Ready
        assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Ready(()));
    }
}
