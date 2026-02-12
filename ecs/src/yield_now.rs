use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// Default yield interval: 3ms.
///
/// Yield points that fire more frequently than this are skipped,
/// reducing context-switch overhead while keeping cooperative
/// multitasking responsive.
static YIELD_INTERVAL_US: AtomicU64 = AtomicU64::new(3000);

thread_local! {
    static LAST_YIELD: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// Sets the minimum interval between actual yields.
///
/// Calls to [`yield_now`] that happen sooner than `interval` after the
/// previous yield will complete immediately without suspending.
///
/// Use `Duration::ZERO` to restore always-yield behaviour.
///
/// The default is 3 ms.
pub fn set_yield_interval(interval: Duration) {
    YIELD_INTERVAL_US.store(interval.as_micros() as u64, Ordering::Relaxed);
}

/// Resets the per-thread yield timer.
///
/// The next [`yield_now`] on this thread will always suspend,
/// regardless of the configured interval.
///
/// Called automatically by [`ComputePool::new()`](crate::ComputePool::new).
pub fn reset_yield_timer() {
    LAST_YIELD.set(None);
}

/// Yields control back to the executor, allowing other tasks to run.
///
/// This is the primary mechanism for cooperative multitasking in
/// async compute tasks. Developers can call this liberally — the
/// runtime will only actually suspend when enough wall-clock time
/// has elapsed since the last real yield (see [`set_yield_interval`]).
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

/// Future returned by [`yield_now`].
///
/// The first poll checks the per-thread yield timer:
/// - If no previous yield, or enough time has elapsed → suspends (`Pending`).
/// - Otherwise → completes immediately (`Ready`).
///
/// The second poll (after a real suspend) always returns `Ready`.
pub struct YieldNow {
    yielded: bool,
}

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            return Poll::Ready(());
        }

        if should_yield() {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        } else {
            // Not enough time since last yield — skip.
            Poll::Ready(())
        }
    }
}

/// Returns `true` when enough wall-clock time has passed since the last
/// real yield on this thread (or if there has been no yield yet).
#[inline]
fn should_yield() -> bool {
    let interval_us = YIELD_INTERVAL_US.load(Ordering::Relaxed);

    // Duration::ZERO → always yield.
    if interval_us == 0 {
        LAST_YIELD.set(Some(Instant::now()));
        return true;
    }

    let now = Instant::now();
    let should = LAST_YIELD
        .get()
        .is_none_or(|last| now.duration_since(last).as_micros() as u64 >= interval_us);

    if should {
        LAST_YIELD.set(Some(now));
    }

    should
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
        reset_yield_timer();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let mut fut = yield_now();
        let mut pinned = Pin::new(&mut fut);

        // First poll: Pending (no previous yield on this thread)
        assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Pending);
        // Second poll: Ready
        assert_eq!(pinned.as_mut().poll(&mut cx), Poll::Ready(()));
    }

    #[test]
    fn skips_yield_within_interval() {
        reset_yield_timer();
        set_yield_interval(Duration::from_secs(10)); // very long interval

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // First yield: always fires (LAST_YIELD is None)
        let mut fut1 = yield_now();
        assert_eq!(Pin::new(&mut fut1).poll(&mut cx), Poll::Pending);

        // Second yield immediately after: should be skipped
        let mut fut2 = yield_now();
        assert_eq!(Pin::new(&mut fut2).poll(&mut cx), Poll::Ready(()));

        // Restore default
        set_yield_interval(Duration::from_millis(3));
    }

    #[test]
    fn zero_interval_always_yields() {
        reset_yield_timer();
        set_yield_interval(Duration::ZERO);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let mut fut1 = yield_now();
        assert_eq!(Pin::new(&mut fut1).poll(&mut cx), Poll::Pending);

        // Even immediately after, should still yield
        let mut fut2 = yield_now();
        assert_eq!(Pin::new(&mut fut2).poll(&mut cx), Poll::Pending);

        // Restore default
        set_yield_interval(Duration::from_millis(3));
    }
}
