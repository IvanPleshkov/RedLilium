use super::waiters::Waiters;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Waker};

use super::yield_now::{YieldNow, yield_now};

/// Error returned when a task is cancelled at a checkpoint.
///
/// Compute tasks that use [`checkpoint()`](super::ComputeContext::checkpoint)
/// receive this error when the task's handle signals cancellation. Tasks can
/// propagate it with `?` to stop early at the next yield point.
///
/// # Example
///
/// ```ignore
/// ctx.compute().spawn(Priority::Low, |cctx| async move {
///     for chunk in data.chunks(256) {
///         process(chunk);
///         cctx.checkpoint().await?;
///     }
///     Ok(result)
/// });
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("task cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// Token that signals cancellation to cooperative async tasks.
///
/// Cloning a token creates another handle to the same cancellation flag.
/// Calling [`cancel()`](CancellationToken::cancel) on any clone affects all.
#[derive(Clone)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

struct CancellationState {
    flag: AtomicBool,
    waiters: Waiters,
}

/// Removes a cancellation listener when its task is destroyed.
pub struct CancellationRegistration {
    state: Arc<CancellationState>,
    key: Option<u64>,
}
impl Drop for CancellationRegistration {
    fn drop(&mut self) {
        self.state.waiters.remove(self.key);
    }
}

impl CancellationToken {
    /// Creates a new cancellation token (not cancelled).
    pub fn new() -> Self {
        Self {
            state: Arc::new(CancellationState {
                flag: AtomicBool::new(false),
                waiters: Waiters::default(),
            }),
        }
    }

    /// Signals cancellation.
    pub fn cancel(&self) {
        self.state.flag.store(true, Ordering::Release);
        self.state.waiters.wake_all();
    }

    /// Wake a task when any clone requests cancellation. Keep the registration
    /// alive while waiting; dropping it releases the stored waker.
    pub fn on_cancel(&self, waker: &Waker) -> CancellationRegistration {
        let mut key = None;
        self.state.waiters.register(&mut key, waker);
        if self.is_cancelled() {
            self.state.waiters.wake_all();
        }
        CancellationRegistration {
            state: self.state.clone(),
            key,
        }
    }

    /// Returns whether cancellation has been signalled.
    pub fn is_cancelled(&self) -> bool {
        self.state.flag.load(Ordering::Acquire)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Future returned by [`ComputeContext::checkpoint`](super::ComputeContext::checkpoint).
///
/// Behaves like [`YieldNow`] but also checks a cancellation token.
/// If the token is cancelled, returns `Err(Cancelled)` immediately.
/// Otherwise yields (when the interval has elapsed) and returns `Ok(())`.
pub struct Checkpoint {
    inner: YieldNow,
    token: Option<CancellationToken>,
}

impl Checkpoint {
    /// Creates a checkpoint that only yields (no cancellation).
    pub fn yield_only() -> Self {
        Self {
            inner: yield_now(),
            token: None,
        }
    }

    /// Creates a checkpoint that yields and checks the given token.
    pub fn with_token(token: CancellationToken) -> Self {
        Self {
            inner: yield_now(),
            token: Some(token),
        }
    }
}

impl Future for Checkpoint {
    type Output = Result<(), Cancelled>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Cancelled>> {
        if let Some(token) = &self.token
            && token.is_cancelled()
        {
            return Poll::Ready(Err(Cancelled));
        }

        match Pin::new(&mut self.inner).poll(cx) {
            Poll::Ready(()) => Poll::Ready(Ok(())),
            Poll::Pending => Poll::Pending,
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
    fn checkpoint_without_token_yields() {
        super::super::yield_now::reset_yield_timer();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let mut cp = Checkpoint::yield_only();
        // First poll: yields (Pending)
        assert_eq!(Pin::new(&mut cp).poll(&mut cx), Poll::Pending);
        // Second poll: completes with Ok
        assert_eq!(Pin::new(&mut cp).poll(&mut cx), Poll::Ready(Ok(())));
    }

    #[test]
    fn checkpoint_with_uncancelled_token_yields() {
        super::super::yield_now::reset_yield_timer();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let token = CancellationToken::new();
        let mut cp = Checkpoint::with_token(token);

        assert_eq!(Pin::new(&mut cp).poll(&mut cx), Poll::Pending);
        assert_eq!(Pin::new(&mut cp).poll(&mut cx), Poll::Ready(Ok(())));
    }

    #[test]
    fn checkpoint_returns_cancelled_immediately() {
        super::super::yield_now::reset_yield_timer();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let token = CancellationToken::new();
        token.cancel();

        let mut cp = Checkpoint::with_token(token);
        assert_eq!(Pin::new(&mut cp).poll(&mut cx), Poll::Ready(Err(Cancelled)));
    }

    #[test]
    fn checkpoint_cancelled_mid_yield() {
        super::super::yield_now::reset_yield_timer();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        let token = CancellationToken::new();
        let mut cp = Checkpoint::with_token(token.clone());

        // First poll: yields (not yet cancelled)
        assert_eq!(Pin::new(&mut cp).poll(&mut cx), Poll::Pending);

        // Cancel between polls
        token.cancel();

        // Second poll: cancelled
        assert_eq!(Pin::new(&mut cp).poll(&mut cx), Poll::Ready(Err(Cancelled)));
    }

    #[test]
    fn cancellation_token_clone_shares_state() {
        let token1 = CancellationToken::new();
        let token2 = token1.clone();

        assert!(!token1.is_cancelled());
        assert!(!token2.is_cancelled());

        token2.cancel();

        assert!(token1.is_cancelled());
        assert!(token2.is_cancelled());
    }
}
