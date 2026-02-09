use std::future::Future;
use std::marker::PhantomData;
use std::marker::PhantomPinned;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Maximum inline storage for a system future (bytes).
///
/// Most system futures capture only a few references and small state,
/// so 256 bytes is generous. If a future exceeds this, `SystemFuture::new`
/// will panic with a clear message at runtime.
const SYSTEM_FUTURE_SIZE: usize = 256;

/// A type-erased future stored inline without heap allocation.
///
/// Replaces `Pin<Box<dyn Future<Output = ()> + Send + 'a>>` for system futures.
/// The future is stored in a fixed-size inline buffer, avoiding a heap allocation
/// per system per frame.
///
/// # Safety invariants
///
/// - The inner future is stored at the start of `data` with proper alignment.
/// - After the first `poll`, the `SystemFuture` must not be moved (enforced by `!Unpin`).
/// - `poll_fn` and `drop_fn` are set by `new()` and match the concrete future type.
/// - `Send` is safe because `new()` requires `F: Send`.
#[repr(C)]
pub struct SystemFuture<'a> {
    // Function pointers first for consistent layout
    poll_fn: unsafe fn(*mut u8, &mut Context<'_>) -> Poll<()>,
    drop_fn: unsafe fn(*mut u8),
    // Aligned inline storage for the future
    data: AlignedStorage,
    _marker: PhantomData<&'a ()>,
    _pin: PhantomPinned,
}

/// 16-byte aligned storage for inline futures.
#[repr(C, align(16))]
struct AlignedStorage([MaybeUninit<u8>; SYSTEM_FUTURE_SIZE]);

impl<'a> SystemFuture<'a> {
    /// Creates a new `SystemFuture` from any future that fits in inline storage.
    ///
    /// # Panics
    ///
    /// Panics if the future exceeds `SYSTEM_FUTURE_SIZE` bytes or requires
    /// alignment greater than 16 bytes.
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = ()> + Send + 'a,
    {
        assert!(
            std::mem::size_of::<F>() <= SYSTEM_FUTURE_SIZE,
            "System future is too large ({} bytes, max {}). \
             Consider reducing captured state or filing an issue to increase the limit.",
            std::mem::size_of::<F>(),
            SYSTEM_FUTURE_SIZE,
        );
        assert!(
            std::mem::align_of::<F>() <= std::mem::align_of::<AlignedStorage>(),
            "System future requires alignment {} (max {}).",
            std::mem::align_of::<F>(),
            std::mem::align_of::<AlignedStorage>(),
        );

        let mut data = AlignedStorage([MaybeUninit::uninit(); SYSTEM_FUTURE_SIZE]);

        // Safety: we checked size and alignment above.
        unsafe {
            let ptr = data.0.as_mut_ptr() as *mut F;
            ptr.write(future);
        }

        Self {
            poll_fn: poll_inner::<F>,
            drop_fn: drop_inner::<F>,
            data,
            _marker: PhantomData,
            _pin: PhantomPinned,
        }
    }
}

/// Polls the stored future.
///
/// # Safety
///
/// - `ptr` must point to a valid, initialized `F` that is pinned in place.
unsafe fn poll_inner<F: Future<Output = ()>>(ptr: *mut u8, cx: &mut Context<'_>) -> Poll<()> {
    unsafe {
        let future = &mut *(ptr as *mut F);
        Pin::new_unchecked(future).poll(cx)
    }
}

/// Drops the stored future.
///
/// # Safety
///
/// - `ptr` must point to a valid, initialized `F`.
unsafe fn drop_inner<F>(ptr: *mut u8) {
    unsafe {
        let future = ptr as *mut F;
        future.drop_in_place();
    }
}

impl Future for SystemFuture<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // Safety: the outer Pin guarantees we won't move, so the inner future
        // at data.0.as_mut_ptr() is also pinned. poll_fn was set by new() to
        // match the concrete type stored there.
        unsafe {
            let this = self.get_unchecked_mut();
            (this.poll_fn)(this.data.0.as_mut_ptr() as *mut u8, cx)
        }
    }
}

impl Drop for SystemFuture<'_> {
    fn drop(&mut self) {
        // Safety: drop_fn matches the concrete type written by new().
        unsafe {
            (self.drop_fn)(self.data.0.as_mut_ptr() as *mut u8);
        }
    }
}

// Safety: SystemFuture is Send because new() requires F: Send.
unsafe impl Send for SystemFuture<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::Waker;

    fn noop_waker() -> Waker {
        use std::task::{RawWaker, RawWakerVTable};
        fn no_op(_: *const ()) {}
        fn clone(p: *const ()) -> RawWaker {
            RawWaker::new(p, &VTABLE)
        }
        const VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
    }

    #[test]
    fn ready_future_completes() {
        let mut future = SystemFuture::new(async {});
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let pinned = unsafe { Pin::new_unchecked(&mut future) };
        assert!(pinned.poll(&mut cx).is_ready());
    }

    #[test]
    fn captures_and_runs() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = flag.clone();
        let mut future = SystemFuture::new(async move {
            flag2.store(true, Ordering::Relaxed);
        });
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let pinned = unsafe { Pin::new_unchecked(&mut future) };
        assert!(pinned.poll(&mut cx).is_ready());
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn drop_runs_on_incomplete() {
        let flag = Arc::new(AtomicBool::new(false));
        let guard = DropGuard(flag.clone());
        {
            // DropGuard is captured by the async block.
            // When SystemFuture is dropped without polling,
            // the captured guard must still be dropped.
            let _future = SystemFuture::new(async move {
                let _g = guard;
                std::future::pending::<()>().await;
            });
            // _future is dropped here
        }
        assert!(flag.load(Ordering::Relaxed));
    }

    struct DropGuard(Arc<AtomicBool>);
    impl Drop for DropGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    #[test]
    #[should_panic(expected = "too large")]
    fn oversized_future_panics() {
        let big = [0u8; 512];
        SystemFuture::new(async move {
            // Force capture of `big` by using it.
            std::hint::black_box(&big);
        });
    }
}
