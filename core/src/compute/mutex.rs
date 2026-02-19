use std::cell::UnsafeCell;
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

/// A mutex designed for cooperative async executors with noop wakers.
///
/// Unlike `std::sync::Mutex`, calling `.lock().await` on a contended
/// `ComputeMutex` returns `Poll::Pending` instead of blocking the thread.
/// The executor polls the task again later, at which point the lock may
/// be available.
///
/// This is critical for [`ComputePool`](crate::compute) tasks: if two tasks
/// on the same thread contend for a `std::sync::Mutex`, the thread blocks
/// and can never poll the holder to release it — a deadlock.
/// `ComputeMutex` avoids this by yielding to the executor when contended.
///
/// # Sync access
///
/// For sync systems or non-async code, use
/// [`try_lock()`](ComputeMutex::try_lock) which returns `None` if contended.
///
/// # Example
///
/// ```ignore
/// use redlilium_core::compute::ComputeMutex;
/// use std::sync::Arc;
///
/// let mutex = Arc::new(ComputeMutex::new(Vec::new()));
///
/// pool.spawn(Priority::Low, |_ctx| {
///     let m = mutex.clone();
///     async move {
///         let mut guard = m.lock().await;
///         guard.push(42);
///     }
/// });
/// ```
pub struct ComputeMutex<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

// SAFETY: The mutex synchronizes access to `T` via atomic operations.
// `T: Send` is required because the data can be accessed from different threads.
unsafe impl<T: Send> Send for ComputeMutex<T> {}
unsafe impl<T: Send> Sync for ComputeMutex<T> {}

impl<T> ComputeMutex<T> {
    /// Creates a new unlocked mutex containing `value`.
    pub fn new(value: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(value),
        }
    }

    /// Attempts to acquire the lock without blocking.
    ///
    /// Returns `Some(guard)` if the lock was acquired, `None` if contended.
    pub fn try_lock(&self) -> Option<ComputeMutexGuard<'_, T>> {
        self.locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| ComputeMutexGuard {
                mutex: self,
                _not_send: PhantomData,
            })
    }

    /// Returns a future that acquires the lock.
    ///
    /// Each poll attempts [`try_lock()`](Self::try_lock). If contended,
    /// the future returns `Poll::Pending` so the cooperative executor
    /// can poll other tasks. Does not register wakers — relies on the
    /// executor re-polling all pending tasks (noop waker model).
    pub fn lock(&self) -> ComputeMutexLock<'_, T> {
        ComputeMutexLock { mutex: self }
    }

    /// Consumes the mutex and returns the inner value.
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    /// Returns a mutable reference to the inner value.
    ///
    /// Since this requires `&mut self`, no locking is needed.
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for ComputeMutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.try_lock() {
            Some(guard) => f
                .debug_struct("ComputeMutex")
                .field("data", &*guard)
                .finish(),
            None => f
                .debug_struct("ComputeMutex")
                .field("data", &"<locked>")
                .finish(),
        }
    }
}

impl<T: Default> Default for ComputeMutex<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<T> for ComputeMutex<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

/// RAII guard that releases the [`ComputeMutex`] on drop.
///
/// This guard is `!Send` to prevent holding it across `.await` points,
/// which would risk deadlocks in cooperative executors.
pub struct ComputeMutexGuard<'a, T> {
    mutex: &'a ComputeMutex<T>,
    /// Make `!Send` so the guard cannot be held across `.await` points.
    _not_send: PhantomData<*mut ()>,
}

impl<T> Deref for ComputeMutexGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: We hold the lock, so exclusive access is guaranteed.
        unsafe { &*self.mutex.data.get() }
    }
}

impl<T> DerefMut for ComputeMutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: We hold the lock, so exclusive access is guaranteed.
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<T> Drop for ComputeMutexGuard<'_, T> {
    fn drop(&mut self) {
        self.mutex.locked.store(false, Ordering::Release);
    }
}

impl<T: fmt::Debug> fmt::Debug for ComputeMutexGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

/// Future returned by [`ComputeMutex::lock`].
///
/// On each poll, attempts to acquire the lock via
/// [`try_lock()`](ComputeMutex::try_lock). If contended, returns
/// `Poll::Pending` so the cooperative executor can poll other tasks.
pub struct ComputeMutexLock<'a, T> {
    mutex: &'a ComputeMutex<T>,
}

impl<'a, T> Future for ComputeMutexLock<'a, T> {
    type Output = ComputeMutexGuard<'a, T>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.mutex.try_lock() {
            Some(guard) => Poll::Ready(guard),
            None => Poll::Pending,
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
    fn new_and_try_lock() {
        let mutex = ComputeMutex::new(42u32);
        let guard = mutex.try_lock().unwrap();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn try_lock_contended() {
        let mutex = ComputeMutex::new(0u32);
        let _guard = mutex.try_lock().unwrap();
        assert!(mutex.try_lock().is_none());
    }

    #[test]
    fn guard_drop_releases() {
        let mutex = ComputeMutex::new(0u32);
        {
            let _guard = mutex.try_lock().unwrap();
        }
        // Lock released — should succeed again.
        assert!(mutex.try_lock().is_some());
    }

    #[test]
    fn lock_future_ready_when_free() {
        let mutex = ComputeMutex::new(7u32);
        let mut fut = mutex.lock();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut fut).poll(&mut cx) {
            Poll::Ready(guard) => assert_eq!(*guard, 7),
            Poll::Pending => panic!("expected Ready"),
        }
    }

    #[test]
    fn lock_future_pending_when_contended() {
        let mutex = ComputeMutex::new(0u32);
        let _guard = mutex.try_lock().unwrap();

        let mut fut = mutex.lock();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());
    }

    #[test]
    fn lock_future_ready_after_release() {
        let mutex = ComputeMutex::new(0u32);
        let guard = mutex.try_lock().unwrap();

        let mut fut = mutex.lock();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // Contended — Pending.
        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());

        // Release the lock.
        drop(guard);

        // Now free — Ready.
        assert!(Pin::new(&mut fut).poll(&mut cx).is_ready());
    }

    #[test]
    fn into_inner() {
        let mutex = ComputeMutex::new(String::from("hello"));
        assert_eq!(mutex.into_inner(), "hello");
    }

    #[test]
    fn get_mut() {
        let mut mutex = ComputeMutex::new(10u32);
        *mutex.get_mut() = 20;
        assert_eq!(*mutex.get_mut(), 20);
    }

    #[test]
    fn debug_format_unlocked() {
        let mutex = ComputeMutex::new(42u32);
        let s = format!("{mutex:?}");
        assert!(s.contains("42"), "expected data in debug output: {s}");
    }

    #[test]
    fn debug_format_locked() {
        let mutex = ComputeMutex::new(42u32);
        let _guard = mutex.try_lock().unwrap();
        let s = format!("{mutex:?}");
        assert!(
            s.contains("<locked>"),
            "expected <locked> in debug output: {s}"
        );
    }

    #[test]
    fn default_impl() {
        let mutex: ComputeMutex<u32> = ComputeMutex::default();
        assert_eq!(*mutex.try_lock().unwrap(), 0);
    }

    #[test]
    fn from_impl() {
        let mutex = ComputeMutex::from(99u32);
        assert_eq!(*mutex.try_lock().unwrap(), 99);
    }

    #[test]
    fn send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ComputeMutex<Vec<u8>>>();
    }
}
