use std::cell::UnsafeCell;
use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::atomic::{AtomicI32, Ordering};
use std::task::{Context, Poll};

/// State value indicating an active writer.
const WRITER: i32 = -1;
/// State value indicating no readers or writers.
const UNLOCKED: i32 = 0;

/// A reader-writer lock designed for cooperative async executors with noop wakers.
///
/// Allows multiple concurrent readers **or** a single exclusive writer.
/// When contended in an async context, returns `Poll::Pending` instead of
/// blocking — same principle as [`ComputeMutex`](super::ComputeMutex).
///
/// # State encoding
///
/// - `0` — unlocked (no readers or writers)
/// - Positive N — N active readers
/// - `-1` — one active writer
///
/// # Sync access
///
/// For sync systems, use [`try_read()`](Self::try_read) and
/// [`try_write()`](Self::try_write) which return `None` if contended.
///
/// # Example
///
/// ```ignore
/// use redlilium_core::compute::ComputeRwLock;
/// use std::sync::Arc;
///
/// let lock = Arc::new(ComputeRwLock::new(Config::default()));
///
/// // Multiple readers concurrently:
/// let guard1 = lock.try_read().unwrap();
/// let guard2 = lock.try_read().unwrap();
/// assert_eq!(*guard1, *guard2);
/// ```
pub struct ComputeRwLock<T> {
    /// 0 = unlocked, positive = reader count, -1 = writer.
    state: AtomicI32,
    data: UnsafeCell<T>,
}

// SAFETY: The lock synchronizes access to `T` via atomic operations.
// `T: Send` is required because the data can move between threads.
// `T: Sync` is required because multiple readers may access `&T` concurrently.
unsafe impl<T: Send> Send for ComputeRwLock<T> {}
unsafe impl<T: Send + Sync> Sync for ComputeRwLock<T> {}

impl<T> ComputeRwLock<T> {
    /// Creates a new unlocked reader-writer lock containing `value`.
    pub fn new(value: T) -> Self {
        Self {
            state: AtomicI32::new(UNLOCKED),
            data: UnsafeCell::new(value),
        }
    }

    /// Attempts to acquire a shared read lock without blocking.
    ///
    /// Returns `Some(guard)` if successful, `None` if a writer is active.
    pub fn try_read(&self) -> Option<ComputeReadGuard<'_, T>> {
        loop {
            let current = self.state.load(Ordering::Relaxed);
            if current == WRITER {
                return None;
            }
            match self.state.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(ComputeReadGuard {
                        lock: self,
                        _not_send: PhantomData,
                    });
                }
                // Another reader changed the count concurrently — retry.
                Err(_) => continue,
            }
        }
    }

    /// Attempts to acquire an exclusive write lock without blocking.
    ///
    /// Returns `Some(guard)` if successful, `None` if any readers or
    /// another writer is active.
    pub fn try_write(&self) -> Option<ComputeWriteGuard<'_, T>> {
        self.state
            .compare_exchange(UNLOCKED, WRITER, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| ComputeWriteGuard {
                lock: self,
                _not_send: PhantomData,
            })
    }

    /// Returns a future that acquires a shared read lock.
    ///
    /// Each poll attempts [`try_read()`](Self::try_read). If a writer is
    /// active, returns `Poll::Pending`.
    pub fn read(&self) -> ComputeRwLockRead<'_, T> {
        ComputeRwLockRead { lock: self }
    }

    /// Returns a future that acquires an exclusive write lock.
    ///
    /// Each poll attempts [`try_write()`](Self::try_write). If any readers
    /// or another writer is active, returns `Poll::Pending`.
    pub fn write(&self) -> ComputeRwLockWrite<'_, T> {
        ComputeRwLockWrite { lock: self }
    }

    /// Consumes the lock and returns the inner value.
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

impl<T: fmt::Debug> fmt::Debug for ComputeRwLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.try_read() {
            Some(guard) => f
                .debug_struct("ComputeRwLock")
                .field("data", &*guard)
                .finish(),
            None => f
                .debug_struct("ComputeRwLock")
                .field("data", &"<locked>")
                .finish(),
        }
    }
}

impl<T: Default> Default for ComputeRwLock<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<T> for ComputeRwLock<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

// ---------------------------------------------------------------------------
// Read guard
// ---------------------------------------------------------------------------

/// RAII guard for shared read access to a [`ComputeRwLock`].
///
/// `!Send` to prevent holding across `.await` points.
pub struct ComputeReadGuard<'a, T> {
    lock: &'a ComputeRwLock<T>,
    _not_send: PhantomData<*mut ()>,
}

impl<T> Deref for ComputeReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: Read access is safe — the state guarantees no writer is active.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> Drop for ComputeReadGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.fetch_sub(1, Ordering::Release);
    }
}

impl<T: fmt::Debug> fmt::Debug for ComputeReadGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

// ---------------------------------------------------------------------------
// Write guard
// ---------------------------------------------------------------------------

/// RAII guard for exclusive write access to a [`ComputeRwLock`].
///
/// `!Send` to prevent holding across `.await` points.
pub struct ComputeWriteGuard<'a, T> {
    lock: &'a ComputeRwLock<T>,
    _not_send: PhantomData<*mut ()>,
}

impl<T> Deref for ComputeWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: Exclusive access is guaranteed — the state is WRITER.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for ComputeWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: Exclusive access is guaranteed — the state is WRITER.
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for ComputeWriteGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.store(UNLOCKED, Ordering::Release);
    }
}

impl<T: fmt::Debug> fmt::Debug for ComputeWriteGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

// ---------------------------------------------------------------------------
// Lock futures
// ---------------------------------------------------------------------------

/// Future returned by [`ComputeRwLock::read`].
///
/// On each poll, attempts to acquire a shared read lock. If a writer is
/// active, returns `Poll::Pending`.
pub struct ComputeRwLockRead<'a, T> {
    lock: &'a ComputeRwLock<T>,
}

impl<'a, T> Future for ComputeRwLockRead<'a, T> {
    type Output = ComputeReadGuard<'a, T>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.lock.try_read() {
            Some(guard) => Poll::Ready(guard),
            None => Poll::Pending,
        }
    }
}

/// Future returned by [`ComputeRwLock::write`].
///
/// On each poll, attempts to acquire an exclusive write lock. If any
/// readers or another writer is active, returns `Poll::Pending`.
pub struct ComputeRwLockWrite<'a, T> {
    lock: &'a ComputeRwLock<T>,
}

impl<'a, T> Future for ComputeRwLockWrite<'a, T> {
    type Output = ComputeWriteGuard<'a, T>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.lock.try_write() {
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
    fn try_read_when_free() {
        let lock = ComputeRwLock::new(42u32);
        let guard = lock.try_read().unwrap();
        assert_eq!(*guard, 42);
    }

    #[test]
    fn multiple_readers() {
        let lock = ComputeRwLock::new(10u32);
        let g1 = lock.try_read().unwrap();
        let g2 = lock.try_read().unwrap();
        let g3 = lock.try_read().unwrap();
        assert_eq!(*g1, 10);
        assert_eq!(*g2, 10);
        assert_eq!(*g3, 10);
    }

    #[test]
    fn try_write_when_free() {
        let lock = ComputeRwLock::new(0u32);
        let mut guard = lock.try_write().unwrap();
        *guard = 99;
        assert_eq!(*guard, 99);
    }

    #[test]
    fn try_write_fails_with_reader() {
        let lock = ComputeRwLock::new(0u32);
        let _reader = lock.try_read().unwrap();
        assert!(lock.try_write().is_none());
    }

    #[test]
    fn try_read_fails_with_writer() {
        let lock = ComputeRwLock::new(0u32);
        let _writer = lock.try_write().unwrap();
        assert!(lock.try_read().is_none());
    }

    #[test]
    fn try_write_fails_with_writer() {
        let lock = ComputeRwLock::new(0u32);
        let _writer = lock.try_write().unwrap();
        assert!(lock.try_write().is_none());
    }

    #[test]
    fn read_releases_correctly() {
        let lock = ComputeRwLock::new(0u32);
        {
            let _reader = lock.try_read().unwrap();
        }
        // Fully released — write should succeed.
        assert!(lock.try_write().is_some());
    }

    #[test]
    fn write_releases_correctly() {
        let lock = ComputeRwLock::new(0u32);
        {
            let _writer = lock.try_write().unwrap();
        }
        assert!(lock.try_read().is_some());
        assert!(lock.try_write().is_some());
    }

    #[test]
    fn read_future_ready_when_free() {
        let lock = ComputeRwLock::new(7u32);
        let mut fut = lock.read();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut fut).poll(&mut cx) {
            Poll::Ready(guard) => assert_eq!(*guard, 7),
            Poll::Pending => panic!("expected Ready"),
        }
    }

    #[test]
    fn read_future_pending_with_writer() {
        let lock = ComputeRwLock::new(0u32);
        let _writer = lock.try_write().unwrap();

        let mut fut = lock.read();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());
    }

    #[test]
    fn write_future_ready_when_free() {
        let lock = ComputeRwLock::new(0u32);
        let mut fut = lock.write();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        assert!(Pin::new(&mut fut).poll(&mut cx).is_ready());
    }

    #[test]
    fn write_future_pending_with_reader() {
        let lock = ComputeRwLock::new(0u32);
        let _reader = lock.try_read().unwrap();

        let mut fut = lock.write();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());
    }

    #[test]
    fn write_future_ready_after_readers_release() {
        let lock = ComputeRwLock::new(0u32);
        let r1 = lock.try_read().unwrap();
        let r2 = lock.try_read().unwrap();

        let mut fut = lock.write();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // Contended — Pending.
        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());

        drop(r1);
        // Still one reader — Pending.
        assert!(Pin::new(&mut fut).poll(&mut cx).is_pending());

        drop(r2);
        // All readers gone — Ready.
        assert!(Pin::new(&mut fut).poll(&mut cx).is_ready());
    }

    #[test]
    fn into_inner() {
        let lock = ComputeRwLock::new(String::from("hello"));
        assert_eq!(lock.into_inner(), "hello");
    }

    #[test]
    fn get_mut() {
        let mut lock = ComputeRwLock::new(10u32);
        *lock.get_mut() = 20;
        assert_eq!(*lock.get_mut(), 20);
    }

    #[test]
    fn debug_format_unlocked() {
        let lock = ComputeRwLock::new(42u32);
        let s = format!("{lock:?}");
        assert!(s.contains("42"), "expected data in debug output: {s}");
    }

    #[test]
    fn debug_format_locked() {
        let lock = ComputeRwLock::new(42u32);
        let _writer = lock.try_write().unwrap();
        let s = format!("{lock:?}");
        assert!(
            s.contains("<locked>"),
            "expected <locked> in debug output: {s}"
        );
    }

    #[test]
    fn default_impl() {
        let lock: ComputeRwLock<u32> = ComputeRwLock::default();
        assert_eq!(*lock.try_read().unwrap(), 0);
    }

    #[test]
    fn from_impl() {
        let lock = ComputeRwLock::from(77u32);
        assert_eq!(*lock.try_read().unwrap(), 77);
    }

    #[test]
    fn send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ComputeRwLock<Vec<u8>>>();
    }
}
