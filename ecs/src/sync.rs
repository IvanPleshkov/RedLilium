//! Cross-image-safe `Mutex`/`RwLock` for the ECS.
//!
//! # Why not `parking_lot`
//!
//! The ECS runs game systems that live in a **separately compiled image** (the
//! game cdylib, ADR-020 #45) while the host holds the same locks. Under static
//! linking both images embed their *own* copy of every dependency's statics and
//! thread-locals. `parking_lot` keeps its wait queue in a per-image
//! `static HASHTABLE` plus a `thread_local! THREAD_DATA`: a thread that parks
//! through the cdylib's copy is enqueued in the cdylib's table, but an unpark
//! issued through the host's copy scans the *host's* table, finds nothing, and
//! clears `PARKED_BIT` without waking anyone. The result is a **lost wakeup →
//! permanent deadlock** the moment a lock is contended across the boundary. A
//! spike reproduced exactly this.
//!
//! # Why `std` is safe here
//!
//! `std`'s `Mutex`/`RwLock` keep all waiter state either *inside the lock
//! object* (shared heap memory) or *in the kernel keyed by the object's
//! address* — never in a per-image `static`:
//!
//! - **Linux / Windows ≥10 / BSD** use the `futex` backend: the futex words
//!   live inside the lock; `futex_wait`/`futex_wake` (Linux `SYS_futex`,
//!   Windows `WaitOnAddress`/`WakeByAddress*`) key the wait queue in the kernel
//!   by the atomic's *address*. Address space is shared across images, so a park
//!   in the cdylib and an unpark in the host meet on the same kernel object.
//! - **macOS** uses the `queue.rs` backend: the queue head is a tagged pointer
//!   in the lock word and the waiter `Node`s live on the parked threads' stacks
//!   (plus a per-`Thread` dispatch semaphore) — again no per-image static.
//!
//! Precondition: both images are built by the **same `rustc`** (guaranteed by
//! #45's single-`cargo build` model and the ABI fingerprint gate), so the lock
//! internals have identical layout on both sides.
//!
//! # Semantics
//!
//! Drop-in for the `parking_lot` subset the ECS uses, with two deliberate
//! differences hidden behind the same API:
//!
//! - **Poison is ignored.** A panic while a guard is held does not poison the
//!   lock; `read`/`write`/`lock` return the guard as `parking_lot` would rather
//!   than a `Result`. The ECS treats a poisoned lock as merely locked.
//! - **`data` is stored beside a `RwLock<()>`/`Mutex<()>`.** The `std` primitive
//!   guards only `()`; the real value sits in an adjacent `UnsafeCell<T>`. This
//!   gives us [`RwLock::data_ptr`] (used by the ECS's up-front
//!   [`acquire_sorted`](crate::World)-style borrow bypass) and keeps the guard
//!   free of a borrow of the `std` guard's payload.

use std::cell::UnsafeCell;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::TryLockError;

// ---------------------------------------------------------------------------
// RwLock
// ---------------------------------------------------------------------------

/// A reader-writer lock backed by [`std::sync::RwLock`], cross-image safe (see
/// the [module docs](self)). Poison is ignored.
///
/// The value lives in an `UnsafeCell<T>` *after* the (zero-payload) `std` lock,
/// so `RwLock<Concrete>` unsizes to `RwLock<dyn Trait>` (the last field carries
/// the type parameter) exactly like `parking_lot::RwLock`.
pub struct RwLock<T: ?Sized> {
    inner: std::sync::RwLock<()>,
    data: UnsafeCell<T>,
}

// SAFETY: the `std` lock serializes access; `T: Send` lets a writer move `T`
// between threads, `T: Sync` lets concurrent readers share `&T`. Mirrors
// `parking_lot::RwLock`'s bounds. The manual impls are required because
// `UnsafeCell<T>` is never `Sync` on its own.
unsafe impl<T: ?Sized + Send> Send for RwLock<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    /// Creates a new, unlocked `RwLock`.
    pub const fn new(value: T) -> Self {
        Self {
            inner: std::sync::RwLock::new(()),
            data: UnsafeCell::new(value),
        }
    }

    /// Consumes the lock, returning the wrapped value.
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }
}

impl<T: ?Sized> RwLock<T> {
    /// Locks for reading, blocking until the lock is available. Ignores poison.
    pub fn read(&self) -> RwLockReadGuard<'_, T> {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        RwLockReadGuard {
            _inner: inner,
            data: self.data.get(),
        }
    }

    /// Attempts to acquire a read lock without blocking. Returns `None` if a
    /// writer holds the lock; ignores poison.
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
        match self.inner.try_read() {
            Ok(inner) => Some(RwLockReadGuard {
                _inner: inner,
                data: self.data.get(),
            }),
            Err(TryLockError::Poisoned(e)) => Some(RwLockReadGuard {
                _inner: e.into_inner(),
                data: self.data.get(),
            }),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// Locks for writing, blocking until the lock is available. Ignores poison.
    pub fn write(&self) -> RwLockWriteGuard<'_, T> {
        let inner = self.inner.write().unwrap_or_else(|e| e.into_inner());
        RwLockWriteGuard {
            _inner: inner,
            data: self.data.get(),
        }
    }

    /// Attempts to acquire a write lock without blocking. Returns `None` if the
    /// lock is held by any reader or writer; ignores poison.
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
        match self.inner.try_write() {
            Ok(inner) => Some(RwLockWriteGuard {
                _inner: inner,
                data: self.data.get(),
            }),
            Err(TryLockError::Poisoned(e)) => Some(RwLockWriteGuard {
                _inner: e.into_inner(),
                data: self.data.get(),
            }),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// Returns a raw pointer to the underlying data.
    ///
    /// The caller is responsible for synchronization: the ECS uses this only
    /// after acquiring the lock up front (see the query-access layer), to reach
    /// the value without threading a guard's lifetime through the borrow.
    pub fn data_ptr(&self) -> *mut T {
        self.data.get()
    }

    /// Returns a mutable reference to the underlying data. No locking is needed
    /// because the borrow checker guarantees exclusive access.
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }
}

impl<T: Default> Default for RwLock<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<T> for RwLock<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for RwLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.try_read() {
            Some(guard) => f.debug_struct("RwLock").field("data", &&*guard).finish(),
            None => f
                .debug_struct("RwLock")
                .field("data", &format_args!("<locked>"))
                .finish(),
        }
    }
}

/// RAII read guard for [`RwLock`]. Releases the lock on drop.
pub struct RwLockReadGuard<'a, T: ?Sized> {
    // Field order matters: `data` (a raw pointer into `RwLock::data`) must drop
    // before `_inner` releases the lock — Rust drops fields in declaration
    // order, so keep the pointer first and the guard last.
    data: *const T,
    _inner: std::sync::RwLockReadGuard<'a, ()>,
}

// SAFETY: while the read guard lives, no writer can run, so `&T` may be shared
// across threads when `T: Sync`. The guard is `!Send` (raw pointer + the `!Send`
// `std` guard), matching `parking_lot`: a read lock must be released on the
// thread that took it.
unsafe impl<T: ?Sized + Sync> Sync for RwLockReadGuard<'_, T> {}

impl<T: ?Sized> Deref for RwLockReadGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: we hold the read lock; the pointer is valid and no `&mut`
        // exists to the data.
        unsafe { &*self.data }
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for RwLockReadGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

/// RAII write guard for [`RwLock`]. Releases the lock on drop.
pub struct RwLockWriteGuard<'a, T: ?Sized> {
    data: *mut T,
    _inner: std::sync::RwLockWriteGuard<'a, ()>,
}

// SAFETY: while the write guard lives, access is exclusive, so `&mut T` is
// uniquely owned; sharing `&Guard` (hence `&T`) across threads is safe when
// `T: Sync`. `!Send` for the same reason as the read guard.
unsafe impl<T: ?Sized + Sync> Sync for RwLockWriteGuard<'_, T> {}

impl<T: ?Sized> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: we hold the write lock; access is exclusive.
        unsafe { &*self.data }
    }
}

impl<T: ?Sized> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: we hold the write lock; access is exclusive.
        unsafe { &mut *self.data }
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for RwLockWriteGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

// ---------------------------------------------------------------------------
// Mutex
// ---------------------------------------------------------------------------

/// A mutual-exclusion lock backed by [`std::sync::Mutex`], cross-image safe
/// (see the [module docs](self)). Poison is ignored.
pub struct Mutex<T: ?Sized> {
    inner: std::sync::Mutex<()>,
    data: UnsafeCell<T>,
}

// SAFETY: the `std` lock serializes all access, so only `T: Send` is needed for
// both `Send` and `Sync`. Mirrors `parking_lot::Mutex`.
unsafe impl<T: ?Sized + Send> Send for Mutex<T> {}
unsafe impl<T: ?Sized + Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    /// Creates a new, unlocked `Mutex`.
    pub const fn new(value: T) -> Self {
        Self {
            inner: std::sync::Mutex::new(()),
            data: UnsafeCell::new(value),
        }
    }

    /// Consumes the mutex, returning the wrapped value.
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }
}

impl<T: ?Sized> Mutex<T> {
    /// Locks the mutex, blocking until it is available. Ignores poison.
    pub fn lock(&self) -> MutexGuard<'_, T> {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        MutexGuard {
            _inner: inner,
            data: self.data.get(),
        }
    }

    /// Attempts to lock without blocking. Returns `None` if already locked;
    /// ignores poison.
    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        match self.inner.try_lock() {
            Ok(inner) => Some(MutexGuard {
                _inner: inner,
                data: self.data.get(),
            }),
            Err(TryLockError::Poisoned(e)) => Some(MutexGuard {
                _inner: e.into_inner(),
                data: self.data.get(),
            }),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// Returns a raw pointer to the underlying data. See [`RwLock::data_ptr`].
    pub fn data_ptr(&self) -> *mut T {
        self.data.get()
    }

    /// Returns a mutable reference to the underlying data. No locking needed —
    /// the exclusive borrow guarantees no other access exists.
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }
}

impl<T: Default> Default for Mutex<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T> From<T> for Mutex<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.try_lock() {
            Some(guard) => f.debug_struct("Mutex").field("data", &&*guard).finish(),
            None => f
                .debug_struct("Mutex")
                .field("data", &format_args!("<locked>"))
                .finish(),
        }
    }
}

/// RAII guard for [`Mutex`]. Releases the lock on drop.
pub struct MutexGuard<'a, T: ?Sized> {
    data: *mut T,
    _inner: std::sync::MutexGuard<'a, ()>,
}

// SAFETY: exclusive access while held; `&Guard` (hence `&T`) is shareable when
// `T: Sync`. `!Send`, matching `parking_lot`.
unsafe impl<T: ?Sized + Sync> Sync for MutexGuard<'_, T> {}

impl<T: ?Sized> Deref for MutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: we hold the lock; access is exclusive.
        unsafe { &*self.data }
    }
}

impl<T: ?Sized> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: we hold the lock; access is exclusive.
        unsafe { &mut *self.data }
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for MutexGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn rwlock_read_write() {
        let lock = RwLock::new(1u32);
        assert_eq!(*lock.read(), 1);
        *lock.write() = 2;
        assert_eq!(*lock.read(), 2);
    }

    #[test]
    fn rwlock_multiple_readers() {
        let lock = RwLock::new(5u32);
        let a = lock.read();
        let b = lock.read();
        assert_eq!(*a + *b, 10);
    }

    #[test]
    fn rwlock_try_write_blocked_by_reader() {
        let lock = RwLock::new(0u32);
        let _r = lock.read();
        assert!(lock.try_write().is_none());
    }

    #[test]
    fn rwlock_try_read_blocked_by_writer() {
        let lock = RwLock::new(0u32);
        let _w = lock.write();
        assert!(lock.try_read().is_none());
    }

    #[test]
    fn rwlock_data_ptr() {
        let lock = RwLock::new(42u32);
        // SAFETY: no guard outstanding, single-threaded.
        unsafe {
            *lock.data_ptr() = 99;
        }
        assert_eq!(*lock.read(), 99);
    }

    #[test]
    fn rwlock_poison_is_ignored() {
        let lock = Arc::new(RwLock::new(7u32));
        let l2 = lock.clone();
        let _ = std::thread::spawn(move || {
            let _g = l2.write();
            panic!("poison it");
        })
        .join();
        // parking_lot semantics: still usable, no `Result`.
        assert_eq!(*lock.read(), 7);
        *lock.write() = 8;
        assert_eq!(*lock.read(), 8);
    }

    // `dyn` unsizing coercion must work exactly like `parking_lot`, so the
    // resource store can keep `Arc<RwLock<dyn Any>>`.
    #[test]
    fn rwlock_dyn_coercion() {
        use std::any::Any;
        let concrete: Arc<RwLock<u32>> = Arc::new(RwLock::new(3));
        let erased: Arc<RwLock<dyn Any + Send + Sync>> = concrete;
        assert_eq!(*erased.read().downcast_ref::<u32>().expect("downcast"), 3);
    }

    #[test]
    fn mutex_lock_and_poison_ignored() {
        let m = Arc::new(Mutex::new(0u32));
        *m.lock() = 5;
        assert_eq!(*m.lock(), 5);

        let m2 = m.clone();
        let _ = std::thread::spawn(move || {
            let _g = m2.lock();
            panic!("poison it");
        })
        .join();
        assert_eq!(*m.lock(), 5);
    }

    #[test]
    fn mutex_try_lock() {
        let m = Mutex::new(0u32);
        let g = m.lock();
        assert!(m.try_lock().is_none());
        drop(g);
        assert!(m.try_lock().is_some());
    }
}
