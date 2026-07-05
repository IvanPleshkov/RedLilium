use std::any::{Any, TypeId, type_name}; // type_name used in panic messages
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::thread::ThreadId;

/// A single main-thread resource entry.
struct MainThreadEntry {
    value: Box<dyn Any>,
}

/// Container for main-thread-only resources.
///
/// Unlike [`Resources`](crate::resource::Resources), main-thread resources
/// do **not** require `Send + Sync`. They are stored in `UnsafeCell` and may
/// only be accessed from the main thread.
///
/// # Safety invariant
///
/// The stored values are not `Send`: every touch of the container —
/// insert, borrow, remove and the final **drop** — must happen on one
/// thread, the one that first inserted a resource (normally the main
/// thread). The [`MainThreadDispatcher`](crate::main_thread_dispatcher::MainThreadDispatcher)
/// in the runner guarantees this for system access by routing closures
/// that touch these resources to the main thread; the world owner must
/// uphold it for setup and for dropping the `World` itself.
///
/// The owning thread is recorded on first insert and every access (and
/// the drop) is checked against it in debug builds.
pub(crate) struct MainThreadResources {
    entries: UnsafeCell<HashMap<TypeId, MainThreadEntry>>,
    /// Thread that first inserted a resource. Every later touch of the
    /// container must happen on it (debug-checked).
    owner: OnceLock<ThreadId>,
}

// SAFETY: MainThreadResources is stored in World (which must be Send + Sync).
// The UnsafeCell contents are only accessed from the owning thread (see the
// safety invariant above) via dispatched closures or during setup; that
// confinement — not these impls — is what makes the non-Send contents sound.
// Send additionally permits moving/dropping the container on another thread,
// which is only allowed while it is empty (debug-asserted in Drop).
unsafe impl Send for MainThreadResources {}
unsafe impl Sync for MainThreadResources {}

impl MainThreadResources {
    pub fn new() -> Self {
        Self {
            entries: UnsafeCell::new(HashMap::new()),
            owner: OnceLock::new(),
        }
    }

    /// Debug-checks that the current thread is the owning thread (the one
    /// that first inserted a resource). No-op while nothing was inserted.
    #[track_caller]
    fn debug_assert_owner(&self) {
        if cfg!(debug_assertions)
            && let Some(owner) = self.owner.get()
        {
            let current = std::thread::current().id();
            assert_eq!(
                *owner, current,
                "main-thread resources are confined to the thread that first \
                 inserted one ({owner:?}); this access is on {current:?}"
            );
        }
    }

    /// Inserts a main-thread resource, replacing any previous value of the same type.
    ///
    /// # Safety
    ///
    /// Caller must be on the owning thread (see type-level invariant), and
    /// no borrows returned by [`borrow`](Self::borrow) /
    /// [`borrow_mut`](Self::borrow_mut) may be alive: replacing a value
    /// drops the old box that outstanding borrows point into.
    pub unsafe fn insert<T: 'static>(&self, value: T) {
        let _ = self.owner.get_or_init(|| std::thread::current().id());
        self.debug_assert_owner();
        let entries = unsafe { &mut *self.entries.get() };
        entries.insert(
            TypeId::of::<T>(),
            MainThreadEntry {
                value: Box::new(value),
            },
        );
    }

    /// Borrows a main-thread resource immutably.
    ///
    /// # Safety
    ///
    /// Caller must be on the owning thread.
    ///
    /// # Panics
    ///
    /// Panics if the resource has not been inserted.
    pub unsafe fn borrow<T: 'static>(&self) -> &T {
        self.debug_assert_owner();
        let entries = unsafe { &*self.entries.get() };
        let entry = entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Main-thread resource `{}` not found", type_name::<T>()));
        entry.value.downcast_ref::<T>().unwrap()
    }

    /// Borrows a main-thread resource mutably.
    ///
    /// # Safety
    ///
    /// Caller must be on the owning thread. No other borrows to this
    /// resource may be active.
    ///
    /// # Panics
    ///
    /// Panics if the resource has not been inserted.
    #[allow(clippy::mut_from_ref)] // SAFETY: caller ensures exclusive main-thread access
    pub unsafe fn borrow_mut<T: 'static>(&self) -> &mut T {
        self.debug_assert_owner();
        let entries = unsafe { &mut *self.entries.get() };
        let entry = entries
            .get_mut(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Main-thread resource `{}` not found", type_name::<T>()));
        entry.value.downcast_mut::<T>().unwrap()
    }

    /// Returns `true` if a resource of type `T` has been inserted.
    ///
    /// # Safety
    ///
    /// Caller must be on the owning thread (the map itself is not
    /// thread-safe, even for key lookups).
    pub unsafe fn contains<T: 'static>(&self) -> bool {
        self.debug_assert_owner();
        let entries = unsafe { &*self.entries.get() };
        entries.contains_key(&TypeId::of::<T>())
    }

    /// Removes a main-thread resource and returns it, or `None` if absent.
    ///
    /// # Safety
    ///
    /// Caller must be on the owning thread, and no borrows returned by
    /// [`borrow`](Self::borrow) / [`borrow_mut`](Self::borrow_mut) may be
    /// alive (removal drops the box that outstanding borrows point into).
    pub unsafe fn remove<T: 'static>(&self) -> Option<T> {
        self.debug_assert_owner();
        let entries = unsafe { &mut *self.entries.get() };
        entries
            .remove(&TypeId::of::<T>())
            .map(|entry| *entry.value.downcast::<T>().unwrap())
    }
}

impl Default for MainThreadResources {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MainThreadResources {
    fn drop(&mut self) {
        // Dropping non-Send values on a foreign thread runs their
        // destructors there (wrong-thread GL/UI teardown, racy Rc counts).
        // An empty container is fine to drop anywhere.
        if cfg!(debug_assertions)
            && !self.entries.get_mut().is_empty()
            && let Some(owner) = self.owner.get()
        {
            let current = std::thread::current().id();
            assert_eq!(
                *owner, current,
                "a World holding main-thread resources must be dropped on the \
                 thread that owns them ({owner:?}); this drop is on {current:?}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Non-Send values must not be dropped on a foreign thread; the debug
    /// assert in Drop catches a World/container moved elsewhere and dropped.
    #[test]
    #[cfg_attr(not(debug_assertions), ignore = "contract is debug-checked only")]
    fn drop_with_resources_on_other_thread_panics() {
        let storage = MainThreadResources::new();
        unsafe { storage.insert(42u32) };

        let handle = std::thread::spawn(move || {
            drop(storage); // wrong thread: destructors would run here
        });
        assert!(handle.join().is_err());
    }

    /// An empty container may be dropped anywhere (nothing non-Send inside).
    #[test]
    fn drop_empty_on_other_thread_is_fine() {
        let storage = MainThreadResources::new();
        let handle = std::thread::spawn(move || drop(storage));
        assert!(handle.join().is_ok());
    }

    /// Access from a thread other than the owning one is debug-rejected.
    #[test]
    #[cfg_attr(not(debug_assertions), ignore = "contract is debug-checked only")]
    fn borrow_on_other_thread_panics() {
        let storage = std::sync::Arc::new(MainThreadResources::new());
        unsafe { storage.insert(42u32) };

        let remote = std::sync::Arc::clone(&storage);
        let handle = std::thread::spawn(move || {
            let _ = unsafe { remote.borrow::<u32>() };
        });
        assert!(handle.join().is_err());
    }

    /// The owner is pinned by the FIRST insert, wherever it happens.
    #[test]
    #[cfg_attr(not(debug_assertions), ignore = "contract is debug-checked only")]
    fn owner_is_first_insert_thread() {
        let storage = std::sync::Arc::new(MainThreadResources::new());

        let remote = std::sync::Arc::clone(&storage);
        let handle = std::thread::spawn(move || {
            unsafe { remote.insert(1u32) };
            assert_eq!(unsafe { *remote.borrow::<u32>() }, 1);
        });
        handle.join().unwrap();

        // This thread is now foreign: inserting here must trip the check.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            storage.insert(2u32);
        }));
        assert!(result.is_err());
        // Silence the Drop assert: this thread is not the owner. Leak the
        // Arc instead of dropping non-empty storage on a foreign thread.
        std::mem::forget(storage);
    }

    #[test]
    fn insert_and_borrow() {
        let storage = MainThreadResources::new();
        unsafe {
            storage.insert(42u32);
            assert_eq!(*storage.borrow::<u32>(), 42);
        }
    }

    #[test]
    fn insert_and_borrow_mut() {
        let storage = MainThreadResources::new();
        unsafe {
            storage.insert(String::from("hello"));
            let s = storage.borrow_mut::<String>();
            s.push_str(" world");
            assert_eq!(*storage.borrow::<String>(), "hello world");
        }
    }

    #[test]
    fn contains_check() {
        let storage = MainThreadResources::new();
        unsafe {
            assert!(!storage.contains::<u32>());
            storage.insert(10u32);
            assert!(storage.contains::<u32>());
        }
    }

    #[test]
    fn remove_resource() {
        let storage = MainThreadResources::new();
        unsafe {
            storage.insert(99u32);
            let removed = storage.remove::<u32>();
            assert_eq!(removed, Some(99));
            assert!(!storage.contains::<u32>());
        }
    }

    #[test]
    fn remove_absent_returns_none() {
        let storage = MainThreadResources::new();
        unsafe {
            assert_eq!(storage.remove::<u32>(), None);
        }
    }

    #[test]
    fn replace_existing() {
        let storage = MainThreadResources::new();
        unsafe {
            storage.insert(1u32);
            storage.insert(2u32);
            assert_eq!(*storage.borrow::<u32>(), 2);
        }
    }

    #[test]
    #[should_panic(expected = "not found")]
    fn borrow_missing_panics() {
        let storage = MainThreadResources::new();
        unsafe {
            storage.borrow::<u32>();
        }
    }

    #[test]
    fn non_send_type() {
        use std::cell::Cell;
        use std::rc::Rc;

        let storage = MainThreadResources::new();
        unsafe {
            let counter = Rc::new(Cell::new(0u32));
            storage.insert(counter.clone());
            let borrowed = storage.borrow::<Rc<Cell<u32>>>();
            borrowed.set(42);
            assert_eq!(counter.get(), 42);
        }
    }
}
