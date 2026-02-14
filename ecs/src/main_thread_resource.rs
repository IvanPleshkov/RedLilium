use std::any::{Any, TypeId, type_name}; // type_name used in panic messages
use std::cell::UnsafeCell;
use std::collections::HashMap;

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
/// All methods that access `entries` must only be called from the main thread.
/// The [`MainThreadDispatcher`](crate::main_thread_dispatcher::MainThreadDispatcher)
/// in the runner guarantees this by routing closures that touch these resources
/// to the main thread.
pub(crate) struct MainThreadResources {
    entries: UnsafeCell<HashMap<TypeId, MainThreadEntry>>,
}

// SAFETY: MainThreadResources is stored in World (which must be Send + Sync).
// The UnsafeCell contents are only accessed from the main thread via dispatched
// closures or during setup (when &mut World guarantees exclusive access).
unsafe impl Send for MainThreadResources {}
unsafe impl Sync for MainThreadResources {}

impl MainThreadResources {
    pub fn new() -> Self {
        Self {
            entries: UnsafeCell::new(HashMap::new()),
        }
    }

    /// Inserts a main-thread resource, replacing any previous value of the same type.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread (or hold `&mut self`).
    pub unsafe fn insert<T: 'static>(&self, value: T) {
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
    /// Caller must be on the main thread.
    ///
    /// # Panics
    ///
    /// Panics if the resource has not been inserted.
    pub unsafe fn borrow<T: 'static>(&self) -> &T {
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
    /// Caller must be on the main thread. No other borrows to this resource
    /// may be active.
    ///
    /// # Panics
    ///
    /// Panics if the resource has not been inserted.
    #[allow(clippy::mut_from_ref)] // SAFETY: caller ensures exclusive main-thread access
    pub unsafe fn borrow_mut<T: 'static>(&self) -> &mut T {
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
    /// Caller must be on the main thread.
    pub unsafe fn contains<T: 'static>(&self) -> bool {
        let entries = unsafe { &*self.entries.get() };
        entries.contains_key(&TypeId::of::<T>())
    }

    /// Removes a main-thread resource and returns it, or `None` if absent.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread (or hold `&mut self`).
    pub unsafe fn remove<T: 'static>(&self) -> Option<T> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
