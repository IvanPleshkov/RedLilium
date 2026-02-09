use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicI32, Ordering};

/// A single type-erased resource with runtime borrow checking.
struct ResourceEntry {
    value: Box<dyn Any + Send + Sync>,
    /// Borrow state: 0 = free, positive = N shared readers, -1 = exclusive writer.
    borrow_state: AtomicI32,
    type_name: &'static str,
}

/// Container for typed singleton resources.
///
/// Resources are global values stored once per World. They use the same
/// runtime borrow checking as component storages: multiple shared borrows
/// are allowed, but exclusive borrows require no other active borrows.
/// Thread-safe via atomic borrow tracking.
pub(crate) struct Resources {
    entries: HashMap<TypeId, ResourceEntry>,
}

impl Resources {
    /// Creates a new empty resource container.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Inserts or replaces a resource of type T.
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) {
        self.entries.insert(
            TypeId::of::<T>(),
            ResourceEntry {
                value: Box::new(value),
                borrow_state: AtomicI32::new(0),
                type_name: type_name::<T>(),
            },
        );
    }

    /// Removes a resource of type T, returning it if present.
    pub fn remove<T: 'static>(&mut self) -> Option<T> {
        let entry = self.entries.remove(&TypeId::of::<T>())?;
        Some(*entry.value.downcast::<T>().unwrap())
    }

    /// Returns whether a resource of type T exists.
    pub fn contains<T: 'static>(&self) -> bool {
        self.entries.contains_key(&TypeId::of::<T>())
    }

    /// Borrows a resource of type T immutably.
    ///
    /// Panics if the resource does not exist or is exclusively borrowed.
    pub fn borrow<T: 'static>(&self) -> ResourceRef<'_, T> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        let prev = entry.borrow_state.fetch_add(1, Ordering::Acquire);
        if prev < 0 {
            entry.borrow_state.fetch_sub(1, Ordering::Release);
            panic!(
                "Cannot borrow resource `{}` immutably: already borrowed mutably",
                entry.type_name
            );
        }

        ResourceRef {
            value: entry.value.downcast_ref::<T>().unwrap(),
            entry,
        }
    }

    /// Borrows a resource of type T mutably.
    ///
    /// Panics if the resource does not exist or any borrow is active.
    pub fn borrow_mut<T: 'static>(&self) -> ResourceRefMut<'_, T> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        match entry
            .borrow_state
            .compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(_) => {}
            Err(state) => {
                if state > 0 {
                    panic!(
                        "Cannot borrow resource `{}` mutably: already borrowed immutably ({} readers)",
                        entry.type_name, state
                    );
                } else {
                    panic!(
                        "Cannot borrow resource `{}` mutably: already borrowed mutably",
                        entry.type_name
                    );
                }
            }
        }

        // SAFETY: borrow_state tracking guarantees exclusive access.
        let value = entry.value.downcast_ref::<T>().unwrap() as *const T as *mut T;

        ResourceRefMut {
            value,
            entry,
            _marker: PhantomData,
        }
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared borrow of a resource.
///
/// Automatically releases the shared borrow when dropped.
pub struct ResourceRef<'a, T: 'static> {
    value: &'a T,
    entry: &'a ResourceEntry,
}

impl<T: 'static> Deref for ResourceRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

impl<T: 'static> Drop for ResourceRef<'_, T> {
    fn drop(&mut self) {
        let prev = self.entry.borrow_state.fetch_sub(1, Ordering::Release);
        debug_assert!(prev > 0);
    }
}

// SAFETY: ResourceRef only provides shared access. Atomic borrow tracking
// ensures no exclusive access exists.
unsafe impl<T: Send + Sync + 'static> Send for ResourceRef<'_, T> {}
unsafe impl<T: Send + Sync + 'static> Sync for ResourceRef<'_, T> {}

/// Exclusive borrow of a resource.
///
/// Automatically releases the exclusive borrow when dropped.
pub struct ResourceRefMut<'a, T: 'static> {
    value: *mut T,
    entry: &'a ResourceEntry,
    _marker: PhantomData<&'a mut T>,
}

impl<T: 'static> Deref for ResourceRefMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Exclusive access guaranteed by borrow tracking.
        unsafe { &*self.value }
    }
}

impl<T: 'static> DerefMut for ResourceRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Exclusive access guaranteed by borrow tracking.
        unsafe { &mut *self.value }
    }
}

impl<T: 'static> Drop for ResourceRefMut<'_, T> {
    fn drop(&mut self) {
        let prev = self.entry.borrow_state.swap(0, Ordering::Release);
        debug_assert_eq!(prev, -1);
    }
}

// SAFETY: ResourceRefMut has exclusive access. Atomic borrow tracking
// ensures no other access exists.
unsafe impl<T: Send + Sync + 'static> Send for ResourceRefMut<'_, T> {}
unsafe impl<T: Send + Sync + 'static> Sync for ResourceRefMut<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_borrow() {
        let mut resources = Resources::new();
        resources.insert(42u32);
        let val = resources.borrow::<u32>();
        assert_eq!(*val, 42);
    }

    #[test]
    fn borrow_mut_and_modify() {
        let mut resources = Resources::new();
        resources.insert(42u32);
        {
            let mut val = resources.borrow_mut::<u32>();
            *val = 99;
        }
        let val = resources.borrow::<u32>();
        assert_eq!(*val, 99);
    }

    #[test]
    fn replace_resource() {
        let mut resources = Resources::new();
        resources.insert(42u32);
        resources.insert(99u32);
        let val = resources.borrow::<u32>();
        assert_eq!(*val, 99);
    }

    #[test]
    fn remove_resource() {
        let mut resources = Resources::new();
        resources.insert(42u32);
        assert_eq!(resources.remove::<u32>(), Some(42));
        assert!(!resources.contains::<u32>());
    }

    #[test]
    fn shared_borrows_coexist() {
        let mut resources = Resources::new();
        resources.insert(42u32);
        let _a = resources.borrow::<u32>();
        let _b = resources.borrow::<u32>();
        // Both borrows succeed simultaneously
    }

    #[test]
    #[should_panic(expected = "Cannot borrow resource `u32` mutably: already borrowed immutably")]
    fn exclusive_conflicts_shared() {
        let mut resources = Resources::new();
        resources.insert(42u32);
        let _a = resources.borrow::<u32>();
        let _b = resources.borrow_mut::<u32>(); // Should panic
    }

    #[test]
    #[should_panic(expected = "Cannot borrow resource `u32` immutably: already borrowed mutably")]
    fn shared_conflicts_exclusive() {
        let mut resources = Resources::new();
        resources.insert(42u32);
        let _a = resources.borrow_mut::<u32>();
        let _b = resources.borrow::<u32>(); // Should panic
    }

    #[test]
    #[should_panic(expected = "does not exist")]
    fn missing_resource_panics() {
        let resources = Resources::new();
        let _val = resources.borrow::<u32>();
    }
}
