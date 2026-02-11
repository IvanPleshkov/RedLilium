use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::sparse_set::LockGuard;

/// A single type-erased resource with per-resource RwLock synchronization.
struct ResourceEntry {
    value: Box<dyn Any + Send + Sync>,
    /// Per-resource lock for thread-safe borrow management.
    lock: RwLock<()>,
    type_name: &'static str,
}

/// Container for typed singleton resources.
///
/// Resources are global values stored once per World. They use per-resource
/// RwLock synchronization: multiple shared borrows are allowed, but exclusive
/// borrows require no other active borrows. Thread-safe via RwLock.
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
                lock: RwLock::new(()),
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

    /// Returns the TypeIds of all registered resource types.
    pub fn type_ids(&self) -> impl Iterator<Item = TypeId> + '_ {
        self.entries.keys().copied()
    }

    /// Borrows a resource of type T immutably.
    ///
    /// Panics if the resource does not exist or is exclusively borrowed.
    pub fn borrow<T: 'static>(&self) -> ResourceRef<'_, T> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        let guard = entry.lock.try_read().unwrap_or_else(|_| {
            panic!(
                "Cannot borrow resource `{}` immutably: already borrowed mutably",
                entry.type_name
            )
        });

        ResourceRef {
            value: entry.value.downcast_ref::<T>().unwrap(),
            _guard: Some(guard),
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

        let guard = entry.lock.try_write().unwrap_or_else(|_| {
            panic!(
                "Cannot borrow resource `{}` mutably: already borrowed",
                entry.type_name
            )
        });

        // SAFETY: write lock guarantees exclusive access.
        let value = entry.value.downcast_ref::<T>().unwrap() as *const T as *mut T;

        ResourceRefMut {
            value,
            _guard: Some(guard),
            _marker: PhantomData,
        }
    }

    /// Borrows a resource immutably without acquiring a lock.
    ///
    /// The caller must ensure the lock is already held externally.
    pub(crate) fn borrow_unlocked<T: 'static>(&self) -> ResourceRef<'_, T> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        ResourceRef {
            value: entry.value.downcast_ref::<T>().unwrap(),
            _guard: None,
        }
    }

    /// Borrows a resource mutably without acquiring a lock.
    ///
    /// The caller must ensure the write lock is already held externally.
    pub(crate) fn borrow_mut_unlocked<T: 'static>(&self) -> ResourceRefMut<'_, T> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        let value = entry.value.downcast_ref::<T>().unwrap() as *const T as *mut T;

        ResourceRefMut {
            value,
            _guard: None,
            _marker: PhantomData,
        }
    }

    /// Acquires a lock on a resource by TypeId for sorted lock acquisition.
    ///
    /// Returns `None` if the TypeId does not correspond to any resource.
    pub(crate) fn acquire_lock(&self, type_id: TypeId, is_write: bool) -> Option<LockGuard<'_>> {
        let entry = self.entries.get(&type_id)?;
        if is_write {
            Some(LockGuard::Write(entry.lock.write().unwrap()))
        } else {
            Some(LockGuard::Read(entry.lock.read().unwrap()))
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
/// Automatically releases the lock when dropped.
pub struct ResourceRef<'a, T: 'static> {
    value: &'a T,
    _guard: Option<RwLockReadGuard<'a, ()>>,
}

impl<T: 'static> Deref for ResourceRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

// ResourceRef holds either an RwLockReadGuard (auto-released on drop) or nothing.
// No manual Drop needed.

// SAFETY: ResourceRef only provides shared access. The RwLock ensures
// no exclusive access exists when the guard is held.
unsafe impl<T: Send + Sync + 'static> Send for ResourceRef<'_, T> {}
unsafe impl<T: Send + Sync + 'static> Sync for ResourceRef<'_, T> {}

/// Exclusive borrow of a resource.
///
/// Automatically releases the lock when dropped.
pub struct ResourceRefMut<'a, T: 'static> {
    value: *mut T,
    _guard: Option<RwLockWriteGuard<'a, ()>>,
    _marker: PhantomData<&'a mut T>,
}

impl<T: 'static> Deref for ResourceRefMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Exclusive access guaranteed by the write lock.
        unsafe { &*self.value }
    }
}

impl<T: 'static> DerefMut for ResourceRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Exclusive access guaranteed by the write lock.
        unsafe { &mut *self.value }
    }
}

// ResourceRefMut holds either an RwLockWriteGuard (auto-released on drop) or nothing.
// No manual Drop needed.

// SAFETY: ResourceRefMut has exclusive access. The RwLock ensures
// no other access exists when the guard is held.
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
    #[should_panic(expected = "Cannot borrow resource `u32` mutably: already borrowed")]
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
