use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Trait that all resources must implement.
///
/// Provides type-erased downcasting via [`Any`]. A blanket implementation
/// covers all `Send + Sync + 'static` types automatically, so most users
/// never need to implement this manually.
///
/// The World stores resources as `Arc<RwLock<dyn Resource>>`. External code
/// (e.g. inspector, editor) can hold `Arc<RwLock<T>>` clones returned by
/// [`Resources::insert`] for direct access outside the ECS runner.
pub trait Resource: Send + Sync + 'static {
    /// Returns a shared reference for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns an exclusive reference for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Send + Sync + 'static> Resource for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A single type-erased resource entry.
struct ResourceEntry {
    /// The resource value, stored as `Arc<RwLock<dyn Resource>>`.
    handle: Arc<RwLock<dyn Resource>>,
    type_name: &'static str,
}

/// Container for typed singleton resources backed by `Arc<RwLock<dyn Resource>>`.
///
/// Resources are global values stored once per World. Each resource is
/// wrapped in `Arc<RwLock<dyn Resource>>`, enabling:
///
/// - **External access**: hold a typed `Arc<RwLock<T>>` clone (from
///   [`insert`](Resources::insert)) to read/write the resource outside
///   the ECS runner (e.g. inspector, editor UI).
/// - **Dynamic dispatch**: all resources share a common `dyn Resource`
///   interface with downcasting support.
///
/// Locking is done through the `Arc`'s own `RwLock` — no separate
/// scheduler-level lock is needed.
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

    /// Inserts a resource, wrapping it in `Arc<RwLock<T>>`.
    ///
    /// Returns the typed `Arc` handle so the caller can keep a clone
    /// for external access (e.g. inspector). The world stores a coerced
    /// `Arc<RwLock<dyn Resource>>` that shares the same underlying data.
    pub fn insert<T: Resource>(&mut self, value: T) -> Arc<RwLock<T>> {
        let arc = Arc::new(RwLock::new(value));
        self.entries.insert(
            TypeId::of::<T>(),
            ResourceEntry {
                handle: arc.clone(),
                type_name: type_name::<T>(),
            },
        );
        arc
    }

    /// Inserts a pre-existing `Arc<RwLock<T>>` as a resource.
    ///
    /// The Arc is coerced to `Arc<RwLock<dyn Resource>>` for storage;
    /// both the caller's clone and the stored clone share the same
    /// underlying lock and data.
    pub fn insert_shared<T: Resource>(&mut self, resource: Arc<RwLock<T>>) {
        self.entries.insert(
            TypeId::of::<T>(),
            ResourceEntry {
                handle: resource,
                type_name: type_name::<T>(),
            },
        );
    }

    /// Removes a resource, returning the `Arc<RwLock<dyn Resource>>` if present.
    pub fn remove<T: 'static>(&mut self) -> Option<Arc<RwLock<dyn Resource>>> {
        self.entries.remove(&TypeId::of::<T>()).map(|e| e.handle)
    }

    /// Returns whether a resource of type T exists.
    pub fn contains<T: 'static>(&self) -> bool {
        self.entries.contains_key(&TypeId::of::<T>())
    }

    /// Returns the TypeIds of all registered resource types.
    pub fn type_ids(&self) -> impl Iterator<Item = TypeId> + '_ {
        self.entries.keys().copied()
    }

    /// Returns the `Arc<RwLock<dyn Resource>>` handle for a resource.
    ///
    /// Panics if the resource does not exist.
    pub fn get_handle<T: 'static>(&self) -> Arc<RwLock<dyn Resource>> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        entry.handle.clone()
    }

    /// Borrows a resource of type T immutably.
    ///
    /// Acquires the `RwLock` read lock on the stored `Arc<RwLock<dyn Resource>>`.
    /// The returned guard downcasts to `&T` via [`Deref`].
    ///
    /// Panics if the resource does not exist or is exclusively borrowed.
    pub fn borrow<T: 'static>(&self) -> ResourceRef<'_, T> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        let guard = entry.handle.try_read().unwrap_or_else(|_| {
            panic!(
                "Cannot borrow resource `{}` immutably: already borrowed mutably",
                entry.type_name
            )
        });

        ResourceRef {
            guard,
            _marker: PhantomData,
        }
    }

    /// Borrows a resource of type T mutably.
    ///
    /// Acquires the `RwLock` write lock on the stored `Arc<RwLock<dyn Resource>>`.
    /// The returned guard downcasts to `&mut T` via [`DerefMut`].
    ///
    /// Panics if the resource does not exist or any borrow is active.
    pub fn borrow_mut<T: 'static>(&self) -> ResourceRefMut<'_, T> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        let guard = entry.handle.try_write().unwrap_or_else(|_| {
            panic!(
                "Cannot borrow resource `{}` mutably: already borrowed",
                entry.type_name
            )
        });

        ResourceRefMut {
            guard,
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
/// Holds an `RwLockReadGuard<dyn Resource>` and downcasts to `&T` in [`Deref`].
/// Automatically releases the lock when dropped.
pub struct ResourceRef<'a, T: 'static> {
    guard: RwLockReadGuard<'a, dyn Resource>,
    _marker: PhantomData<&'a T>,
}

impl<T: 'static> Deref for ResourceRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.as_any().downcast_ref::<T>().unwrap()
    }
}

/// Exclusive borrow of a resource.
///
/// Holds an `RwLockWriteGuard<dyn Resource>` and downcasts to `&mut T` in
/// [`DerefMut`]. Automatically releases the lock when dropped.
pub struct ResourceRefMut<'a, T: 'static> {
    guard: RwLockWriteGuard<'a, dyn Resource>,
    _marker: PhantomData<&'a mut T>,
}

impl<T: 'static> Deref for ResourceRefMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.as_any().downcast_ref::<T>().unwrap()
    }
}

impl<T: 'static> DerefMut for ResourceRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.as_any_mut().downcast_mut::<T>().unwrap()
    }
}

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
        let removed = resources.remove::<u32>();
        assert!(removed.is_some());
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

    #[test]
    fn insert_returns_arc_handle() {
        let mut resources = Resources::new();
        let handle = resources.insert(42u32);

        // External access via typed handle
        assert_eq!(*handle.read().unwrap(), 42);

        // Modify through typed handle
        *handle.write().unwrap() = 99;

        // World sees the change (same underlying Arc)
        let val = resources.borrow::<u32>();
        assert_eq!(*val, 99);
    }

    #[test]
    fn insert_shared_same_arc() {
        let mut resources = Resources::new();
        let handle = Arc::new(RwLock::new(42u32));
        resources.insert_shared(handle.clone());

        // Modify through external handle
        *handle.write().unwrap() = 99;

        // World sees the change
        let val = resources.borrow::<u32>();
        assert_eq!(*val, 99);
    }

    #[test]
    fn get_handle_returns_dyn_resource() {
        let mut resources = Resources::new();
        resources.insert(42u32);

        let dyn_handle = resources.get_handle::<u32>();
        let guard = dyn_handle.read().unwrap();
        assert_eq!(*guard.as_any().downcast_ref::<u32>().unwrap(), 42);
    }
}
