use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::main_thread_resource::MainThreadResources;

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

/// Marker for a resource that participates in whole-world snapshots.
///
/// Snapshots (Play mode, scene save/load, warm-restart reload) move data
/// across the [`serialize_world`](crate::World::serialize_world) /
/// [`deserialize_world_into`](crate::World::deserialize_world_into) boundary.
/// Only resources that opt in — by implementing this trait and calling
/// [`register_snapshot_resource`](crate::World::register_snapshot_resource) —
/// are captured and restored. Host/GPU managers (the editor's `EngineContext`)
/// and ephemeral resources (`RenderSchedule`, `Time`, `CommandBuffer`,
/// `Events`, editor UI) deliberately do **not** implement it: they are
/// re-injected or rebuilt by world setup rather than carried in the snapshot.
pub trait SnapshotResource: Resource + serde::Serialize + serde::de::DeserializeOwned {
    /// Stable type name used to match a captured resource to its restore hook.
    const NAME: &'static str;

    /// The schema version of this resource type.
    ///
    /// Incremented when the resource's serialized fields change, allowing
    /// migrations to transform old schema data to new schema during snapshot
    /// restore. Defaults to 1.
    const SCHEMA_VERSION: u32 = 1;
}

/// Trait for resources that respond to Play/Pause/Resume/Stop lifecycle events.
///
/// Resources like physics engines, audio systems, or GPU managers need to
/// react to gameplay state changes. Implement this trait to hook into the
/// Play/Pause/Resume/Stop cycle (e.g. pause physics, mute audio, stall GPU
/// uploads during pauses).
///
/// The default implementation (empty methods) is provided for resources that
/// don't need any lifecycle hooks.
///
/// **Design note:** This trait is separate from [`SnapshotResource`]. A
/// `PlayModeAware` resource can also be a `SnapshotResource`, but they're
/// independent concerns. Snapshot determines whether a resource is captured
/// and restored; PlayModeAware determines how it responds to state changes.
pub trait PlayModeAware: Resource {
    /// Called when Play mode starts. Use this to initialize playback state
    /// (e.g., zero out physics accumulators, reset audio positions).
    fn on_play_start(&mut self) {}

    /// Called when Play mode pauses. Use this to save state or halt activity
    /// that should not accumulate during the pause (e.g., pause physics
    /// simulation, stop audio, freeze GPU uploads).
    fn on_pause(&mut self) {}

    /// Called when Play mode resumes from a pause. Use this to resume activity
    /// (e.g., un-pause physics simulation, resume audio playback).
    fn on_resume(&mut self) {}

    /// Called when Play mode stops. Use this to clean up anything that should
    /// not persist after Play ends (e.g., clear transient GPU buffers,
    /// reset audio state, despawn dynamic entities spawned during play).
    fn on_stop(&mut self) {}
}

/// A single type-erased resource entry.
struct ResourceEntry {
    /// The resource value, stored as `Arc<RwLock<dyn Resource>>`.
    handle: Arc<RwLock<dyn Resource>>,
    type_name: &'static str,
    /// The origin ([`SourceId`](crate::SourceId)) recorded when this entry was
    /// inserted, per the ADR-020 type-identity amendment. Compared against the
    /// source `T` currently resolves to by the downcast guard (see
    /// [`World::resource`](crate::World::resource)). `HOST` in single-process
    /// builds.
    source: crate::type_identity::SourceId,
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
    main_thread: MainThreadResources,
}

impl Resources {
    /// Creates a new empty resource container.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            main_thread: MainThreadResources::new(),
        }
    }

    /// Inserts a resource, wrapping it in `Arc<RwLock<T>>`.
    ///
    /// Returns the typed `Arc` handle so the caller can keep a clone
    /// for external access (e.g. inspector). The world stores a coerced
    /// `Arc<RwLock<dyn Resource>>` that shares the same underlying data.
    pub fn insert<T: Resource>(
        &mut self,
        value: T,
        source: crate::type_identity::SourceId,
    ) -> Arc<RwLock<T>> {
        let arc = Arc::new(RwLock::new(value));
        self.entries.insert(
            TypeId::of::<T>(),
            ResourceEntry {
                handle: arc.clone(),
                type_name: type_name::<T>(),
                source,
            },
        );
        arc
    }

    /// Inserts a pre-existing `Arc<RwLock<T>>` as a resource.
    ///
    /// The Arc is coerced to `Arc<RwLock<dyn Resource>>` for storage;
    /// both the caller's clone and the stored clone share the same
    /// underlying lock and data.
    pub fn insert_shared<T: Resource>(
        &mut self,
        resource: Arc<RwLock<T>>,
        source: crate::type_identity::SourceId,
    ) {
        self.entries.insert(
            TypeId::of::<T>(),
            ResourceEntry {
                handle: resource,
                type_name: type_name::<T>(),
                source,
            },
        );
    }

    /// The source recorded for a resource entry, if present. Consulted by the
    /// downcast guard at the [`World`](crate::World) boundary.
    pub(crate) fn entry_source<T: 'static>(&self) -> Option<crate::type_identity::SourceId> {
        self.entries.get(&TypeId::of::<T>()).map(|e| e.source)
    }

    /// Removes a resource, returning the `Arc<RwLock<dyn Resource>>` if present.
    pub fn remove<T: 'static>(&mut self) -> Option<Arc<RwLock<dyn Resource>>> {
        self.entries.remove(&TypeId::of::<T>()).map(|e| e.handle)
    }

    /// Returns whether a resource of type T exists.
    pub fn contains<T: 'static>(&self) -> bool {
        self.entries.contains_key(&TypeId::of::<T>())
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

        let guard = entry.handle.try_read().unwrap_or_else(|| {
            panic!(
                "Cannot borrow resource `{}` immutably: already borrowed mutably",
                entry.type_name
            )
        });

        let ptr = guard
            .as_any()
            .downcast_ref::<T>()
            .unwrap_or_else(|| panic!("Resource `{}` type mismatch", entry.type_name))
            as *const T;

        ResourceRef {
            ptr,
            _guard: Some(guard),
            _marker: PhantomData,
        }
    }

    /// Borrows a resource of type T immutably **without** acquiring its lock.
    ///
    /// # Safety
    ///
    /// The caller must already hold a read lock on this resource (e.g. acquired
    /// up-front by `World::acquire_sorted`) for the lifetime of the returned
    /// reference, and no `&mut` to it may exist.
    pub(crate) unsafe fn borrow_unlocked<T: 'static>(&self) -> ResourceRef<'_, T> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        // SAFETY: the caller guarantees the read lock is held externally.
        let data: &dyn Resource = unsafe { &*entry.handle.data_ptr() };
        let ptr = data
            .as_any()
            .downcast_ref::<T>()
            .unwrap_or_else(|| panic!("Resource `{}` type mismatch", entry.type_name))
            as *const T;

        ResourceRef {
            ptr,
            _guard: None,
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

        let mut guard = entry.handle.try_write().unwrap_or_else(|| {
            panic!(
                "Cannot borrow resource `{}` mutably: already borrowed",
                entry.type_name
            )
        });

        let ptr = guard
            .as_any_mut()
            .downcast_mut::<T>()
            .unwrap_or_else(|| panic!("Resource `{}` type mismatch", entry.type_name))
            as *mut T;

        ResourceRefMut {
            ptr,
            _guard: Some(guard),
            _marker: PhantomData,
            borrowed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Borrows a resource of type T mutably **without** acquiring its lock.
    ///
    /// # Safety
    ///
    /// The caller must already hold a write lock on this resource (e.g. acquired
    /// up-front by `World::acquire_sorted`) for the lifetime of the returned
    /// reference, and no other reference to it may exist.
    pub(crate) unsafe fn borrow_mut_unlocked<T: 'static>(&self) -> ResourceRefMut<'_, T> {
        let entry = self
            .entries
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Resource `{}` does not exist", type_name::<T>()));

        // SAFETY: the caller guarantees the write lock is held externally.
        let data: &mut dyn Resource = unsafe { &mut *entry.handle.data_ptr() };
        let ptr = data
            .as_any_mut()
            .downcast_mut::<T>()
            .unwrap_or_else(|| panic!("Resource `{}` type mismatch", entry.type_name))
            as *mut T;

        ResourceRefMut {
            ptr,
            _guard: None,
            _marker: PhantomData,
            borrowed: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Acquires the read lock on a resource by `TypeId`, blocking until
    /// available. Returns the type-erased guard, or `None` if the resource does
    /// not exist. Used by `World::acquire_sorted` to lock resources up-front in
    /// a globally-consistent order (deadlock-free, blocking instead of the
    /// panic-on-contention `try_*` path).
    pub(crate) fn read_guard_dyn(
        &self,
        type_id: TypeId,
    ) -> Option<RwLockReadGuard<'_, dyn Resource>> {
        Some(self.entries.get(&type_id)?.handle.read())
    }

    /// Acquires the write lock on a resource by `TypeId`, blocking until
    /// available. See [`read_guard_dyn`](Self::read_guard_dyn).
    pub(crate) fn write_guard_dyn(
        &self,
        type_id: TypeId,
    ) -> Option<RwLockWriteGuard<'_, dyn Resource>> {
        Some(self.entries.get(&type_id)?.handle.write())
    }

    // ---- Main-thread resource delegation ----

    /// Inserts a main-thread resource.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread (or hold `&mut self`).
    pub unsafe fn insert_main_thread<T: 'static>(&self, value: T) {
        unsafe { self.main_thread.insert(value) }
    }

    /// Returns whether a main-thread resource of type `T` exists.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread.
    pub unsafe fn has_main_thread<T: 'static>(&self) -> bool {
        unsafe { self.main_thread.contains::<T>() }
    }

    /// Removes a main-thread resource and returns it, or `None` if absent.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread (or hold `&mut self`).
    pub unsafe fn remove_main_thread<T: 'static>(&self) -> Option<T> {
        unsafe { self.main_thread.remove::<T>() }
    }

    /// Borrows a main-thread resource immutably.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread.
    pub unsafe fn borrow_main_thread<T: 'static>(&self) -> &T {
        unsafe { self.main_thread.borrow::<T>() }
    }

    /// Borrows a main-thread resource mutably.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread. No other borrows to this resource
    /// may be active.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn borrow_main_thread_mut<T: 'static>(&self) -> &mut T {
        unsafe { self.main_thread.borrow_mut::<T>() }
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared borrow of a resource.
///
/// Downcasts to `&T` via [`Deref`]. Either holds its own `RwLockReadGuard`
/// (standalone [`Resources::borrow`]) or borrows data whose read lock is held
/// externally — e.g. acquired up-front in a globally-sorted order by
/// `World::acquire_sorted` (see [`Resources::borrow_unlocked`]).
pub struct ResourceRef<'a, T: 'static> {
    /// Points into the locked resource data; valid for `'a` because either
    /// `_guard` is held or the caller holds the read lock externally.
    ptr: *const T,
    _guard: Option<RwLockReadGuard<'a, dyn Resource>>,
    _marker: PhantomData<&'a T>,
}

// SAFETY (Sync): `ResourceRef` is a read-only view, equivalent to `&T`,
// which is `Sync` exactly when `T: Sync`.
//
// Deliberately NOT Send: the held read guard (when present)
// must be released on the thread that acquired it (lock_api contract;
// also incompatible with deadlock detection). Guards are created, used
// and dropped on the same thread (audit A7).
unsafe impl<T: Sync> Sync for ResourceRef<'_, T> {}

impl<T: 'static> Deref for ResourceRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: `ptr` points into resource data kept alive by the held guard
        // or by an externally-held read lock for `'a`.
        unsafe { &*self.ptr }
    }
}

/// Exclusive borrow of a resource.
///
/// Downcasts to `&mut T` via [`DerefMut`]. Either holds its own
/// `RwLockWriteGuard` (standalone [`Resources::borrow_mut`]) or borrows data
/// whose write lock is held externally (see [`Resources::borrow_mut_unlocked`]).
///
/// Contains a runtime borrow flag used by [`QueryItem`](crate::QueryItem)
/// iteration to detect aliasing `&mut T` references (similar to `RefCell`).
/// The flag lives behind an `Arc` so a [`ResMutRef`](crate::query::ResMutRef)
/// holding it stays valid even if this struct is moved or dropped first.
/// The raw `ptr` keeps this type `!Sync`, which intentionally excludes
/// `ResMut` from `par_for_each`.
pub struct ResourceRefMut<'a, T: 'static> {
    ptr: *mut T,
    _guard: Option<RwLockWriteGuard<'a, dyn Resource>>,
    _marker: PhantomData<&'a mut T>,
    /// Runtime borrow tracking for iterator safety.
    /// `true` while a [`ResMutRef`](crate::query::ResMutRef) derived
    /// from this resource is alive.
    pub(crate) borrowed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

// Deliberately neither Send nor Sync: the raw `ptr` keeps it `!Sync`
// (aliasing `&mut T` across threads), and the held write guard
// must be released on the thread that acquired it, so no `Send` either
// (audit A7). Fully thread-confined.

impl<T: 'static> ResourceRefMut<'_, T> {
    /// Returns a raw mutable pointer to the underlying resource data.
    pub(crate) fn as_ptr_mut(&self) -> *mut T {
        self.ptr
    }
}

impl<T: 'static> Deref for ResourceRefMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: exclusive access held via the write guard or externally.
        unsafe { &*self.ptr }
    }
}

impl<T: 'static> DerefMut for ResourceRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: exclusive access held via the write guard or externally.
        unsafe { &mut *self.ptr }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_identity::SourceId;

    const HOST: SourceId = SourceId::HOST;

    #[test]
    fn insert_and_borrow() {
        let mut resources = Resources::new();
        resources.insert(42u32, HOST);
        let val = resources.borrow::<u32>();
        assert_eq!(*val, 42);
    }

    #[test]
    fn borrow_mut_and_modify() {
        let mut resources = Resources::new();
        resources.insert(42u32, HOST);
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
        resources.insert(42u32, HOST);
        resources.insert(99u32, HOST);
        let val = resources.borrow::<u32>();
        assert_eq!(*val, 99);
    }

    #[test]
    fn remove_resource() {
        let mut resources = Resources::new();
        resources.insert(42u32, HOST);
        let removed = resources.remove::<u32>();
        assert!(removed.is_some());
        assert!(!resources.contains::<u32>());
    }

    #[test]
    fn shared_borrows_coexist() {
        let mut resources = Resources::new();
        resources.insert(42u32, HOST);
        let _a = resources.borrow::<u32>();
        let _b = resources.borrow::<u32>();
        // Both borrows succeed simultaneously
    }

    #[test]
    #[should_panic(expected = "Cannot borrow resource `u32` mutably: already borrowed")]
    fn exclusive_conflicts_shared() {
        let mut resources = Resources::new();
        resources.insert(42u32, HOST);
        let _a = resources.borrow::<u32>();
        let _b = resources.borrow_mut::<u32>(); // Should panic
    }

    #[test]
    #[should_panic(expected = "Cannot borrow resource `u32` immutably: already borrowed mutably")]
    fn shared_conflicts_exclusive() {
        let mut resources = Resources::new();
        resources.insert(42u32, HOST);
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
        let handle = resources.insert(42u32, HOST);

        // External access via typed handle
        assert_eq!(*handle.read(), 42);

        // Modify through typed handle
        *handle.write() = 99;

        // World sees the change (same underlying Arc)
        let val = resources.borrow::<u32>();
        assert_eq!(*val, 99);
    }

    #[test]
    fn insert_shared_same_arc() {
        let mut resources = Resources::new();
        let handle = Arc::new(RwLock::new(42u32));
        resources.insert_shared(handle.clone(), HOST);

        // Modify through external handle
        *handle.write() = 99;

        // World sees the change
        let val = resources.borrow::<u32>();
        assert_eq!(*val, 99);
    }

    #[test]
    fn get_handle_returns_dyn_resource() {
        let mut resources = Resources::new();
        resources.insert(42u32, HOST);

        let dyn_handle = resources.get_handle::<u32>();
        let guard = dyn_handle.read();
        assert_eq!(*guard.as_any().downcast_ref::<u32>().unwrap(), 42);
    }
}
