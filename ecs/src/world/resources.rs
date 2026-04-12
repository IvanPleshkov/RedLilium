use std::sync::Arc;

use crate::commands::CommandBuffer;
use crate::events::Events;
use crate::resource::{Resource, ResourceRef, ResourceRefMut};

use super::World;

impl World {
    // ---- Resource management ----

    /// Inserts or replaces a resource, wrapping it in `Arc<RwLock<T>>`.
    ///
    /// Returns the typed `Arc` handle for external access (e.g. inspector,
    /// editor). The world stores a coerced `Arc<RwLock<dyn Resource>>` that
    /// shares the same underlying data and lock.
    pub fn insert_resource<T: Resource>(&mut self, value: T) -> Arc<parking_lot::RwLock<T>> {
        self.resources.insert(value)
    }

    /// Inserts a pre-existing `Arc<RwLock<T>>` as a resource.
    ///
    /// The Arc is coerced to `Arc<RwLock<dyn Resource>>` for storage;
    /// both the caller's clone and the stored clone share the same lock.
    pub fn insert_resource_shared<T: Resource>(&mut self, resource: Arc<parking_lot::RwLock<T>>) {
        self.resources.insert_shared(resource);
    }

    /// Removes a resource, returning the `Arc<RwLock<dyn Resource>>` if present.
    pub fn remove_resource<T: 'static>(
        &mut self,
    ) -> Option<Arc<parking_lot::RwLock<dyn Resource>>> {
        self.resources.remove::<T>()
    }

    /// Returns whether a resource of type T exists.
    pub fn has_resource<T: 'static>(&self) -> bool {
        self.resources.contains::<T>()
    }

    /// Returns the `Arc<RwLock<dyn Resource>>` handle for a resource.
    ///
    /// For typed access, keep the `Arc<RwLock<T>>` returned by
    /// [`insert_resource`](World::insert_resource) instead.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist.
    pub fn resource_handle<T: 'static>(&self) -> Arc<parking_lot::RwLock<dyn Resource>> {
        self.resources.get_handle::<T>()
    }

    /// Borrows a resource of type T immutably.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist or is exclusively borrowed.
    pub fn resource<T: 'static>(&self) -> ResourceRef<'_, T> {
        self.resources.borrow::<T>()
    }

    /// Borrows a resource of type T mutably.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist or any borrow is active.
    pub fn resource_mut<T: 'static>(&self) -> ResourceRefMut<'_, T> {
        self.resources.borrow_mut::<T>()
    }

    // ---- Main-thread resource management ----

    /// Inserts a main-thread resource during world setup.
    ///
    /// The resource does **not** need to implement `Send` or `Sync`.
    /// Takes `&mut self`, so it can only be called before systems run
    /// (during setup on the main thread).
    pub fn insert_main_thread_resource<T: 'static>(&mut self, value: T) {
        // SAFETY: &mut self guarantees exclusive access; setup is on main thread.
        unsafe { self.resources.insert_main_thread(value) }
    }

    /// Returns whether a main-thread resource of type `T` exists.
    pub fn has_main_thread_resource<T: 'static>(&self) -> bool {
        // SAFETY: contains() only reads HashMap keys (TypeId), no data access.
        unsafe { self.resources.has_main_thread::<T>() }
    }

    /// Removes a main-thread resource and returns it, or `None` if absent.
    ///
    /// Takes `&mut self`, so it can only be called outside system execution.
    pub fn remove_main_thread_resource<T: 'static>(&mut self) -> Option<T> {
        // SAFETY: &mut self guarantees exclusive access.
        unsafe { self.resources.remove_main_thread::<T>() }
    }

    /// Borrows a main-thread resource immutably.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread.
    pub(crate) unsafe fn main_thread_resource<T: 'static>(&self) -> &T {
        unsafe { self.resources.borrow_main_thread::<T>() }
    }

    /// Borrows a main-thread resource mutably.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread. No other borrows to this resource
    /// may be active.
    #[allow(clippy::mut_from_ref)] // SAFETY: caller ensures exclusive main-thread access
    pub(crate) unsafe fn main_thread_resource_mut<T: 'static>(&self) -> &mut T {
        unsafe { self.resources.borrow_main_thread_mut::<T>() }
    }

    // ---- Change detection ----

    /// Returns the current world tick.
    ///
    /// The tick advances each frame via [`advance_tick`](World::advance_tick)
    /// and is used for change detection.
    pub fn current_tick(&self) -> u64 {
        self.tick
    }

    /// Advances the world tick by one.
    ///
    /// Call this at the start of each frame, before running systems.
    pub fn advance_tick(&mut self) {
        self.tick += 1;
    }

    /// Clears all removal tracking records for all component types.
    ///
    /// Call this at the start of each frame (after systems have had a chance
    /// to observe removals via [`removed`](World::removed)) to prevent
    /// unbounded growth of removal records.
    pub fn clear_removed_tracking(&mut self) {
        for lock in self.components.values_mut() {
            lock.get_mut().clear_removed();
        }
    }

    // ---- Commands ----

    /// Initializes a [`CommandBuffer`] resource if not already present.
    ///
    /// Call this before running systems that use commands.
    pub fn init_commands(&mut self) {
        if !self.has_resource::<CommandBuffer>() {
            self.insert_resource(CommandBuffer::new());
        }
    }

    /// Drains and applies all queued commands from the [`CommandBuffer`] resource.
    ///
    /// Each command receives `&mut World` and can perform structural changes
    /// (spawn, despawn, insert, remove). Commands execute in the order they
    /// were queued.
    ///
    /// Call this after `schedule.run()` or between schedule stages.
    ///
    /// # Panics
    ///
    /// Panics if the `CommandBuffer` resource does not exist.
    /// Call [`init_commands`](World::init_commands) first.
    pub fn apply_commands(&mut self) {
        let cmds = {
            let buffer = self.resources.borrow::<CommandBuffer>();
            buffer.drain()
        };
        for cmd in cmds {
            cmd(self);
        }
    }

    // ---- Events ----

    /// Registers an event type by inserting an empty [`Events<T>`] resource.
    ///
    /// Call this during setup, before running systems that send or receive
    /// events of type T.
    pub fn add_event<T: Send + Sync + 'static>(&mut self) {
        self.insert_resource(Events::<T>::new());
    }
}
