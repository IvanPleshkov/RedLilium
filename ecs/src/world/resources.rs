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
    pub fn insert_resource<T: Resource>(&mut self, value: T) -> Arc<crate::sync::RwLock<T>> {
        // A value insert resolves to the type's declared origin (respecting it),
        // declaring the current source only for a never-before-seen type — so a
        // warm-reload snapshot restore never conflicts. See `resolve_type_source`.
        let source = self.resolve_type_source(std::any::TypeId::of::<T>());
        self.resources.insert(value, source)
    }

    /// Inserts a pre-existing `Arc<RwLock<T>>` as a resource.
    ///
    /// The Arc is coerced to `Arc<RwLock<dyn Resource>>` for storage;
    /// both the caller's clone and the stored clone share the same lock.
    pub fn insert_resource_shared<T: Resource>(&mut self, resource: Arc<crate::sync::RwLock<T>>) {
        // See `insert_resource`: value inserts resolve to the declared origin.
        let source = self.resolve_type_source(std::any::TypeId::of::<T>());
        self.resources.insert_shared(resource, source);
    }

    /// Removes a resource, returning the `Arc<RwLock<dyn Resource>>` if present.
    pub fn remove_resource<T: 'static>(
        &mut self,
    ) -> Option<Arc<crate::sync::RwLock<dyn Resource>>> {
        self.resources.remove::<T>()
    }

    /// Moves a resource out of `from` into this world — the **same** `Arc`
    /// and lock, so external handles stay valid. The entry's recorded source
    /// and the type-identity record carry over. Returns `false` when `from`
    /// has no such resource.
    ///
    /// The world-rebuild flows (editor game reload) use this to carry
    /// shell-owned resources (remote transport, GPU-backed renderers) across
    /// to the replacement world instead of recreating them.
    pub fn adopt_resource_from<T: Resource>(&mut self, from: &mut World) -> bool {
        let Some(entry) = from.resources.remove_entry::<T>() else {
            return false;
        };
        let type_id = std::any::TypeId::of::<T>();
        let source = from
            .type_sources
            .remove(&type_id)
            .unwrap_or_else(|| self.resolve_type_source(type_id));
        self.resources.insert_entry::<T>(entry);
        self.type_sources.insert(type_id, source);
        true
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
    pub fn resource_shared<T: 'static>(&self) -> Arc<crate::sync::RwLock<dyn Resource>> {
        self.guard_resource_source::<T>();
        self.resources.get_handle::<T>()
    }

    /// Borrows a resource of type T immutably.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist or is exclusively borrowed.
    pub fn resource<T: 'static>(&self) -> ResourceRef<'_, T> {
        self.guard_resource_source::<T>();
        self.resources.borrow::<T>()
    }

    /// Borrows a resource of type T mutably.
    ///
    /// # Panics
    ///
    /// Panics if the resource does not exist or any borrow is active.
    pub fn resource_mut<T: 'static>(&self) -> ResourceRefMut<'_, T> {
        self.guard_resource_source::<T>();
        self.resources.borrow_mut::<T>()
    }

    /// Try to get mutable access to the shared generation registry, if injected.
    pub(crate) fn with_generation_registry_mut<R>(
        &self,
        f: impl FnOnce(&mut crate::GameGenerationRegistry) -> R,
    ) -> Option<R> {
        if !self.has_resource::<crate::GameGenerationRegistry>() {
            return None;
        }
        let handle = self.resources.get_handle::<crate::GameGenerationRegistry>();
        handle.try_write().and_then(|mut guard| {
            guard
                .as_any_mut()
                .downcast_mut::<crate::GameGenerationRegistry>()
                .map(f)
        })
    }

    /// Try to get read access to the shared generation registry, if injected.
    pub(crate) fn with_generation_registry<R>(
        &self,
        f: impl FnOnce(&crate::GameGenerationRegistry) -> R,
    ) -> Option<R> {
        if !self.has_resource::<crate::GameGenerationRegistry>() {
            return None;
        }
        let handle = self.resources.get_handle::<crate::GameGenerationRegistry>();
        handle.try_read().and_then(|guard| {
            guard
                .as_any()
                .downcast_ref::<crate::GameGenerationRegistry>()
                .map(f)
        })
    }

    /// Downcast source-guard (ADR-020 type-identity amendment): asserts the
    /// stored entry's recorded source matches the source `T` currently resolves
    /// to via the registration map, refusing to reinterpret data left by a
    /// stale generation. It only bites once #45 introduces real per-reload
    /// generations; today every registration is `HOST` so it is always a no-op.
    ///
    /// The whole body is behind `cfg!(debug_assertions)`, so in release the two
    /// map lookups are compiled out entirely (`if false`) rather than relying on
    /// the optimizer to prove them dead — the guard costs exactly nothing on the
    /// hot resource-access path in release. Skipped when either side is unknown;
    /// the one such case, the bootstrap `CommandBuffer`, is now recorded in
    /// [`World::new`], so a skip in practice means an unregistered type.
    #[inline]
    fn guard_resource_source<T: 'static>(&self) {
        if cfg!(debug_assertions)
            && let (Some(entry), Some(expected)) = (
                self.resources.entry_source::<T>(),
                self.resolved_source::<T>(),
            )
        {
            crate::type_identity::assert_source_match(entry, expected, std::any::type_name::<T>());
        }
    }

    /// Borrows a resource immutably **without** locking it.
    ///
    /// # Safety
    ///
    /// The caller must already hold this resource's read lock for the returned
    /// lifetime (e.g. via [`acquire_sorted`](Self::acquire_sorted)).
    pub(crate) unsafe fn resource_unlocked<T: 'static>(&self) -> ResourceRef<'_, T> {
        // The production system-access path (`Res<T>::fetch_unlocked`) flows
        // through here, so the source-guard must sit on it too — not only on the
        // convenience `resource()` path. Reads only the registration maps; the
        // caller's lock is unaffected.
        self.guard_resource_source::<T>();
        unsafe { self.resources.borrow_unlocked::<T>() }
    }

    /// Borrows a resource mutably **without** locking it.
    ///
    /// # Safety
    ///
    /// The caller must already hold this resource's write lock for the returned
    /// lifetime (e.g. via [`acquire_sorted`](Self::acquire_sorted)).
    pub(crate) unsafe fn resource_mut_unlocked<T: 'static>(&self) -> ResourceRefMut<'_, T> {
        // See `resource_unlocked`: guard the production `ResMut<T>` path too.
        self.guard_resource_source::<T>();
        unsafe { self.resources.borrow_mut_unlocked::<T>() }
    }

    /// Read-locks a resource by `TypeId` (blocking), returning the type-erased
    /// guard. Used by `acquire_sorted` for up-front globally-ordered locking.
    pub(crate) fn resource_read_guard_dyn(
        &self,
        type_id: std::any::TypeId,
    ) -> Option<crate::sync::RwLockReadGuard<'_, dyn crate::resource::Resource>> {
        self.resources.read_guard_dyn(type_id)
    }

    /// Write-locks a resource by `TypeId` (blocking). See
    /// [`resource_read_guard_dyn`](Self::resource_read_guard_dyn).
    pub(crate) fn resource_write_guard_dyn(
        &self,
        type_id: std::any::TypeId,
    ) -> Option<crate::sync::RwLockWriteGuard<'_, dyn crate::resource::Resource>> {
        self.resources.write_guard_dyn(type_id)
    }

    // ---- Main-thread resource management ----

    /// Inserts a main-thread resource during world setup.
    ///
    /// The resource does **not** need to implement `Send` or `Sync`.
    /// Takes `&mut self`, so it can only be called before systems run
    /// (during setup on the main thread).
    ///
    /// # Thread confinement
    ///
    /// The first insert pins main-thread resources to the current thread:
    /// all later access happens there (systems reach them via the runner's
    /// main-thread dispatcher), and the `World` itself must be **dropped on
    /// that thread** while main-thread resources are present — otherwise
    /// their destructors would run on a foreign thread. Both are checked
    /// in debug builds.
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
    /// The tick is a **change-detection sequence number**, not a clock: it
    /// advances once per executed system inside a runner (see
    /// [`next_tick`](World::next_tick)) plus at explicit barriers
    /// ([`advance_tick`](World::advance_tick)). It is fully independent of
    /// `RealTime`/`GameTime` — pausing or scaling game time never touches it.
    ///
    /// INVARIANT: the tick doubles as the **entity generation counter**.
    /// Entities are identified as `(index, spawn_tick)`; snapshot restore
    /// matches entities by that pair, and the Stop transition compares entity
    /// world ticks against `PlayStartTick` to separate pre-play entities from
    /// play-spawned ones. Because identity hangs off it, the tick must stay
    /// monotonic for the lifetime of the `World` — never reset or rewind it,
    /// however tempting a "fresh ticks on Play" cleanup may look.
    pub fn current_tick(&self) -> u64 {
        self.tick.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Advances the world tick by one.
    ///
    /// Called by runners before applying deferred commands (so command
    /// writes are stamped after every system's `last_run`) and by
    /// `Schedules::run_frame` after the frame (so out-of-band edits by the
    /// world owner are visible to every system next frame).
    pub fn advance_tick(&mut self) {
        self.tick.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Advances the tick and returns the new value.
    ///
    /// Runners call this to assign each system run a unique `this_run`
    /// stamp: the system's writes are stamped with it, and after the run it
    /// becomes the system's `last_run`, so change filters see exactly what
    /// happened since — including mutations later in the same frame.
    /// Atomic (`&self`) because the multi-threaded runner assigns ticks
    /// while worker systems hold `&World`.
    pub(crate) fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
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

    /// Drains and applies all queued commands from the [`CommandBuffer`] resource.
    ///
    /// Each command receives `&mut World` and can perform structural changes
    /// (spawn, despawn, insert, remove). Commands execute in the order they
    /// were queued.
    ///
    /// Call this after `schedule.run()` or between schedule stages.
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
    /// events of type T. The queue's double buffer is advanced once per
    /// frame by `Schedules::run_frame` (via [`update_events`](World::update_events)).
    pub fn add_event<T: Send + Sync + 'static>(&mut self) {
        // Idempotent: a second registration would add a duplicate update fn,
        // making the buffer advance twice per frame and drop events after
        // one frame instead of two.
        if self.has_resource::<Events<T>>() {
            return;
        }
        self.insert_resource(Events::<T>::new());
        self.event_update_fns
            .push((self.current_source, super::update_event_buffer::<T>));
    }

    /// Advances all registered event queues (drops last frame's events,
    /// current frame's become readable-as-previous).
    ///
    /// Called by `Schedules::run_frame` once at the start of each frame —
    /// NOT per `runner.run`, so event lifetime does not depend on how many
    /// schedules (or FixedUpdate iterations) ran this frame. Drive it
    /// manually when running containers without `run_frame`.
    pub fn update_events(&mut self) {
        if self.event_update_fns.is_empty() {
            return;
        }
        let fns = std::mem::take(&mut self.event_update_fns);
        for (_, f) in &fns {
            f(self);
        }
        self.event_update_fns = fns;
    }
}
