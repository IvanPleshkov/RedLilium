use crate::entity::Entity;
use crate::observer::{OnAdd, OnInsert, OnRemove};
use crate::reactive::Triggers;
use crate::sparse_set::ComponentHookFn;

use super::World;

impl World {
    // ---- Lifecycle hooks ----

    /// Sets the `on_add` hook for component type `T`.
    ///
    /// The hook fires after a component is inserted on an entity that
    /// did **not** previously have it. It does not fire on replacement.
    ///
    /// # Panics
    ///
    /// Panics if `T` has not been registered.
    pub fn set_on_add<T: 'static>(&mut self, hook: ComponentHookFn) -> &mut Self {
        self.components
            .get_mut(&std::any::TypeId::of::<T>())
            .expect("Component not registered")
            .get_mut()
            .on_add = Some(hook);
        self
    }

    /// Sets the `on_insert` hook for component type `T`.
    ///
    /// The hook fires after every insertion — both new additions and
    /// replacements of existing values.
    ///
    /// # Panics
    ///
    /// Panics if `T` has not been registered.
    pub fn set_on_insert<T: 'static>(&mut self, hook: ComponentHookFn) -> &mut Self {
        self.components
            .get_mut(&std::any::TypeId::of::<T>())
            .expect("Component not registered")
            .get_mut()
            .on_insert = Some(hook);
        self
    }

    /// Sets the `on_replace` hook for component type `T`.
    ///
    /// The hook fires just **before** an existing component value is
    /// overwritten by a new insertion. The old value is still readable
    /// via `world.get::<T>(entity)` inside the hook.
    ///
    /// # Panics
    ///
    /// Panics if `T` has not been registered.
    pub fn set_on_replace<T: 'static>(&mut self, hook: ComponentHookFn) -> &mut Self {
        self.components
            .get_mut(&std::any::TypeId::of::<T>())
            .expect("Component not registered")
            .get_mut()
            .on_replace = Some(hook);
        self
    }

    /// Sets the `on_remove` hook for component type `T`.
    ///
    /// The hook fires just **before** the component is removed from the
    /// entity (including during despawn). The value is still readable
    /// via `world.get::<T>(entity)` inside the hook.
    ///
    /// # Panics
    ///
    /// Panics if `T` has not been registered.
    pub fn set_on_remove<T: 'static>(&mut self, hook: ComponentHookFn) -> &mut Self {
        self.components
            .get_mut(&std::any::TypeId::of::<T>())
            .expect("Component not registered")
            .get_mut()
            .on_remove = Some(hook);
        self
    }

    // ---- Deferred observers ----

    /// Registers a deferred observer that fires when component `T` is
    /// added for the first time on an entity.
    ///
    /// The component is readable via `world.get::<T>(entity)` inside the handler.
    ///
    /// # Example
    ///
    /// ```ignore
    /// world.observe_add::<Health>(|world, entity| {
    ///     let hp = world.get::<Health>(entity).unwrap();
    ///     println!("Entity {entity} gained {hp:?}");
    /// });
    /// ```
    pub fn observe_add<T: 'static>(
        &mut self,
        handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
    ) {
        self.observers.add_on_add::<T>(handler);
    }

    /// Registers a deferred observer that fires on every insertion of
    /// component `T` (both first-time additions and replacements).
    ///
    /// The new value is readable via `world.get::<T>(entity)`.
    pub fn observe_insert<T: 'static>(
        &mut self,
        handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
    ) {
        self.observers.add_on_insert::<T>(handler);
    }

    /// Registers a deferred observer that fires when component `T` is
    /// removed or the entity is despawned.
    ///
    /// **Note**: By the time the observer runs, the component has already
    /// been removed. For cleanup that requires reading the component value,
    /// use [`set_on_remove`](World::set_on_remove) hooks instead.
    pub fn observe_remove<T: 'static>(
        &mut self,
        handler: impl Fn(&mut World, Entity) + Send + Sync + 'static,
    ) {
        self.observers.add_on_remove::<T>(handler);
    }

    /// Drains and fires all pending observer triggers.
    ///
    /// Called by the runner after applying deferred commands. Supports
    /// cascading: observer handlers that perform mutations will queue
    /// new triggers, which are processed in subsequent iterations.
    ///
    /// # Panics
    ///
    /// Panics if cascading exceeds 100 iterations.
    pub fn flush_observers(&mut self) {
        if !self.observers.has_pending() {
            return;
        }
        let world_ptr: *mut World = self;
        self.observers.flush(world_ptr);
    }

    /// Returns `true` if there are pending observer triggers.
    pub fn has_pending_observers(&self) -> bool {
        self.observers.has_pending()
    }

    // ---- Reactive trigger buffers ----

    /// Enables trigger collection for `OnAdd<T>`.
    ///
    /// Creates a [`Triggers<OnAdd<T>>`] resource and registers an internal
    /// observer that collects triggered entities. Systems can then read
    /// `Res<Triggers<OnAdd<T>>>` to get entities that had `T` added last tick.
    ///
    /// The component type `T` must be registered before calling this.
    pub fn enable_add_triggers<T: Send + Sync + 'static>(&mut self) {
        self.insert_resource(Triggers::<OnAdd<T>>::new());
        self.observe_add::<T>(|world, entity| {
            world.resource_mut::<Triggers<OnAdd<T>>>().push(entity);
        });
        self.trigger_swap_fns
            .push(super::swap_trigger_buffer::<OnAdd<T>>);
    }

    /// Enables trigger collection for `OnInsert<T>`.
    ///
    /// Creates a [`Triggers<OnInsert<T>>`] resource and registers an internal
    /// observer that collects triggered entities. Fires on both first-time
    /// addition and replacement of an existing value.
    pub fn enable_insert_triggers<T: Send + Sync + 'static>(&mut self) {
        self.insert_resource(Triggers::<OnInsert<T>>::new());
        self.observe_insert::<T>(|world, entity| {
            world.resource_mut::<Triggers<OnInsert<T>>>().push(entity);
        });
        self.trigger_swap_fns
            .push(super::swap_trigger_buffer::<OnInsert<T>>);
    }

    /// Enables trigger collection for `OnRemove<T>`.
    ///
    /// Creates a [`Triggers<OnRemove<T>>`] resource and registers an internal
    /// observer that collects triggered entities. Fires on explicit removal
    /// and on despawn.
    pub fn enable_remove_triggers<T: Send + Sync + 'static>(&mut self) {
        self.insert_resource(Triggers::<OnRemove<T>>::new());
        self.observe_remove::<T>(|world, entity| {
            world.resource_mut::<Triggers<OnRemove<T>>>().push(entity);
        });
        self.trigger_swap_fns
            .push(super::swap_trigger_buffer::<OnRemove<T>>);
    }

    /// Swaps all reactive trigger buffers.
    ///
    /// Moves `collecting` → `readable` and clears `collecting` for each
    /// registered trigger buffer. Called by the runner at the start of
    /// each tick, before any systems execute.
    pub fn update_triggers(&mut self) {
        if self.trigger_swap_fns.is_empty() {
            return;
        }
        let fns = std::mem::take(&mut self.trigger_swap_fns);
        for f in &fns {
            f(self);
        }
        self.trigger_swap_fns = fns;
    }

    /// Returns the TypeIds of all registered resource types.
    pub fn resource_type_ids(&self) -> impl Iterator<Item = std::any::TypeId> + '_ {
        self.resources.type_ids()
    }
}
