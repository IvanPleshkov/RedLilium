use std::any::TypeId;

use crate::bundle::Bundle;
use crate::entity::Entity;
use crate::observer::{OnAdd, OnInsert, OnRemove};
use crate::world::component_ops::InsertPending;

use super::{World, WorldError};

impl World {
    // ---- Batch entity operations ----

    /// Spawns `count` empty entities at once.
    ///
    /// More efficient than calling [`spawn`](World::spawn) in a loop because
    /// the entity allocator grows its internal arrays in bulk.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let entities = world.spawn_batch(100);
    /// assert_eq!(entities.len(), 100);
    /// ```
    pub fn spawn_batch(&mut self, count: u32) -> Vec<Entity> {
        self.entities.allocate_many(count, self.tick)
    }

    /// Spawns `count` entities, each with a clone of the given bundle.
    ///
    /// All entities are spawned and fully populated before any hooks fire.
    /// This means `on_add` handlers can observe not only all components on
    /// their own entity, but also all other entities in the batch. Use this
    /// when spawning a group of related entities that reference each other.
    ///
    /// # Errors
    ///
    /// Returns [`WorldError::ComponentNotRegistered`] if any component type
    /// in the bundle has not been registered. Already-spawned entities
    /// remain alive but may be incomplete.
    pub fn spawn_batch_with(
        &mut self,
        count: u32,
        bundle: impl Bundle + Clone,
    ) -> Result<Vec<Entity>, WorldError> {
        let entities = self.entities.allocate_many(count, self.tick);

        // Phase 1: insert all components for all entities (no hooks)
        let mut all_pending: Vec<(Entity, Vec<InsertPending>)> = Vec::with_capacity(count as usize);
        for &entity in &entities {
            let mut pending = Vec::new();
            bundle.clone().collect_pending(self, entity, &mut pending)?;
            all_pending.push((entity, pending));
        }

        // Phase 2: apply hooks for all entities (all components present)
        for (entity, pending) in &all_pending {
            self.apply_pending(*entity, pending);
        }

        Ok(entities)
    }

    /// Spawns `count` entities, calling `f(index)` to produce each entity's bundle.
    ///
    /// Use this when each entity needs different component data. All entities
    /// are spawned and fully populated before any hooks fire, so `on_add`
    /// handlers can observe the entire batch.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let entities = world.spawn_batch_with_fn(10, |i| {
    ///     (Position { x: i as f32, y: 0.0 }, Health(100))
    /// }).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`WorldError::ComponentNotRegistered`] if any component type
    /// in the bundle has not been registered.
    pub fn spawn_batch_with_fn<B: Bundle>(
        &mut self,
        count: u32,
        f: impl Fn(usize) -> B,
    ) -> Result<Vec<Entity>, WorldError> {
        let entities = self.entities.allocate_many(count, self.tick);

        // Phase 1: insert all components for all entities (no hooks)
        let mut all_pending: Vec<(Entity, Vec<InsertPending>)> = Vec::with_capacity(count as usize);
        for (i, &entity) in entities.iter().enumerate() {
            let mut pending = Vec::new();
            f(i).collect_pending(self, entity, &mut pending)?;
            all_pending.push((entity, pending));
        }

        // Phase 2: apply hooks for all entities (all components present)
        for (entity, pending) in &all_pending {
            self.apply_pending(*entity, pending);
        }

        Ok(entities)
    }

    /// Despawns multiple entities at once.
    ///
    /// Skips entities that are already dead.
    /// Fires `on_remove` hooks before removal for each entity.
    /// Records removals for [`removed`](World::removed) filter queries.
    pub fn despawn_batch(&mut self, entities: &[Entity]) {
        for &entity in entities {
            self.despawn(entity);
        }
    }

    /// Inserts a component on each entity from parallel slices.
    ///
    /// `entities` and `components` must have the same length.
    /// Uses tick 0 (untracked). For change-tracked insertion,
    /// use [`insert_batch_tracked`](World::insert_batch_tracked).
    /// Fires lifecycle hooks (`on_add`, `on_insert`, `on_replace`) per entity.
    ///
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if `T` has never been registered.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths or if any entity is dead.
    pub fn insert_batch<T: Send + Sync + 'static>(
        &mut self,
        entities: &[Entity],
        components: Vec<T>,
    ) -> Result<(), WorldError> {
        assert_eq!(
            entities.len(),
            components.len(),
            "insert_batch: entities and components must have the same length"
        );

        let type_id = TypeId::of::<T>();

        // Verify registered and extract hooks
        let (on_add, on_insert, on_replace, has_required) = {
            let storage = self
                .components
                .get_mut(&type_id)
                .map(|l| l.get_mut())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
            (
                storage.on_add,
                storage.on_insert,
                storage.on_replace,
                storage.has_required_components(),
            )
        };
        let has_hooks = on_add.is_some() || on_insert.is_some() || on_replace.is_some();

        // Reserve capacity upfront
        self.storage_mut(&type_id)
            .unwrap()
            .typed_mut::<T>()
            .reserve(components.len());

        if has_hooks || has_required {
            for (entity, component) in entities.iter().zip(components) {
                assert!(
                    self.entities.is_alive(*entity),
                    "Cannot insert component on dead entity {entity}"
                );
                let had = self
                    .storage_mut(&type_id)
                    .unwrap()
                    .contains_untyped(entity.index());

                if had && let Some(hook) = on_replace {
                    hook(self, *entity);
                }

                self.storage_mut(&type_id).unwrap().typed_mut::<T>().insert(
                    entity.index(),
                    component,
                    0,
                );

                if !had && let Some(hook) = on_add {
                    hook(self, *entity);
                }
                if let Some(hook) = on_insert {
                    hook(self, *entity);
                }

                // Apply required components (only on first add)
                if !had {
                    let required = self
                        .storage_mut(&type_id)
                        .unwrap()
                        .required_components
                        .clone();
                    for req_fn in required {
                        req_fn(self, *entity);
                    }
                }

                // Queue deferred observer triggers
                if !had {
                    self.observers.push_typed_trigger::<OnAdd<T>>(*entity);
                }
                self.observers.push_typed_trigger::<OnInsert<T>>(*entity);
            }
        } else {
            // Fast path: no hooks and no required components, direct sparse set insert
            let storage = self.components.get_mut(&type_id).unwrap().get_mut();
            let set = storage.typed_mut::<T>();
            for (entity, component) in entities.iter().zip(components) {
                assert!(
                    self.entities.is_alive(*entity),
                    "Cannot insert component on dead entity {entity}"
                );
                let had = set.contains(entity.index());
                set.insert(entity.index(), component, 0);

                // Queue deferred observer triggers
                if !had {
                    self.observers.push_typed_trigger::<OnAdd<T>>(*entity);
                }
                self.observers.push_typed_trigger::<OnInsert<T>>(*entity);
            }
        }
        Ok(())
    }

    /// Inserts a component on each entity with change tracking.
    ///
    /// Like [`insert_batch`](World::insert_batch) but records the current tick.
    /// Fires lifecycle hooks (`on_add`, `on_insert`, `on_replace`) per entity.
    ///
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if `T` has never been registered.
    ///
    /// # Panics
    ///
    /// Panics if the slices have different lengths or if any entity is dead.
    pub fn insert_batch_tracked<T: Send + Sync + 'static>(
        &mut self,
        entities: &[Entity],
        components: Vec<T>,
    ) -> Result<(), WorldError> {
        assert_eq!(
            entities.len(),
            components.len(),
            "insert_batch_tracked: entities and components must have the same length"
        );

        let type_id = TypeId::of::<T>();
        let tick = self.tick;

        // Verify registered and extract hooks
        let (on_add, on_insert, on_replace, has_required) = {
            let storage = self
                .components
                .get_mut(&type_id)
                .map(|l| l.get_mut())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
            (
                storage.on_add,
                storage.on_insert,
                storage.on_replace,
                storage.has_required_components(),
            )
        };
        let has_hooks = on_add.is_some() || on_insert.is_some() || on_replace.is_some();

        // Reserve capacity upfront
        self.storage_mut(&type_id)
            .unwrap()
            .typed_mut::<T>()
            .reserve(components.len());

        if has_hooks || has_required {
            for (entity, component) in entities.iter().zip(components) {
                assert!(
                    self.entities.is_alive(*entity),
                    "Cannot insert component on dead entity {entity}"
                );
                let had = self
                    .storage_mut(&type_id)
                    .unwrap()
                    .contains_untyped(entity.index());

                if had && let Some(hook) = on_replace {
                    hook(self, *entity);
                }

                self.storage_mut(&type_id).unwrap().typed_mut::<T>().insert(
                    entity.index(),
                    component,
                    tick,
                );

                if !had && let Some(hook) = on_add {
                    hook(self, *entity);
                }
                if let Some(hook) = on_insert {
                    hook(self, *entity);
                }

                // Apply required components (only on first add)
                if !had {
                    let required = self
                        .storage_mut(&type_id)
                        .unwrap()
                        .required_components
                        .clone();
                    for req_fn in required {
                        req_fn(self, *entity);
                    }
                }

                // Queue deferred observer triggers
                if !had {
                    self.observers.push_typed_trigger::<OnAdd<T>>(*entity);
                }
                self.observers.push_typed_trigger::<OnInsert<T>>(*entity);
            }
        } else {
            // Fast path: no hooks and no required components, direct sparse set insert
            let storage = self.components.get_mut(&type_id).unwrap().get_mut();
            let set = storage.typed_mut::<T>();
            for (entity, component) in entities.iter().zip(components) {
                assert!(
                    self.entities.is_alive(*entity),
                    "Cannot insert component on dead entity {entity}"
                );
                let had = set.contains(entity.index());
                set.insert(entity.index(), component, tick);

                // Queue deferred observer triggers
                if !had {
                    self.observers.push_typed_trigger::<OnAdd<T>>(*entity);
                }
                self.observers.push_typed_trigger::<OnInsert<T>>(*entity);
            }
        }
        Ok(())
    }

    /// Removes a component from multiple entities at once.
    ///
    /// Skips entities that don't have the component.
    /// Fires `on_remove` hook for each entity before removal.
    /// Records removals for [`removed`](World::removed) filter queries.
    pub fn remove_batch<T: 'static>(&mut self, entities: &[Entity]) {
        let tick = self.tick;
        let type_id = TypeId::of::<T>();

        // Extract on_remove hook
        let on_remove = {
            let Some(storage) = self.storage_mut(&type_id) else {
                return;
            };
            storage.on_remove
        };

        // Fire on_remove hooks before removal
        if let Some(hook) = on_remove {
            let with_component: Vec<Entity> = entities
                .iter()
                .copied()
                .filter(|e| {
                    self.storage_mut(&type_id)
                        .is_some_and(|s| s.contains_untyped(e.index()))
                })
                .collect();
            for entity in with_component {
                hook(self, entity);
            }
        }

        // Perform removals
        let Some(storage) = self.storage_mut(&type_id) else {
            return;
        };
        let mut removed_entities = Vec::new();
        {
            let set = storage.typed_mut::<T>();
            for &entity in entities {
                if set.remove(entity.index()).is_some() {
                    removed_entities.push(entity);
                }
            }
        }
        for &entity in &removed_entities {
            storage.record_removal(entity.index(), tick);
        }

        // Queue deferred observer triggers
        for entity in removed_entities {
            self.observers.push_typed_trigger::<OnRemove<T>>(entity);
        }
    }
}
