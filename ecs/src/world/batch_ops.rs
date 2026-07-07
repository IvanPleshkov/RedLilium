use std::any::TypeId;

use smallvec::SmallVec;

use crate::bundle::Bundle;
use crate::entity::Entity;
use crate::observer::OnRemove;

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
        let tick = self.current_tick();
        self.entities.allocate_many(count, tick)
    }

    /// Spawns `count` entities, each with a clone of the given bundle.
    ///
    /// Component types are validated up-front, so if validation fails no
    /// entities are spawned and the world is unchanged. Once validation passes
    /// the inserts are infallible, so there is no mid-batch rollback path.
    /// All hooks fire after every entity is fully populated, so `on_add`
    /// handlers can observe the entire batch.
    ///
    /// # Errors
    ///
    /// Returns [`WorldError::ComponentNotRegistered`] if any component type
    /// in the bundle has not been registered.
    pub fn spawn_batch_with(
        &mut self,
        count: u32,
        bundle: impl Bundle + Clone,
    ) -> Result<Vec<Entity>, WorldError> {
        // Validate upfront — no entities spawned if types are wrong
        bundle.validate(self)?;

        let tick = self.current_tick();
        let entities = self.entities.allocate_many(count, tick);
        for &entity in &entities {
            bundle.clone().insert_into(self, entity);
        }
        Ok(entities)
    }

    /// Spawns `count` entities, calling `f(index)` to produce each entity's bundle.
    ///
    /// Use this when each entity needs different component data. All
    /// component types are validated upfront before any entities are spawned.
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
        // Validate with a sample bundle — all bundles have the same types
        if count > 0 {
            f(0).validate(self)?;
        }

        let tick = self.current_tick();
        let entities = self.entities.allocate_many(count, tick);
        for (i, &entity) in entities.iter().enumerate() {
            f(i).insert_into(self, entity);
        }
        Ok(entities)
    }

    /// Despawns multiple entities at once.
    ///
    /// `on_remove` hooks fire first (every batch entity still alive) and
    /// exactly once per component; hook-less components stay readable for
    /// the whole hook phase (see #43 for ordering caveats between hooked
    /// components). Then all entities are deallocated and their remaining
    /// components removed.
    ///
    /// Skips entities that are already dead (a stale handle never touches
    /// the slot's current owner).
    /// Records removals for [`removed`](World::removed) filter queries.
    pub fn despawn_batch(&mut self, entities: &[Entity]) {
        // Phase 1: collect (entity, hooked component) pairs. Only TypeIds
        // are snapshotted — presence is re-checked per hook, so a hook that
        // cascades a removal or despawn does not double-fire.
        let mut hooked: SmallVec<[(Entity, TypeId); 8]> = Default::default();
        for &entity in entities {
            if !self.entities.is_alive(entity) {
                continue;
            }
            let index = entity.index();
            for (type_id, lock) in self.components.iter_mut() {
                let storage = lock.get_mut();
                if !storage.on_remove.is_empty() && storage.contains_untyped(index) {
                    hooked.push((entity, *type_id));
                }
            }
        }

        // Phase 2: per (entity, hooked component) — fire its hooks
        // (re-validating liveness and presence between calls), then remove
        // the component immediately so a cascading remove() from a later
        // hook cannot re-fire it (#43).
        let tick = self.current_tick();
        for (entity, type_id) in hooked {
            if !self.entities.is_alive(entity) {
                continue;
            }
            let hooks = match self.storage_mut(&type_id) {
                Some(s) => s.on_remove.clone(),
                None => continue,
            };
            if !self.fire_hooks_checked(entity, type_id, &hooks) {
                continue; // despawned by a hook; nothing left to remove
            }
            let index = entity.index();
            let removed = self.storage_mut(&type_id).is_some_and(|s| {
                if s.remove_untyped(index) {
                    s.record_removal(index, tick);
                    true
                } else {
                    false
                }
            });
            if removed && let Some(trigger_key) = self.observers.remove_trigger_key(&type_id) {
                self.observers.push_trigger(trigger_key, entity);
            }
        }

        // Phase 3: deallocate all entities and remove components
        for &entity in entities {
            if !self.entities.is_alive(entity) {
                continue;
            }
            let index = entity.index();
            self.entities.deallocate(entity);
            let components = &mut self.components;
            let observers = &mut self.observers;
            for (type_id, lock) in components.iter_mut() {
                let storage = lock.get_mut();
                if storage.remove_untyped(index) {
                    storage.record_removal(index, tick);
                    if let Some(trigger_key) = observers.remove_trigger_key(type_id) {
                        observers.push_trigger(trigger_key, entity);
                    }
                }
            }
        }
    }

    /// Inserts a component on each entity from an iterator of `(Entity, T)` pairs.
    ///
    /// All entities are validated upfront — if any is dead, the entire
    /// operation fails before any mutation.
    ///
    /// Each entity is then processed in turn exactly like [`insert`](Self::insert):
    /// for a given entity all of its hooks run (`on_replace` → write →
    /// `on_add`/`on_insert`) before the next entity is processed. Hooks therefore
    /// observe a partially-applied batch (later entities not yet written) — do
    /// not rely on the whole batch being visible from within a hook.
    ///
    /// Records the current tick for change detection.
    ///
    /// # Errors
    ///
    /// Returns [`WorldError::EntityNotAlive`] if any entity is dead.
    /// Returns [`WorldError::ComponentNotRegistered`] if `T` has never been registered.
    pub fn insert_batch<T: Send + Sync + 'static>(
        &mut self,
        items: impl IntoIterator<Item = (Entity, T)>,
    ) -> Result<(), WorldError> {
        let items: Vec<(Entity, T)> = items.into_iter().collect();

        // Validate all entities upfront — fail before any mutation.
        for (entity, _) in &items {
            if !self.entities.is_alive(*entity) {
                return Err(WorldError::EntityNotAlive { entity: *entity });
            }
        }

        // Validate type registered
        let type_id = TypeId::of::<T>();
        if !self.components.contains_key(&type_id) {
            return Err(WorldError::ComponentNotRegistered {
                type_name: std::any::type_name::<T>(),
            });
        }

        for (entity, component) in items {
            self.insert(entity, component)?;
        }
        Ok(())
    }

    /// Removes a component from multiple entities at once.
    ///
    /// Skips dead entities (a stale handle must never touch the slot's
    /// current owner) and entities that don't have the component.
    /// Fires `on_remove` hook for each entity before removal,
    /// re-validating liveness and presence between hook calls.
    /// Records removals for [`removed`](World::removed) filter queries.
    pub fn remove_batch<T: 'static>(&mut self, entities: &[Entity]) {
        let tick = self.current_tick();
        let type_id = TypeId::of::<T>();

        // Extract on_remove hooks
        let on_remove = {
            let Some(storage) = self.storage_mut(&type_id) else {
                return;
            };
            storage.on_remove.clone()
        };

        // Fire on_remove hooks before removal
        if !on_remove.is_empty() {
            for &entity in entities {
                if !self.entities.is_alive(entity) {
                    continue;
                }
                let present = self
                    .storage_mut(&type_id)
                    .is_some_and(|s| s.contains_untyped(entity.index()));
                if present {
                    self.fire_hooks_checked(entity, type_id, &on_remove);
                }
            }
        }

        // Perform removals — only for entities still alive after the hooks
        // (hooks may have despawned some; their slots may already be owned
        // by a respawn).
        let alive: Vec<Entity> = entities
            .iter()
            .copied()
            .filter(|e| self.entities.is_alive(*e))
            .collect();
        let Some(storage) = self.storage_mut(&type_id) else {
            return;
        };
        let mut removed_entities = Vec::new();
        {
            let set = storage.typed_mut::<T>();
            for &entity in &alive {
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
