use std::any::TypeId;

use smallvec::SmallVec;

use crate::bundle::Bundle;
use crate::component::Component;
use crate::entity::{Entities, Entity};
use crate::observer::{OnAdd, OnInsert, OnRemove};
use crate::sparse_set::{
    ComponentHooks, ComponentMeta, ComponentStorage, Mut, RequiredComponentFn,
};

use super::{World, WorldError, deserialize_component_fn, serialize_component_fn};

/// Pending hook work from a raw component insertion.
///
/// Returned by [`World::insert_raw`] and consumed by [`World::apply_pending`].
/// Separates the storage write from hook/observer processing so that bundles
/// can insert all components first, then fire hooks when the entity is complete.
pub(crate) struct InsertPending {
    pub(crate) was_new: bool,
    pub(crate) on_add: ComponentHooks,
    pub(crate) on_insert: ComponentHooks,
    pub(crate) required: SmallVec<[RequiredComponentFn; 2]>,
    pub(crate) add_trigger_fn: Option<fn(&mut World, Entity)>,
    pub(crate) insert_trigger_fn: fn(&mut World, Entity),
}

fn push_add_trigger<T: 'static>(world: &mut World, entity: Entity) {
    world.observers.push_typed_trigger::<OnAdd<T>>(entity);
}

fn push_insert_trigger<T: 'static>(world: &mut World, entity: Entity) {
    world.observers.push_typed_trigger::<OnInsert<T>>(entity);
}

impl World {
    // ---- Component management (structural changes, require &mut self) ----

    /// Registers a component type without inserting any data.
    ///
    /// This is only needed if you want to query a component type
    /// before any entity has been given that component.
    /// Does not register inspector metadata — use [`register_inspector`](World::register_inspector)
    /// or [`register_inspector_default`](World::register_inspector_default) for that.
    pub fn register_component<T: Send + Sync + 'static>(&mut self) {
        // Record the type's origin (ADR-020 type-identity amendment) and
        // fail-fast on a source conflict. This is the single choke point every
        // component registration path funnels through.
        self.record_type_source(TypeId::of::<T>(), std::any::type_name::<T>());
        let type_id = TypeId::of::<T>();
        if !self.components.contains_key(&type_id) {
            // First registration of this type: stamp it with a monotonic
            // sequence so despawn can fire cross-type `on_remove` hooks in a
            // deterministic order rather than `HashMap` iteration order (#43).
            let seq = self.next_registration_seq;
            self.next_registration_seq += 1;
            let mut storage = ComponentStorage::new::<T>();
            storage.registration_seq = seq;
            self.components
                .insert(type_id, crate::sync::RwLock::new(storage));
        }
    }

    /// Registers a component type with inspector support.
    ///
    /// Creates storage and stores type-erased inspector metadata so the
    /// component can be enumerated, inspected, and removed via the UI.
    ///
    /// The component will be visible in the inspector but cannot be added
    /// via the "Add Component" button. Use [`register_inspector_default`](World::register_inspector_default)
    /// for that.
    pub fn register_inspector<T: Component>(&mut self) {
        self.register_component::<T>();
        T::register_required(self);
        self.name_index.insert(T::NAME, TypeId::of::<T>());
        let storage = self
            .components
            .get_mut(&TypeId::of::<T>())
            .unwrap()
            .get_mut();
        storage.meta = Some(ComponentMeta {
            name: T::NAME,
            schema_version: T::SCHEMA_VERSION,
            schema_hash: T::schema_hash(),
            has_fn: |world, entity| world.get::<T>(entity).is_some(),
            inspect_fn: |world, entity, ui| {
                let comp = world.get::<T>(entity)?;
                comp.inspect_ui(ui, world, entity)
            },
            remove_fn: |world, entity| world.remove::<T>(entity).is_ok_and(|v| v.is_some()),
            insert_default_fn: None,
            collect_entities_fn: |world, entity, collector| {
                if let Some(comp) = world.get::<T>(entity) {
                    comp.collect_entities(collector);
                }
            },
            remap_entities_fn: |world, entity, map| {
                if let Some(mut comp) = world.get_mut::<T>(entity) {
                    comp.remap_entities(map);
                }
            },
            clone_fn: Some(|world, src, dst| {
                let cloned = world.get::<T>(src).cloned();
                if let Some(val) = cloned {
                    let _ = world.insert(dst, val);
                    true
                } else {
                    false
                }
            }),
            extract_fn: Some(|world, entity| {
                world.get::<T>(entity).cloned().map(|v| {
                    Box::new(crate::prefab::TypedBag(v)) as Box<dyn crate::prefab::ComponentBag>
                })
            }),
            aabb_fn: |world, entity| {
                let comp = world.get::<T>(entity)?;
                comp.aabb(world)
            },
            scan_asset_refs_fn: |world, since, f| {
                if let Ok(storage) = world.read_all::<T>() {
                    for (idx, comp) in storage.iter() {
                        if let Some(t) = since
                            && !storage.changed_since(idx, t)
                        {
                            continue;
                        }
                        comp.visit_asset_refs(&mut |r| f(idx, r));
                    }
                }
            },
            patch_asset_refs_fn: |world, idx, f| {
                if let Ok(mut storage) = world.write_all_storage::<T>()
                    && let Some(mut comp) = storage.get_mut(idx)
                {
                    comp.visit_asset_refs_mut(f);
                }
            },
            serialize_fn: serialize_component_fn::<T>,
            deserialize_fn: deserialize_component_fn::<T>,
            display_order: 100,
            gizmo_anchors_fn: None,
            gizmo_drag_fn: None,
        });
    }

    /// Registers a component type with full inspector support including "Add Component".
    ///
    /// Like [`register_inspector`](World::register_inspector) but also enables
    /// inserting a default instance via the inspector "Add Component" button.
    pub fn register_inspector_default<T: Component + Default>(&mut self) {
        self.register_component::<T>();
        T::register_required(self);
        self.name_index.insert(T::NAME, TypeId::of::<T>());
        let storage = self
            .components
            .get_mut(&TypeId::of::<T>())
            .unwrap()
            .get_mut();
        storage.meta = Some(ComponentMeta {
            name: T::NAME,
            schema_version: T::SCHEMA_VERSION,
            schema_hash: T::schema_hash(),
            has_fn: |world, entity| world.get::<T>(entity).is_some(),
            inspect_fn: |world, entity, ui| {
                let comp = world.get::<T>(entity)?;
                comp.inspect_ui(ui, world, entity)
            },
            remove_fn: |world, entity| world.remove::<T>(entity).is_ok_and(|v| v.is_some()),
            insert_default_fn: Some(|world, entity| {
                let _ = world.insert(entity, T::default());
            }),
            collect_entities_fn: |world, entity, collector| {
                if let Some(comp) = world.get::<T>(entity) {
                    comp.collect_entities(collector);
                }
            },
            remap_entities_fn: |world, entity, map| {
                if let Some(mut comp) = world.get_mut::<T>(entity) {
                    comp.remap_entities(map);
                }
            },
            clone_fn: Some(|world, src, dst| {
                let cloned = world.get::<T>(src).cloned();
                if let Some(val) = cloned {
                    let _ = world.insert(dst, val);
                    true
                } else {
                    false
                }
            }),
            extract_fn: Some(|world, entity| {
                world.get::<T>(entity).cloned().map(|v| {
                    Box::new(crate::prefab::TypedBag(v)) as Box<dyn crate::prefab::ComponentBag>
                })
            }),
            aabb_fn: |world, entity| {
                let comp = world.get::<T>(entity)?;
                comp.aabb(world)
            },
            scan_asset_refs_fn: |world, since, f| {
                if let Ok(storage) = world.read_all::<T>() {
                    for (idx, comp) in storage.iter() {
                        if let Some(t) = since
                            && !storage.changed_since(idx, t)
                        {
                            continue;
                        }
                        comp.visit_asset_refs(&mut |r| f(idx, r));
                    }
                }
            },
            patch_asset_refs_fn: |world, idx, f| {
                if let Ok(mut storage) = world.write_all_storage::<T>()
                    && let Some(mut comp) = storage.get_mut(idx)
                {
                    comp.visit_asset_refs_mut(f);
                }
            },
            serialize_fn: serialize_component_fn::<T>,
            deserialize_fn: deserialize_component_fn::<T>,
            display_order: 100,
            gizmo_anchors_fn: None,
            gizmo_drag_fn: None,
        });
    }

    /// Sets the display order for a previously registered inspector component.
    ///
    /// Lower values appear first in the component inspector panel.
    /// The default order is `100`.
    pub fn set_inspector_order<T: Component>(&mut self, order: u32) {
        if let Some(storage) = self
            .components
            .get_mut(&TypeId::of::<T>())
            .map(|l| l.get_mut())
            && let Some(meta) = storage.meta_mut()
        {
            meta.display_order = order;
        }
    }

    /// Registers a requirement: inserting component `T` on an entity will
    /// automatically insert `R::default()` if the entity does not already
    /// have `R`.
    ///
    /// Requirements are transitive. If `T` requires `R` and `R` requires `S`,
    /// inserting `T` will also insert `S` (because inserting `R` triggers
    /// its own requirements).
    ///
    /// Auto-registers `R` if not already registered.
    ///
    /// # Panics
    ///
    /// Panics if `T` has not been registered.
    ///
    /// # Example
    ///
    /// ```ignore
    /// world.register_component::<Camera>();
    /// world.register_required::<Camera, Transform>();
    /// world.register_required::<Camera, Visibility>();
    ///
    /// let entity = world.spawn();
    /// world.insert(entity, camera).unwrap();
    /// // Transform and Visibility are now also on the entity
    /// ```
    pub fn register_required<T: Send + Sync + 'static, R: Send + Sync + Default + 'static>(
        &mut self,
    ) -> &mut Self {
        self.register_component::<R>();

        let storage = self
            .components
            .get_mut(&TypeId::of::<T>())
            .expect("Component T not registered — call register_component::<T>() first")
            .get_mut();

        storage.required_components.push(|world, entity| {
            if world.get::<R>(entity).is_none() {
                let _ = world.insert(entity, R::default());
            }
        });

        self
    }

    /// Enables type-erased cloning for a component type.
    ///
    /// Inserts a component on an entity.
    ///
    /// If the entity already has this component, the value is replaced.
    /// Records the current world tick for change detection — the component
    /// will be visible to [`Changed<T>`](crate::Changed) filters.
    ///
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if `T` has never been registered
    /// via [`register_component`](World::register_component).
    ///
    /// # Panics
    ///
    /// Panics if the entity is not alive.
    /// Inserts a component on an entity.
    ///
    /// If the entity already has this component, the value is replaced.
    /// Records the current world tick for change detection — the component
    /// will be visible to [`Changed<T>`](crate::Changed) filters.
    ///
    /// # Errors
    ///
    /// Returns [`WorldError::EntityNotAlive`] if the entity is dead.
    /// Returns [`WorldError::ComponentNotRegistered`] if `T` has never been registered
    /// via [`register_component`](World::register_component).
    pub fn insert<T: Send + Sync + 'static>(
        &mut self,
        entity: Entity,
        component: T,
    ) -> Result<(), WorldError> {
        if !self.entities.is_alive(entity) {
            return Err(WorldError::EntityNotAlive { entity });
        }
        let pending = self.insert_raw(entity, component)?;
        self.apply_pending(entity, &[pending]);
        Ok(())
    }

    /// Fires a component's hooks one at a time, re-validating between calls.
    ///
    /// A hook may despawn the entity or remove the component (directly or
    /// via a cascade) — the remaining hooks are then skipped instead of
    /// firing on dead or absent data, and a component removed by a hook is
    /// not double-processed by the caller.
    ///
    /// Returns `false` if the entity is no longer alive afterwards.
    pub(crate) fn fire_hooks_checked(
        &mut self,
        entity: Entity,
        type_id: TypeId,
        hooks: &ComponentHooks,
    ) -> bool {
        for hook in hooks.iter() {
            if !self.entities.is_alive(entity) {
                return false;
            }
            let present = self
                .storage_mut(&type_id)
                .is_some_and(|s| s.contains_untyped(entity.index()));
            if !present {
                return true;
            }
            hook(self, entity);
        }
        self.entities.is_alive(entity)
    }

    /// Low-level insert: writes component to storage and fires on_replace.
    ///
    /// Does NOT fire on_add/on_insert hooks, apply required components, or
    /// queue observer triggers. Returns [`InsertPending`] for the caller
    /// to process via [`apply_pending`].
    ///
    /// The caller must have already verified the entity is alive.
    /// Re-checks liveness after `on_replace` hooks: a hook that despawns
    /// the entity aborts the insert with [`WorldError::EntityNotAlive`]
    /// instead of writing phantom data into the freed slot.
    pub(crate) fn insert_raw<T: Send + Sync + 'static>(
        &mut self,
        entity: Entity,
        component: T,
    ) -> Result<InsertPending, WorldError> {
        let type_id = TypeId::of::<T>();

        // Extract hook info and requirements (borrow released after this block)
        let (had_component, on_add, on_insert, on_replace, required) = {
            let storage = self
                .components
                .get_mut(&type_id)
                .map(|l| l.get_mut())
                .ok_or(WorldError::ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
            (
                storage.contains_untyped(entity.index()),
                storage.on_add.clone(),
                storage.on_insert.clone(),
                storage.on_replace.clone(),
                storage.required_components.clone(),
            )
        };

        // Fire on_replace BEFORE overwriting (old value still readable),
        // re-validating between hooks.
        if had_component && !self.fire_hooks_checked(entity, type_id, &on_replace) {
            // A hook despawned the entity — do NOT write into the freed
            // slot (the next spawn would inherit phantom data).
            return Err(WorldError::EntityNotAlive { entity });
        }

        // A replace hook may have removed the component; recompute so the
        // write below counts as a fresh add (on_add, required components
        // and add triggers fire) rather than a replacement.
        let had_component = had_component
            && self
                .storage_mut(&type_id)
                .is_some_and(|s| s.contains_untyped(entity.index()));

        // Perform the actual insert (always tracked at current tick)
        let tick = self.current_tick();
        self.storage_mut(&type_id).unwrap().typed_mut::<T>().insert(
            entity.index(),
            component,
            tick,
        );

        Ok(InsertPending {
            was_new: !had_component,
            on_add: if !had_component {
                on_add
            } else {
                ComponentHooks::new()
            },
            on_insert,
            required: if !had_component {
                required
            } else {
                SmallVec::new()
            },
            add_trigger_fn: if !had_component {
                Some(push_add_trigger::<T>)
            } else {
                None
            },
            insert_trigger_fn: push_insert_trigger::<T>,
        })
    }

    /// Processes deferred work from one or more [`insert_raw`] calls.
    ///
    /// Order:
    /// 1. Required components (for new components only)
    /// 2. on_add hooks (for new components only)
    /// 3. on_insert hooks (for all components)
    /// 4. Observer triggers
    pub(crate) fn apply_pending(&mut self, entity: Entity, pending: &[InsertPending]) {
        for d in pending {
            if d.was_new {
                for req_fn in &d.required {
                    req_fn(self, entity);
                }
            }
        }
        for d in pending {
            d.on_add.fire(self, entity);
        }
        for d in pending {
            d.on_insert.fire(self, entity);
        }
        for d in pending {
            if let Some(trigger_fn) = d.add_trigger_fn {
                trigger_fn(self, entity);
            }
            (d.insert_trigger_fn)(self, entity);
        }
    }

    /// Checks whether all component types in a bundle are registered.
    ///
    /// Call this before writing to ensure no partial inserts can occur.
    pub fn is_component_registered_by_id(&self, type_id: TypeId) -> bool {
        self.components.contains_key(&type_id)
    }

    /// Inserts a bundle of components on an entity.
    ///
    /// Validates all component types upfront. All components are written
    /// before any hooks fire, so hooks see the complete entity.
    ///
    /// # Errors
    ///
    /// Returns [`WorldError::EntityNotAlive`] if the entity is dead.
    /// Returns [`WorldError::ComponentNotRegistered`] if any component type
    /// has never been registered.
    pub fn insert_bundle(&mut self, entity: Entity, bundle: impl Bundle) -> Result<(), WorldError> {
        if !self.entities.is_alive(entity) {
            return Err(WorldError::EntityNotAlive { entity });
        }
        bundle.validate(self)?;
        bundle.insert_into(self, entity);
        Ok(())
    }

    /// Spawns a new entity with a bundle of components.
    ///
    /// Validates all component types upfront. If validation fails, no
    /// entity is spawned and the world is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`WorldError::ComponentNotRegistered`] if any component type
    /// has not been registered.
    pub fn spawn_with(&mut self, bundle: impl Bundle) -> Result<Entity, WorldError> {
        bundle.validate(self)?;
        let entity = self.spawn();
        bundle.insert_into(self, entity);
        Ok(entity)
    }

    /// Removes a component from an entity.
    ///
    /// Removes a component from an entity.
    ///
    /// Returns `Ok(Some(value))` if the component was present and removed,
    /// `Ok(None)` if the entity is alive but does not have this component,
    /// or `Err(WorldError::EntityNotAlive)` if the entity is dead.
    ///
    /// Fires `on_remove` hook before removal. Records the removal for
    /// [`removed`](World::removed) filter queries.
    ///
    /// If an `on_remove` hook despawns the entity or removes the component
    /// itself, the cleanup performed by that hook wins and this call
    /// returns `Ok(None)` — it never touches the slot by raw index after
    /// the entity died (a respawn may already own it).
    pub fn remove<T: 'static>(&mut self, entity: Entity) -> Result<Option<T>, WorldError> {
        if !self.entities.is_alive(entity) {
            return Err(WorldError::EntityNotAlive { entity });
        }

        let tick = self.current_tick();
        let type_id = TypeId::of::<T>();

        // Check presence and extract hooks
        let on_remove = {
            let Some(storage) = self.storage_mut(&type_id) else {
                return Ok(None);
            };
            if !storage.contains_untyped(entity.index()) {
                return Ok(None);
            }
            storage.on_remove.clone()
        };

        // Fire on_remove BEFORE removal (value still readable),
        // re-validating between hooks.
        if !self.fire_hooks_checked(entity, type_id, &on_remove) {
            // A hook despawned the entity; despawn already removed the
            // component. The freed slot may belong to a new entity — do
            // not touch it by raw index.
            return Ok(None);
        }

        // Perform removal. If a hook removed the component itself, the
        // sparse set returns None and nothing is double-recorded.
        let Some(storage) = self.storage_mut(&type_id) else {
            return Ok(None);
        };
        let result = storage.typed_mut::<T>().remove(entity.index());
        if result.is_some() {
            storage.record_removal(entity.index(), tick);
            // Queue deferred observer trigger
            self.observers.push_typed_trigger::<OnRemove<T>>(entity);
        }
        Ok(result)
    }

    /// Returns a reference to a component on an entity.
    ///
    /// We intentionally bypass the `RwLock` and return a plain `&T` instead
    /// of a lock guard. This avoids deadlocks in hooks (which read components
    /// while inside an `insert`/`remove` call path) and keeps the API
    /// ergonomic for the common single-threaded access pattern.
    ///
    /// # Safety of the `unsafe` block
    ///
    /// `data_ptr()` returns a raw pointer to the storage without locking.
    /// This is sound because every path that can mutate component storage is
    /// exclusive with the `&self` borrow this reference derives from:
    ///
    /// - all public mutating APIs (`insert`, `remove`, `despawn`, `write`,
    ///   `try_write`, `write_all`, `try_write_all`, `query`) take `&mut self`;
    /// - systems mutate storage only inside a runner, and every runner entry
    ///   point (`EcsRunner::run`, `Schedules::run_frame`) takes `&mut World`,
    ///   so no external `&self` can coexist with a running schedule;
    /// - the crate-internal `&self` accessors (`write_storage`,
    ///   `write_unlocked`, …) are used only by fetch paths and systems under
    ///   the discipline above.
    ///
    /// Crate-internal rule: code running *inside* a schedule (systems holding
    /// `ctx.world()`) must not call the unlocked readers (`get`, the bare
    /// filter constructors) for storages a sibling system may write — use the
    /// locking accessors instead.
    pub fn get<T: 'static>(&self, entity: Entity) -> Option<&T> {
        // Reject stale/dead handles so a recycled slot does not return the new
        // occupant's component (component storage is indexed by slot only).
        if !self.entities.is_alive(entity) {
            return None;
        }
        let lock = self.components.get(&TypeId::of::<T>())?;
        // SAFETY: see above — no mutation path can overlap the `&self` this
        // reference is tied to. Not taking the RwLock also keeps hook
        // callbacks (inside insert/remove) deadlock-free.
        let storage = unsafe { &*lock.data_ptr() };
        storage.typed::<T>().get(entity.index())
    }

    /// Returns a [`Mut`] wrapper for a component on an entity.
    ///
    /// The component is marked as changed only when [`DerefMut`] is invoked.
    pub fn get_mut<T: 'static>(&mut self, entity: Entity) -> Option<Mut<'_, T>> {
        // Reject stale/dead handles (see `get`).
        if !self.entities.is_alive(entity) {
            return None;
        }
        let tick = self.current_tick();
        let storage = self
            .components
            .get_mut(&TypeId::of::<T>())
            .map(|l| l.get_mut())?;
        storage
            .typed_mut::<T>()
            .get_mut_tracked(entity.index(), tick)
    }

    /// Returns a reference to the entity store.
    ///
    /// Used by `Ref`/`RefMut` to filter disabled entities during iteration.
    pub(crate) fn entities(&self) -> &Entities {
        &self.entities
    }

    /// Returns whether the given entity is currently disabled.
    pub fn is_disabled(&self, entity: Entity) -> bool {
        self.entities.get_flags(entity.index()) & Entity::DISABLED != 0
    }

    /// Returns whether the given entity is currently static.
    pub fn is_static(&self, entity: Entity) -> bool {
        self.entities.get_flags(entity.index()) & Entity::STATIC != 0
    }

    /// Returns whether the given entity is an editor entity.
    pub fn is_editor(&self, entity: Entity) -> bool {
        self.entities.get_flags(entity.index()) & Entity::EDITOR != 0
    }

    /// Returns whether the given entity is excluded from game systems.
    ///
    /// An entity is excluded if it matches the default game query exclude mask:
    /// disabled, static, or editor.
    ///
    /// Use this in side-table invalidation predicates (e.g., physics bodies, asset residents)
    /// to ensure they stay in sync with default query visibility.
    pub fn is_excluded_from_game(&self, entity: Entity) -> bool {
        let flags = self.get_entity_flags(entity);
        flags & Entity::DEFAULT_GAME_QUERY_EXCLUDE_MASK != 0
    }

    /// Returns the current flags for an entity.
    pub fn get_entity_flags(&self, entity: Entity) -> u32 {
        self.entities.get_flags(entity.index())
    }

    /// Returns the world tick when the entity was spawned.
    pub fn get_entity_world_tick(&self, entity: Entity) -> u64 {
        self.entities.get_world_tick(entity.index())
    }

    /// Sets flag bits on an entity (OR operation).
    ///
    /// # Panics
    ///
    /// Panics if the entity is not alive.
    pub fn set_entity_flags(&mut self, entity: Entity, bits: u32) {
        assert!(
            self.entities.is_alive(entity),
            "Cannot set flags on dead entity {entity}"
        );
        self.entities.set_flags(entity.index(), bits);
    }

    /// Clears flag bits on an entity (AND-NOT operation).
    ///
    /// # Panics
    ///
    /// Panics if the entity is not alive.
    pub fn clear_entity_flags(&mut self, entity: Entity, bits: u32) {
        assert!(
            self.entities.is_alive(entity),
            "Cannot clear flags on dead entity {entity}"
        );
        self.entities.clear_flags(entity.index(), bits);
    }
}
