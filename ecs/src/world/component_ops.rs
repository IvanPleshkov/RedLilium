use std::any::TypeId;

use crate::bundle::Bundle;
use crate::component::Component;
use crate::entity::{Entities, Entity};
use crate::observer::{OnAdd, OnInsert, OnRemove};
use crate::sparse_set::{ComponentMeta, ComponentStorage, Mut};

use super::{ComponentNotRegistered, World, deserialize_component_fn, serialize_component_fn};

impl World {
    // ---- Component management (structural changes, require &mut self) ----

    /// Registers a component type without inserting any data.
    ///
    /// This is only needed if you want to query a component type
    /// before any entity has been given that component.
    /// Does not register inspector metadata — use [`register_inspector`](World::register_inspector)
    /// or [`register_inspector_default`](World::register_inspector_default) for that.
    pub fn register_component<T: Send + Sync + 'static>(&mut self) {
        self.components
            .entry(TypeId::of::<T>())
            .or_insert_with(|| parking_lot::RwLock::new(ComponentStorage::new::<T>()));
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
            has_fn: |world, entity| world.get::<T>(entity).is_some(),
            inspect_fn: |world, entity, ui| {
                let comp = world.get::<T>(entity)?;
                comp.inspect_ui(ui, world, entity)
            },
            remove_fn: |world, entity| world.remove::<T>(entity).is_some(),
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
            serialize_fn: serialize_component_fn::<T>,
            deserialize_fn: deserialize_component_fn::<T>,
            display_order: 100,
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
            has_fn: |world, entity| world.get::<T>(entity).is_some(),
            inspect_fn: |world, entity, ui| {
                let comp = world.get::<T>(entity)?;
                comp.inspect_ui(ui, world, entity)
            },
            remove_fn: |world, entity| world.remove::<T>(entity).is_some(),
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
            serialize_fn: serialize_component_fn::<T>,
            deserialize_fn: deserialize_component_fn::<T>,
            display_order: 100,
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
    pub fn insert<T: Send + Sync + 'static>(
        &mut self,
        entity: Entity,
        component: T,
    ) -> Result<(), ComponentNotRegistered> {
        assert!(
            self.entities.is_alive(entity),
            "Cannot insert component on dead entity {entity}"
        );

        let type_id = TypeId::of::<T>();

        // Extract hook info and requirements (borrow released after this block)
        let (had_component, on_add, on_insert, on_replace, required) = {
            let storage = self
                .components
                .get_mut(&type_id)
                .map(|l| l.get_mut())
                .ok_or(ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
            (
                storage.contains_untyped(entity.index()),
                storage.on_add,
                storage.on_insert,
                storage.on_replace,
                storage.required_components.clone(),
            )
        };

        // Fire on_replace BEFORE overwriting (old value still readable)
        if had_component && let Some(hook) = on_replace {
            hook(self, entity);
        }

        // Perform the actual insert (always tracked at current tick)
        let tick = self.tick;
        self.storage_mut(&type_id)
            .unwrap()
            .typed_mut::<T>()
            .insert_with_tick(entity.index(), component, tick);

        // Fire on_add / on_insert AFTER insertion
        if !had_component && let Some(hook) = on_add {
            hook(self, entity);
        }
        if let Some(hook) = on_insert {
            hook(self, entity);
        }

        // Apply required components (only on first add)
        if !had_component {
            for req_fn in required {
                req_fn(self, entity);
            }
        }

        // Queue deferred observer triggers
        if !had_component {
            self.observers.push_typed_trigger::<OnAdd<T>>(entity);
        }
        self.observers.push_typed_trigger::<OnInsert<T>>(entity);

        Ok(())
    }

    /// Inserts a bundle of components on an entity.
    ///
    /// A bundle is a tuple of components, e.g. `(Position, Velocity, Health)`.
    /// All components are inserted at once.
    ///
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if any component type has never
    /// been registered.
    ///
    /// # Panics
    ///
    /// Panics if the entity is not alive.
    pub fn insert_bundle(
        &mut self,
        entity: Entity,
        bundle: impl Bundle,
    ) -> Result<(), ComponentNotRegistered> {
        assert!(
            self.entities.is_alive(entity),
            "Cannot insert bundle on dead entity {entity}"
        );
        bundle.insert_into(self, entity)
    }

    /// Spawns a new entity with a bundle of components.
    ///
    /// Convenience for `spawn()` + `insert_bundle()`.
    ///
    /// # Panics
    ///
    /// Panics if any component type has not been registered.
    pub fn spawn_with(&mut self, bundle: impl Bundle) -> Entity {
        let entity = self.spawn();
        bundle
            .insert_into(self, entity)
            .expect("Component in bundle not registered");
        entity
    }

    /// Removes a component from an entity.
    ///
    /// Returns the removed value, or `None` if the entity did not have it.
    /// Fires `on_remove` hook before removal. Records the removal for
    /// [`removed`](World::removed) filter queries.
    pub fn remove<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        let tick = self.tick;
        let type_id = TypeId::of::<T>();

        // Check presence and extract hook
        let on_remove = {
            let storage = self.storage_mut(&type_id)?;
            if !storage.contains_untyped(entity.index()) {
                return None;
            }
            storage.on_remove
        };

        // Fire on_remove BEFORE removal (value still readable)
        if let Some(hook) = on_remove {
            hook(self, entity);
        }

        // Perform removal
        let storage = self.storage_mut(&type_id)?;
        let result = storage.typed_mut::<T>().remove(entity.index());
        if result.is_some() {
            storage.record_removal(entity.index(), tick);
            // Queue deferred observer trigger
            self.observers.push_typed_trigger::<OnRemove<T>>(entity);
        }
        result
    }

    /// Returns a reference to a component on an entity.
    pub fn get<T: 'static>(&self, entity: Entity) -> Option<&T> {
        let lock = self.components.get(&TypeId::of::<T>())?;
        let storage = unsafe { &*lock.data_ptr() };
        storage.typed::<T>().get(entity.index())
    }

    /// Returns a [`Mut`] wrapper for a component on an entity.
    ///
    /// The component is marked as changed only when [`DerefMut`] is invoked.
    pub fn get_mut<T: 'static>(&mut self, entity: Entity) -> Option<Mut<'_, T>> {
        let tick = self.tick;
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

    /// Returns the current flags for an entity.
    pub fn get_entity_flags(&self, entity: Entity) -> u32 {
        self.entities.get_flags(entity.index())
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
