use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};

use fixedbitset::FixedBitSet;

use crate::access_set::AccessInfo;
use crate::bundle::Bundle;
use crate::commands::CommandBuffer;
use crate::component::Component;
use crate::components::Disabled;
use crate::entity::{Entity, EntityAllocator};
use crate::events::Events;
use crate::main_thread_resource::MainThreadResources;
use crate::observer::{Observers, OnAdd, OnInsert, OnRemove};
use crate::query::{AddedFilter, ChangedFilter, ContainsChecker, RemovedFilter};
use crate::reactive::Triggers;
use crate::resource::{Resource, ResourceRef, ResourceRefMut, Resources};
use crate::sparse_set::{ComponentHookFn, ComponentStorage, LockGuard, Ref, RefMut};
use std::sync::{Arc, RwLock};

/// Error returned when a component type has not been registered in the [`World`].
///
/// This happens when calling [`World::insert`], [`World::read`], or [`World::write`]
/// on a type that was never passed to [`World::register_component`] or inserted.
#[derive(Debug)]
pub struct ComponentNotRegistered {
    /// The name of the unregistered component type.
    pub type_name: &'static str,
}

impl std::fmt::Display for ComponentNotRegistered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Component type `{}` has never been registered. Call register_component() first.",
            self.type_name
        )
    }
}

impl std::error::Error for ComponentNotRegistered {}

// ---------------------------------------------------------------------------
// Inspector metadata (stored in World, separate from component storages)
// ---------------------------------------------------------------------------

/// Type-erased inspector operations for a single component type.
struct InspectorEntry {
    /// Check if an entity has this component.
    has_fn: fn(&World, Entity) -> bool,
    /// Render the component's inspector UI. Returns true if entity had it.
    inspect_fn: fn(&mut World, Entity, &mut egui::Ui) -> bool,
    /// Remove this component from an entity. Returns true if removed.
    remove_fn: fn(&mut World, Entity) -> bool,
    /// Insert a default instance on an entity (None if T doesn't impl Default).
    insert_default_fn: Option<fn(&mut World, Entity)>,
    /// Collect all entity references from this component on an entity.
    collect_entities_fn: fn(&World, Entity, &mut Vec<Entity>),
    /// Remap all entity references in this component on an entity.
    remap_entities_fn: fn(&mut World, Entity, &mut dyn FnMut(Entity) -> Entity),
    /// Clone this component from src entity to dst entity. None if T is not Clone.
    clone_fn: Option<fn(&mut World, Entity, Entity) -> bool>,
}

/// An independent ECS world containing entities, components, and resources.
///
/// Each World is fully self-contained. Multiple worlds can coexist
/// in the same process, sharing no data between them.
///
/// # Example
///
/// ```
/// use redlilium_ecs::World;
///
/// struct Position { x: f32, y: f32 }
/// struct Velocity { x: f32, y: f32 }
///
/// let mut world = World::new();
/// world.register_component::<Position>();
/// world.register_component::<Velocity>();
///
/// let entity = world.spawn();
/// world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
/// world.insert(entity, Velocity { x: 1.0, y: 0.0 }).unwrap();
///
/// // Query components
/// let positions = world.read::<Position>().unwrap();
/// let velocities = world.read::<Velocity>().unwrap();
/// for (idx, pos) in positions.iter() {
///     if let Some(vel) = velocities.get(idx) {
///         println!("pos ({}, {}) vel ({}, {})", pos.x, pos.y, vel.x, vel.y);
///     }
/// }
/// ```
pub struct World {
    entities: EntityAllocator,
    components: HashMap<TypeId, ComponentStorage>,
    resources: Resources,
    main_thread_resources: MainThreadResources,
    /// Global tick counter for change detection.
    tick: u64,
    /// Inspector metadata for registered component types, keyed by name.
    inspector_entries: BTreeMap<&'static str, InspectorEntry>,
    /// Deferred observer registry and pending triggers.
    observers: Observers,
    /// Monomorphized swap functions for each registered `Triggers<M>` resource.
    trigger_swap_fns: Vec<fn(&mut World)>,
    /// Tracks which entity indices are disabled (have the `Disabled` component).
    /// Updated by hooks on the `Disabled` component. Indexed by entity index.
    pub(crate) disabled_entities: FixedBitSet,
    /// Empty bitset returned for `Disabled` component queries (so they see all entities).
    empty_bitset: FixedBitSet,
}

impl World {
    /// Creates a new empty world.
    pub fn new() -> Self {
        Self {
            entities: EntityAllocator::new(),
            components: HashMap::new(),
            resources: Resources::new(),
            main_thread_resources: MainThreadResources::new(),
            tick: 0,
            inspector_entries: BTreeMap::new(),
            observers: Observers::new(),
            trigger_swap_fns: Vec::new(),
            disabled_entities: FixedBitSet::new(),
            empty_bitset: FixedBitSet::new(),
        }
    }

    // ---- Entity management ----

    /// Spawns a new entity and returns its ID.
    pub fn spawn(&mut self) -> Entity {
        self.entities.allocate()
    }

    /// Despawns an entity, removing all its components.
    ///
    /// Returns `true` if the entity was alive and is now despawned.
    /// Returns `false` if the entity was already dead.
    /// Fires `on_remove` hooks before removal (entity still alive, components still readable).
    /// Records removals for [`removed`](World::removed) filter queries.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }

        let index = entity.index();
        let tick = self.tick;

        // Pass 1: collect on_remove hooks for components this entity has
        let hooks: Vec<ComponentHookFn> = self
            .components
            .values()
            .filter(|s| s.contains_untyped(index))
            .filter_map(|s| s.on_remove)
            .collect();

        // Pass 2: fire hooks (entity still alive, components still readable)
        for hook in hooks {
            hook(self, entity);
        }

        // Collect deferred OnRemove observer triggers before removing components.
        // We need to check which component types have registered remove observers.
        let observer_triggers: Vec<TypeId> = self
            .components
            .iter()
            .filter(|(_, storage)| storage.contains_untyped(index))
            .filter_map(|(type_id, _)| self.observers.remove_trigger_key(type_id))
            .collect();

        // Deallocate entity and remove all components (including any added by hooks)
        self.entities.deallocate(entity);
        for storage in self.components.values_mut() {
            if storage.remove_untyped(index) {
                storage.record_removal(index, tick);
            }
        }

        // Queue deferred observer triggers
        for trigger_key in observer_triggers {
            self.observers.push_trigger(trigger_key, entity);
        }

        true
    }

    /// Returns whether the entity is currently alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Returns the number of alive entities.
    pub fn entity_count(&self) -> u32 {
        self.entities.count()
    }

    /// Iterates over all currently alive entity IDs.
    pub fn iter_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.entities.iter_alive()
    }

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
            .or_insert_with(ComponentStorage::new::<T>);
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
        self.inspector_entries.insert(
            T::NAME,
            InspectorEntry {
                has_fn: |world, entity| world.get::<T>(entity).is_some(),
                inspect_fn: |world, entity, ui| {
                    let Some(comp) = world.get_mut::<T>(entity) else {
                        return false;
                    };
                    comp.inspect_ui(ui);
                    true
                },
                remove_fn: |world, entity| world.remove::<T>(entity).is_some(),
                insert_default_fn: None,
                collect_entities_fn: |world, entity, collector| {
                    if let Some(comp) = world.get::<T>(entity) {
                        comp.collect_entities(collector);
                    }
                },
                remap_entities_fn: |world, entity, map| {
                    if let Some(comp) = world.get_mut::<T>(entity) {
                        comp.remap_entities(map);
                    }
                },
                clone_fn: None,
            },
        );
    }

    /// Registers a component type with full inspector support including "Add Component".
    ///
    /// Like [`register_inspector`](World::register_inspector) but also enables
    /// inserting a default instance via the inspector "Add Component" button.
    pub fn register_inspector_default<T: Component + Default>(&mut self) {
        self.register_component::<T>();
        T::register_required(self);
        self.inspector_entries.insert(
            T::NAME,
            InspectorEntry {
                has_fn: |world, entity| world.get::<T>(entity).is_some(),
                inspect_fn: |world, entity, ui| {
                    let Some(comp) = world.get_mut::<T>(entity) else {
                        return false;
                    };
                    comp.inspect_ui(ui);
                    true
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
                    if let Some(comp) = world.get_mut::<T>(entity) {
                        comp.remap_entities(map);
                    }
                },
                clone_fn: None,
            },
        );
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
            .expect("Component T not registered — call register_component::<T>() first");

        storage.required_components.push(|world, entity| {
            if world.get::<R>(entity).is_none() {
                let _ = world.insert(entity, R::default());
            }
        });

        self
    }

    /// Enables type-erased cloning for a component type.
    ///
    /// After calling this, the component will be included when using
    /// [`clone_entity`](World::clone_entity) or
    /// [`clone_entity_tree`](World::clone_entity_tree).
    ///
    /// The component must have been previously registered via
    /// [`register_inspector`](World::register_inspector) or
    /// [`register_inspector_default`](World::register_inspector_default).
    ///
    /// # Example
    ///
    /// ```ignore
    /// world.register_inspector_default::<Transform>();
    /// world.enable_clone::<Transform>();
    /// ```
    pub fn enable_clone<T: Component + Clone>(&mut self) {
        if let Some(entry) = self.inspector_entries.get_mut(T::NAME) {
            entry.clone_fn = Some(|world, src, dst| {
                let cloned = world.get::<T>(src).cloned();
                if let Some(val) = cloned {
                    let _ = world.insert(dst, val);
                    true
                } else {
                    false
                }
            });
        }
    }

    /// Inserts a component on an entity.
    ///
    /// If the entity already has this component, the value is replaced.
    /// Uses tick 0 (untracked). For change-tracked insertion, use
    /// [`insert_tracked`](World::insert_tracked).
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
                .get(&type_id)
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

        // Perform the actual insert
        self.components
            .get_mut(&type_id)
            .unwrap()
            .typed_mut::<T>()
            .insert(entity.index(), component);

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

    /// Inserts a component with change tracking at the current world tick.
    ///
    /// Like [`insert`](World::insert) but records the current tick for
    /// change detection queries.
    ///
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if `T` has never been registered.
    ///
    /// # Panics
    ///
    /// Panics if the entity is not alive.
    pub fn insert_tracked<T: Send + Sync + 'static>(
        &mut self,
        entity: Entity,
        component: T,
    ) -> Result<(), ComponentNotRegistered> {
        assert!(
            self.entities.is_alive(entity),
            "Cannot insert component on dead entity {entity}"
        );

        let type_id = TypeId::of::<T>();
        let tick = self.tick;

        let (had_component, on_add, on_insert, on_replace, required) = {
            let storage = self
                .components
                .get(&type_id)
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

        if had_component && let Some(hook) = on_replace {
            hook(self, entity);
        }

        self.components
            .get_mut(&type_id)
            .unwrap()
            .typed_mut::<T>()
            .insert_with_tick(entity.index(), component, tick);

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
        self.entities.allocate_many(count)
    }

    /// Spawns `count` entities, each with a clone of the given bundle.
    ///
    /// More efficient than calling [`spawn_with`](World::spawn_with) in a loop
    /// because entity allocation is batched and component storage is pre-reserved.
    ///
    /// # Panics
    ///
    /// Panics if any component type in the bundle has not been registered.
    pub fn spawn_batch_with(&mut self, count: u32, bundle: impl Bundle + Clone) -> Vec<Entity> {
        let entities = self.entities.allocate_many(count);
        for entity in &entities {
            bundle
                .clone()
                .insert_into(self, *entity)
                .expect("Component in bundle not registered");
        }
        entities
    }

    /// Spawns `count` entities, calling `f(index)` to produce each entity's bundle.
    ///
    /// Use this when each entity needs different component data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let entities = world.spawn_batch_with_fn(10, |i| {
    ///     (Position { x: i as f32, y: 0.0 }, Health(100))
    /// });
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if any component type in the bundle has not been registered.
    pub fn spawn_batch_with_fn<B: Bundle>(
        &mut self,
        count: u32,
        f: impl Fn(usize) -> B,
    ) -> Vec<Entity> {
        let entities = self.entities.allocate_many(count);
        for (i, entity) in entities.iter().enumerate() {
            f(i).insert_into(self, *entity)
                .expect("Component in bundle not registered");
        }
        entities
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
    ) -> Result<(), ComponentNotRegistered> {
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
                .get(&type_id)
                .ok_or(ComponentNotRegistered {
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
        self.components
            .get_mut(&type_id)
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
                    .components
                    .get(&type_id)
                    .unwrap()
                    .contains_untyped(entity.index());

                if had && let Some(hook) = on_replace {
                    hook(self, *entity);
                }

                self.components
                    .get_mut(&type_id)
                    .unwrap()
                    .typed_mut::<T>()
                    .insert(entity.index(), component);

                if !had && let Some(hook) = on_add {
                    hook(self, *entity);
                }
                if let Some(hook) = on_insert {
                    hook(self, *entity);
                }

                // Apply required components (only on first add)
                if !had {
                    let required = self
                        .components
                        .get(&type_id)
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
            let storage = self.components.get_mut(&type_id).unwrap();
            let set = storage.typed_mut::<T>();
            for (entity, component) in entities.iter().zip(components) {
                assert!(
                    self.entities.is_alive(*entity),
                    "Cannot insert component on dead entity {entity}"
                );
                let had = set.contains(entity.index());
                set.insert(entity.index(), component);

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
    ) -> Result<(), ComponentNotRegistered> {
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
                .get(&type_id)
                .ok_or(ComponentNotRegistered {
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
        self.components
            .get_mut(&type_id)
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
                    .components
                    .get(&type_id)
                    .unwrap()
                    .contains_untyped(entity.index());

                if had && let Some(hook) = on_replace {
                    hook(self, *entity);
                }

                self.components
                    .get_mut(&type_id)
                    .unwrap()
                    .typed_mut::<T>()
                    .insert_with_tick(entity.index(), component, tick);

                if !had && let Some(hook) = on_add {
                    hook(self, *entity);
                }
                if let Some(hook) = on_insert {
                    hook(self, *entity);
                }

                // Apply required components (only on first add)
                if !had {
                    let required = self
                        .components
                        .get(&type_id)
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
            let storage = self.components.get_mut(&type_id).unwrap();
            let set = storage.typed_mut::<T>();
            for (entity, component) in entities.iter().zip(components) {
                assert!(
                    self.entities.is_alive(*entity),
                    "Cannot insert component on dead entity {entity}"
                );
                let had = set.contains(entity.index());
                set.insert_with_tick(entity.index(), component, tick);

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
            let Some(storage) = self.components.get(&type_id) else {
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
                    self.components
                        .get(&type_id)
                        .is_some_and(|s| s.contains_untyped(e.index()))
                })
                .collect();
            for entity in with_component {
                hook(self, entity);
            }
        }

        // Perform removals
        let Some(storage) = self.components.get_mut(&type_id) else {
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
            let storage = self.components.get(&type_id)?;
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
        let storage = self.components.get_mut(&type_id)?;
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
        let storage = self.components.get(&TypeId::of::<T>())?;
        storage.typed::<T>().get(entity.index())
    }

    /// Returns a mutable reference to a component on an entity.
    pub fn get_mut<T: 'static>(&mut self, entity: Entity) -> Option<&mut T> {
        let storage = self.components.get_mut(&TypeId::of::<T>())?;
        storage.typed_mut::<T>().get_mut(entity.index())
    }

    /// Returns the disabled-entities slice to pass to `Ref`/`RefMut`.
    ///
    /// For the `Disabled` component itself, returns an empty slice so that
    /// `Read<Disabled>` can iterate all disabled entities without self-filtering.
    fn disabled_for<T: 'static>(&self) -> &FixedBitSet {
        if TypeId::of::<T>() == TypeId::of::<Disabled>() {
            &self.empty_bitset
        } else {
            &self.disabled_entities
        }
    }

    /// Returns whether the given entity is currently disabled.
    pub fn is_disabled(&self, entity: Entity) -> bool {
        self.disabled_entities.contains(entity.index() as usize)
    }

    // ---- Query access (runtime borrow-checked, take &self) ----

    /// Gets shared read access to all components of type T.
    ///
    /// Returns a guard that dereferences to [`SparseSetInner<T>`](crate::SparseSetInner),
    /// allowing iteration and lookup.
    ///
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if `T` has never been registered or inserted.
    ///
    /// # Panics
    ///
    /// Panics if T is exclusively borrowed by a [`write`](World::write) call.
    pub fn read<T: 'static>(&self) -> Result<Ref<'_, T>, ComponentNotRegistered> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .ok_or(ComponentNotRegistered {
                type_name: std::any::type_name::<T>(),
            })?;
        Ok(Ref::new(storage, self.disabled_for::<T>()))
    }

    /// Gets shared read access to all components of type T, returning `None`
    /// if the type has never been registered.
    ///
    /// Non-panicking variant of [`read`](World::read). Used by `OptionalRead<T>`.
    pub fn try_read<T: 'static>(&self) -> Option<Ref<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(Ref::new(storage, self.disabled_for::<T>()))
    }

    /// Gets exclusive write access to all components of type T.
    ///
    /// Returns a guard that dereferences to [`SparseSetInner<T>`](crate::SparseSetInner),
    /// allowing iteration, lookup, and mutation.
    ///
    /// # Errors
    ///
    /// Returns [`ComponentNotRegistered`] if `T` has never been registered or inserted.
    ///
    /// # Panics
    ///
    /// Panics if T is borrowed by any [`read`](World::read) or [`write`](World::write) call.
    pub fn write<T: 'static>(&self) -> Result<RefMut<'_, T>, ComponentNotRegistered> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .ok_or(ComponentNotRegistered {
                type_name: std::any::type_name::<T>(),
            })?;
        Ok(RefMut::new(storage, self.disabled_for::<T>()))
    }

    /// Gets exclusive write access to all components of type T, returning `None`
    /// if the type has never been registered.
    ///
    /// Non-panicking variant of [`write`](World::write). Used by `OptionalWrite<T>`.
    pub fn try_write<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new(storage, self.disabled_for::<T>()))
    }

    // ---- Unlocked access (for use when locks are held externally) ----

    /// Gets shared read access without acquiring a lock.
    ///
    /// The caller must ensure the read lock is already held externally.
    pub(crate) fn read_unlocked<T: 'static>(&self) -> Result<Ref<'_, T>, ComponentNotRegistered> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .ok_or(ComponentNotRegistered {
                type_name: std::any::type_name::<T>(),
            })?;
        Ok(Ref::new_unlocked(storage, self.disabled_for::<T>()))
    }

    /// Gets exclusive write access without acquiring a lock.
    ///
    /// The caller must ensure the write lock is already held externally.
    pub(crate) fn write_unlocked<T: 'static>(
        &self,
    ) -> Result<RefMut<'_, T>, ComponentNotRegistered> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .ok_or(ComponentNotRegistered {
                type_name: std::any::type_name::<T>(),
            })?;
        Ok(RefMut::new_unlocked(storage, self.disabled_for::<T>()))
    }

    /// Gets optional shared read access without acquiring a lock.
    pub(crate) fn try_read_unlocked<T: 'static>(&self) -> Option<Ref<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(Ref::new_unlocked(storage, self.disabled_for::<T>()))
    }

    /// Gets optional exclusive write access without acquiring a lock.
    pub(crate) fn try_write_unlocked<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new_unlocked(storage, self.disabled_for::<T>()))
    }

    /// Acquires component locks in TypeId-sorted order.
    ///
    /// Used by `LockRequest` during system execution. Sorted acquisition
    /// prevents deadlocks when multiple systems run concurrently.
    ///
    /// Resources are NOT included — they lock themselves via their own
    /// `Arc<RwLock<T>>` when accessed.
    pub(crate) fn acquire_sorted(&self, infos: &[AccessInfo]) -> Vec<LockGuard<'_>> {
        let mut sorted = infos.to_vec();
        sorted.sort_by_key(|info| info.type_id);
        sorted.dedup_by(|a, b| {
            if a.type_id == b.type_id {
                b.is_write = b.is_write || a.is_write;
                true
            } else {
                false
            }
        });

        sorted
            .iter()
            .filter_map(|info| {
                // Only component storages — resources self-lock via Arc<RwLock<T>>
                let storage = self.components.get(&info.type_id)?;
                let lock = storage.rw_lock();
                Some(if info.is_write {
                    LockGuard::Write(lock.write().unwrap())
                } else {
                    LockGuard::Read(lock.read().unwrap())
                })
            })
            .collect()
    }

    /// Non-blocking variant of [`acquire_sorted`](World::acquire_sorted).
    ///
    /// Returns the human-readable type name for a component TypeId, if registered.
    pub(crate) fn component_type_name(&self, type_id: TypeId) -> Option<&'static str> {
        self.components.get(&type_id).map(|s| s.type_name())
    }

    /// Returns whether a component type has been registered.
    pub fn is_component_registered<T: 'static>(&self) -> bool {
        self.components.contains_key(&TypeId::of::<T>())
    }

    /// Returns the TypeIds of all registered component types.
    pub fn component_type_ids(&self) -> impl Iterator<Item = TypeId> + '_ {
        self.components.keys().copied()
    }

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
            .get_mut(&TypeId::of::<T>())
            .expect("Component not registered")
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
            .get_mut(&TypeId::of::<T>())
            .expect("Component not registered")
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
            .get_mut(&TypeId::of::<T>())
            .expect("Component not registered")
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
            .get_mut(&TypeId::of::<T>())
            .expect("Component not registered")
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
        self.trigger_swap_fns.push(swap_trigger_buffer::<OnAdd<T>>);
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
            .push(swap_trigger_buffer::<OnInsert<T>>);
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
            .push(swap_trigger_buffer::<OnRemove<T>>);
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
    pub fn resource_type_ids(&self) -> impl Iterator<Item = TypeId> + '_ {
        self.resources.type_ids()
    }

    // ---- Filters ----

    /// Creates a `With<T>` filter that checks for component presence.
    ///
    /// Returns a [`ContainsChecker`] that does not borrow component data.
    /// If T has never been registered, the filter matches nothing.
    pub fn with<T: 'static>(&self) -> ContainsChecker<'_> {
        let storage = self.components.get(&TypeId::of::<T>());
        ContainsChecker::with(storage)
    }

    /// Creates a `Without<T>` filter that checks for component absence.
    ///
    /// Returns a [`ContainsChecker`] that does not borrow component data.
    /// If T has never been registered, the filter matches everything.
    pub fn without<T: 'static>(&self) -> ContainsChecker<'_> {
        let storage = self.components.get(&TypeId::of::<T>());
        ContainsChecker::without(storage)
    }

    /// Creates a filter matching entities whose component T was changed
    /// since (strictly after) `since_tick`.
    ///
    /// Does not borrow component data. If T has never been registered,
    /// the filter matches nothing.
    pub fn changed<T: 'static>(&self, since_tick: u64) -> ChangedFilter<'_> {
        let storage = self.components.get(&TypeId::of::<T>());
        ChangedFilter::new(storage, since_tick)
    }

    /// Creates a filter matching entities whose component T was added
    /// since (strictly after) `since_tick`.
    ///
    /// Does not borrow component data. If T has never been registered,
    /// the filter matches nothing.
    pub fn added<T: 'static>(&self, since_tick: u64) -> AddedFilter<'_> {
        let storage = self.components.get(&TypeId::of::<T>());
        AddedFilter::new(storage, since_tick)
    }

    /// Creates a filter matching entities whose component T was removed
    /// since (strictly after) `since_tick`.
    ///
    /// Does not borrow component data. If T has never been registered,
    /// the filter matches nothing.
    ///
    /// Removal records are accumulated across frames. Call
    /// [`clear_removed_tracking`](World::clear_removed_tracking) to reset them.
    pub fn removed<T: 'static>(&self, since_tick: u64) -> RemovedFilter<'_> {
        let storage = self.components.get(&TypeId::of::<T>());
        RemovedFilter::new(storage, since_tick)
    }

    // ---- Inspector ----

    /// Returns the names of all inspector-registered components that an entity has.
    ///
    /// Only includes components registered via [`register_inspector`](World::register_inspector)
    /// or [`register_inspector_default`](World::register_inspector_default).
    pub fn inspectable_components_of(&self, entity: Entity) -> Vec<&'static str> {
        self.inspector_entries
            .iter()
            .filter(|(_, e)| (e.has_fn)(self, entity))
            .map(|(name, _)| *name)
            .collect()
    }

    /// Returns component names that the entity does NOT have and that support Default insertion.
    pub fn addable_components_of(&self, entity: Entity) -> Vec<&'static str> {
        self.inspector_entries
            .iter()
            .filter(|(_, e)| e.insert_default_fn.is_some() && !(e.has_fn)(self, entity))
            .map(|(name, _)| *name)
            .collect()
    }

    /// Renders the inspector UI for a component by name.
    ///
    /// Returns `true` if the entity had the component and it was rendered.
    pub fn inspect_by_name(&mut self, entity: Entity, name: &str, ui: &mut egui::Ui) -> bool {
        // Copy fn pointer out to release the immutable borrow on self
        let inspect_fn = self.inspector_entries.get(name).map(|e| e.inspect_fn);
        if let Some(f) = inspect_fn {
            f(self, entity, ui)
        } else {
            false
        }
    }

    /// Removes a component by name from an entity.
    ///
    /// Returns `true` if the component was removed.
    pub fn remove_by_name(&mut self, entity: Entity, name: &str) -> bool {
        let remove_fn = self.inspector_entries.get(name).map(|e| e.remove_fn);
        if let Some(f) = remove_fn {
            f(self, entity)
        } else {
            false
        }
    }

    /// Inserts a default instance of a component by name on an entity.
    ///
    /// Does nothing if the component was not registered with Default support
    /// or the name is unknown.
    pub fn insert_default_by_name(&mut self, entity: Entity, name: &str) {
        let insert_fn = self
            .inspector_entries
            .get(name)
            .and_then(|e| e.insert_default_fn);
        if let Some(f) = insert_fn {
            f(self, entity);
        }
    }

    /// Collects all entity references from a component by name on an entity.
    ///
    /// Appends referenced entities to `collector`. Does nothing if the
    /// component name is unknown or the entity doesn't have it.
    pub fn collect_entities_by_name(
        &self,
        entity: Entity,
        name: &str,
        collector: &mut Vec<Entity>,
    ) {
        if let Some(entry) = self.inspector_entries.get(name) {
            (entry.collect_entities_fn)(self, entity, collector);
        }
    }

    /// Remaps all entity references in a component by name on an entity.
    ///
    /// Does nothing if the component name is unknown or the entity doesn't have it.
    pub fn remap_entities_by_name(
        &mut self,
        entity: Entity,
        name: &str,
        map: &mut dyn FnMut(Entity) -> Entity,
    ) {
        let remap_fn = self
            .inspector_entries
            .get(name)
            .map(|e| e.remap_entities_fn);
        if let Some(f) = remap_fn {
            f(self, entity, map);
        }
    }

    /// Collects all entity references from all registered components on an entity.
    ///
    /// Iterates every inspector-registered component type and appends any
    /// entity references found to `collector`.
    pub fn collect_all_entities(&self, entity: Entity, collector: &mut Vec<Entity>) {
        for entry in self.inspector_entries.values() {
            (entry.collect_entities_fn)(self, entity, collector);
        }
    }

    /// Remaps all entity references in all registered components on an entity.
    ///
    /// Iterates every inspector-registered component type and remaps any
    /// entity references found using the provided mapping function.
    pub fn remap_all_entities(&mut self, entity: Entity, map: &mut dyn FnMut(Entity) -> Entity) {
        let fns: Vec<_> = self
            .inspector_entries
            .values()
            .map(|e| e.remap_entities_fn)
            .collect();
        for f in fns {
            f(self, entity, map);
        }
    }

    /// Clones all clone-enabled components from one entity to a new entity.
    ///
    /// Spawns a new entity and copies every component that was registered
    /// with [`enable_clone`](World::enable_clone). Components without clone
    /// support are silently skipped.
    ///
    /// Does **not** traverse the hierarchy or remap entity references.
    /// For hierarchy-aware cloning, use [`clone_entity_tree`](World::clone_entity_tree).
    ///
    /// Returns `None` if the source entity is not alive.
    pub fn clone_entity(&mut self, src: Entity) -> Option<Entity> {
        if !self.is_alive(src) {
            return None;
        }
        let dst = self.spawn();

        let clone_fns: Vec<_> = self
            .inspector_entries
            .values()
            .filter_map(|e| {
                if (e.has_fn)(self, src) {
                    e.clone_fn
                } else {
                    None
                }
            })
            .collect();

        for f in clone_fns {
            f(self, src, dst);
        }

        Some(dst)
    }

    /// Clones an entity subtree, remapping all internal entity references.
    ///
    /// Performs a breadth-first walk from `root` through [`Children`](crate::Children)
    /// components, clones all clone-enabled components on every entity in the
    /// subtree, then remaps all entity references (via [`remap_all_entities`])
    /// so that internal references point to the new cloned entities.
    ///
    /// Entity references that point **outside** the subtree are left unchanged.
    ///
    /// Returns a mapping from old entity IDs to new entity IDs.
    /// The cloned root is `mapping[&root]`.
    ///
    /// Returns an empty map if `root` is not alive.
    pub fn clone_entity_tree(&mut self, root: Entity) -> HashMap<Entity, Entity> {
        if !self.is_alive(root) {
            return HashMap::new();
        }

        // 1. Collect all entities in subtree via Children (BFS)
        let mut old_entities = vec![root];
        let mut i = 0;
        while i < old_entities.len() {
            let entity = old_entities[i];
            if let Some(children) = self.get::<crate::Children>(entity) {
                old_entities.extend(children.0.iter().copied());
            }
            i += 1;
        }

        // 2. Spawn new entities and build old→new mapping
        let mut mapping = HashMap::with_capacity(old_entities.len());
        for &old in &old_entities {
            let new = self.spawn();
            mapping.insert(old, new);
        }

        // 3. Clone all components from each old entity to corresponding new entity
        let clone_fns: Vec<_> = self
            .inspector_entries
            .values()
            .filter_map(|e| e.clone_fn.map(|clone| (e.has_fn, clone)))
            .collect();

        for (&old, &new) in &mapping {
            for &(has_fn, clone_fn) in &clone_fns {
                if has_fn(self, old) {
                    clone_fn(self, old, new);
                }
            }
        }

        // 4. Remap all entity references in new entities
        let remap_fns: Vec<_> = self
            .inspector_entries
            .values()
            .map(|e| e.remap_entities_fn)
            .collect();

        for &new in mapping.values() {
            for &f in &remap_fns {
                f(self, new, &mut |e| *mapping.get(&e).unwrap_or(&e));
            }
        }

        mapping
    }

    // ---- Resource management ----

    /// Inserts or replaces a resource, wrapping it in `Arc<RwLock<T>>`.
    ///
    /// Returns the typed `Arc` handle for external access (e.g. inspector,
    /// editor). The world stores a coerced `Arc<RwLock<dyn Resource>>` that
    /// shares the same underlying data and lock.
    pub fn insert_resource<T: Resource>(&mut self, value: T) -> Arc<RwLock<T>> {
        self.resources.insert(value)
    }

    /// Inserts a pre-existing `Arc<RwLock<T>>` as a resource.
    ///
    /// The Arc is coerced to `Arc<RwLock<dyn Resource>>` for storage;
    /// both the caller's clone and the stored clone share the same lock.
    pub fn insert_resource_shared<T: Resource>(&mut self, resource: Arc<RwLock<T>>) {
        self.resources.insert_shared(resource);
    }

    /// Removes a resource, returning the `Arc<RwLock<dyn Resource>>` if present.
    pub fn remove_resource<T: 'static>(&mut self) -> Option<Arc<RwLock<dyn Resource>>> {
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
    pub fn resource_handle<T: 'static>(&self) -> Arc<RwLock<dyn Resource>> {
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
        unsafe { self.main_thread_resources.insert(value) }
    }

    /// Returns whether a main-thread resource of type `T` exists.
    pub fn has_main_thread_resource<T: 'static>(&self) -> bool {
        // SAFETY: contains() only reads HashMap keys (TypeId), no data access.
        unsafe { self.main_thread_resources.contains::<T>() }
    }

    /// Removes a main-thread resource and returns it, or `None` if absent.
    ///
    /// Takes `&mut self`, so it can only be called outside system execution.
    pub fn remove_main_thread_resource<T: 'static>(&mut self) -> Option<T> {
        // SAFETY: &mut self guarantees exclusive access.
        unsafe { self.main_thread_resources.remove::<T>() }
    }

    /// Borrows a main-thread resource immutably.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread.
    pub(crate) unsafe fn main_thread_resource<T: 'static>(&self) -> &T {
        unsafe { self.main_thread_resources.borrow::<T>() }
    }

    /// Borrows a main-thread resource mutably.
    ///
    /// # Safety
    ///
    /// Caller must be on the main thread. No other borrows to this resource
    /// may be active.
    #[allow(clippy::mut_from_ref)] // SAFETY: caller ensures exclusive main-thread access
    pub(crate) unsafe fn main_thread_resource_mut<T: 'static>(&self) -> &mut T {
        unsafe { self.main_thread_resources.borrow_mut::<T>() }
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
        for storage in self.components.values_mut() {
            storage.clear_removed();
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

/// Swaps the trigger buffer for marker type `M`.
///
/// Used as a monomorphized function pointer stored in `World::trigger_swap_fns`.
fn swap_trigger_buffer<M: 'static>(world: &mut World) {
    world.resource_mut::<Triggers<M>>().swap();
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Velocity {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Health(u32);

    struct Frozen;

    #[test]
    fn spawn_and_check_alive() {
        let mut world = World::new();
        let entity = world.spawn();
        assert!(world.is_alive(entity));
        assert_eq!(world.entity_count(), 1);
    }

    #[test]
    fn despawn_removes_entity() {
        let mut world = World::new();
        let entity = world.spawn();
        assert!(world.despawn(entity));
        assert!(!world.is_alive(entity));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn despawn_dead_entity_returns_false() {
        let mut world = World::new();
        let entity = world.spawn();
        world.despawn(entity);
        assert!(!world.despawn(entity));
    }

    #[test]
    fn insert_and_get_component() {
        let mut world = World::new();
        world.register_component::<Position>();
        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 1.0, y: 2.0 })
        );
    }

    #[test]
    fn insert_unregistered_returns_err() {
        let mut world = World::new();
        let entity = world.spawn();
        let result = world.insert(entity, Position { x: 0.0, y: 0.0 });
        assert!(result.is_err());
        assert!(result.unwrap_err().type_name.contains("Position"));
    }

    #[test]
    fn read_unregistered_returns_err() {
        let world = World::new();
        assert!(world.read::<Position>().is_err());
    }

    #[test]
    fn write_unregistered_returns_err() {
        let world = World::new();
        assert!(world.write::<Position>().is_err());
    }

    #[test]
    #[should_panic(expected = "Cannot insert component on dead entity")]
    fn insert_on_dead_entity_panics() {
        let mut world = World::new();
        world.register_component::<Position>();
        let entity = world.spawn();
        world.despawn(entity);
        let _ = world.insert(entity, Position { x: 0.0, y: 0.0 });
    }

    #[test]
    fn remove_component() {
        let mut world = World::new();
        world.register_component::<Health>();
        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        assert_eq!(world.remove::<Health>(entity), Some(Health(100)));
        assert!(world.get::<Health>(entity).is_none());
    }

    #[test]
    fn despawn_removes_all_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();
        let entity = world.spawn();
        world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
        world.insert(entity, Health(100)).unwrap();

        world.despawn(entity);

        // Spawn a new entity that reuses the same index
        let new_entity = world.spawn();
        assert_eq!(new_entity.index(), entity.index());

        // New entity should not have old components
        assert!(world.get::<Position>(new_entity).is_none());
        assert!(world.get::<Health>(new_entity).is_none());
    }

    #[test]
    fn read_query_iterates_all() {
        let mut world = World::new();
        world.register_component::<Position>();
        for i in 0..3 {
            let e = world.spawn();
            world
                .insert(
                    e,
                    Position {
                        x: i as f32,
                        y: 0.0,
                    },
                )
                .unwrap();
        }

        let positions = world.read::<Position>().unwrap();
        assert_eq!(positions.len(), 3);

        let xs: Vec<f32> = positions.iter().map(|(_, p)| p.x).collect();
        assert!(xs.contains(&0.0));
        assert!(xs.contains(&1.0));
        assert!(xs.contains(&2.0));
    }

    #[test]
    fn write_query_allows_mutation() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0, y: 2.0 }).unwrap();

        {
            let mut positions = world.write::<Position>().unwrap();
            for (_, pos) in positions.iter_mut() {
                pos.x += 10.0;
            }
        }

        assert_eq!(
            world.get::<Position>(e),
            Some(&Position { x: 11.0, y: 2.0 })
        );
    }

    #[test]
    fn double_read_succeeds() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0, y: 0.0 }).unwrap();

        let _a = world.read::<Position>().unwrap();
        let _b = world.read::<Position>().unwrap();
    }

    #[test]
    #[should_panic(expected = "already borrowed")]
    fn read_write_conflict_panics() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0, y: 0.0 }).unwrap();

        let _r = world.read::<Position>().unwrap();
        let _w = world.write::<Position>().unwrap();
    }

    #[test]
    fn resource_insert_and_get() {
        let mut world = World::new();
        world.insert_resource(42u32);

        let val = world.resource::<u32>();
        assert_eq!(*val, 42);
    }

    #[test]
    fn resource_mut_modify() {
        let mut world = World::new();
        world.insert_resource(42u32);

        {
            let mut val = world.resource_mut::<u32>();
            *val = 99;
        }

        let val = world.resource::<u32>();
        assert_eq!(*val, 99);
    }

    #[test]
    fn entity_recycling_invalidates_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        let old = world.spawn();
        world.insert(old, Position { x: 1.0, y: 2.0 }).unwrap();

        world.despawn(old);
        let new = world.spawn();

        // Same index, different generation
        assert_eq!(new.index(), old.index());
        assert_ne!(new.generation(), old.generation());

        // New entity should not have old entity's components
        assert!(world.get::<Position>(new).is_none());
    }

    #[test]
    fn with_filter_in_query() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();

        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0, y: 0.0 }).unwrap();
        world.insert(e1, Health(100)).unwrap();

        let e2 = world.spawn();
        world.insert(e2, Position { x: 2.0, y: 0.0 }).unwrap();

        let positions = world.read::<Position>().unwrap();
        let has_health = world.with::<Health>();

        let healthy_positions: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| has_health.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();

        assert_eq!(healthy_positions, vec![1.0]);
    }

    #[test]
    fn without_filter_in_query() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Frozen>();

        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0, y: 0.0 }).unwrap();
        world.insert(e1, Frozen).unwrap();

        let e2 = world.spawn();
        world.insert(e2, Position { x: 2.0, y: 0.0 }).unwrap();

        let positions = world.read::<Position>().unwrap();
        let not_frozen = world.without::<Frozen>();

        let unfrozen_positions: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| not_frozen.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();

        assert_eq!(unfrozen_positions, vec![2.0]);
    }

    #[test]
    fn combined_read_iteration() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        let e1 = world.spawn();
        world.insert(e1, Position { x: 0.0, y: 0.0 }).unwrap();
        world.insert(e1, Velocity { x: 1.0, y: 0.0 }).unwrap();

        let e2 = world.spawn();
        world.insert(e2, Position { x: 5.0, y: 5.0 }).unwrap();
        // e2 has no Velocity

        let positions = world.read::<Position>().unwrap();
        let velocities = world.read::<Velocity>().unwrap();

        let mut count = 0;
        for (idx, _pos) in positions.iter() {
            if velocities.get(idx).is_some() {
                count += 1;
            }
        }
        assert_eq!(count, 1); // Only e1 has both
    }

    #[test]
    fn removed_filter_after_remove() {
        let mut world = World::new();
        world.register_component::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        world.advance_tick(); // tick = 1
        let before_remove = world.current_tick();

        world.advance_tick(); // tick = 2
        world.remove::<Health>(entity);

        let removed = world.removed::<Health>(before_remove);
        assert!(removed.matches(entity.index()));
    }

    #[test]
    fn removed_filter_not_matching_before_tick() {
        let mut world = World::new();
        world.register_component::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        world.advance_tick(); // tick = 1
        world.remove::<Health>(entity); // removed at tick 1

        // Query with since_tick = 1, removal at tick 1 is NOT strictly after 1
        let removed = world.removed::<Health>(1);
        assert!(!removed.matches(entity.index()));

        // Query with since_tick = 0, removal at tick 1 IS strictly after 0
        let removed = world.removed::<Health>(0);
        assert!(removed.matches(entity.index()));
    }

    #[test]
    fn removed_filter_after_despawn() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();

        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();
        world.insert(entity, Health(100)).unwrap();

        world.advance_tick(); // tick = 1
        world.despawn(entity);

        // Both components should be tracked as removed
        let removed_pos = world.removed::<Position>(0);
        let removed_health = world.removed::<Health>(0);
        assert!(removed_pos.matches(entity.index()));
        assert!(removed_health.matches(entity.index()));
    }

    #[test]
    fn removed_filter_iter() {
        let mut world = World::new();
        world.register_component::<Health>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        let e3 = world.spawn();
        world.insert(e1, Health(100)).unwrap();
        world.insert(e2, Health(200)).unwrap();
        world.insert(e3, Health(300)).unwrap();

        world.advance_tick(); // tick = 1
        world.remove::<Health>(e1);
        world.remove::<Health>(e3);

        let removed = world.removed::<Health>(0);
        let mut entities: Vec<u32> = removed.iter().collect();
        entities.sort();
        assert_eq!(entities, vec![e1.index(), e3.index()]);
    }

    #[test]
    fn clear_removed_tracking_works() {
        let mut world = World::new();
        world.register_component::<Health>();

        let entity = world.spawn();
        world.insert(entity, Health(100)).unwrap();

        world.advance_tick(); // tick = 1
        world.remove::<Health>(entity);

        assert!(world.removed::<Health>(0).matches(entity.index()));

        world.clear_removed_tracking();

        assert!(!world.removed::<Health>(0).matches(entity.index()));
    }

    #[test]
    fn removed_filter_unregistered_matches_nothing() {
        let world = World::new();
        let removed = world.removed::<Health>(0);
        assert!(!removed.matches(0));
        assert_eq!(removed.iter().count(), 0);
    }

    #[test]
    fn remove_nonexistent_component_not_tracked() {
        let mut world = World::new();
        world.register_component::<Health>();

        let entity = world.spawn();
        // Don't insert Health, just try to remove it
        world.advance_tick();
        world.remove::<Health>(entity);

        let removed = world.removed::<Health>(0);
        assert!(!removed.matches(entity.index()));
    }

    // ---- Batch operation tests ----

    #[test]
    fn spawn_batch_creates_entities() {
        let mut world = World::new();
        let entities = world.spawn_batch(5);

        assert_eq!(entities.len(), 5);
        assert_eq!(world.entity_count(), 5);
        for e in &entities {
            assert!(world.is_alive(*e));
        }
    }

    #[test]
    fn spawn_batch_zero() {
        let mut world = World::new();
        let entities = world.spawn_batch(0);
        assert!(entities.is_empty());
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn spawn_batch_with_inserts_components() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();

        let entities = world.spawn_batch_with(3, (Position { x: 1.0, y: 2.0 }, Health(100)));

        assert_eq!(entities.len(), 3);
        for e in &entities {
            assert_eq!(
                world.get::<Position>(*e),
                Some(&Position { x: 1.0, y: 2.0 })
            );
            assert_eq!(world.get::<Health>(*e), Some(&Health(100)));
        }
    }

    #[test]
    fn spawn_batch_with_fn_unique_data() {
        let mut world = World::new();
        world.register_component::<Position>();

        let entities = world.spawn_batch_with_fn(4, |i| {
            (Position {
                x: i as f32,
                y: (i * 10) as f32,
            },)
        });

        assert_eq!(entities.len(), 4);
        for (i, e) in entities.iter().enumerate() {
            assert_eq!(
                world.get::<Position>(*e),
                Some(&Position {
                    x: i as f32,
                    y: (i * 10) as f32
                })
            );
        }
    }

    #[test]
    fn despawn_batch_removes_all() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();

        let entities = world.spawn_batch(4);
        for e in &entities {
            world.insert(*e, Position { x: 0.0, y: 0.0 }).unwrap();
            world.insert(*e, Health(50)).unwrap();
        }

        world.advance_tick(); // tick = 1
        world.despawn_batch(&entities);

        assert_eq!(world.entity_count(), 0);
        for e in &entities {
            assert!(!world.is_alive(*e));
        }

        // Removal tracking should work
        for e in &entities {
            assert!(world.removed::<Position>(0).matches(e.index()));
            assert!(world.removed::<Health>(0).matches(e.index()));
        }
    }

    #[test]
    fn despawn_batch_skips_dead() {
        let mut world = World::new();
        let entities = world.spawn_batch(3);
        world.despawn(entities[1]); // pre-despawn one

        world.despawn_batch(&entities); // should not panic on already dead entity[1]
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn insert_batch_adds_components() {
        let mut world = World::new();
        world.register_component::<Health>();

        let entities = world.spawn_batch(3);
        let healths = vec![Health(10), Health(20), Health(30)];

        world.insert_batch(&entities, healths).unwrap();

        assert_eq!(world.get::<Health>(entities[0]), Some(&Health(10)));
        assert_eq!(world.get::<Health>(entities[1]), Some(&Health(20)));
        assert_eq!(world.get::<Health>(entities[2]), Some(&Health(30)));
    }

    #[test]
    fn insert_batch_tracked_records_tick() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.advance_tick(); // tick = 1

        let entities = world.spawn_batch(2);
        let healths = vec![Health(10), Health(20)];

        world.insert_batch_tracked(&entities, healths).unwrap();

        assert!(world.added::<Health>(0).matches(entities[0].index()));
        assert!(world.added::<Health>(0).matches(entities[1].index()));
    }

    #[test]
    #[should_panic(expected = "entities and components must have the same length")]
    fn insert_batch_mismatched_lengths_panics() {
        let mut world = World::new();
        world.register_component::<Health>();

        let entities = world.spawn_batch(2);
        let healths = vec![Health(10)];

        let _ = world.insert_batch(&entities, healths);
    }

    #[test]
    fn insert_batch_unregistered_returns_err() {
        let mut world = World::new();
        let entities = world.spawn_batch(1);
        let result = world.insert_batch(&entities, vec![Health(10)]);
        assert!(result.is_err());
    }

    #[test]
    fn remove_batch_removes_components() {
        let mut world = World::new();
        world.register_component::<Health>();

        let entities = world.spawn_batch(3);
        for (i, e) in entities.iter().enumerate() {
            world.insert(*e, Health(i as u32 * 10)).unwrap();
        }

        world.advance_tick();
        world.remove_batch::<Health>(&entities[0..2]);

        assert!(world.get::<Health>(entities[0]).is_none());
        assert!(world.get::<Health>(entities[1]).is_none());
        assert_eq!(world.get::<Health>(entities[2]), Some(&Health(20)));

        // Removal tracking
        assert!(world.removed::<Health>(0).matches(entities[0].index()));
        assert!(world.removed::<Health>(0).matches(entities[1].index()));
        assert!(!world.removed::<Health>(0).matches(entities[2].index()));
    }

    #[test]
    fn remove_batch_unregistered_no_panic() {
        let mut world = World::new();
        let entities = world.spawn_batch(2);
        // Should not panic when component type is not registered
        world.remove_batch::<Health>(&entities);
    }

    // ---- Lifecycle hook tests ----

    #[derive(Debug, Clone, PartialEq)]
    struct Marker(u32);

    #[test]
    fn on_add_fires_on_first_insert() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();
        world.set_on_add::<Position>(|world, entity| {
            let _ = world.insert(entity, Marker(1));
        });

        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();

        // Marker should have been added by on_add hook
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));
    }

    #[test]
    fn on_add_does_not_fire_on_replace() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();
        world.set_on_add::<Position>(|world, entity| {
            let _ = world.insert(entity, Marker(1));
        });

        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));

        // Remove marker, then replace position — on_add should NOT fire
        world.remove::<Marker>(entity);
        world.insert(entity, Position { x: 3.0, y: 4.0 }).unwrap();
        assert!(world.get::<Marker>(entity).is_none());
    }

    #[test]
    fn on_insert_fires_on_every_insert() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();
        world.set_on_insert::<Position>(|world, entity| {
            let count = world.get::<Marker>(entity).map(|m| m.0).unwrap_or(0);
            let _ = world.insert(entity, Marker(count + 1));
        });

        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 0.0 }).unwrap();
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));

        world.insert(entity, Position { x: 2.0, y: 0.0 }).unwrap();
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(2)));
    }

    #[test]
    fn on_replace_fires_before_overwrite() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();
        world.set_on_replace::<Position>(|world, entity| {
            // Read old value and store it in Marker
            if let Some(pos) = world.get::<Position>(entity) {
                let _ = world.insert(entity, Marker(pos.x as u32));
            }
        });

        let entity = world.spawn();
        world.insert(entity, Position { x: 10.0, y: 0.0 }).unwrap();
        // on_replace should NOT fire on first insert
        assert!(world.get::<Marker>(entity).is_none());

        world.insert(entity, Position { x: 20.0, y: 0.0 }).unwrap();
        // Hook read old value x=10
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(10)));

        world.insert(entity, Position { x: 30.0, y: 0.0 }).unwrap();
        // Hook read old value x=20
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(20)));
    }

    #[test]
    fn on_remove_fires_before_removal() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();
        world.set_on_remove::<Position>(|world, entity| {
            // Read component before it's removed
            if let Some(pos) = world.get::<Position>(entity) {
                let _ = world.insert(entity, Marker(pos.x as u32));
            }
        });

        let entity = world.spawn();
        world.insert(entity, Position { x: 42.0, y: 0.0 }).unwrap();
        world.remove::<Position>(entity);

        // Hook stored Position.x in Marker before removal
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(42)));
        assert!(world.get::<Position>(entity).is_none());
    }

    #[test]
    fn on_remove_fires_during_despawn() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.insert_resource(0u32);

        world.set_on_remove::<Position>(|world, entity| {
            if let Some(pos) = world.get::<Position>(entity) {
                let mut counter = world.resource_mut::<u32>();
                *counter = pos.x as u32;
            }
        });

        let entity = world.spawn();
        world.insert(entity, Position { x: 77.0, y: 0.0 }).unwrap();
        world.despawn(entity);

        let val = world.resource::<u32>();
        assert_eq!(*val, 77);
    }

    #[test]
    fn on_remove_fires_during_despawn_batch() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.insert_resource(0u32);

        world.set_on_remove::<Health>(|world, _entity| {
            let mut counter = world.resource_mut::<u32>();
            *counter += 1;
        });

        let entities = world.spawn_batch(3);
        for e in &entities {
            world.insert(*e, Health(10)).unwrap();
        }
        world.despawn_batch(&entities);

        let count = world.resource::<u32>();
        assert_eq!(*count, 3);
    }

    #[test]
    fn on_remove_entity_still_alive_during_despawn() {
        let mut world = World::new();
        world.register_component::<Health>();
        world.insert_resource(false);

        world.set_on_remove::<Health>(|world, entity| {
            let mut was_alive = world.resource_mut::<bool>();
            *was_alive = world.is_alive(entity);
        });

        let entity = world.spawn();
        world.insert(entity, Health(1)).unwrap();
        world.despawn(entity);

        let was_alive = world.resource::<bool>();
        assert!(*was_alive);
    }

    #[test]
    fn hooks_fire_during_insert_batch() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();

        world.set_on_add::<Position>(|world, entity| {
            let _ = world.insert(entity, Marker(1));
        });

        let entities = world.spawn_batch(3);
        let positions = vec![
            Position { x: 1.0, y: 0.0 },
            Position { x: 2.0, y: 0.0 },
            Position { x: 3.0, y: 0.0 },
        ];
        world.insert_batch(&entities, positions).unwrap();

        for e in &entities {
            assert_eq!(world.get::<Marker>(*e), Some(&Marker(1)));
        }
    }

    #[test]
    fn hooks_fire_during_insert_batch_tracked() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();
        world.advance_tick(); // tick = 1

        world.set_on_add::<Position>(|world, entity| {
            let _ = world.insert(entity, Marker(1));
        });

        let entities = world.spawn_batch(2);
        let positions = vec![Position { x: 1.0, y: 0.0 }, Position { x: 2.0, y: 0.0 }];
        world.insert_batch_tracked(&entities, positions).unwrap();

        for e in &entities {
            assert_eq!(world.get::<Marker>(*e), Some(&Marker(1)));
        }
        // Verify tick tracking still works
        assert!(world.added::<Position>(0).matches(entities[0].index()));
    }

    #[test]
    fn hooks_fire_during_remove_batch() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();

        world.set_on_remove::<Position>(|world, entity| {
            if let Some(pos) = world.get::<Position>(entity) {
                let _ = world.insert(entity, Marker(pos.x as u32));
            }
        });

        let entities = world.spawn_batch(3);
        for (i, e) in entities.iter().enumerate() {
            world
                .insert(
                    *e,
                    Position {
                        x: (i + 1) as f32,
                        y: 0.0,
                    },
                )
                .unwrap();
        }
        world.remove_batch::<Position>(&entities);

        assert_eq!(world.get::<Marker>(entities[0]), Some(&Marker(1)));
        assert_eq!(world.get::<Marker>(entities[1]), Some(&Marker(2)));
        assert_eq!(world.get::<Marker>(entities[2]), Some(&Marker(3)));
    }

    #[test]
    fn on_add_required_component_pattern() {
        // Classic use case: inserting A automatically inserts B
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();

        world.set_on_add::<Position>(|world, entity| {
            if world.get::<Velocity>(entity).is_none() {
                let _ = world.insert(entity, Velocity { x: 0.0, y: 0.0 });
            }
        });

        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 2.0 }).unwrap();

        assert_eq!(
            world.get::<Velocity>(entity),
            Some(&Velocity { x: 0.0, y: 0.0 })
        );
    }

    #[test]
    fn multiple_hooks_on_same_component() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();
        world.insert_resource(0u32);

        world.set_on_add::<Position>(|world, entity| {
            let _ = world.insert(entity, Marker(1));
        });
        world.set_on_insert::<Position>(|world, _entity| {
            let mut counter = world.resource_mut::<u32>();
            *counter += 1;
        });

        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 0.0 }).unwrap();

        // Both hooks should have fired
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(1)));
        assert_eq!(*world.resource::<u32>(), 1);

        // Replace — only on_insert fires, not on_add
        world.remove::<Marker>(entity);
        world.insert(entity, Position { x: 2.0, y: 0.0 }).unwrap();

        assert!(world.get::<Marker>(entity).is_none());
        assert_eq!(*world.resource::<u32>(), 2);
    }

    #[test]
    fn hooks_via_commands() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Marker>();
        world.init_commands();

        world.set_on_add::<Position>(|world, entity| {
            let _ = world.insert(entity, Marker(99));
        });

        // Queue insertion via command buffer
        let entity = world.spawn();
        {
            let cmds = world.resource::<CommandBuffer>();
            cmds.push(move |world: &mut World| {
                let _ = world.insert(entity, Position { x: 1.0, y: 0.0 });
            });
        }
        world.apply_commands();

        // Hook should have fired when command was applied
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(99)));
    }

    #[test]
    fn no_hooks_batch_fast_path() {
        // Ensure batch operations still work efficiently without hooks
        let mut world = World::new();
        world.register_component::<Health>();

        let entities = world.spawn_batch(100);
        let healths: Vec<Health> = (0..100).map(Health).collect();
        world.insert_batch(&entities, healths).unwrap();

        for (i, e) in entities.iter().enumerate() {
            assert_eq!(world.get::<Health>(*e), Some(&Health(i as u32)));
        }
    }

    #[test]
    fn despawn_multiple_components_hooks() {
        // Despawn fires on_remove for each component type
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();
        world.insert_resource(0u32);

        world.set_on_remove::<Position>(|world, _entity| {
            let mut c = world.resource_mut::<u32>();
            *c += 10;
        });
        world.set_on_remove::<Health>(|world, _entity| {
            let mut c = world.resource_mut::<u32>();
            *c += 1;
        });

        let entity = world.spawn();
        world.insert(entity, Position { x: 0.0, y: 0.0 }).unwrap();
        world.insert(entity, Health(100)).unwrap();
        world.despawn(entity);

        // Both hooks should have fired
        assert_eq!(*world.resource::<u32>(), 11);
    }

    // --- Required components tests ---

    #[derive(Debug, Clone, PartialEq, Default)]
    struct ReqA(u32);
    #[derive(Debug, Clone, PartialEq, Default)]
    struct ReqB(u32);
    #[derive(Debug, Clone, PartialEq, Default)]
    struct ReqC(u32);

    #[test]
    fn required_component_inserted_automatically() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        world.register_required::<ReqA, ReqB>();

        let entity = world.spawn();
        world.insert(entity, ReqA(1)).unwrap();

        assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
    }

    #[test]
    fn required_component_not_overwritten() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        world.register_component::<ReqB>();
        world.register_required::<ReqA, ReqB>();

        let entity = world.spawn();
        world.insert(entity, ReqB(42)).unwrap();
        world.insert(entity, ReqA(1)).unwrap();

        // Existing ReqB should NOT be overwritten
        assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(42)));
    }

    #[test]
    fn required_component_not_applied_on_replace() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        world.register_required::<ReqA, ReqB>();

        let entity = world.spawn();
        world.insert(entity, ReqA(1)).unwrap();
        assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));

        // Remove ReqB, then replace ReqA — requirements should NOT fire again
        world.remove::<ReqB>(entity);
        world.insert(entity, ReqA(2)).unwrap();
        assert!(world.get::<ReqB>(entity).is_none());
    }

    #[test]
    fn transitive_requirements() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        world.register_required::<ReqA, ReqB>();
        world.register_required::<ReqB, ReqC>();

        let entity = world.spawn();
        world.insert(entity, ReqA(1)).unwrap();

        assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
        assert_eq!(world.get::<ReqC>(entity), Some(&ReqC(0)));
    }

    #[test]
    fn required_components_coexist_with_on_add_hook() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        world.register_component::<Marker>();
        world.register_required::<ReqA, ReqB>();

        world.set_on_add::<ReqA>(|world, entity| {
            let _ = world.insert(entity, Marker(99));
        });

        let entity = world.spawn();
        world.insert(entity, ReqA(1)).unwrap();

        assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
        assert_eq!(world.get::<Marker>(entity), Some(&Marker(99)));
    }

    #[test]
    fn required_component_auto_registers() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        // Don't manually register ReqB — register_required should do it
        world.register_required::<ReqA, ReqB>();

        let entity = world.spawn();
        world.insert(entity, ReqA(1)).unwrap();
        assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
    }

    #[test]
    fn required_components_in_batch_insert() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        world.register_required::<ReqA, ReqB>();

        let entities = world.spawn_batch(3);
        let components = vec![ReqA(1), ReqA(2), ReqA(3)];
        world.insert_batch(&entities, components).unwrap();

        for e in &entities {
            assert_eq!(world.get::<ReqB>(*e), Some(&ReqB(0)));
        }
    }

    #[test]
    fn required_components_in_batch_tracked() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        world.register_required::<ReqA, ReqB>();

        let entities = world.spawn_batch(2);
        let components = vec![ReqA(1), ReqA(2)];
        world.insert_batch_tracked(&entities, components).unwrap();

        for e in &entities {
            assert_eq!(world.get::<ReqB>(*e), Some(&ReqB(0)));
        }
    }

    #[test]
    fn required_components_via_bundle() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        world.register_component::<Health>();
        world.register_required::<ReqA, ReqB>();

        let entity = world.spawn_with((ReqA(1), Health(100)));
        assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
    }

    #[test]
    fn multiple_required_components() {
        let mut world = World::new();
        world.register_component::<ReqA>();
        world.register_required::<ReqA, ReqB>();
        world.register_required::<ReqA, ReqC>();

        let entity = world.spawn();
        world.insert(entity, ReqA(1)).unwrap();

        assert_eq!(world.get::<ReqB>(entity), Some(&ReqB(0)));
        assert_eq!(world.get::<ReqC>(entity), Some(&ReqC(0)));
    }

    // --- Entity collect/remap tests ---

    #[test]
    fn collect_entities_from_parent() {
        use crate::components::{Children, Parent};

        let mut world = World::new();
        world.register_inspector::<Parent>();
        world.register_inspector_default::<Children>();

        let parent = world.spawn();
        let child = world.spawn();
        world.insert(child, Parent(parent)).unwrap();

        let mut collected = Vec::new();
        world.collect_entities_by_name(child, "Parent", &mut collected);
        assert_eq!(collected, vec![parent]);
    }

    #[test]
    fn collect_entities_from_children() {
        use crate::components::{Children, Parent};

        let mut world = World::new();
        world.register_inspector::<Parent>();
        world.register_inspector_default::<Children>();

        let parent = world.spawn();
        let c1 = world.spawn();
        let c2 = world.spawn();
        world.insert(parent, Children(vec![c1, c2])).unwrap();

        let mut collected = Vec::new();
        world.collect_entities_by_name(parent, "Children", &mut collected);
        assert_eq!(collected, vec![c1, c2]);
    }

    #[test]
    fn remap_entities_in_parent() {
        use crate::components::{Children, Parent};

        let mut world = World::new();
        world.register_inspector::<Parent>();
        world.register_inspector_default::<Children>();

        let old_parent = world.spawn();
        let new_parent = world.spawn();
        let child = world.spawn();
        world.insert(child, Parent(old_parent)).unwrap();

        world.remap_entities_by_name(child, "Parent", &mut |e| {
            if e == old_parent { new_parent } else { e }
        });

        assert_eq!(world.get::<Parent>(child), Some(&Parent(new_parent)));
    }

    #[test]
    fn remap_entities_in_children() {
        use crate::components::{Children, Parent};

        let mut world = World::new();
        world.register_inspector::<Parent>();
        world.register_inspector_default::<Children>();

        let parent = world.spawn();
        let old_c1 = world.spawn();
        let old_c2 = world.spawn();
        let new_c1 = world.spawn();
        let new_c2 = world.spawn();
        world
            .insert(parent, Children(vec![old_c1, old_c2]))
            .unwrap();

        world.remap_entities_by_name(parent, "Children", &mut |e| {
            if e == old_c1 {
                new_c1
            } else if e == old_c2 {
                new_c2
            } else {
                e
            }
        });

        assert_eq!(
            world.get::<Children>(parent),
            Some(&Children(vec![new_c1, new_c2]))
        );
    }

    #[test]
    fn collect_all_entities_gathers_from_all_components() {
        use crate::components::{Children, Parent};

        let mut world = World::new();
        world.register_inspector::<Parent>();
        world.register_inspector_default::<Children>();

        let parent = world.spawn();
        let c1 = world.spawn();
        let c2 = world.spawn();
        // Entity has both Parent (pointing at parent) and Children (containing c1, c2)
        let entity = world.spawn();
        world.insert(entity, Parent(parent)).unwrap();
        world.insert(entity, Children(vec![c1, c2])).unwrap();

        let mut collected = Vec::new();
        world.collect_all_entities(entity, &mut collected);
        assert_eq!(collected.len(), 3);
        assert!(collected.contains(&parent));
        assert!(collected.contains(&c1));
        assert!(collected.contains(&c2));
    }

    #[test]
    fn collect_noop_for_non_entity_component() {
        let mut world = World::new();
        world.register_inspector_default::<crate::components::Transform>();

        let entity = world.spawn();
        world
            .insert(entity, crate::components::Transform::IDENTITY)
            .unwrap();

        let mut collected = Vec::new();
        world.collect_entities_by_name(entity, "Transform", &mut collected);
        assert!(collected.is_empty());
    }

    // --- Clone entity tests ---

    #[test]
    fn clone_entity_copies_components() {
        let mut world = World::new();
        crate::register_std_components(&mut world);

        let src = world.spawn();
        let t = crate::components::Transform::from_translation(redlilium_core::math::Vec3::new(
            1.0, 2.0, 3.0,
        ));
        world.insert(src, t).unwrap();
        world
            .insert(src, crate::components::Name::new("original"))
            .unwrap();

        let dst = world.clone_entity(src).unwrap();

        assert_ne!(src, dst);
        assert_eq!(world.get::<crate::components::Transform>(dst), Some(&t));
        assert_eq!(
            world
                .get::<crate::components::Name>(dst)
                .map(|n| n.as_str()),
            Some("original"),
        );
    }

    #[test]
    fn clone_entity_dead_source_returns_none() {
        let mut world = World::new();
        crate::register_std_components(&mut world);

        let src = world.spawn();
        world.despawn(src);

        assert!(world.clone_entity(src).is_none());
    }

    #[test]
    fn clone_entity_tree_flat() {
        let mut world = World::new();
        crate::register_std_components(&mut world);

        let parent = world.spawn();
        world
            .insert(parent, crate::components::Name::new("parent"))
            .unwrap();
        world
            .insert(parent, crate::components::Transform::IDENTITY)
            .unwrap();

        let child_a = world.spawn();
        world
            .insert(child_a, crate::components::Name::new("child_a"))
            .unwrap();
        crate::hierarchy::set_parent(&mut world, child_a, parent);

        let child_b = world.spawn();
        world
            .insert(child_b, crate::components::Name::new("child_b"))
            .unwrap();
        crate::hierarchy::set_parent(&mut world, child_b, parent);

        // 3 original + 3 cloned = 6
        let entity_count_before = world.entity_count();
        let mapping = world.clone_entity_tree(parent);
        assert_eq!(mapping.len(), 3);
        assert_eq!(world.entity_count(), entity_count_before + 3);

        let new_parent = mapping[&parent];
        let new_child_a = mapping[&child_a];
        let new_child_b = mapping[&child_b];

        // Verify component data cloned
        assert_eq!(
            world
                .get::<crate::components::Name>(new_parent)
                .map(|n| n.as_str()),
            Some("parent"),
        );
        assert_eq!(
            world
                .get::<crate::components::Name>(new_child_a)
                .map(|n| n.as_str()),
            Some("child_a"),
        );

        // Verify hierarchy remapped
        let children = world.get::<crate::Children>(new_parent).unwrap();
        assert_eq!(children.0, vec![new_child_a, new_child_b]);

        let parent_of_a = world.get::<crate::Parent>(new_child_a).unwrap();
        assert_eq!(parent_of_a.0, new_parent);

        let parent_of_b = world.get::<crate::Parent>(new_child_b).unwrap();
        assert_eq!(parent_of_b.0, new_parent);

        // Cloned root should have no parent (original didn't)
        assert!(world.get::<crate::Parent>(new_parent).is_none());
    }

    #[test]
    fn clone_entity_tree_deep() {
        let mut world = World::new();
        crate::register_std_components(&mut world);

        // root -> mid -> leaf
        let root = world.spawn();
        world
            .insert(root, crate::components::Name::new("root"))
            .unwrap();

        let mid = world.spawn();
        world
            .insert(mid, crate::components::Name::new("mid"))
            .unwrap();
        crate::hierarchy::set_parent(&mut world, mid, root);

        let leaf = world.spawn();
        world
            .insert(leaf, crate::components::Name::new("leaf"))
            .unwrap();
        crate::hierarchy::set_parent(&mut world, leaf, mid);

        let mapping = world.clone_entity_tree(root);
        assert_eq!(mapping.len(), 3);

        let new_root = mapping[&root];
        let new_mid = mapping[&mid];
        let new_leaf = mapping[&leaf];

        // root -> mid
        let root_children = world.get::<crate::Children>(new_root).unwrap();
        assert_eq!(root_children.0, vec![new_mid]);

        // mid -> leaf
        let mid_children = world.get::<crate::Children>(new_mid).unwrap();
        assert_eq!(mid_children.0, vec![new_leaf]);

        // leaf has parent = mid
        assert_eq!(world.get::<crate::Parent>(new_leaf).unwrap().0, new_mid);

        // mid has parent = root
        assert_eq!(world.get::<crate::Parent>(new_mid).unwrap().0, new_root);

        // root has no parent
        assert!(world.get::<crate::Parent>(new_root).is_none());
    }

    #[test]
    fn clone_entity_tree_dead_root_returns_empty() {
        let mut world = World::new();
        crate::register_std_components(&mut world);

        let root = world.spawn();
        world.despawn(root);

        let mapping = world.clone_entity_tree(root);
        assert!(mapping.is_empty());
    }
}
