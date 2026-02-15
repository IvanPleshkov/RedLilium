use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};

use crate::access_set::AccessInfo;
use crate::bundle::Bundle;
use crate::commands::CommandBuffer;
use crate::component::Component;
use crate::entity::{Entity, EntityAllocator};
use crate::events::Events;
use crate::main_thread_resource::MainThreadResources;
use crate::query::{AddedFilter, ChangedFilter, ContainsChecker, RemovedFilter};
use crate::resource::{Resource, ResourceRef, ResourceRefMut, Resources};
use crate::sparse_set::{ComponentStorage, LockGuard, Ref, RefMut};
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
    /// Records removals for [`removed`](World::removed) filter queries.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.deallocate(entity) {
            return false;
        }

        let index = entity.index();
        let tick = self.tick;
        for storage in self.components.values_mut() {
            if storage.remove_untyped(index) {
                storage.record_removal(index, tick);
            }
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
            },
        );
    }

    /// Registers a component type with full inspector support including "Add Component".
    ///
    /// Like [`register_inspector`](World::register_inspector) but also enables
    /// inserting a default instance via the inspector "Add Component" button.
    pub fn register_inspector_default<T: Component + Default>(&mut self) {
        self.register_component::<T>();
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
            },
        );
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

        let storage =
            self.components
                .get_mut(&TypeId::of::<T>())
                .ok_or(ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;

        storage.typed_mut::<T>().insert(entity.index(), component);
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

        let tick = self.tick;
        let storage =
            self.components
                .get_mut(&TypeId::of::<T>())
                .ok_or(ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;

        storage
            .typed_mut::<T>()
            .insert_with_tick(entity.index(), component, tick);
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
    /// Records removals for [`removed`](World::removed) filter queries.
    pub fn despawn_batch(&mut self, entities: &[Entity]) {
        let tick = self.tick;
        for &entity in entities {
            if !self.entities.deallocate(entity) {
                continue;
            }
            let index = entity.index();
            for storage in self.components.values_mut() {
                if storage.remove_untyped(index) {
                    storage.record_removal(index, tick);
                }
            }
        }
    }

    /// Inserts a component on each entity from parallel slices.
    ///
    /// `entities` and `components` must have the same length.
    /// Uses tick 0 (untracked). For change-tracked insertion,
    /// use [`insert_batch_tracked`](World::insert_batch_tracked).
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

        let storage =
            self.components
                .get_mut(&TypeId::of::<T>())
                .ok_or(ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        let set = storage.typed_mut::<T>();
        set.reserve(components.len());

        for (entity, component) in entities.iter().zip(components) {
            assert!(
                self.entities.is_alive(*entity),
                "Cannot insert component on dead entity {entity}"
            );
            set.insert(entity.index(), component);
        }
        Ok(())
    }

    /// Inserts a component on each entity with change tracking.
    ///
    /// Like [`insert_batch`](World::insert_batch) but records the current tick.
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

        let tick = self.tick;
        let storage =
            self.components
                .get_mut(&TypeId::of::<T>())
                .ok_or(ComponentNotRegistered {
                    type_name: std::any::type_name::<T>(),
                })?;
        let set = storage.typed_mut::<T>();
        set.reserve(components.len());

        for (entity, component) in entities.iter().zip(components) {
            assert!(
                self.entities.is_alive(*entity),
                "Cannot insert component on dead entity {entity}"
            );
            set.insert_with_tick(entity.index(), component, tick);
        }
        Ok(())
    }

    /// Removes a component from multiple entities at once.
    ///
    /// Skips entities that don't have the component.
    /// Records removals for [`removed`](World::removed) filter queries.
    pub fn remove_batch<T: 'static>(&mut self, entities: &[Entity]) {
        let tick = self.tick;
        let Some(storage) = self.components.get_mut(&TypeId::of::<T>()) else {
            return;
        };

        // Collect which entities were actually removed, then record.
        let mut removed_indices = Vec::new();
        {
            let set = storage.typed_mut::<T>();
            for &entity in entities {
                if set.remove(entity.index()).is_some() {
                    removed_indices.push(entity.index());
                }
            }
        }
        for index in removed_indices {
            storage.record_removal(index, tick);
        }
    }

    /// Removes a component from an entity.
    ///
    /// Returns the removed value, or `None` if the entity did not have it.
    /// Records the removal for [`removed`](World::removed) filter queries.
    pub fn remove<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        let tick = self.tick;
        let storage = self.components.get_mut(&TypeId::of::<T>())?;
        let result = storage.typed_mut::<T>().remove(entity.index());
        if result.is_some() {
            storage.record_removal(entity.index(), tick);
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
        Ok(Ref::new(storage))
    }

    /// Gets shared read access to all components of type T, returning `None`
    /// if the type has never been registered.
    ///
    /// Non-panicking variant of [`read`](World::read). Used by `OptionalRead<T>`.
    pub fn try_read<T: 'static>(&self) -> Option<Ref<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(Ref::new(storage))
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
        Ok(RefMut::new(storage))
    }

    /// Gets exclusive write access to all components of type T, returning `None`
    /// if the type has never been registered.
    ///
    /// Non-panicking variant of [`write`](World::write). Used by `OptionalWrite<T>`.
    pub fn try_write<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new(storage))
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
        Ok(Ref::new_unlocked(storage))
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
        Ok(RefMut::new_unlocked(storage))
    }

    /// Gets optional shared read access without acquiring a lock.
    pub(crate) fn try_read_unlocked<T: 'static>(&self) -> Option<Ref<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(Ref::new_unlocked(storage))
    }

    /// Gets optional exclusive write access without acquiring a lock.
    pub(crate) fn try_write_unlocked<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new_unlocked(storage))
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
}
