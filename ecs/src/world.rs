use std::any::TypeId;
use std::collections::HashMap;

use crate::commands::CommandBuffer;
use crate::entity::{Entity, EntityAllocator};
use crate::events::Events;
use crate::query::{AddedFilter, ChangedFilter, ContainsChecker};
use crate::resource::{ResourceRef, ResourceRefMut, Resources};
use crate::sparse_set::{ComponentStorage, Ref, RefMut};

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
///
/// let entity = world.spawn();
/// world.insert(entity, Position { x: 0.0, y: 0.0 });
/// world.insert(entity, Velocity { x: 1.0, y: 0.0 });
///
/// // Query components
/// let positions = world.read::<Position>();
/// let velocities = world.read::<Velocity>();
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
    /// Global tick counter for change detection.
    tick: u64,
}

impl World {
    /// Creates a new empty world.
    pub fn new() -> Self {
        Self {
            entities: EntityAllocator::new(),
            components: HashMap::new(),
            resources: Resources::new(),
            tick: 0,
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
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.deallocate(entity) {
            return false;
        }

        let index = entity.index();
        for storage in self.components.values_mut() {
            storage.remove_untyped(index);
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
    pub fn register_component<T: Send + Sync + 'static>(&mut self) {
        self.components
            .entry(TypeId::of::<T>())
            .or_insert_with(ComponentStorage::new::<T>);
    }

    /// Inserts a component on an entity. Creates the storage for T if needed.
    ///
    /// If the entity already has this component, the value is replaced.
    /// Uses tick 0 (untracked). For change-tracked insertion, use
    /// [`insert_tracked`](World::insert_tracked).
    ///
    /// # Panics
    ///
    /// Panics if the entity is not alive.
    pub fn insert<T: Send + Sync + 'static>(&mut self, entity: Entity, component: T) {
        assert!(
            self.entities.is_alive(entity),
            "Cannot insert component on dead entity {entity}"
        );

        let storage = self
            .components
            .entry(TypeId::of::<T>())
            .or_insert_with(ComponentStorage::new::<T>);

        storage.typed_mut::<T>().insert(entity.index(), component);
    }

    /// Inserts a component with change tracking at the current world tick.
    ///
    /// Like [`insert`](World::insert) but records the current tick for
    /// change detection queries.
    ///
    /// # Panics
    ///
    /// Panics if the entity is not alive.
    pub fn insert_tracked<T: Send + Sync + 'static>(&mut self, entity: Entity, component: T) {
        assert!(
            self.entities.is_alive(entity),
            "Cannot insert component on dead entity {entity}"
        );

        let tick = self.tick;
        let storage = self
            .components
            .entry(TypeId::of::<T>())
            .or_insert_with(ComponentStorage::new::<T>);

        storage
            .typed_mut::<T>()
            .insert_with_tick(entity.index(), component, tick);
    }

    /// Removes a component from an entity.
    ///
    /// Returns the removed value, or `None` if the entity did not have it.
    pub fn remove<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        let storage = self.components.get_mut(&TypeId::of::<T>())?;
        storage.typed_mut::<T>().remove(entity.index())
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
    /// # Panics
    ///
    /// Panics if T is exclusively borrowed by a [`write`](World::write) call,
    /// or if T has never been registered or inserted.
    pub fn read<T: 'static>(&self) -> Ref<'_, T> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| {
                panic!(
                    "Component type `{}` has never been registered. Call insert() or register_component() first.",
                    std::any::type_name::<T>()
                )
            });
        Ref::new(storage)
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
    /// # Panics
    ///
    /// Panics if T is borrowed by any [`read`](World::read) or [`write`](World::write) call,
    /// or if T has never been registered or inserted.
    pub fn write<T: 'static>(&self) -> RefMut<'_, T> {
        let storage = self
            .components
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| {
                panic!(
                    "Component type `{}` has never been registered. Call insert() or register_component() first.",
                    std::any::type_name::<T>()
                )
            });
        RefMut::new(storage)
    }

    /// Gets exclusive write access to all components of type T, returning `None`
    /// if the type has never been registered.
    ///
    /// Non-panicking variant of [`write`](World::write). Used by `OptionalWrite<T>`.
    pub fn try_write<T: 'static>(&self) -> Option<RefMut<'_, T>> {
        let storage = self.components.get(&TypeId::of::<T>())?;
        Some(RefMut::new(storage))
    }

    /// Returns the TypeIds of all registered component types.
    ///
    /// Used by [`WorldLocks`](crate::world_locks) to create per-component locks.
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

    // ---- Resource management ----

    /// Inserts or replaces a resource.
    pub fn insert_resource<T: Send + Sync + 'static>(&mut self, value: T) {
        self.resources.insert(value);
    }

    /// Removes a resource, returning it if present.
    pub fn remove_resource<T: 'static>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }

    /// Returns whether a resource of type T exists.
    pub fn has_resource<T: 'static>(&self) -> bool {
        self.resources.contains::<T>()
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

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Velocity {
        x: f32,
        y: f32,
    }

    #[derive(Debug, PartialEq)]
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
        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 2.0 });

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 1.0, y: 2.0 })
        );
    }

    #[test]
    #[should_panic(expected = "Cannot insert component on dead entity")]
    fn insert_on_dead_entity_panics() {
        let mut world = World::new();
        let entity = world.spawn();
        world.despawn(entity);
        world.insert(entity, Position { x: 0.0, y: 0.0 });
    }

    #[test]
    fn remove_component() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Health(100));

        assert_eq!(world.remove::<Health>(entity), Some(Health(100)));
        assert!(world.get::<Health>(entity).is_none());
    }

    #[test]
    fn despawn_removes_all_components() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Position { x: 0.0, y: 0.0 });
        world.insert(entity, Health(100));

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
        for i in 0..3 {
            let e = world.spawn();
            world.insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                },
            );
        }

        let positions = world.read::<Position>();
        assert_eq!(positions.len(), 3);

        let xs: Vec<f32> = positions.iter().map(|(_, p)| p.x).collect();
        assert!(xs.contains(&0.0));
        assert!(xs.contains(&1.0));
        assert!(xs.contains(&2.0));
    }

    #[test]
    fn write_query_allows_mutation() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0, y: 2.0 });

        {
            let mut positions = world.write::<Position>();
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
        let e = world.spawn();
        world.insert(e, Position { x: 0.0, y: 0.0 });

        let _a = world.read::<Position>();
        let _b = world.read::<Position>();
    }

    #[test]
    #[should_panic(expected = "already borrowed")]
    fn read_write_conflict_panics() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0, y: 0.0 });

        let _r = world.read::<Position>();
        let _w = world.write::<Position>();
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
        let old = world.spawn();
        world.insert(old, Position { x: 1.0, y: 2.0 });

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

        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0, y: 0.0 });
        world.insert(e1, Health(100));

        let e2 = world.spawn();
        world.insert(e2, Position { x: 2.0, y: 0.0 });

        let positions = world.read::<Position>();
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

        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0, y: 0.0 });
        world.insert(e1, Frozen);

        let e2 = world.spawn();
        world.insert(e2, Position { x: 2.0, y: 0.0 });

        let positions = world.read::<Position>();
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

        let e1 = world.spawn();
        world.insert(e1, Position { x: 0.0, y: 0.0 });
        world.insert(e1, Velocity { x: 1.0, y: 0.0 });

        let e2 = world.spawn();
        world.insert(e2, Position { x: 5.0, y: 5.0 });
        // e2 has no Velocity

        let positions = world.read::<Position>();
        let velocities = world.read::<Velocity>();

        let mut count = 0;
        for (idx, _pos) in positions.iter() {
            if velocities.get(idx).is_some() {
                count += 1;
            }
        }
        assert_eq!(count, 1); // Only e1 has both
    }
}
