use std::any::TypeId;
use std::collections::HashMap;

use crate::entity::{Entity, EntityAllocator};
use crate::query::ContainsChecker;
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
}

impl World {
    /// Creates a new empty world.
    pub fn new() -> Self {
        Self {
            entities: EntityAllocator::new(),
            components: HashMap::new(),
            resources: Resources::new(),
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
