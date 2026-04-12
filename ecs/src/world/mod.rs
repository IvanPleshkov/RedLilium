mod actions;
mod batch_ops;
mod component_ops;
mod hooks;
mod inspector;
mod query_access;
mod resources;
#[cfg(test)]
mod tests;

use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};

use crate::component::Component;
use crate::entity::{Entities, Entity};
use crate::observer::Observers;
use crate::reactive::Triggers;
use crate::resource::Resources;
use crate::sparse_set::ComponentStorage;

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

// Re-export InspectResult for public API consumers.
pub use crate::sparse_set::InspectResult;

pub use actions::set_component_actions;

/// Type-erased serialize helper: reads `T` from the world and serializes it.
fn serialize_component_fn<T: Component>(
    world: &World,
    entity: Entity,
    ctx: &mut crate::serialize::SerializeContext<'_>,
) -> Result<Option<crate::serialize::SerializedComponent>, crate::serialize::SerializeError> {
    let Some(comp) = world.get::<T>(entity) else {
        return Ok(None);
    };
    match comp.serialize_component(ctx) {
        Ok(value) => Ok(Some(crate::serialize::SerializedComponent {
            type_name: T::NAME.to_owned(),
            data: value,
        })),
        Err(crate::serialize::SerializeError::NotSerializable { .. }) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Type-erased deserialize helper: deserializes `T` and inserts it on an entity.
fn deserialize_component_fn<T: Component>(
    entity: Entity,
    data: &crate::serialize::Value,
    ctx: &mut crate::serialize::DeserializeContext<'_>,
) -> Result<(), crate::serialize::DeserializeError> {
    ctx.load_data(data)?;
    let comp = T::deserialize_component(ctx)?;
    ctx.world_mut().insert(entity, comp).map_err(|e| {
        crate::serialize::DeserializeError::UnknownComponent {
            type_name: e.type_name.to_string(),
        }
    })?;
    Ok(())
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
    entities: Entities,
    components: HashMap<TypeId, parking_lot::RwLock<ComponentStorage>>,
    resources: Resources,
    /// Global tick counter for change detection.
    tick: u64,
    /// Reverse index from component name to TypeId for name-based lookups.
    name_index: BTreeMap<&'static str, TypeId>,
    /// Deferred observer registry and pending triggers.
    observers: Observers,
    /// Monomorphized swap functions for each registered `Triggers<M>` resource.
    trigger_swap_fns: Vec<fn(&mut World)>,
}

impl redlilium_core::abstract_editor::Editable for World {}

impl World {
    /// Creates a new empty world.
    pub fn new() -> Self {
        Self {
            entities: Entities::new(),
            components: HashMap::new(),
            resources: Resources::new(),
            tick: 0,
            name_index: BTreeMap::new(),
            observers: Observers::new(),
            trigger_swap_fns: Vec::new(),
        }
    }

    // ---- Component lock helpers ----

    /// Returns a reference to the component storage without locking.
    ///
    /// # Safety
    ///
    /// Returns a mutable reference to the component storage.
    ///
    /// Uses `RwLock::get_mut()` which bypasses locking via the borrow checker
    /// (`&mut RwLock` guarantees no other references exist).
    fn storage_mut(&mut self, type_id: &TypeId) -> Option<&mut ComponentStorage> {
        self.components.get_mut(type_id).map(|lock| lock.get_mut())
    }

    // ---- Entity management ----

    /// Spawns a new entity and returns its ID.
    pub fn spawn(&mut self) -> Entity {
        self.entities.allocate(self.tick)
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
        let hooks: Vec<crate::sparse_set::ComponentHookFn> = self
            .components
            .values_mut()
            .map(|lock| lock.get_mut())
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
            .filter(|(_, lock)| lock.read().contains_untyped(index))
            .filter_map(|(type_id, _)| self.observers.remove_trigger_key(type_id))
            .collect();

        // Deallocate entity and remove all components (including any added by hooks)
        self.entities.deallocate(entity);
        for lock in self.components.values_mut() {
            let storage = lock.get_mut();
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

    /// Returns the alive entity at the given slot index, or `None` if the
    /// slot is empty or has been recycled.
    pub fn entity_at_index(&self, index: u32) -> Option<Entity> {
        self.entities.entity_at_index(index)
    }

    /// Iterates over all currently alive entity IDs.
    pub fn iter_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.entities.iter_alive()
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
