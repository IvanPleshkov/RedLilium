//! Prefab extraction and instantiation.
//!
//! A [`Prefab`] is a portable snapshot of an entity tree that can be
//! instantiated into any [`World`](crate::World). It stores type-erased
//! component data via the [`ComponentBag`] trait, so it is fully
//! self-contained and independent of the source world.
//!
//! # Example
//!
//! ```ignore
//! // Extract a prefab from an existing entity tree
//! let prefab = world.extract_prefab(root_entity);
//!
//! // Instantiate it (can be called multiple times)
//! let entities_a = prefab.instantiate(&mut world);
//! let entities_b = prefab.instantiate(&mut other_world);
//! ```

use std::collections::HashMap;

use crate::component::Component;
use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// ComponentBag — type-erased component storage
// ---------------------------------------------------------------------------

/// Type-erased component storage for prefabs.
///
/// Each bag holds a single component value of a concrete type. The trait
/// provides clone, entity collection/remapping, and insertion operations
/// without exposing the concrete type.
pub(crate) trait ComponentBag: Send + Sync {
    /// Clone this bag into a new heap allocation.
    fn clone_box(&self) -> Box<dyn ComponentBag>;

    /// Collect entity references stored in this component.
    #[allow(dead_code)]
    fn collect_entities(&self, collector: &mut Vec<Entity>);

    /// Remap entity references stored in this component.
    fn remap_entities(&mut self, map: &mut dyn FnMut(Entity) -> Entity);

    /// Consume this bag and insert the component into the world on the given entity.
    fn consume_into(self: Box<Self>, world: &mut World, entity: Entity);
}

/// Concrete [`ComponentBag`] implementation for a specific component type.
pub(crate) struct TypedBag<T>(pub T);

impl<T: Component + Clone> ComponentBag for TypedBag<T> {
    fn clone_box(&self) -> Box<dyn ComponentBag> {
        Box::new(TypedBag(self.0.clone()))
    }

    fn collect_entities(&self, collector: &mut Vec<Entity>) {
        self.0.collect_entities(collector);
    }

    fn remap_entities(&mut self, map: &mut dyn FnMut(Entity) -> Entity) {
        self.0.remap_entities(map);
    }

    fn consume_into(self: Box<Self>, world: &mut World, entity: Entity) {
        let _ = world.insert(entity, self.0);
    }
}

// ---------------------------------------------------------------------------
// Prefab
// ---------------------------------------------------------------------------

/// A portable snapshot of an entity tree.
///
/// Created via [`World::extract_prefab`] and instantiated via
/// [`Prefab::instantiate`]. The prefab is fully self-contained and can
/// be instantiated into any world (as long as the component types are
/// registered there).
///
/// The first entity in the list is always the root.
pub struct Prefab {
    /// Each entry: (original source entity ID, component bags).
    ///
    /// The source entity ID is used only to build the remap table during
    /// instantiation. Index 0 is the root.
    entities: Vec<(Entity, Vec<Box<dyn ComponentBag>>)>,
}

impl Clone for Prefab {
    fn clone(&self) -> Self {
        Prefab {
            entities: self
                .entities
                .iter()
                .map(|(e, bags)| (*e, bags.iter().map(|b| b.clone_box()).collect()))
                .collect(),
        }
    }
}

impl Prefab {
    /// Creates an empty prefab with no entities.
    pub fn empty() -> Self {
        Prefab {
            entities: Vec::new(),
        }
    }

    /// Creates a prefab from a list of (source entity, component bags) pairs.
    pub(crate) fn new(entities: Vec<(Entity, Vec<Box<dyn ComponentBag>>)>) -> Self {
        Prefab { entities }
    }

    /// Returns the number of entities in this prefab.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns `true` if the prefab contains no entities.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Instantiate this prefab into a world.
    ///
    /// Spawns new entities, clones component data with entity references
    /// remapped to the new entities, and inserts all components.
    ///
    /// Entity references that point **outside** the prefab (i.e., not
    /// matching any source entity in the prefab) are left unchanged.
    ///
    /// Returns the list of spawned entities in the same order as the
    /// prefab's internal entity list. Index 0 is the root.
    pub fn instantiate(&self, world: &mut World) -> Vec<Entity> {
        if self.entities.is_empty() {
            return Vec::new();
        }

        // 1. Spawn new entities
        let new_entities: Vec<Entity> = (0..self.entities.len()).map(|_| world.spawn()).collect();

        // 2. Build mapping: original source entity → new entity
        let mapping: HashMap<Entity, Entity> = self
            .entities
            .iter()
            .zip(new_entities.iter())
            .map(|((src, _), &dst)| (*src, dst))
            .collect();

        // 3. For each prefab entity: clone bags, remap, insert
        for ((_, bags), &new_entity) in self.entities.iter().zip(new_entities.iter()) {
            for bag in bags {
                let mut cloned = bag.clone_box();
                cloned.remap_entities(&mut |e| *mapping.get(&e).unwrap_or(&e));
                cloned.consume_into(world, new_entity);
            }
        }

        new_entities
    }
}
