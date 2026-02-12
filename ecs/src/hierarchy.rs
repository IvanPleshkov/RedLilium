//! Parent-child hierarchy operations.
//!
//! Provides functions for managing entity relationships. All operations
//! maintain consistency between [`Parent`] and [`Children`] components.
//!
//! # Usage
//!
//! ```ignore
//! // Direct mutation (requires &mut World)
//! set_parent(&mut world, child, parent);
//! remove_parent(&mut world, child);
//! despawn_recursive(&mut world, entity);
//!
//! // Deferred via commands (from within systems)
//! let commands = world.resource::<CommandBuffer>();
//! commands.set_parent(child, parent);
//! commands.remove_parent(child);
//! commands.despawn_recursive(entity);
//! ```

use crate::{CommandBuffer, Entity, World};

use crate::components::{Children, Parent};

/// Sets `entity` as a child of `parent`.
///
/// Updates both the [`Parent`] component on `entity` and the [`Children`]
/// component on `parent`. If `entity` already has a different parent,
/// it is removed from the old parent's children first.
///
/// # Panics
///
/// Panics if `entity == parent` (cannot parent to self).
pub fn set_parent(world: &mut World, entity: Entity, parent: Entity) {
    assert_ne!(
        entity, parent,
        "Cannot set entity as its own parent: {entity}"
    );

    // Remove from old parent if any
    if let Some(old_parent) = world.get::<Parent>(entity).map(|p| p.0) {
        if old_parent == parent {
            return; // Already parented correctly
        }
        // Remove entity from old parent's children
        if let Some(children) = world.get_mut::<Children>(old_parent) {
            children.0.retain(|&e| e != entity);
        }
    }

    // Set Parent component on entity
    world
        .insert(entity, Parent(parent))
        .expect("Parent not registered");

    // Add to new parent's Children
    if let Some(children) = world.get_mut::<Children>(parent) {
        if !children.0.contains(&entity) {
            children.0.push(entity);
        }
    } else {
        world
            .insert(parent, Children(vec![entity]))
            .expect("Children not registered");
    }
}

/// Removes the parent relationship from `entity`.
///
/// Removes the [`Parent`] component from `entity` and removes `entity`
/// from its parent's [`Children`] list. Does nothing if `entity` has
/// no parent.
pub fn remove_parent(world: &mut World, entity: Entity) {
    let Some(parent) = world.remove::<Parent>(entity) else {
        return;
    };

    // Remove from parent's children
    if let Some(children) = world.get_mut::<Children>(parent.0) {
        children.0.retain(|&e| e != entity);
    }
}

/// Despawns an entity and all its descendants recursively.
///
/// First removes the entity from its parent's children list (if any),
/// then despawns the entity and all descendants depth-first.
pub fn despawn_recursive(world: &mut World, entity: Entity) {
    // Remove from parent first
    if let Some(parent) = world.remove::<Parent>(entity)
        && let Some(children) = world.get_mut::<Children>(parent.0)
    {
        children.0.retain(|&e| e != entity);
    }

    despawn_subtree(world, entity);
}

/// Despawns an entity and all children depth-first (internal).
fn despawn_subtree(world: &mut World, entity: Entity) {
    // Collect children first to avoid borrow issues
    let child_entities = world
        .remove::<Children>(entity)
        .map(|c| c.0)
        .unwrap_or_default();

    for child in child_entities {
        despawn_subtree(world, child);
    }

    world.despawn(entity);
}

// ---- CommandBuffer extensions ----

/// Extension trait adding hierarchy commands to [`CommandBuffer`].
///
/// Import this trait to use `commands.cmd_set_parent()`, etc.
pub trait HierarchyCommands {
    /// Queues a [`set_parent`] command.
    fn cmd_set_parent(&self, entity: Entity, parent: Entity);

    /// Queues a [`remove_parent`] command.
    fn cmd_remove_parent(&self, entity: Entity);

    /// Queues a [`despawn_recursive`] command.
    fn cmd_despawn_recursive(&self, entity: Entity);
}

impl HierarchyCommands for CommandBuffer {
    fn cmd_set_parent(&self, entity: Entity, parent: Entity) {
        self.push(move |world| {
            set_parent(world, entity, parent);
        });
    }

    fn cmd_remove_parent(&self, entity: Entity) {
        self.push(move |world| {
            remove_parent(world, entity);
        });
    }

    fn cmd_despawn_recursive(&self, entity: Entity) {
        self.push(move |world| {
            despawn_recursive(world, entity);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn register_hierarchy(world: &mut World) {
        world.register_component::<Parent>();
        world.register_component::<Children>();
    }

    #[test]
    fn set_parent_creates_relationship() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let parent = world.spawn();
        let child = world.spawn();

        set_parent(&mut world, child, parent);

        assert_eq!(world.get::<Parent>(child), Some(&Parent(parent)));
        let children = world.get::<Children>(parent).unwrap();
        assert_eq!(children.0, vec![child]);
    }

    #[test]
    fn set_parent_multiple_children() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let parent = world.spawn();
        let child_a = world.spawn();
        let child_b = world.spawn();

        set_parent(&mut world, child_a, parent);
        set_parent(&mut world, child_b, parent);

        let children = world.get::<Children>(parent).unwrap();
        assert_eq!(children.len(), 2);
        assert!(children.0.contains(&child_a));
        assert!(children.0.contains(&child_b));
    }

    #[test]
    fn set_parent_idempotent() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let parent = world.spawn();
        let child = world.spawn();

        set_parent(&mut world, child, parent);
        set_parent(&mut world, child, parent); // Same parent again

        let children = world.get::<Children>(parent).unwrap();
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn set_parent_reparents() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let parent_a = world.spawn();
        let parent_b = world.spawn();
        let child = world.spawn();

        set_parent(&mut world, child, parent_a);
        set_parent(&mut world, child, parent_b);

        assert_eq!(world.get::<Parent>(child), Some(&Parent(parent_b)));

        // Old parent should have no children
        let children_a = world.get::<Children>(parent_a).unwrap();
        assert!(children_a.is_empty());

        // New parent should have the child
        let children_b = world.get::<Children>(parent_b).unwrap();
        assert_eq!(children_b.0, vec![child]);
    }

    #[test]
    #[should_panic(expected = "Cannot set entity as its own parent")]
    fn set_parent_self_panics() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let entity = world.spawn();
        set_parent(&mut world, entity, entity);
    }

    #[test]
    fn remove_parent_clears_relationship() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let parent = world.spawn();
        let child = world.spawn();

        set_parent(&mut world, child, parent);
        remove_parent(&mut world, child);

        assert!(world.get::<Parent>(child).is_none());
        let children = world.get::<Children>(parent).unwrap();
        assert!(children.is_empty());
    }

    #[test]
    fn remove_parent_noop_without_parent() {
        let mut world = World::new();
        let entity = world.spawn();
        remove_parent(&mut world, entity); // Should not panic
    }

    #[test]
    fn despawn_recursive_removes_subtree() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let root = world.spawn();
        let child_a = world.spawn();
        let child_b = world.spawn();
        let grandchild = world.spawn();

        set_parent(&mut world, child_a, root);
        set_parent(&mut world, child_b, root);
        set_parent(&mut world, grandchild, child_a);

        assert_eq!(world.entity_count(), 4);

        despawn_recursive(&mut world, root);

        assert_eq!(world.entity_count(), 0);
        assert!(!world.is_alive(root));
        assert!(!world.is_alive(child_a));
        assert!(!world.is_alive(child_b));
        assert!(!world.is_alive(grandchild));
    }

    #[test]
    fn despawn_recursive_removes_from_parent() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let parent = world.spawn();
        let child = world.spawn();
        let grandchild = world.spawn();

        set_parent(&mut world, child, parent);
        set_parent(&mut world, grandchild, child);

        // Despawn child subtree (child + grandchild)
        despawn_recursive(&mut world, child);

        assert!(world.is_alive(parent));
        assert!(!world.is_alive(child));
        assert!(!world.is_alive(grandchild));

        let children = world.get::<Children>(parent).unwrap();
        assert!(children.is_empty());
    }

    #[test]
    fn despawn_recursive_leaf_entity() {
        let mut world = World::new();
        let entity = world.spawn();

        despawn_recursive(&mut world, entity);

        assert!(!world.is_alive(entity));
    }

    #[test]
    fn command_set_parent() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        world.init_commands();

        let parent = world.spawn();
        let child = world.spawn();

        {
            let commands = world.resource::<CommandBuffer>();
            commands.cmd_set_parent(child, parent);
        }

        world.apply_commands();

        assert_eq!(world.get::<Parent>(child), Some(&Parent(parent)));
    }

    #[test]
    fn command_despawn_recursive() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        world.init_commands();

        let parent = world.spawn();
        let child = world.spawn();
        set_parent(&mut world, child, parent);

        {
            let commands = world.resource::<CommandBuffer>();
            commands.cmd_despawn_recursive(parent);
        }

        world.apply_commands();

        assert!(!world.is_alive(parent));
        assert!(!world.is_alive(child));
    }
}
