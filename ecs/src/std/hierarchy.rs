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

use crate::{CommandBuffer, CommandCollector, Entity, World};

use super::components::{Children, Parent, Transform};

/// Returns whether `ancestor` is `descendant` itself or reachable by walking
/// `descendant`'s `Parent` chain upward. Used to reject hierarchy cycles.
///
/// The walk is bounded by a visited set so a pre-existing (manually created)
/// cycle cannot make this loop forever.
fn is_ancestor_of(world: &World, ancestor: Entity, descendant: Entity) -> bool {
    let mut current = descendant;
    let mut seen = std::collections::HashSet::new();
    loop {
        if current == ancestor {
            return true;
        }
        if !seen.insert(current) {
            return false; // hit a pre-existing cycle; stop
        }
        match world.get::<Parent>(current) {
            Some(p) => current = p.0,
            None => return false,
        }
    }
}

/// Marks an entity's [`Transform`] as changed (if it has one) so the
/// global-transform propagation system recomputes its subtree.
///
/// Reparenting changes an entity's *world* position without touching its local
/// [`Transform`], so without this the `Changed<Transform>` gate would skip the
/// moved subtree and it would keep a stale `GlobalTransform`.
fn mark_transform_changed(world: &mut World, entity: Entity) {
    if let Some(mut transform) = world.get_mut::<Transform>(entity) {
        transform.set_changed();
    }
}

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

    // Reject cycles: if `parent` is already a descendant of `entity`, parenting
    // `entity` under it would form a loop, which makes transform propagation and
    // other hierarchy walks recurse forever (stack overflow).
    assert!(
        !is_ancestor_of(world, entity, parent),
        "Cannot set parent {parent} of {entity}: would create a hierarchy cycle"
    );

    // Remove from old parent if any
    if let Some(old_parent) = world.get::<Parent>(entity).map(|p| p.0) {
        if old_parent == parent {
            return; // Already parented correctly
        }
        // Remove entity from old parent's children
        if let Some(mut children) = world.get_mut::<Children>(old_parent) {
            children.0.retain(|&e| e != entity);
        }
    }

    // Set Parent component on entity
    world
        .insert(entity, Parent(parent))
        .expect("Parent not registered");

    // Add to new parent's Children
    if let Some(mut children) = world.get_mut::<Children>(parent) {
        if !children.0.contains(&entity) {
            children.0.push(entity);
        }
    } else {
        world
            .insert(parent, Children(vec![entity]))
            .expect("Children not registered");
    }

    // The entity's world transform changed even though its local Transform did
    // not — dirty it so propagation recomputes the moved subtree.
    mark_transform_changed(world, entity);

    // Re-derive the subtree's inherited flags from the new parent (#68): a
    // node moved under a disabled/static/editor/hidden parent inherits its
    // effective state, and one moved out from under it sheds it.
    refresh_inherited_flags(world, entity);
}

/// Removes the parent relationship from `entity`.
///
/// Removes the [`Parent`] component from `entity` and removes `entity`
/// from its parent's [`Children`] list. Does nothing if `entity` has
/// no parent.
pub fn remove_parent(world: &mut World, entity: Entity) {
    let Ok(Some(parent)) = world.remove::<Parent>(entity) else {
        return;
    };

    // Remove from parent's children
    if let Some(mut children) = world.get_mut::<Children>(parent.0) {
        children.0.retain(|&e| e != entity);
    }

    // The entity's world transform now equals its local Transform; dirty it so
    // propagation recomputes the (now-unparented) subtree.
    mark_transform_changed(world, entity);

    // A root has no parent to inherit from — shed inherited flags (#68).
    refresh_node_inherited(world, entity, 0);
}

/// The three flag families that propagate down the hierarchy: the manual flag
/// and its inherited counterpart. `set_parent`/`remove_parent` re-derive
/// these; the dedicated walkers (`disable`, `mark_static`, `mark_editor`, ...)
/// propagate in-place flag changes.
const INHERITED_FLAG_FAMILIES: [(u32, u32); 3] = [
    (Entity::DISABLED, Entity::INHERITED_DISABLED),
    (Entity::STATIC, Entity::INHERITED_STATIC),
    (Entity::EDITOR, Entity::INHERITED_EDITOR),
];

/// Re-derives `entity`'s (and, where affected, its subtree's) inherited
/// flags from its current parent's effective flags. Manual flags (set
/// without the inherited bit) are the user's intent and are never touched —
/// a manually disabled node stays disabled wherever it moves, and its
/// subtree keeps deriving from it.
pub fn refresh_inherited_flags(world: &mut World, entity: Entity) {
    let parent_flags = world
        .get::<Parent>(entity)
        .map(|p| p.0)
        .map(|p| world.get_entity_flags(p))
        .unwrap_or(0);
    refresh_node_inherited(world, entity, parent_flags);
}

fn refresh_node_inherited(world: &mut World, entity: Entity, parent_flags: u32) {
    let flags = world.get_entity_flags(entity);
    let mut set_mask = 0u32;
    let mut clear_mask = 0u32;
    for (flag, inherited) in INHERITED_FLAG_FAMILIES {
        let manual = flags & flag != 0 && flags & inherited == 0;
        if manual {
            // User intent: the node (and the derivation below it) stands.
            continue;
        }
        // Parent's *effective* state: manual or inherited both propagate.
        let should_inherit = parent_flags & flag != 0;
        let has_inherited = flags & inherited != 0;
        if should_inherit && !has_inherited {
            set_mask |= flag | inherited;
        } else if !should_inherit && has_inherited {
            clear_mask |= flag | inherited;
        }
    }
    if set_mask == 0 && clear_mask == 0 {
        // Nothing changed at this node — its subtree already derives from an
        // unchanged effective state (prior consistency), so stop here.
        return;
    }
    if set_mask != 0 {
        world.set_entity_flags(entity, set_mask);
    }
    if clear_mask != 0 {
        world.clear_entity_flags(entity, clear_mask);
    }
    let new_flags = world.get_entity_flags(entity);
    let child_entities = world
        .get::<Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();
    for child in child_entities {
        refresh_node_inherited(world, child, new_flags);
    }
}

/// Despawns an entity and all its descendants recursively.
///
/// First removes the entity from its parent's children list (if any),
/// then despawns the entity and all descendants depth-first.
pub fn despawn_recursive(world: &mut World, entity: Entity) {
    // Remove from parent first
    if let Ok(Some(parent)) = world.remove::<Parent>(entity)
        && let Some(mut children) = world.get_mut::<Children>(parent.0)
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
        .ok()
        .flatten()
        .map(|c| c.0)
        .unwrap_or_default();

    for child in child_entities {
        despawn_subtree(world, child);
    }

    world.despawn(entity);
}

// ---- Entity disabling (always recursive, flag-based) ----

/// Disables an entity and all descendants recursively.
///
/// The target entity gets the `DISABLED` flag set and `INHERITED_DISABLED`
/// cleared (marking it as manually disabled). Descendants that are not
/// already disabled receive both `DISABLED` and `INHERITED_DISABLED` flags.
/// Already-disabled descendants keep their manual status.
pub fn disable(world: &mut World, entity: Entity) {
    // Mark as manually disabled (DISABLED set, INHERITED_DISABLED cleared)
    world.set_entity_flags(entity, Entity::DISABLED);
    world.clear_entity_flags(entity, Entity::INHERITED_DISABLED);
    disable_subtree(world, entity);
}

fn disable_subtree(world: &mut World, entity: Entity) {
    let child_entities = world
        .get::<Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();
    for child in child_entities {
        let flags = world.get_entity_flags(child);
        if flags & Entity::DISABLED == 0 {
            // Child was enabled — disable it as inherited
            world.set_entity_flags(child, Entity::DISABLED | Entity::INHERITED_DISABLED);
        }
        disable_subtree(world, child);
    }
}

/// Enables an entity and re-enables inherited-disabled descendants.
///
/// Descendants that were manually disabled (have `DISABLED` without
/// `INHERITED_DISABLED`) are left alone — their subtrees are not traversed.
///
/// Enabling a node inside a disabled subtree clears only the node's *manual*
/// intent: the final state re-derives from the parent, so the node stays
/// effectively disabled until its ancestor is enabled (#68).
pub fn enable(world: &mut World, entity: Entity) {
    world.clear_entity_flags(entity, Entity::DISABLED | Entity::INHERITED_DISABLED);
    enable_subtree(world, entity);
    // Re-derive from the parent: under a disabled ancestor the subtree
    // returns to inherited-disabled instead of becoming a hole in it.
    refresh_inherited_flags(world, entity);
}

fn enable_subtree(world: &mut World, entity: Entity) {
    let child_entities = world
        .get::<Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();
    for child in child_entities {
        let flags = world.get_entity_flags(child);
        if flags & Entity::INHERITED_DISABLED != 0 {
            // Child was inherited-disabled — re-enable it and recurse
            world.clear_entity_flags(child, Entity::DISABLED | Entity::INHERITED_DISABLED);
            enable_subtree(world, child);
        } else if flags & Entity::DISABLED == 0 {
            // Child is enabled — recurse in case deeper descendants are inherited-disabled
            enable_subtree(world, child);
        }
        // If child has DISABLED but NOT INHERITED_DISABLED, it was manually disabled — skip
    }
}

// ---- Entity static marking (always recursive, flag-based) ----

/// Marks an entity and all descendants as static recursively.
///
/// The target entity gets the `STATIC` flag set and `INHERITED_STATIC`
/// cleared (marking it as manually static). Descendants that are not
/// already static receive both `STATIC` and `INHERITED_STATIC` flags.
/// Already-static descendants keep their manual status.
///
/// Static entities are excluded from both `Read<T>` and `Write<T>` queries.
/// Use `ReadAll<T>` to include them in read-only queries, or access them
/// directly via exclusive systems (`&mut World`).
pub fn mark_static(world: &mut World, entity: Entity) {
    world.set_entity_flags(entity, Entity::STATIC);
    world.clear_entity_flags(entity, Entity::INHERITED_STATIC);
    mark_static_subtree(world, entity);
}

fn mark_static_subtree(world: &mut World, entity: Entity) {
    let child_entities = world
        .get::<Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();
    for child in child_entities {
        let flags = world.get_entity_flags(child);
        if flags & Entity::STATIC == 0 {
            // Child was not static — mark it as inherited
            world.set_entity_flags(child, Entity::STATIC | Entity::INHERITED_STATIC);
        }
        mark_static_subtree(world, child);
    }
}

/// Unmarks an entity as static and re-enables inherited-static descendants.
///
/// Descendants that were manually marked static (have `STATIC` without
/// `INHERITED_STATIC`) are left alone — their subtrees are not traversed.
pub fn unmark_static(world: &mut World, entity: Entity) {
    world.clear_entity_flags(entity, Entity::STATIC | Entity::INHERITED_STATIC);
    unmark_static_subtree(world, entity);
}

fn unmark_static_subtree(world: &mut World, entity: Entity) {
    let child_entities = world
        .get::<Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();
    for child in child_entities {
        let flags = world.get_entity_flags(child);
        if flags & Entity::INHERITED_STATIC != 0 {
            // Child was inherited-static — clear it and recurse
            world.clear_entity_flags(child, Entity::STATIC | Entity::INHERITED_STATIC);
            unmark_static_subtree(world, child);
        } else if flags & Entity::STATIC == 0 {
            // Child is not static — recurse in case deeper descendants are inherited-static
            unmark_static_subtree(world, child);
        }
        // If child has STATIC but NOT INHERITED_STATIC, it was manually marked — skip
    }
}

// ---- Entity editor marking (always recursive, flag-based) ----

/// Marks an entity and all descendants as editor entities recursively.
///
/// The target entity gets the `EDITOR` flag set and `INHERITED_EDITOR`
/// cleared (marking it as manually editor). Descendants that are not
/// already editor receive both `EDITOR` and `INHERITED_EDITOR` flags.
/// Already-editor descendants keep their manual status.
///
/// Editor entities are excluded from both `Read<T>` and `Write<T>` queries.
/// Use `ReadAll<T>` / `WriteAll<T>` to include them, or access them
/// directly via exclusive systems (`&mut World`).
pub fn mark_editor(world: &mut World, entity: Entity) {
    world.set_entity_flags(entity, Entity::EDITOR);
    world.clear_entity_flags(entity, Entity::INHERITED_EDITOR);
    mark_editor_subtree(world, entity);
}

fn mark_editor_subtree(world: &mut World, entity: Entity) {
    let child_entities = world
        .get::<Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();
    for child in child_entities {
        let flags = world.get_entity_flags(child);
        if flags & Entity::EDITOR == 0 {
            // Child was not editor — mark it as inherited
            world.set_entity_flags(child, Entity::EDITOR | Entity::INHERITED_EDITOR);
        }
        mark_editor_subtree(world, child);
    }
}

/// Unmarks an entity as editor and re-enables inherited-editor descendants.
///
/// Descendants that were manually marked editor (have `EDITOR` without
/// `INHERITED_EDITOR`) are left alone — their subtrees are not traversed.
pub fn unmark_editor(world: &mut World, entity: Entity) {
    world.clear_entity_flags(entity, Entity::EDITOR | Entity::INHERITED_EDITOR);
    unmark_editor_subtree(world, entity);
}

fn unmark_editor_subtree(world: &mut World, entity: Entity) {
    let child_entities = world
        .get::<Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();
    for child in child_entities {
        let flags = world.get_entity_flags(child);
        if flags & Entity::INHERITED_EDITOR != 0 {
            // Child was inherited-editor — clear it and recurse
            world.clear_entity_flags(child, Entity::EDITOR | Entity::INHERITED_EDITOR);
            unmark_editor_subtree(world, child);
        } else if flags & Entity::EDITOR == 0 {
            // Child is not editor — recurse in case deeper descendants are inherited-editor
            unmark_editor_subtree(world, child);
        }
        // If child has EDITOR but NOT INHERITED_EDITOR, it was manually marked — skip
    }
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

    /// Queues a [`disable`] command (always recursive).
    fn cmd_disable(&self, entity: Entity);

    /// Queues an [`enable`] command (always recursive).
    fn cmd_enable(&self, entity: Entity);

    /// Queues a [`mark_static`] command (always recursive).
    fn cmd_mark_static(&self, entity: Entity);

    /// Queues an [`unmark_static`] command (always recursive).
    fn cmd_unmark_static(&self, entity: Entity);

    /// Queues a [`mark_editor`] command (always recursive).
    fn cmd_mark_editor(&self, entity: Entity);

    /// Queues an [`unmark_editor`] command (always recursive).
    fn cmd_unmark_editor(&self, entity: Entity);
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

    fn cmd_disable(&self, entity: Entity) {
        self.push(move |world| {
            disable(world, entity);
        });
    }

    fn cmd_enable(&self, entity: Entity) {
        self.push(move |world| {
            enable(world, entity);
        });
    }

    fn cmd_mark_static(&self, entity: Entity) {
        self.push(move |world| {
            mark_static(world, entity);
        });
    }

    fn cmd_unmark_static(&self, entity: Entity) {
        self.push(move |world| {
            unmark_static(world, entity);
        });
    }

    fn cmd_mark_editor(&self, entity: Entity) {
        self.push(move |world| {
            mark_editor(world, entity);
        });
    }

    fn cmd_unmark_editor(&self, entity: Entity) {
        self.push(move |world| {
            unmark_editor(world, entity);
        });
    }
}

impl HierarchyCommands for CommandCollector {
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

    fn cmd_disable(&self, entity: Entity) {
        self.push(move |world| {
            disable(world, entity);
        });
    }

    fn cmd_enable(&self, entity: Entity) {
        self.push(move |world| {
            enable(world, entity);
        });
    }

    fn cmd_mark_static(&self, entity: Entity) {
        self.push(move |world| {
            mark_static(world, entity);
        });
    }

    fn cmd_unmark_static(&self, entity: Entity) {
        self.push(move |world| {
            unmark_static(world, entity);
        });
    }

    fn cmd_mark_editor(&self, entity: Entity) {
        self.push(move |world| {
            mark_editor(world, entity);
        });
    }

    fn cmd_unmark_editor(&self, entity: Entity) {
        self.push(move |world| {
            unmark_editor(world, entity);
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
    #[should_panic(expected = "would create a hierarchy cycle")]
    fn set_parent_cycle_panics() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let a = world.spawn();
        let b = world.spawn();
        let c = world.spawn();
        set_parent(&mut world, b, a); // a -> b
        set_parent(&mut world, c, b); // a -> b -> c
        set_parent(&mut world, a, c); // would close the loop a -> b -> c -> a
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
    fn set_parent_marks_transform_changed() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        world.register_component::<Transform>();
        let parent = world.spawn();
        let child = world.spawn();
        world.insert(child, Transform::default()).unwrap();

        // Advance so the insert no longer counts as changed this frame.
        world.advance_tick();
        world.advance_tick();
        let since = world.current_tick().saturating_sub(1);
        assert!(!world.changed::<Transform>(since).matches(child.index()));

        set_parent(&mut world, child, parent);

        let since = world.current_tick().saturating_sub(1);
        assert!(
            world.changed::<Transform>(since).matches(child.index()),
            "reparenting must mark the child's Transform as changed"
        );
    }

    #[test]
    fn remove_parent_marks_transform_changed() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        world.register_component::<Transform>();
        let parent = world.spawn();
        let child = world.spawn();
        world.insert(child, Transform::default()).unwrap();
        set_parent(&mut world, child, parent);

        world.advance_tick();
        world.advance_tick();
        let since = world.current_tick().saturating_sub(1);
        assert!(!world.changed::<Transform>(since).matches(child.index()));

        remove_parent(&mut world, child);

        let since = world.current_tick().saturating_sub(1);
        assert!(
            world.changed::<Transform>(since).matches(child.index()),
            "unparenting must mark the child's Transform as changed"
        );
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

    // ---- #68: inherited-flag propagation on reparenting ----

    #[test]
    fn reparent_under_disabled_parent_inherits() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let parent = world.spawn();
        let child = world.spawn();
        let grandchild = world.spawn();
        set_parent(&mut world, grandchild, child);

        disable(&mut world, parent);
        set_parent(&mut world, child, parent);

        for e in [child, grandchild] {
            let flags = world.get_entity_flags(e);
            assert_ne!(flags & Entity::DISABLED, 0, "{e} inherits DISABLED");
            assert_ne!(
                flags & Entity::INHERITED_DISABLED,
                0,
                "{e} marked as inherited, not manual"
            );
        }
    }

    #[test]
    fn reparent_out_sheds_inherited_flags() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let parent = world.spawn();
        let other = world.spawn();
        let child = world.spawn();
        let grandchild = world.spawn();
        set_parent(&mut world, grandchild, child);

        mark_editor(&mut world, parent);
        set_parent(&mut world, child, parent);
        assert_ne!(world.get_entity_flags(grandchild) & Entity::EDITOR, 0);

        // Move the subtree under a clean parent: inherited state sheds.
        set_parent(&mut world, child, other);
        for e in [child, grandchild] {
            assert_eq!(
                world.get_entity_flags(e) & (Entity::EDITOR | Entity::INHERITED_EDITOR),
                0,
                "{e} sheds inherited editor"
            );
        }
    }

    #[test]
    fn remove_parent_sheds_inherited_flags() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let parent = world.spawn();
        let child = world.spawn();
        disable(&mut world, parent);
        set_parent(&mut world, child, parent);
        assert_ne!(world.get_entity_flags(child) & Entity::DISABLED, 0);

        remove_parent(&mut world, child);
        assert_eq!(
            world.get_entity_flags(child) & (Entity::DISABLED | Entity::INHERITED_DISABLED),
            0,
            "orphaned entity sheds inherited flags"
        );
    }

    /// The #68 acceptance scenario: a manually flagged node keeps the user's
    /// intent wherever it moves — an editor gizmo stays EDITOR under a game
    /// parent, a manually disabled node stays disabled under a clean parent.
    #[test]
    fn manual_flags_survive_reparenting() {
        let mut world = World::new();
        register_hierarchy(&mut world);
        let game_parent = world.spawn();
        let gizmo = world.spawn();
        crate::mark_editor(&mut world, gizmo);
        let disabled = world.spawn();
        disable(&mut world, disabled);

        set_parent(&mut world, gizmo, game_parent);
        let flags = world.get_entity_flags(gizmo);
        assert_ne!(flags & Entity::EDITOR, 0, "gizmo keeps manual EDITOR");
        assert_eq!(flags & Entity::INHERITED_EDITOR, 0, "still manual");

        set_parent(&mut world, disabled, game_parent);
        let flags = world.get_entity_flags(disabled);
        assert_ne!(flags & Entity::DISABLED, 0, "manual DISABLED survives");
        assert_eq!(flags & Entity::INHERITED_DISABLED, 0);
    }

    /// #71 property test: random hierarchies mutated by random reparent /
    /// flag operations always settle to the inherited-flag fixed point —
    /// every node's inherited bits equal the derivation from its parent's
    /// effective flags, and manual bits are never disturbed. Seeded LCG so
    /// failures reproduce.
    #[test]
    fn random_hierarchy_inherited_flags_reach_fixed_point() {
        let mut rng_state: u64 = 0x5EED_CAFE_F00D_0001;
        let mut rng = move || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng_state >> 33) as usize
        };

        for _round in 0..20 {
            let mut world = World::new();
            register_hierarchy(&mut world);
            let n = 12;
            let entities: Vec<Entity> = (0..n).map(|_| world.spawn()).collect();

            // Random forest (parent strictly earlier -> acyclic), random flags.
            for i in 1..n {
                if rng() % 4 != 0 {
                    let p = entities[rng() % i];
                    set_parent(&mut world, entities[i], p);
                }
            }
            // Random mutations: manual flag walkers + reparenting.
            for _ in 0..30 {
                let e = entities[rng() % n];
                match rng() % 4 {
                    0 => disable(&mut world, e),
                    1 => enable(&mut world, e),
                    2 => {
                        let p = entities[rng() % n];
                        if e != p && !is_ancestor_of(&world, e, p) {
                            set_parent(&mut world, e, p);
                        }
                    }
                    _ => remove_parent(&mut world, e),
                }
            }

            // Fixed point: every node's inherited bits derive exactly from
            // its parent's effective flags.
            for &e in &entities {
                let flags = world.get_entity_flags(e);
                let parent_flags = world
                    .get::<Parent>(e)
                    .map(|p| world.get_entity_flags(p.0))
                    .unwrap_or(0);
                for (flag, inherited) in super::INHERITED_FLAG_FAMILIES {
                    let manual = flags & flag != 0 && flags & inherited == 0;
                    if manual {
                        continue;
                    }
                    let expect = parent_flags & flag != 0;
                    let actual = flags & inherited != 0;
                    assert_eq!(
                        actual, expect,
                        "{e}: inherited bit for flag {flag:#x} diverged from \
                         parent derivation (flags {flags:#x}, parent {parent_flags:#x})"
                    );
                    assert_eq!(
                        flags & flag != 0,
                        expect,
                        "{e}: flag {flag:#x} inconsistent with inherited bit"
                    );
                }
            }
        }
    }
}
