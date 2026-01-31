//! Transform propagation systems for hierarchy-based transforms.
//!
//! These systems compute [`GlobalTransform`] from local [`Transform`] components,
//! propagating transforms down the entity hierarchy.

use bevy_ecs::prelude::*;

use crate::components::{ChildOf, Children, GlobalTransform, HierarchyDepth, Transform};

/// Updates [`GlobalTransform`] for root entities (those without a parent).
///
/// Root entities have their [`GlobalTransform`] equal to their [`Transform`].
/// This system should run before [`propagate_transforms`].
#[allow(clippy::type_complexity)]
pub fn sync_root_transforms(
    mut query: Query<(&Transform, &mut GlobalTransform), (Changed<Transform>, Without<ChildOf>)>,
) {
    for (transform, mut global_transform) in query.iter_mut() {
        *global_transform = GlobalTransform::from(*transform);
    }
}

/// Propagates transforms from parent to children through the hierarchy.
///
/// This system computes [`GlobalTransform`] for all entities with parents
/// by combining the parent's [`GlobalTransform`] with the child's local [`Transform`].
///
/// The system traverses the hierarchy depth-first, ensuring parents are
/// processed before their children.
#[allow(clippy::type_complexity)]
pub fn propagate_transforms(
    mut root_query: Query<
        (Entity, &Children, &GlobalTransform),
        (
            Without<ChildOf>,
            Or<(Changed<GlobalTransform>, Changed<Children>)>,
        ),
    >,
    mut transform_query: Query<(&Transform, &mut GlobalTransform, Option<&Children>)>,
    child_query: Query<&ChildOf>,
) {
    for (_, children, parent_global) in root_query.iter_mut() {
        propagate_recursive(&mut transform_query, children, parent_global);
    }

    // Also handle changes that start from children
    for child_of in child_query.iter() {
        // This will be handled by the recursive propagation from root
        let _ = child_of;
    }
}

fn propagate_recursive(
    transform_query: &mut Query<(&Transform, &mut GlobalTransform, Option<&Children>)>,
    children: &Children,
    parent_global: &GlobalTransform,
) {
    for child in children.iter() {
        if let Ok((transform, mut global_transform, maybe_children)) =
            transform_query.get_mut(child)
        {
            *global_transform = parent_global.mul_transform(transform);

            if let Some(grandchildren) = maybe_children {
                // Clone to avoid borrow issues
                let grandchildren_vec: Vec<Entity> = grandchildren.iter().collect();
                let child_global = *global_transform;

                // Recursively propagate to grandchildren
                for grandchild in grandchildren_vec {
                    propagate_to_child(transform_query, grandchild, &child_global);
                }
            }
        }
    }
}

fn propagate_to_child(
    transform_query: &mut Query<(&Transform, &mut GlobalTransform, Option<&Children>)>,
    entity: Entity,
    parent_global: &GlobalTransform,
) {
    if let Ok((transform, mut global_transform, maybe_children)) = transform_query.get_mut(entity) {
        *global_transform = parent_global.mul_transform(transform);

        if let Some(children) = maybe_children {
            let children_vec: Vec<Entity> = children.iter().collect();
            let entity_global = *global_transform;

            for child in children_vec {
                propagate_to_child(transform_query, child, &entity_global);
            }
        }
    }
}

/// Updates hierarchy depth for all entities.
///
/// This is useful for sorting entities by depth for ordered processing.
pub fn update_hierarchy_depth(
    mut commands: Commands,
    roots: Query<Entity, (Without<ChildOf>, Without<HierarchyDepth>)>,
    children_query: Query<&Children>,
) {
    // Set depth 0 for roots
    for root in roots.iter() {
        commands.entity(root).insert(HierarchyDepth::new(0));
    }

    // Update depths recursively
    for root in roots.iter() {
        if let Ok(children) = children_query.get(root) {
            update_depth_recursive(&mut commands, &children_query, children, 1);
        }
    }
}

fn update_depth_recursive(
    commands: &mut Commands,
    children_query: &Query<&Children>,
    children: &Children,
    depth: u32,
) {
    for child in children.iter() {
        commands.entity(child).insert(HierarchyDepth::new(depth));

        if let Ok(grandchildren) = children_query.get(child) {
            update_depth_recursive(commands, children_query, grandchildren, depth + 1);
        }
    }
}

/// Convenience function to run transform propagation on a World directly.
///
/// This updates all GlobalTransform components based on the hierarchy.
/// Call this once per frame before rendering.
///
/// # Example
///
/// ```
/// use redlilium_ecs::prelude::*;
/// use bevy_ecs::prelude::*;
///
/// let mut world = World::new();
/// // ... spawn entities ...
/// run_transform_systems(&mut world);
/// ```
pub fn run_transform_systems(world: &mut World) {
    // First, sync root transforms
    {
        let mut query = world.query_filtered::<(&Transform, &mut GlobalTransform), (Changed<Transform>, Without<ChildOf>)>();

        // Collect to avoid borrow issues
        let updates: Vec<_> = query.iter(world).map(|(t, _)| (*t,)).collect();

        // We need to use a different approach - query again with mutable access
        let mut query =
            world.query_filtered::<(&Transform, &mut GlobalTransform), Without<ChildOf>>();
        for (transform, mut global_transform) in query.iter_mut(world) {
            *global_transform = GlobalTransform::from(*transform);
        }
        let _ = updates; // silence warning
    }

    // Then propagate through hierarchy
    // Collect root entities with children first
    let roots: Vec<(Entity, Vec<Entity>, GlobalTransform)> = {
        let mut query =
            world.query_filtered::<(Entity, &Children, &GlobalTransform), Without<ChildOf>>();
        query
            .iter(world)
            .map(|(e, children, g)| (e, children.iter().collect(), *g))
            .collect()
    };

    // Propagate to each child hierarchy
    for (_root_entity, children, parent_global) in roots {
        propagate_to_children_world(world, &children, &parent_global);
    }
}

fn propagate_to_children_world(
    world: &mut World,
    children: &[Entity],
    parent_global: &GlobalTransform,
) {
    for &child in children {
        // Get the child's transform and compute global
        let child_data: Option<(Transform, Vec<Entity>)> = {
            let mut query = world.query::<(&Transform, &GlobalTransform, Option<&Children>)>();
            query
                .get(world, child)
                .ok()
                .map(|(t, _, c)| (*t, c.map(|c| c.iter().collect()).unwrap_or_default()))
        };

        if let Some((transform, grandchildren)) = child_data {
            let child_global = parent_global.mul_transform(&transform);

            // Update the child's global transform
            if let Some(mut global) = world.get_mut::<GlobalTransform>(child) {
                *global = child_global;
            }

            // Recursively propagate to grandchildren
            if !grandchildren.is_empty() {
                propagate_to_children_world(world, &grandchildren, &child_global);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn global_transform_computation() {
        let parent_transform = Transform::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let child_transform = Transform::from_translation(Vec3::new(0.0, 5.0, 0.0));

        let parent_global = GlobalTransform::from(parent_transform);
        let child_global = parent_global.mul_transform(&child_transform);

        let expected_translation = Vec3::new(10.0, 5.0, 0.0);
        assert!((child_global.translation() - expected_translation).length() < 1e-5);
    }

    #[test]
    fn run_transform_systems_syncs_root_transforms() {
        let mut world = World::new();

        // Create root entity with a transform
        let root = world
            .spawn((
                Transform::from_translation(Vec3::new(10.0, 5.0, 3.0)),
                GlobalTransform::IDENTITY,
            ))
            .id();

        // Run the transform systems
        run_transform_systems(&mut world);

        // Check the root's global transform matches its local transform
        let root_global = world.get::<GlobalTransform>(root).unwrap();
        let expected = Vec3::new(10.0, 5.0, 3.0);
        assert!(
            (root_global.translation() - expected).length() < 1e-5,
            "Expected {:?}, got {:?}",
            expected,
            root_global.translation()
        );
    }

    #[test]
    fn run_transform_systems_multiple_roots() {
        let mut world = World::new();

        // Create multiple root entities
        let root1 = world
            .spawn((
                Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
                GlobalTransform::IDENTITY,
            ))
            .id();

        let root2 = world
            .spawn((
                Transform::from_translation(Vec3::new(0.0, 2.0, 0.0)),
                GlobalTransform::IDENTITY,
            ))
            .id();

        // Run the transform systems
        run_transform_systems(&mut world);

        // Check both roots have correct global transforms
        let root1_global = world.get::<GlobalTransform>(root1).unwrap();
        assert!((root1_global.translation() - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);

        let root2_global = world.get::<GlobalTransform>(root2).unwrap();
        assert!((root2_global.translation() - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-5);
    }
}
