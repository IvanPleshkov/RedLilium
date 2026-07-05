use crate::query::ChangedFilter;
use crate::{Ref, RefMut, SystemContext};
use redlilium_core::math::Mat4;

use crate::std::components::{Children, GlobalTransform, Parent, Transform};

/// System that updates all [`GlobalTransform`] components from local [`Transform`],
/// respecting the parent-child hierarchy.
///
/// Uses [`Changed<Transform>`](crate::Changed) to skip unchanged subtrees —
/// only entities whose local `Transform` was modified (or whose ancestor was
/// modified) get their `GlobalTransform` rewritten.
///
/// # Access
///
/// - Reads: `Transform`, `Parent`, `Children`, `Changed<Transform>`
/// - Writes: `GlobalTransform`
pub struct UpdateGlobalTransforms;

impl crate::System for UpdateGlobalTransforms {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::ReadAll<Transform>,
            crate::WriteAll<GlobalTransform>,
            crate::ReadAll<Children>,
            crate::ReadAll<Parent>,
            crate::MaybeChanged<Transform>,
        )>()
        .execute(
            |(transforms, mut globals, children_storage, parents, changed)| {
                update_global_transforms(
                    &transforms,
                    &mut globals,
                    &children_storage,
                    &parents,
                    &changed,
                    &|e| ctx.world().is_alive(e),
                );
            },
        );
        Ok(())
    }
}

fn update_global_transforms(
    transforms: &Ref<Transform>,
    globals: &mut RefMut<GlobalTransform>,
    children_storage: &Ref<Children>,
    parents: &Ref<Parent>,
    changed: &ChangedFilter<'_>,
    is_alive: &dyn Fn(crate::Entity) -> bool,
) {
    redlilium_core::profile_scope!("update_global_transforms");

    // Process root entities. An entity is a propagation root if it has no
    // `Parent`, or its `Parent` points to a dead entity (e.g. after the parent
    // was despawned without despawning the subtree) — otherwise such an orphan
    // would never be reached by the downward `Children` walk and would keep a
    // stale `GlobalTransform`.
    for (idx, transform) in transforms.iter() {
        let is_root = match parents.get(idx) {
            None => true,
            Some(parent) => !is_alive(parent.0),
        };
        if is_root {
            if changed.matches(idx) {
                let local_matrix = transform.to_matrix();
                if let Some(mut gt) = globals.get_mut(idx) {
                    gt.0 = local_matrix;
                }
                propagate_children(
                    idx,
                    local_matrix,
                    transforms,
                    globals,
                    children_storage,
                    changed,
                    true,
                );
            } else {
                // Root unchanged — read existing GlobalTransform, still check children
                let parent_world = globals.get(idx).map(|gt| gt.0).unwrap_or(Mat4::identity());
                propagate_children(
                    idx,
                    parent_world,
                    transforms,
                    globals,
                    children_storage,
                    changed,
                    false,
                );
            }
        }
    }
}

/// Recursively propagates world transforms from parent to children.
///
/// When `parent_changed` is `true`, all descendants are recomputed.
/// When `false`, only descendants with a changed `Transform` are updated;
/// unchanged nodes reuse their existing `GlobalTransform`.
fn propagate_children(
    parent_idx: u32,
    parent_world: Mat4,
    transforms: &Ref<Transform>,
    globals: &mut RefMut<GlobalTransform>,
    children_storage: &Ref<Children>,
    changed: &ChangedFilter<'_>,
    parent_changed: bool,
) {
    let Some(children) = children_storage.get(parent_idx) else {
        return;
    };
    for &child in children.0.iter() {
        let child_idx = child.index();
        let needs_update = parent_changed || changed.matches(child_idx);

        if needs_update {
            let child_world = if let Some(local) = transforms.get(child_idx) {
                parent_world * local.to_matrix()
            } else {
                parent_world
            };
            if let Some(mut gt) = globals.get_mut(child_idx) {
                gt.0 = child_world;
            }
            propagate_children(
                child_idx,
                child_world,
                transforms,
                globals,
                children_storage,
                changed,
                true,
            );
        } else {
            // Neither parent nor this child changed — reuse existing GlobalTransform
            let child_world = globals
                .get(child_idx)
                .map(|gt| gt.0)
                .unwrap_or(parent_world);
            propagate_children(
                child_idx,
                child_world,
                transforms,
                globals,
                children_storage,
                changed,
                false,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::World;
    use crate::std::hierarchy::set_parent;
    use redlilium_core::math::{Vec3, quat_from_rotation_y};

    /// Helper: register hierarchy + transform components so tests don't panic.
    fn register_hierarchy(world: &mut World) {
        world.register_component::<Transform>();
        world.register_component::<GlobalTransform>();
        world.register_component::<Parent>();
        world.register_component::<Children>();
        // Advance tick so insert stamps ticks_changed = 1,
        // allowing Changed<Transform> to detect them (since_tick = 0, 1 > 0 = true).
        world.advance_tick();
    }

    /// Helper: get component borrows and run update.
    fn run_update(world: &World) {
        let transforms = world.read::<Transform>().unwrap();
        let mut globals = world.write_storage::<GlobalTransform>().unwrap();
        let children_storage = world.read::<Children>().unwrap();
        let parents = world.read::<Parent>().unwrap();
        let since_tick = world.current_tick().saturating_sub(1);
        let changed = world.changed::<Transform>(since_tick);
        update_global_transforms(
            &transforms,
            &mut globals,
            &children_storage,
            &parents,
            &changed,
            &|e| world.is_alive(e),
        );
    }

    #[test]
    fn updates_global_from_local() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let e = world.spawn();
        let t = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        world.insert(e, t).unwrap();
        world.insert(e, GlobalTransform::IDENTITY).unwrap();

        run_update(&world);

        let globals = world.read::<GlobalTransform>().unwrap();
        let global = globals.get(e.index()).unwrap();
        assert!((global.translation() - Vec3::new(1.0, 2.0, 3.0)).norm() < 1e-6);
    }

    #[test]
    fn orphan_with_dead_parent_is_updated() {
        // A child whose parent was despawned (leaving a stale Parent) must still
        // have its GlobalTransform recomputed as a root, not silently skipped.
        let mut world = World::new();
        register_hierarchy(&mut world);

        let parent = world.spawn();
        world.insert(parent, Transform::default()).unwrap();
        world.insert(parent, GlobalTransform::IDENTITY).unwrap();

        let child = world.spawn();
        world
            .insert(child, Transform::from_translation(Vec3::new(5.0, 0.0, 0.0)))
            .unwrap();
        world.insert(child, GlobalTransform::IDENTITY).unwrap();
        set_parent(&mut world, child, parent);

        // Despawn only the parent, leaving the child with a stale Parent.
        world.despawn(parent);
        assert!(!world.is_alive(parent));

        run_update(&world);

        let globals = world.read::<GlobalTransform>().unwrap();
        let global = globals.get(child.index()).unwrap();
        assert!(
            (global.translation() - Vec3::new(5.0, 0.0, 0.0)).norm() < 1e-6,
            "orphaned child should use its local transform as world transform"
        );
    }

    #[test]
    fn propagation_works_without_manual_tick_advance() {
        // Historical regression: the world tick used to start at 0, so writes
        // stamped at tick 0 could never satisfy the strict `>` change
        // comparison and demos that never called advance_tick had frozen
        // cameras. The tick now starts at 1, so setup-time inserts are
        // visible to change filters with no manual advancing at all.
        let mut world = World::new();
        world.register_component::<Transform>();
        world.register_component::<GlobalTransform>();
        world.register_component::<Parent>();
        world.register_component::<Children>();
        // Deliberately no advance_tick anywhere in this test.

        let e = world.spawn();
        world
            .insert(e, Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        world.insert(e, GlobalTransform::IDENTITY).unwrap();

        run_update(&world);
        {
            let globals = world.read::<GlobalTransform>().unwrap();
            assert!(
                (globals.get(e.index()).unwrap().translation() - Vec3::new(1.0, 0.0, 0.0)).norm()
                    < 1e-6,
                "setup-time insert must be visible to Changed<Transform> and propagate"
            );
        }
    }

    #[test]
    fn rotation_propagates() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let e = world.spawn();
        let t = Transform::from_rotation(quat_from_rotation_y(std::f32::consts::FRAC_PI_2));
        world.insert(e, t).unwrap();
        world.insert(e, GlobalTransform::IDENTITY).unwrap();

        run_update(&world);

        let globals = world.read::<GlobalTransform>().unwrap();
        let global = globals.get(e.index()).unwrap();
        let forward = global.forward();
        assert!((forward - Vec3::new(-1.0, 0.0, 0.0)).norm() < 1e-5);
    }

    #[test]
    fn skips_entities_without_global() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let e = world.spawn();
        world.insert(e, Transform::IDENTITY).unwrap();
        world.register_component::<GlobalTransform>();

        run_update(&world);
    }

    #[test]
    fn child_inherits_parent_translation() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let parent = world.spawn();
        world
            .insert(
                parent,
                Transform::from_translation(Vec3::new(10.0, 0.0, 0.0)),
            )
            .unwrap();
        world.insert(parent, GlobalTransform::IDENTITY).unwrap();

        let child = world.spawn();
        world
            .insert(child, Transform::from_translation(Vec3::new(0.0, 5.0, 0.0)))
            .unwrap();
        world.insert(child, GlobalTransform::IDENTITY).unwrap();

        set_parent(&mut world, child, parent);
        run_update(&world);

        let globals = world.read::<GlobalTransform>().unwrap();
        let child_global = globals.get(child.index()).unwrap();
        assert!((child_global.translation() - Vec3::new(10.0, 5.0, 0.0)).norm() < 1e-6);
    }

    #[test]
    fn grandchild_inherits_full_chain() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let root = world.spawn();
        world
            .insert(root, Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)))
            .unwrap();
        world.insert(root, GlobalTransform::IDENTITY).unwrap();

        let mid = world.spawn();
        world
            .insert(mid, Transform::from_translation(Vec3::new(0.0, 2.0, 0.0)))
            .unwrap();
        world.insert(mid, GlobalTransform::IDENTITY).unwrap();

        let leaf = world.spawn();
        world
            .insert(leaf, Transform::from_translation(Vec3::new(0.0, 0.0, 3.0)))
            .unwrap();
        world.insert(leaf, GlobalTransform::IDENTITY).unwrap();

        set_parent(&mut world, mid, root);
        set_parent(&mut world, leaf, mid);
        run_update(&world);

        let globals = world.read::<GlobalTransform>().unwrap();
        let leaf_global = globals.get(leaf.index()).unwrap();
        assert!((leaf_global.translation() - Vec3::new(1.0, 2.0, 3.0)).norm() < 1e-6);
    }

    #[test]
    fn parent_rotation_affects_child_position() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let parent = world.spawn();
        world
            .insert(
                parent,
                Transform::from_rotation(quat_from_rotation_y(std::f32::consts::FRAC_PI_2)),
            )
            .unwrap();
        world.insert(parent, GlobalTransform::IDENTITY).unwrap();

        let child = world.spawn();
        world
            .insert(child, Transform::from_translation(Vec3::new(0.0, 0.0, 1.0)))
            .unwrap();
        world.insert(child, GlobalTransform::IDENTITY).unwrap();

        set_parent(&mut world, child, parent);
        run_update(&world);

        let globals = world.read::<GlobalTransform>().unwrap();
        let child_global = globals.get(child.index()).unwrap();
        assert!((child_global.translation() - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-5);
    }
}
