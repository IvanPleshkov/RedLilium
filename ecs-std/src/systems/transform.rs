use glam::Mat4;
use redlilium_ecs::{Access, System, SystemContext, World};

use crate::components::{Children, GlobalTransform, Parent, Transform};

/// System that updates all [`GlobalTransform`] components from local [`Transform`],
/// respecting the parent-child hierarchy.
///
/// Root entities get `GlobalTransform` directly from `Transform`. Children
/// have their parent's world matrix multiplied with their local matrix.
///
/// # Access
///
/// - Reads: `Transform`, `Parent`, `Children`
/// - Writes: `GlobalTransform`
pub struct UpdateGlobalTransforms;

impl System for UpdateGlobalTransforms {
    fn run(&self, ctx: &SystemContext) {
        update_global_transforms(ctx.world());
    }

    fn access(&self) -> Access {
        let mut access = Access::new();
        access.add_read::<Transform>();
        access.add_read::<Parent>();
        access.add_read::<Children>();
        access.add_write::<GlobalTransform>();
        access
    }
}

fn update_global_transforms(world: &World) {
    redlilium_core::profile_scope!("update_global_transforms");

    let transforms = world.read::<Transform>();
    let mut globals = world.write::<GlobalTransform>();
    let children_storage = world.read::<Children>();
    let has_parent = world.with::<Parent>();

    // Process root entities (no Parent component)
    for (idx, transform) in transforms.iter() {
        if !has_parent.matches(idx) {
            let local_matrix = transform.to_matrix();
            if let Some(gt) = globals.get_mut(idx) {
                gt.0 = local_matrix;
            }
            // Recursively propagate to children
            propagate_children(
                idx,
                local_matrix,
                &transforms,
                &mut globals,
                &children_storage,
            );
        }
    }
}

/// Recursively propagates world transforms from parent to children.
fn propagate_children(
    parent_idx: u32,
    parent_world: Mat4,
    transforms: &redlilium_ecs::Ref<Transform>,
    globals: &mut redlilium_ecs::RefMut<GlobalTransform>,
    children_storage: &redlilium_ecs::Ref<Children>,
) {
    let Some(children) = children_storage.get(parent_idx) else {
        return;
    };
    for &child in children.0.iter() {
        let child_idx = child.index();
        let child_world = if let Some(local) = transforms.get(child_idx) {
            parent_world * local.to_matrix()
        } else {
            parent_world
        };
        if let Some(gt) = globals.get_mut(child_idx) {
            gt.0 = child_world;
        }
        propagate_children(
            child_idx,
            child_world,
            transforms,
            globals,
            children_storage,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::set_parent;
    use glam::{Quat, Vec3};

    /// Helper: register hierarchy components so tests don't panic.
    fn register_hierarchy(world: &mut World) {
        world.register_component::<Parent>();
        world.register_component::<Children>();
    }

    #[test]
    fn updates_global_from_local() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let e = world.spawn();
        let t = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        world.insert(e, t);
        world.insert(e, GlobalTransform::IDENTITY);

        update_global_transforms(&world);

        let globals = world.read::<GlobalTransform>();
        let global = globals.get(e.index()).unwrap();
        assert!((global.translation() - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
    }

    #[test]
    fn rotation_propagates() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let e = world.spawn();
        let t = Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
        world.insert(e, t);
        world.insert(e, GlobalTransform::IDENTITY);

        update_global_transforms(&world);

        let globals = world.read::<GlobalTransform>();
        let global = globals.get(e.index()).unwrap();
        // After 90-degree Y rotation, forward (-Z) rotates to -X
        let forward = global.forward();
        assert!((forward - Vec3::NEG_X).length() < 1e-5);
    }

    #[test]
    fn skips_entities_without_global() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        // Entity with Transform but no GlobalTransform — should not panic
        let e = world.spawn();
        world.insert(e, Transform::IDENTITY);
        world.register_component::<GlobalTransform>();

        update_global_transforms(&world);
    }

    #[test]
    fn child_inherits_parent_translation() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let parent = world.spawn();
        world.insert(
            parent,
            Transform::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        );
        world.insert(parent, GlobalTransform::IDENTITY);

        let child = world.spawn();
        world.insert(child, Transform::from_translation(Vec3::new(0.0, 5.0, 0.0)));
        world.insert(child, GlobalTransform::IDENTITY);

        set_parent(&mut world, child, parent);
        update_global_transforms(&world);

        let globals = world.read::<GlobalTransform>();
        let child_global = globals.get(child.index()).unwrap();
        // Child at (0,5,0) relative to parent at (10,0,0) → world (10,5,0)
        assert!((child_global.translation() - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn grandchild_inherits_full_chain() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let root = world.spawn();
        world.insert(root, Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)));
        world.insert(root, GlobalTransform::IDENTITY);

        let mid = world.spawn();
        world.insert(mid, Transform::from_translation(Vec3::new(0.0, 2.0, 0.0)));
        world.insert(mid, GlobalTransform::IDENTITY);

        let leaf = world.spawn();
        world.insert(leaf, Transform::from_translation(Vec3::new(0.0, 0.0, 3.0)));
        world.insert(leaf, GlobalTransform::IDENTITY);

        set_parent(&mut world, mid, root);
        set_parent(&mut world, leaf, mid);
        update_global_transforms(&world);

        let globals = world.read::<GlobalTransform>();
        let leaf_global = globals.get(leaf.index()).unwrap();
        // root(1,0,0) + mid(0,2,0) + leaf(0,0,3) = (1,2,3)
        assert!((leaf_global.translation() - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
    }

    #[test]
    fn parent_rotation_affects_child_position() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        // Parent rotated 90° around Y
        let parent = world.spawn();
        world.insert(
            parent,
            Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
        );
        world.insert(parent, GlobalTransform::IDENTITY);

        // Child offset along +Z
        let child = world.spawn();
        world.insert(child, Transform::from_translation(Vec3::new(0.0, 0.0, 1.0)));
        world.insert(child, GlobalTransform::IDENTITY);

        set_parent(&mut world, child, parent);
        update_global_transforms(&world);

        let globals = world.read::<GlobalTransform>();
        let child_global = globals.get(child.index()).unwrap();
        // Parent 90° Y rotation turns child's +Z into +X
        assert!((child_global.translation() - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
    }
}
