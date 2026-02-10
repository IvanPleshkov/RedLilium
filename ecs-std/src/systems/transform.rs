use glam::Mat4;
use redlilium_ecs::{Ref, RefMut, SystemContext};

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

impl redlilium_ecs::System for UpdateGlobalTransforms {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(
            redlilium_ecs::Read<Transform>,
            redlilium_ecs::Write<GlobalTransform>,
            redlilium_ecs::Read<Children>,
            redlilium_ecs::Read<Parent>,
        )>()
        .execute(|(transforms, mut globals, children_storage, parents)| {
            update_global_transforms(&transforms, &mut globals, &children_storage, &parents);
        })
        .await;
    }
}

fn update_global_transforms(
    transforms: &Ref<Transform>,
    globals: &mut RefMut<GlobalTransform>,
    children_storage: &Ref<Children>,
    parents: &Ref<Parent>,
) {
    redlilium_core::profile_scope!("update_global_transforms");

    // Process root entities (no Parent component)
    for (idx, transform) in transforms.iter() {
        if parents.get(idx).is_none() {
            let local_matrix = transform.to_matrix();
            if let Some(gt) = globals.get_mut(idx) {
                gt.0 = local_matrix;
            }
            // Recursively propagate to children
            propagate_children(idx, local_matrix, transforms, globals, children_storage);
        }
    }
}

/// Recursively propagates world transforms from parent to children.
fn propagate_children(
    parent_idx: u32,
    parent_world: Mat4,
    transforms: &Ref<Transform>,
    globals: &mut RefMut<GlobalTransform>,
    children_storage: &Ref<Children>,
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
    use redlilium_ecs::World;

    /// Helper: register hierarchy + transform components so tests don't panic.
    fn register_hierarchy(world: &mut World) {
        world.register_component::<Transform>();
        world.register_component::<GlobalTransform>();
        world.register_component::<Parent>();
        world.register_component::<Children>();
    }

    /// Helper: get component borrows and run update.
    fn run_update(world: &World) {
        let transforms = world.read::<Transform>().unwrap();
        let mut globals = world.write::<GlobalTransform>().unwrap();
        let children_storage = world.read::<Children>().unwrap();
        let parents = world.read::<Parent>().unwrap();
        update_global_transforms(&transforms, &mut globals, &children_storage, &parents);
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
        assert!((global.translation() - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
    }

    #[test]
    fn rotation_propagates() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let e = world.spawn();
        let t = Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
        world.insert(e, t).unwrap();
        world.insert(e, GlobalTransform::IDENTITY).unwrap();

        run_update(&world);

        let globals = world.read::<GlobalTransform>().unwrap();
        let global = globals.get(e.index()).unwrap();
        let forward = global.forward();
        assert!((forward - Vec3::NEG_X).length() < 1e-5);
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
        assert!((child_global.translation() - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-6);
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
        assert!((leaf_global.translation() - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
    }

    #[test]
    fn parent_rotation_affects_child_position() {
        let mut world = World::new();
        register_hierarchy(&mut world);

        let parent = world.spawn();
        world
            .insert(
                parent,
                Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),
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
        assert!((child_global.translation() - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
    }
}
