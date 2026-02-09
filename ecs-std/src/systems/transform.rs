use redlilium_ecs::World;

use crate::components::{GlobalTransform, Transform};

/// Updates all [`GlobalTransform`] components from their local [`Transform`].
///
/// Currently computes the world matrix directly from the local TRS since
/// hierarchy (Parent/Children) is not yet available. When hierarchy is
/// added (ECS Phase 3), this system will be extended to multiply the
/// parent's GlobalTransform with the child's local Transform.
///
/// # Access
///
/// - Reads: `Transform`
/// - Writes: `GlobalTransform`
pub fn update_global_transforms(world: &World) {
    redlilium_core::profile_scope!("update_global_transforms");

    let transforms = world.read::<Transform>();
    let mut globals = world.write::<GlobalTransform>();

    for (idx, transform) in transforms.iter() {
        if let Some(global) = globals.get_mut(idx) {
            global.0 = transform.to_matrix();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn updates_global_from_local() {
        let mut world = World::new();

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

        // Entity with Transform but no GlobalTransform — should not panic
        let e = world.spawn();
        world.insert(e, Transform::IDENTITY);
        world.register_component::<GlobalTransform>();

        update_global_transforms(&world);
    }
}
