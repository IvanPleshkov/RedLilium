use redlilium_ecs::World;

use crate::components::{Camera, GlobalTransform};

/// Updates view matrices for all [`Camera`] components.
///
/// Reads [`GlobalTransform`] to compute the view matrix (inverse of the
/// camera's world transform). Projection matrices are set at construction
/// and not modified by this system.
///
/// Must run after [`update_global_transforms`](crate::systems::update_global_transforms).
///
/// # Access
///
/// - Reads: `GlobalTransform`
/// - Writes: `Camera`
pub fn update_camera_matrices(world: &World) {
    redlilium_core::profile_scope!("update_camera_matrices");

    let globals = world.read::<GlobalTransform>();
    let mut cameras = world.write::<Camera>();

    for (idx, camera) in cameras.iter_mut() {
        if let Some(global) = globals.get(idx) {
            camera.view_matrix = global.0.inverse();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Transform;
    use glam::{Mat4, Vec3};

    #[test]
    fn updates_view_matrix() {
        let mut world = World::new();

        let e = world.spawn();
        let t = Transform::from_translation(Vec3::new(0.0, 5.0, 10.0));
        world.insert(e, GlobalTransform(t.to_matrix()));
        world.insert(e, Camera::perspective(1.0, 1.0, 0.1, 100.0));

        update_camera_matrices(&world);

        let cameras = world.read::<Camera>();
        let cam = cameras.get(e.index()).unwrap();

        // View matrix should be the inverse of the global transform
        let expected_view = t.to_matrix().inverse();
        assert!((cam.view_matrix - expected_view).abs_diff_eq(Mat4::ZERO, 1e-5));

        // Projection should still be what was set at construction (not identity)
        assert_ne!(cam.projection_matrix, Mat4::IDENTITY);
    }

    #[test]
    fn skips_cameras_without_global() {
        let mut world = World::new();

        let e = world.spawn();
        world.insert(e, Camera::perspective(1.0, 1.0, 0.1, 100.0));
        world.register_component::<GlobalTransform>();

        update_camera_matrices(&world);

        // Camera view matrix should remain identity (not updated)
        let cameras = world.read::<Camera>();
        let cam = cameras.get(e.index()).unwrap();
        assert_eq!(cam.view_matrix, Mat4::IDENTITY);
    }

    #[test]
    fn projection_preserved() {
        let mut world = World::new();

        let e = world.spawn();
        let cam_original = Camera::perspective(1.0, 1.0, 0.1, 100.0);
        let proj_before = cam_original.projection_matrix;
        world.insert(
            e,
            GlobalTransform(Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))),
        );
        world.insert(e, cam_original);

        update_camera_matrices(&world);

        let cameras = world.read::<Camera>();
        let cam = cameras.get(e.index()).unwrap();
        // Projection matrix unchanged by system
        assert_eq!(cam.projection_matrix, proj_before);
    }
}
