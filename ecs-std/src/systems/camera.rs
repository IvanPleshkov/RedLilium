use redlilium_ecs::{Ref, RefMut, SystemContext};

use crate::components::{Camera, GlobalTransform};

/// System that updates view matrices for all [`Camera`] components.
///
/// Reads [`GlobalTransform`] to compute the inverse world transform.
/// Must run after [`UpdateGlobalTransforms`](super::UpdateGlobalTransforms).
///
/// # Access
///
/// - Reads: `GlobalTransform`
/// - Writes: `Camera`
pub struct UpdateCameraMatrices;

impl redlilium_ecs::System for UpdateCameraMatrices {
    async fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) {
        ctx.lock::<(
            redlilium_ecs::Read<GlobalTransform>,
            redlilium_ecs::Write<Camera>,
        )>()
        .execute(|(globals, mut cameras)| {
            update_camera_matrices(&globals, &mut cameras);
        })
        .await;
    }
}

fn update_camera_matrices(globals: &Ref<GlobalTransform>, cameras: &mut RefMut<Camera>) {
    redlilium_core::profile_scope!("update_camera_matrices");

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
    use redlilium_ecs::World;

    #[test]
    fn updates_view_matrix() {
        let mut world = World::new();
        world.register_component::<GlobalTransform>();
        world.register_component::<Camera>();

        let e = world.spawn();
        let t = Transform::from_translation(Vec3::new(0.0, 5.0, 10.0));
        world.insert(e, GlobalTransform(t.to_matrix())).unwrap();
        world
            .insert(e, Camera::perspective(1.0, 1.0, 0.1, 100.0))
            .unwrap();

        let globals = world.read::<GlobalTransform>().unwrap();
        let mut cameras = world.write::<Camera>().unwrap();
        update_camera_matrices(&globals, &mut cameras);
        drop(cameras);
        drop(globals);

        let cameras = world.read::<Camera>().unwrap();
        let cam = cameras.get(e.index()).unwrap();

        let expected_view = t.to_matrix().inverse();
        assert!((cam.view_matrix - expected_view).abs_diff_eq(Mat4::ZERO, 1e-5));
        assert_ne!(cam.projection_matrix, Mat4::IDENTITY);
    }

    #[test]
    fn skips_cameras_without_global() {
        let mut world = World::new();
        world.register_component::<Camera>();
        world.register_component::<GlobalTransform>();

        let e = world.spawn();
        world
            .insert(e, Camera::perspective(1.0, 1.0, 0.1, 100.0))
            .unwrap();

        let globals = world.read::<GlobalTransform>().unwrap();
        let mut cameras = world.write::<Camera>().unwrap();
        update_camera_matrices(&globals, &mut cameras);
        drop(cameras);
        drop(globals);

        let cameras = world.read::<Camera>().unwrap();
        let cam = cameras.get(e.index()).unwrap();
        assert_eq!(cam.view_matrix, Mat4::IDENTITY);
    }

    #[test]
    fn projection_preserved() {
        let mut world = World::new();
        world.register_component::<GlobalTransform>();
        world.register_component::<Camera>();

        let e = world.spawn();
        let cam_original = Camera::perspective(1.0, 1.0, 0.1, 100.0);
        let proj_before = cam_original.projection_matrix;
        world
            .insert(
                e,
                GlobalTransform(Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))),
            )
            .unwrap();
        world.insert(e, cam_original).unwrap();

        let globals = world.read::<GlobalTransform>().unwrap();
        let mut cameras = world.write::<Camera>().unwrap();
        update_camera_matrices(&globals, &mut cameras);
        drop(cameras);
        drop(globals);

        let cameras = world.read::<Camera>().unwrap();
        let cam = cameras.get(e.index()).unwrap();
        assert_eq!(cam.projection_matrix, proj_before);
    }
}
