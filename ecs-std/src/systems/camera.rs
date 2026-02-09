use glam::Mat4;
use redlilium_core::scene::CameraProjection;
use redlilium_ecs::World;

use crate::components::{Camera, GlobalTransform};

/// Updates view and projection matrices for all [`Camera`] components.
///
/// Reads [`GlobalTransform`] to compute the view matrix (inverse of the
/// camera's world transform). Computes the projection matrix from
/// the camera's [`CameraProjection`] configuration.
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
            camera.projection_matrix = compute_projection(&camera.projection);
        }
    }
}

/// Compute projection matrix from CameraProjection.
///
/// Uses right-handed projection with [0, 1] depth range,
/// matching the engine's coordinate system convention.
fn compute_projection(projection: &CameraProjection) -> Mat4 {
    match *projection {
        CameraProjection::Perspective {
            yfov,
            aspect,
            znear,
            zfar,
        } => {
            let aspect = aspect.unwrap_or(16.0 / 9.0);
            match zfar {
                Some(far) => Mat4::perspective_rh(yfov, aspect, znear, far),
                None => Mat4::perspective_infinite_rh(yfov, aspect, znear),
            }
        }
        CameraProjection::Orthographic {
            xmag,
            ymag,
            znear,
            zfar,
        } => Mat4::orthographic_rh(-xmag, xmag, -ymag, ymag, znear, zfar),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Transform;
    use glam::Vec3;

    #[test]
    fn updates_camera_matrices() {
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

        // Projection should not be identity anymore
        assert_ne!(cam.projection_matrix, Mat4::IDENTITY);
    }

    #[test]
    fn skips_cameras_without_global() {
        let mut world = World::new();

        let e = world.spawn();
        world.insert(e, Camera::perspective(1.0, 1.0, 0.1, 100.0));
        world.register_component::<GlobalTransform>();

        update_camera_matrices(&world);

        // Camera matrices should remain identity (not updated)
        let cameras = world.read::<Camera>();
        let cam = cameras.get(e.index()).unwrap();
        assert_eq!(cam.view_matrix, Mat4::IDENTITY);
    }

    #[test]
    fn orthographic_projection() {
        let proj = compute_projection(&CameraProjection::Orthographic {
            xmag: 10.0,
            ymag: 10.0,
            znear: 0.1,
            zfar: 100.0,
        });
        assert_ne!(proj, Mat4::IDENTITY);
    }

    #[test]
    fn infinite_perspective() {
        let proj = compute_projection(&CameraProjection::Perspective {
            yfov: 1.0,
            aspect: Some(1.0),
            znear: 0.1,
            zfar: None,
        });
        assert_ne!(proj, Mat4::IDENTITY);
    }
}
