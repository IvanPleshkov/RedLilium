use glam::Mat4;
use redlilium_core::scene::CameraProjection;

/// Camera component storing projection configuration and computed matrices.
///
/// The projection configuration reuses core's [`CameraProjection`].
/// The view and projection matrices are computed by the
/// [`update_camera_matrices`](crate::systems::update_camera_matrices) system.
#[derive(Debug, Clone, redlilium_ecs::Component)]
pub struct Camera {
    /// Projection type and parameters (perspective or orthographic).
    pub projection: CameraProjection,
    /// Whether this camera is the active/main camera.
    pub active: bool,
    /// Computed view matrix (world-to-camera). Updated by system.
    pub view_matrix: Mat4,
    /// Computed projection matrix. Updated by system.
    pub projection_matrix: Mat4,
}

impl Camera {
    /// Create a new perspective camera.
    pub fn perspective(yfov: f32, aspect: f32, znear: f32, zfar: f32) -> Self {
        Self {
            projection: CameraProjection::Perspective {
                yfov,
                aspect: Some(aspect),
                znear,
                zfar: Some(zfar),
            },
            active: true,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
        }
    }

    /// Create a new orthographic camera.
    pub fn orthographic(xmag: f32, ymag: f32, znear: f32, zfar: f32) -> Self {
        Self {
            projection: CameraProjection::Orthographic {
                xmag,
                ymag,
                znear,
                zfar,
            },
            active: true,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
        }
    }

    /// Create from a core CameraProjection.
    pub fn from_projection(projection: CameraProjection) -> Self {
        Self {
            projection,
            active: true,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
        }
    }

    /// Compute the view-projection matrix (projection * view).
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix * self.view_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perspective_constructor() {
        let cam = Camera::perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
        assert!(cam.active);
        assert!(matches!(
            cam.projection,
            CameraProjection::Perspective { .. }
        ));
    }

    #[test]
    fn orthographic_constructor() {
        let cam = Camera::orthographic(10.0, 10.0, 0.1, 100.0);
        assert!(cam.active);
        assert!(matches!(
            cam.projection,
            CameraProjection::Orthographic { .. }
        ));
    }

    #[test]
    fn view_projection_identity() {
        let cam = Camera::perspective(1.0, 1.0, 0.1, 100.0);
        // Both matrices are identity initially
        assert_eq!(cam.view_projection(), Mat4::IDENTITY);
    }
}
