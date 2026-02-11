use redlilium_core::math::{Mat4, orthographic_rh, perspective_rh};

/// Camera component storing computed view and projection matrices.
///
/// Projection is computed eagerly in constructors. The
/// [`update_camera_matrices`](crate::systems::update_camera_matrices) system
/// updates only the view matrix from the entity's world transform.
#[derive(
    Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, redlilium_ecs::Component,
)]
#[repr(C)]
pub struct Camera {
    /// Computed view matrix (world-to-camera). Updated by system.
    pub view_matrix: Mat4,
    /// Computed projection matrix. Set at construction.
    pub projection_matrix: Mat4,
}

impl Camera {
    /// Create a new perspective camera.
    pub fn perspective(yfov: f32, aspect: f32, znear: f32, zfar: f32) -> Self {
        Self {
            view_matrix: Mat4::identity(),
            projection_matrix: perspective_rh(yfov, aspect, znear, zfar),
        }
    }

    /// Create a new orthographic camera.
    pub fn orthographic(xmag: f32, ymag: f32, znear: f32, zfar: f32) -> Self {
        Self {
            view_matrix: Mat4::identity(),
            projection_matrix: orthographic_rh(-xmag, xmag, -ymag, ymag, znear, zfar),
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
        assert_ne!(cam.projection_matrix, Mat4::identity());
        assert_eq!(cam.view_matrix, Mat4::identity());
    }

    #[test]
    fn orthographic_constructor() {
        let cam = Camera::orthographic(10.0, 10.0, 0.1, 100.0);
        assert_ne!(cam.projection_matrix, Mat4::identity());
        assert_eq!(cam.view_matrix, Mat4::identity());
    }

    #[test]
    fn view_projection_identity_view() {
        let cam = Camera::perspective(1.0, 1.0, 0.1, 100.0);
        // With identity view, view_projection == projection
        assert_eq!(cam.view_projection(), cam.projection_matrix);
    }
}
