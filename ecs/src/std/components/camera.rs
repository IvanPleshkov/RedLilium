use redlilium_core::math::{DepthConvention, Mat4};

/// Camera component storing computed view and projection matrices.
///
/// Projection is computed eagerly in constructors. The
/// [`update_camera_matrices`](crate::systems::update_camera_matrices) system
/// updates only the view matrix from the entity's world transform.
///
/// Constructors default to the engine's **reversed-Z** depth convention
/// (ADR-038): the convention is baked into the projection matrix at
/// construction — the component stores only matrices, no convention field.
/// The std render path assumes reversed-Z targets (clear 0.0, `GreaterEqual`);
/// a camera built with [`DepthConvention::Classic`] must render through a
/// pipeline whose clear and compare match, or depth testing silently breaks.
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable, crate::Component)]
#[require(crate::Transform, crate::GlobalTransform, crate::Visibility)]
#[repr(C)]
pub struct Camera {
    /// Computed view matrix (world-to-camera). Updated by system.
    pub view_matrix: Mat4,
    /// Computed projection matrix. Set at construction.
    pub projection_matrix: Mat4,
}

impl Camera {
    /// Create a new perspective camera (reversed-Z, the engine default).
    pub fn perspective(yfov: f32, aspect: f32, znear: f32, zfar: f32) -> Self {
        Self::perspective_with(DepthConvention::default(), yfov, aspect, znear, zfar)
    }

    /// Create a new perspective camera with the classic depth convention
    /// (near → 0, far → 1). Opt-out escape hatch — see the type-level docs.
    pub fn perspective_classic(yfov: f32, aspect: f32, znear: f32, zfar: f32) -> Self {
        Self::perspective_with(DepthConvention::Classic, yfov, aspect, znear, zfar)
    }

    /// Create a new perspective camera with an explicit depth convention.
    pub fn perspective_with(
        convention: DepthConvention,
        yfov: f32,
        aspect: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            view_matrix: Mat4::identity(),
            projection_matrix: convention.perspective(yfov, aspect, znear, zfar),
        }
    }

    /// Create a new orthographic camera (reversed-Z, the engine default).
    pub fn orthographic(xmag: f32, ymag: f32, znear: f32, zfar: f32) -> Self {
        Self::orthographic_with(DepthConvention::default(), xmag, ymag, znear, zfar)
    }

    /// Create a new orthographic camera with the classic depth convention
    /// (near → 0, far → 1). Opt-out escape hatch — see the type-level docs.
    pub fn orthographic_classic(xmag: f32, ymag: f32, znear: f32, zfar: f32) -> Self {
        Self::orthographic_with(DepthConvention::Classic, xmag, ymag, znear, zfar)
    }

    /// Create a new orthographic camera with an explicit depth convention.
    pub fn orthographic_with(
        convention: DepthConvention,
        xmag: f32,
        ymag: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            view_matrix: Mat4::identity(),
            projection_matrix: convention.orthographic(-xmag, xmag, -ymag, ymag, znear, zfar),
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
    use redlilium_core::math::{perspective_rh, perspective_rh_reversed};

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
    fn perspective_defaults_to_reversed_z() {
        let cam = Camera::perspective(1.0, 1.0, 0.1, 100.0);
        assert_eq!(
            cam.projection_matrix,
            perspective_rh_reversed(1.0, 1.0, 0.1, 100.0)
        );
    }

    #[test]
    fn perspective_classic_opt_out() {
        let cam = Camera::perspective_classic(1.0, 1.0, 0.1, 100.0);
        assert_eq!(cam.projection_matrix, perspective_rh(1.0, 1.0, 0.1, 100.0));
        assert_eq!(
            Camera::perspective_with(DepthConvention::Classic, 1.0, 1.0, 0.1, 100.0)
                .projection_matrix,
            cam.projection_matrix
        );
    }

    #[test]
    fn view_projection_identity_view() {
        let cam = Camera::perspective(1.0, 1.0, 0.1, 100.0);
        // With identity view, view_projection == projection
        assert_eq!(cam.view_projection(), cam.projection_matrix);
    }
}
