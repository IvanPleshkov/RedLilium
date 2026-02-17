use redlilium_core::math::Vec3;

/// Orbit camera component with configurable orbital parameters.
///
/// Stores a target point, distance, azimuth/elevation angles, and
/// sensitivity/clamp settings. The [`UpdateOrbitCamera`](crate::std::systems::UpdateOrbitCamera)
/// system reads these parameters plus the [`WindowInput`](super::WindowInput)
/// resource to update the entity's [`Transform`](super::Transform).
///
/// Entities with this component should also have `Camera`, `Transform`,
/// and `GlobalTransform` (add them manually since `Camera` has no `Default`).
#[derive(Debug, Clone, Copy, PartialEq, crate::Component)]
pub struct OrbitCamera {
    /// The point the camera orbits around.
    pub target: Vec3,
    /// Distance from the target.
    pub distance: f32,
    /// Horizontal angle in radians (around the Y axis).
    pub azimuth: f32,
    /// Vertical angle in radians (above/below the horizontal plane).
    pub elevation: f32,
    /// Mouse rotation sensitivity (radians per pixel).
    pub rotate_sensitivity: f32,
    /// Scroll zoom sensitivity (distance units per scroll unit).
    pub zoom_sensitivity: f32,
    /// Minimum allowed distance.
    pub min_distance: f32,
    /// Maximum allowed distance.
    pub max_distance: f32,
    /// Minimum elevation in radians (typically negative).
    pub min_elevation: f32,
    /// Maximum elevation in radians (typically positive).
    pub max_elevation: f32,
}

impl OrbitCamera {
    /// Create an orbit camera targeting `target` at the given `distance`.
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            azimuth: 0.0,
            elevation: 0.3,
            rotate_sensitivity: 0.005,
            zoom_sensitivity: 0.5,
            min_distance: 2.0,
            max_distance: 60.0,
            min_elevation: -1.5,
            max_elevation: 1.5,
        }
    }

    /// Set the initial azimuth (radians).
    pub fn with_azimuth(mut self, azimuth: f32) -> Self {
        self.azimuth = azimuth;
        self
    }

    /// Set the initial elevation (radians).
    pub fn with_elevation(mut self, elevation: f32) -> Self {
        self.elevation = elevation;
        self
    }

    /// Set the rotation sensitivity (radians per pixel).
    pub fn with_rotate_sensitivity(mut self, sensitivity: f32) -> Self {
        self.rotate_sensitivity = sensitivity;
        self
    }

    /// Set the zoom sensitivity (distance per scroll unit).
    pub fn with_zoom_sensitivity(mut self, sensitivity: f32) -> Self {
        self.zoom_sensitivity = sensitivity;
        self
    }

    /// Set the allowed distance range.
    pub fn with_distance_range(mut self, min: f32, max: f32) -> Self {
        self.min_distance = min;
        self.max_distance = max;
        self
    }

    /// Set the allowed elevation range (radians).
    pub fn with_elevation_range(mut self, min: f32, max: f32) -> Self {
        self.min_elevation = min;
        self.max_elevation = max;
        self
    }

    /// Apply rotation deltas (in radians).
    pub fn rotate(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        self.azimuth += delta_azimuth;
        self.elevation =
            (self.elevation + delta_elevation).clamp(self.min_elevation, self.max_elevation);
    }

    /// Apply zoom delta. Positive values move closer.
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta).clamp(self.min_distance, self.max_distance);
    }

    /// Compute the camera eye position from spherical coordinates.
    pub fn position(&self) -> Vec3 {
        let x = self.distance * self.elevation.cos() * self.azimuth.sin();
        let y = self.distance * self.elevation.sin();
        let z = self.distance * self.elevation.cos() * self.azimuth.cos();
        self.target + Vec3::new(x, y, z)
    }

    /// Compute the [`Transform`](crate::Transform) for this camera (position + look-at rotation).
    pub fn to_transform(&self) -> crate::Transform {
        let eye = self.position();
        let eye_point = redlilium_core::math::nalgebra::Point3::from(eye);
        let target_point = redlilium_core::math::nalgebra::Point3::from(self.target);
        let up = Vec3::new(0.0, 1.0, 0.0);

        let view_iso =
            redlilium_core::math::nalgebra::Isometry3::look_at_rh(&eye_point, &target_point, &up);
        let camera_iso = view_iso.inverse();
        let rotation = camera_iso.rotation.into_inner();

        crate::Transform::new(eye, rotation, Vec3::new(1.0, 1.0, 1.0))
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 0.0, 0.0), 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_defaults() {
        let cam = OrbitCamera::new(Vec3::new(0.0, 3.0, 0.0), 20.0);
        assert_eq!(cam.distance, 20.0);
        assert_eq!(cam.target, Vec3::new(0.0, 3.0, 0.0));
        assert!((cam.rotate_sensitivity - 0.005).abs() < f32::EPSILON);
    }

    #[test]
    fn rotate_clamps_elevation() {
        let mut cam = OrbitCamera::default();
        cam.rotate(0.0, 10.0); // way above max
        assert!(cam.elevation <= cam.max_elevation);

        cam.rotate(0.0, -20.0); // way below min
        assert!(cam.elevation >= cam.min_elevation);
    }

    #[test]
    fn zoom_clamps_distance() {
        let mut cam = OrbitCamera::default();
        cam.zoom(1000.0); // zoom in very far
        assert!(cam.distance >= cam.min_distance);

        cam.zoom(-10000.0); // zoom out very far
        assert!(cam.distance <= cam.max_distance);
    }

    #[test]
    fn position_at_zero_angles() {
        let cam = OrbitCamera::new(Vec3::zeros(), 10.0)
            .with_azimuth(0.0)
            .with_elevation(0.0);
        let pos = cam.position();
        // At azimuth=0, elevation=0: eye is along +Z
        assert!((pos.x).abs() < 1e-6);
        assert!((pos.y).abs() < 1e-6);
        assert!((pos.z - 10.0).abs() < 1e-6);
    }

    #[test]
    fn to_transform_position_matches() {
        let cam = OrbitCamera::new(Vec3::new(0.0, 3.0, 0.0), 15.0)
            .with_azimuth(0.5)
            .with_elevation(0.4);
        let transform = cam.to_transform();
        let expected_pos = cam.position();
        assert!((transform.translation - expected_pos).norm() < 1e-5);
    }

    #[test]
    fn builder_pattern() {
        let cam = OrbitCamera::new(Vec3::zeros(), 5.0)
            .with_azimuth(1.0)
            .with_elevation(0.5)
            .with_rotate_sensitivity(0.01)
            .with_zoom_sensitivity(1.0)
            .with_distance_range(1.0, 100.0)
            .with_elevation_range(-1.0, 1.0);

        assert_eq!(cam.azimuth, 1.0);
        assert_eq!(cam.elevation, 0.5);
        assert_eq!(cam.rotate_sensitivity, 0.01);
        assert_eq!(cam.zoom_sensitivity, 1.0);
        assert_eq!(cam.min_distance, 1.0);
        assert_eq!(cam.max_distance, 100.0);
        assert_eq!(cam.min_elevation, -1.0);
        assert_eq!(cam.max_elevation, 1.0);
    }
}
