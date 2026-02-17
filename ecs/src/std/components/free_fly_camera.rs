use redlilium_core::math::Vec3;

/// Free-fly camera component combining fly and orbit modes (Unity-editor style).
///
/// **Orbit mode** (default): mouse drag orbits the camera around the
/// fixed target point.
///
/// **Fly mode** (hold Ctrl): mouse drag rotates the view direction.
/// WASD moves forward/back/left/right. Q/E moves down/up.
/// The target point moves with the camera.
///
/// Scroll wheel zooms (adjusts distance to target) in both modes.
///
/// Entities with this component should also have `Camera`, `Transform`,
/// and `GlobalTransform` (add them manually since `Camera` has no `Default`).
#[derive(Debug, Clone, Copy, PartialEq, crate::Component)]
pub struct FreeFlyCamera {
    /// The point the camera looks at / orbits around.
    pub target: Vec3,
    /// Distance from the camera to the target point.
    pub distance: f32,
    /// Horizontal angle in radians (rotation around the Y axis).
    pub yaw: f32,
    /// Vertical angle in radians (above/below the horizontal plane).
    pub pitch: f32,
    /// Mouse rotation sensitivity (radians per pixel).
    pub rotate_sensitivity: f32,
    /// Movement speed (units per frame) for WASD/QE keys.
    pub move_speed: f32,
    /// Scroll zoom sensitivity (distance units per scroll unit).
    pub zoom_sensitivity: f32,
    /// Minimum allowed distance (zoom-in limit).
    pub min_distance: f32,
    /// Maximum allowed distance (zoom-out limit).
    pub max_distance: f32,
    /// Minimum pitch in radians (looking-down limit, typically negative).
    pub min_pitch: f32,
    /// Maximum pitch in radians (looking-up limit, typically positive).
    pub max_pitch: f32,
    /// Current speed multiplier (managed by the system, ramps up while Shift is held).
    pub speed_multiplier: f32,
    /// Per-frame multiplicative growth of `speed_multiplier` while Shift is held (e.g. 1.02).
    pub speed_boost_acceleration: f32,
    /// Maximum value `speed_multiplier` can reach.
    pub max_speed_multiplier: f32,
}

impl FreeFlyCamera {
    /// Create a free-fly camera looking at `target` from the given `distance`.
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            yaw: 0.0,
            pitch: 0.3,
            rotate_sensitivity: 0.005,
            move_speed: 0.15,
            zoom_sensitivity: 0.5,
            min_distance: 0.5,
            max_distance: 200.0,
            min_pitch: -1.5,
            max_pitch: 1.5,
            speed_multiplier: 1.0,
            speed_boost_acceleration: 1.02,
            max_speed_multiplier: 10.0,
        }
    }

    /// Set the initial yaw (radians).
    pub fn with_yaw(mut self, yaw: f32) -> Self {
        self.yaw = yaw;
        self
    }

    /// Set the initial pitch (radians).
    pub fn with_pitch(mut self, pitch: f32) -> Self {
        self.pitch = pitch;
        self
    }

    /// Set the rotation sensitivity (radians per pixel).
    pub fn with_rotate_sensitivity(mut self, sensitivity: f32) -> Self {
        self.rotate_sensitivity = sensitivity;
        self
    }

    /// Set the movement speed (units per frame).
    pub fn with_move_speed(mut self, speed: f32) -> Self {
        self.move_speed = speed;
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

    /// Set the allowed pitch range (radians).
    pub fn with_pitch_range(mut self, min: f32, max: f32) -> Self {
        self.min_pitch = min;
        self.max_pitch = max;
        self
    }

    /// Orbit rotation: rotate around the target (target stays fixed, camera moves).
    pub fn orbit_rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(self.min_pitch, self.max_pitch);
    }

    /// Free-look rotation: rotate the camera direction (eye stays, target moves).
    pub fn free_rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        let old_eye = self.eye_position();
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(self.min_pitch, self.max_pitch);
        // Recompute target so the eye remains at the same position
        self.target = old_eye - self.eye_offset();
    }

    /// Move the camera and target together in the camera's local frame.
    ///
    /// `forward`: movement toward the target (the actual look direction including pitch).
    /// `right`: movement to the camera's right (horizontal).
    /// `up`: movement along the world Y axis.
    pub fn fly_move(&mut self, forward: f32, right: f32, up: f32) {
        // Forward includes pitch — moves toward where the camera is looking
        let fwd = Vec3::new(
            -self.yaw.sin() * self.pitch.cos(),
            -self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        );
        // Right stays horizontal
        let rgt = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
        let speed = self.move_speed * self.speed_multiplier;
        let displacement = fwd * forward * speed + rgt * right * speed + Vec3::y() * up * speed;
        self.target += displacement;
    }

    /// Apply zoom delta. Positive values move closer to the target.
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta).clamp(self.min_distance, self.max_distance);
    }

    /// Offset vector from target to eye in world space.
    fn eye_offset(&self) -> Vec3 {
        Vec3::new(
            self.distance * self.pitch.cos() * self.yaw.sin(),
            self.distance * self.pitch.sin(),
            self.distance * self.pitch.cos() * self.yaw.cos(),
        )
    }

    /// Compute the camera eye position from the target and spherical parameters.
    pub fn eye_position(&self) -> Vec3 {
        self.target + self.eye_offset()
    }

    /// Compute the [`Transform`](crate::Transform) for this camera (position + look-at rotation).
    pub fn to_transform(&self) -> crate::Transform {
        let eye = self.eye_position();
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

impl Default for FreeFlyCamera {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 0.0, 0.0), 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_defaults() {
        let cam = FreeFlyCamera::new(Vec3::new(0.0, 3.0, 0.0), 20.0);
        assert_eq!(cam.distance, 20.0);
        assert_eq!(cam.target, Vec3::new(0.0, 3.0, 0.0));
        assert!((cam.rotate_sensitivity - 0.005).abs() < f32::EPSILON);
        assert!((cam.move_speed - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn orbit_rotate_clamps_pitch() {
        let mut cam = FreeFlyCamera::default();
        cam.orbit_rotate(0.0, 10.0);
        assert!(cam.pitch <= cam.max_pitch);

        cam.orbit_rotate(0.0, -20.0);
        assert!(cam.pitch >= cam.min_pitch);
    }

    #[test]
    fn free_rotate_preserves_eye() {
        let cam_before = FreeFlyCamera::new(Vec3::new(1.0, 2.0, 3.0), 15.0)
            .with_yaw(0.5)
            .with_pitch(0.3);
        let eye_before = cam_before.eye_position();

        let mut cam = cam_before;
        cam.free_rotate(0.2, -0.1);
        let eye_after = cam.eye_position();

        assert!((eye_before - eye_after).norm() < 1e-4);
    }

    #[test]
    fn zoom_clamps_distance() {
        let mut cam = FreeFlyCamera::default();
        cam.zoom(1000.0);
        assert!(cam.distance >= cam.min_distance);

        cam.zoom(-10000.0);
        assert!(cam.distance <= cam.max_distance);
    }

    #[test]
    fn fly_move_shifts_target() {
        let mut cam = FreeFlyCamera::new(Vec3::zeros(), 10.0)
            .with_yaw(0.0)
            .with_pitch(0.0);
        let old_target = cam.target;
        cam.fly_move(1.0, 0.0, 0.0);
        // Forward at yaw=0, pitch=0 is -Z direction
        assert!((cam.target.z - (old_target.z - cam.move_speed)).abs() < 1e-5);
        assert!(cam.target.y.abs() < 1e-5);
    }

    #[test]
    fn eye_position_at_zero_angles() {
        let cam = FreeFlyCamera::new(Vec3::zeros(), 10.0)
            .with_yaw(0.0)
            .with_pitch(0.0);
        let pos = cam.eye_position();
        assert!((pos.x).abs() < 1e-6);
        assert!((pos.y).abs() < 1e-6);
        assert!((pos.z - 10.0).abs() < 1e-6);
    }

    #[test]
    fn to_transform_position_matches() {
        let cam = FreeFlyCamera::new(Vec3::new(0.0, 3.0, 0.0), 15.0)
            .with_yaw(0.5)
            .with_pitch(0.4);
        let transform = cam.to_transform();
        let expected_pos = cam.eye_position();
        assert!((transform.translation - expected_pos).norm() < 1e-5);
    }

    #[test]
    fn builder_pattern() {
        let cam = FreeFlyCamera::new(Vec3::zeros(), 5.0)
            .with_yaw(1.0)
            .with_pitch(0.5)
            .with_rotate_sensitivity(0.01)
            .with_move_speed(0.5)
            .with_zoom_sensitivity(1.0)
            .with_distance_range(1.0, 100.0)
            .with_pitch_range(-1.0, 1.0);

        assert_eq!(cam.yaw, 1.0);
        assert_eq!(cam.pitch, 0.5);
        assert_eq!(cam.rotate_sensitivity, 0.01);
        assert_eq!(cam.move_speed, 0.5);
        assert_eq!(cam.zoom_sensitivity, 1.0);
        assert_eq!(cam.min_distance, 1.0);
        assert_eq!(cam.max_distance, 100.0);
        assert_eq!(cam.min_pitch, -1.0);
        assert_eq!(cam.max_pitch, 1.0);
    }
}
