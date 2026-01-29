//! Camera controller system
//!
//! Provides abstract camera control with implementations for:
//! - FreeFly: WASD movement, mouse look, scroll speed
//! - Orbit: Rotate around a target point

use glam::{Mat4, Vec2, Vec3};

use super::Camera;

/// Input state for camera controllers
#[derive(Debug, Clone, Default)]
pub struct CameraInput {
    /// Movement keys (WASD, QE for up/down)
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,

    /// Sprint modifier (shift)
    pub sprint: bool,

    /// Mouse delta since last frame (in pixels)
    pub mouse_delta: Vec2,

    /// Mouse scroll delta (positive = scroll up)
    pub scroll_delta: f32,

    /// Whether mouse look is active (e.g., right mouse button held)
    pub mouse_look_active: bool,
}

impl CameraInput {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset per-frame deltas (call after update)
    pub fn reset_deltas(&mut self) {
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = 0.0;
    }
}

/// Abstract camera controller trait
pub trait CameraController {
    /// Update the camera based on input and delta time
    fn update(&mut self, camera: &mut Camera, input: &CameraInput, dt: f32);

    /// Get the controller name for debugging
    fn name(&self) -> &'static str;

    /// Reset the controller to default state
    fn reset(&mut self);
}

/// Free-fly camera controller (FPS-style)
///
/// - WASD: Move forward/backward/left/right
/// - QE or Space/Ctrl: Move up/down
/// - Mouse: Look around (when mouse_look_active)
/// - Scroll: Adjust movement speed
/// - Shift: Sprint (2x speed)
pub struct FreeFlyController {
    /// Current yaw angle (horizontal rotation) in radians
    pub yaw: f32,
    /// Current pitch angle (vertical rotation) in radians
    pub pitch: f32,
    /// Base movement speed in units per second
    pub move_speed: f32,
    /// Minimum movement speed
    pub min_speed: f32,
    /// Maximum movement speed
    pub max_speed: f32,
    /// Mouse sensitivity (radians per pixel)
    pub mouse_sensitivity: f32,
    /// Speed multiplier when sprinting
    pub sprint_multiplier: f32,
    /// Speed change per scroll unit
    pub scroll_speed_factor: f32,
}

impl Default for FreeFlyController {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            move_speed: 5.0,
            min_speed: 0.5,
            max_speed: 50.0,
            mouse_sensitivity: 0.003,
            sprint_multiplier: 2.0,
            scroll_speed_factor: 1.2,
        }
    }
}

impl FreeFlyController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom speed settings
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.move_speed = speed;
        self
    }

    /// Create with custom sensitivity
    pub fn with_sensitivity(mut self, sensitivity: f32) -> Self {
        self.mouse_sensitivity = sensitivity;
        self
    }

    /// Initialize yaw/pitch from camera's current orientation
    pub fn sync_with_camera(&mut self, camera: &Camera) {
        let forward = (camera.target - camera.position).normalize();
        self.yaw = forward.z.atan2(forward.x);
        self.pitch = (-forward.y).asin();
    }

    /// Get the forward direction based on yaw/pitch
    fn forward_direction(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            -self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    /// Get the right direction (perpendicular to forward, on XZ plane)
    fn right_direction(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }
}

impl CameraController for FreeFlyController {
    fn update(&mut self, camera: &mut Camera, input: &CameraInput, dt: f32) {
        // Handle scroll wheel - adjust speed
        if input.scroll_delta != 0.0 {
            if input.scroll_delta > 0.0 {
                self.move_speed *= self.scroll_speed_factor;
            } else {
                self.move_speed /= self.scroll_speed_factor;
            }
            self.move_speed = self.move_speed.clamp(self.min_speed, self.max_speed);
        }

        // Handle mouse look
        if input.mouse_look_active && input.mouse_delta != Vec2::ZERO {
            self.yaw += input.mouse_delta.x * self.mouse_sensitivity;
            self.pitch += input.mouse_delta.y * self.mouse_sensitivity;

            // Clamp pitch to avoid gimbal lock
            let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
            self.pitch = self.pitch.clamp(-max_pitch, max_pitch);

            // Keep yaw in reasonable range
            self.yaw = self.yaw % (2.0 * std::f32::consts::PI);
        }

        // Calculate movement direction
        let forward = self.forward_direction();
        let right = self.right_direction();
        let up = Vec3::Y;

        let mut velocity = Vec3::ZERO;

        if input.forward {
            velocity += forward;
        }
        if input.backward {
            velocity -= forward;
        }
        if input.right {
            velocity += right;
        }
        if input.left {
            velocity -= right;
        }
        if input.up {
            velocity += up;
        }
        if input.down {
            velocity -= up;
        }

        // Normalize if moving diagonally
        if velocity.length_squared() > 0.0 {
            velocity = velocity.normalize();
        }

        // Apply speed
        let speed = if input.sprint {
            self.move_speed * self.sprint_multiplier
        } else {
            self.move_speed
        };

        camera.position += velocity * speed * dt;

        // Update camera target based on new position and orientation
        camera.target = camera.position + forward;
    }

    fn name(&self) -> &'static str {
        "FreeFly"
    }

    fn reset(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.move_speed = 5.0;
    }
}

/// Orbit camera controller
///
/// Rotates around a target point at a fixed distance.
/// - Mouse drag: Orbit around target
/// - Scroll: Zoom in/out (change distance)
/// - WASD: Pan the target point
pub struct OrbitController {
    /// Target point to orbit around
    pub target: Vec3,
    /// Distance from target
    pub distance: f32,
    /// Minimum distance
    pub min_distance: f32,
    /// Maximum distance
    pub max_distance: f32,
    /// Current azimuth angle (horizontal) in radians
    pub azimuth: f32,
    /// Current elevation angle (vertical) in radians
    pub elevation: f32,
    /// Minimum elevation (prevent going below ground)
    pub min_elevation: f32,
    /// Maximum elevation (prevent going over top)
    pub max_elevation: f32,
    /// Orbit sensitivity (radians per pixel)
    pub orbit_sensitivity: f32,
    /// Zoom factor per scroll unit
    pub zoom_factor: f32,
    /// Pan speed for moving target
    pub pan_speed: f32,
}

impl Default for OrbitController {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 10.0,
            min_distance: 1.0,
            max_distance: 100.0,
            azimuth: 0.0,
            elevation: std::f32::consts::FRAC_PI_6, // 30 degrees
            min_elevation: 0.05,
            max_elevation: std::f32::consts::FRAC_PI_2 - 0.05,
            orbit_sensitivity: 0.005,
            zoom_factor: 1.1,
            pan_speed: 5.0,
        }
    }
}

impl OrbitController {
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            ..Default::default()
        }
    }

    /// Create with specific angles
    pub fn with_angles(mut self, azimuth_degrees: f32, elevation_degrees: f32) -> Self {
        self.azimuth = azimuth_degrees.to_radians();
        self.elevation = elevation_degrees.to_radians();
        self
    }

    /// Initialize from camera's current position and target
    pub fn sync_with_camera(&mut self, camera: &Camera) {
        self.target = camera.target;
        let offset = camera.position - camera.target;
        self.distance = offset.length();

        // Calculate angles from offset
        let horizontal_dist = (offset.x * offset.x + offset.z * offset.z).sqrt();
        self.elevation = (offset.y / self.distance).asin();
        self.azimuth = offset.z.atan2(offset.x);
    }

    /// Calculate camera position from orbit parameters
    fn calculate_position(&self) -> Vec3 {
        let x = self.distance * self.elevation.cos() * self.azimuth.cos();
        let y = self.distance * self.elevation.sin();
        let z = self.distance * self.elevation.cos() * self.azimuth.sin();
        self.target + Vec3::new(x, y, z)
    }

    /// Get right direction for panning (on XZ plane based on azimuth)
    fn right_direction(&self) -> Vec3 {
        Vec3::new(-self.azimuth.sin(), 0.0, self.azimuth.cos()).normalize()
    }

    /// Get forward direction for panning (on XZ plane, toward target)
    fn forward_direction(&self) -> Vec3 {
        Vec3::new(self.azimuth.cos(), 0.0, self.azimuth.sin()).normalize()
    }
}

impl CameraController for OrbitController {
    fn update(&mut self, camera: &mut Camera, input: &CameraInput, dt: f32) {
        // Handle scroll wheel - zoom
        if input.scroll_delta != 0.0 {
            if input.scroll_delta > 0.0 {
                self.distance /= self.zoom_factor;
            } else {
                self.distance *= self.zoom_factor;
            }
            self.distance = self.distance.clamp(self.min_distance, self.max_distance);
        }

        // Handle mouse orbit
        if input.mouse_look_active && input.mouse_delta != Vec2::ZERO {
            self.azimuth += input.mouse_delta.x * self.orbit_sensitivity;
            self.elevation += input.mouse_delta.y * self.orbit_sensitivity;

            // Clamp elevation
            self.elevation = self.elevation.clamp(self.min_elevation, self.max_elevation);

            // Keep azimuth in reasonable range
            self.azimuth = self.azimuth % (2.0 * std::f32::consts::PI);
        }

        // Handle panning with WASD
        let forward = self.forward_direction();
        let right = self.right_direction();

        let mut pan = Vec3::ZERO;

        if input.forward {
            pan += forward;
        }
        if input.backward {
            pan -= forward;
        }
        if input.right {
            pan += right;
        }
        if input.left {
            pan -= right;
        }
        if input.up {
            pan += Vec3::Y;
        }
        if input.down {
            pan -= Vec3::Y;
        }

        if pan.length_squared() > 0.0 {
            pan = pan.normalize();
            let speed = if input.sprint {
                self.pan_speed * 2.0
            } else {
                self.pan_speed
            };
            self.target += pan * speed * dt;
        }

        // Update camera position and target
        camera.position = self.calculate_position();
        camera.target = self.target;
    }

    fn name(&self) -> &'static str {
        "Orbit"
    }

    fn reset(&mut self) {
        self.target = Vec3::ZERO;
        self.distance = 10.0;
        self.azimuth = 0.0;
        self.elevation = std::f32::consts::FRAC_PI_6;
    }
}

/// Camera controller that can switch between different control modes
pub struct MultiModeController {
    controllers: Vec<Box<dyn CameraController>>,
    active_index: usize,
}

impl MultiModeController {
    pub fn new() -> Self {
        Self {
            controllers: Vec::new(),
            active_index: 0,
        }
    }

    pub fn add_controller<C: CameraController + 'static>(mut self, controller: C) -> Self {
        self.controllers.push(Box::new(controller));
        self
    }

    pub fn switch_to(&mut self, index: usize) {
        if index < self.controllers.len() {
            self.active_index = index;
        }
    }

    pub fn switch_next(&mut self) {
        if !self.controllers.is_empty() {
            self.active_index = (self.active_index + 1) % self.controllers.len();
        }
    }

    pub fn active_name(&self) -> &'static str {
        self.controllers
            .get(self.active_index)
            .map(|c| c.name())
            .unwrap_or("None")
    }

    pub fn active_index(&self) -> usize {
        self.active_index
    }
}

impl Default for MultiModeController {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraController for MultiModeController {
    fn update(&mut self, camera: &mut Camera, input: &CameraInput, dt: f32) {
        if let Some(controller) = self.controllers.get_mut(self.active_index) {
            controller.update(camera, input, dt);
        }
    }

    fn name(&self) -> &'static str {
        "MultiMode"
    }

    fn reset(&mut self) {
        if let Some(controller) = self.controllers.get_mut(self.active_index) {
            controller.reset();
        }
    }
}
