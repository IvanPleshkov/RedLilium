//! Orbit camera input controller for the PBR demo.
//!
//! Handles rotate/zoom input and computes the camera position in world space.
//! View and projection matrices are computed by the ECS camera system.

use std::f32::consts::PI;

use redlilium_core::math::Vec3;

/// Orbit camera controller that tracks azimuth, elevation, and distance
/// around a target point. The resulting position is fed into the ECS
/// camera entity's Transform each frame.
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub azimuth: f32,
    pub elevation: f32,
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self {
            target: Vec3::zeros(),
            distance: 8.0,
            azimuth: 0.5,
            elevation: 0.4,
        }
    }

    pub fn rotate(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        self.azimuth += delta_azimuth;
        self.elevation = (self.elevation + delta_elevation).clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta).clamp(2.0, 20.0);
    }

    pub fn position(&self) -> Vec3 {
        let x = self.distance * self.elevation.cos() * self.azimuth.sin();
        let y = self.distance * self.elevation.sin();
        let z = self.distance * self.elevation.cos() * self.azimuth.cos();
        self.target + Vec3::new(x, y, z)
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new()
    }
}
