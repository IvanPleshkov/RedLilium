//! Light types for the scene

use bevy_ecs::prelude::*;
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};

/// Point light component
/// Position comes from the Transform component on the same entity
#[derive(Component, Debug, Clone)]
pub struct PointLight {
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: Vec3::ONE,
            intensity: 1.0,
            radius: 10.0,
        }
    }
}

impl PointLight {
    pub fn new(color: Vec3, intensity: f32, radius: f32) -> Self {
        Self { color, intensity, radius }
    }

    /// Convert to GPU data format
    pub fn to_gpu_data(&self, position: Vec3) -> GpuLightData {
        GpuLightData {
            position: position.extend(self.radius),
            color_intensity: Vec4::new(self.color.x, self.color.y, self.color.z, self.intensity),
            direction_type: Vec4::new(0.0, 0.0, 0.0, 0.0), // type 0 = point
            spot_params: Vec4::ZERO,
        }
    }
}

/// Spot light component
/// Position comes from the Transform component on the same entity
#[derive(Component, Debug, Clone)]
pub struct SpotLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub inner_angle: f32, // radians
    pub outer_angle: f32, // radians
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            direction: -Vec3::Y,
            color: Vec3::ONE,
            intensity: 1.0,
            radius: 10.0,
            inner_angle: 0.3,
            outer_angle: 0.5,
        }
    }
}

impl SpotLight {
    pub fn new(direction: Vec3, color: Vec3, intensity: f32, radius: f32, inner_angle: f32, outer_angle: f32) -> Self {
        Self { direction: direction.normalize(), color, intensity, radius, inner_angle, outer_angle }
    }

    /// Convert to GPU data format
    pub fn to_gpu_data(&self, position: Vec3) -> GpuLightData {
        GpuLightData {
            position: position.extend(self.radius),
            color_intensity: Vec4::new(self.color.x, self.color.y, self.color.z, self.intensity),
            direction_type: Vec4::new(self.direction.x, self.direction.y, self.direction.z, 1.0), // type 1 = spot
            spot_params: Vec4::new(
                self.inner_angle.cos(),
                self.outer_angle.cos(),
                0.0,
                0.0,
            ),
        }
    }
}

/// Directional light component (like the sun)
#[derive(Component, Debug, Clone)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(-0.5, -1.0, -0.5).normalize(),
            color: Vec3::ONE,
            intensity: 1.0,
        }
    }
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self { direction: direction.normalize(), color, intensity }
    }

    /// Convert to GPU data format
    pub fn to_gpu_data(&self) -> GpuLightData {
        GpuLightData {
            position: Vec4::new(0.0, 0.0, 0.0, f32::INFINITY),
            color_intensity: Vec4::new(self.color.x, self.color.y, self.color.z, self.intensity),
            direction_type: Vec4::new(self.direction.x, self.direction.y, self.direction.z, 2.0), // type 2 = directional
            spot_params: Vec4::ZERO,
        }
    }
}

/// GPU-friendly light data structure
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuLightData {
    /// xyz = position, w = radius
    pub position: Vec4,
    /// xyz = color, w = intensity
    pub color_intensity: Vec4,
    /// xyz = direction, w = light type (0=point, 1=spot, 2=directional)
    pub direction_type: Vec4,
    /// x = cos(inner_angle), y = cos(outer_angle), zw = unused
    pub spot_params: Vec4,
}

/// Light culling tile data
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TileLightData {
    /// Number of lights affecting this tile
    pub light_count: u32,
    /// Padding for alignment
    pub _padding: [u32; 3],
    /// Indices of lights affecting this tile (max 256 per tile)
    pub light_indices: [u32; 256],
}

impl Default for TileLightData {
    fn default() -> Self {
        Self {
            light_count: 0,
            _padding: [0; 3],
            light_indices: [0; 256],
        }
    }
}

/// Constants for Forward+ lighting
pub const MAX_LIGHTS_PER_TILE: u32 = 256;
pub const DEFAULT_TILE_SIZE: u32 = 16;
