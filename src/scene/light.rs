//! Light types for the scene

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};

/// Light types
#[derive(Debug, Clone)]
pub enum Light {
    Point(PointLight),
    Spot(SpotLight),
    Directional(DirectionalLight),
}

impl Light {
    /// Get the light position (returns zero for directional lights)
    pub fn position(&self) -> Vec3 {
        match self {
            Light::Point(p) => p.position,
            Light::Spot(s) => s.position,
            Light::Directional(_) => Vec3::ZERO,
        }
    }

    /// Get the light radius (returns infinity for directional lights)
    pub fn radius(&self) -> f32 {
        match self {
            Light::Point(p) => p.radius,
            Light::Spot(s) => s.radius,
            Light::Directional(_) => f32::INFINITY,
        }
    }

    /// Convert to GPU data format
    pub fn to_gpu_data(&self) -> GpuLightData {
        match self {
            Light::Point(p) => GpuLightData {
                position: p.position.extend(p.radius),
                color_intensity: Vec4::new(p.color.x, p.color.y, p.color.z, p.intensity),
                direction_type: Vec4::new(0.0, 0.0, 0.0, 0.0), // type 0 = point
                spot_params: Vec4::ZERO,
            },
            Light::Spot(s) => GpuLightData {
                position: s.position.extend(s.radius),
                color_intensity: Vec4::new(s.color.x, s.color.y, s.color.z, s.intensity),
                direction_type: Vec4::new(s.direction.x, s.direction.y, s.direction.z, 1.0), // type 1 = spot
                spot_params: Vec4::new(
                    s.inner_angle.cos(),
                    s.outer_angle.cos(),
                    0.0,
                    0.0,
                ),
            },
            Light::Directional(d) => GpuLightData {
                position: Vec4::new(0.0, 0.0, 0.0, f32::INFINITY),
                color_intensity: Vec4::new(d.color.x, d.color.y, d.color.z, d.intensity),
                direction_type: Vec4::new(d.direction.x, d.direction.y, d.direction.z, 2.0), // type 2 = directional
                spot_params: Vec4::ZERO,
            },
        }
    }
}

/// Point light
#[derive(Debug, Clone)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1.0,
            radius: 10.0,
        }
    }
}

/// Spot light
#[derive(Debug, Clone)]
pub struct SpotLight {
    pub position: Vec3,
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
            position: Vec3::ZERO,
            direction: -Vec3::Y,
            color: Vec3::ONE,
            intensity: 1.0,
            radius: 10.0,
            inner_angle: 0.3,
            outer_angle: 0.5,
        }
    }
}

/// Directional light (like the sun)
#[derive(Debug, Clone)]
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
