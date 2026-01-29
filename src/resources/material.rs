//! Material definitions for PBR rendering

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};

/// PBR material properties
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub base_color: Vec4,
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: Vec3,
    pub emissive_strength: f32,

    /// Texture IDs (None means use default)
    pub base_color_texture: Option<usize>,
    pub normal_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub emissive_texture: Option<usize>,
    pub occlusion_texture: Option<usize>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            base_color: Vec4::new(1.0, 1.0, 1.0, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            emissive: Vec3::ZERO,
            emissive_strength: 1.0,
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
            occlusion_texture: None,
        }
    }
}

impl Material {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    pub fn with_base_color(mut self, color: Vec4) -> Self {
        self.base_color = color;
        self
    }

    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.metallic = metallic;
        self
    }

    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness;
        self
    }

    pub fn with_emissive(mut self, emissive: Vec3, strength: f32) -> Self {
        self.emissive = emissive;
        self.emissive_strength = strength;
        self
    }

    /// Create a uniform data struct for GPU
    pub fn uniform_data(&self) -> MaterialUniformData {
        MaterialUniformData {
            base_color: self.base_color,
            metallic_roughness: [self.metallic, self.roughness, 0.0, 0.0],
            emissive: self.emissive.extend(self.emissive_strength),
        }
    }

    // Preset materials

    pub fn plastic(color: Vec3) -> Self {
        Self::new("plastic")
            .with_base_color(color.extend(1.0))
            .with_metallic(0.0)
            .with_roughness(0.4)
    }

    pub fn metal(color: Vec3, roughness: f32) -> Self {
        Self::new("metal")
            .with_base_color(color.extend(1.0))
            .with_metallic(1.0)
            .with_roughness(roughness)
    }

    pub fn gold() -> Self {
        Self::metal(Vec3::new(1.0, 0.766, 0.336), 0.3)
    }

    pub fn silver() -> Self {
        Self::metal(Vec3::new(0.972, 0.960, 0.915), 0.2)
    }

    pub fn copper() -> Self {
        Self::metal(Vec3::new(0.955, 0.637, 0.538), 0.4)
    }

    pub fn iron() -> Self {
        Self::metal(Vec3::new(0.56, 0.57, 0.58), 0.5)
    }

    pub fn rubber(color: Vec3) -> Self {
        Self::new("rubber")
            .with_base_color(color.extend(1.0))
            .with_metallic(0.0)
            .with_roughness(0.9)
    }

    pub fn glass() -> Self {
        Self::new("glass")
            .with_base_color(Vec4::new(1.0, 1.0, 1.0, 0.3))
            .with_metallic(0.0)
            .with_roughness(0.1)
    }

    pub fn emissive(color: Vec3, strength: f32) -> Self {
        Self::new("emissive")
            .with_base_color(Vec4::ONE)
            .with_emissive(color, strength)
    }
}

/// Material uniform data for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialUniformData {
    pub base_color: Vec4,
    pub metallic_roughness: [f32; 4], // x=metallic, y=roughness, zw=padding
    pub emissive: Vec4,               // xyz=emissive, w=strength
}
