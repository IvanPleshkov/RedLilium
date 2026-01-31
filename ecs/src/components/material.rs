//! Material components for defining surface appearance of rendered entities.

use bevy_ecs::component::Component;
use glam::Vec4;

/// Basic PBR (Physically Based Rendering) material component.
///
/// Defines the visual appearance of a rendered mesh using a metallic-roughness workflow.
///
/// # Example
///
/// ```
/// use redlilium_ecs::components::Material;
/// use glam::Vec4;
///
/// // Create a red metallic material
/// let material = Material::default()
///     .with_base_color(Vec4::new(1.0, 0.0, 0.0, 1.0))
///     .with_metallic(1.0)
///     .with_roughness(0.3);
/// ```
#[derive(Component, Debug, Clone, PartialEq)]
pub struct Material {
    /// Base color (albedo) in linear RGBA. Default is white (1, 1, 1, 1).
    pub base_color: Vec4,

    /// Metallic factor from 0.0 (dielectric) to 1.0 (metal). Default is 0.0.
    pub metallic: f32,

    /// Roughness factor from 0.0 (smooth) to 1.0 (rough). Default is 0.5.
    pub roughness: f32,

    /// Emissive color in linear RGB. Default is black (no emission).
    pub emissive: glam::Vec3,

    /// Optional base color texture handle.
    pub base_color_texture: Option<TextureHandle>,

    /// Optional normal map texture handle.
    pub normal_texture: Option<TextureHandle>,

    /// Optional metallic-roughness texture handle (metallic in B, roughness in G).
    pub metallic_roughness_texture: Option<TextureHandle>,

    /// Optional emissive texture handle.
    pub emissive_texture: Option<TextureHandle>,

    /// Optional occlusion texture handle.
    pub occlusion_texture: Option<TextureHandle>,

    /// Alpha mode for transparency handling.
    pub alpha_mode: AlphaMode,

    /// Double-sided rendering flag.
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Vec4::ONE,
            metallic: 0.0,
            roughness: 0.5,
            emissive: glam::Vec3::ZERO,
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
            occlusion_texture: None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
        }
    }
}

impl Material {
    /// Creates a new material with default PBR values.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns this material with a different base color.
    #[inline]
    #[must_use]
    pub fn with_base_color(mut self, color: Vec4) -> Self {
        self.base_color = color;
        self
    }

    /// Returns this material with a different metallic value.
    #[inline]
    #[must_use]
    pub fn with_metallic(mut self, metallic: f32) -> Self {
        self.metallic = metallic.clamp(0.0, 1.0);
        self
    }

    /// Returns this material with a different roughness value.
    #[inline]
    #[must_use]
    pub fn with_roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness.clamp(0.0, 1.0);
        self
    }

    /// Returns this material with a different emissive color.
    #[inline]
    #[must_use]
    pub fn with_emissive(mut self, emissive: glam::Vec3) -> Self {
        self.emissive = emissive;
        self
    }

    /// Returns this material with a different alpha mode.
    #[inline]
    #[must_use]
    pub fn with_alpha_mode(mut self, alpha_mode: AlphaMode) -> Self {
        self.alpha_mode = alpha_mode;
        self
    }

    /// Returns this material with double-sided rendering enabled.
    #[inline]
    #[must_use]
    pub fn with_double_sided(mut self, double_sided: bool) -> Self {
        self.double_sided = double_sided;
        self
    }

    /// Creates a simple unlit color material.
    #[inline]
    pub fn unlit(color: Vec4) -> Self {
        Self {
            base_color: color,
            emissive: color.truncate(),
            ..Default::default()
        }
    }
}

/// Handle to a texture resource.
///
/// This is a lightweight identifier that references a texture stored elsewhere
/// (e.g., in an asset manager or GPU resource pool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u64);

impl TextureHandle {
    /// Creates a new texture handle from an ID.
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying ID.
    #[inline]
    pub const fn id(&self) -> u64 {
        self.0
    }
}

/// Alpha blending mode for materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaMode {
    /// Fully opaque, alpha channel is ignored.
    #[default]
    Opaque,
    /// Alpha values below cutoff are discarded, rest are opaque.
    Mask {
        /// Alpha cutoff threshold (default 0.5).
        cutoff: u8,
    },
    /// Full alpha blending for transparent materials.
    Blend,
}

impl AlphaMode {
    /// Creates a mask alpha mode with the given cutoff (0.0 to 1.0).
    #[inline]
    pub fn mask(cutoff: f32) -> Self {
        Self::Mask {
            cutoff: (cutoff.clamp(0.0, 1.0) * 255.0) as u8,
        }
    }

    /// Returns the cutoff value for mask mode, or None for other modes.
    #[inline]
    pub fn cutoff(&self) -> Option<f32> {
        match self {
            Self::Mask { cutoff } => Some(*cutoff as f32 / 255.0),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_default() {
        let mat = Material::default();
        assert_eq!(mat.base_color, Vec4::ONE);
        assert_eq!(mat.metallic, 0.0);
        assert_eq!(mat.roughness, 0.5);
    }

    #[test]
    fn material_builder() {
        let mat = Material::new()
            .with_base_color(Vec4::new(1.0, 0.0, 0.0, 1.0))
            .with_metallic(0.8)
            .with_roughness(0.2);

        assert_eq!(mat.base_color, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(mat.metallic, 0.8);
        assert_eq!(mat.roughness, 0.2);
    }

    #[test]
    fn alpha_mode_mask() {
        let mode = AlphaMode::mask(0.5);
        let cutoff = mode.cutoff().unwrap();
        // Allow for u8 quantization error
        assert!((cutoff - 0.5).abs() < 0.01);
    }
}
