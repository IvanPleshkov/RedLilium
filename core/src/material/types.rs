//! Material data types for CPU-side material definitions.
//!
//! Materials use a property-based system where each property has a
//! [`MaterialSemantic`] tag and a typed [`MaterialValue`]. This design
//! bridges format-specific loaders (glTF, FBX, etc.) with the generic
//! graphics material system.

/// Well-known material property semantics.
///
/// Standard PBR metallic-roughness properties plus extensibility via [`Custom`](Self::Custom).
/// The graphics material system maps these semantics to shader binding slots.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MaterialSemantic {
    // -- PBR Metallic-Roughness --
    /// Base color factor `[r, g, b, a]`.
    BaseColorFactor,
    /// Base color texture.
    BaseColorTexture,
    /// Metallic factor (0.0–1.0).
    MetallicFactor,
    /// Roughness factor (0.0–1.0).
    RoughnessFactor,
    /// Metallic-roughness texture (B=metallic, G=roughness).
    MetallicRoughnessTexture,
    /// Normal map texture.
    NormalTexture,
    /// Normal map scale factor.
    NormalScale,
    /// Occlusion texture.
    OcclusionTexture,
    /// Occlusion strength (0.0–1.0).
    OcclusionStrength,
    /// Emissive factor `[r, g, b]`.
    EmissiveFactor,
    /// Emissive texture.
    EmissiveTexture,
    /// Alpha cutoff threshold (for [`AlphaMode::Mask`]).
    AlphaCutoff,

    // -- Extension --
    /// Custom property for non-standard semantics.
    Custom(String),
}

/// A typed material property value.
#[derive(Debug, Clone, PartialEq)]
pub enum MaterialValue {
    /// Single float (metallic, roughness, normal scale, occlusion strength, alpha cutoff).
    Float(f32),
    /// 3-component vector (emissive factor).
    Vec3([f32; 3]),
    /// 4-component vector (base color factor).
    Vec4([f32; 4]),
    /// Texture reference.
    Texture(TextureRef),
}

/// Reference to a texture with sampler and UV set.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureRef {
    /// Index into the owning container's texture array.
    pub texture: usize,
    /// Index into the owning container's sampler array.
    pub sampler: Option<usize>,
    /// Texture coordinate set index (0, 1, …).
    pub tex_coord: u32,
}

/// A single material property: semantic tag + typed value.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialProperty {
    /// What this property represents.
    pub semantic: MaterialSemantic,
    /// The property value.
    pub value: MaterialValue,
}

/// Alpha rendering mode.
///
/// Affects pipeline state (blend configuration), not shader bindings.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AlphaMode {
    /// Fully opaque (alpha ignored).
    #[default]
    Opaque,
    /// Alpha masking with cutoff threshold.
    Mask,
    /// Full alpha blending.
    Blend,
}

/// CPU-side material definition.
///
/// All material data is stored as a flat list of [`MaterialProperty`] entries
/// with semantic tags. Pipeline state ([`alpha_mode`](Self::alpha_mode),
/// [`double_sided`](Self::double_sided)) is separate since it affects
/// rendering configuration, not shader bindings.
///
/// # Example
///
/// ```ignore
/// use redlilium_core::material::*;
///
/// let mat = CpuMaterial::new()
///     .with_name("red_metal")
///     .with_property(MaterialProperty {
///         semantic: MaterialSemantic::BaseColorFactor,
///         value: MaterialValue::Vec4([1.0, 0.0, 0.0, 1.0]),
///     })
///     .with_property(MaterialProperty {
///         semantic: MaterialSemantic::MetallicFactor,
///         value: MaterialValue::Float(1.0),
///     });
/// ```
#[derive(Debug, Clone)]
pub struct CpuMaterial {
    /// Material name.
    pub name: Option<String>,
    /// Alpha rendering mode.
    pub alpha_mode: AlphaMode,
    /// Whether the material is double-sided.
    pub double_sided: bool,
    /// Material properties (PBR factors, textures, custom data).
    pub properties: Vec<MaterialProperty>,
}

impl CpuMaterial {
    /// Creates a new empty material (opaque, single-sided, no properties).
    pub fn new() -> Self {
        Self {
            name: None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            properties: Vec::new(),
        }
    }

    /// Set the material name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the alpha rendering mode.
    #[must_use]
    pub fn with_alpha_mode(mut self, alpha_mode: AlphaMode) -> Self {
        self.alpha_mode = alpha_mode;
        self
    }

    /// Set double-sided rendering.
    #[must_use]
    pub fn with_double_sided(mut self, double_sided: bool) -> Self {
        self.double_sided = double_sided;
        self
    }

    /// Add a property.
    #[must_use]
    pub fn with_property(mut self, property: MaterialProperty) -> Self {
        self.properties.push(property);
        self
    }

    /// Find a property value by semantic.
    pub fn get(&self, semantic: &MaterialSemantic) -> Option<&MaterialValue> {
        self.properties
            .iter()
            .find(|p| &p.semantic == semantic)
            .map(|p| &p.value)
    }

    /// Get a float property by semantic.
    pub fn get_float(&self, semantic: &MaterialSemantic) -> Option<f32> {
        match self.get(semantic)? {
            MaterialValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a vec3 property by semantic.
    pub fn get_vec3(&self, semantic: &MaterialSemantic) -> Option<[f32; 3]> {
        match self.get(semantic)? {
            MaterialValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a vec4 property by semantic.
    pub fn get_vec4(&self, semantic: &MaterialSemantic) -> Option<[f32; 4]> {
        match self.get(semantic)? {
            MaterialValue::Vec4(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a texture reference by semantic.
    pub fn get_texture(&self, semantic: &MaterialSemantic) -> Option<&TextureRef> {
        match self.get(semantic)? {
            MaterialValue::Texture(t) => Some(t),
            _ => None,
        }
    }
}

impl Default for CpuMaterial {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_material_default() {
        let mat = CpuMaterial::new();
        assert!(mat.name.is_none());
        assert_eq!(mat.alpha_mode, AlphaMode::Opaque);
        assert!(!mat.double_sided);
        assert!(mat.properties.is_empty());
    }

    #[test]
    fn cpu_material_builder() {
        let mat = CpuMaterial::new()
            .with_name("test")
            .with_alpha_mode(AlphaMode::Blend)
            .with_double_sided(true)
            .with_property(MaterialProperty {
                semantic: MaterialSemantic::BaseColorFactor,
                value: MaterialValue::Vec4([1.0, 0.0, 0.0, 1.0]),
            })
            .with_property(MaterialProperty {
                semantic: MaterialSemantic::MetallicFactor,
                value: MaterialValue::Float(0.8),
            });

        assert_eq!(mat.name.as_deref(), Some("test"));
        assert_eq!(mat.alpha_mode, AlphaMode::Blend);
        assert!(mat.double_sided);
        assert_eq!(mat.properties.len(), 2);
    }

    #[test]
    fn cpu_material_get_properties() {
        let mat = CpuMaterial::new()
            .with_property(MaterialProperty {
                semantic: MaterialSemantic::BaseColorFactor,
                value: MaterialValue::Vec4([1.0, 0.0, 0.0, 1.0]),
            })
            .with_property(MaterialProperty {
                semantic: MaterialSemantic::MetallicFactor,
                value: MaterialValue::Float(0.5),
            })
            .with_property(MaterialProperty {
                semantic: MaterialSemantic::EmissiveFactor,
                value: MaterialValue::Vec3([0.1, 0.2, 0.3]),
            })
            .with_property(MaterialProperty {
                semantic: MaterialSemantic::BaseColorTexture,
                value: MaterialValue::Texture(TextureRef {
                    texture: 0,
                    sampler: Some(1),
                    tex_coord: 0,
                }),
            });

        assert_eq!(
            mat.get_vec4(&MaterialSemantic::BaseColorFactor),
            Some([1.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(mat.get_float(&MaterialSemantic::MetallicFactor), Some(0.5));
        assert_eq!(
            mat.get_vec3(&MaterialSemantic::EmissiveFactor),
            Some([0.1, 0.2, 0.3])
        );

        let tex = mat
            .get_texture(&MaterialSemantic::BaseColorTexture)
            .unwrap();
        assert_eq!(tex.texture, 0);
        assert_eq!(tex.sampler, Some(1));

        assert!(mat.get(&MaterialSemantic::RoughnessFactor).is_none());
    }

    #[test]
    fn cpu_material_custom_semantic() {
        let mat = CpuMaterial::new().with_property(MaterialProperty {
            semantic: MaterialSemantic::Custom("clearcoat_factor".into()),
            value: MaterialValue::Float(0.7),
        });

        assert_eq!(
            mat.get_float(&MaterialSemantic::Custom("clearcoat_factor".into())),
            Some(0.7)
        );
    }
}
