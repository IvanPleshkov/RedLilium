//! CPU-side material definitions with declaration/instance split.
//!
//! Materials are split into two types mirroring the graphics crate:
//!
//! - [`CpuMaterial`] — **Declaration** defining pipeline state and binding
//!   layout (slot definitions). Shared via `Arc` across instances that use
//!   the same shader variant.
//! - [`CpuMaterialInstance`] — **Bindings** holding actual values in indexed
//!   slots that match the parent material's binding definitions.
//!
//! Supporting types:
//! - [`MaterialBindingDef`] — A single binding slot definition
//! - [`MaterialValueType`] — Expected value type for a binding slot
//! - [`MaterialValue`] — Typed property value (float, vec3, vec4, texture)
//! - [`TextureRef`] — Texture + sampler + UV set reference
//! - [`AlphaMode`] — Alpha rendering mode (opaque, mask with cutoff, blend)

use std::sync::Arc;

use crate::mesh::{PrimitiveTopology, VertexLayout};
use crate::sampler::CpuSampler;
use crate::texture::{CpuTexture, TextureFormat};

/// Type of value expected in a material binding slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialValueType {
    /// Single f32 value.
    Float,
    /// 3-component float vector.
    Vec3,
    /// 4-component float vector.
    Vec4,
    /// Texture reference.
    Texture,
}

/// Describes one binding slot in a [`CpuMaterial`].
///
/// Each slot has a human-readable name, an expected value type, and a GPU
/// binding index. Multiple uniform-type slots may share the same binding
/// (e.g., all scalar/vector uniforms packed into binding 0), while texture
/// slots each get their own binding index.
#[derive(Debug, Clone)]
pub struct MaterialBindingDef {
    /// Human-readable name (e.g., "base_color", "metallic_roughness_texture").
    pub name: String,
    /// Expected value type for this slot.
    pub value_type: MaterialValueType,
    /// GPU binding slot index this maps to.
    pub binding: u32,
}

/// A typed material property value.
#[derive(Debug, Clone, PartialEq)]
pub enum MaterialValue {
    /// Single float (metallic, roughness, normal scale, occlusion strength).
    Float(f32),
    /// 3-component vector (emissive factor).
    Vec3([f32; 3]),
    /// 4-component vector (base color factor).
    Vec4([f32; 4]),
    /// Texture reference.
    Texture(TextureRef),
}

/// How a texture is sourced.
#[derive(Debug, Clone)]
pub enum TextureSource {
    /// Owned CPU texture data (Arc-shared across materials).
    Cpu(Arc<CpuTexture>),
    /// Named texture reference (resolved externally by the application).
    Named(String),
}

/// Reference to a texture with sampler and UV set.
#[derive(Debug, Clone)]
pub struct TextureRef {
    /// The texture source (owned data or named reference).
    pub texture: TextureSource,
    /// Shared sampler configuration (Arc-shared across materials).
    pub sampler: Option<Arc<CpuSampler>>,
    /// Texture coordinate set index (0, 1, …).
    pub tex_coord: u32,
}

impl PartialEq for TextureRef {
    fn eq(&self, other: &Self) -> bool {
        self.tex_coord == other.tex_coord
            && match (&self.texture, &other.texture) {
                (TextureSource::Cpu(a), TextureSource::Cpu(b)) => Arc::ptr_eq(a, b),
                (TextureSource::Named(a), TextureSource::Named(b)) => a == b,
                _ => false,
            }
            && match (&self.sampler, &other.sampler) {
                (Some(a), Some(b)) => **a == **b,
                (None, None) => true,
                _ => false,
            }
    }
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
    Mask {
        /// Cutoff value (0.0–1.0). Fragments with alpha below this are discarded.
        cutoff: f32,
    },
    /// Full alpha blending.
    Blend,
}

/// CPU-side material declaration.
///
/// Defines the material "shape": pipeline state and binding slot layout.
/// Shared via `Arc` — two [`CpuMaterialInstance`]s with the same `CpuMaterial`
/// use the same shader variant.
///
/// # Example
///
/// ```ignore
/// use redlilium_core::material::*;
///
/// let layout = Arc::new(VertexLayout::new());
/// let mat = CpuMaterial::pbr_metallic_roughness(
///     layout,
///     AlphaMode::Opaque,
///     false,
///     true,  // has base color texture
///     false, // no metallic-roughness texture
///     false, // no normal texture
///     false, // no occlusion texture
///     false, // no emissive texture
/// );
/// ```
#[derive(Debug, Clone)]
pub struct CpuMaterial {
    /// Material name.
    pub name: Option<String>,
    /// Alpha rendering mode.
    pub alpha_mode: AlphaMode,
    /// Whether the material is double-sided.
    pub double_sided: bool,
    /// Expected vertex layout for meshes rendered with this material.
    /// Shared via `Arc` for pointer-based batching.
    pub vertex_layout: Arc<VertexLayout>,
    /// Primitive topology (how vertices are assembled).
    pub topology: PrimitiveTopology,
    /// Color attachment formats for the render pass this material targets.
    pub color_formats: Vec<TextureFormat>,
    /// Depth attachment format, if any.
    pub depth_format: Option<TextureFormat>,
    /// Binding slot definitions — ordered list of expected properties.
    pub bindings: Vec<MaterialBindingDef>,
}

impl CpuMaterial {
    /// Creates a new empty material (opaque, single-sided, empty layout, no bindings).
    pub fn new() -> Self {
        Self {
            name: None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            vertex_layout: Arc::new(VertexLayout::new()),
            topology: PrimitiveTopology::TriangleList,
            color_formats: Vec::new(),
            depth_format: None,
            bindings: Vec::new(),
        }
    }

    /// Set the expected vertex layout.
    #[must_use]
    pub fn with_vertex_layout(mut self, layout: Arc<VertexLayout>) -> Self {
        self.vertex_layout = layout;
        self
    }

    /// Set the primitive topology.
    #[must_use]
    pub fn with_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Add a color attachment format.
    #[must_use]
    pub fn with_color_format(mut self, format: TextureFormat) -> Self {
        self.color_formats.push(format);
        self
    }

    /// Set the depth attachment format.
    #[must_use]
    pub fn with_depth_format(mut self, format: TextureFormat) -> Self {
        self.depth_format = Some(format);
        self
    }

    /// Creates a PBR metallic-roughness material definition for glTF.
    ///
    /// Uniform properties (base color, metallic, roughness, emissive, normal
    /// scale, occlusion strength) always map to binding 0. Texture bindings
    /// are assigned incrementally starting from binding 1.
    ///
    /// # Binding slot order
    ///
    /// | Index | Name | Type | Binding |
    /// |-------|------|------|---------|
    /// | 0 | `base_color` | Vec4 | 0 |
    /// | 1 | `metallic` | Float | 0 |
    /// | 2 | `roughness` | Float | 0 |
    /// | 3 | `emissive` | Vec3 | 0 |
    /// | 4 | `normal_scale` | Float | 0 |
    /// | 5 | `occlusion_strength` | Float | 0 |
    /// | 6+ | texture slots | Texture | 1+ |
    #[allow(clippy::too_many_arguments)]
    pub fn pbr_metallic_roughness(
        vertex_layout: Arc<VertexLayout>,
        alpha_mode: AlphaMode,
        double_sided: bool,
        has_base_color_tex: bool,
        has_metallic_roughness_tex: bool,
        has_normal_tex: bool,
        has_occlusion_tex: bool,
        has_emissive_tex: bool,
    ) -> Self {
        use MaterialValueType::*;

        let mut bindings = vec![
            MaterialBindingDef {
                name: "base_color".into(),
                value_type: Vec4,
                binding: 0,
            },
            MaterialBindingDef {
                name: "metallic".into(),
                value_type: Float,
                binding: 0,
            },
            MaterialBindingDef {
                name: "roughness".into(),
                value_type: Float,
                binding: 0,
            },
            MaterialBindingDef {
                name: "emissive".into(),
                value_type: Vec3,
                binding: 0,
            },
            MaterialBindingDef {
                name: "normal_scale".into(),
                value_type: Float,
                binding: 0,
            },
            MaterialBindingDef {
                name: "occlusion_strength".into(),
                value_type: Float,
                binding: 0,
            },
        ];

        let mut next_binding = 1u32;
        for (has_tex, name) in [
            (has_base_color_tex, "base_color_texture"),
            (has_metallic_roughness_tex, "metallic_roughness_texture"),
            (has_normal_tex, "normal_texture"),
            (has_occlusion_tex, "occlusion_texture"),
            (has_emissive_tex, "emissive_texture"),
        ] {
            if has_tex {
                bindings.push(MaterialBindingDef {
                    name: name.into(),
                    value_type: Texture,
                    binding: next_binding,
                });
                next_binding += 1;
            }
        }

        Self {
            name: None,
            alpha_mode,
            double_sided,
            vertex_layout,
            topology: PrimitiveTopology::TriangleList,
            color_formats: Vec::new(),
            depth_format: None,
            bindings,
        }
    }

    /// Find a binding definition by name.
    pub fn find_binding(&self, name: &str) -> Option<(usize, &MaterialBindingDef)> {
        self.bindings
            .iter()
            .enumerate()
            .find(|(_, b)| b.name == name)
    }
}

impl Default for CpuMaterial {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU-side material instance holding actual binding values.
///
/// `values[i]` corresponds to `material.bindings[i]`. The material declaration
/// is shared via `Arc` so that instances with the same pipeline configuration
/// can be batched.
///
/// # Example
///
/// ```ignore
/// use redlilium_core::material::*;
/// use std::sync::Arc;
///
/// let layout = Arc::new(VertexLayout::new());
/// let mat = Arc::new(CpuMaterial::pbr_metallic_roughness(
///     layout, AlphaMode::Opaque, false,
///     false, false, false, false, false,
/// ));
/// let instance = CpuMaterialInstance::new(Arc::clone(&mat))
///     .with_name("red_metal")
///     .with_value(0, MaterialValue::Vec4([1.0, 0.0, 0.0, 1.0]))
///     .with_value(1, MaterialValue::Float(1.0))
///     .with_value(2, MaterialValue::Float(0.3));
/// ```
#[derive(Debug, Clone)]
pub struct CpuMaterialInstance {
    /// The material declaration (pipeline state + binding layout).
    pub material: Arc<CpuMaterial>,
    /// Instance name.
    pub name: Option<String>,
    /// Values for each binding slot, indexed to match `material.bindings`.
    pub values: Vec<MaterialValue>,
}

impl CpuMaterialInstance {
    /// Create a new instance with default values for all binding slots.
    pub fn new(material: Arc<CpuMaterial>) -> Self {
        let values = material
            .bindings
            .iter()
            .map(|b| match b.value_type {
                MaterialValueType::Float => MaterialValue::Float(0.0),
                MaterialValueType::Vec3 => MaterialValue::Vec3([0.0; 3]),
                MaterialValueType::Vec4 => MaterialValue::Vec4([0.0, 0.0, 0.0, 1.0]),
                MaterialValueType::Texture => MaterialValue::Texture(TextureRef {
                    texture: TextureSource::Named(String::new()),
                    sampler: None,
                    tex_coord: 0,
                }),
            })
            .collect();
        Self {
            material,
            name: None,
            values,
        }
    }

    /// Set the instance name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set a value at the given binding index.
    #[must_use]
    pub fn with_value(mut self, index: usize, value: MaterialValue) -> Self {
        if index < self.values.len() {
            self.values[index] = value;
        }
        self
    }

    /// Get a value by binding name.
    pub fn get(&self, name: &str) -> Option<&MaterialValue> {
        let (idx, _) = self.material.find_binding(name)?;
        self.values.get(idx)
    }

    /// Get a float value by binding name.
    pub fn get_float(&self, name: &str) -> Option<f32> {
        match self.get(name)? {
            MaterialValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a vec3 value by binding name.
    pub fn get_vec3(&self, name: &str) -> Option<[f32; 3]> {
        match self.get(name)? {
            MaterialValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a vec4 value by binding name.
    pub fn get_vec4(&self, name: &str) -> Option<[f32; 4]> {
        match self.get(name)? {
            MaterialValue::Vec4(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a texture reference by binding name.
    pub fn get_texture(&self, name: &str) -> Option<&TextureRef> {
        match self.get(name)? {
            MaterialValue::Texture(t) => Some(t),
            _ => None,
        }
    }

    /// Iterator over all texture values in this instance.
    pub fn textures(&self) -> impl Iterator<Item = &TextureRef> {
        self.values.iter().filter_map(|v| match v {
            MaterialValue::Texture(t) => Some(t),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::VertexLayout;

    fn test_layout() -> Arc<VertexLayout> {
        Arc::new(VertexLayout::new())
    }

    #[test]
    fn cpu_material_default() {
        let mat = CpuMaterial::new();
        assert!(mat.name.is_none());
        assert_eq!(mat.alpha_mode, AlphaMode::Opaque);
        assert!(!mat.double_sided);
        assert!(mat.bindings.is_empty());
    }

    #[test]
    fn pbr_no_textures() {
        let mat = CpuMaterial::pbr_metallic_roughness(
            test_layout(),
            AlphaMode::Opaque,
            false,
            false,
            false,
            false,
            false,
            false,
        );
        assert_eq!(mat.bindings.len(), 6);
        assert!(mat.find_binding("base_color").is_some());
        assert!(mat.find_binding("metallic").is_some());
        assert!(mat.find_binding("roughness").is_some());
        assert!(mat.find_binding("emissive").is_some());
        assert!(mat.find_binding("normal_scale").is_some());
        assert!(mat.find_binding("occlusion_strength").is_some());
        assert!(mat.find_binding("base_color_texture").is_none());
    }

    #[test]
    fn pbr_all_textures() {
        let mat = CpuMaterial::pbr_metallic_roughness(
            test_layout(),
            AlphaMode::Blend,
            true,
            true,
            true,
            true,
            true,
            true,
        );
        assert_eq!(mat.bindings.len(), 11);
        assert_eq!(mat.alpha_mode, AlphaMode::Blend);
        assert!(mat.double_sided);

        let tex_bindings: Vec<u32> = mat
            .bindings
            .iter()
            .filter(|b| b.value_type == MaterialValueType::Texture)
            .map(|b| b.binding)
            .collect();
        assert_eq!(tex_bindings, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn alpha_mode_mask_cutoff() {
        let mat = CpuMaterial::pbr_metallic_roughness(
            test_layout(),
            AlphaMode::Mask { cutoff: 0.5 },
            false,
            false,
            false,
            false,
            false,
            false,
        );
        assert_eq!(mat.alpha_mode, AlphaMode::Mask { cutoff: 0.5 });
    }

    #[test]
    fn cpu_material_instance_getters() {
        let mat = Arc::new(CpuMaterial::pbr_metallic_roughness(
            test_layout(),
            AlphaMode::Opaque,
            false,
            true,
            false,
            false,
            false,
            false,
        ));

        let instance = CpuMaterialInstance::new(Arc::clone(&mat))
            .with_name("test")
            .with_value(0, MaterialValue::Vec4([1.0, 0.0, 0.0, 1.0]))
            .with_value(1, MaterialValue::Float(0.8))
            .with_value(2, MaterialValue::Float(0.5))
            .with_value(3, MaterialValue::Vec3([0.1, 0.2, 0.3]));

        assert_eq!(instance.name.as_deref(), Some("test"));
        assert_eq!(instance.get_vec4("base_color"), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(instance.get_float("metallic"), Some(0.8));
        assert_eq!(instance.get_float("roughness"), Some(0.5));
        assert_eq!(instance.get_vec3("emissive"), Some([0.1, 0.2, 0.3]));
        assert!(instance.get("nonexistent").is_none());
    }

    #[test]
    fn cpu_material_instance_textures() {
        let mat = Arc::new(CpuMaterial::pbr_metallic_roughness(
            test_layout(),
            AlphaMode::Opaque,
            false,
            true,
            false,
            false,
            false,
            true,
        ));

        let instance = CpuMaterialInstance::new(Arc::clone(&mat))
            .with_value(
                6,
                MaterialValue::Texture(TextureRef {
                    texture: TextureSource::Named("base_tex".into()),
                    sampler: Some(Arc::new(CpuSampler::linear())),
                    tex_coord: 0,
                }),
            )
            .with_value(
                7,
                MaterialValue::Texture(TextureRef {
                    texture: TextureSource::Named("emissive_tex".into()),
                    sampler: None,
                    tex_coord: 1,
                }),
            );

        let tex = instance.get_texture("base_color_texture").unwrap();
        assert!(matches!(&tex.texture, TextureSource::Named(n) if n == "base_tex"));
        assert!(tex.sampler.is_some());

        let tex2 = instance.get_texture("emissive_texture").unwrap();
        assert!(matches!(&tex2.texture, TextureSource::Named(n) if n == "emissive_tex"));

        let all_textures: Vec<_> = instance.textures().collect();
        assert_eq!(all_textures.len(), 2);
    }
}
