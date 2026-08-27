//! Engine shading models — the code/registry side of the material system.
//!
//! A shading model is **engine-defined** (not an asset, per `docs/MATERIAL_ASSETS.md`
//! Decision 3): it declares the material property *schema* (what a material fills)
//! and which shader implements it. Materials reference a shading model by id.
//! For the first slice there is one model, `opaque`.

use std::collections::HashMap;

use redlilium_assets::Guid;

use crate::std::rendering::loaders::TextureSource;

/// A serializable material property value (asset-layer; **ref-based**, unlike
/// the inline `redlilium_core::material::MaterialValue`). A texture property
/// holds a [`TextureSource`] (an asset reference / solid color), not pixels.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PropValue {
    Float(f32),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Texture(TextureSource),
}

impl PropValue {
    /// Append this value's little-endian float bytes to `out` (uniform packing).
    /// Textures contribute nothing — they bind as texture/sampler slots, not
    /// uniform bytes (see [`texture_props`]).
    pub fn pack_into(&self, out: &mut Vec<u8>) {
        match self {
            PropValue::Float(v) => out.extend_from_slice(&v.to_le_bytes()),
            PropValue::Vec3(v) => v
                .iter()
                .for_each(|f| out.extend_from_slice(&f.to_le_bytes())),
            PropValue::Vec4(v) => v
                .iter()
                .for_each(|f| out.extend_from_slice(&f.to_le_bytes())),
            PropValue::Texture(_) => {}
        }
    }
}

/// Pack a resolved property list (schema order) into a contiguous uniform byte
/// buffer for upload to the material's static binding (group 1). Texture
/// properties are skipped — they bind as texture/sampler slots.
pub fn pack_props(props: &[(String, PropValue)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (_, v) in props {
        v.pack_into(&mut out);
    }
    out
}

/// The texture properties of a resolved list, in schema order — the canonical
/// binding order. In group 1, binding 0 is the packed uniform buffer; the
/// `i`-th texture property binds as texture `1 + 2*i`, sampler `2 + 2*i` (the
/// shader implementing the model declares the matching slots).
pub fn texture_props(props: &[(String, PropValue)]) -> Vec<(String, TextureSource)> {
    props
        .iter()
        .filter_map(|(name, v)| match v {
            PropValue::Texture(source) => Some((name.clone(), source.clone())),
            _ => None,
        })
        .collect()
}

/// A serializable feature-axis selection on a material — Decision 5's
/// *material half* of the shader variant split. Mirrors the graphics
/// [`VariantValue`](redlilium_graphics::VariantValue) shape: `Bool` for
/// `//#pragma variant NAME` axes, `Value` for enum axes.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FeatureValue {
    Bool(bool),
    Value(String),
}

impl From<&FeatureValue> for redlilium_graphics::VariantValue {
    fn from(value: &FeatureValue) -> Self {
        match value {
            FeatureValue::Bool(b) => Self::Bool(*b),
            FeatureValue::Value(v) => Self::Value(v.clone()),
        }
    }
}

/// One property slot in a shading model's schema: a name and its default value
/// (the value type is implied by the default's variant).
#[derive(Debug, Clone, PartialEq)]
pub struct PropDef {
    pub name: String,
    pub default: PropValue,
}

/// An engine-defined shading model — the contract between a material (which fills
/// the schema) and the shader that implements it.
#[derive(Debug, Clone)]
pub struct ShadingModel {
    /// Stable id that materials reference (e.g. `"opaque"`).
    pub id: String,
    /// The shader source asset implementing this model.
    pub shader: Guid,
    /// Property schema: ordered slots with defaults.
    pub schema: Vec<PropDef>,
}

impl ShadingModel {
    /// The default value for a named property, if the schema declares it.
    pub fn default_of(&self, name: &str) -> Option<&PropValue> {
        self.schema
            .iter()
            .find(|p| p.name == name)
            .map(|p| &p.default)
    }

    /// Resolve property values against this model's schema: every schema slot in
    /// order, taking the supplied value (matched by name) or the model default.
    /// The result is the canonical packing order for the material binding.
    pub fn resolve(&self, values: &[(String, PropValue)]) -> Vec<(String, PropValue)> {
        self.schema
            .iter()
            .map(|slot| {
                let value = values
                    .iter()
                    .find(|(n, _)| n == &slot.name)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_else(|| slot.default.clone());
                (slot.name.clone(), value)
            })
            .collect()
    }
}

/// The engine's shading-model registry (becomes an ECS resource in a later step).
/// Starts simple — a fixed built-in set; game-extensible registration can come
/// later (cheap to change).
#[derive(Debug, Clone)]
pub struct ShadingRegistry {
    models: HashMap<String, ShadingModel>,
}

impl ShadingRegistry {
    /// Build the registry with the built-in engine shading models.
    pub fn with_builtins() -> Self {
        let mut models = HashMap::new();
        for model in [
            Self::opaque(),
            Self::opaque_textured(),
            Self::pbr(),
            Self::pbr_textured(),
            Self::layered_decal(),
        ] {
            models.insert(model.id.clone(), model);
        }
        Self { models }
    }

    /// The `opaque` model: Blinn-Phong, one `base_color` (Vec4) property, backed
    /// by the std `opaque_color.slang` shader (bound by its stable guid).
    fn opaque() -> ShadingModel {
        ShadingModel {
            id: "opaque".to_owned(),
            shader: Guid::stable("shaders/opaque_color.slang"),
            schema: vec![PropDef {
                name: "base_color".to_owned(),
                default: PropValue::Vec4([1.0, 1.0, 1.0, 1.0]),
            }],
        }
    }

    /// The `opaque_textured` model: `opaque` plus a `base_texture` sampled in
    /// UV space and multiplied into the base color. Needs a UV-carrying vertex
    /// layout (e.g. position_normal_uv). Backed by `opaque_textured.slang`.
    fn opaque_textured() -> ShadingModel {
        ShadingModel {
            id: "opaque_textured".to_owned(),
            shader: Guid::stable("shaders/opaque_textured.slang"),
            schema: vec![
                PropDef {
                    name: "base_color".to_owned(),
                    default: PropValue::Vec4([1.0, 1.0, 1.0, 1.0]),
                },
                PropDef {
                    // x: alpha cutoff for the ALPHA_CUTOUT feature (a full
                    // Vec4 slot keeps the packed uniform 16-byte aligned,
                    // matching the shader's cbuffer layout).
                    name: "cutout_params".to_owned(),
                    default: PropValue::Vec4([0.5, 0.0, 0.0, 0.0]),
                },
                PropDef {
                    name: "base_texture".to_owned(),
                    default: PropValue::Texture(TextureSource::WHITE),
                },
            ],
        }
    }

    /// The `pbr` model (#144): metallic-roughness PBR drawn by the deferred
    /// path's G-buffer pass and lit by the resolve (IBL + key light). Backed
    /// by `deferred_gbuffer.slang`, which writes the properties to MRT
    /// instead of shading directly — so `pbr` materials render only through
    /// the `"deferred"` pipeline (the forward path's single color target
    /// cannot satisfy the shader's three outputs).
    fn pbr() -> ShadingModel {
        ShadingModel {
            id: "pbr".to_owned(),
            shader: Guid::stable("shaders/deferred_gbuffer.slang"),
            schema: vec![
                PropDef {
                    name: "base_color".to_owned(),
                    default: PropValue::Vec4([1.0, 1.0, 1.0, 1.0]),
                },
                PropDef {
                    // x: metallic, y: roughness (z, w reserved — a full Vec4
                    // slot keeps the packed uniform 16-byte aligned, matching
                    // the shader's cbuffer layout).
                    name: "pbr_params".to_owned(),
                    default: PropValue::Vec4([0.0, 0.5, 0.0, 0.0]),
                },
            ],
        }
    }

    /// The `pbr_textured` model (ADR-039): `pbr` plus the glTF
    /// metallic-roughness texture pair — sRGB base color and linear packed
    /// ORM (R = AO, G = roughness, B = metallic), each multiplying the
    /// factor properties. Needs a UV-carrying vertex layout. White 1×1
    /// defaults make an untextured instance render exactly like `pbr`.
    fn pbr_textured() -> ShadingModel {
        ShadingModel {
            id: "pbr_textured".to_owned(),
            shader: Guid::stable("shaders/deferred_gbuffer_textured.slang"),
            schema: vec![
                PropDef {
                    name: "base_color".to_owned(),
                    default: PropValue::Vec4([1.0, 1.0, 1.0, 1.0]),
                },
                PropDef {
                    // x: metallic factor, y: roughness factor (z, w
                    // reserved). Both default to 1: the ORM texture decides.
                    name: "pbr_params".to_owned(),
                    default: PropValue::Vec4([1.0, 1.0, 0.0, 0.0]),
                },
                PropDef {
                    name: "base_color_texture".to_owned(),
                    default: PropValue::Texture(TextureSource::WHITE),
                },
                PropDef {
                    name: "orm_texture".to_owned(),
                    default: PropValue::Texture(TextureSource::WHITE),
                },
            ],
        }
    }

    /// The `layered_decal` model (procedural: design/decals-design.md, baked-decal
    /// channel): `opaque_textured` plus up to 3 decal albedo layers composited over
    /// the base albedo. Each layer has a surface-UV → decal-UV affine (`layerN_uv` =
    /// offset.xy/scale.xy), a params slot (`layerN_params.x` = strength; 0 disables
    /// the layer so a decal-less instance renders as the base), and its own texture
    /// (white 1×1 default). Schema order MUST equal the shader's MaterialParams field
    /// order — uniforms first (base_color, then each layer's uv+params), then the
    /// textures in declaration order. Needs a UV-carrying vertex layout.
    fn layered_decal() -> ShadingModel {
        let uv_default = PropValue::Vec4([0.0, 0.0, 1.0, 1.0]);
        let params_default = PropValue::Vec4([0.0, 0.0, 0.0, 0.0]);
        ShadingModel {
            id: "layered_decal".to_owned(),
            shader: Guid::stable("shaders/layered_decal.slang"),
            schema: vec![
                PropDef {
                    name: "base_color".to_owned(),
                    default: PropValue::Vec4([1.0, 1.0, 1.0, 1.0]),
                },
                PropDef {
                    name: "layer0_uv".to_owned(),
                    default: uv_default.clone(),
                },
                PropDef {
                    name: "layer0_params".to_owned(),
                    default: params_default.clone(),
                },
                PropDef {
                    name: "layer1_uv".to_owned(),
                    default: uv_default.clone(),
                },
                PropDef {
                    name: "layer1_params".to_owned(),
                    default: params_default.clone(),
                },
                PropDef {
                    name: "layer2_uv".to_owned(),
                    default: uv_default.clone(),
                },
                PropDef {
                    name: "layer2_params".to_owned(),
                    default: params_default.clone(),
                },
                PropDef {
                    name: "base_texture".to_owned(),
                    default: PropValue::Texture(TextureSource::WHITE),
                },
                PropDef {
                    name: "layer0_texture".to_owned(),
                    default: PropValue::Texture(TextureSource::WHITE),
                },
                PropDef {
                    name: "layer1_texture".to_owned(),
                    default: PropValue::Texture(TextureSource::WHITE),
                },
                PropDef {
                    name: "layer2_texture".to_owned(),
                    default: PropValue::Texture(TextureSource::WHITE),
                },
            ],
        }
    }

    /// Look up a model by id.
    pub fn get(&self, id: &str) -> Option<&ShadingModel> {
        self.models.get(id)
    }

    /// All registered model ids, sorted (stable order for UI lists).
    pub fn ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.models.keys().map(String::as_str).collect();
        ids.sort_unstable();
        ids
    }
}

impl Default for ShadingRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Texture properties contribute no uniform bytes (they bind as
    /// texture/sampler slots) and are extracted in schema order.
    #[test]
    fn texture_props_bind_not_pack() {
        let props = vec![
            ("base_color".to_owned(), PropValue::Vec4([1.0; 4])),
            (
                "base_texture".to_owned(),
                PropValue::Texture(TextureSource::WHITE),
            ),
        ];
        assert_eq!(pack_props(&props).len(), 16, "only the Vec4 packs");
        assert_eq!(
            texture_props(&props),
            vec![("base_texture".to_owned(), TextureSource::WHITE)]
        );
    }

    #[test]
    fn opaque_textured_model_registered() {
        let reg = ShadingRegistry::with_builtins();
        let m = reg.get("opaque_textured").expect("model present");
        assert_eq!(m.shader, Guid::stable("shaders/opaque_textured.slang"));
        assert_eq!(
            m.default_of("base_texture"),
            Some(&PropValue::Texture(TextureSource::WHITE))
        );
    }

    #[test]
    fn opaque_model_registered_with_schema() {
        let reg = ShadingRegistry::with_builtins();
        let m = reg.get("opaque").expect("opaque model present");
        assert_eq!(m.shader, Guid::stable("shaders/opaque_color.slang"));
        assert_eq!(
            m.default_of("base_color"),
            Some(&PropValue::Vec4([1.0, 1.0, 1.0, 1.0]))
        );
        assert!(reg.get("nonexistent").is_none());
    }
}
