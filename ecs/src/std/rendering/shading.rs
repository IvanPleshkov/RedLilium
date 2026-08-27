//! Engine shading models — the code/registry side of the material system.
//!
//! A shading model is **engine-defined** (not an asset, per `docs/MATERIAL_ASSETS.md`
//! Decision 3): it declares the material property *schema* (what a material fills)
//! and which shader implements it. Materials reference a shading model by id.
//! For the first slice there is one model, `opaque`.

use std::collections::HashMap;

use redlilium_assets::Guid;

use crate::std::rendering::loaders::TextureSource;

/// Where a storage-buffer material property's contents come from (asset-layer).
/// Both bind at a `StructuredBuffer` / `ByteAddressBuffer` slot; they differ in
/// who owns the bytes:
/// - `Inline` — read-only data authored in the material record. The instance
///   manager allocates a `STORAGE` buffer and uploads it once through the frame
///   graph (like the packed uniform), so it needs no external producer.
/// - `Ref` — a GPU buffer published at runtime by another system (a compute
///   pass output, an ECS-owned buffer) under a guid via
///   [`MaterialInstanceManager::publish_buffer`]. May be read-write. The
///   instance stays unresolved until the buffer is published (like a
///   still-loading texture).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StorageBufferSource {
    /// Read-only bytes authored in the material record.
    Inline(Vec<u8>),
    /// A buffer published at runtime under this guid.
    Ref(Guid),
}

/// A serializable material property value (asset-layer; **ref-based**, unlike
/// the inline `redlilium_core::material::MaterialValue`). A texture property
/// holds a [`TextureSource`] (an asset reference / solid color), not pixels; a
/// storage-buffer property holds a [`StorageBufferSource`] (inline bytes / a
/// published-buffer ref).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PropValue {
    Float(f32),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Texture(TextureSource),
    /// A layered texture bound at a `Texture2DArray` slot. The source must
    /// resolve to a D2Array texture (a `File` KTX2 array, or the
    /// [`TextureSource::WHITE_ARRAY`] solid default).
    TextureArray(TextureSource),
    /// A structured/byte-address storage buffer bound at the material set.
    StorageBuffer(StorageBufferSource),
}

impl PropValue {
    /// Append this value's little-endian float bytes to `out` (uniform packing).
    /// Textures and storage buffers contribute nothing — they bind as
    /// descriptor slots, not uniform bytes (see [`opaque_bindings`]).
    pub fn pack_into(&self, out: &mut Vec<u8>) {
        match self {
            PropValue::Float(v) => out.extend_from_slice(&v.to_le_bytes()),
            PropValue::Vec3(v) => v
                .iter()
                .for_each(|f| out.extend_from_slice(&f.to_le_bytes())),
            PropValue::Vec4(v) => v
                .iter()
                .for_each(|f| out.extend_from_slice(&f.to_le_bytes())),
            PropValue::Texture(_) | PropValue::TextureArray(_) | PropValue::StorageBuffer(_) => {}
        }
    }
}

/// One opaque (non-uniform) material property, in schema order — the canonical
/// binding order for the material set. After the packed uniform buffer at
/// binding 0, each opaque property takes the next descriptor slot(s): a texture
/// (plain or array) takes a texture + sampler pair; a storage buffer takes one
/// buffer slot. This must match the shader's `MaterialParams` field order (the
/// engine reflects opaque fields to consecutive descriptor slots — Decision 7).
#[derive(Debug, Clone, PartialEq)]
pub enum OpaqueBinding {
    /// A sampled texture (plain 2D or D2Array) — binds texture + sampler.
    Texture(TextureSource),
    /// A storage buffer — binds a single buffer slot.
    StorageBuffer(StorageBufferSource),
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

/// The texture properties of a resolved list, in schema order — every property
/// bound at a texture slot: both plain [`PropValue::Texture`] and layered
/// [`PropValue::TextureArray`] (each resolves to a GPU texture + sampler pair).
/// Used for hot-reload pull-validation of the bound textures; the actual slot
/// assignment (which interleaves storage buffers) goes through
/// [`opaque_bindings`].
pub fn texture_props(props: &[(String, PropValue)]) -> Vec<(String, TextureSource)> {
    props
        .iter()
        .filter_map(|(name, v)| match v {
            PropValue::Texture(source) | PropValue::TextureArray(source) => {
                Some((name.clone(), source.clone()))
            }
            _ => None,
        })
        .collect()
}

/// The opaque (non-uniform) properties of a resolved list, in schema order —
/// the canonical descriptor-slot order for the material set. Binding 0 is the
/// packed uniform buffer; then each entry here takes the next slot(s): a
/// [`OpaqueBinding::Texture`] a texture + sampler pair, a
/// [`OpaqueBinding::StorageBuffer`] one buffer slot. The shader's
/// `MaterialParams` declares the matching fields in this same order.
pub fn opaque_bindings(props: &[(String, PropValue)]) -> Vec<(String, OpaqueBinding)> {
    props
        .iter()
        .filter_map(|(name, v)| match v {
            PropValue::Texture(source) | PropValue::TextureArray(source) => {
                Some((name.clone(), OpaqueBinding::Texture(source.clone())))
            }
            PropValue::StorageBuffer(source) => {
                Some((name.clone(), OpaqueBinding::StorageBuffer(source.clone())))
            }
            PropValue::Float(_) | PropValue::Vec3(_) | PropValue::Vec4(_) => None,
        })
        .collect()
}

/// The first descriptor slot each opaque binding occupies, in schema order —
/// the spec the instance manager builds the material set to. Binding 0 is the
/// packed uniform constant buffer when the model has uniform props
/// (`has_uniform`); opaque slots follow. A texture occupies two consecutive
/// slots (texture, then sampler); a storage buffer occupies one. Matches the
/// consecutive descriptor slots Slang reflects the shader's opaque
/// `MaterialParams` fields to (Decision 7).
pub fn opaque_slot_layout(has_uniform: bool, bindings: &[OpaqueBinding]) -> Vec<u32> {
    let mut next = u32::from(has_uniform);
    bindings
        .iter()
        .map(|binding| {
            let slot = next;
            next += match binding {
                OpaqueBinding::Texture(_) => 2,
                OpaqueBinding::StorageBuffer(_) => 1,
            };
            slot
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
            Self::array_storage_demo(),
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

    /// The `array_storage_demo` model — the reference/demo surface exercising
    /// the two newer material-property kinds end-to-end (the ones a plain
    /// scalar/vector/texture schema can't express): a `Texture2DArray` layered
    /// texture and a read-only `StructuredBuffer`. Backed by
    /// `array_storage_demo.slang` (forward, single color target, UV layout).
    /// Schema order MUST equal the shader's `MaterialParams` field order:
    /// uniform (`base_color`) first, then the array texture, then the storage
    /// buffer. The defaults (white 1-layer array + a single white tint) make an
    /// unassigned instance render as a flat `base_color`.
    fn array_storage_demo() -> ShadingModel {
        // One float4 tint, white — 16 read-only bytes uploaded as a storage
        // buffer (the shader reads `tints[0]`).
        let white_tint: Vec<u8> = [1.0f32; 4].iter().flat_map(|f| f.to_le_bytes()).collect();
        ShadingModel {
            id: "array_storage_demo".to_owned(),
            shader: Guid::stable("shaders/array_storage_demo.slang"),
            schema: vec![
                PropDef {
                    name: "base_color".to_owned(),
                    default: PropValue::Vec4([1.0, 1.0, 1.0, 1.0]),
                },
                PropDef {
                    name: "layers".to_owned(),
                    default: PropValue::TextureArray(TextureSource::WHITE_ARRAY),
                },
                PropDef {
                    name: "tints".to_owned(),
                    default: PropValue::StorageBuffer(StorageBufferSource::Inline(white_tint)),
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

    /// Texture-array and storage-buffer properties pack no uniform bytes and are
    /// extracted as opaque bindings in schema order (the descriptor-slot order).
    #[test]
    fn opaque_bindings_order_and_no_packing() {
        let props = vec![
            ("base_color".to_owned(), PropValue::Vec4([1.0; 4])),
            (
                "layers".to_owned(),
                PropValue::TextureArray(TextureSource::WHITE_ARRAY),
            ),
            (
                "tints".to_owned(),
                PropValue::StorageBuffer(StorageBufferSource::Inline(vec![0u8; 16])),
            ),
        ];
        // Only the Vec4 packs; array/buffer contribute no uniform bytes.
        assert_eq!(pack_props(&props).len(), 16);

        let bindings = opaque_bindings(&props);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].0, "layers");
        assert!(matches!(bindings[0].1, OpaqueBinding::Texture(_)));
        assert_eq!(bindings[1].0, "tints");
        assert!(matches!(bindings[1].1, OpaqueBinding::StorageBuffer(_)));

        // A texture array is also collected by `texture_props` (it resolves to a
        // GPU texture + sampler), the storage buffer is not.
        assert_eq!(
            texture_props(&props),
            vec![("layers".to_owned(), TextureSource::WHITE_ARRAY)]
        );
    }

    /// Slot layout after the uniform buffer: a texture takes two consecutive
    /// slots (texture, sampler), a storage buffer takes one — interleaved in
    /// schema order.
    #[test]
    fn opaque_slot_layout_interleaves_widths() {
        let bindings = vec![
            OpaqueBinding::Texture(TextureSource::WHITE),
            OpaqueBinding::StorageBuffer(StorageBufferSource::Ref(Guid::stable("buf"))),
            OpaqueBinding::Texture(TextureSource::WHITE_ARRAY),
        ];
        // With a uniform at 0: tex@1(+sampler 2), buffer@3, tex@4(+sampler 5).
        assert_eq!(opaque_slot_layout(true, &bindings), vec![1, 3, 4]);
        // Without a uniform, opaque slots start at 0.
        assert_eq!(opaque_slot_layout(false, &bindings), vec![0, 2, 3]);
    }

    /// The demo model is registered with the array + buffer schema in the field
    /// order its shader declares.
    #[test]
    fn array_storage_demo_model_registered() {
        let reg = ShadingRegistry::with_builtins();
        let m = reg.get("array_storage_demo").expect("model present");
        assert_eq!(m.shader, Guid::stable("shaders/array_storage_demo.slang"));
        let names: Vec<&str> = m.schema.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["base_color", "layers", "tints"]);
        assert!(matches!(
            m.default_of("layers"),
            Some(PropValue::TextureArray(_))
        ));
        assert!(matches!(
            m.default_of("tints"),
            Some(PropValue::StorageBuffer(StorageBufferSource::Inline(_)))
        ));
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
