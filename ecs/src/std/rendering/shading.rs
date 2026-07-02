//! Engine shading models — the code/registry side of the material system.
//!
//! A shading model is **engine-defined** (not an asset, per `docs/MATERIAL_ASSETS.md`
//! Decision 3): it declares the material property *schema* (what a material fills)
//! and which shader implements it. Materials reference a shading model by id.
//! For the first slice there is one model, `opaque`.

use std::collections::HashMap;

use redlilium_assets::Guid;

/// A serializable material property value (asset-layer; **ref-based**, unlike the
/// inline `redlilium_core::material::MaterialValue`). Textures will later be a
/// `Guid` asset reference rather than inline data.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PropValue {
    Float(f32),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
}

impl PropValue {
    /// Append this value's little-endian float bytes to `out` (uniform packing).
    pub fn pack_into(&self, out: &mut Vec<u8>) {
        match self {
            PropValue::Float(v) => out.extend_from_slice(&v.to_le_bytes()),
            PropValue::Vec3(v) => v
                .iter()
                .for_each(|f| out.extend_from_slice(&f.to_le_bytes())),
            PropValue::Vec4(v) => v
                .iter()
                .for_each(|f| out.extend_from_slice(&f.to_le_bytes())),
        }
    }
}

/// Pack a resolved property list (schema order) into a contiguous uniform byte
/// buffer for upload to the material's static binding (group 1).
pub fn pack_props(props: &[(String, PropValue)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (_, v) in props {
        v.pack_into(&mut out);
    }
    out
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
        let opaque = Self::opaque();
        models.insert(opaque.id.clone(), opaque);
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

    /// Look up a model by id.
    pub fn get(&self, id: &str) -> Option<&ShadingModel> {
        self.models.get(id)
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
