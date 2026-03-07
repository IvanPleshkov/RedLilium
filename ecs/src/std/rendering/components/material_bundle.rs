//! Render pass type and material bundle.

use std::collections::HashMap;
use std::sync::Arc;

use redlilium_graphics::MaterialInstance;

use super::super::resources::TextureManager;
use crate::serialize::Value;

/// Identifies which render pass a material instance is intended for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPassType {
    /// Main forward color pass.
    Forward,
    /// Depth-only prepass (for early-Z or screen-space effects).
    DepthPrepass,
    /// Shadow map pass.
    Shadow,
    /// Deferred G-buffer pass.
    Deferred,
    /// Entity index pass — renders entity ID to an R32Uint target for picking.
    EntityIndex,
}

impl RenderPassType {
    /// All known render pass types.
    pub const ALL: &[Self] = &[
        Self::Forward,
        Self::DepthPrepass,
        Self::Shadow,
        Self::Deferred,
        Self::EntityIndex,
    ];

    /// Serialize to a string key.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Forward => "Forward",
            Self::DepthPrepass => "DepthPrepass",
            Self::Shadow => "Shadow",
            Self::Deferred => "Deferred",
            Self::EntityIndex => "EntityIndex",
        }
    }

    /// Parse from a string key.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Forward" => Some(Self::Forward),
            "DepthPrepass" => Some(Self::DepthPrepass),
            "Shadow" => Some(Self::Shadow),
            "Deferred" => Some(Self::Deferred),
            "EntityIndex" => Some(Self::EntityIndex),
            _ => None,
        }
    }
}

/// A collection of [`MaterialInstance`]s for different render passes.
///
/// All instances in a bundle share the same binding groups (textures, uniform
/// buffers) but reference different GPU [`Material`](redlilium_graphics::Material)
/// pipelines — one per pass type.
#[derive(Debug, Clone)]
pub struct MaterialBundle {
    /// Material instance per pass type.
    passes: HashMap<RenderPassType, Arc<MaterialInstance>>,
    /// Shared binding groups referenced by each instance.
    shared_bindings: Vec<Arc<redlilium_graphics::BindingGroup>>,
    /// Optional debug label.
    label: Option<String>,
}

impl MaterialBundle {
    /// Create an empty material bundle.
    pub fn new() -> Self {
        Self {
            passes: HashMap::new(),
            shared_bindings: Vec::new(),
            label: None,
        }
    }

    /// Add a material instance for a specific render pass.
    pub fn with_pass(mut self, pass_type: RenderPassType, instance: Arc<MaterialInstance>) -> Self {
        self.passes.insert(pass_type, instance);
        self
    }

    /// Set the shared binding groups.
    pub fn with_shared_bindings(
        mut self,
        bindings: Vec<Arc<redlilium_graphics::BindingGroup>>,
    ) -> Self {
        self.shared_bindings = bindings;
        self
    }

    /// Set a debug label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the material instance for a specific pass type.
    pub fn get(&self, pass_type: RenderPassType) -> Option<&Arc<MaterialInstance>> {
        self.passes.get(&pass_type)
    }

    /// Get all pass entries.
    pub fn passes(&self) -> &HashMap<RenderPassType, Arc<MaterialInstance>> {
        &self.passes
    }

    /// Get the shared binding groups.
    pub fn shared_bindings(&self) -> &[Arc<redlilium_graphics::BindingGroup>] {
        &self.shared_bindings
    }

    /// Get the bundle label, if set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl Default for MaterialBundle {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Material value serialization helpers (used by render_material)
// ---------------------------------------------------------------------------

use redlilium_core::material::{MaterialValue, TextureRef, TextureSource};

pub(super) fn serialize_material_value(
    value: &MaterialValue,
    tex_manager: &TextureManager,
) -> Result<Value, crate::serialize::SerializeError> {
    match value {
        MaterialValue::Float(v) => Ok(Value::Map(vec![
            ("t".to_owned(), Value::String("f".to_owned())),
            ("v".to_owned(), Value::F32(*v)),
        ])),
        MaterialValue::Vec3(v) => Ok(Value::Map(vec![
            ("t".to_owned(), Value::String("v3".to_owned())),
            (
                "v".to_owned(),
                Value::List(v.iter().map(|f| Value::F32(*f)).collect()),
            ),
        ])),
        MaterialValue::Vec4(v) => Ok(Value::Map(vec![
            ("t".to_owned(), Value::String("v4".to_owned())),
            (
                "v".to_owned(),
                Value::List(v.iter().map(|f| Value::F32(*f)).collect()),
            ),
        ])),
        MaterialValue::Texture(tex_ref) => {
            let texture_name = match &tex_ref.texture {
                TextureSource::Named(name) => name.clone(),
                TextureSource::Cpu(cpu_tex) => cpu_tex
                    .name
                    .clone()
                    .unwrap_or_else(|| "<unnamed>".to_owned()),
            };

            let sampler_val = if let Some(sampler) = &tex_ref.sampler {
                Value::String(
                    sampler
                        .name
                        .clone()
                        .unwrap_or_else(|| "<unnamed>".to_owned()),
                )
            } else {
                Value::Null
            };

            let _ = tex_manager; // used for future texture name resolution

            Ok(Value::Map(vec![
                ("t".to_owned(), Value::String("tex".to_owned())),
                ("texture".to_owned(), Value::String(texture_name)),
                ("sampler".to_owned(), sampler_val),
                ("tex_coord".to_owned(), Value::U64(tex_ref.tex_coord as u64)),
            ]))
        }
    }
}

pub(super) fn deserialize_material_value(
    value: Value,
) -> Result<MaterialValue, crate::serialize::DeserializeError> {
    let fields = match value {
        Value::Map(fields) => fields,
        _ => {
            return Err(crate::serialize::DeserializeError::FormatError(
                "expected Map for material value".into(),
            ));
        }
    };

    let mut map: std::collections::HashMap<String, Value> = fields.into_iter().collect();
    let type_tag = match map.remove("t") {
        Some(Value::String(s)) => s,
        _ => {
            return Err(crate::serialize::DeserializeError::FormatError(
                "missing or invalid 't' field in material value".into(),
            ));
        }
    };

    match type_tag.as_str() {
        "f" => {
            let v = extract_f32(&map, "v")?;
            Ok(MaterialValue::Float(v))
        }
        "v3" => {
            let list = extract_f32_list(&map, "v", 3)?;
            Ok(MaterialValue::Vec3([list[0], list[1], list[2]]))
        }
        "v4" => {
            let list = extract_f32_list(&map, "v", 4)?;
            Ok(MaterialValue::Vec4([list[0], list[1], list[2], list[3]]))
        }
        "tex" => {
            let texture_name = match map.get("texture") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "missing 'texture' field in texture value".into(),
                    ));
                }
            };

            let sampler = match map.get("sampler") {
                Some(Value::String(s)) => {
                    let mut cpu_sampler = redlilium_core::sampler::CpuSampler::linear();
                    cpu_sampler.name = Some(s.clone());
                    Some(Arc::new(cpu_sampler))
                }
                Some(Value::Null) | None => None,
                _ => {
                    return Err(crate::serialize::DeserializeError::FormatError(
                        "invalid 'sampler' field in texture value".into(),
                    ));
                }
            };

            let tex_coord = match map.get("tex_coord") {
                Some(Value::U64(n)) => *n as u32,
                _ => 0,
            };

            Ok(MaterialValue::Texture(TextureRef {
                texture: TextureSource::Named(texture_name),
                sampler,
                tex_coord,
            }))
        }
        _ => Err(crate::serialize::DeserializeError::FormatError(format!(
            "unknown material value type tag: '{type_tag}'"
        ))),
    }
}

fn extract_f32(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
) -> Result<f32, crate::serialize::DeserializeError> {
    match map.get(key) {
        Some(Value::F32(v)) => Ok(*v),
        Some(Value::F64(v)) => Ok(*v as f32),
        Some(Value::I64(v)) => Ok(*v as f32),
        Some(Value::U64(v)) => Ok(*v as f32),
        _ => Err(crate::serialize::DeserializeError::FormatError(format!(
            "expected numeric value for '{key}'"
        ))),
    }
}

fn extract_f32_list(
    map: &std::collections::HashMap<String, Value>,
    key: &str,
    expected_len: usize,
) -> Result<Vec<f32>, crate::serialize::DeserializeError> {
    match map.get(key) {
        Some(Value::List(list)) => {
            if list.len() != expected_len {
                return Err(crate::serialize::DeserializeError::FormatError(format!(
                    "expected {expected_len} elements for '{key}', got {}",
                    list.len()
                )));
            }
            list.iter()
                .map(|v| match v {
                    Value::F32(f) => Ok(*f),
                    Value::F64(f) => Ok(*f as f32),
                    Value::I64(i) => Ok(*i as f32),
                    Value::U64(u) => Ok(*u as f32),
                    _ => Err(crate::serialize::DeserializeError::FormatError(
                        "expected numeric in list".into(),
                    )),
                })
                .collect()
        }
        _ => Err(crate::serialize::DeserializeError::FormatError(format!(
            "expected List for '{key}'"
        ))),
    }
}
