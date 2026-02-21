use std::sync::Arc;

use redlilium_core::material::{CpuMaterialInstance, MaterialValue, TextureRef, TextureSource};
use redlilium_graphics::{MaterialInstance, Mesh, Texture};

use crate::serialize::Value;

/// GPU mesh component.
///
/// Wraps an `Arc<Mesh>` (GPU-uploaded mesh) so it can be attached to entities.
/// Entities with both `RenderMesh` and [`RenderMaterial`] are collected by the
/// forward render system and drawn each frame.
#[derive(Debug, Clone)]
pub struct RenderMesh(pub Arc<Mesh>);

impl crate::Component for RenderMesh {
    const NAME: &'static str = "RenderMesh";

    fn inspect_ui(&self, ui: &mut crate::egui::Ui) -> Option<Self> {
        ui.horizontal(|ui| {
            ui.label("mesh");
            match self.0.label() {
                Some(label) => ui.label(format!("Mesh: {label}")),
                None => ui.weak("Mesh (unnamed)"),
            };
        });
        None
    }

    fn collect_entities(&self, _collector: &mut Vec<crate::Entity>) {}

    fn remap_entities(&mut self, _map: &mut dyn FnMut(crate::Entity) -> crate::Entity) {}

    fn register_required(world: &mut crate::World) {
        world.register_required::<Self, crate::Transform>();
        world.register_required::<Self, crate::GlobalTransform>();
        world.register_required::<Self, crate::Visibility>();
    }

    fn serialize_component(
        &self,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<crate::serialize::Value, crate::serialize::SerializeError> {
        let mesh_name = {
            let world = ctx.world();
            if !world.has_resource::<super::MeshManager>() {
                return Err(crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "MeshManager resource not found".into(),
                });
            }
            let manager = world.resource::<super::MeshManager>();
            manager
                .find_name(&self.0)
                .or_else(|| self.0.label())
                .ok_or_else(|| crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "mesh has no registered name and no label".into(),
                })?
                .to_owned()
        };
        ctx.begin_struct(Self::NAME)?;
        ctx.write_serde("0", &mesh_name)?;
        ctx.end_struct()
    }

    fn deserialize_component(
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.begin_struct(Self::NAME)?;
        let mesh_name: String = ctx.read_serde("0")?;
        let world = ctx.world();
        if !world.has_resource::<super::MeshManager>() {
            return Err(crate::serialize::DeserializeError::FormatError(
                "MeshManager resource not found".into(),
            ));
        }
        let manager = world.resource::<super::MeshManager>();
        let mesh = manager.get_mesh(&mesh_name).ok_or_else(|| {
            crate::serialize::DeserializeError::FormatError(format!(
                "mesh '{mesh_name}' not found in MeshManager"
            ))
        })?;
        let mesh = Arc::clone(mesh);
        drop(manager);
        ctx.end_struct()?;
        Ok(Self(mesh))
    }
}

impl RenderMesh {
    /// Create a new render mesh component.
    pub fn new(mesh: Arc<Mesh>) -> Self {
        Self(mesh)
    }

    /// Get the inner GPU mesh.
    pub fn mesh(&self) -> &Arc<Mesh> {
        &self.0
    }
}

/// GPU material instance component.
///
/// Wraps an `Arc<MaterialInstance>` containing bound shader resources.
/// Attach alongside [`RenderMesh`] to make an entity renderable.
#[derive(Debug, Clone)]
pub struct RenderMaterial(pub Arc<MaterialInstance>);

impl crate::Component for RenderMaterial {
    const NAME: &'static str = "RenderMaterial";

    fn inspect_ui(&self, ui: &mut crate::egui::Ui) -> Option<Self> {
        ui.horizontal(|ui| {
            ui.label("material");
            match self.0.label() {
                Some(label) => ui.label(format!("Material: {label}")),
                None => ui.weak("Material (unnamed)"),
            };
        });
        None
    }

    fn collect_entities(&self, _collector: &mut Vec<crate::Entity>) {}

    fn remap_entities(&mut self, _map: &mut dyn FnMut(crate::Entity) -> crate::Entity) {}

    fn register_required(_world: &mut crate::World) {}

    fn serialize_component(
        &self,
        ctx: &mut crate::serialize::SerializeContext<'_>,
    ) -> Result<Value, crate::serialize::SerializeError> {
        // Extract all data from world resources in a scoped borrow
        let (material_name, instance_name, values) = {
            let world = ctx.world();
            if !world.has_resource::<super::MaterialManager>() {
                return Err(crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "MaterialManager resource not found".into(),
                });
            }
            if !world.has_resource::<super::TextureManager>() {
                return Err(crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "TextureManager resource not found".into(),
                });
            }

            let mat_manager = world.resource::<super::MaterialManager>();
            let tex_manager = world.resource::<super::TextureManager>();

            let cpu_instance = mat_manager.get_cpu_instance(&self.0).ok_or_else(|| {
                crate::serialize::SerializeError::FieldError {
                    field: "0".to_owned(),
                    message: "material instance not tracked in MaterialManager".into(),
                }
            })?;

            let material_name = cpu_instance
                .material
                .name
                .as_deref()
                .or_else(|| mat_manager.find_material_name(self.0.material()))
                .ok_or_else(|| crate::serialize::SerializeError::FieldError {
                    field: "material".to_owned(),
                    message: "material has no registered name".into(),
                })?
                .to_owned();

            let values: Vec<Value> = cpu_instance
                .values
                .iter()
                .map(|v| serialize_material_value(v, &tex_manager))
                .collect::<Result<_, _>>()?;

            let instance_name = cpu_instance.name.clone();

            (material_name, instance_name, values)
        };

        ctx.begin_struct(Self::NAME)?;
        ctx.write_serde("material", &material_name)?;
        if let Some(name) = &instance_name {
            ctx.write_serde("name", name)?;
        } else {
            ctx.write_field("name", Value::Null)?;
        }
        ctx.write_field("values", Value::List(values))?;
        ctx.end_struct()
    }

    fn deserialize_component(
        ctx: &mut crate::serialize::DeserializeContext<'_>,
    ) -> Result<Self, crate::serialize::DeserializeError> {
        ctx.begin_struct(Self::NAME)?;

        let material_name: String = ctx.read_serde("material")?;
        let instance_name: Option<String> = {
            let val = ctx.read_field("name")?;
            match val {
                Value::Null => None,
                Value::String(s) => Some(s),
                _ => {
                    return Err(crate::serialize::DeserializeError::TypeMismatch {
                        field: "name".to_owned(),
                        expected: "String or Null".into(),
                        found: format!("{val:?}"),
                    });
                }
            }
        };
        let values_val = ctx.read_field("values")?;
        let value_list = match values_val {
            Value::List(list) => list,
            _ => {
                return Err(crate::serialize::DeserializeError::TypeMismatch {
                    field: "values".to_owned(),
                    expected: "List".into(),
                    found: format!("{values_val:?}"),
                });
            }
        };

        // Deserialize values (no world access needed)
        let values: Vec<MaterialValue> = value_list
            .into_iter()
            .map(deserialize_material_value)
            .collect::<Result<_, _>>()?;

        // Create GPU instance from world resources (scoped to release ctx borrow)
        let gpu_instance = {
            let world = ctx.world_mut();
            if !world.has_resource::<super::MaterialManager>() {
                return Err(crate::serialize::DeserializeError::FormatError(
                    "MaterialManager resource not found".into(),
                ));
            }
            if !world.has_resource::<super::TextureManager>() {
                return Err(crate::serialize::DeserializeError::FormatError(
                    "TextureManager resource not found".into(),
                ));
            }

            // Get CPU material declaration
            let cpu_material = {
                let mat_manager = world.resource::<super::MaterialManager>();
                let cpu = mat_manager
                    .get_cpu_material(&material_name)
                    .ok_or_else(|| {
                        crate::serialize::DeserializeError::FormatError(format!(
                            "material '{material_name}' not found in MaterialManager"
                        ))
                    })?;
                Arc::clone(cpu)
            };

            // Build CpuMaterialInstance
            let mut cpu_instance = CpuMaterialInstance::new(cpu_material);
            cpu_instance.name = instance_name;
            cpu_instance.values = values;

            // Create GPU instance
            let mut mat_manager = world.resource_mut::<super::MaterialManager>();
            let mut tex_manager = world.resource_mut::<super::TextureManager>();
            mat_manager
                .create_instance(&cpu_instance, &mut tex_manager)
                .map_err(|e| {
                    crate::serialize::DeserializeError::FormatError(format!(
                        "failed to create material instance: {e}"
                    ))
                })?
        };

        ctx.end_struct()?;
        Ok(Self(gpu_instance))
    }
}

impl RenderMaterial {
    /// Create a new render material component.
    pub fn new(material: Arc<MaterialInstance>) -> Self {
        Self(material)
    }

    /// Get the inner material instance.
    pub fn material(&self) -> &Arc<MaterialInstance> {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// Material value serialization helpers
// ---------------------------------------------------------------------------

fn serialize_material_value(
    value: &MaterialValue,
    tex_manager: &super::TextureManager,
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

fn deserialize_material_value(
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

/// Render target for a camera entity.
///
/// Specifies which textures the camera renders to. Attach this to an entity
/// that already has a [`Camera`](crate::Camera) component. The forward render
/// system will create a graphics pass for each camera that has a `CameraTarget`.
///
/// The color and depth textures must be created with `TextureUsage::RENDER_ATTACHMENT`.
#[derive(Debug, Clone, crate::Component)]
#[skip_serialization]
pub struct CameraTarget {
    /// Color texture to render to.
    pub color: Arc<Texture>,
    /// Depth texture for depth testing.
    pub depth: Arc<Texture>,
    /// Clear color (RGBA) applied at the start of the render pass.
    pub clear_color: [f32; 4],
}

impl CameraTarget {
    /// Create a new camera target with the given textures and clear color.
    pub fn new(color: Arc<Texture>, depth: Arc<Texture>, clear_color: [f32; 4]) -> Self {
        Self {
            color,
            depth,
            clear_color,
        }
    }
}
