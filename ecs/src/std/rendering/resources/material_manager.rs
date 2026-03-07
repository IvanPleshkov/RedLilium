//! GPU material management.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use redlilium_core::material::{
    CpuMaterial, CpuMaterialInstance, MaterialValue, MaterialValueType, TextureRef, TextureSource,
};
use redlilium_graphics::{
    BindingGroup, Buffer, BufferDescriptor, BufferUsage, CpuSampler, GraphicsDevice, GraphicsError,
    Material, MaterialInstance, Sampler, Texture,
};

use super::texture_manager::TextureManager;
use crate::std::rendering::components::{MaterialBundle, RenderPassType};

/// Errors that can occur in [`MaterialManager`] operations.
#[derive(Debug)]
pub enum MaterialManagerError {
    /// GPU resource creation error.
    Graphics(GraphicsError),
    /// Material not found by name.
    MaterialNotFound(String),
    /// Texture not found in [`TextureManager`].
    TextureNotFound(String),
    /// Sampler not found in [`TextureManager`].
    SamplerNotFound(String),
    /// Value count doesn't match binding count.
    BindingMismatch {
        /// Number of bindings defined.
        expected: usize,
        /// Number of values provided.
        got: usize,
    },
}

impl fmt::Display for MaterialManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graphics(err) => write!(f, "graphics error: {err}"),
            Self::MaterialNotFound(name) => write!(f, "material '{name}' not found"),
            Self::TextureNotFound(name) => write!(f, "texture '{name}' not found"),
            Self::SamplerNotFound(name) => write!(f, "sampler '{name}' not found"),
            Self::BindingMismatch { expected, got } => {
                write!(f, "binding mismatch: expected {expected} values, got {got}")
            }
        }
    }
}

impl std::error::Error for MaterialManagerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graphics(err) => Some(err),
            _ => None,
        }
    }
}

impl From<GraphicsError> for MaterialManagerError {
    fn from(err: GraphicsError) -> Self {
        Self::Graphics(err)
    }
}

/// Tracks the CPU-side source data for a [`MaterialBundle`] to enable serialization.
pub struct CpuBundleInfo {
    /// The CPU material instance (binding values).
    pub cpu_instance: Arc<CpuMaterialInstance>,
    /// Pass type → registered material name mapping.
    pub pass_materials: Vec<(RenderPassType, String)>,
}

/// Resource for managing GPU materials and converting between CPU and GPU representations.
pub struct MaterialManager {
    device: Arc<GraphicsDevice>,
    /// Registered materials: name → (cpu declaration, gpu pipeline).
    materials: HashMap<String, (Arc<CpuMaterial>, Arc<Material>)>,
    /// CPU bundle info keyed by MaterialBundle Arc pointer.
    cpu_bundles: HashMap<usize, CpuBundleInfo>,
}

impl MaterialManager {
    /// Create a new material manager for the given device.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            materials: HashMap::new(),
            cpu_bundles: HashMap::new(),
        }
    }

    /// Get the graphics device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    // --- Material registration ---

    /// Register a material definition under the given name.
    pub fn register_material(
        &mut self,
        name: impl Into<String>,
        cpu_material: Arc<CpuMaterial>,
        gpu_material: Arc<Material>,
    ) {
        self.materials
            .insert(name.into(), (cpu_material, gpu_material));
    }

    /// Look up a registered GPU material by name.
    pub fn get_material(&self, name: &str) -> Option<&Arc<Material>> {
        self.materials.get(name).map(|(_, gpu)| gpu)
    }

    /// Look up a registered CPU material by name.
    pub fn get_cpu_material(&self, name: &str) -> Option<&Arc<CpuMaterial>> {
        self.materials.get(name).map(|(cpu, _)| cpu)
    }

    /// Find the registered name for a GPU material by Arc pointer identity.
    pub fn find_material_name(&self, material: &Arc<Material>) -> Option<&str> {
        self.materials
            .iter()
            .find(|(_, (_, gpu))| Arc::ptr_eq(gpu, material))
            .map(|(k, _)| k.as_str())
    }

    /// Remove a registered material by name.
    pub fn remove_material(&mut self, name: &str) -> Option<(Arc<CpuMaterial>, Arc<Material>)> {
        self.materials.remove(name)
    }

    // --- Bundle creation ---

    /// Create a [`MaterialBundle`] from a [`CpuMaterialInstance`] and a set of
    /// pass type → material name mappings.
    pub fn create_bundle(
        &mut self,
        cpu_instance: &CpuMaterialInstance,
        pass_materials: &[(RenderPassType, &str)],
        textures: &mut TextureManager,
    ) -> Result<Arc<MaterialBundle>, MaterialManagerError> {
        let cpu_mat = &cpu_instance.material;

        if cpu_instance.values.len() != cpu_mat.bindings.len() {
            return Err(MaterialManagerError::BindingMismatch {
                expected: cpu_mat.bindings.len(),
                got: cpu_instance.values.len(),
            });
        }

        // Build shared binding group once
        let shared_binding = Arc::new(self.build_binding_group(cpu_instance, textures)?);
        let shared_bindings = vec![Arc::clone(&shared_binding)];

        // Create a MaterialInstance per pass
        let mut bundle = MaterialBundle::new().with_shared_bindings(shared_bindings);

        for (pass_type, mat_name) in pass_materials {
            let gpu_material = self
                .get_material(mat_name)
                .cloned()
                .ok_or_else(|| MaterialManagerError::MaterialNotFound(mat_name.to_string()))?;

            let mut instance =
                MaterialInstance::new(gpu_material).with_binding_group(Arc::clone(&shared_binding));
            if let Some(name) = &cpu_instance.name {
                instance = instance.with_label(format!("{name}_{}", pass_type.as_str()));
            }

            bundle = bundle.with_pass(*pass_type, Arc::new(instance));
        }

        if let Some(name) = &cpu_instance.name {
            bundle = bundle.with_label(name.clone());
        }

        let bundle = Arc::new(bundle);

        // Track for serialization
        let info = CpuBundleInfo {
            cpu_instance: Arc::new(cpu_instance.clone()),
            pass_materials: pass_materials
                .iter()
                .map(|(p, n)| (*p, n.to_string()))
                .collect(),
        };
        let ptr = Arc::as_ptr(&bundle) as usize;
        self.cpu_bundles.insert(ptr, info);

        Ok(bundle)
    }

    /// Create a single-pass [`MaterialBundle`] (Forward only) from a [`CpuMaterialInstance`].
    pub fn create_instance(
        &mut self,
        cpu_instance: &CpuMaterialInstance,
        textures: &mut TextureManager,
    ) -> Result<Arc<MaterialBundle>, MaterialManagerError> {
        let cpu_mat = &cpu_instance.material;

        // Find the registered material name
        let mat_name = self
            .materials
            .iter()
            .find(|(_, (cpu, _))| Arc::ptr_eq(cpu, cpu_mat))
            .map(|(k, _)| k.clone())
            .or_else(|| {
                cpu_mat
                    .name
                    .as_ref()
                    .filter(|n| self.materials.contains_key(n.as_str()))
                    .cloned()
            })
            .ok_or_else(|| {
                MaterialManagerError::MaterialNotFound(
                    cpu_mat
                        .name
                        .clone()
                        .unwrap_or_else(|| "<unnamed>".to_owned()),
                )
            })?;

        self.create_bundle(
            cpu_instance,
            &[(RenderPassType::Forward, mat_name.as_str())],
            textures,
        )
    }

    /// Register an externally-created bundle with its CPU source data.
    pub fn register_bundle(
        &mut self,
        bundle: &Arc<MaterialBundle>,
        cpu_instance: Arc<CpuMaterialInstance>,
        pass_materials: Vec<(RenderPassType, String)>,
    ) {
        let ptr = Arc::as_ptr(bundle) as usize;
        self.cpu_bundles.insert(
            ptr,
            CpuBundleInfo {
                cpu_instance,
                pass_materials,
            },
        );
    }

    /// Get the CPU source data for a material bundle (for serialization).
    pub fn get_cpu_bundle(&self, bundle: &Arc<MaterialBundle>) -> Option<&CpuBundleInfo> {
        let ptr = Arc::as_ptr(bundle) as usize;
        self.cpu_bundles.get(&ptr)
    }

    // --- Iteration ---

    /// Get a reference to all registered materials.
    pub fn materials(&self) -> &HashMap<String, (Arc<CpuMaterial>, Arc<Material>)> {
        &self.materials
    }

    /// Iterate over all registered material names.
    pub fn material_names(&self) -> impl Iterator<Item = &str> {
        self.materials.keys().map(|s| s.as_str())
    }

    /// Returns the number of registered materials.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Returns the number of tracked bundles.
    pub fn bundle_count(&self) -> usize {
        self.cpu_bundles.len()
    }

    // --- Internal helpers ---

    /// Build a [`BindingGroup`] from a CPU material instance's values.
    pub(in crate::std::rendering) fn build_binding_group(
        &self,
        cpu_instance: &CpuMaterialInstance,
        textures: &mut TextureManager,
    ) -> Result<BindingGroup, MaterialManagerError> {
        let cpu_mat = &cpu_instance.material;

        // Pack uniform values (Float/Vec3/Vec4) into a byte buffer for binding 0
        let uniform_buffer = self.pack_uniforms(cpu_mat, &cpu_instance.values)?;

        let mut binding_group = BindingGroup::new();

        if let Some(buf) = uniform_buffer {
            binding_group = binding_group.with_buffer(0, buf);
        }

        // Add texture+sampler bindings
        for (i, value) in cpu_instance.values.iter().enumerate() {
            let binding_def = &cpu_mat.bindings[i];
            if binding_def.value_type != MaterialValueType::Texture {
                continue;
            }
            if let MaterialValue::Texture(tex_ref) = value {
                let (texture, sampler) = self.resolve_texture_ref(tex_ref, textures)?;
                binding_group = binding_group.with_combined(binding_def.binding, texture, sampler);
            }
        }

        Ok(binding_group)
    }

    /// Pack uniform (Float/Vec3/Vec4) values into a GPU buffer at binding 0.
    fn pack_uniforms(
        &self,
        cpu_mat: &CpuMaterial,
        values: &[MaterialValue],
    ) -> Result<Option<Arc<Buffer>>, MaterialManagerError> {
        let uniform_data = pack_uniform_bytes(cpu_mat, values);

        if uniform_data.is_empty() {
            return Ok(None);
        }

        let buffer = self.device.create_buffer(
            &BufferDescriptor::new(
                uniform_data.len() as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            )
            .with_label("material_uniforms"),
        )?;
        self.device.write_buffer(&buffer, 0, &uniform_data)?;
        Ok(Some(buffer))
    }

    /// Resolve a [`TextureRef`] to GPU texture + sampler Arcs.
    fn resolve_texture_ref(
        &self,
        tex_ref: &TextureRef,
        textures: &mut TextureManager,
    ) -> Result<(Arc<Texture>, Arc<Sampler>), MaterialManagerError> {
        let texture = match &tex_ref.texture {
            TextureSource::Named(name) => textures
                .get_texture(name)
                .cloned()
                .ok_or_else(|| MaterialManagerError::TextureNotFound(name.clone()))?,
            TextureSource::Cpu(cpu_tex) => textures.create_texture(cpu_tex)?,
        };

        let sampler = if let Some(cpu_sampler) = &tex_ref.sampler {
            if let Some(name) = &cpu_sampler.name {
                if let Some(s) = textures.get_sampler(name) {
                    Arc::clone(s)
                } else {
                    textures.create_sampler(cpu_sampler)?
                }
            } else {
                textures.create_sampler(cpu_sampler)?
            }
        } else {
            let default_sampler = CpuSampler::linear().with_name("__default_linear");
            if let Some(s) = textures.get_sampler("__default_linear") {
                Arc::clone(s)
            } else {
                textures.create_sampler(&default_sampler)?
            }
        };

        Ok((texture, sampler))
    }
}

/// Pack uniform (Float/Vec3/Vec4) values from a [`CpuMaterial`]'s binding
/// definitions into a contiguous byte vector suitable for GPU upload.
pub fn pack_uniform_bytes(
    cpu_mat: &redlilium_core::material::CpuMaterial,
    values: &[MaterialValue],
) -> Vec<u8> {
    let mut data = Vec::new();
    for (i, value) in values.iter().enumerate() {
        if i >= cpu_mat.bindings.len() {
            break;
        }
        if cpu_mat.bindings[i].value_type == MaterialValueType::Texture {
            continue;
        }
        match value {
            MaterialValue::Float(v) => {
                data.extend_from_slice(&v.to_le_bytes());
            }
            MaterialValue::Vec3(v) => {
                for f in v {
                    data.extend_from_slice(&f.to_le_bytes());
                }
            }
            MaterialValue::Vec4(v) => {
                for f in v {
                    data.extend_from_slice(&f.to_le_bytes());
                }
            }
            MaterialValue::Texture(_) => {}
        }
    }
    data
}
