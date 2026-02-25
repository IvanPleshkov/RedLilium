use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use redlilium_core::material::{
    CpuMaterial, CpuMaterialInstance, MaterialValue, MaterialValueType, TextureRef, TextureSource,
};
use redlilium_graphics::{
    BindingGroup, Buffer, BufferDescriptor, BufferUsage, CpuMesh, CpuSampler, CpuTexture,
    FrameSchedule, GraphicsDevice, GraphicsError, Material, MaterialInstance, Mesh, Sampler,
    Texture, TextureFormat,
};

use super::components::{MaterialBundle, RenderPassType};

/// Errors that can occur in [`TextureManager`] operations.
#[derive(Debug)]
pub enum TextureManagerError {
    /// File I/O error (e.g., file not found).
    Io(std::io::Error),
    /// Image decoding error.
    ImageDecode(String),
    /// GPU resource creation error.
    Graphics(GraphicsError),
}

impl fmt::Display for TextureManagerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::ImageDecode(msg) => write!(f, "image decode error: {msg}"),
            Self::Graphics(err) => write!(f, "graphics error: {err}"),
        }
    }
}

impl std::error::Error for TextureManagerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::ImageDecode(_) => None,
            Self::Graphics(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for TextureManagerError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<image::ImageError> for TextureManagerError {
    fn from(err: image::ImageError) -> Self {
        Self::ImageDecode(err.to_string())
    }
}

impl From<GraphicsError> for TextureManagerError {
    fn from(err: GraphicsError) -> Self {
        Self::Graphics(err)
    }
}

const DEFAULT_WHITE: &str = "__default_white";
const DEFAULT_BLACK: &str = "__default_black";
const DEFAULT_NORMAL: &str = "__default_normal";

/// Resource for managing GPU textures and samplers.
///
/// Holds a reference to the [`GraphicsDevice`] and caches created textures
/// and samplers by name for reuse.
///
/// # Example
///
/// ```ignore
/// let manager = TextureManager::new(device.clone());
/// world.insert_resource(manager);
///
/// // In a system:
/// ctx.lock::<(ResMut<TextureManager>,)>()
///     .execute(|(mut textures,)| {
///         let tex = textures.create_texture(&cpu_texture).unwrap();
///     });
/// ```
pub struct TextureManager {
    device: Arc<GraphicsDevice>,
    textures: HashMap<String, Arc<Texture>>,
    samplers: HashMap<String, Arc<Sampler>>,
}

impl TextureManager {
    /// Create a new texture manager for the given device.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            textures: HashMap::new(),
            samplers: HashMap::new(),
        }
    }

    /// Get the graphics device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    // --- Texture creation & lookup ---

    /// Create a GPU texture from CPU data.
    ///
    /// If the texture has a name, it is cached for future lookups via [`get_texture`](Self::get_texture).
    pub fn create_texture(
        &mut self,
        cpu_texture: &CpuTexture,
    ) -> Result<Arc<Texture>, GraphicsError> {
        let texture = self.device.create_texture_from_cpu(cpu_texture)?;
        if let Some(name) = &cpu_texture.name {
            self.textures.insert(name.clone(), Arc::clone(&texture));
        }
        Ok(texture)
    }

    /// Look up a previously created texture by name.
    pub fn get_texture(&self, name: &str) -> Option<&Arc<Texture>> {
        self.textures.get(name)
    }

    /// Insert a texture into the cache under a given name.
    pub fn insert_texture(&mut self, name: impl Into<String>, texture: Arc<Texture>) {
        self.textures.insert(name.into(), texture);
    }

    /// Remove a texture from the cache by name, returning it if present.
    pub fn remove_texture(&mut self, name: &str) -> Option<Arc<Texture>> {
        self.textures.remove(name)
    }

    // --- Sampler creation & lookup ---

    /// Create a GPU sampler from CPU descriptor.
    ///
    /// If the sampler has a name, it is cached for future lookups via [`get_sampler`](Self::get_sampler).
    pub fn create_sampler(
        &mut self,
        cpu_sampler: &CpuSampler,
    ) -> Result<Arc<Sampler>, GraphicsError> {
        let sampler = self.device.create_sampler_from_cpu(cpu_sampler)?;
        if let Some(name) = &cpu_sampler.name {
            self.samplers.insert(name.clone(), Arc::clone(&sampler));
        }
        Ok(sampler)
    }

    /// Look up a previously created sampler by name.
    pub fn get_sampler(&self, name: &str) -> Option<&Arc<Sampler>> {
        self.samplers.get(name)
    }

    /// Insert a sampler into the cache under a given name.
    pub fn insert_sampler(&mut self, name: impl Into<String>, sampler: Arc<Sampler>) {
        self.samplers.insert(name.into(), sampler);
    }

    /// Remove a sampler from the cache by name, returning it if present.
    pub fn remove_sampler(&mut self, name: &str) -> Option<Arc<Sampler>> {
        self.samplers.remove(name)
    }

    // --- File loading ---

    /// Load a texture from a file path.
    ///
    /// The image is decoded to RGBA8, uploaded to the GPU, and cached using
    /// the file path as the key. Supported formats depend on the `image` crate
    /// features (PNG and JPEG by default).
    ///
    /// If a texture with this path is already cached, the existing one is returned.
    pub fn load_texture(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<Arc<Texture>, TextureManagerError> {
        let path = path.as_ref();
        let path_str = path.to_string_lossy().into_owned();

        if let Some(texture) = self.textures.get(&path_str) {
            return Ok(Arc::clone(texture));
        }

        let bytes = std::fs::read(path)?;
        let img = image::load_from_memory(&bytes)?;
        let rgba = img.to_rgba8();
        let (width, height) = (img.width(), img.height());

        let cpu_texture =
            CpuTexture::new(width, height, TextureFormat::Rgba8Unorm, rgba.into_raw())
                .with_name(path_str);
        let texture = self.create_texture(&cpu_texture)?;
        Ok(texture)
    }

    // --- Iteration ---

    /// Get a reference to all cached textures.
    pub fn textures(&self) -> &HashMap<String, Arc<Texture>> {
        &self.textures
    }

    /// Get a reference to all cached samplers.
    pub fn samplers(&self) -> &HashMap<String, Arc<Sampler>> {
        &self.samplers
    }

    /// Iterate over all cached texture names.
    pub fn texture_names(&self) -> impl Iterator<Item = &str> {
        self.textures.keys().map(|s| s.as_str())
    }

    /// Iterate over all cached sampler names.
    pub fn sampler_names(&self) -> impl Iterator<Item = &str> {
        self.samplers.keys().map(|s| s.as_str())
    }

    /// Returns the number of cached textures.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// Returns the number of cached samplers.
    pub fn sampler_count(&self) -> usize {
        self.samplers.len()
    }

    // --- Reverse lookup ---

    /// Find the registered name for a texture by Arc pointer identity.
    pub fn find_texture_name(&self, texture: &Arc<Texture>) -> Option<&str> {
        self.textures
            .iter()
            .find(|(_, v)| Arc::ptr_eq(v, texture))
            .map(|(k, _)| k.as_str())
    }

    /// Find the registered name for a sampler by Arc pointer identity.
    pub fn find_sampler_name(&self, sampler: &Arc<Sampler>) -> Option<&str> {
        self.samplers
            .iter()
            .find(|(_, v)| Arc::ptr_eq(v, sampler))
            .map(|(k, _)| k.as_str())
    }

    // --- Default textures ---

    /// Get or create a 1x1 white texture `[255, 255, 255, 255]`.
    pub fn white_texture(&mut self) -> Result<Arc<Texture>, GraphicsError> {
        if let Some(tex) = self.textures.get(DEFAULT_WHITE) {
            return Ok(Arc::clone(tex));
        }
        let cpu = CpuTexture::new(1, 1, TextureFormat::Rgba8Unorm, vec![255, 255, 255, 255])
            .with_name(DEFAULT_WHITE);
        self.create_texture(&cpu)
    }

    /// Get or create a 1x1 black texture `[0, 0, 0, 255]`.
    pub fn black_texture(&mut self) -> Result<Arc<Texture>, GraphicsError> {
        if let Some(tex) = self.textures.get(DEFAULT_BLACK) {
            return Ok(Arc::clone(tex));
        }
        let cpu = CpuTexture::new(1, 1, TextureFormat::Rgba8Unorm, vec![0, 0, 0, 255])
            .with_name(DEFAULT_BLACK);
        self.create_texture(&cpu)
    }

    /// Get or create a 1x1 default normal map texture `[128, 128, 255, 255]`.
    pub fn normal_texture(&mut self) -> Result<Arc<Texture>, GraphicsError> {
        if let Some(tex) = self.textures.get(DEFAULT_NORMAL) {
            return Ok(Arc::clone(tex));
        }
        let cpu = CpuTexture::new(1, 1, TextureFormat::Rgba8Unorm, vec![128, 128, 255, 255])
            .with_name(DEFAULT_NORMAL);
        self.create_texture(&cpu)
    }
}

/// Resource for managing GPU meshes by name.
///
/// Holds a reference to the [`GraphicsDevice`] and caches created meshes
/// by name for reuse. Also enables serialization of `Arc<Mesh>` references
/// by mapping between mesh names and GPU mesh handles.
///
/// # Example
///
/// ```ignore
/// let manager = MeshManager::new(device.clone());
/// world.insert_resource(manager);
///
/// // In a system:
/// ctx.lock::<(ResMut<MeshManager>,)>()
///     .execute(|(mut meshes,)| {
///         let mesh = meshes.create_mesh(&cpu_mesh).unwrap();
///     });
/// ```
pub struct MeshManager {
    device: Arc<GraphicsDevice>,
    meshes: HashMap<String, Arc<Mesh>>,
    /// Cached local-space AABBs keyed by mesh name.
    aabbs: HashMap<String, redlilium_core::math::Aabb>,
}

impl MeshManager {
    /// Create a new mesh manager for the given device.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            meshes: HashMap::new(),
            aabbs: HashMap::new(),
        }
    }

    /// Get the graphics device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    // --- Mesh creation & lookup ---

    /// Create a GPU mesh from CPU data.
    ///
    /// If the mesh has a label, it is cached for future lookups via [`get_mesh`](Self::get_mesh).
    pub fn create_mesh(&mut self, cpu_mesh: &CpuMesh) -> Result<Arc<Mesh>, GraphicsError> {
        let aabb = cpu_mesh.compute_aabb();
        let mesh = self.device.create_mesh_from_cpu(cpu_mesh)?;
        if let Some(label) = mesh.label() {
            self.meshes.insert(label.to_owned(), Arc::clone(&mesh));
            if let Some(aabb) = aabb {
                self.aabbs.insert(label.to_owned(), aabb);
            }
        }
        Ok(mesh)
    }

    /// Look up a previously created mesh by name.
    pub fn get_mesh(&self, name: &str) -> Option<&Arc<Mesh>> {
        self.meshes.get(name)
    }

    /// Insert a mesh into the cache under a given name.
    pub fn insert_mesh(&mut self, name: impl Into<String>, mesh: Arc<Mesh>) {
        self.meshes.insert(name.into(), mesh);
    }

    /// Remove a mesh from the cache by name, returning it if present.
    pub fn remove_mesh(&mut self, name: &str) -> Option<Arc<Mesh>> {
        self.meshes.remove(name)
    }

    /// Find the registered name for a mesh by Arc pointer identity.
    pub fn find_name(&self, mesh: &Arc<Mesh>) -> Option<&str> {
        self.meshes
            .iter()
            .find(|(_, v)| Arc::ptr_eq(v, mesh))
            .map(|(k, _)| k.as_str())
    }

    // --- AABB ---

    /// Look up the cached local-space AABB for a mesh by Arc pointer identity.
    pub fn get_aabb_by_mesh(&self, mesh: &Arc<Mesh>) -> Option<redlilium_core::math::Aabb> {
        let name = self.find_name(mesh)?;
        self.aabbs.get(name).copied()
    }

    // --- Iteration ---

    /// Get a reference to all cached meshes.
    pub fn meshes(&self) -> &HashMap<String, Arc<Mesh>> {
        &self.meshes
    }

    /// Iterate over all cached mesh names.
    pub fn mesh_names(&self) -> impl Iterator<Item = &str> {
        self.meshes.keys().map(|s| s.as_str())
    }

    /// Returns the number of cached meshes.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }
}

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
///
/// Stores registered material definitions (`CpuMaterial` + `Material` pairs) by name,
/// and tracks the CPU-side source data for each [`MaterialBundle`] to enable
/// serialization and deserialization.
///
/// # Example
///
/// ```ignore
/// let mut manager = MaterialManager::new(device.clone());
/// manager.register_material("pbr", cpu_mat, gpu_mat);
///
/// let bundle = manager.create_instance(&cpu_instance, &textures)?;
/// ```
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
    ///
    /// Both the CPU declaration (binding layout) and GPU material (pipeline)
    /// are stored together for later instance creation.
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
    ///
    /// Shared binding groups (uniform buffer, textures) are built once from the
    /// CPU instance values, then each pass gets its own [`MaterialInstance`]
    /// referencing a different GPU [`Material`] pipeline but sharing the same
    /// bindings.
    ///
    /// The resulting bundle is tracked for serialization via
    /// [`get_cpu_bundle`](Self::get_cpu_bundle).
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
    ///
    /// This is a convenience method equivalent to calling [`create_bundle`](Self::create_bundle)
    /// with a single `(Forward, material_name)` pair. The material is resolved by
    /// matching the CPU material's Arc pointer or name.
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
    ///
    /// Use this when the bundle was created outside MaterialManager but
    /// you still want serialization support.
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
    pub(super) fn build_binding_group(
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
///
/// Texture bindings are skipped — only scalar/vector values are packed.
/// Returns an empty `Vec` if there are no uniform values to pack.
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

/// Resource wrapping a [`FrameSchedule`] for the current frame.
///
/// The application layer inserts this before running ECS systems and
/// extracts it after, using [`take`](Self::take).
///
/// # Integration flow
///
/// ```ignore
/// // Before ECS systems:
/// let schedule = pipeline.begin_frame();
/// world.insert_resource(RenderSchedule::new(schedule));
///
/// // Run ECS systems (ForwardRenderSystem submits graphs)
/// runner.run(&mut world, &systems);
///
/// // After ECS systems:
/// let mut res = world.get_resource_mut::<RenderSchedule>();
/// let schedule = res.take().unwrap();
/// pipeline.end_frame(schedule);
/// ```
pub struct RenderSchedule {
    schedule: Option<FrameSchedule>,
}

impl RenderSchedule {
    /// Create a new render schedule resource holding the given frame schedule.
    pub fn new(schedule: FrameSchedule) -> Self {
        Self {
            schedule: Some(schedule),
        }
    }

    /// Create an empty render schedule (no active frame).
    pub fn empty() -> Self {
        Self { schedule: None }
    }

    /// Take the frame schedule out, leaving this resource empty.
    pub fn take(&mut self) -> Option<FrameSchedule> {
        self.schedule.take()
    }

    /// Replace the current schedule with a new one.
    pub fn set(&mut self, schedule: FrameSchedule) {
        self.schedule = Some(schedule);
    }

    /// Get a reference to the frame schedule, if present.
    pub fn schedule(&self) -> Option<&FrameSchedule> {
        self.schedule.as_ref()
    }

    /// Get a mutable reference to the frame schedule, if present.
    pub fn schedule_mut(&mut self) -> Option<&mut FrameSchedule> {
        self.schedule.as_mut()
    }

    /// Returns `true` if a frame schedule is currently held.
    pub fn is_active(&self) -> bool {
        self.schedule.is_some()
    }
}
