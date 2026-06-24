//! GPU texture and sampler management.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use redlilium_graphics::{
    CpuSampler, CpuTexture, Extent3d, GraphicsDevice, GraphicsError, RenderGraph, Sampler, Texture,
    TextureDescriptor, TextureFormat, TextureUsage, TransferConfig, TransferOperation,
    TransferPass,
};

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
pub struct TextureManager {
    device: Arc<GraphicsDevice>,
    textures: HashMap<String, Arc<Texture>>,
    samplers: HashMap<String, Arc<Sampler>>,
    /// Pending texture uploads, flushed into the frame graph by
    /// [`flush_uploads`](Self::flush_uploads). Data is uploaded through the
    /// render graph (never synchronously), so a freshly created texture becomes
    /// resident one or more frames later.
    pending_uploads: Vec<TransferOperation>,
}

impl TextureManager {
    /// Create a new texture manager for the given device.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            textures: HashMap::new(),
            samplers: HashMap::new(),
            pending_uploads: Vec::new(),
        }
    }

    /// Flush queued texture uploads into the current frame's render graph.
    ///
    /// Call once per frame while building the frame graph. Adds a transfer pass
    /// that copies staged CPU data into the textures created since the last
    /// flush. Does nothing if there are no pending uploads.
    pub fn flush_uploads(&mut self, graph: &mut RenderGraph) {
        if self.pending_uploads.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.pending_uploads);
        let mut pass = TransferPass::new("texture_uploads".into());
        pass.set_transfer_config(TransferConfig::new().with_operations(ops));
        graph.add_transfer_pass(pass);
    }

    /// Get the graphics device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    // --- Texture creation & lookup ---

    /// Create a GPU texture from CPU data.
    ///
    /// The texture is allocated immediately, but its pixel data is uploaded
    /// **through the frame graph** on the next [`flush_uploads`](Self::flush_uploads)
    /// — it is not written synchronously. The texture is therefore not sampled
    /// correctly until that upload executes (one or more frames later).
    pub fn create_texture(
        &mut self,
        cpu_texture: &CpuTexture,
    ) -> Result<Arc<Texture>, GraphicsError> {
        let descriptor = TextureDescriptor {
            label: cpu_texture.name.clone(),
            size: Extent3d::new_2d(cpu_texture.width, cpu_texture.height),
            mip_level_count: 1,
            sample_count: 1,
            dimension: cpu_texture.dimension,
            format: cpu_texture.format,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        };
        let texture = self.device.create_texture(&descriptor)?;

        if !cpu_texture.data.is_empty() {
            self.pending_uploads
                .push(TransferOperation::upload_texture_data(
                    &self.device,
                    Arc::clone(&texture),
                    &cpu_texture.data,
                )?);
        }

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
