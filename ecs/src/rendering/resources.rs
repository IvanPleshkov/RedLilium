use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use redlilium_graphics::{
    CpuSampler, CpuTexture, FrameSchedule, GraphicsDevice, GraphicsError, Sampler, Texture,
    TextureFormat,
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
