use std::collections::HashMap;
use std::sync::Arc;

use redlilium_graphics::{
    CpuSampler, CpuTexture, FrameSchedule, GraphicsDevice, GraphicsError, Sampler, Texture,
};

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
