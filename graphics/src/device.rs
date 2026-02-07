//! Graphics device.
//!
//! The [`GraphicsDevice`] is the main interface for creating GPU resources.
//! It is created by [`GraphicsInstance::create_device`].

use std::sync::{Arc, RwLock, Weak};

use crate::error::GraphicsError;
use crate::instance::GraphicsInstance;
use crate::materials::{Material, MaterialDescriptor};
use crate::mesh::{CpuMesh, Mesh, MeshDescriptor};
use crate::pipeline::FramePipeline;
use crate::resources::{Buffer, Sampler, Texture};
use crate::types::{
    BufferDescriptor, BufferUsage, CpuSampler, CpuTexture, Extent3d, SamplerDescriptor,
    TextureDescriptor, TextureUsage,
};
use redlilium_core::profiling::profile_scope;

/// Capabilities of a graphics device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceCapabilities {
    /// Maximum texture dimension.
    pub max_texture_dimension: u32,
    /// Maximum buffer size.
    pub max_buffer_size: u64,
    /// Whether compute shaders are supported.
    pub compute_shaders: bool,
    /// Whether ray tracing is supported.
    pub ray_tracing: bool,
    /// Whether mesh shaders are supported.
    pub mesh_shaders: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            max_texture_dimension: 16384,
            max_buffer_size: 1 << 30, // 1 GB
            compute_shaders: true,
            ray_tracing: false,
            mesh_shaders: false,
        }
    }
}

/// A graphics device for creating GPU resources.
///
/// The device is created by [`GraphicsInstance::create_device`] and provides
/// methods for creating buffers, textures, and samplers.
///
/// # Thread Safety
///
/// `GraphicsDevice` is `Send + Sync` and can be safely shared across threads.
/// All resource creation methods use interior mutability where needed.
///
/// # Example
///
/// ```ignore
/// let instance = GraphicsInstance::new()?;
/// let device = instance.create_device()?;
///
/// let buffer = device.create_buffer(&BufferDescriptor::new(1024, BufferUsage::VERTEX))?;
/// let texture = device.create_texture(&TextureDescriptor::new_2d(
///     1920, 1080,
///     TextureFormat::Rgba8Unorm,
///     TextureUsage::RENDER_ATTACHMENT,
/// ))?;
/// ```
pub struct GraphicsDevice {
    instance: Arc<GraphicsInstance>,
    name: String,
    capabilities: DeviceCapabilities,
    // Track allocated resources (weak references for cleanup/debugging)
    buffers: RwLock<Vec<Weak<Buffer>>>,
    textures: RwLock<Vec<Weak<Texture>>>,
    samplers: RwLock<Vec<Weak<Sampler>>>,
    materials: RwLock<Vec<Weak<Material>>>,
    meshes: RwLock<Vec<Weak<Mesh>>>,
}

impl GraphicsDevice {
    /// Create a new graphics device (called by GraphicsInstance).
    pub(crate) fn new(instance: Arc<GraphicsInstance>, name: String) -> Self {
        Self {
            instance,
            name,
            capabilities: DeviceCapabilities::default(),
            buffers: RwLock::new(Vec::new()),
            textures: RwLock::new(Vec::new()),
            samplers: RwLock::new(Vec::new()),
            materials: RwLock::new(Vec::new()),
            meshes: RwLock::new(Vec::new()),
        }
    }

    /// Get the parent instance.
    pub fn instance(&self) -> &Arc<GraphicsInstance> {
        &self.instance
    }

    /// Get the device name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the device capabilities.
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    /// Create a GPU buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer size exceeds device limits or allocation fails.
    pub fn create_buffer(
        self: &Arc<Self>,
        descriptor: &BufferDescriptor,
    ) -> Result<Arc<Buffer>, GraphicsError> {
        profile_scope!("create_buffer");

        // Validate
        if descriptor.size > self.capabilities.max_buffer_size {
            return Err(GraphicsError::InvalidParameter(format!(
                "buffer size {} exceeds maximum {}",
                descriptor.size, self.capabilities.max_buffer_size
            )));
        }

        if descriptor.size == 0 {
            return Err(GraphicsError::InvalidParameter(
                "buffer size cannot be zero".to_string(),
            ));
        }

        // Create the GPU buffer via backend
        let gpu_handle = self.instance.backend().create_buffer(descriptor)?;

        // Create the buffer
        let buffer = Arc::new(Buffer::new(
            Arc::clone(self),
            descriptor.clone(),
            gpu_handle,
        ));

        // Track it
        if let Ok(mut buffers) = self.buffers.write() {
            buffers.push(Arc::downgrade(&buffer));
        }

        Ok(buffer)
    }

    /// Create a GPU texture.
    ///
    /// # Errors
    ///
    /// Returns an error if the texture dimensions exceed device limits or allocation fails.
    pub fn create_texture(
        self: &Arc<Self>,
        descriptor: &TextureDescriptor,
    ) -> Result<Arc<Texture>, GraphicsError> {
        profile_scope!("create_texture");

        // Validate
        let max_dim = self.capabilities.max_texture_dimension;
        if descriptor.size.width > max_dim
            || descriptor.size.height > max_dim
            || descriptor.size.depth > max_dim
        {
            return Err(GraphicsError::InvalidParameter(format!(
                "texture dimension exceeds maximum {max_dim}"
            )));
        }

        if descriptor.size.width == 0 || descriptor.size.height == 0 {
            return Err(GraphicsError::InvalidParameter(
                "texture dimensions cannot be zero".to_string(),
            ));
        }

        // Create the GPU texture via backend
        let gpu_handle = self.instance.backend().create_texture(descriptor)?;

        // Create the texture
        let texture = Arc::new(Texture::new(
            Arc::clone(self),
            descriptor.clone(),
            gpu_handle,
        ));

        // Track it
        if let Ok(mut textures) = self.textures.write() {
            textures.push(Arc::downgrade(&texture));
        }

        Ok(texture)
    }

    /// Create a GPU texture from a CPU-side texture.
    ///
    /// This is a convenience method that:
    /// 1. Creates a GPU texture with `TEXTURE_BINDING | COPY_DST` usage
    /// 2. Writes the pixel data from the `CpuTexture` into it
    ///
    /// For textures that need custom usage flags, mip levels, or multisampling,
    /// use [`create_texture`] with a [`TextureDescriptor`] instead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use redlilium_core::texture::{CpuTexture, TextureFormat};
    ///
    /// let cpu_tex = CpuTexture::new(256, 256, TextureFormat::Rgba8Unorm, pixel_data);
    /// let gpu_tex = device.create_texture_from_cpu(&cpu_tex)?;
    /// ```
    pub fn create_texture_from_cpu(
        self: &Arc<Self>,
        cpu_texture: &CpuTexture,
    ) -> Result<Arc<Texture>, GraphicsError> {
        profile_scope!("create_texture_from_cpu");

        let descriptor = TextureDescriptor {
            label: cpu_texture.name.clone(),
            size: Extent3d::new_2d(cpu_texture.width, cpu_texture.height),
            mip_level_count: 1,
            sample_count: 1,
            dimension: cpu_texture.dimension,
            format: cpu_texture.format,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        };
        let texture = self.create_texture(&descriptor)?;

        if !cpu_texture.data.is_empty() {
            self.write_texture(&texture, &cpu_texture.data)?;
        }

        Ok(texture)
    }

    /// Create a texture sampler.
    ///
    /// # Errors
    ///
    /// Returns an error if sampler creation fails.
    pub fn create_sampler(
        self: &Arc<Self>,
        descriptor: &SamplerDescriptor,
    ) -> Result<Arc<Sampler>, GraphicsError> {
        profile_scope!("create_sampler");

        // Create the GPU sampler via backend
        let gpu_handle = self.instance.backend().create_sampler(descriptor)?;

        // Create the sampler
        let sampler = Arc::new(Sampler::new(
            Arc::clone(self),
            descriptor.clone(),
            gpu_handle,
        ));

        // Track it
        if let Ok(mut samplers) = self.samplers.write() {
            samplers.push(Arc::downgrade(&sampler));
        }

        Ok(sampler)
    }

    /// Create a GPU sampler from a CPU-side sampler.
    ///
    /// This is a convenience method that converts a [`CpuSampler`] to a
    /// [`SamplerDescriptor`] and creates the GPU sampler.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use redlilium_core::sampler::CpuSampler;
    ///
    /// let cpu_sampler = CpuSampler::linear().with_name("my_sampler");
    /// let gpu_sampler = device.create_sampler_from_cpu(&cpu_sampler)?;
    /// ```
    pub fn create_sampler_from_cpu(
        self: &Arc<Self>,
        cpu_sampler: &CpuSampler,
    ) -> Result<Arc<Sampler>, GraphicsError> {
        let descriptor = SamplerDescriptor {
            label: cpu_sampler.name.clone(),
            address_mode_u: cpu_sampler.address_mode_u,
            address_mode_v: cpu_sampler.address_mode_v,
            address_mode_w: cpu_sampler.address_mode_w,
            mag_filter: cpu_sampler.mag_filter,
            min_filter: cpu_sampler.min_filter,
            mipmap_filter: cpu_sampler.mipmap_filter,
            lod_min_clamp: cpu_sampler.lod_min_clamp,
            lod_max_clamp: cpu_sampler.lod_max_clamp,
            compare: cpu_sampler.compare,
            anisotropy_clamp: cpu_sampler.anisotropy_clamp,
        };
        self.create_sampler(&descriptor)
    }

    /// Create a material.
    ///
    /// # Errors
    ///
    /// Returns an error if material creation fails.
    pub fn create_material(
        self: &Arc<Self>,
        descriptor: &MaterialDescriptor,
    ) -> Result<Arc<Material>, GraphicsError> {
        profile_scope!("create_material");

        // Create the material
        let material = Arc::new(Material::new(Arc::clone(self), descriptor.clone()));

        // Track it
        if let Ok(mut materials) = self.materials.write() {
            materials.push(Arc::downgrade(&material));
        }

        Ok(material)
    }

    /// Create a mesh with vertex and optional index buffers.
    ///
    /// Creates one vertex buffer per buffer slot defined in the layout.
    /// For animated meshes, this allows separating static and dynamic data.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer creation fails or the layout has no buffers.
    ///
    /// # Example - Single Buffer
    ///
    /// ```ignore
    /// let layout = VertexLayout::position_normal_uv();
    /// let mesh = device.create_mesh(&MeshDescriptor::new(layout)
    ///     .with_vertex_count(24)
    ///     .with_indices(IndexFormat::Uint16, 36)
    ///     .with_label("cube"))?;
    /// ```
    ///
    /// # Example - Multiple Buffers (Animated)
    ///
    /// ```ignore
    /// let layout = VertexLayout::animated_dynamic(); // 2 buffers
    /// let mesh = device.create_mesh(&MeshDescriptor::new(layout)
    ///     .with_vertex_count(1000)
    ///     .with_label("character"))?;
    ///
    /// // Buffer 0: static data (texcoords)
    /// // Buffer 1: dynamic data (positions/normals) - update each frame
    /// ```
    pub fn create_mesh(
        self: &Arc<Self>,
        descriptor: &MeshDescriptor,
    ) -> Result<Arc<Mesh>, GraphicsError> {
        profile_scope!("create_mesh");

        // Validate
        if descriptor.vertex_count == 0 {
            return Err(GraphicsError::InvalidParameter(
                "mesh must have at least one vertex".to_string(),
            ));
        }

        let buffer_count = descriptor.buffer_count();
        if buffer_count == 0 {
            return Err(GraphicsError::InvalidParameter(
                "mesh layout must have at least one buffer".to_string(),
            ));
        }

        // Validate the layout
        if let Err(e) = descriptor.layout.validate() {
            return Err(GraphicsError::InvalidParameter(format!(
                "invalid vertex layout: {e}"
            )));
        }

        let mesh_label = descriptor.label.as_deref().unwrap_or("mesh");

        // Create vertex buffers (one per layout buffer slot)
        let mut vertex_buffers = Vec::with_capacity(buffer_count);
        for i in 0..buffer_count {
            let buffer_size = descriptor.vertex_buffer_size(i);
            if buffer_size == 0 {
                return Err(GraphicsError::InvalidParameter(format!(
                    "vertex buffer {i} has zero size (stride may be zero)"
                )));
            }

            let label = if buffer_count == 1 {
                format!("{mesh_label}_vertices")
            } else {
                format!("{mesh_label}_vb{i}")
            };

            let buffer = self.create_buffer(
                &BufferDescriptor::new(buffer_size, BufferUsage::VERTEX | BufferUsage::COPY_DST)
                    .with_label(label),
            )?;
            vertex_buffers.push(buffer);
        }

        // Create index buffer if needed
        let (index_buffer, index_format, index_count) = if descriptor.is_indexed() {
            let index_size = descriptor.index_buffer_size();
            let buffer = self.create_buffer(
                &BufferDescriptor::new(index_size, BufferUsage::INDEX | BufferUsage::COPY_DST)
                    .with_label(format!("{mesh_label}_indices")),
            )?;
            (
                Some(buffer),
                descriptor.index_format,
                descriptor.index_count,
            )
        } else {
            (None, None, 0)
        };

        // Create the mesh
        let mesh = Arc::new(Mesh::new(
            Arc::clone(self),
            descriptor.layout.clone(),
            descriptor.topology,
            vertex_buffers,
            descriptor.vertex_count,
            index_buffer,
            index_format,
            index_count,
            descriptor.label.clone(),
        ));

        // Track it
        if let Ok(mut meshes) = self.meshes.write() {
            meshes.push(Arc::downgrade(&mesh));
        }

        Ok(mesh)
    }

    /// Create a GPU mesh from a CPU mesh, uploading vertex and index data.
    ///
    /// This is a convenience method that:
    /// 1. Creates a GPU mesh with the appropriate buffers
    /// 2. Writes the vertex data from the CpuMesh into each buffer slot
    /// 3. Writes the index data if present
    ///
    /// # Example
    ///
    /// ```ignore
    /// use redlilium_core::mesh::generators;
    ///
    /// let sphere = generators::generate_sphere(1.0, 32, 16);
    /// let gpu_mesh = device.create_mesh_from_cpu(&sphere)?;
    /// ```
    pub fn create_mesh_from_cpu(
        self: &Arc<Self>,
        cpu_mesh: &CpuMesh,
    ) -> Result<Arc<Mesh>, GraphicsError> {
        profile_scope!("create_mesh_from_cpu");

        let descriptor = cpu_mesh.to_descriptor();
        let mesh = self.create_mesh(&descriptor)?;

        // Write vertex data for each buffer slot
        for i in 0..cpu_mesh.buffer_count() {
            if let (Some(gpu_buffer), Some(cpu_data)) =
                (mesh.vertex_buffer(i), cpu_mesh.vertex_buffer_data(i))
                && !cpu_data.is_empty()
            {
                self.write_buffer(gpu_buffer, 0, cpu_data)?;
            }
        }

        // Write index data if present
        if let (Some(gpu_index_buffer), Some(cpu_index_data)) =
            (mesh.index_buffer(), cpu_mesh.index_data())
            && !cpu_index_data.is_empty()
        {
            self.write_buffer(gpu_index_buffer, 0, cpu_index_data)?;
        }

        Ok(mesh)
    }

    /// Create a frame pipeline for managing multiple frames in flight.
    ///
    /// The pipeline coordinates CPU-GPU synchronization and enables frame overlap
    /// for maximum throughput.
    ///
    /// # Arguments
    ///
    /// * `frames_in_flight` - Number of frames that can be in flight simultaneously.
    ///   Typically 2 or 3. Must be at least 1.
    ///
    /// # Panics
    ///
    /// Panics if `frames_in_flight` is 0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut pipeline = device.create_pipeline(2);
    ///
    /// while running {
    ///     let mut schedule = pipeline.begin_frame();
    ///     // ... submit graphs ...
    ///     schedule.present("present", &graph, &[deps]);
    ///     pipeline.end_frame(schedule);
    /// }
    ///
    /// pipeline.wait_idle();
    /// ```
    pub fn create_pipeline(self: &Arc<Self>, frames_in_flight: usize) -> FramePipeline {
        FramePipeline::new(Arc::clone(self), frames_in_flight)
    }

    /// Get the number of live buffers created by this device.
    pub fn buffer_count(&self) -> usize {
        self.buffers
            .read()
            .map(|b| b.iter().filter(|w| w.strong_count() > 0).count())
            .unwrap_or(0)
    }

    /// Get the number of live textures created by this device.
    pub fn texture_count(&self) -> usize {
        self.textures
            .read()
            .map(|t| t.iter().filter(|w| w.strong_count() > 0).count())
            .unwrap_or(0)
    }

    /// Get the number of live samplers created by this device.
    pub fn sampler_count(&self) -> usize {
        self.samplers
            .read()
            .map(|s| s.iter().filter(|w| w.strong_count() > 0).count())
            .unwrap_or(0)
    }

    /// Get the number of live materials created by this device.
    pub fn material_count(&self) -> usize {
        self.materials
            .read()
            .map(|m| m.iter().filter(|w| w.strong_count() > 0).count())
            .unwrap_or(0)
    }

    /// Get the number of live meshes created by this device.
    pub fn mesh_count(&self) -> usize {
        self.meshes
            .read()
            .map(|m| m.iter().filter(|w| w.strong_count() > 0).count())
            .unwrap_or(0)
    }

    /// Write data to a buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to write to
    /// * `offset` - Byte offset into the buffer
    /// * `data` - The data to write
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vertices: &[f32] = &[0.0, 0.0, 1.0, 0.0, 0.5, 1.0];
    /// device.write_buffer(&buffer, 0, bytemuck::cast_slice(vertices))?;
    /// ```
    pub fn write_buffer(
        &self,
        buffer: &Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), crate::error::GraphicsError> {
        self.instance
            .backend()
            .write_buffer(buffer.gpu_handle(), offset, data)
    }

    /// Read data from a buffer.
    ///
    /// This is a blocking operation that waits for the GPU to finish.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to read from
    /// * `offset` - Byte offset into the buffer
    /// * `size` - Number of bytes to read
    ///
    /// # Returns
    ///
    /// A vector containing the read data.
    pub fn read_buffer(&self, buffer: &Buffer, offset: u64, size: u64) -> Vec<u8> {
        self.instance
            .backend()
            .read_buffer(buffer.gpu_handle(), offset, size)
    }

    /// Write data to a texture.
    ///
    /// Uploads pixel data to the texture. The data should be in the format
    /// matching the texture's format.
    ///
    /// # Arguments
    ///
    /// * `texture` - The texture to write to
    /// * `data` - The pixel data to upload
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pixels: Vec<u8> = vec![255, 0, 0, 255, 0, 255, 0, 255, ...]; // RGBA data
    /// device.write_texture(&texture, &pixels)?;
    /// ```
    pub fn write_texture(
        &self,
        texture: &Texture,
        data: &[u8],
    ) -> Result<(), crate::error::GraphicsError> {
        self.instance
            .backend()
            .write_texture(texture.gpu_handle(), data, texture.descriptor())
    }

    /// Clean up dead weak references to released resources.
    pub fn cleanup_dead_resources(&self) {
        if let Ok(mut buffers) = self.buffers.write() {
            buffers.retain(|w| w.strong_count() > 0);
        }
        if let Ok(mut textures) = self.textures.write() {
            textures.retain(|w| w.strong_count() > 0);
        }
        if let Ok(mut samplers) = self.samplers.write() {
            samplers.retain(|w| w.strong_count() > 0);
        }
        if let Ok(mut materials) = self.materials.write() {
            materials.retain(|w| w.strong_count() > 0);
        }
        if let Ok(mut meshes) = self.meshes.write() {
            meshes.retain(|w| w.strong_count() > 0);
        }
    }

    /// Advance the deferred destruction system.
    ///
    /// This should be called after a frame fence has been waited on, indicating
    /// that the GPU has finished with resources from older frames. For the Vulkan
    /// backend, this processes the deferred destruction queue and destroys resources
    /// that are no longer in use.
    ///
    /// # Safety Note
    ///
    /// This method is safe to call - it internally handles the unsafe aspects
    /// of GPU resource destruction. However, it should only be called after
    /// waiting on a frame fence to ensure proper synchronization.
    pub(crate) fn advance_deferred_destruction(&self) {
        use crate::backend::GpuBackend;

        // Only Vulkan backend needs deferred destruction
        #[cfg(feature = "vulkan-backend")]
        if let GpuBackend::Vulkan(vulkan_backend) = &*self.instance.backend() {
            // SAFETY: This is called after waiting on a frame fence, which guarantees
            // the GPU has finished with resources from MAX_FRAMES_IN_FLIGHT frames ago.
            unsafe { vulkan_backend.advance_frame() };
        }

        // Also clean up dead weak references while we're at it
        self.cleanup_dead_resources();
    }
}

impl std::fmt::Debug for GraphicsDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphicsDevice")
            .field("name", &self.name)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

// Ensure GraphicsDevice is Send + Sync
static_assertions::assert_impl_all!(GraphicsDevice: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::ShaderSource;
    use crate::types::{BufferUsage, TextureFormat, TextureUsage};

    fn create_test_device() -> Arc<GraphicsDevice> {
        let instance = GraphicsInstance::new().unwrap();
        instance.create_device().unwrap()
    }

    #[test]
    fn test_device_name() {
        let device = create_test_device();
        assert_eq!(device.name(), "Dummy Adapter");
    }

    #[test]
    fn test_create_buffer() {
        let device = create_test_device();
        let buffer = device
            .create_buffer(&BufferDescriptor::new(1024, BufferUsage::VERTEX))
            .unwrap();
        assert_eq!(buffer.size(), 1024);
        assert_eq!(device.buffer_count(), 1);
    }

    #[test]
    fn test_create_buffer_zero_size() {
        let device = create_test_device();
        let result = device.create_buffer(&BufferDescriptor::new(0, BufferUsage::VERTEX));
        assert!(result.is_err());
    }

    #[test]
    fn test_create_texture() {
        let device = create_test_device();
        let texture = device
            .create_texture(&TextureDescriptor::new_2d(
                512,
                512,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();
        assert_eq!(texture.width(), 512);
        assert_eq!(texture.height(), 512);
        assert_eq!(device.texture_count(), 1);
    }

    #[test]
    fn test_create_texture_zero_size() {
        let device = create_test_device();
        let result = device.create_texture(&TextureDescriptor::new_2d(
            0,
            512,
            TextureFormat::Rgba8Unorm,
            TextureUsage::TEXTURE_BINDING,
        ));
        assert!(result.is_err());
    }

    #[test]
    fn test_create_sampler() {
        let device = create_test_device();
        let sampler = device.create_sampler(&SamplerDescriptor::linear()).unwrap();
        assert!(sampler.label().is_none());
        assert_eq!(device.sampler_count(), 1);
    }

    #[test]
    fn test_resource_cleanup() {
        let device = create_test_device();
        {
            let _buffer = device
                .create_buffer(&BufferDescriptor::new(1024, BufferUsage::VERTEX))
                .unwrap();
            assert_eq!(device.buffer_count(), 1);
        }
        // Buffer dropped
        device.cleanup_dead_resources();
        assert_eq!(device.buffer_count(), 0);
    }

    #[test]
    fn test_create_material() {
        let device = create_test_device();
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::vertex(b"vs".to_vec(), "main"))
                    .with_label("test_material"),
            )
            .unwrap();
        assert_eq!(material.label(), Some("test_material"));
        assert_eq!(device.material_count(), 1);
    }
}
