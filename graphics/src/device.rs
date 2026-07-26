//! Graphics device.
//!
//! The [`GraphicsDevice`] is the main interface for creating GPU resources.
//! It is created by [`GraphicsInstance::create_device`].

use std::sync::{Arc, RwLock, Weak};

use crate::error::GraphicsError;
use crate::graph::TransferOperation;
use crate::instance::GraphicsInstance;
use crate::materials::{Material, MaterialDescriptor};
use crate::mesh::{CpuMesh, Mesh, MeshDescriptor};
use crate::pipeline::FramePipeline;
use crate::resources::{
    Blas, BlasDescriptor, Buffer, Sampler, Texture, Tlas, TlasDescriptor, instance_buffer_size,
};
use crate::types::{
    BufferDescriptor, BufferUsage, CompressionFamily, CpuSampler, SamplerDescriptor,
    TextureDescriptor, TextureUsage,
};
use redlilium_core::profiling::profile_scope;

/// Renderer-architecture tier of a device (ADR-027).
///
/// An ordered ladder: each tier includes everything below it. A tier names a
/// renderer architecture — a feature set the renderer builds a *distinct code
/// path* on — not a bag of individual capabilities. Orthogonal facts whose
/// consequence is a clamp or a local fallback (anisotropy, sample counts,
/// wireframe, async compute) live in [`DeviceCapabilities`] instead.
///
/// A new tier is added only together with the render path that stands on it
/// (e.g. bindless materials, ray-traced lighting). Devices below `Baseline`
/// fail device creation with a named-feature error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceTier {
    /// The floor every render path assumes: dynamic rendering, timeline
    /// semaphores, `shaderDrawParameters` (instance-id in shaders).
    Baseline,
}

/// Capabilities of a graphics device (ADR-027).
///
/// Filled from real adapter queries by every backend at device creation —
/// never fabricated — and read-only afterwards. Use-sites clamp or validate
/// against these values instead of hardcoding limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceCapabilities {
    /// Renderer-architecture tier (see [`DeviceTier`]).
    pub tier: DeviceTier,
    /// Maximum texture dimension the backend accepts.
    pub max_texture_dimension: u32,
    /// Maximum buffer size the backend accepts.
    pub max_buffer_size: u64,
    /// Maximum sampler anisotropy; `1` when anisotropic filtering is
    /// unsupported. Sampler creation clamps to this.
    pub max_sampler_anisotropy: u16,
    /// Bitmask of supported MSAA sample counts for render attachments
    /// (bit `n` set means a sample count of `n` is supported, matching
    /// Vulkan's `VkSampleCountFlagBits` encoding — bit 1 = 1x, bit 4 = 4x…).
    pub sample_count_mask: u32,
    /// Whether wireframe rasterization (`PolygonMode::Line`) is available.
    /// When false, pipelines downgrade line mode to fill.
    pub wireframe: bool,
    /// Whether the device has a distinct async compute queue (#47).
    pub async_compute: bool,
    /// Whether the device has a dedicated transfer queue — DMA engines for
    /// host↔device streaming that overlaps both graphics and compute (#89).
    pub transfer_queue: bool,
    /// Whether compute shaders are supported (false on WebGL downlevel).
    pub compute_shaders: bool,
    /// Whether per-pass GPU timestamp queries are available (#95). Requires the
    /// device's graphics queue family to expose a non-zero `timestampValidBits`
    /// (Vulkan). When false, [`GraphicsDevice::latest_gpu_timings`] returns an
    /// empty result and the editor stats panel degrades to "unavailable".
    pub gpu_timestamps: bool,
    /// Whether the backend can generate mip chains on the GPU via the frame
    /// graph (#96, `TransferOperation::GenerateMipmaps` — a `vkCmdBlitImage`
    /// chain). This is the format-independent "backend supports the op" bit;
    /// per-format blit eligibility is a separate query
    /// ([`GraphicsDevice::supports_mipmap_generation`]). `false` on wgpu/dummy
    /// (no blit path), so the texture loader falls back to a single mip.
    pub mip_generation: bool,
    /// Whether `VK_EXT_memory_budget` is enabled on the device (#98). When true,
    /// [`GraphicsDevice::latest_memory_stats`] fills each heap's `budget`/`usage`
    /// from `VkPhysicalDeviceMemoryBudgetPropertiesEXT` (queried once per frame,
    /// never cached across frames — the spec allows the values to change at any
    /// moment). When false, heap `budget`/`usage` are `None` and the stats panel
    /// shows heap sizes + allocator totals only. Always false on wgpu/dummy.
    pub memory_budget: bool,
    /// Whether inline ray tracing (ray queries against acceleration
    /// structures) is available (#110). True only on the Vulkan backend when
    /// `VK_KHR_acceleration_structure` + `VK_KHR_ray_query` (and the
    /// `bufferDeviceAddress` feature they build on) are enabled. This is a
    /// capability, not a [`DeviceTier`]: no renderer code path stands on it
    /// yet (ADR-032) — a future ray-traced-lighting render path introduces
    /// the tier. Gates [`GraphicsDevice::create_blas`] /
    /// [`GraphicsDevice::create_tlas`] and
    /// [`BindingType::AccelerationStructure`](crate::BindingType) bindings.
    /// Always false on wgpu/dummy and on wasm.
    pub ray_query: bool,
    /// Whether mesh-shading pipelines (task + mesh shader stages, #111) are
    /// available. True only on the Vulkan backend when `VK_EXT_mesh_shader`
    /// (with its `taskShader` + `meshShader` feature bits) is enabled. Like
    /// [`ray_query`](Self::ray_query) this is a capability, not a
    /// [`DeviceTier`]: no renderer path stands on it yet. Gates materials
    /// with [`ShaderStage::Task`]/[`ShaderStage::Mesh`](crate::ShaderStage)
    /// stages and [`GraphicsPass::add_draw_mesh_tasks`](crate::GraphicsPass)
    /// draws. Always false on wgpu/dummy and on wasm (WebGPU has no mesh
    /// shaders).
    pub mesh_shading: bool,
    /// Maximum task/mesh work-group count per dimension accepted by a mesh-
    /// tasks draw (`VkPhysicalDeviceMeshShaderPropertiesEXT::
    /// maxTaskWorkGroupCount`, #116). `[0; 3]` when
    /// [`mesh_shading`](Self::mesh_shading) is false. Mesh-tasks draws are
    /// validated per-axis against this at encode time so an over-large group
    /// count cannot reach the driver as undefined behaviour. The indirect
    /// variant cannot be checked (its counts live in a GPU buffer).
    pub mesh_tasks_max_group_count: [u32; 3],
    /// Maximum product of the three task/mesh work-group dimensions
    /// (`maxTaskWorkGroupTotalCount`, #116). `0` when mesh shading is false.
    /// Validated alongside [`mesh_tasks_max_group_count`](Self::mesh_tasks_max_group_count).
    pub mesh_tasks_max_total_count: u32,
    /// Whether BC1–BC7 (DXT/S3TC) block-compressed textures can be created
    /// (#119). Vulkan `textureCompressionBC` / wgpu `TEXTURE_COMPRESSION_BC`.
    /// The desktop family; compressed textures are sample-only (no
    /// `RENDER_ATTACHMENT`/`STORAGE_BINDING` usage).
    pub texture_compression_bc: bool,
    /// Whether ETC2 + EAC block-compressed textures can be created (#119).
    /// Vulkan `textureCompressionETC2` / wgpu `TEXTURE_COMPRESSION_ETC2`.
    /// The mobile/GLES family; one bit covers ETC2 and EAC together.
    pub texture_compression_etc2: bool,
    /// Whether ASTC (LDR profile) block-compressed textures can be created
    /// (#119). Vulkan `textureCompressionASTC_LDR` / wgpu
    /// `TEXTURE_COMPRESSION_ASTC`. Covers every block size, Unorm + sRGB.
    pub texture_compression_astc: bool,
    /// Whether the bindless texture/sampler heap (#117) is available:
    /// update-after-bind, partially-bound, runtime-sized descriptor arrays
    /// indexed non-uniformly from shaders. True only on the Vulkan backend
    /// when the five Vulkan 1.2 descriptor-indexing feature bits are present
    /// (core since 1.2; the engine baseline is 1.3 — in practice every
    /// desktop GPU the baseline admits). Like [`ray_query`](Self::ray_query)
    /// this is a capability, not a [`DeviceTier`]. Gates the bindless
    /// registration APIs and the `BindlessTextures`/`BindlessSamplers`
    /// binding types. Always false on wgpu/dummy and on wasm.
    pub bindless: bool,
}

impl DeviceCapabilities {
    /// Whether MSAA render attachments with `count` samples are supported.
    pub fn supports_sample_count(&self, count: u32) -> bool {
        count.is_power_of_two() && (self.sample_count_mask & count) != 0
    }

    /// Whether textures of the given block-compression family can be created
    /// (#119). Support is per-family, not per-format — one feature bit gates
    /// the whole family on every backend.
    pub fn supports_compression_family(&self, family: CompressionFamily) -> bool {
        match family {
            CompressionFamily::Bc => self.texture_compression_bc,
            CompressionFamily::Etc2 => self.texture_compression_etc2,
            CompressionFamily::Astc => self.texture_compression_astc,
        }
    }
}

/// Per-pass GPU timings for one submitted queue (#95).
///
/// Durations only — begin/end deltas resolved *within a single queue*. Cross-
/// queue timelines are not built (timestamps from different queues are not
/// comparable without `VK_EXT_calibrated_timestamps`, which is out of scope).
#[derive(Debug, Clone, PartialEq)]
pub struct SubmitTiming {
    /// Which queue this submit ran on.
    pub queue: crate::graph::QueuePreference,
    /// Wall-clock GPU time for the whole submit (begin→end), in milliseconds.
    pub total_ms: f32,
    /// Per-pass GPU time in milliseconds, keyed by the graph's pass label.
    pub passes: Vec<(String, f32)>,
}

/// GPU timings for the most recently *retired* frame slot (#95).
///
/// Because timestamp results are read back only after a slot's fence has
/// signaled, these values are always ~`MAX_FRAMES_IN_FLIGHT` frames stale —
/// which is fine for a stats panel. Empty on backends without
/// [`DeviceCapabilities::gpu_timestamps`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrameGpuTimings {
    /// One entry per submit recorded during the retired frame.
    pub submits: Vec<SubmitTiming>,
}

impl FrameGpuTimings {
    /// Sum of every submit's `total_ms` — a coarse whole-frame GPU figure.
    /// Note this over-counts when submits on different queues overlap in time.
    pub fn frame_total_ms(&self) -> f32 {
        self.submits.iter().map(|s| s.total_ms).sum()
    }

    /// Whether any timing data is present.
    pub fn is_empty(&self) -> bool {
        self.submits.is_empty()
    }
}

/// One Vulkan memory heap's driver-level figures (#98).
///
/// `budget`/`usage` come from `VK_EXT_memory_budget` and are `None` when the
/// extension is unavailable ([`DeviceCapabilities::memory_budget`]). Usage
/// reflects the *whole process* (and, for budget, system-wide pressure) — it
/// shows headroom, not just this engine's footprint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeapStats {
    /// Vulkan memory-heap index.
    pub index: u32,
    /// Whether this heap is `DEVICE_LOCAL` (VRAM on discrete GPUs).
    pub device_local: bool,
    /// Total heap size in bytes (`VkMemoryHeap::size`).
    pub size: u64,
    /// Driver-reported budget for this heap in bytes — how much the process may
    /// use before eviction pressure. `None` without `VK_EXT_memory_budget`.
    pub budget: Option<u64>,
    /// Driver-reported current usage of this heap in bytes (process-wide).
    /// `None` without `VK_EXT_memory_budget`.
    pub usage: Option<u64>,
}

/// Live GPU-memory counts for this device ([`GraphicsDevice`] resource trackers).
///
/// These are the engine's own live-resource tallies (not byte figures), shown
/// next to the allocator totals so the panel is meaningful even on backends
/// without heap/budget data.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceCounts {
    pub buffers: usize,
    pub textures: usize,
    pub samplers: usize,
    pub materials: usize,
    pub meshes: usize,
}

/// GPU-memory statistics sampled once per frame (#98).
///
/// Two data sources: driver-level heap budgets (`VK_EXT_memory_budget`, in
/// [`Self::heaps`]) and the engine's `gpu-allocator` totals
/// ([`Self::allocator_allocated`] / [`Self::allocator_reserved`] /
/// [`Self::allocation_count`]), plus the engine's live-resource counts
/// ([`Self::resources`]). Empty/zeroed on backends without data (dummy); the
/// wgpu backend leaves [`Self::heaps`] empty but the resource counts are still
/// filled by [`GraphicsDevice::latest_memory_stats`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuMemoryStats {
    /// Per-heap driver figures, `DEVICE_LOCAL` heaps not guaranteed first
    /// (the UI sorts). Empty without a Vulkan backend.
    pub heaps: Vec<HeapStats>,
    /// Bytes currently sub-allocated to live resources by our allocator
    /// (`AllocatorReport::total_allocated_bytes`).
    pub allocator_allocated: u64,
    /// Bytes reserved in the allocator's memory blocks, including unused slack
    /// (`AllocatorReport::total_capacity_bytes`).
    pub allocator_reserved: u64,
    /// Number of live allocations tracked by the allocator.
    pub allocation_count: usize,
    /// Engine live-resource counts, filled at the device layer regardless of
    /// backend.
    pub resources: ResourceCounts,
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
    /// The bindless heap (#117): slot table + the shared binding group,
    /// created lazily on first use (group construction needs `Arc<Self>`).
    /// `None` until then; never populated when `capabilities.bindless` is
    /// false.
    bindless: parking_lot::Mutex<Option<BindlessShared>>,
    /// Transparent BLAS compaction driver (#110 phase 3): tracks compactable
    /// BLASes and drives their swap/retirement each frame.
    as_compaction: parking_lot::Mutex<crate::as_compaction::AsCompactionDriver>,
}

/// The lazily created bindless heap (#117): the slot table and the one
/// device-owned binding group every bindless material binds.
#[derive(Clone)]
struct BindlessShared {
    slots: Arc<crate::bindless::BindlessSlots>,
    layout: Arc<crate::materials::BindingLayout>,
    group: Arc<crate::materials::BindingGroup>,
}

impl GraphicsDevice {
    /// Create a new graphics device (called by GraphicsInstance).
    pub(crate) fn new(instance: Arc<GraphicsInstance>, name: String) -> Self {
        let capabilities = instance.backend().capabilities();
        Self {
            instance,
            name,
            capabilities,
            buffers: RwLock::new(Vec::new()),
            textures: RwLock::new(Vec::new()),
            samplers: RwLock::new(Vec::new()),
            materials: RwLock::new(Vec::new()),
            meshes: RwLock::new(Vec::new()),
            bindless: parking_lot::Mutex::new(None),
            as_compaction: parking_lot::Mutex::new(Default::default()),
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

    /// GPU timings for the most recently retired frame slot (#95).
    ///
    /// Values are always ~`MAX_FRAMES_IN_FLIGHT` frames stale (results are read
    /// back only once a slot's fence has signaled) — fine for a stats panel.
    /// Returns an empty [`FrameGpuTimings`] when
    /// [`DeviceCapabilities::gpu_timestamps`] is false.
    pub fn latest_gpu_timings(&self) -> FrameGpuTimings {
        self.instance.backend().latest_gpu_timings()
    }

    /// GPU-memory statistics for the current frame (#98).
    ///
    /// Heap budgets and allocator totals come from the backend (sampled once
    /// per frame on the render thread in `advance_frame` — never queried from
    /// this call, which may run on the UI thread). The engine's live-resource
    /// counts are filled here from this device's trackers, so the result is
    /// meaningful even on backends without heap/budget data (wgpu) or with none
    /// at all (dummy → empty heaps, zero allocator totals).
    pub fn latest_memory_stats(&self) -> GpuMemoryStats {
        let mut stats = self.instance.backend().latest_memory_stats();
        stats.resources = ResourceCounts {
            buffers: self.buffer_count(),
            textures: self.texture_count(),
            samplers: self.sampler_count(),
            materials: self.material_count(),
            meshes: self.mesh_count(),
        };
        stats
    }

    /// Whether the GPU can generate a mip chain for `format` (#96).
    ///
    /// True only when the backend supports the op
    /// ([`DeviceCapabilities::mip_generation`]) AND the format is blit-eligible
    /// (`BLIT_SRC | BLIT_DST | SAMPLED_IMAGE_FILTER_LINEAR` on Vulkan).
    /// Block-compressed and other non-blittable formats return false; the
    /// texture loader then keeps a single mip. Always false on wgpu/dummy.
    pub fn supports_mipmap_generation(&self, format: crate::types::TextureFormat) -> bool {
        self.capabilities.mip_generation && self.instance.backend().supports_blit_mipgen(format)
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

        if descriptor.sample_count > 1
            && !self
                .capabilities
                .supports_sample_count(descriptor.sample_count)
        {
            return Err(GraphicsError::InvalidParameter(format!(
                "sample count {} not supported by this device (supported mask: {:#x})",
                descriptor.sample_count, self.capabilities.sample_count_mask
            )));
        }

        // Block-compressed formats (#119): sample-only on every backend, and
        // gated per-family by the device feature bit. The usage check comes
        // first so it fails the same way on every device. NPOT sizes are fine —
        // partial edge blocks are legal, and the transfer path already sizes
        // uploads in blocks.
        if let Some(family) = descriptor.format.compression_family() {
            if descriptor
                .usage
                .intersects(TextureUsage::RENDER_ATTACHMENT | TextureUsage::STORAGE_BINDING)
            {
                return Err(GraphicsError::InvalidParameter(format!(
                    "compressed texture format {:?} is sample-only; \
                     RENDER_ATTACHMENT/STORAGE_BINDING usage is not supported",
                    descriptor.format
                )));
            }
            if !self.capabilities.supports_compression_family(family) {
                return Err(GraphicsError::FeatureNotSupported(format!(
                    "texture format {:?} requires {family} texture compression, \
                     which this device does not support",
                    descriptor.format
                )));
            }
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

    /// Create a bottom-level acceleration structure over triangle geometry
    /// (#110, ADR-032).
    ///
    /// Allocates the AS backing storage and build scratch (engine [`Buffer`]s,
    /// tracked by the automatic barrier system) and creates the GPU handle.
    /// The BLAS holds no geometry yet: add it to an
    /// [`AccelerationStructureBuildPass`](crate::AccelerationStructureBuildPass)
    /// to build it on the GPU — there is deliberately no direct build method
    /// (GPU work goes through the frame graph).
    ///
    /// Requires [`DeviceCapabilities::ray_query`]; geometry buffers must carry
    /// [`BufferUsage::ACCELERATION_STRUCTURE_INPUT`](crate::types::BufferUsage).
    ///
    /// # Errors
    ///
    /// Fails on backends without ray-query support (wgpu; Vulkan devices
    /// lacking the extensions), on an empty descriptor, or when a geometry
    /// buffer lacks the AS-input usage flag.
    pub fn create_blas(
        self: &Arc<Self>,
        descriptor: &BlasDescriptor,
    ) -> Result<Arc<Blas>, GraphicsError> {
        profile_scope!("create_blas");

        if descriptor.geometries.is_empty() {
            return Err(GraphicsError::InvalidParameter(
                "BLAS descriptor has no geometry".to_string(),
            ));
        }
        for geometry in &descriptor.geometries {
            let input_ok = |buffer: &Buffer| {
                buffer
                    .descriptor()
                    .usage
                    .contains(crate::types::BufferUsage::ACCELERATION_STRUCTURE_INPUT)
            };
            if !input_ok(&geometry.vertex_buffer)
                || geometry
                    .index_buffer
                    .as_deref()
                    .is_some_and(|b| !input_ok(b))
            {
                return Err(GraphicsError::InvalidParameter(format!(
                    "BLAS '{}' geometry buffers must be created with \
                     BufferUsage::ACCELERATION_STRUCTURE_INPUT",
                    descriptor.label.as_deref().unwrap_or("unnamed")
                )));
            }
        }

        let backend = self.instance.backend();
        let sizes = backend.blas_build_sizes(descriptor)?;
        let label = descriptor.label.as_deref().unwrap_or("blas");
        let backing = self.create_buffer(
            &BufferDescriptor::new(
                sizes.acceleration_structure_size,
                BufferUsage::ACCELERATION_STRUCTURE_STORAGE,
            )
            .with_label(format!("{label} storage")),
        )?;
        let scratch = self.create_buffer(
            &BufferDescriptor::new(
                sizes.build_scratch_size,
                BufferUsage::STORAGE | BufferUsage::DEVICE_ADDRESS,
            )
            .with_label(format!("{label} scratch")),
        )?;
        let gpu_handle =
            backend.create_blas_handle(backing.gpu_handle(), sizes.acceleration_structure_size)?;

        let blas = Arc::new(Blas::new(
            Arc::clone(self),
            descriptor.clone(),
            backing,
            scratch,
            gpu_handle,
        ));
        // Register for transparent compaction (#110 phase 3) so the per-frame
        // maintenance flush finds it once its compacted size comes back.
        if descriptor.allow_compaction {
            self.as_compaction.lock().register(&blas);
        }
        Ok(blas)
    }

    /// Create a top-level acceleration structure over BLAS instances (#110,
    /// ADR-032).
    ///
    /// Allocates backing storage, build scratch, and one host-visible
    /// instance buffer per frame slot. Per frame: write the instance array
    /// with [`Tlas::write_instances`], then add the TLAS to that frame's
    /// [`AccelerationStructureBuildPass`](crate::AccelerationStructureBuildPass).
    /// Shaders bind the built TLAS through
    /// [`BindingType::AccelerationStructure`](crate::BindingType).
    ///
    /// # Errors
    ///
    /// Fails on backends without ray-query support or when `max_instances`
    /// is zero.
    pub fn create_tlas(
        self: &Arc<Self>,
        descriptor: &TlasDescriptor,
    ) -> Result<Arc<Tlas>, GraphicsError> {
        profile_scope!("create_tlas");

        if descriptor.max_instances == 0 {
            return Err(GraphicsError::InvalidParameter(
                "TLAS max_instances cannot be zero".to_string(),
            ));
        }

        let backend = self.instance.backend();
        let sizes = backend.tlas_build_sizes(descriptor.max_instances)?;
        let label = descriptor.label.as_deref().unwrap_or("tlas");
        let backing = self.create_buffer(
            &BufferDescriptor::new(
                sizes.acceleration_structure_size,
                BufferUsage::ACCELERATION_STRUCTURE_STORAGE,
            )
            .with_label(format!("{label} storage")),
        )?;
        let scratch = self.create_buffer(
            &BufferDescriptor::new(
                sizes.build_scratch_size,
                BufferUsage::STORAGE | BufferUsage::DEVICE_ADDRESS,
            )
            .with_label(format!("{label} scratch")),
        )?;
        // One instance buffer per frame slot: `write_instances` rotates, so
        // the CPU never touches data an in-flight frame's build still reads.
        let instance_buffers = (0..crate::pipeline::MAX_FRAMES_IN_FLIGHT)
            .map(|slot| {
                self.create_buffer(
                    &BufferDescriptor::new(
                        instance_buffer_size(descriptor.max_instances),
                        BufferUsage::ACCELERATION_STRUCTURE_INPUT | BufferUsage::MAP_WRITE,
                    )
                    .with_label(format!("{label} instances[{slot}]")),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let gpu_handle =
            backend.create_tlas_handle(backing.gpu_handle(), sizes.acceleration_structure_size)?;

        Ok(Arc::new(Tlas::new(
            Arc::clone(self),
            descriptor.clone(),
            backing,
            scratch,
            instance_buffers,
            gpu_handle,
        )))
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

    /// Apply reflected binding layouts to a material descriptor.
    ///
    /// Reclassifies a whole set's uniform buffers as
    /// [`DynamicUniformBuffer`](crate::materials::BindingType::DynamicUniformBuffer)
    /// when its update rate is `Dynamic`/`External` (rate-classified sets bind a
    /// ring buffer once with a per-draw offset; `docs/MATERIAL_ASSETS.md`
    /// Decision 7), honors the legacy explicit `dynamic_uniforms`, and stores the
    /// layouts + update rates. Shared by the native (live Slang reflection) and
    /// wasm (offline baked reflection) paths so the reclassification rules cannot
    /// drift between them (#33).
    fn apply_reflected_layouts(
        desc: &mut MaterialDescriptor,
        mut layouts: Vec<crate::materials::BindingLayout>,
        update_rates: Vec<Option<crate::materials::UpdateRate>>,
    ) {
        use crate::materials::{BindingType, UpdateRate};
        for (layout, rate) in layouts.iter_mut().zip(&update_rates) {
            if matches!(
                *rate,
                Some(UpdateRate::Dynamic) | Some(UpdateRate::External)
            ) {
                for entry in &mut layout.entries {
                    if entry.binding_type == BindingType::UniformBuffer {
                        entry.binding_type = BindingType::DynamicUniformBuffer;
                    }
                }
            }
        }
        for &(group, binding) in &desc.dynamic_uniforms {
            if let Some(layout) = layouts.get_mut(group as usize) {
                for entry in &mut layout.entries {
                    if entry.binding == binding && entry.binding_type == BindingType::UniformBuffer
                    {
                        entry.binding_type = BindingType::DynamicUniformBuffer;
                    }
                }
            }
        }
        desc.binding_layouts = layouts.into_iter().map(std::sync::Arc::new).collect();
        desc.set_update_rates = update_rates;
    }

    /// Create a material with its GPU pipeline.
    ///
    /// The pipeline is compiled eagerly from the descriptor's shaders,
    /// vertex layout, topology, color/depth formats, and blend state.
    ///
    /// # Errors
    ///
    /// Returns an error if shader compilation or pipeline creation fails.
    pub fn create_material(
        self: &Arc<Self>,
        descriptor: &MaterialDescriptor,
    ) -> Result<Arc<Material>, GraphicsError> {
        profile_scope!("create_material");

        // Resolve the shader variant (#6) into per-stage defines first, so
        // reflection, baked lookups and pipeline compilation below all see
        // the same effective defines.
        let descriptor = &descriptor.resolve_variant()?;

        // Stage-combination sanity (#111): catch a task stage without a mesh
        // stage, mesh+vertex mixes, and mesh materials with vertex layouts
        // here, with the material named — not as driver validation errors.
        descriptor.validate_stage_combination()?;

        // Mesh-shading materials require the capability (#111). The dummy
        // backend is exempt — it fabricates no capabilities but must keep
        // descriptor-level tests runnable headless.
        if descriptor.uses_mesh_shading()
            && !self.capabilities.mesh_shading
            && !matches!(
                &*self.instance.backend(),
                crate::backend::GpuBackend::Dummy(_)
            )
        {
            return Err(GraphicsError::FeatureNotSupported(format!(
                "material {:?} uses task/mesh shader stages, which require \
                 DeviceCapabilities::mesh_shading (#111)",
                descriptor.label
            )));
        }

        // Auto-reflect binding layouts from Slang shaders when none are specified
        #[cfg(feature = "slang-shaders")]
        let descriptor = &{
            let mut desc = descriptor.clone();
            if desc.binding_layouts.is_empty()
                && !desc.shaders.is_empty()
                && desc
                    .shaders
                    .iter()
                    .all(|s| s.language == crate::materials::ShaderSourceLanguage::Slang)
            {
                let compiler = crate::shader::SlangCompiler::new()?;
                compiler.write_library_modules(&crate::shader::ShaderLibrary::standard_slang())?;

                // Pre-compute owned data (source strings and define refs)
                let sources: Vec<String> = desc
                    .shaders
                    .iter()
                    .map(|s| String::from_utf8_lossy(&s.source).into_owned())
                    .collect();
                let defines: Vec<Vec<(&str, &str)>> = desc
                    .shaders
                    .iter()
                    .map(|s| {
                        s.defines
                            .iter()
                            .map(|(k, v)| (k.as_str(), v.as_str()))
                            .collect()
                    })
                    .collect();

                let shaders_for_reflect: Vec<crate::shader::ShaderReflectInput<'_>> = desc
                    .shaders
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        (
                            sources[i].as_str(),
                            s.entry_point.as_str(),
                            s.stage,
                            defines[i].as_slice(),
                        )
                    })
                    .collect();

                let (layouts, update_rates) =
                    compiler.reflect_all_bindings(&shaders_for_reflect)?;
                Self::apply_reflected_layouts(&mut desc, layouts, update_rates);
            }
            desc
        };

        // On wasm there is no Slang compiler, so binding layouts come from the
        // offline reflection bake (#33), keyed by the shader SET. A miss is a
        // hard error naming the material — never a silent empty layout (which
        // would panic later at the first `binding_layouts()[0]` and leave the
        // pipeline without bind-group layouts).
        #[cfg(not(feature = "slang-shaders"))]
        let descriptor = &{
            let mut desc = descriptor.clone();
            if desc.binding_layouts.is_empty()
                && !desc.shaders.is_empty()
                && desc
                    .shaders
                    .iter()
                    .all(|s| s.language == crate::materials::ShaderSourceLanguage::Slang)
            {
                let sources: Vec<String> = desc
                    .shaders
                    .iter()
                    .map(|s| String::from_utf8_lossy(&s.source).into_owned())
                    .collect();
                let defines: Vec<Vec<(&str, &str)>> = desc
                    .shaders
                    .iter()
                    .map(|s| {
                        s.defines
                            .iter()
                            .map(|(k, v)| (k.as_str(), v.as_str()))
                            .collect()
                    })
                    .collect();
                let set: Vec<crate::shader::baked::ShaderSetInput<'_>> = desc
                    .shaders
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        (
                            sources[i].as_str(),
                            s.entry_point.as_str(),
                            s.stage,
                            defines[i].as_slice(),
                        )
                    })
                    .collect();

                let key = crate::shader::baked::shader_set_key(&set);
                let (layouts, update_rates) = crate::shader::baked::lookup_reflection(key)
                    .ok_or_else(|| {
                        GraphicsError::ResourceCreationFailed(format!(
                            "no baked binding-layout reflection for material {:?} (entries: {:?}); \
                             add its shader set to the xtask registry and rebake (#33)",
                            desc.label,
                            desc.shaders
                                .iter()
                                .map(|s| s.entry_point.as_str())
                                .collect::<Vec<_>>(),
                        ))
                    })?;
                Self::apply_reflected_layouts(&mut desc, layouts, update_rates);
            }
            desc
        };

        // Create the GPU pipeline via backend
        let gpu_handle = self.instance.backend().create_pipeline(descriptor)?;

        // Create the material
        let material = Arc::new(Material::new(
            Arc::clone(self),
            descriptor.clone(),
            gpu_handle,
        ));

        // Track it
        if let Ok(mut materials) = self.materials.write() {
            materials.push(Arc::downgrade(&material));
        }

        Ok(material)
    }

    /// Create a binding group: the compiled, immutable form of a
    /// [`BindingGroupDescriptor`], created eagerly against `layout` (like
    /// [`create_material`](Self::create_material) compiles a pipeline). The
    /// backend descriptor set / bind group is built and written **once** here;
    /// encoding a draw then only binds it — no per-draw descriptor allocations
    /// or writes.
    ///
    /// # Errors
    ///
    /// Returns an error if an entry's binding is not declared in `layout`, if a
    /// bound resource's kind does not match the layout's declared
    /// [`BindingType`](crate::BindingType), or if backend resource creation
    /// fails.
    pub fn create_binding_group(
        self: &Arc<Self>,
        layout: Arc<crate::materials::BindingLayout>,
        descriptor: crate::materials::BindingGroupDescriptor,
    ) -> Result<Arc<crate::materials::BindingGroup>, GraphicsError> {
        profile_scope!("create_binding_group");

        // The bindless heap group is device-owned (#117): its set lives in a
        // dedicated update-after-bind pool and is written by registration,
        // not here. User-built groups cannot declare its binding types.
        if layout.entries.iter().any(|e| {
            matches!(
                e.binding_type,
                crate::materials::BindingType::BindlessTextures
                    | crate::materials::BindingType::BindlessSamplers
            )
        }) {
            return Err(GraphicsError::InvalidParameter(
                "bindless heap bindings cannot be created via create_binding_group; bind \
                 GraphicsDevice::bindless_heap_group() instead (#117)"
                    .to_string(),
            ));
        }

        // Validate every entry against the layout up front (this replaces the
        // per-draw validation the backends used to do in encode_draw_command).
        for entry in &descriptor.entries {
            let Some(decl) = layout.entries.iter().find(|e| e.binding == entry.binding) else {
                return Err(GraphicsError::InvalidParameter(format!(
                    "binding {} is not declared by the binding layout",
                    entry.binding
                )));
            };
            // AS bindings require the ray-query capability (#110): fail here
            // with the engine-level cause instead of an invalid descriptor
            // type reaching the driver. The dummy backend is exempt — it
            // fabricates no capabilities but must keep graph tests runnable.
            if matches!(
                decl.binding_type,
                crate::materials::BindingType::AccelerationStructure
            ) && !self.capabilities.ray_query
                && !matches!(
                    &*self.instance.backend(),
                    crate::backend::GpuBackend::Dummy(_)
                )
            {
                return Err(GraphicsError::FeatureNotSupported(format!(
                    "binding {} declares an acceleration structure, which requires \
                     DeviceCapabilities::ray_query (#110)",
                    entry.binding
                )));
            }
            if !resource_matches_binding_type(&entry.resource, decl.binding_type) {
                return Err(GraphicsError::InvalidParameter(format!(
                    "binding {} bound resource does not match the layout's declared type {:?}",
                    entry.binding, decl.binding_type
                )));
            }
            // A depth-read-only sampled layout is only meaningful for a depth
            // texture (it records DEPTH_STENCIL_READ_ONLY_OPTIMAL).
            if entry.sampled_depth_layout
                == crate::materials::SampledDepthLayout::DepthStencilReadOnly
            {
                let depth = matches!(
                    &entry.resource,
                    crate::materials::BoundResource::Texture(t)
                        if t.format().is_depth_stencil()
                );
                if !depth {
                    return Err(GraphicsError::InvalidParameter(format!(
                        "binding {} declares DepthStencilReadOnly sampled layout but is not a \
                         depth texture",
                        entry.binding
                    )));
                }
            }
        }

        let gpu_handle = self
            .instance
            .backend()
            .create_binding_group(&layout, &descriptor)?;

        Ok(Arc::new(crate::materials::BindingGroup::new(
            Arc::clone(self),
            layout,
            descriptor,
            gpu_handle,
        )))
    }

    // ======================== Bindless heap (#117) ========================

    /// The heap's slot table + shared group, created on first use.
    fn bindless_shared(self: &Arc<Self>) -> Result<BindlessShared, GraphicsError> {
        use crate::materials::{
            BindingEntry, BindingGroupDescriptor, BindingLayout, BindingLayoutEntry, BindingType,
            BoundResource,
        };

        if !self.capabilities.bindless {
            return Err(GraphicsError::FeatureNotSupported(
                "the bindless heap requires DeviceCapabilities::bindless (#117; Vulkan \
                 backend with the descriptor-indexing feature bits)"
                    .to_string(),
            ));
        }

        let mut guard = self.bindless.lock();
        if let Some(shared) = &*guard {
            return Ok(shared.clone());
        }

        #[cfg(feature = "vulkan-backend")]
        let (gpu_handle, capacities, layout) = {
            let backend = self.instance.backend();
            let crate::backend::GpuBackend::Vulkan(vulkan) = &*backend else {
                return Err(GraphicsError::FeatureNotSupported(
                    "the bindless heap is Vulkan-only (#117)".to_string(),
                ));
            };
            let capacities = vulkan.bindless_capacities();
            // Phase-1 visibility: the classic shader stages. Task/mesh
            // visibility joins once a mesh material consumes the heap
            // (TASK/MESH stage flags require VK_EXT_mesh_shader).
            let visibility = crate::materials::ShaderStageFlags::VERTEX
                | crate::materials::ShaderStageFlags::FRAGMENT
                | crate::materials::ShaderStageFlags::COMPUTE;
            let layout = std::sync::Arc::new(
                BindingLayout::new()
                    .with_entry(
                        BindingLayoutEntry::new(
                            crate::bindless::BINDLESS_TEXTURES_BINDING,
                            BindingType::BindlessTextures,
                        )
                        .with_visibility(visibility),
                    )
                    .with_entry(
                        BindingLayoutEntry::new(
                            crate::bindless::BINDLESS_SAMPLERS_BINDING,
                            BindingType::BindlessSamplers,
                        )
                        .with_visibility(visibility),
                    )
                    .with_label("bindless heap"),
            );
            (vulkan.create_bindless_group(&layout)?, capacities, layout)
        };
        #[cfg(not(feature = "vulkan-backend"))]
        {
            return Err(GraphicsError::FeatureNotSupported(
                "the bindless heap is Vulkan-only (#117)".to_string(),
            ));
        }

        #[cfg(feature = "vulkan-backend")]
        {
            let slots = Arc::new(crate::bindless::BindlessSlots::new(
                capacities.0,
                capacities.1,
            ));
            // The group's entries reference the slot table so pass resource
            // inference can declare the live textures (see
            // `extract_material_resources`).
            let mut descriptor = BindingGroupDescriptor::new();
            descriptor.entries.push(BindingEntry::new(
                crate::bindless::BINDLESS_TEXTURES_BINDING,
                BoundResource::BindlessHeap(Arc::clone(&slots)),
            ));
            descriptor.entries.push(BindingEntry::new(
                crate::bindless::BINDLESS_SAMPLERS_BINDING,
                BoundResource::BindlessHeap(Arc::clone(&slots)),
            ));
            descriptor.label = Some("bindless heap".to_string());

            let group = Arc::new(crate::materials::BindingGroup::new(
                Arc::clone(self),
                Arc::clone(&layout),
                descriptor,
                gpu_handle,
            ));
            let shared = BindlessShared {
                slots,
                layout,
                group,
            };
            *guard = Some(shared.clone());
            Ok(shared)
        }
    }

    /// The device-owned bindless heap group (#117) — the one binding group
    /// every bindless material binds at its heap set. Created on first call.
    ///
    /// Requires [`DeviceCapabilities::bindless`].
    pub fn bindless_heap_group(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::materials::BindingGroup>, GraphicsError> {
        Ok(self.bindless_shared()?.group)
    }

    /// The heap group's binding layout (#117) — pass this `Arc` to
    /// [`MaterialDescriptor::with_binding_layout`](crate::MaterialDescriptor)
    /// for the heap set, so material and group share the layout by pointer.
    pub fn bindless_heap_layout(
        self: &Arc<Self>,
    ) -> Result<Arc<crate::materials::BindingLayout>, GraphicsError> {
        Ok(self.bindless_shared()?.layout)
    }

    /// Register a sampled texture in the bindless heap (#117) and return its
    /// slot index — the value shaders index the heap's texture array with
    /// (`NonUniformResourceIndex`). The heap keeps the texture alive until
    /// [`bindless_unregister_texture`](Self::bindless_unregister_texture).
    ///
    /// # Errors
    ///
    /// Fails without [`DeviceCapabilities::bindless`], when the heap is
    /// full, when the texture lacks `TEXTURE_BINDING` usage, or for
    /// depth-format textures (phase 1 registers color-sampled textures; the
    /// heap records `SHADER_READ_ONLY_OPTIMAL`).
    pub fn bindless_register_texture(
        self: &Arc<Self>,
        texture: &Arc<Texture>,
    ) -> Result<u32, GraphicsError> {
        if !texture
            .usage()
            .contains(crate::types::TextureUsage::TEXTURE_BINDING)
        {
            return Err(GraphicsError::InvalidParameter(format!(
                "bindless textures need TextureUsage::TEXTURE_BINDING (texture {:?})",
                texture.label()
            )));
        }
        if texture.format().is_depth_stencil() {
            return Err(GraphicsError::InvalidParameter(format!(
                "depth-format textures cannot join the bindless heap yet (#117 phase 1; \
                 texture {:?})",
                texture.label()
            )));
        }

        let shared = self.bindless_shared()?;
        let slot = shared.slots.allocate_texture(Arc::clone(texture))?;

        #[cfg(feature = "vulkan-backend")]
        if let crate::backend::GpuBackend::Vulkan(vulkan) = &*self.instance.backend()
            && let Err(e) = vulkan.bindless_write_texture(
                shared.group.gpu_handle(),
                crate::bindless::BINDLESS_TEXTURES_BINDING,
                slot,
                texture.gpu_handle(),
            )
        {
            // Roll the slot back through the deferred path (harmless: the
            // descriptor was never written, the delay only costs reuse).
            let _ = shared.slots.release_texture(slot);
            return Err(e);
        }

        Ok(slot)
    }

    /// Unregister a bindless texture slot (#117). The texture stays alive —
    /// and the slot unreusable — until every frame that might still index it
    /// has passed its fence.
    pub fn bindless_unregister_texture(&self, slot: u32) -> Result<(), GraphicsError> {
        let guard = self.bindless.lock();
        let Some(shared) = &*guard else {
            return Err(GraphicsError::InvalidParameter(
                "bindless heap was never created".to_string(),
            ));
        };
        shared.slots.release_texture(slot)
    }

    /// Register a sampler in the bindless heap (#117); sampler sibling of
    /// [`bindless_register_texture`](Self::bindless_register_texture).
    pub fn bindless_register_sampler(
        self: &Arc<Self>,
        sampler: &Arc<Sampler>,
    ) -> Result<u32, GraphicsError> {
        let shared = self.bindless_shared()?;
        let slot = shared.slots.allocate_sampler(Arc::clone(sampler))?;

        #[cfg(feature = "vulkan-backend")]
        if let crate::backend::GpuBackend::Vulkan(vulkan) = &*self.instance.backend()
            && let Err(e) = vulkan.bindless_write_sampler(
                shared.group.gpu_handle(),
                crate::bindless::BINDLESS_SAMPLERS_BINDING,
                slot,
                sampler.gpu_handle(),
            )
        {
            let _ = shared.slots.release_sampler(slot);
            return Err(e);
        }

        Ok(slot)
    }

    /// Unregister a bindless sampler slot (#117); sampler sibling of
    /// [`bindless_unregister_texture`](Self::bindless_unregister_texture).
    pub fn bindless_unregister_sampler(&self, slot: u32) -> Result<(), GraphicsError> {
        let guard = self.bindless.lock();
        let Some(shared) = &*guard else {
            return Err(GraphicsError::InvalidParameter(
                "bindless heap was never created".to_string(),
            ));
        };
        shared.slots.release_sampler(slot)
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

        // Opt-in ray-tracing AS build input (#110): when set, the vertex/index
        // buffers additionally carry `ACCELERATION_STRUCTURE_INPUT` so the mesh
        // can feed a BLAS build directly. The bit is Vulkan/ray-query-only, so
        // buffer creation fails here on other backends — callers must gate the
        // opt-in on `DeviceCapabilities::ray_query`.
        let as_input = if descriptor.acceleration_structure_input {
            BufferUsage::ACCELERATION_STRUCTURE_INPUT
        } else {
            BufferUsage::empty()
        };

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
                &BufferDescriptor::new(
                    buffer_size,
                    BufferUsage::VERTEX | BufferUsage::COPY_DST | as_input,
                )
                .with_label(label),
            )?;
            vertex_buffers.push(buffer);
        }

        // Create index buffer if needed
        let (index_buffer, index_format, index_count) = if descriptor.is_indexed() {
            let index_size = descriptor.index_buffer_size();
            let buffer = self.create_buffer(
                &BufferDescriptor::new(
                    index_size,
                    BufferUsage::INDEX | BufferUsage::COPY_DST | as_input,
                )
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
        let mesh_data = Mesh::new(
            Arc::clone(self),
            descriptor.layout.clone(),
            descriptor.topology,
            vertex_buffers,
            descriptor.vertex_count,
            index_buffer,
            index_format,
            index_count,
            descriptor.label.clone(),
        );
        let mesh = Arc::new(mesh_data);

        // Track it
        if let Ok(mut meshes) = self.meshes.write() {
            meshes.push(Arc::downgrade(&mesh));
        }

        Ok(mesh)
    }

    /// Allocate a GPU mesh from CPU data and return the transfer operations that
    /// upload its vertex/index data **through the frame graph**.
    ///
    /// Unlike [`create_mesh_from_cpu`](Self::create_mesh_from_cpu), this performs
    /// no synchronous GPU writes: the returned [`TransferOperation`]s must be
    /// added to a frame's render graph (e.g. via a manager's `flush_uploads`).
    /// Mesh data is static (written once before first use), so the upload is
    /// safe and effectively same-frame once the graph executes.
    pub fn create_mesh_deferred(
        self: &Arc<Self>,
        cpu_mesh: &CpuMesh,
    ) -> Result<(Arc<Mesh>, Vec<TransferOperation>), GraphicsError> {
        self.create_mesh_deferred_with(cpu_mesh, &cpu_mesh.to_descriptor())
    }

    /// [`create_mesh_deferred`](Self::create_mesh_deferred) with a caller-built
    /// descriptor, so opt-ins like
    /// [`MeshDescriptor::with_acceleration_structure_input`](redlilium_core::mesh::MeshDescriptor::with_acceleration_structure_input)
    /// (#110) can flow onto the GPU buffers. The descriptor must match
    /// `cpu_mesh`'s layout and counts (start from `cpu_mesh.to_descriptor()`).
    pub fn create_mesh_deferred_with(
        self: &Arc<Self>,
        cpu_mesh: &CpuMesh,
        descriptor: &MeshDescriptor,
    ) -> Result<(Arc<Mesh>, Vec<TransferOperation>), GraphicsError> {
        profile_scope!("create_mesh_deferred");

        let mesh = self.create_mesh(descriptor)?;
        let mut ops = Vec::new();

        for i in 0..cpu_mesh.buffer_count() {
            if let (Some(gpu_buffer), Some(cpu_data)) =
                (mesh.vertex_buffer(i), cpu_mesh.vertex_buffer_data(i))
                && !cpu_data.is_empty()
            {
                ops.push(TransferOperation::write_buffer(
                    Arc::clone(gpu_buffer),
                    0,
                    Arc::from(cpu_data),
                ));
            }
        }

        if let (Some(gpu_index_buffer), Some(cpu_index_data)) =
            (mesh.index_buffer(), cpu_mesh.index_data())
            && !cpu_index_data.is_empty()
        {
            ops.push(TransferOperation::write_buffer(
                Arc::clone(gpu_index_buffer),
                0,
                Arc::from(cpu_index_data),
            ));
        }

        Ok((mesh, ops))
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

    /// Current monotonic frame index (#110 phase 3), advanced once per
    /// [`advance_frame`](Self::advance_frame). `0` on non-Vulkan backends,
    /// which have no compaction pipeline.
    pub(crate) fn frame_index(&self) -> u64 {
        #[cfg(feature = "vulkan-backend")]
        if let crate::backend::GpuBackend::Vulkan(vb) = &*self.instance.backend() {
            return vb.frame_index();
        }
        0
    }

    /// Drive transparent BLAS compaction into `graph` (#110 phase 3): for every
    /// compactable BLAS whose GPU-side compacted size has come back, allocate
    /// the compacted structure and add its copy to the frame's graph. Public
    /// entry point is
    /// [`RenderGraph::flush_acceleration_structure_compaction`](crate::graph::RenderGraph::flush_acceleration_structure_compaction).
    pub(crate) fn flush_blas_compaction(self: &Arc<Self>, graph: &mut crate::graph::RenderGraph) {
        let frame_index = self.frame_index();
        self.as_compaction.lock().flush(self, graph, frame_index);
    }

    /// Advance per-frame backend state.
    ///
    /// This should be called after a frame fence has been waited on. For the
    /// Vulkan backend, this frees command buffers from the completed frame,
    /// advances the layout tracker, and resets the descriptor pool.
    ///
    /// Also cleans up dead weak references to released resources.
    pub(crate) fn advance_frame(&self) {
        use crate::backend::GpuBackend;

        #[cfg(feature = "vulkan-backend")]
        if let GpuBackend::Vulkan(vulkan_backend) = &*self.instance.backend() {
            // SAFETY: This is called after waiting on a frame fence, which guarantees
            // the GPU has finished with resources from the oldest frame slot.
            unsafe { vulkan_backend.advance_frame() };
            // Drive transparent BLAS compaction at the new frame index (#110
            // phase 3): swap in completed compacted structures and free
            // originals whose retirement window has passed.
            self.as_compaction
                .lock()
                .advance(vulkan_backend.frame_index());
        }

        // Recycle bindless slots whose retirement window has passed (#117).
        if let Some(shared) = &*self.bindless.lock() {
            shared.slots.advance_frame();
        }

        // Clean up dead weak references
        self.cleanup_dead_resources();
    }
}

impl Drop for GraphicsDevice {
    fn drop(&mut self) {
        // Keeps the instance's live-device guard counter accurate; `instance`
        // (a field) drops after this, so the call is always safe.
        self.instance.device_dropped();
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

/// Whether a bound resource's kind is compatible with a layout's declared
/// [`BindingType`](crate::BindingType). Used to validate a binding group at
/// creation time.
fn resource_matches_binding_type(
    resource: &crate::materials::BoundResource,
    binding_type: crate::materials::BindingType,
) -> bool {
    use crate::materials::{BindingType, BoundResource};
    match resource {
        BoundResource::Buffer(_) | BoundResource::BufferRange { .. } => matches!(
            binding_type,
            BindingType::UniformBuffer
                | BindingType::DynamicUniformBuffer
                | BindingType::StorageBuffer
                | BindingType::StorageBufferReadOnly
        ),
        BoundResource::Texture(_) => matches!(
            binding_type,
            BindingType::Texture
                | BindingType::TextureCube
                | BindingType::Texture2DArray
                | BindingType::DepthTexture
        ),
        BoundResource::Sampler(_) => {
            matches!(
                binding_type,
                BindingType::Sampler | BindingType::ComparisonSampler
            )
        }
        BoundResource::CombinedTextureSampler { .. } => {
            matches!(binding_type, BindingType::CombinedTextureSampler)
        }
        BoundResource::AccelerationStructure(_) => {
            matches!(binding_type, BindingType::AccelerationStructure)
        }
        BoundResource::BindlessHeap(_) => {
            matches!(
                binding_type,
                BindingType::BindlessTextures | BindingType::BindlessSamplers
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::ShaderSource;
    use crate::types::{BufferUsage, TextureFormat, TextureUsage};

    fn create_test_device() -> Arc<GraphicsDevice> {
        let instance = GraphicsInstance::new().unwrap();
        instance.create_device().unwrap()
    }

    /// User binding groups can never declare the bindless heap's binding
    /// types (#117) — the heap set is device-owned.
    #[test]
    fn bindless_layouts_rejected_in_create_binding_group() {
        let device = create_test_device();
        let layout = Arc::new(crate::materials::BindingLayout::new().with_bindless_textures(0));
        let result =
            device.create_binding_group(layout, crate::materials::BindingGroupDescriptor::new());
        assert!(matches!(result, Err(GraphicsError::InvalidParameter(_))));
    }

    /// Registration round-trips where the capability exists and gates
    /// cleanly where it does not; invalid textures are rejected either way.
    #[test]
    fn bindless_registration_gated_and_validated() {
        let device = create_test_device();
        let texture = device
            .create_texture(&crate::types::TextureDescriptor::new_2d(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap();

        if device.capabilities().bindless {
            let slot = device.bindless_register_texture(&texture).unwrap();
            device.bindless_unregister_texture(slot).unwrap();
            // Double-unregister of the same slot is an error.
            assert!(device.bindless_unregister_texture(slot).is_err());
            // The heap group and layout exist and agree.
            let group = device.bindless_heap_group().unwrap();
            let layout = device.bindless_heap_layout().unwrap();
            assert!(Arc::ptr_eq(group.layout(), &layout));
        } else {
            assert!(matches!(
                device.bindless_register_texture(&texture),
                Err(GraphicsError::FeatureNotSupported(_))
            ));
        }

        // A texture without TEXTURE_BINDING usage is rejected before any
        // capability question.
        let attachment_only = device
            .create_texture(&crate::types::TextureDescriptor::new_2d(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureUsage::RENDER_ATTACHMENT,
            ))
            .unwrap();
        assert!(matches!(
            device.bindless_register_texture(&attachment_only),
            Err(GraphicsError::InvalidParameter(_))
        ));
    }

    /// The device is named after the real adapter (XB-L3) — "Dummy Adapter"
    /// only when the dummy backend was actually selected.
    #[test]
    fn test_device_name() {
        let device = create_test_device();
        assert!(!device.name().is_empty());
    }

    /// ADR-027: capabilities come from the backend, not fabricated defaults.
    #[test]
    fn test_capabilities_come_from_backend() {
        let device = create_test_device();
        let caps = device.capabilities();
        assert_eq!(caps.tier, DeviceTier::Baseline);
        assert!(caps.supports_sample_count(1));
        assert!(caps.supports_sample_count(4));
        assert!(!caps.supports_sample_count(16));
        // Not a power of two → never supported, regardless of the mask.
        assert!(!caps.supports_sample_count(3));
    }

    /// VK-M12 (engine side): a sample count outside the queried mask is
    /// rejected at validation, not silently downgraded by a backend.
    #[test]
    fn test_create_texture_unsupported_sample_count() {
        let device = create_test_device();
        let mut descriptor = TextureDescriptor::new_2d(
            256,
            256,
            TextureFormat::Rgba8Unorm,
            TextureUsage::RENDER_ATTACHMENT,
        );
        descriptor.sample_count = 16;
        let result = device.create_texture(&descriptor);
        assert!(result.is_err());

        descriptor.sample_count = 4;
        assert!(device.create_texture(&descriptor).is_ok());
    }

    /// #119: NPOT compressed textures are legal — partial edge blocks are a
    /// normal part of the format, and the transfer path sizes uploads in
    /// blocks. 5×3 BC1 = 2×1 blocks.
    #[test]
    fn test_create_texture_compressed_npot() {
        let device = create_test_device();
        let descriptor = TextureDescriptor::new_2d(
            5,
            3,
            TextureFormat::Bc1RgbaUnorm,
            TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
        );
        if device.capabilities().texture_compression_bc {
            assert!(device.create_texture(&descriptor).is_ok());
        } else {
            assert!(matches!(
                device.create_texture(&descriptor),
                Err(GraphicsError::FeatureNotSupported(_))
            ));
        }
    }

    /// #119: compressed formats are sample-only. The usage check runs before
    /// the capability check, so this is `InvalidParameter` on every device —
    /// with or without BC support.
    #[test]
    fn test_create_texture_compressed_rejects_attachment_and_storage() {
        let device = create_test_device();
        for usage in [
            TextureUsage::RENDER_ATTACHMENT,
            TextureUsage::STORAGE_BINDING,
        ] {
            let descriptor = TextureDescriptor::new_2d(
                256,
                256,
                TextureFormat::Bc7RgbaUnorm,
                TextureUsage::TEXTURE_BINDING | usage,
            );
            assert!(matches!(
                device.create_texture(&descriptor),
                Err(GraphicsError::InvalidParameter(_))
            ));
        }
    }

    /// #119: the per-family capability lookup, independent of any backend.
    #[test]
    fn test_supports_compression_family() {
        let mut caps = *create_test_device().capabilities();
        caps.texture_compression_bc = true;
        caps.texture_compression_etc2 = false;
        caps.texture_compression_astc = false;
        use crate::types::CompressionFamily;
        assert!(caps.supports_compression_family(CompressionFamily::Bc));
        assert!(!caps.supports_compression_family(CompressionFamily::Etc2));
        assert!(!caps.supports_compression_family(CompressionFamily::Astc));
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
        use crate::types::TextureFormat;

        // Minimal valid WGSL shader for pipeline creation
        let shader_src = b"\
@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
";
        let device = create_test_device();
        let material = device
            .create_material(
                &MaterialDescriptor::new()
                    .with_shader(ShaderSource::vertex(shader_src.to_vec(), "vs_main"))
                    .with_shader(ShaderSource::fragment(shader_src.to_vec(), "fs_main"))
                    .with_color_format(TextureFormat::Bgra8Unorm)
                    .with_label("test_material"),
            )
            .unwrap();
        assert_eq!(material.label(), Some("test_material"));
        assert_eq!(device.material_count(), 1);
    }
}
