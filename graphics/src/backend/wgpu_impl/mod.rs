//! wgpu GPU backend implementation.
//!
//! This backend uses wgpu for cross-platform GPU access, supporting
//! Vulkan, Metal, DX12, and WebGPU.

pub(crate) mod conversion;
mod pass_encoding;
mod resources;
pub mod swapchain;

use std::collections::HashMap;
use std::sync::Arc;

use crate::materials::{BindingType, ShaderStageFlags};

/// Content key for bind-group-layout dedup: one entry per binding, in
/// declaration order (binding index, type, visibility). Labels excluded —
/// layouts differing only by label are wgpu-equivalent. Mirrors the Vulkan
/// `ds_layout_cache` key so a binding group and the pipelines that use it share
/// the same `wgpu::BindGroupLayout` object (wgpu requires them to be
/// compatible).
type BindGroupLayoutKey = Vec<(u32, BindingType, ShaderStageFlags)>;

pub(crate) fn bind_group_layout_key(
    layout: &crate::materials::BindingLayout,
) -> BindGroupLayoutKey {
    layout
        .entries
        .iter()
        .map(|e| (e.binding, e.binding_type, e.visibility))
        .collect()
}

/// A texture view for a surface texture (swapchain image).
///
/// This wraps the wgpu::TextureView from the surface texture for use in render passes.
/// Note: Uses Arc because this type needs to be Clone for use in RenderTarget,
/// and wgpu::TextureView doesn't implement Clone.
#[derive(Clone)]
pub struct SurfaceTextureView {
    pub(crate) view: Arc<wgpu::TextureView>,
}

impl std::fmt::Debug for SurfaceTextureView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceTextureView").finish()
    }
}

impl SurfaceTextureView {
    /// Get the underlying wgpu texture view.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, RenderGraph};
use redlilium_core::profiling::profile_scope;

use super::GpuFence;

/// wgpu-based GPU backend.
pub struct WgpuBackend {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Content-keyed dedup of bind group layouts. Both pipeline creation and
    /// `create_binding_group` pull from here, so a binding group's
    /// `wgpu::BindGroupLayout` is the *same object* as the pipelines that use
    /// it — the clean way to satisfy wgpu's bind-group/pipeline compatibility.
    // parking_lot: no poisoning.
    bind_group_layout_cache: parking_lot::Mutex<HashMap<BindGroupLayoutKey, wgpu::BindGroupLayout>>,
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBackend")
            .field("adapter", &self.adapter.get_info().name)
            .finish()
    }
}

impl WgpuBackend {
    /// Create a new wgpu backend with default parameters.
    pub fn new() -> Result<Self, GraphicsError> {
        Self::with_params(&crate::instance::InstanceParameters::default())
    }

    /// Create a new wgpu backend with custom parameters (blocking).
    ///
    /// Native path: drives `request_adapter`/`request_device` to completion with
    /// `pollster::block_on`. That cannot make progress on the single browser
    /// thread, so on wasm this returns an error instead of hanging — the web path
    /// builds the backend through [`with_params_async`](Self::with_params_async)
    /// (#33).
    pub fn with_params(
        params: &crate::instance::InstanceParameters,
    ) -> Result<Self, GraphicsError> {
        let instance = Self::build_instance(params);

        #[cfg(target_arch = "wasm32")]
        {
            let _ = instance;
            Err(GraphicsError::ResourceCreationFailed(
                "synchronous wgpu backend creation is unavailable on wasm (block_on cannot \
                 progress on the browser thread); use with_params_async (#33)"
                    .to_string(),
            ))
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let adapter = if let Some(explicit) = Self::explicit_adapter(&instance, params)? {
                explicit
            } else {
                let mut adapter = None;
                for pref in Self::power_preferences(&params.adapter) {
                    match pollster::block_on(instance.request_adapter(&Self::adapter_options(pref)))
                    {
                        Ok(a) => {
                            adapter = Some(a);
                            break;
                        }
                        Err(e) => log::warn!("request_adapter({pref:?}) found no adapter: {e}"),
                    }
                }
                adapter.ok_or_else(|| {
                    GraphicsError::ResourceCreationFailed(
                        "No compatible GPU adapter (tried the preferred and default power \
                         preference)"
                            .to_string(),
                    )
                })?
            };
            let (device, queue) =
                pollster::block_on(adapter.request_device(&Self::device_descriptor(&adapter)))
                    .map_err(|e| {
                        GraphicsError::ResourceCreationFailed(format!(
                            "Device creation failed: {e}"
                        ))
                    })?;
            Ok(Self::assemble(instance, adapter, device, queue))
        }
    }

    /// Create a new wgpu backend, awaiting the adapter/device requests.
    ///
    /// The browser path (#33): `request_adapter`/`request_device` resolve as JS
    /// promises driven by the event loop, so the caller must be inside an async
    /// task (`wasm_bindgen_futures::spawn_local`) and must never block on it.
    /// Also correct on native (the futures resolve immediately).
    pub async fn with_params_async(
        params: &crate::instance::InstanceParameters,
    ) -> Result<Self, GraphicsError> {
        let instance = Self::build_instance(params);

        // Explicit adapter identity: resolvable by enumeration on native
        // only. WebGPU has no adapter enumeration (the browser decides), so
        // Explicit degrades to the default power-preference path with a
        // warning (ADR-028).
        #[cfg(not(target_arch = "wasm32"))]
        let explicit = Self::explicit_adapter(&instance, params)?;
        #[cfg(target_arch = "wasm32")]
        let explicit: Option<wgpu::Adapter> = {
            if matches!(
                params.adapter,
                crate::instance::AdapterPreference::Explicit(_)
            ) {
                log::warn!(
                    "AdapterPreference::Explicit is not supported on WebGPU \
                     (no adapter enumeration in the browser); using the default adapter"
                );
            }
            None
        };

        let adapter = if let Some(explicit) = explicit {
            explicit
        } else {
            let mut adapter = None;
            for pref in Self::power_preferences(&params.adapter) {
                match instance.request_adapter(&Self::adapter_options(pref)).await {
                    Ok(a) => {
                        adapter = Some(a);
                        break;
                    }
                    Err(e) => log::warn!("request_adapter({pref:?}) found no adapter: {e}"),
                }
            }
            adapter.ok_or_else(|| {
                GraphicsError::ResourceCreationFailed(
                    "No compatible GPU adapter (tried the preferred and default power preference)"
                        .to_string(),
                )
            })?
        };
        let (device, queue) = adapter
            .request_device(&Self::device_descriptor(&adapter))
            .await
            .map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!("Device creation failed: {e}"))
            })?;
        Ok(Self::assemble(instance, adapter, device, queue))
    }

    /// Build the `wgpu::Instance` from engine params. Shared by the blocking and
    /// async constructors.
    fn build_instance(params: &crate::instance::InstanceParameters) -> wgpu::Instance {
        let backends = params.wgpu_backend.to_wgpu_backends();

        let mut flags = wgpu::InstanceFlags::default();
        if params.validation {
            flags |= wgpu::InstanceFlags::VALIDATION;
            flags |= wgpu::InstanceFlags::GPU_BASED_VALIDATION;
        }
        if params.debug {
            flags |= wgpu::InstanceFlags::DEBUG;
        }

        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            flags,
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        })
    }

    /// Adapter request options for the given power preference (no surface hint —
    /// on WebGPU the single adapter presents to the canvas; compatibility is
    /// verified later in
    /// [`ensure_compatible_with_surface`](Self::ensure_compatible_with_surface)).
    fn adapter_options(
        power_preference: wgpu::PowerPreference,
    ) -> wgpu::RequestAdapterOptions<'static, 'static> {
        wgpu::RequestAdapterOptions {
            power_preference,
            compatible_surface: None,
            force_fallback_adapter: false,
        }
    }

    /// Power preferences to try, in order, for the given adapter policy
    /// (ADR-028). The preferred setting first; `None` as a fallback because
    /// on some setups (notably dual-GPU laptops where the high-performance
    /// GPU is powered down, and browsers generally) a specific preference
    /// returns no adapter while the default does (#33).
    fn power_preferences(
        preference: &crate::instance::AdapterPreference,
    ) -> [wgpu::PowerPreference; 2] {
        use crate::instance::AdapterPreference;
        let preferred = match preference {
            AdapterPreference::LowPower => wgpu::PowerPreference::LowPower,
            // Explicit is resolved by enumeration before this is consulted;
            // reaching here means enumeration was unavailable (web).
            AdapterPreference::Auto
            | AdapterPreference::HighPerformance
            | AdapterPreference::Explicit(_) => wgpu::PowerPreference::HighPerformance,
        };
        [preferred, wgpu::PowerPreference::None]
    }

    /// Resolve an [`AdapterPreference::Explicit`] identity by enumerating
    /// adapters (native only). `Ok(None)` when the policy is not Explicit;
    /// an error when it is Explicit and nothing matches — an explicitly
    /// requested adapter must never silently degrade (ADR-028).
    #[cfg(not(target_arch = "wasm32"))]
    fn explicit_adapter(
        instance: &wgpu::Instance,
        params: &crate::instance::InstanceParameters,
    ) -> Result<Option<wgpu::Adapter>, GraphicsError> {
        let crate::instance::AdapterPreference::Explicit(id) = &params.adapter else {
            return Ok(None);
        };
        let candidates =
            pollster::block_on(instance.enumerate_adapters(params.wgpu_backend.to_wgpu_backends()));
        let mut names = Vec::new();
        for adapter in candidates {
            let info = adapter.get_info();
            if id.matches(info.vendor, info.device, &info.name) {
                log::info!("Explicit adapter matched: {} ({id:?})", info.name);
                return Ok(Some(adapter));
            }
            names.push(format!(
                "{} ({:04x}:{:04x})",
                info.name, info.vendor, info.device
            ));
        }
        Err(GraphicsError::InitializationFailed(format!(
            "Requested adapter {id:?} not found among [{}] \
             (check REDLILIUM_ADAPTER / adapter preference)",
            names.join(", ")
        )))
    }

    /// Features requested opportunistically when the adapter supports them:
    /// compressed texture formats (BC/ETC2/ASTC) so `create_texture` with
    /// those formats works wherever the hardware can, and filterable 32-bit
    /// float textures.
    fn optional_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        let supported = adapter.features();
        [
            wgpu::Features::TEXTURE_COMPRESSION_BC,
            wgpu::Features::TEXTURE_COMPRESSION_ETC2,
            wgpu::Features::TEXTURE_COMPRESSION_ASTC,
            wgpu::Features::FLOAT32_FILTERABLE,
            // Wireframe (`PolygonMode::Line`) is native-only — absent on WebGPU
            // and WebGL. Requesting it unconditionally makes `request_device`
            // fail outright on web (WG-C2). Request it only where supported and
            // treat it as a runtime capability (`supports_wireframe`): pipeline
            // creation downgrades `Line` to `Fill` when it is missing rather
            // than tripping a validation error.
            wgpu::Features::POLYGON_MODE_LINE,
        ]
        .into_iter()
        .filter(|f| supported.contains(*f))
        .fold(wgpu::Features::empty(), |acc, f| acc | f)
    }

    /// Device descriptor with the engine's opportunistic feature set and
    /// WebGPU-safe limits raised to the adapter's actual resolution caps.
    ///
    /// `Limits::default()` are the WebGPU spec-guaranteed defaults — satisfiable
    /// on any WebGPU adapter (unlike the native defaults that exceeded WebGL
    /// limits, WG-C2). `using_resolution` raises the texture/buffer size caps to
    /// what this adapter actually reports, so a HiDPI canvas or large viewport
    /// isn't clamped to the 2048 floor.
    fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: Some("RedLilium Device"),
            required_features: Self::optional_features(adapter),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }
    }

    /// Install the uncaptured-error handler on a device.
    ///
    /// Without a handler every uncaptured wgpu validation error aborts the
    /// process; routing them to the log makes a bad draw degrade (missing
    /// output + error message) instead of killing the app.
    fn install_error_handler(device: &wgpu::Device) {
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            log::error!("wgpu uncaptured error: {error}");
        }));
    }

    /// Assemble the backend from a freshly created adapter/device/queue, logging
    /// the adapter and installing the uncaptured-error handler. Shared by the
    /// blocking and async constructors.
    fn assemble(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        log::info!("wgpu adapter: {:?}", adapter.get_info());
        Self::install_error_handler(&device);
        Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            bind_group_layout_cache: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    /// Get the wgpu instance.
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// Get the wgpu adapter.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// Get the wgpu device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Get the wgpu queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Check if the current adapter is compatible with a surface.
    pub fn is_adapter_compatible_with_surface(&self, surface: &wgpu::Surface<'_>) -> bool {
        self.adapter.is_surface_supported(surface)
    }

    /// Re-request an adapter that is compatible with the given surface.
    ///
    /// This creates a new device and queue if the current adapter is not compatible.
    /// Returns true if a compatible adapter was found and the backend was updated.
    pub fn ensure_compatible_with_surface(
        &mut self,
        surface: &wgpu::Surface<'_>,
    ) -> Result<bool, GraphicsError> {
        // Check if current adapter is already compatible
        if self.adapter.is_surface_supported(surface) {
            return Ok(true);
        }
        self.reacquire_for_surface(surface)
    }

    /// Re-request a surface-compatible adapter+device (native). The blocking
    /// re-request is impossible on wasm, so the wasm variant below errors instead
    /// — a split by platform keeps `block_on` out of the wasm build path (#33).
    #[cfg(not(target_arch = "wasm32"))]
    fn reacquire_for_surface(
        &mut self,
        surface: &wgpu::Surface<'_>,
    ) -> Result<bool, GraphicsError> {
        log::info!(
            "Current adapter '{}' not compatible with surface, requesting new adapter",
            self.adapter.get_info().name
        );

        // Request a new adapter that is compatible with the surface
        let new_adapter =
            pollster::block_on(self.instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            }))
            .map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "No GPU adapter compatible with surface: {e}"
                ))
            })?;

        log::info!(
            "Found compatible adapter: {:?}",
            new_adapter.get_info().name
        );

        // Request device from the new adapter
        let (new_device, new_queue) =
            pollster::block_on(new_adapter.request_device(&Self::device_descriptor(&new_adapter)))
                .map_err(|e| {
                    GraphicsError::ResourceCreationFailed(format!("Device creation failed: {e}"))
                })?;
        Self::install_error_handler(&new_device);

        // Update the backend with new adapter and device.
        // Note: Pipelines are now owned by Materials (GpuPipeline),
        // so there's no pipeline cache to clear here. Materials created
        // with the old device will need to be recreated by the caller.
        self.adapter = new_adapter;
        self.device = Arc::new(new_device);
        self.queue = Arc::new(new_queue);
        // The cached bind group layouts belong to the old device; drop them so
        // materials/binding groups recreated against the new device get fresh,
        // compatible layouts.
        self.bind_group_layout_cache.lock().clear();

        Ok(true)
    }

    /// wasm variant: a synchronous re-request can't progress on the browser
    /// thread. Unreachable in practice on WebGPU (the sole adapter presents to
    /// the canvas), so surface a clean error rather than hang (#33).
    #[cfg(target_arch = "wasm32")]
    fn reacquire_for_surface(
        &mut self,
        _surface: &wgpu::Surface<'_>,
    ) -> Result<bool, GraphicsError> {
        Err(GraphicsError::ResourceCreationFailed(
            "current WebGPU adapter is not compatible with the canvas surface, and a \
             synchronous re-request is impossible on wasm"
                .to_string(),
        ))
    }

    /// Get the backend name.
    pub fn name(&self) -> &'static str {
        "wgpu Backend"
    }

    /// Capabilities from the *granted* device limits and features (ADR-027)
    /// — what `request_device` actually returned, not what the engine wishes
    /// for. This is what makes engine-level validation agree with wgpu's own:
    /// a texture that passes `create_texture` validation here cannot trip a
    /// wgpu limit later (XB-M7).
    pub fn capabilities(&self) -> crate::device::DeviceCapabilities {
        let limits = self.device.limits();
        let features = self.device.features();
        let downlevel = self.adapter.get_downlevel_capabilities();
        crate::device::DeviceCapabilities {
            tier: crate::device::DeviceTier::Baseline,
            max_texture_dimension: limits.max_texture_dimension_2d,
            max_buffer_size: limits.max_buffer_size,
            // WebGPU validates anisotropy in [1, 16] and clamps to hardware
            // internally; there is no per-adapter query to be more precise.
            max_sampler_anisotropy: 16,
            // WebGPU guarantees exactly 1x and 4x for renderable formats.
            sample_count_mask: 1 | 4,
            wireframe: features.contains(wgpu::Features::POLYGON_MODE_LINE),
            async_compute: false,
            // wgpu exposes a single queue; transfer routing is Vulkan-only.
            transfer_queue: false,
            compute_shaders: downlevel
                .flags
                .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS),
            // Per-pass timestamp collection is wired for the Vulkan backend
            // only (#95); the wgpu pass encoders pass `timestamp_writes: None`.
            gpu_timestamps: false,
            // No blit path in the wgpu backend (#96); GenerateMipmaps is a no-op
            // and the loader falls back to a single mip. A wgpu render/compute
            // downsampler is a named follow-up.
            mip_generation: false,
            // No VRAM budget API on wgpu (#98); heaps are empty, the panel
            // falls back to the engine's allocator/resource figures.
            memory_budget: false,
            // Inline ray tracing is Vulkan-only (#110). wgpu's experimental
            // ray-query support is a named follow-up (issue #110 phase 4).
            ray_query: false,
        }
    }

    /// Latest retired-slot GPU timings — always empty; the wgpu backend does
    /// not record timestamps (#95).
    pub fn latest_gpu_timings(&self) -> crate::device::FrameGpuTimings {
        crate::device::FrameGpuTimings::default()
    }

    /// GPU-memory statistics — no budget API on wgpu, so heaps and allocator
    /// totals are empty (#98). The device layer still fills the engine's
    /// live-resource counts so the panel shows something meaningful.
    pub fn latest_memory_stats(&self) -> crate::device::GpuMemoryStats {
        crate::device::GpuMemoryStats::default()
    }

    /// Adapter info from the wgpu adapter.
    pub fn adapter_info(&self) -> crate::instance::AdapterInfo {
        let info = self.adapter.get_info();
        let device_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => crate::instance::AdapterType::Discrete,
            wgpu::DeviceType::IntegratedGpu => crate::instance::AdapterType::Integrated,
            wgpu::DeviceType::Cpu => crate::instance::AdapterType::Software,
            wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => {
                crate::instance::AdapterType::Unknown
            }
        };
        crate::instance::AdapterInfo {
            name: info.name,
            vendor: crate::instance::vendor_name(info.vendor),
            device_type,
            vendor_id: info.vendor,
            device_id: info.device,
        }
    }

    /// Execute a compiled render graph.
    ///
    /// # Ordering & synchronization (wgpu)
    ///
    /// wgpu does not expose GPU semaphores, so `_wait_semaphores` /
    /// `_signal_semaphores` are intentionally ignored. Cross-graph ordering is
    /// instead guaranteed by wgpu's **in-order execution of queue submissions**:
    /// graphs are submitted in dependency order (the scheduler executes
    /// `submit()`/`present()` in call order to a single queue), so a dependent
    /// graph's work always runs after its dependencies' work.
    ///
    /// # Fence behavior
    ///
    /// - If `signal_fence` is provided: returns immediately after submission;
    ///   the submission index is stored in the fence for async polling.
    /// - If `signal_fence` is `None`: currently blocks until the submission
    ///   completes (`poll(Wait)`) and returns an error on timeout/poll
    ///   failure. Per the trait-level contract this blocking is an
    ///   implementation detail — callers must not rely on it.
    ///
    /// NOTE: the `None`-fence block is conservative. CPU/GPU overlap is bounded
    /// per frame by the frame-in-flight fence waited in
    /// [`FramePipeline::begin_frame`](crate::pipeline::FramePipeline::begin_frame),
    /// so making intermediate submits non-blocking is possible — but only once
    /// every frame's GPU completion is represented by a real submission-tied
    /// fence (today `FrameSchedule::finish` creates an untied fence and relies on
    /// this block to mean "GPU is idle"). Removing the block without that change
    /// would let `begin_frame` recycle resources still in use by the GPU.
    pub fn execute_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError> {
        profile_scope!("wgpu_execute_graph");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderGraph Encoder"),
            });

        // Get all passes from the graph
        let passes = graph.passes();

        // Process each pass in compiled order
        {
            profile_scope!("record_passes");
            for handle in compiled.pass_order() {
                let pass = &passes[handle.index()];
                self.encode_pass(&mut encoder, pass)?;
            }
        }

        // Submit commands
        let command_buffer = encoder.finish();
        let submission_index = {
            profile_scope!("queue_submit");
            self.queue.submit(std::iter::once(command_buffer))
        };

        // Async path: tie the fence to this submission and register a completion
        // callback. `on_submitted_work_done` fires when the GPU finishes — on
        // native when `device.poll` runs it, on wasm from the browser event loop
        // (the only completion signal there; `poll` can't block, #33). The
        // generation guards against ABA: a slot fence re-submitted before this
        // callback fires carries a newer generation, so this callback becomes a
        // no-op (see `WgpuFenceState::Submitted`).
        if let Some(GpuFence::Wgpu {
            state, generation, ..
        }) = signal_fence
        {
            let this_gen = generation.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            {
                let mut guard = state.lock().unwrap_or_else(|e| e.into_inner());
                *guard = crate::backend::WgpuFenceState::Submitted {
                    index: submission_index,
                    generation: this_gen,
                };
            }
            let state_cb = std::sync::Arc::clone(state);
            self.queue.on_submitted_work_done(move || {
                let mut guard = state_cb.lock().unwrap_or_else(|e| e.into_inner());
                if let crate::backend::WgpuFenceState::Submitted { generation, .. } = &*guard
                    && *generation == this_gen
                {
                    *guard = crate::backend::WgpuFenceState::Signaled;
                }
            });
            return Ok(());
        }

        // No-fence path: the caller wants CPU-synchronous completion. The frame
        // pipeline always ties a fence, so this is never on the wasm hot path;
        // still, a browser thread cannot block, so guard it loudly rather than
        // silently returning stale (any real wasm need here must go async).
        #[cfg(target_arch = "wasm32")]
        {
            let _ = submission_index;
            log::error!(
                "execute_graph without a fence cannot block for completion on wasm; \
                 returning without waiting — the caller must use a fenced submit + \
                 non-blocking readiness polling instead"
            );
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        // A timeout or poll failure must be surfaced — pretending the work
        // finished lets callers recycle in-flight resources.
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(10)),
        }) {
            Ok(status) if status.wait_finished() => Ok(()),
            Ok(_) => Err(GraphicsError::Timeout(
                "render graph submission did not complete within 10 s; GPU may be hung".into(),
            )),
            Err(e) => Err(GraphicsError::Internal(format!(
                "device poll failed after graph submission: {e}"
            ))),
        }
    }
}
