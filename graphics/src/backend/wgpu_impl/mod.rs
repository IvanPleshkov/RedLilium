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

    /// Create a new wgpu backend with custom parameters.
    pub fn with_params(
        params: &crate::instance::InstanceParameters,
    ) -> Result<Self, GraphicsError> {
        // Determine which wgpu backends to enable
        let backends = params.wgpu_backend.to_wgpu_backends();

        // Configure instance flags based on validation/debug settings
        let mut flags = wgpu::InstanceFlags::default();
        if params.validation {
            flags |= wgpu::InstanceFlags::VALIDATION;
            flags |= wgpu::InstanceFlags::GPU_BASED_VALIDATION;
        }
        if params.debug {
            flags |= wgpu::InstanceFlags::DEBUG;
        }

        // Create instance with configured backends
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            flags,
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        });

        // Request adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("No compatible GPU adapter: {e}"))
        })?;

        log::info!("wgpu adapter: {:?}", adapter.get_info());

        // Request device
        let (device, queue) = Self::request_device(&adapter)?;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            bind_group_layout_cache: parking_lot::Mutex::new(HashMap::new()),
        })
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

    /// Request a device from `adapter` with the engine's feature set and an
    /// uncaptured-error handler installed.
    ///
    /// Without a handler every uncaptured wgpu validation error aborts the
    /// process; routing them to the log makes a bad draw degrade (missing
    /// output + error message) instead of killing the app.
    fn request_device(
        adapter: &wgpu::Adapter,
    ) -> Result<(wgpu::Device, wgpu::Queue), GraphicsError> {
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RedLilium Device"),
            required_features: Self::optional_features(adapter),
            // `Limits::default()` are the WebGPU spec-guaranteed defaults —
            // satisfiable on any WebGPU adapter (unlike the native defaults that
            // exceeded WebGL limits, WG-C2). `using_resolution` raises the
            // texture/buffer size caps to what this adapter actually reports, so
            // a HiDPI canvas or large viewport isn't clamped to the 2048 floor.
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Device creation failed: {e}"))
        })?;

        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            log::error!("wgpu uncaptured error: {error}");
        }));

        Ok((device, queue))
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
        let (new_device, new_queue) = Self::request_device(&new_adapter)?;

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

    /// Get the backend name.
    pub fn name(&self) -> &'static str {
        "wgpu Backend"
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
