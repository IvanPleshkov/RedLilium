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

/// Scratch buffers reused across draw commands to avoid per-draw heap allocations.
///
/// Only contains types without Rust lifetimes (can be stored long-term).
/// Vecs are cleared between draws but retain their capacity across frames.
#[derive(Default)]
struct WgpuEncoderScratch {
    // Backing storage for types without Rust lifetimes:
    color_targets: Vec<Option<wgpu::ColorTargetState>>,
    bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    vertex_attributes: Vec<Vec<wgpu::VertexAttribute>>,
    color_formats: Vec<Option<wgpu::TextureFormat>>,
    // GPU handle Vecs (no lifetimes, safe to pool):
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    bind_groups: Vec<wgpu::BindGroup>,
}

/// Cached render pipeline with its bind group layouts.
///
/// wgpu resource types are internally reference-counted; cloning is a refcount bump.
/// Bind group layouts are cached alongside the pipeline because they are needed
/// to create per-frame bind groups.
struct CachedPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

/// Cached compute pipeline with its bind group layouts.
struct CachedComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
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
    encoder_scratch: std::sync::Mutex<WgpuEncoderScratch>,
    pipeline_cache: std::sync::Mutex<HashMap<u64, CachedPipeline>>,
    compute_pipeline_cache: std::sync::Mutex<HashMap<u64, CachedComputePipeline>>,
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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RedLilium Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Device creation failed: {e}"))
        })?;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            encoder_scratch: std::sync::Mutex::new(WgpuEncoderScratch::default()),
            pipeline_cache: std::sync::Mutex::new(HashMap::new()),
            compute_pipeline_cache: std::sync::Mutex::new(HashMap::new()),
        })
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
        let (new_device, new_queue) =
            pollster::block_on(new_adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("RedLilium Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            }))
            .map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!("Device creation failed: {e}"))
            })?;

        // Update the backend with new adapter and device.
        // Clear pipeline cache since cached pipelines belong to the old device.
        self.adapter = new_adapter;
        self.device = Arc::new(new_device);
        self.queue = Arc::new(new_queue);
        self.pipeline_cache.lock().unwrap().clear();

        Ok(true)
    }

    /// Get the backend name.
    pub fn name(&self) -> &'static str {
        "wgpu Backend"
    }

    /// Execute a compiled render graph.
    ///
    /// # Async Behavior
    ///
    /// - If `signal_fence` is provided: Returns immediately after submission (async).
    ///   The caller can wait on the fence using `wait_fence()` or poll with `is_fence_signaled()`.
    /// - If `signal_fence` is `None`: Blocks until GPU completes (sync, for backwards compatibility).
    ///
    /// For true async rendering with multiple frames in flight, always provide a fence.
    pub fn execute_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        _wait_semaphores: &[&super::GpuSemaphore],
        _signal_semaphores: &[&super::GpuSemaphore],
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

        // Store submission index in fence for async polling
        if let Some(GpuFence::Wgpu {
            submission_index: fence_idx,
            ..
        }) = signal_fence
            && let Ok(mut guard) = fence_idx.lock()
        {
            *guard = Some(submission_index);
            // Async path: return immediately, caller will wait on fence
            return Ok(());
        }

        // Sync path: no fence provided, wait for GPU to complete before returning
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(10)),
        });

        Ok(())
    }
}
