//! wgpu GPU backend implementation.
//!
//! This backend uses wgpu for cross-platform GPU access, supporting
//! Vulkan, Metal, DX12, and WebGPU.

pub(crate) mod conversion;
mod pass_encoding;
mod resources;
pub mod swapchain;

use std::sync::Arc;

/// A texture view for a surface texture (swapchain image).
///
/// This wraps the wgpu::TextureView from the surface texture for use in render passes.
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

use super::GpuFence;

/// wgpu-based GPU backend.
pub struct WgpuBackend {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
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

    /// Get the backend name.
    pub fn name(&self) -> &'static str {
        "wgpu Backend"
    }

    /// Execute a compiled render graph.
    pub fn execute_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderGraph Encoder"),
            });

        // Get all passes from the graph
        let passes = graph.passes();

        // Process each pass in compiled order
        for handle in compiled.pass_order() {
            let pass = &passes[handle.index()];
            self.encode_pass(&mut encoder, pass)?;
        }

        // Submit commands
        let command_buffer = encoder.finish();
        let submission_index = self.queue.submit(std::iter::once(command_buffer));

        // Store submission index in fence for polling
        if let Some(GpuFence::Wgpu {
            submission_index: fence_idx,
            ..
        }) = signal_fence
            && let Ok(mut guard) = fence_idx.lock()
        {
            *guard = Some(submission_index.clone());
        }

        // Wait for GPU to complete before returning
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(10)),
        });

        Ok(())
    }
}
