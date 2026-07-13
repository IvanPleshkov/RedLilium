//! GPU backend abstraction layer.
//!
//! This module provides an enum-based abstraction for GPU backends,
//! allowing the graphics crate to work with different GPU APIs without
//! dynamic dispatch.
//!
//! # Available Backends
//!
//! - `Dummy` (default): No-op backend for testing and development
//! - `Wgpu`: Cross-platform backend using wgpu (requires `wgpu-backend` feature)
//! - `Vulkan`: Native Vulkan backend using ash (requires `vulkan-backend` feature)
//!
//! # Architecture
//!
//! The [`GpuBackend`] enum provides:
//! - Instance and device creation
//! - Resource creation (buffers, textures, samplers)
//! - Command buffer recording and submission
//! - Synchronization primitives

#[cfg(feature = "wgpu-backend")]
pub mod wgpu_impl;

#[cfg(feature = "vulkan-backend")]
pub mod vulkan;

pub mod dummy;

use std::sync::Arc;

#[cfg(feature = "vulkan-backend")]
use ash::vk;
#[cfg(feature = "vulkan-backend")]
use ash::vk::Handle;
#[cfg(feature = "vulkan-backend")]
use gpu_allocator::vulkan::Allocation;
#[cfg(feature = "vulkan-backend")]
use gpu_allocator::vulkan::Allocator;
#[cfg(feature = "vulkan-backend")]
use parking_lot::Mutex;
#[cfg(feature = "vulkan-backend")]
use parking_lot::RwLock;

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, RenderGraph};
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

/// Handle to a GPU buffer resource.
#[allow(clippy::large_enum_variant)]
pub enum GpuBuffer {
    /// Dummy backend (no GPU allocation)
    Dummy,
    /// wgpu backend buffer
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu::Buffer),
    /// Vulkan backend buffer
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        buffer: vk::Buffer,
        allocation: Mutex<Option<Allocation>>,
        size: u64,
        /// Allocator for freeing memory on drop.
        allocator: Arc<Mutex<Allocator>>,
        /// Shared retirement queue: `Drop` pushes the buffer handle here so
        /// the backend removes its access-tracker entry in `advance_frame`.
        retired: Arc<Mutex<vulkan::RetiredTrackerHandles>>,
    },
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuBuffer::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(buffer) => f.debug_tuple("GpuBuffer::Wgpu").field(buffer).finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { buffer, size, .. } => f
                .debug_struct("GpuBuffer::Vulkan")
                .field("buffer", buffer)
                .field("size", size)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to a GPU texture resource.
#[allow(clippy::large_enum_variant)]
pub enum GpuTexture {
    /// Dummy backend (no GPU allocation)
    Dummy,
    /// wgpu backend texture
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
    },
    /// Vulkan backend texture
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        image: vk::Image,
        view: vk::ImageView,
        allocation: Mutex<Option<Allocation>>,
        format: vk::Format,
        extent: vk::Extent3D,
        /// Allocator for freeing memory on drop.
        allocator: Arc<Mutex<Allocator>>,
        /// Stable, process-unique id used to key the layout tracker. Unlike the
        /// raw `image` handle, this is never reused after the texture is
        /// destroyed, so it can't alias a recreated texture's tracked layout.
        id: u64,
        /// Shared retirement queue: `Drop` pushes `id` here so the backend
        /// removes its layout-tracker entry in `advance_frame`.
        retired: Arc<Mutex<vulkan::RetiredTrackerHandles>>,
    },
}

impl std::fmt::Debug for GpuTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuTexture::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { texture, view } => f
                .debug_struct("GpuTexture::Wgpu")
                .field("texture", texture)
                .field("view", view)
                .finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan {
                image,
                view,
                format,
                extent,
                ..
            } => f
                .debug_struct("GpuTexture::Vulkan")
                .field("image", image)
                .field("view", view)
                .field("format", format)
                .field("extent", extent)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to a GPU sampler resource.
#[allow(clippy::large_enum_variant)]
pub enum GpuSampler {
    /// Dummy backend (no GPU allocation)
    Dummy,
    /// wgpu backend sampler
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu::Sampler),
    /// Vulkan backend sampler
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        sampler: vk::Sampler,
    },
}

impl std::fmt::Debug for GpuSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuSampler::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(sampler) => f.debug_tuple("GpuSampler::Wgpu").field(sampler).finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { sampler, .. } => f
                .debug_struct("GpuSampler::Vulkan")
                .field("sampler", sampler)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to a GPU pipeline (graphics or compute).
///
/// Material owns its pipeline directly — one Material, one pipeline.
/// Created at `GraphicsDevice::create_material()` time.
#[allow(clippy::large_enum_variant)]
pub enum GpuPipeline {
    /// Dummy backend (no GPU pipeline)
    Dummy,
    /// wgpu backend graphics pipeline
    #[cfg(feature = "wgpu-backend")]
    WgpuGraphics {
        pipeline: wgpu::RenderPipeline,
        bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    },
    /// wgpu backend compute pipeline
    #[cfg(feature = "wgpu-backend")]
    WgpuCompute {
        pipeline: wgpu::ComputePipeline,
        bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    },
    /// Vulkan backend pipeline (graphics or compute)
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        /// Layouts for per-draw descriptor-set allocation. Borrowed copies of
        /// handles owned by the backend's `PipelineManager` dedup cache —
        /// shared between materials with identical binding layouts and valid
        /// for the backend's lifetime. `Drop` must not destroy them.
        descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    },
}

impl std::fmt::Debug for GpuPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuPipeline::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::WgpuGraphics { .. } => f
                .debug_struct("GpuPipeline::WgpuGraphics")
                .finish_non_exhaustive(),
            #[cfg(feature = "wgpu-backend")]
            Self::WgpuCompute { .. } => f
                .debug_struct("GpuPipeline::WgpuCompute")
                .finish_non_exhaustive(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { pipeline, .. } => f
                .debug_struct("GpuPipeline::Vulkan")
                .field("pipeline", pipeline)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to a GPU binding group (a compiled [`BindingGroup`]).
///
/// This is the eagerly-created descriptor set (Vulkan) / bind group (wgpu) that
/// backs a [`BindingGroup`](crate::BindingGroup). It is created once at
/// `GraphicsDevice::create_binding_group()` time and bound — never rebuilt — on
/// every draw. Its lifetime equals the owning `Arc<BindingGroup>`; `Drop` frees
/// the GPU resource.
#[allow(clippy::large_enum_variant)]
pub enum GpuBindingGroup {
    /// Dummy backend (no GPU allocation).
    Dummy,
    /// wgpu backend bind group.
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu::BindGroup),
    /// Vulkan backend descriptor set, allocated from the persistent pool chain.
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        descriptor_set: vk::DescriptorSet,
        /// The specific pool this set was allocated from — `vkFreeDescriptorSets`
        /// must target it.
        pool: vk::DescriptorPool,
        /// The shared persistent pool chain, held so `Drop` can free the set
        /// back to it from any thread (host access to the pool must be
        /// externally synchronized — the shared mutex provides that).
        pools: Arc<Mutex<vulkan::PersistentDescriptorPools>>,
    },
}

impl std::fmt::Debug for GpuBindingGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuBindingGroup::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(bg) => f.debug_tuple("GpuBindingGroup::Wgpu").field(bg).finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { descriptor_set, .. } => f
                .debug_struct("GpuBindingGroup::Vulkan")
                .field("descriptor_set", descriptor_set)
                .finish_non_exhaustive(),
        }
    }
}

/// Handle to a GPU fence for CPU-GPU synchronization.
///
/// # Backend Differences
///
/// Fences are implemented differently across backends due to API constraints:
///
/// ## Vulkan (timeline semaphore value)
/// A `(semaphore, value)` pair on the backend's per-queue timeline semaphore:
/// each `execute_graph` submit signals the next monotonically increasing value
/// and stamps it into the fence it was given. The CPU waits or polls via
/// `vkWaitSemaphores` / `vkGetSemaphoreCounterValue`. One primitive serves
/// host waits, cross-queue GPU waits (#47), and resource retirement — and no
/// per-fence Vulkan object is created or destroyed.
///
/// ## wgpu ([`WgpuFenceState`])
/// wgpu abstracts over multiple backends (Vulkan, Metal, DX12, WebGPU) and doesn't
/// expose native fence handles to maintain portability. The fence emulates
/// Vulkan's binary semantics with a small state machine:
/// - `Unsignaled` / `Signaled` — the initial state chosen by `create_fence`
/// - `Submitted(index)` — set by `execute_graph`; polling/waiting resolves the
///   state by asking the device about that submission
///
/// **Why not expose GPU fences in wgpu?** Each backend (Metal, DX12, WebGPU) has
/// different synchronization primitives. wgpu's submission index abstraction works
/// across all backends at the cost of less precise control.
///
/// ## Dummy (`AtomicBool`)
/// Simple CPU-side flag for testing without GPU hardware.
#[allow(clippy::large_enum_variant)]
pub enum GpuFence {
    /// Dummy backend - CPU-side atomic boolean for testing.
    Dummy {
        signaled: std::sync::atomic::AtomicBool,
    },
    /// wgpu backend - binary fence emulated over submission indices.
    ///
    /// `state` is `Arc<Mutex>` (not a plain `Mutex`) so the `queue
    /// .on_submitted_work_done` callback registered at submit can hold a clone
    /// and flip it to `Signaled` when the GPU finishes — the only completion
    /// signal available on wasm, where `device.poll` cannot block (#33).
    /// `generation` mints a fresh id per submit; the callback only signals if
    /// the state still carries its generation, so a late callback for a slot
    /// fence that has since been re-submitted is a no-op (ABA guard).
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        device: Arc<wgpu::Device>,
        state: Arc<std::sync::Mutex<WgpuFenceState>>,
        generation: Arc<std::sync::atomic::AtomicU64>,
    },
    /// Vulkan backend - a wait value on a queue's timeline semaphore.
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        device: ash::Device,
        /// Raw handle of the backend's per-queue timeline semaphore. NOT
        /// owned: the backend destroys it at teardown; the fence merely
        /// references it (each `Fence` holds an `Arc<GraphicsInstance>` that
        /// keeps the backend alive, #50). Stored as the raw `u64` so
        /// `execute_graph` can re-stamp it atomically — which queue a graph
        /// lands on (graphics or async compute, #47) is only known at submit.
        semaphore: std::sync::atomic::AtomicU64,
        /// Timeline value this fence is satisfied at. `0` = trivially
        /// signaled (every counter starts at 0); `u64::MAX` = never (created
        /// unsignaled and not yet tied to a submit); otherwise stamped by
        /// `execute_graph` when the submit is enqueued.
        value: std::sync::atomic::AtomicU64,
    },
}

impl std::fmt::Debug for GpuFence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy { signaled } => f
                .debug_struct("GpuFence::Dummy")
                .field("signaled", signaled)
                .finish(),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { state, .. } => f
                .debug_struct("GpuFence::Wgpu")
                .field("state", state)
                .finish_non_exhaustive(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { value, .. } => f
                .debug_struct("GpuFence::Vulkan")
                .field("value", value)
                .finish_non_exhaustive(),
        }
    }
}

/// State of a wgpu fence (see [`GpuFence::Wgpu`]).
///
/// Emulates Vulkan's binary fence semantics on top of wgpu submission
/// indices: `create_fence` picks `Unsignaled` or `Signaled`, and
/// `execute_graph` moves the fence to `Submitted` so waits/polls resolve
/// against the actual GPU submission.
#[cfg(feature = "wgpu-backend")]
#[derive(Debug, Clone)]
pub enum WgpuFenceState {
    /// Created unsignaled; no submission will ever signal it until
    /// `execute_graph` ties one. Polls as unsignaled; waiting is an error.
    Unsignaled,
    /// Created signaled (or trivially complete); polls as signaled.
    Signaled,
    /// Tied to a queue submission. Native waits/polls resolve via
    /// `device.poll(index)`; on all targets the `on_submitted_work_done`
    /// callback flips this to `Signaled` when its `generation` matches.
    Submitted {
        index: wgpu::SubmissionIndex,
        generation: u64,
    },
}

/// Handle to an acquired surface texture for presentation.
///
/// This encapsulates all backend-specific state needed to render to a surface
/// and present the result.
#[allow(clippy::large_enum_variant)]
pub enum GpuSurfaceTexture {
    /// Dummy backend (no GPU texture)
    Dummy,
    /// wgpu backend surface texture
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        /// The raw surface texture (needed for presentation).
        texture: wgpu::SurfaceTexture,
        /// The texture view for rendering.
        view: wgpu_impl::SurfaceTextureView,
    },
    /// Vulkan backend surface texture
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        /// The texture view for rendering.
        view: vulkan::VulkanSurfaceTextureView,
        /// The swapchain image index.
        image_index: u32,
        /// The frame-in-flight index.
        #[allow(dead_code)] // Reserved for future use
        frame_index: usize,
        /// The swapchain handle.
        swapchain: vk::SwapchainKHR,
        /// The image available semaphore.
        image_available_semaphore: vk::Semaphore,
        /// Semaphore signaled by the render submit that writes the swapchain
        /// image (waited on by the present layout-transition submit).
        image_render_finished_semaphore: vk::Semaphore,
        /// The render finished semaphore.
        render_finished_semaphore: vk::Semaphore,
        /// The in-flight fence.
        in_flight_fence: vk::Fence,
        /// The command buffer for presentation.
        present_command_buffer: vk::CommandBuffer,
        /// Whether the acquire reported the swapchain as suboptimal — the
        /// frame is rendered and presented normally, but present reports
        /// `SurfaceOutdated` so the caller reconfigures.
        suboptimal: bool,
    },
}

impl std::fmt::Debug for GpuSurfaceTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuSurfaceTexture::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { .. } => f.debug_struct("GpuSurfaceTexture::Wgpu").finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { image_index, .. } => f
                .debug_struct("GpuSurfaceTexture::Vulkan")
                .field("image_index", image_index)
                .finish_non_exhaustive(),
        }
    }
}

impl GpuSurfaceTexture {
    /// Get the wgpu texture view for rendering (only available with wgpu backend).
    #[cfg(feature = "wgpu-backend")]
    pub fn wgpu_view(&self) -> Option<wgpu_impl::SurfaceTextureView> {
        match self {
            Self::Wgpu { view, .. } => Some(view.clone()),
            _ => None,
        }
    }

    /// Get the wgpu texture view for rendering (stub for non-wgpu builds).
    #[cfg(not(feature = "wgpu-backend"))]
    pub fn wgpu_view(&self) -> Option<()> {
        None
    }

    /// Get the Vulkan texture view for rendering (only available with vulkan backend).
    #[cfg(feature = "vulkan-backend")]
    pub fn vulkan_view(&self) -> Option<vulkan::VulkanSurfaceTextureView> {
        match self {
            Self::Vulkan { view, .. } => Some(view.clone()),
            _ => None,
        }
    }

    /// Get the Vulkan texture view for rendering (stub for non-vulkan builds).
    #[cfg(not(feature = "vulkan-backend"))]
    pub fn vulkan_view(&self) -> Option<()> {
        None
    }

    /// Present the surface texture.
    ///
    /// Takes the backend for Vulkan presentation.
    ///
    /// # Errors
    ///
    /// `SurfaceOutdated` / `SurfaceLost` mean the surface must be
    /// reconfigured before the next acquire (the frame may still have been
    /// shown). Other errors indicate the frame was not presented.
    pub fn present(self, backend: &GpuBackend, frame_index: u64) -> Result<(), GraphicsError> {
        match self {
            Self::Dummy => Ok(()),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { texture, .. } => wgpu_impl::swapchain::present_surface_texture(texture),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan {
                view,
                image_index,
                swapchain,
                image_available_semaphore,
                image_render_finished_semaphore,
                render_finished_semaphore,
                in_flight_fence,
                present_command_buffer,
                suboptimal,
                ..
            } => {
                if let GpuBackend::Vulkan(vulkan_backend) = backend {
                    vulkan::swapchain::present_vulkan_frame(
                        vulkan_backend,
                        &view,
                        swapchain,
                        image_index,
                        image_available_semaphore,
                        image_render_finished_semaphore,
                        render_finished_semaphore,
                        in_flight_fence,
                        present_command_buffer,
                        frame_index,
                    )?;
                    if suboptimal {
                        Err(GraphicsError::SurfaceOutdated)
                    } else {
                        Ok(())
                    }
                } else {
                    Err(GraphicsError::Internal(
                        "Vulkan surface texture requires Vulkan backend for presentation"
                            .to_string(),
                    ))
                }
            }
        }
    }

    /// Release an acquired surface texture that will not be rendered.
    ///
    /// On Vulkan an acquired image carries pending binary-semaphore signals
    /// that can only be drained by presenting, so this presents the
    /// (undrawn) image — one undefined frame on an error path beats
    /// permanently poisoned sync objects. On wgpu, dropping the texture is
    /// the supported cancellation path.
    pub fn abandon(self, backend: &GpuBackend, frame_index: u64) {
        match self {
            Self::Dummy => {}
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { .. } => {}
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { .. } => {
                if let Err(e) = self.present(backend, frame_index) {
                    log::warn!("Failed to release abandoned surface texture: {e}");
                }
            }
        }
    }
}

/// Represents a GPU surface for presentation.
///
/// This encapsulates all backend-specific state needed to create and manage
/// a presentation surface (swapchain).
#[allow(clippy::large_enum_variant)]
pub enum GpuSurface {
    /// Dummy backend (no GPU surface)
    Dummy,
    /// wgpu backend surface
    #[cfg(feature = "wgpu-backend")]
    Wgpu {
        /// The wgpu surface.
        surface: wgpu::Surface<'static>,
    },
    /// Vulkan backend surface
    #[cfg(feature = "vulkan-backend")]
    Vulkan {
        /// The Vulkan surface handle.
        surface: vk::SurfaceKHR,
        /// The Vulkan swapchain (created on configure).
        swapchain: RwLock<Option<vulkan::swapchain::VulkanSwapchain>>,
        /// Surface-extension fn table, kept so `Drop` can destroy the
        /// `VkSurfaceKHR` (the frontend `Surface` keeps the `VkInstance`
        /// alive until after this drops).
        surface_loader: ash::khr::surface::Instance,
        /// Keeps the window alive for as long as the `VkSurfaceKHR` references
        /// it. The `wgpu` variant needs no equivalent — its `Surface<'static>`
        /// owns an `Arc<Window>` clone internally.
        _window: Arc<dyn std::any::Any + Send + Sync>,
    },
}

impl std::fmt::Debug for GpuSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy => write!(f, "GpuSurface::Dummy"),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu { .. } => f.debug_struct("GpuSurface::Wgpu").finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan { surface, .. } => f
                .debug_struct("GpuSurface::Vulkan")
                .field("surface", surface)
                .finish_non_exhaustive(),
        }
    }
}

// GpuSurface is Send + Sync because all variant backends are Send + Sync
unsafe impl Send for GpuSurface {}
unsafe impl Sync for GpuSurface {}

impl GpuSurface {
    /// Query the formats actually supported by this surface from the GPU.
    ///
    /// Returns the list of texture formats the surface can use for presentation,
    /// as reported by the GPU driver.
    pub fn get_supported_formats(&self, backend: &GpuBackend) -> Vec<crate::types::TextureFormat> {
        match (self, backend) {
            (Self::Dummy, _) => {
                vec![
                    crate::types::TextureFormat::Bgra8Unorm,
                    crate::types::TextureFormat::Rgba8Unorm,
                ]
            }

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu { surface }, GpuBackend::Wgpu(wgpu_backend)) => {
                let caps = surface.get_capabilities(wgpu_backend.adapter());
                caps.formats
                    .iter()
                    .filter_map(|f| wgpu_impl::conversion::from_wgpu_texture_format(*f))
                    .collect()
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan { surface, .. }, GpuBackend::Vulkan(vulkan_backend)) => {
                match vulkan_backend.get_surface_formats(*surface) {
                    Ok(formats) => formats
                        .iter()
                        .filter_map(|sf| vulkan::conversion::from_vulkan_format(sf.format))
                        .collect(),
                    Err(e) => {
                        // Report honestly: pretending Bgra8Unorm is supported
                        // lets configuration proceed with a format the
                        // backend will reject (or worse, silently mismatch).
                        log::warn!("Failed to query Vulkan surface formats: {}", e);
                        Vec::new()
                    }
                }
            }

            #[allow(unreachable_patterns)]
            _ => Vec::new(),
        }
    }

    /// Query the present modes actually supported by this surface.
    pub fn get_supported_present_modes(
        &self,
        backend: &GpuBackend,
    ) -> Vec<crate::swapchain::PresentMode> {
        match (self, backend) {
            (Self::Dummy, _) => vec![
                crate::swapchain::PresentMode::Fifo,
                crate::swapchain::PresentMode::Immediate,
                crate::swapchain::PresentMode::Mailbox,
                crate::swapchain::PresentMode::FifoRelaxed,
            ],

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu { surface }, GpuBackend::Wgpu(wgpu_backend)) => {
                let caps = surface.get_capabilities(wgpu_backend.adapter());
                caps.present_modes
                    .iter()
                    .filter_map(|m| wgpu_impl::conversion::from_wgpu_present_mode(*m))
                    .collect()
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan { surface, .. }, GpuBackend::Vulkan(vulkan_backend)) => {
                match vulkan_backend.get_surface_present_modes(*surface) {
                    Ok(modes) => modes
                        .iter()
                        .filter_map(|m| vulkan::conversion::from_vulkan_present_mode(*m))
                        .collect(),
                    Err(e) => {
                        log::warn!("Failed to query Vulkan surface present modes: {}", e);
                        // FIFO is guaranteed by the Vulkan spec.
                        vec![crate::swapchain::PresentMode::Fifo]
                    }
                }
            }

            #[allow(unreachable_patterns)]
            _ => vec![crate::swapchain::PresentMode::Fifo],
        }
    }

    /// Configure the surface for rendering.
    ///
    /// This must be called before acquiring textures. It should also be called
    /// when the window is resized.
    pub fn configure(
        &self,
        backend: &GpuBackend,
        config: &crate::swapchain::SurfaceConfiguration,
    ) -> Result<(), GraphicsError> {
        match (self, backend) {
            (Self::Dummy, _) => {
                log::info!("Configured dummy surface");
                Ok(())
            }

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu { surface }, GpuBackend::Wgpu(wgpu_backend)) => {
                wgpu_impl::swapchain::configure_surface(surface, wgpu_backend, config)
            }

            #[cfg(feature = "vulkan-backend")]
            (
                Self::Vulkan {
                    surface, swapchain, ..
                },
                GpuBackend::Vulkan(vulkan_backend),
            ) => {
                use vulkan::swapchain::VulkanSwapchain;

                // Create the new swapchain BEFORE destroying the old one,
                // passing the old handle so the driver can recreate
                // seamlessly. On failure the old swapchain stays in place —
                // rendering continues on the previous configuration instead
                // of losing the surface entirely.
                let old = swapchain.write().take();
                let old_handle = old
                    .as_ref()
                    .map(|s| s.swapchain)
                    .unwrap_or(vk::SwapchainKHR::null());

                match VulkanSwapchain::new(vulkan_backend, *surface, config, old_handle) {
                    Ok(new_swapchain) => {
                        // The old swapchain is retired by the create call;
                        // destroy its images/sync objects now.
                        if let Some(mut old) = old {
                            old.destroy();
                        }
                        *swapchain.write() = Some(new_swapchain);
                        log::info!("Configured Vulkan swapchain");
                        Ok(())
                    }
                    Err(e) => {
                        *swapchain.write() = old;
                        Err(e)
                    }
                }
            }

            #[allow(unreachable_patterns)]
            _ => Err(GraphicsError::Internal(
                "Surface and backend type mismatch".to_string(),
            )),
        }
    }

    /// Acquire the next texture from the swapchain.
    ///
    /// `view_format` is the engine's render format for the frame — the format the
    /// acquired texture *view* is created as. On WebGPU this is the sRGB view over
    /// a non-sRGB canvas (#33); on native it equals the surface format. Ignored by
    /// backends whose view format is fixed at swapchain creation (Vulkan).
    ///
    /// Returns a backend-specific surface texture for rendering.
    pub fn acquire_texture(
        &self,
        backend: &GpuBackend,
        view_format: crate::types::TextureFormat,
    ) -> Result<GpuSurfaceTexture, GraphicsError> {
        match (self, backend) {
            (Self::Dummy, _) => Ok(GpuSurfaceTexture::Dummy),

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu { surface }, GpuBackend::Wgpu(_)) => {
                let result = wgpu_impl::swapchain::acquire_surface_texture(surface, view_format)?;
                Ok(GpuSurfaceTexture::Wgpu {
                    texture: result.texture,
                    view: result.view,
                })
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan { swapchain, .. }, GpuBackend::Vulkan(vulkan_backend)) => {
                if let Some(ref mut swapchain) = *swapchain.write() {
                    let result = swapchain.acquire_next_image(vulkan_backend)?;
                    Ok(GpuSurfaceTexture::Vulkan {
                        view: result.view,
                        image_index: result.image_index,
                        frame_index: result.frame_index,
                        swapchain: result.swapchain,
                        image_available_semaphore: result.image_available_semaphore,
                        image_render_finished_semaphore: result.image_render_finished_semaphore,
                        render_finished_semaphore: result.render_finished_semaphore,
                        in_flight_fence: result.in_flight_fence,
                        present_command_buffer: result.present_command_buffer,
                        suboptimal: result.suboptimal,
                    })
                } else {
                    Err(GraphicsError::Internal(
                        "Vulkan swapchain not configured".to_string(),
                    ))
                }
            }

            #[allow(unreachable_patterns)]
            _ => Err(GraphicsError::Internal(
                "Surface and backend type mismatch".to_string(),
            )),
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuSurface {
    fn drop(&mut self) {
        if let GpuSurface::Vulkan {
            surface,
            swapchain,
            surface_loader,
            ..
        } = self
        {
            // Destroy the swapchain before the surface it was created from.
            // The VulkanSwapchain stores its own device handles for cleanup.
            if let Some(ref mut sc) = *swapchain.write() {
                sc.destroy();
            }
            // SAFETY: the frontend Surface keeps the instance (and thus the
            // VkInstance behind `surface_loader`) alive until after this Drop.
            if *surface != vk::SurfaceKHR::null() {
                unsafe { surface_loader.destroy_surface(*surface, None) };
            }
        }
    }
}

// ============================================================================
// Vulkan Resource Cleanup (Drop implementations)
// ============================================================================
//
// These Drop implementations destroy Vulkan resources directly.
// Safety: Resources are kept alive by Arc references in render graphs until
// FramePipeline::begin_frame() resets them after the fence wait, which
// guarantees the GPU has finished using them.

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let GpuBuffer::Vulkan {
            device,
            buffer,
            allocation,
            allocator,
            retired,
            ..
        } = self
        {
            if let Some(alloc) = allocation.lock().take()
                && let Err(e) = allocator.lock().free(alloc)
            {
                log::error!("Failed to free buffer allocation: {}", e);
            }
            retired.lock().buffers.push(buffer.as_raw());
            unsafe { device.destroy_buffer(*buffer, None) };
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuTexture {
    fn drop(&mut self) {
        if let GpuTexture::Vulkan {
            device,
            image,
            view,
            allocation,
            allocator,
            id,
            retired,
            ..
        } = self
        {
            if let Some(alloc) = allocation.lock().take()
                && let Err(e) = allocator.lock().free(alloc)
            {
                log::error!("Failed to free texture allocation: {}", e);
            }
            retired.lock().textures.push(*id);
            unsafe {
                device.destroy_image_view(*view, None);
                device.destroy_image(*image, None);
            }
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuSampler {
    fn drop(&mut self) {
        if let GpuSampler::Vulkan { device, sampler } = self {
            unsafe { device.destroy_sampler(*sampler, None) };
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuBindingGroup {
    fn drop(&mut self) {
        if let GpuBindingGroup::Vulkan {
            device,
            descriptor_set,
            pool,
            pools,
        } = self
        {
            // Free the set directly, no deferred queue. Safety: submitted render
            // graphs retain their `Arc<MaterialInstance>` → `Arc<BindingGroup>`
            // (and thus this handle) until `FramePipeline` recycles the slot
            // after its fence wait, so the GPU is guaranteed done with the set
            // before this runs — the same invariant that makes `GpuBuffer`'s
            // immediate destroy-on-Drop safe. The shared pool mutex provides the
            // external synchronization `vkFreeDescriptorSets` requires.
            let _guard = pools.lock();
            unsafe {
                if let Err(e) = device.free_descriptor_sets(*pool, &[*descriptor_set]) {
                    log::error!("Failed to free descriptor set: {e:?}");
                }
            }
        }
    }
}

#[cfg(feature = "vulkan-backend")]
impl Drop for GpuPipeline {
    fn drop(&mut self) {
        if let GpuPipeline::Vulkan {
            device,
            pipeline,
            pipeline_layout,
            descriptor_set_layouts: _,
        } = self
        {
            // descriptor_set_layouts are NOT destroyed here: they are shared
            // handles owned by the PipelineManager's dedup cache (other
            // materials may still reference them). The manager destroys them
            // in its destroy().
            unsafe {
                device.destroy_pipeline(*pipeline, None);
                device.destroy_pipeline_layout(*pipeline_layout, None);
            }
        }
    }
}

// ============================================================================
// GPU Backend Enum
// ============================================================================

/// GPU backend enum for abstracting different GPU APIs.
///
/// Unlike a trait-based approach, this enum allows for static dispatch
/// and avoids the overhead of dynamic dispatch.
#[allow(clippy::large_enum_variant)]
pub enum GpuBackend {
    /// Dummy backend for testing and development.
    Dummy(dummy::DummyBackend),
    /// wgpu backend for cross-platform GPU access.
    #[cfg(feature = "wgpu-backend")]
    Wgpu(wgpu_impl::WgpuBackend),
    /// Native Vulkan backend using ash.
    #[cfg(feature = "vulkan-backend")]
    Vulkan(vulkan::VulkanBackend),
}

impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dummy(backend) => f.debug_tuple("GpuBackend::Dummy").field(backend).finish(),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => f.debug_tuple("GpuBackend::Wgpu").field(backend).finish(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => f.debug_tuple("GpuBackend::Vulkan").field(backend).finish(),
        }
    }
}

// GpuBackend is Send + Sync because all variant backends are Send + Sync
unsafe impl Send for GpuBackend {}
unsafe impl Sync for GpuBackend {}

impl GpuBackend {
    /// Get the backend name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dummy(backend) => backend.name(),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.name(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.name(),
        }
    }

    /// Capabilities of the adapter this backend was created on, from real
    /// queries at backend creation (ADR-027) — never fabricated defaults.
    pub fn capabilities(&self) -> crate::device::DeviceCapabilities {
        match self {
            Self::Dummy(backend) => backend.capabilities(),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.capabilities(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.capabilities(),
        }
    }

    /// Information about the adapter this backend was created on.
    pub fn adapter_info(&self) -> crate::instance::AdapterInfo {
        match self {
            Self::Dummy(backend) => backend.adapter_info(),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.adapter_info(),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.adapter_info(),
        }
    }

    /// Create a buffer resource.
    pub fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<GpuBuffer, GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.create_buffer(descriptor),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_buffer(descriptor),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_buffer(descriptor),
        }
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.create_texture(descriptor),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_texture(descriptor),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_texture(descriptor),
        }
    }

    /// Create a sampler resource.
    pub fn create_sampler(
        &self,
        descriptor: &SamplerDescriptor,
    ) -> Result<GpuSampler, GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.create_sampler(descriptor),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_sampler(descriptor),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_sampler(descriptor),
        }
    }

    /// Create a pipeline (graphics or compute) from a material descriptor.
    pub fn create_pipeline(
        &self,
        descriptor: &crate::materials::MaterialDescriptor,
    ) -> Result<GpuPipeline, GraphicsError> {
        match self {
            Self::Dummy(_) => Ok(GpuPipeline::Dummy),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_pipeline(descriptor),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_pipeline(descriptor),
        }
    }

    /// Create a binding group: the eagerly-compiled descriptor set / bind group
    /// for a [`BindingGroupDescriptor`](crate::BindingGroupDescriptor) against
    /// `layout`. The GPU handle is written once here and only bound at draw time.
    pub fn create_binding_group(
        &self,
        layout: &crate::materials::BindingLayout,
        descriptor: &crate::materials::BindingGroupDescriptor,
    ) -> Result<GpuBindingGroup, GraphicsError> {
        match self {
            Self::Dummy(_) => Ok(GpuBindingGroup::Dummy),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_binding_group(layout, descriptor),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_binding_group(layout, descriptor),
        }
    }

    /// Create a fence for CPU-GPU synchronization.
    pub fn create_fence(&self, signaled: bool) -> Result<GpuFence, GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.create_fence(signaled),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.create_fence(signaled),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.create_fence(signaled),
        }
    }

    /// Wait for a fence to be signaled.
    ///
    /// Uses a backend-internal timeout (10 s) to avoid hanging forever on a
    /// hung GPU. Returns an error on timeout or device loss — the caller must
    /// NOT recycle resources guarded by this fence in that case, because the
    /// GPU may still be using them.
    pub fn wait_fence(&self, fence: &GpuFence) -> Result<(), GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.wait_fence(fence),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.wait_fence(fence),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.wait_fence(fence),
        }
    }

    /// Wait for a fence to be signaled with a timeout.
    ///
    /// Returns `Ok(true)` if the fence was signaled, `Ok(false)` if the
    /// timeout elapsed, and an error on device loss or wait failure.
    pub fn wait_fence_timeout(
        &self,
        fence: &GpuFence,
        timeout: std::time::Duration,
    ) -> Result<bool, GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.wait_fence_timeout(fence, timeout),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.wait_fence_timeout(fence, timeout),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.wait_fence_timeout(fence, timeout),
        }
    }

    /// Check if a fence is signaled (non-blocking).
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        match self {
            Self::Dummy(backend) => backend.is_fence_signaled(fence),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.is_fence_signaled(fence),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.is_fence_signaled(fence),
        }
    }

    /// Signal a fence (for testing/dummy backend).
    pub fn signal_fence(&self, fence: &GpuFence) {
        match self {
            Self::Dummy(backend) => backend.signal_fence(fence),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.signal_fence(fence),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.signal_fence(fence),
        }
    }

    /// Execute a compiled render graph.
    ///
    /// This records commands from the graph into a command buffer and submits it.
    ///
    /// # Synchronization contract
    ///
    /// * `signal_fence: Some(_)` — returns immediately after submission on
    ///   every backend; the fence is signaled when the GPU finishes, and the
    ///   caller waits/polls it before recycling frame resources.
    /// * `signal_fence: None` — **fire-and-forget with no CPU-side completion
    ///   guarantee.** Backends may return immediately after submission
    ///   (Vulkan) or block conservatively (wgpu today) — callers must not
    ///   rely on either behavior. If you need to know when the work is done,
    ///   pass a fence.
    ///
    /// One render graph is submitted per frame, so there is no cross-graph
    /// semaphore ordering. Swapchain acquire/present synchronization is handled
    /// internally by the backend.
    pub fn execute_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.execute_graph(graph, compiled, signal_fence),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.execute_graph(graph, compiled, signal_fence),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.execute_graph(graph, compiled, signal_fence),
        }
    }

    /// Low-level host write into a buffer's memory.
    ///
    /// This is the primitive behind [`RingBuffer`](crate::RingBuffer)-style
    /// streaming and is intentionally weak: the caller must guarantee that no
    /// in-flight GPU work reads the written region. Timing differs per
    /// backend — Vulkan writes the mapped memory immediately (visible even to
    /// already-submitted work), wgpu stages the write and applies it at the
    /// start of the next submission. Both satisfy the ring-buffer contract
    /// (fresh region each frame); nothing else should rely on this method.
    ///
    /// Host-visible buffers only: on Vulkan a device-local destination is an
    /// error (#89 — the blocking one-shot staging path was removed). Every
    /// GPU-only upload goes through
    /// [`TransferOperation::WriteBuffer`](crate::TransferOperation::WriteBuffer),
    /// which copies at the transfer pass's position on both backends.
    pub fn write_buffer(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.write_buffer(buffer, offset, data),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.write_buffer(buffer, offset, data),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.write_buffer(buffer, offset, data),
        }
    }

    /// Read a host-visible buffer's mapped memory into a `Vec`.
    ///
    /// Contract (identical on both backends): the buffer must be host-readable
    /// (`BufferUsage::MAP_READ`) and the caller must ensure the GPU has
    /// finished writing it — read *after* the frame fence (this is what the
    /// pipeline's post-fence readback drain does). This method performs **no**
    /// GPU wait and **no** staging copy of its own: reading a device-local
    /// buffer is an error, not a silent zero-fill — copy it to a readback
    /// buffer with [`TransferOperation::ReadbackBuffer`](crate::TransferOperation)
    /// first.
    pub fn read_buffer(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, GraphicsError> {
        match self {
            Self::Dummy(backend) => backend.read_buffer(buffer, offset, size),
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => backend.read_buffer(buffer, offset, size),
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => backend.read_buffer(buffer, offset, size),
        }
    }

    /// Non-blocking readback: begin mapping `buffer[offset..offset+size]` and,
    /// when the map resolves, copy it into `dst` and clear `map_pending`.
    ///
    /// This is the wasm-safe readback: on wgpu the copy happens in the
    /// `map_async` completion callback (a browser thread cannot block for the
    /// map, #33); native backends fill `dst` synchronously (the caller drains
    /// after the frame fence, so the GPU is already done). Fire-and-forget:
    /// failures are logged and `map_pending` is always cleared so the buffer
    /// isn't wedged as permanently in-flight.
    pub fn read_buffer_async(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        size: u64,
        dst: Arc<std::sync::Mutex<Vec<u8>>>,
        map_pending: Arc<std::sync::atomic::AtomicBool>,
    ) {
        match self {
            Self::Dummy(backend) => {
                backend.read_buffer_async(buffer, offset, size, dst, map_pending)
            }
            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(backend) => {
                backend.read_buffer_async(buffer, offset, size, dst, map_pending)
            }
            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(backend) => {
                backend.read_buffer_async(buffer, offset, size, dst, map_pending)
            }
        }
    }

    /// Create a surface from an owned window.
    ///
    /// Takes an `Arc<W>` so the window's lifetime is genuinely tied to the
    /// surface: `wgpu`'s `Surface<'static>` is created from the owned `Arc`
    /// (no lifetime-erasing `transmute`), and the Vulkan variant keeps a clone
    /// alive for as long as the `VkSurfaceKHR`. Dropping the window while the
    /// surface lives is therefore impossible.
    pub fn create_surface<W>(&self, window: Arc<W>) -> Result<GpuSurface, GraphicsError>
    where
        W: raw_window_handle::HasWindowHandle
            + raw_window_handle::HasDisplayHandle
            + Send
            + Sync
            + 'static,
    {
        match self {
            Self::Dummy(_) => {
                log::info!("Created dummy surface");
                Ok(GpuSurface::Dummy)
            }

            #[cfg(feature = "wgpu-backend")]
            Self::Wgpu(wgpu_backend) => {
                // The owned Arc converts into wgpu's `SurfaceTarget<'static>`
                // (it is `HasWindowHandle + HasDisplayHandle + Send + Sync +
                // 'static`), so wgpu returns a genuine `Surface<'static>` that
                // holds the window itself — no `transmute`.
                let surface = wgpu_backend
                    .instance()
                    .create_surface(window)
                    .map_err(|e| {
                        GraphicsError::ResourceCreationFailed(format!(
                            "Failed to create wgpu surface: {e}"
                        ))
                    })?;
                log::info!("Created wgpu surface");
                Ok(GpuSurface::Wgpu { surface })
            }

            #[cfg(feature = "vulkan-backend")]
            Self::Vulkan(vulkan_backend) => {
                // A headless instance has no surface extensions enabled;
                // fail with a clear message instead of calling unloaded
                // surface entry points.
                if !vulkan_backend.surface_support() {
                    return Err(GraphicsError::ResourceCreationFailed(
                        "Vulkan instance has no surface support (headless loader); \
                         cannot create a window surface"
                            .to_string(),
                    ));
                }
                // Create Vulkan surface from window
                let display_handle = window.display_handle().map_err(|e| {
                    GraphicsError::ResourceCreationFailed(format!(
                        "Failed to get display handle: {e}"
                    ))
                })?;
                let window_handle = window.window_handle().map_err(|e| {
                    GraphicsError::ResourceCreationFailed(format!(
                        "Failed to get window handle: {e}"
                    ))
                })?;

                let surface = unsafe {
                    ash_window::create_surface(
                        vulkan_backend.entry(),
                        vulkan_backend.instance(),
                        display_handle.as_raw(),
                        window_handle.as_raw(),
                        None,
                    )
                }
                .map_err(|e| {
                    GraphicsError::ResourceCreationFailed(format!(
                        "Failed to create Vulkan surface: {e}"
                    ))
                })?;
                log::info!("Created Vulkan surface");
                Ok(GpuSurface::Vulkan {
                    surface,
                    swapchain: RwLock::new(None),
                    surface_loader: vulkan_backend.surface_loader().clone(),
                    _window: window,
                })
            }
        }
    }

    /// Ensure the backend is compatible with the given surface.
    ///
    /// For wgpu backend, this may re-request an adapter if the current one is not
    /// compatible with the surface. For Vulkan backend, this verifies the physical
    /// device supports the surface.
    ///
    /// Returns `Ok(true)` if compatible, `Err` if no compatible adapter could be found.
    /// Whether the backend's current adapter can present to `surface`,
    /// without changing any state.
    pub fn surface_compatible(&self, surface: &GpuSurface) -> bool {
        match (self, surface) {
            (Self::Dummy(_), _) => true,

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu(wgpu_backend), GpuSurface::Wgpu { surface }) => {
                wgpu_backend.is_adapter_compatible_with_surface(surface)
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan(vulkan_backend), GpuSurface::Vulkan { surface, .. }) => {
                vulkan_backend.is_surface_supported(*surface)
            }

            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    pub fn ensure_compatible_with_surface(
        &mut self,
        surface: &GpuSurface,
    ) -> Result<bool, GraphicsError> {
        match (self, surface) {
            (Self::Dummy(_), _) => {
                // Dummy backend is always "compatible"
                Ok(true)
            }

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu(wgpu_backend), GpuSurface::Wgpu { surface }) => {
                wgpu_backend.ensure_compatible_with_surface(surface)
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan(vulkan_backend), GpuSurface::Vulkan { surface, .. }) => {
                if vulkan_backend.is_surface_supported(*surface) {
                    Ok(true)
                } else {
                    Err(GraphicsError::ResourceCreationFailed(
                        "Vulkan physical device does not support presentation to this surface"
                            .to_string(),
                    ))
                }
            }

            #[allow(unreachable_patterns)]
            _ => Err(GraphicsError::Internal(
                "Surface and backend type mismatch".to_string(),
            )),
        }
    }

    /// Check if the backend is compatible with the given surface without modifying it.
    pub fn is_compatible_with_surface(&self, surface: &GpuSurface) -> bool {
        match (self, surface) {
            (Self::Dummy(_), _) => true,

            #[cfg(feature = "wgpu-backend")]
            (Self::Wgpu(wgpu_backend), GpuSurface::Wgpu { surface }) => {
                wgpu_backend.is_adapter_compatible_with_surface(surface)
            }

            #[cfg(feature = "vulkan-backend")]
            (Self::Vulkan(vulkan_backend), GpuSurface::Vulkan { surface, .. }) => {
                vulkan_backend.is_surface_supported(*surface)
            }

            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

/// Selects and creates the appropriate backend based on available features.
///
/// This uses default parameters (auto-select best backend).
pub fn create_backend() -> Result<GpuBackend, GraphicsError> {
    create_backend_with_params(&crate::instance::InstanceParameters::default())
}

/// Serializes GPU backend creation process-wide (#93).
///
/// NVIDIA's Windows drivers (observed via wgpu/DX12 on an RTX 3070)
/// intermittently abort the process (exception 0x87D raised through
/// `RaiseException`) when many devices are created and torn down
/// concurrently — the pattern parallel unit tests produce. Real applications
/// create one device once, so this lock is uncontended in production. The
/// Vulkan backend's Drop takes the same lock so teardown is serialized
/// against creation as well.
pub(crate) static BACKEND_LIFECYCLE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Selects and creates the appropriate backend based on parameters.
pub fn create_backend_with_params(
    params: &crate::instance::InstanceParameters,
) -> Result<GpuBackend, GraphicsError> {
    use crate::instance::BackendType;

    let _lifecycle = BACKEND_LIFECYCLE_LOCK.lock();

    match params.backend {
        BackendType::Auto => create_backend_auto(params),
        BackendType::Dummy => {
            log::info!("Using dummy backend (requested)");
            Ok(GpuBackend::Dummy(dummy::DummyBackend::new()))
        }
        BackendType::Wgpu => {
            #[cfg(feature = "wgpu-backend")]
            {
                let backend = wgpu_impl::WgpuBackend::with_params(params)?;
                log::info!("Using wgpu backend (requested)");
                Ok(GpuBackend::Wgpu(backend))
            }
            #[cfg(not(feature = "wgpu-backend"))]
            {
                Err(GraphicsError::ResourceCreationFailed(
                    "wgpu backend requested but wgpu-backend feature is not enabled".to_string(),
                ))
            }
        }
        BackendType::Vulkan => {
            #[cfg(feature = "vulkan-backend")]
            {
                let backend = vulkan::VulkanBackend::with_params(params)?;
                log::info!("Using Vulkan backend (requested)");
                Ok(GpuBackend::Vulkan(backend))
            }
            #[cfg(not(feature = "vulkan-backend"))]
            {
                Err(GraphicsError::ResourceCreationFailed(
                    "Vulkan backend requested but vulkan-backend feature is not enabled"
                        .to_string(),
                ))
            }
        }
    }
}

/// Async variant of [`create_backend_with_params`] for targets that cannot block
/// on device creation (wasm/WebGPU, #33). Covers the Dummy and wgpu backends; the
/// native Vulkan backend keeps its blocking constructor (it is never used on the
/// async/web path).
#[cfg(feature = "wgpu-backend")]
pub async fn create_backend_with_params_async(
    params: &crate::instance::InstanceParameters,
) -> Result<GpuBackend, GraphicsError> {
    use crate::instance::BackendType;

    match params.backend {
        BackendType::Dummy => {
            log::info!("Using dummy backend (requested)");
            Ok(GpuBackend::Dummy(dummy::DummyBackend::new()))
        }
        BackendType::Auto | BackendType::Wgpu => {
            let backend = wgpu_impl::WgpuBackend::with_params_async(params).await?;
            log::info!("Using wgpu backend (async)");
            Ok(GpuBackend::Wgpu(backend))
        }
        #[cfg(feature = "vulkan-backend")]
        BackendType::Vulkan => {
            let backend = vulkan::VulkanBackend::with_params(params)?;
            log::info!("Using Vulkan backend (requested)");
            Ok(GpuBackend::Vulkan(backend))
        }
        #[cfg(not(feature = "vulkan-backend"))]
        BackendType::Vulkan => Err(GraphicsError::ResourceCreationFailed(
            "Vulkan backend requested but vulkan-backend feature is not enabled".to_string(),
        )),
    }
}

/// Auto-select the best available backend.
fn create_backend_auto(
    params: &crate::instance::InstanceParameters,
) -> Result<GpuBackend, GraphicsError> {
    // Try wgpu backend first if available (supports WGSL shaders and full draw commands)
    #[cfg(feature = "wgpu-backend")]
    {
        match wgpu_impl::WgpuBackend::with_params(params) {
            Ok(backend) => {
                log::info!("Using wgpu backend");
                return Ok(GpuBackend::Wgpu(backend));
            }
            Err(e) => {
                log::warn!("Failed to create wgpu backend: {}", e);
            }
        }
    }

    // Try Vulkan backend if wgpu unavailable (native Vulkan via ash)
    #[cfg(feature = "vulkan-backend")]
    {
        match vulkan::VulkanBackend::with_params(params) {
            Ok(backend) => {
                log::info!("Using Vulkan backend (ash)");
                return Ok(GpuBackend::Vulkan(backend));
            }
            Err(e) => {
                log::warn!("Failed to create Vulkan backend: {}", e);
            }
        }
    }

    // Fall back to dummy backend
    log::info!("Using dummy backend");
    Ok(GpuBackend::Dummy(dummy::DummyBackend::new()))
}

/// Check if a real GPU backend is available.
pub fn has_gpu_backend() -> bool {
    cfg!(any(feature = "vulkan-backend", feature = "wgpu-backend"))
}
