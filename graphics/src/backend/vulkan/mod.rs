//! Native Vulkan backend implementation using ash.
//!
//! This backend provides direct Vulkan access for maximum performance and control.
//! It includes support for validation layers in debug builds.

mod accel;
mod allocator;
pub mod barriers;
mod breadcrumbs;
mod command;
mod compaction;
pub(crate) mod conversion;
mod debug;
pub use accel::AccelerationStructureKind;
pub use debug::{reset_validation_error_count, validation_error_count};
mod device;
mod instance;
pub mod layout;
mod maintenance9;
mod memory;
mod pipeline;
mod staging;
pub mod swapchain;
mod timestamps;

use std::collections::HashSet;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use ash::vk;
use ash::vk::Handle as _;
use gpu_allocator::vulkan::Allocator;
use parking_lot::Mutex;

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, Pass, RenderGraph, RenderTarget};
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};
use redlilium_core::profiling::profile_scope;

use super::{GpuAccelerationStructure, GpuBuffer, GpuFence, GpuSampler, GpuTexture};

/// Maximum number of frames in flight for per-slot resource tracking.
///
/// The engine-wide bound lives in [`crate::pipeline`]; this backend sizes its
/// per-slot arrays (command pools, staging) to it, which is why exceeding it
/// is enforced at [`FramePipeline`](crate::pipeline::FramePipeline) creation
/// (VK-M9) — a fourth in-flight frame would reset a pool whose command
/// buffers are still executing.
pub use crate::pipeline::MAX_FRAMES_IN_FLIGHT;

/// Upper bound for every CPU-side GPU wait (fences, swapchain acquire).
///
/// A hung GPU must surface as a [`GraphicsError::Timeout`] instead of
/// freezing the process forever; 10 s is far beyond any legitimate frame.
pub(crate) const FENCE_WAIT_TIMEOUT_NS: u64 = 10_000_000_000;

/// Process-wide source of stable texture ids for the layout tracker. Monotonic
/// so a destroyed texture's id is never reused (no layout aliasing).
static NEXT_TEXTURE_ID: AtomicU64 = AtomicU64::new(1);
pub use layout::{TextureLayout, TextureLayoutTracker, TextureUsageGraph};
pub use pipeline::PersistentDescriptorPools;

/// Handles of destroyed resources awaiting removal from the barrier trackers.
///
/// `GpuTexture`/`GpuBuffer` drops push their tracker keys here (they can't
/// touch the trackers directly — drops happen wherever the last `Arc` dies);
/// [`VulkanBackend::advance_frame`] drains the queue and removes the entries,
/// so tracker maps stay bounded by the number of live resources instead of
/// growing with every texture/buffer ever created.
#[derive(Debug, Default)]
pub struct RetiredTrackerHandles {
    /// Stable texture ids (the layout-tracker keys).
    pub textures: Vec<u64>,
    /// Raw `vk::Buffer` handles (the access-tracker keys).
    pub buffers: Vec<u64>,
}

use self::barriers::{BarrierBatch, BufferAccessTracker, BufferId, QueueId, SubmitWaits};
use self::layout::TextureId;

/// Scratch buffers reused across draw commands to avoid per-draw heap allocations.
///
/// Contains only plain value types without Rust lifetimes. Vecs are cleared
/// between draws but retain their capacity across frames.
#[derive(Default)]
struct VulkanEncoderScratch {
    /// Cached descriptor sets collected per draw/dispatch for
    /// `cmd_bind_descriptor_sets`. Reused across draws to avoid per-draw heap
    /// allocation. (Binding groups are now created eagerly, so there is no
    /// per-draw allocation/write scratch.)
    descriptor_sets: Vec<vk::DescriptorSet>,
}

/// Whether two binding layouts are descriptor-set-layout compatible: identical
/// bindings, in order, with matching type and stage visibility. A binding group
/// created against a layout compatible with the material's set layout binds
/// correctly (content-equal layouts share the same `VkDescriptorSetLayout` via
/// the pipeline manager's dedup).
fn binding_layouts_compatible(
    a: &crate::materials::BindingLayout,
    b: &crate::materials::BindingLayout,
) -> bool {
    a.entries.len() == b.entries.len()
        && a.entries.iter().zip(&b.entries).all(|(x, y)| {
            x.binding == y.binding
                && x.binding_type == y.binding_type
                && x.visibility == y.visibility
        })
}

/// A texture view for a Vulkan surface texture (swapchain image).
///
/// This wraps the Vulkan image view from the swapchain for use in render passes.
#[derive(Clone)]
pub struct VulkanSurfaceTextureView {
    pub(crate) image: vk::Image,
    pub(crate) view: Arc<VulkanImageView>,
}

/// Wrapper for a Vulkan image view that handles cleanup.
pub struct VulkanImageView {
    #[allow(dead_code)] // Reserved for cleanup when needed
    device: ash::Device,
    view: vk::ImageView,
}

impl VulkanImageView {
    /// Create a new VulkanImageView wrapper.
    pub(crate) fn new(device: ash::Device, view: vk::ImageView) -> Self {
        Self { device, view }
    }

    /// Get the raw Vulkan image view handle.
    pub fn view(&self) -> vk::ImageView {
        self.view
    }
}

impl Drop for VulkanImageView {
    fn drop(&mut self) {
        // Note: We don't destroy the view here because swapchain image views
        // are managed by the swapchain. Only destroy views we created ourselves.
    }
}

impl std::fmt::Debug for VulkanSurfaceTextureView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanSurfaceTextureView").finish()
    }
}

impl VulkanSurfaceTextureView {
    /// Get the underlying Vulkan image.
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Get the underlying Vulkan image view.
    pub fn view(&self) -> vk::ImageView {
        self.view.view()
    }
}

use self::conversion::{
    convert_address_mode, convert_buffer_usage, convert_compare_function, convert_filter_mode,
    convert_mipmap_filter_mode, convert_texture_format, convert_texture_usage,
};

/// Vulkan-based GPU backend using ash.
///
/// This backend provides native Vulkan access with:
/// - Validation layers enabled in debug builds
/// - gpu-allocator for memory management
/// - Dynamic rendering (VK_KHR_dynamic_rendering)
/// - Deferred resource destruction for safe GPU resource management
pub struct VulkanBackend {
    /// Vulkan entry points (function loader).
    entry: ash::Entry,
    /// Vulkan instance.
    instance: ash::Instance,
    /// Debug messenger for validation layer output.
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Debug utils extension instance.
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    /// Debug-utils **device** functions (#123): `vkSetDebugUtilsObjectNameEXT`
    /// for naming GPU objects and `vkCmd{Begin,End}DebugUtilsLabelEXT` for
    /// per-pass label regions. `Some` whenever the instance enabled
    /// `VK_EXT_debug_utils` (RenderDoc injects it even without validation);
    /// every naming/label call is a no-op when `None`.
    debug_utils_device: Option<ash::ext::debug_utils::Device>,
    /// Selected physical device.
    physical_device: vk::PhysicalDevice,
    /// Resolved Vulkan format for `TextureFormat::Depth24PlusStencil8`:
    /// `D24_UNORM_S8_UINT` where the device supports it as a depth-stencil
    /// attachment, `D32_SFLOAT_S8_UINT` otherwise (optional on e.g. AMD).
    depth24_stencil8_format: vk::Format,
    /// Logical device.
    device: ash::Device,
    /// Graphics queue.
    graphics_queue: vk::Queue,
    /// Graphics queue family index.
    graphics_queue_family: u32,
    /// Memory allocator (wrapped in Arc for sharing with GPU resource Drop impls).
    ///
    /// Wrapped in `ManuallyDrop` so it can be dropped explicitly in
    /// [`Drop::drop`] *before* the logical device is destroyed — gpu-allocator
    /// frees its pooled memory blocks using the device when the `Allocator` is
    /// dropped, which would be UB against an already-destroyed device.
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    /// Command pool for graphics operations.
    ///
    /// INVARIANT: Vulkan requires the pool (and recording into its buffers) to
    /// be externally synchronized. All users — `execute_graph`, `write_buffer`,
    /// swapchain present, and `advance_frame` — run on the single render thread
    /// that drives `&mut FrameSchedule`, so accesses are serialized by that
    /// ownership. If GPU submission/upload is ever moved off that thread (e.g.
    /// async texture uploads), this pool must become per-thread/per-frame or be
    /// guarded by a mutex, otherwise concurrent alloc/free/reset/record is UB.
    command_pool: vk::CommandPool,
    /// Per-frame swapchain synchronization, set on `acquire_next_image` and
    /// consumed by the render submit that writes the swapchain image so that
    /// submit waits on `image_available` and signals `image_render_finished`.
    swapchain_sync: Mutex<SwapchainSync>,
    /// Whether validation layers are enabled.
    #[allow(dead_code)]
    validation_enabled: bool,
    /// Surface extension.
    surface_loader: ash::khr::surface::Instance,
    /// Swapchain extension.
    swapchain_loader: ash::khr::swapchain::Device,
    /// Per-frame-slot command pools for the render-graph submit. Each slot's
    /// pool is reset wholesale (`vkResetCommandPool`) once its fence signals,
    /// which frees all of that frame's command buffers at once (cheaper than
    /// per-buffer freeing and isolates frames from each other). Staging uploads
    /// and swapchain present use the separate long-lived `command_pool`.
    frame_command_pools: [vk::CommandPool; MAX_FRAMES_IN_FLIGHT],
    /// Current frame slot index for command buffer tracking.
    current_slot: AtomicUsize,
    /// Monotonic frame counter, incremented once per
    /// [`advance_frame`](Self::advance_frame). Unlike `current_slot` it never
    /// wraps, so it drives the multi-frame BLAS-compaction lifecycle (#110
    /// phase 3): work submitted in frame `N` is guaranteed complete by the time
    /// this reaches `N + MAX_FRAMES_IN_FLIGHT`.
    frame_index: AtomicU64,
    /// Layout tracker for automatic barrier placement.
    /// Uses interior mutability since execute_graph takes &self.
    layout_tracker: Mutex<TextureLayoutTracker>,
    /// Per-buffer last-access tracker for precise buffer barriers
    /// (mirrors `layout_tracker` for buffers).
    buffer_tracker: Mutex<BufferAccessTracker>,
    /// Tracker keys of destroyed textures/buffers, pushed by resource drops
    /// and drained in [`advance_frame`](Self::advance_frame) so the tracker
    /// maps don't grow with every resource ever created.
    retired_tracker_handles: Arc<Mutex<RetiredTrackerHandles>>,
    /// Pipeline manager for shader compilation and pipeline creation.
    pipeline_manager: pipeline::PipelineManager,
    /// Scratch buffers for allocation reuse during pass encoding.
    encoder_scratch: Mutex<VulkanEncoderScratch>,
    /// Pooled staging memory for frame-graph uploads (`WriteBuffer` transfer
    /// ops). Chunks written during a frame stay owned by its slot and return
    /// to the pool in [`advance_frame`](Self::advance_frame) once the slot's
    /// fence has signalled, so the GPU is guaranteed to be done with them.
    staging_belt: Mutex<staging::StagingBelt>,
    /// Timeline semaphore for the graphics queue: every `execute_graph`
    /// submit signals the next monotonically increasing value on it. Frame
    /// fences are (this semaphore, value) pairs — see [`GpuFence::Vulkan`].
    /// One timeline per queue; the async compute queue has its own.
    queue_timeline: vk::Semaphore,
    /// Next timeline value to signal (starts at 1; the semaphore's counter
    /// starts at 0, so a fence carrying value 0 reads as already signaled).
    timeline_next: AtomicU64,
    /// The async compute queue, when the device exposes one (#47 phase 4).
    /// `None` is the first-class single-queue mode: everything runs on the
    /// graphics queue exactly as before.
    async_compute: Option<SecondaryQueue>,
    /// The dedicated transfer queue — DMA engines — when the device exposes
    /// a transfer-only family (#89). `None` is first-class: transfer graphs
    /// fall back to async compute, then graphics.
    transfer_queue: Option<SecondaryQueue>,
    /// `minImageTransferGranularity` of the transfer family (`Some` exactly
    /// when [`transfer_queue`](Self::transfer_queue) is). A coarse granularity
    /// (AMD SDMA: 16×16×8) restricts which image copies may be routed to the
    /// transfer queue — buffer copies and whole-subresource image copies are
    /// legal at any granularity; anything else falls back (#92).
    transfer_granularity: Option<vk::Extent3D>,
    /// Stable ids of textures already warned about declining the async
    /// compute hint (#88) — the fallback is correct but silently serializes
    /// work the caller wanted overlapped, so it warns once per texture, not
    /// per frame. Entries are dropped when the texture retires (same drain
    /// as the trackers in [`advance_frame`](Self::advance_frame)).
    async_decline_warned: Mutex<HashSet<u64>>,
    /// maintenance9 image fast path (#88): true when the extension is
    /// enabled AND the graphics and async compute families may mutually
    /// implicitly acquire each other's optimal-tiling images. Declared
    /// cross-queue textures are then created EXCLUSIVE (keeping framebuffer
    /// compression) with no ownership transfers — synchronization is still
    /// the tracker-emitted timeline waits, unchanged.
    implicit_cross_queue_images: bool,
    /// Per-pass GPU timestamp collector (#95), `None` when the device lacks
    /// `timestampValidBits` on the graphics family. Behind a `Mutex` for the
    /// same reason as the other per-frame state: `execute_graph`/`advance_frame`
    /// take `&self` but run on the single render thread.
    timestamps: Option<Mutex<timestamps::TimestampManager>>,
    /// GPU-memory statistics (#98), re-sampled once per frame in
    /// [`advance_frame`](Self::advance_frame) — driver heap budgets
    /// (`VK_EXT_memory_budget`, when enabled) plus allocator totals. Behind a
    /// `Mutex` because the sample runs on the render thread while
    /// [`latest_memory_stats`](Self::latest_memory_stats) may read from the UI
    /// thread. Budget/usage are deliberately not cached across frames (the spec
    /// allows them to change any moment); the once-per-frame sample is the cache.
    memory_stats: Mutex<crate::device::GpuMemoryStats>,
    /// Per-pass GPU crash breadcrumbs (#97), `None` when breadcrumbs are off.
    /// Marks which pass the GPU was in when a `VK_ERROR_DEVICE_LOST` is
    /// observed. Behind a `Mutex` for the same single-render-thread reason as
    /// the other per-frame state.
    breadcrumbs: Option<Mutex<breadcrumbs::BreadcrumbManager>>,
    /// Set on the first observed device loss so the (expensive, log-and-file)
    /// breadcrumb report runs exactly once even though device loss cascades
    /// through every subsequent submit/wait/present.
    device_lost_reported: std::sync::atomic::AtomicBool,
    /// `VK_KHR_acceleration_structure` entry points, `Some` exactly when
    /// [`DeviceCapabilities::ray_query`](crate::device::DeviceCapabilities) is
    /// true (#110). Used for BLAS/TLAS creation, build-size queries, and
    /// `vkCmdBuildAccelerationStructuresKHR` encoding.
    accel_loader: Option<ash::khr::acceleration_structure::Device>,
    /// Compacted-size query pools for transparent BLAS compaction (#110 phase
    /// 3, ADR-032). `Some` alongside `accel_loader` unless pool creation
    /// failed. Behind a `Mutex` for the same single-render-thread reason as the
    /// timestamp pools.
    compaction_queries: Option<Mutex<compaction::CompactionQueryManager>>,
    /// `VK_EXT_mesh_shader` entry points, `Some` exactly when
    /// [`DeviceCapabilities::mesh_shading`](crate::device::DeviceCapabilities)
    /// is true (#111). Used for `vkCmdDrawMeshTasksEXT` encoding.
    mesh_loader: Option<ash::ext::mesh_shader::Device>,
    /// Capabilities queried from the selected physical device at creation
    /// (ADR-027) — the single source of truth downstream clamps against.
    device_caps: crate::device::DeviceCapabilities,
    /// Name/vendor/type of the selected physical device.
    adapter_info: crate::instance::AdapterInfo,
    /// Whether the instance has surface extensions (false = headless loader;
    /// surface creation fails early instead of calling null fn pointers).
    surface_support: bool,
}

/// Distinct queue families for CONCURRENT resource creation (graphics plus
/// up to one per secondary queue), inline to avoid a per-creation allocation.
#[derive(Debug, Clone, Copy)]
struct ConcurrentFamilies {
    families: [u32; barriers::QUEUE_COUNT],
    len: usize,
}

impl ConcurrentFamilies {
    fn new(graphics_family: u32) -> Self {
        Self {
            families: [graphics_family; barriers::QUEUE_COUNT],
            len: 1,
        }
    }

    fn push(&mut self, family: u32) {
        self.families[self.len] = family;
        self.len += 1;
    }

    fn as_slice(&self) -> &[u32] {
        &self.families[..self.len]
    }
}

/// A secondary queue (async compute / dedicated transfer) and its submission
/// state (#47 phase 4, #89).
struct SecondaryQueue {
    /// The queue itself.
    queue: vk::Queue,
    /// Its queue family (async compute may equal the graphics family — second
    /// queue in the same family; CONCURRENT sharing is then unnecessary but
    /// cross-queue semaphores still are. A transfer family is always
    /// distinct by construction).
    family: u32,
    /// This queue's timeline semaphore (mirrors `queue_timeline`).
    timeline: vk::Semaphore,
    /// Next timeline value to signal on `timeline`.
    timeline_next: AtomicU64,
    /// Per-frame-slot command pools, bulk-reset in `advance_frame` alongside
    /// the graphics ones (the same slot-fence waits guarantee this queue's
    /// submits finished — the pipeline waits ALL of a slot's fences).
    command_pools: [vk::CommandPool; MAX_FRAMES_IN_FLIGHT],
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("validation_enabled", &self.validation_enabled)
            .finish()
    }
}

/// Whether a transfer op is legal on a transfer queue whose image granularity
/// is coarser than 1×1×1 (#92). Buffer-only ops are always legal; image copies
/// are legal only when they cover a whole base subresource (mip 0, origin
/// (0,0,0), full descriptor extent), which reaches every subresource edge and
/// so satisfies the granularity rule at any granularity (the spec exempts a
/// copy whose `offset + extent` equals the subresource dimensions). Partial or
/// non-base image copies — none generated today — read as unsafe so the graph
/// falls back to async compute.
/// Name a Vulkan object via `VK_EXT_debug_utils` (#123), usable before the
/// `VulkanBackend` struct exists (device creation) and from sub-modules that
/// hold only the loader. A no-op when `loader` is `None` or `name` is empty /
/// not a valid C string — naming never fails a real operation.
pub(super) fn set_debug_object_name<H: vk::Handle>(
    loader: Option<&ash::ext::debug_utils::Device>,
    handle: H,
    name: &str,
) {
    let Some(loader) = loader else {
        return;
    };
    if name.is_empty() {
        return;
    }
    let Ok(cname) = std::ffi::CString::new(name) else {
        return;
    };
    let info = vk::DebugUtilsObjectNameInfoEXT::default()
        .object_handle(handle)
        .object_name(&cname);
    unsafe {
        let _ = loader.set_debug_utils_object_name(&info);
    }
}

/// A stable per-pass-type colour for the pass's debug label region (#123):
/// RenderDoc tints each region so pass kinds are distinguishable at a glance.
/// Blue = graphics, green = transfer, orange = compute, purple = AS build.
fn pass_label_color(pass: &Pass) -> [f32; 4] {
    match pass {
        Pass::Graphics(_) => [0.26, 0.52, 0.96, 1.0],
        Pass::Transfer(_) => [0.30, 0.69, 0.31, 1.0],
        Pass::Compute(_) => [0.96, 0.60, 0.16, 1.0],
        Pass::AccelerationStructureBuild(_) => [0.61, 0.35, 0.71, 1.0],
    }
}

fn op_ok_on_coarse_transfer(op: &crate::graph::TransferOperation) -> bool {
    use crate::graph::TransferOperation as Op;
    match op {
        Op::BufferToBuffer { .. } | Op::WriteBuffer { .. } | Op::ReadbackBuffer { .. } => true,
        Op::BufferToTexture { dst, regions, .. } => regions
            .iter()
            .all(|r| region_is_whole_base(dst, &r.texture_location, r.extent)),
        Op::TextureToBuffer { src, regions, .. } => regions
            .iter()
            .all(|r| region_is_whole_base(src, &r.texture_location, r.extent)),
        Op::TextureToTexture { src, dst, regions } => regions.iter().all(|r| {
            region_is_whole_base(src, &r.src, r.extent)
                && region_is_whole_base(dst, &r.dst, r.extent)
        }),
        // Mip generation is a blit chain (#96): not a transfer-queue operation
        // at all, and already routed to the graphics queue by
        // `RenderGraph::requires_graphics_queue`. Never legal on a transfer queue.
        Op::GenerateMipmaps { .. } => false,
    }
}

/// Whether a copy region covers the whole base (mip 0) subresource of `tex`.
fn region_is_whole_base(
    tex: &crate::resources::Texture,
    loc: &crate::graph::TextureCopyLocation,
    extent: crate::types::Extent3d,
) -> bool {
    loc.mip_level == 0
        && loc.origin.x == 0
        && loc.origin.y == 0
        && loc.origin.z == 0
        && extent == tex.size()
}

impl VulkanBackend {
    /// Create a new Vulkan backend with explicit validation setting.
    ///
    /// This initializes the Vulkan instance, selects a physical device,
    /// creates a logical device, and sets up the memory allocator.
    ///
    /// Callers arrive through [`super::create_backend_with_params`], which
    /// holds [`super::BACKEND_LIFECYCLE_LOCK`] (#93).
    pub fn with_params(
        params: &crate::instance::InstanceParameters,
    ) -> Result<Self, GraphicsError> {
        // Load Vulkan entry points
        let entry = unsafe { ash::Entry::load() }.map_err(|e| {
            GraphicsError::InitializationFailed(format!("Failed to load Vulkan: {}", e))
        })?;

        let validation_enabled = params.validation;

        // Create instance with validation layers (and sync validation, #99)
        let created =
            instance::create_instance(&entry, validation_enabled, params.sync_validation)?;
        let instance = created.instance;
        let debug_messenger = created.debug_messenger;
        let debug_utils = created.debug_utils;
        let surface_support = created.surface_support;

        // maintenance9 is usable only when validation is off or the layer is
        // new enough to know it — an older layer reports our hand-rolled
        // structs as "unknown VkStructureType" errors and disables its
        // handling of the extension.
        let allow_maintenance9 = created
            .validation_layer_spec
            .is_none_or(|spec| spec >= maintenance9::MIN_AWARE_HEADER_VERSION);

        // Select physical device (filter-then-score against the baseline
        // tier under the adapter preference; a headless instance cannot
        // require the swapchain extension).
        let selected = device::select_physical_device(
            &entry,
            &instance,
            surface_support,
            &params.adapter,
            allow_maintenance9,
        )?;
        let physical_device = selected.physical_device;

        // GPU crash breadcrumbs (#97): on for dev/editor builds (validation on)
        // unless overridden. Resolved before device creation so the vendor
        // extensions are enabled only when breadcrumbs are actually on.
        let breadcrumbs_on = params.breadcrumbs.resolve(validation_enabled);

        // Plan queues (graphics + async compute when available) and create
        // the logical device.
        let queue_plan = device::plan_queues(&instance, physical_device)?;
        let graphics_queue_family = queue_plan.graphics_family;
        let device = device::create_logical_device(
            &instance,
            &selected,
            &queue_plan,
            surface_support,
            breadcrumbs_on,
        )?;

        // Everything downstream clamps/validates against these (ADR-027).
        let device_caps = device::device_capabilities(&instance, &selected, &queue_plan);
        let adapter_info = device::adapter_info(&selected);
        log::info!("Device capabilities: {device_caps:?}");

        // maintenance9 image fast path (#88): with the feature enabled and
        // mutual implicit-acquire support between EVERY pair of distinct
        // families a declared cross-queue texture can be touched from
        // (graphics, async compute, dedicated transfer #89), declared
        // textures stay EXCLUSIVE.
        let distinct_secondary_families: Vec<u32> = queue_plan
            .async_compute
            .into_iter()
            .chain(queue_plan.transfer)
            .map(|(family, _)| family)
            .filter(|&family| family != queue_plan.graphics_family)
            .collect();
        let implicit_cross_queue_images =
            selected.optional.maintenance9 && !distinct_secondary_families.is_empty() && {
                let mut all_families = vec![queue_plan.graphics_family];
                all_families.extend(&distinct_secondary_families);
                all_families.iter().enumerate().all(|(i, &a)| {
                    all_families[i + 1..].iter().all(|&b| {
                        maintenance9::implicit_image_transfer_ok(&instance, physical_device, a, b)
                    })
                })
            };
        if !distinct_secondary_families.is_empty() {
            log::info!(
                "Cross-queue textures: {}",
                if implicit_cross_queue_images {
                    "EXCLUSIVE via maintenance9 implicit ownership (compression retained)"
                } else {
                    "CONCURRENT (maintenance9 implicit image acquire unavailable)"
                }
            );
        }

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

        // Create memory allocator (wrapped in Arc for sharing with GPU resource Drop impls).
        // Buffer device addresses are allocator-wide: with ray query enabled
        // (#110) every allocation carries VK_MEMORY_ALLOCATE_DEVICE_ADDRESS
        // so AS/scratch/geometry buffers can be addressed by the build APIs.
        let allocator = Arc::new(Mutex::new(allocator::create_allocator(
            &instance,
            physical_device,
            device.clone(),
            device_caps.ray_query,
        )?));

        // Debug-utils device functions for object naming + pass labels (#123),
        // loaded whenever the instance enabled the extension.
        let debug_utils_device = debug_utils
            .as_ref()
            .map(|_| ash::ext::debug_utils::Device::new(&instance, &device));

        // Acceleration-structure entry points, loaded only when the extension
        // bundle is enabled (#110).
        let accel_loader = device_caps
            .ray_query
            .then(|| ash::khr::acceleration_structure::Device::new(&instance, &device));

        // Compacted-size query pools for transparent BLAS compaction (#110
        // phase 3), created alongside the AS entry points.
        let compaction_queries = accel_loader
            .is_some()
            .then(|| compaction::CompactionQueryManager::new(&device))
            .flatten()
            .map(Mutex::new);

        // Mesh-shader entry points, loaded only when the extension is
        // enabled (#111).
        let mesh_loader = device_caps
            .mesh_shading
            .then(|| ash::ext::mesh_shader::Device::new(&instance, &device));

        // Create command pool (staging uploads + swapchain present). Present
        // command buffers are reset individually each frame, so this pool
        // needs per-buffer reset.
        let command_pool = command::create_command_pool(&device, graphics_queue_family, true)?;

        // Per-frame-slot pools for the render-graph submit — only ever
        // bulk-reset once the slot's fence signals, so no per-buffer reset.
        let frame_command_pools = {
            let mut pools = [vk::CommandPool::null(); MAX_FRAMES_IN_FLIGHT];
            for pool in &mut pools {
                *pool = command::create_command_pool(&device, graphics_queue_family, false)?;
            }
            pools
        };

        // Per-queue timeline semaphores (initial counter 0). Every
        // render-graph submit signals its queue's next value; frame fences
        // wait on them.
        let create_timeline = |device: &ash::Device| -> Result<vk::Semaphore, GraphicsError> {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            unsafe { device.create_semaphore(&info, None) }.map_err(|e| {
                GraphicsError::InitializationFailed(format!(
                    "Failed to create queue timeline semaphore: {:?}",
                    e
                ))
            })
        };
        let queue_timeline = create_timeline(&device)?;
        set_debug_object_name(
            debug_utils_device.as_ref(),
            queue_timeline,
            "graphics-queue timeline",
        );

        // The secondary queues (async compute #47, dedicated transfer #89),
        // when planned: each gets its own timeline and per-slot command pools
        // (a command pool is tied to one family and one recording thread's
        // pools must not be shared across queues' submit streams anyway).
        let create_secondary =
            |planned: Option<(u32, u32)>| -> Result<Option<SecondaryQueue>, GraphicsError> {
                let Some((family, index)) = planned else {
                    return Ok(None);
                };
                let queue = unsafe { device.get_device_queue(family, index) };
                let timeline = create_timeline(&device)?;
                let command_pools = {
                    let mut pools = [vk::CommandPool::null(); MAX_FRAMES_IN_FLIGHT];
                    for pool in &mut pools {
                        *pool = command::create_command_pool(&device, family, false)?;
                    }
                    pools
                };
                Ok(Some(SecondaryQueue {
                    queue,
                    family,
                    timeline,
                    timeline_next: AtomicU64::new(1),
                    command_pools,
                }))
            };
        let async_compute = create_secondary(queue_plan.async_compute)?;
        let transfer_queue = create_secondary(queue_plan.transfer)?;
        if let Some(q) = &async_compute {
            set_debug_object_name(
                debug_utils_device.as_ref(),
                q.timeline,
                "async-compute-queue timeline",
            );
        }
        if let Some(q) = &transfer_queue {
            set_debug_object_name(
                debug_utils_device.as_ref(),
                q.timeline,
                "transfer-queue timeline",
            );
        }

        // Per-pass GPU timestamps (#95). Build one query pool per (queue, slot)
        // for every queue whose family exposes `timestampValidBits > 0`. The
        // capability bit already gates on the graphics family; a secondary
        // family reporting 0 valid bits is simply skipped (its passes report no
        // timing). Only the graphics queue can report `gpu_timestamps == false`
        // as unsupported outright.
        let timestamps = if device_caps.gpu_timestamps {
            let family_props =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let valid_bits = |family: u32| {
                family_props
                    .get(family as usize)
                    .map_or(0, |p| p.timestamp_valid_bits)
            };
            use self::barriers::QueueId;
            use self::timestamps::QueueTimestampInfo;
            let mut infos = vec![QueueTimestampInfo {
                queue: QueueId::Graphics,
                preference: crate::graph::QueuePreference::Graphics,
                valid_bits: valid_bits(graphics_queue_family),
            }];
            if let Some(ac) = &async_compute {
                infos.push(QueueTimestampInfo {
                    queue: QueueId::AsyncCompute,
                    preference: crate::graph::QueuePreference::AsyncCompute,
                    valid_bits: valid_bits(ac.family),
                });
            }
            if let Some(tq) = &transfer_queue {
                // `vkCmdResetQueryPool` requires a graphics- or compute-capable
                // queue (VUID-vkCmdResetQueryPool-commandBuffer-cmdpool). A
                // dedicated transfer family (#89) exposes neither, so a timestamp
                // pool there could never be reset on its own command buffer —
                // skip it. Resetting host-side would need the `hostQueryReset`
                // feature this module deliberately avoids; transfer-queue DMA
                // timing is a minor loss. (The check is on family flags, so a
                // device whose transfer queue also exposes compute still times.)
                let can_reset = family_props.get(tq.family as usize).is_some_and(|p| {
                    p.queue_flags
                        .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                });
                if can_reset {
                    infos.push(QueueTimestampInfo {
                        queue: QueueId::Transfer,
                        preference: crate::graph::QueuePreference::Transfer,
                        valid_bits: valid_bits(tq.family),
                    });
                } else {
                    log::debug!(
                        "GPU timestamps: skipping transfer queue (family {} is \
                         transfer-only; vkCmdResetQueryPool needs graphics/compute) (#95)",
                        tq.family
                    );
                }
            }
            timestamps::TimestampManager::new(
                &device,
                selected.properties.limits.timestamp_period,
                &infos,
                debug_utils_device.as_ref(),
            )
            .map(Mutex::new)
        } else {
            None
        };

        // Staging chunks must be CONCURRENT-shareable across every distinct
        // family a transfer graph can be routed to (async compute, dedicated
        // transfer).
        let staging_belt = {
            let mut families = vec![graphics_queue_family];
            if let Some(ac) = async_compute
                .as_ref()
                .filter(|ac| ac.family != graphics_queue_family)
            {
                families.push(ac.family);
            }
            if let Some(tq) = &transfer_queue {
                families.push(tq.family);
            }
            let mut belt = staging::StagingBelt::default();
            if families.len() >= 2 {
                belt.set_concurrent_families(&families);
            }
            belt
        };

        // dynamic rendering and synchronization2 are Vulkan 1.3 core: the
        // backend calls `vkCmd{Begin,End}Rendering`, `vkCmdPipelineBarrier2`,
        // `vkQueueSubmit2`, and `vkCmdWriteTimestamp2` directly on `device`.

        // GPU crash breadcrumbs (#97). Pick the mechanism by extension
        // availability (NV checkpoints → AMD buffer marker → portable
        // `vkCmdFillBuffer`), load its handles, and build one marker buffer per
        // (queue, slot) for the buffer mechanisms.
        let breadcrumbs = if breadcrumbs_on {
            use self::barriers::QueueId;
            use self::breadcrumbs::{Mechanism, QueueBreadcrumbInfo};
            let mechanism = if selected.breadcrumbs.nv_checkpoints {
                Mechanism::NvCheckpoints
            } else if selected.breadcrumbs.amd_buffer_marker {
                Mechanism::AmdBufferMarker
            } else {
                Mechanism::Fallback
            };
            let checkpoints = selected
                .breadcrumbs
                .nv_checkpoints
                .then(|| ash::nv::device_diagnostic_checkpoints::Device::new(&instance, &device));
            let buffer_marker = selected
                .breadcrumbs
                .amd_buffer_marker
                .then(|| ash::amd::buffer_marker::Device::new(&instance, &device));
            let device_fault = selected
                .breadcrumbs
                .device_fault
                .then(|| ash::ext::device_fault::Device::new(&instance, &device));

            let mut infos = vec![QueueBreadcrumbInfo {
                queue: QueueId::Graphics,
                preference: crate::graph::QueuePreference::Graphics,
                handle: graphics_queue,
            }];
            if let Some(ac) = &async_compute {
                infos.push(QueueBreadcrumbInfo {
                    queue: QueueId::AsyncCompute,
                    preference: crate::graph::QueuePreference::AsyncCompute,
                    handle: ac.queue,
                });
            }
            if let Some(tq) = &transfer_queue {
                infos.push(QueueBreadcrumbInfo {
                    queue: QueueId::Transfer,
                    preference: crate::graph::QueuePreference::Transfer,
                    handle: tq.queue,
                });
            }
            let manager = breadcrumbs::BreadcrumbManager::new(
                &device,
                &mut allocator.lock(),
                mechanism,
                checkpoints,
                buffer_marker,
                device_fault,
                &infos,
            );
            match &manager {
                Some(_) => log::info!("GPU crash breadcrumbs active: {}", mechanism.label()),
                None => log::warn!("GPU crash breadcrumbs requested but could not be initialized"),
            }
            manager.map(Mutex::new)
        } else {
            log::info!("GPU crash breadcrumbs: off");
            None
        };

        // Load surface extension
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // Load swapchain extension
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        // Create layout tracker for automatic barrier placement
        let layout_tracker = Mutex::new(TextureLayoutTracker::new());

        // D24_UNORM_S8_UINT is optional (commonly absent on AMD); the spec
        // only guarantees that one of D24S8/D32S8 supports depth-stencil
        // attachment. Resolve the actual format for Depth24PlusStencil8 once.
        let depth24_stencil8_format = {
            let props = unsafe {
                instance.get_physical_device_format_properties(
                    physical_device,
                    vk::Format::D24_UNORM_S8_UINT,
                )
            };
            if props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            {
                vk::Format::D24_UNORM_S8_UINT
            } else {
                log::info!(
                    "D24_UNORM_S8_UINT not supported by this device; \
                     Depth24PlusStencil8 maps to D32_SFLOAT_S8_UINT"
                );
                vk::Format::D32_SFLOAT_S8_UINT
            }
        };

        // Create pipeline manager for shader compilation and graphics pipelines.
        // Device properties identify the on-disk pipeline cache's owner
        // (vendor/device/cacheUUID header validation).
        let pipeline_manager = pipeline::PipelineManager::new(
            device.clone(),
            depth24_stencil8_format,
            &selected.properties,
            device_caps.wireframe,
            device_caps.ray_query,
            device_caps.mesh_shading,
            device_caps
                .bindless
                .then(|| device::bindless_capacities(&instance, physical_device)),
        )?;

        // Seed the memory-stats cache so the panel has heap sizes immediately;
        // it is re-sampled every frame in `advance_frame` (#98).
        let memory_stats = Mutex::new(memory::sample(
            &instance,
            physical_device,
            device_caps.memory_budget,
            &allocator.lock(),
        ));

        log::info!(
            "Vulkan backend initialized (validation: {})",
            validation_enabled
        );

        Ok(Self {
            entry,
            instance,
            debug_messenger,
            debug_utils,
            debug_utils_device,
            physical_device,
            device,
            graphics_queue,
            graphics_queue_family,
            allocator: ManuallyDrop::new(allocator),
            command_pool,
            swapchain_sync: Mutex::new(SwapchainSync::default()),
            validation_enabled,
            surface_loader,
            swapchain_loader,
            frame_command_pools,
            current_slot: AtomicUsize::new(0),
            frame_index: AtomicU64::new(0),
            layout_tracker,
            buffer_tracker: Mutex::new({
                let mut tracker = BufferAccessTracker::new();
                tracker.set_mesh_shading(device_caps.mesh_shading);
                tracker
            }),
            retired_tracker_handles: Arc::new(Mutex::new(RetiredTrackerHandles::default())),
            pipeline_manager,
            depth24_stencil8_format,
            encoder_scratch: Mutex::new(VulkanEncoderScratch::default()),
            staging_belt: Mutex::new(staging_belt),
            queue_timeline,
            timeline_next: AtomicU64::new(1),
            async_compute,
            transfer_queue,
            timestamps,
            memory_stats,
            breadcrumbs,
            device_lost_reported: std::sync::atomic::AtomicBool::new(false),
            transfer_granularity: queue_plan.transfer_granularity,
            async_decline_warned: Mutex::new(HashSet::new()),
            implicit_cross_queue_images,
            accel_loader,
            compaction_queries,
            mesh_loader,
            device_caps,
            adapter_info,
            surface_support,
        })
    }

    /// Capabilities queried at backend creation (ADR-027).
    pub fn capabilities(&self) -> crate::device::DeviceCapabilities {
        self.device_caps
    }

    /// GPU timings for the most recently retired frame slot (#95). Empty when
    /// the device lacks timestamp support.
    pub fn latest_gpu_timings(&self) -> crate::device::FrameGpuTimings {
        self.timestamps
            .as_ref()
            .map(|m| m.lock().latest())
            .unwrap_or_default()
    }

    /// GPU-memory statistics as of the last per-frame sample (#98). Read-only:
    /// the actual Vulkan query happens on the render thread in `advance_frame`,
    /// so this is safe to call from the UI thread. `resources` is filled at the
    /// device layer.
    pub fn latest_memory_stats(&self) -> crate::device::GpuMemoryStats {
        self.memory_stats.lock().clone()
    }

    /// Whether `format` supports a `vkCmdBlitImage` mip-generation chain (#96):
    /// the format's optimal tiling must advertise `BLIT_SRC | BLIT_DST |
    /// SAMPLED_IMAGE_FILTER_LINEAR`. Block-compressed formats never do. Queried
    /// straight from the driver — a cheap call made once per texture load.
    pub fn supports_blit_mipgen(&self, format: crate::types::TextureFormat) -> bool {
        let vk_format = self.vk_texture_format(format);
        let props = unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical_device, vk_format)
        };
        let needed = vk::FormatFeatureFlags::BLIT_SRC
            | vk::FormatFeatureFlags::BLIT_DST
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        props.optimal_tiling_features.contains(needed)
    }

    /// Re-arm the timestamp query pool for `queue`/`slot` after a recording
    /// error abandoned the command buffer (#95). No-op without timestamps.
    fn abort_timestamps(&self, queue: QueueId, slot: usize) {
        if let Some(m) = &self.timestamps {
            m.lock().abort_submit(queue, slot);
        }
    }

    /// Name a Vulkan object for RenderDoc / validation output (#123). A no-op
    /// when `VK_EXT_debug_utils` is absent or the name is empty/not a valid
    /// C string — naming never fails a real operation.
    fn set_object_name<H: vk::Handle>(&self, handle: H, name: &str) {
        set_debug_object_name(self.debug_utils_device.as_ref(), handle, name);
    }

    /// Open a labelled command-buffer region (#123): RenderDoc groups the
    /// commands until the matching [`end_debug_label`](Self::end_debug_label)
    /// and tints them `color`. A no-op without `VK_EXT_debug_utils`.
    fn begin_debug_label(&self, cmd: vk::CommandBuffer, name: &str, color: [f32; 4]) {
        let Some(debug_utils) = &self.debug_utils_device else {
            return;
        };
        let Ok(cname) = std::ffi::CString::new(name) else {
            return;
        };
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&cname)
            .color(color);
        unsafe { debug_utils.cmd_begin_debug_utils_label(cmd, &label) };
    }

    /// Close the region opened by [`begin_debug_label`](Self::begin_debug_label).
    fn end_debug_label(&self, cmd: vk::CommandBuffer) {
        if let Some(debug_utils) = &self.debug_utils_device {
            unsafe { debug_utils.cmd_end_debug_utils_label(cmd) };
        }
    }

    /// Drop an abandoned submit's breadcrumbs for `queue`/`slot` (#97). No-op
    /// without breadcrumbs.
    fn abort_breadcrumbs(&self, queue: QueueId, slot: usize) {
        if let Some(m) = &self.breadcrumbs {
            m.lock().abort_submit(queue, slot);
        }
    }

    /// Roll back the slot's compacted-size queries after an abandoned submit
    /// (#110 phase 3): the query writes are lost with the command buffer, so
    /// return the awaiting BLASes to `NeedsQuery` for a retry. The pool is
    /// per-slot (queries are reset per-query on their own command buffer), so
    /// unlike the timestamp abort this takes no queue.
    fn abort_compaction(&self, slot: usize) {
        if let Some(m) = &self.compaction_queries {
            m.lock().abort_slot(slot);
        }
    }

    /// Post-mortem for a `VK_ERROR_DEVICE_LOST` (#97): read the per-queue
    /// breadcrumbs, log a structured report at `error!`, and write it to
    /// `redlilium-gpu-crash-<timestamp>.txt` next to the executable (a hung app
    /// often loses its log tail). Runs exactly once per backend even though
    /// device loss cascades through every subsequent submit/wait/present.
    fn report_device_lost(&self) {
        if self.device_lost_reported.swap(true, Ordering::SeqCst) {
            return;
        }
        let Some(m) = &self.breadcrumbs else {
            log::error!(
                "VK_ERROR_DEVICE_LOST on {} — GPU crash breadcrumbs are off; \
                 set REDLILIUM_BREADCRUMBS=1 to identify the guilty pass (#97)",
                self.adapter_info.name
            );
            return;
        };
        let report = m.lock().collect_report(&self.adapter_info);
        log::error!("VK_ERROR_DEVICE_LOST — GPU crash breadcrumbs:\n{report}");
        if let Some(path) = breadcrumbs::write_crash_file(&report) {
            log::error!("GPU crash report written to {}", path.display());
        }
    }

    /// Info about the selected physical device.
    pub fn adapter_info(&self) -> crate::instance::AdapterInfo {
        self.adapter_info.clone()
    }

    /// Whether the instance was created with surface extensions (false on a
    /// headless loader — offscreen rendering only).
    pub fn surface_support(&self) -> bool {
        self.surface_support
    }

    /// Queue family indices for CONCURRENT sharing, or `None` when EXCLUSIVE
    /// suffices (no secondary queue in a distinct family — sharing modes
    /// only matter across *families*).
    ///
    /// With distinct secondary families (async compute #47, dedicated
    /// transfer #89), buffers (all of them — compression does not apply to
    /// buffers) and textures DECLARED cross-queue (#88) are created
    /// CONCURRENT across every such family so cross-queue access needs no
    /// queue-family ownership transfers — timeline semaphores alone carry the
    /// synchronization (#47 design decision). Undeclared textures stay
    /// EXCLUSIVE to keep framebuffer compression; `execute_graph` never
    /// routes a graph that touches one off the graphics queue. Where the
    /// maintenance9 fast path is active
    /// ([`implicit_cross_queue_images`](Self::implicit_cross_queue_images))
    /// even declared textures stay EXCLUSIVE. Full EXCLUSIVE + QFOT is
    /// deliberately not implemented (#88 / ADR-030).
    fn concurrent_families(&self) -> Option<ConcurrentFamilies> {
        let mut families = ConcurrentFamilies::new(self.graphics_queue_family);
        if let Some(ac) = self
            .async_compute
            .as_ref()
            .filter(|ac| ac.family != self.graphics_queue_family)
        {
            families.push(ac.family);
        }
        if let Some(tq) = &self.transfer_queue {
            families.push(tq.family);
        }
        (families.as_slice().len() >= 2).then_some(families)
    }

    /// Convert an engine texture format to the Vulkan format this device
    /// actually uses for it.
    ///
    /// Identical to [`conversion::convert_texture_format`] except for
    /// `Depth24PlusStencil8`, whose Vulkan format is device-dependent
    /// (D24S8 where supported, D32S8 otherwise). Texture creation and
    /// pipeline attachment formats must both go through this so they agree.
    pub(crate) fn vk_texture_format(&self, format: crate::types::TextureFormat) -> vk::Format {
        if format == crate::types::TextureFormat::Depth24PlusStencil8 {
            self.depth24_stencil8_format
        } else {
            convert_texture_format(format)
        }
    }

    /// Get the Vulkan device.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Get the Vulkan entry.
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// Get the Vulkan instance.
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Get the physical device.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Get the graphics queue family index.
    pub fn graphics_queue_family(&self) -> u32 {
        self.graphics_queue_family
    }

    /// Get the graphics queue.
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// Get the surface loader.
    pub fn surface_loader(&self) -> &ash::khr::surface::Instance {
        &self.surface_loader
    }

    /// Get the swapchain loader.
    pub fn swapchain_loader(&self) -> &ash::khr::swapchain::Device {
        &self.swapchain_loader
    }

    /// Get the command pool.
    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    /// Get the allocator for sharing with GPU resource Drop impls.
    pub fn allocator(&self) -> &Arc<Mutex<Allocator>> {
        &self.allocator
    }

    /// Registers this frame's swapchain acquire/render-done semaphores.
    ///
    /// Called by `acquire_next_image`. The render submit that writes the
    /// swapchain image (detected in `execute_graph`) consumes these to wait on
    /// `image_available` and signal `image_render_finished`.
    pub(crate) fn begin_swapchain_frame(
        &self,
        image_available: vk::Semaphore,
        image_render_finished: vk::Semaphore,
    ) {
        let mut sync = self.swapchain_sync.lock();
        sync.image_available = Some(image_available);
        sync.image_render_finished = Some(image_render_finished);
        sync.consumed = false;
        sync.surface_transitioned = false;
    }

    /// If a swapchain acquire is pending and not yet consumed, returns the
    /// `(wait_on_image_available, signal_image_render_finished)` semaphore pair
    /// and marks the frame consumed. Returns `None` otherwise.
    pub(crate) fn take_swapchain_render_sync(&self) -> Option<(vk::Semaphore, vk::Semaphore)> {
        let mut sync = self.swapchain_sync.lock();
        if sync.consumed {
            return None;
        }
        match (sync.image_available.take(), sync.image_render_finished) {
            (Some(ia), Some(irf)) => {
                sync.consumed = true;
                Some((ia, irf))
            }
            _ => None,
        }
    }

    /// Whether the swapchain image was written (and `image_available` consumed)
    /// by a render submit this frame. Used by present to pick its wait semaphore.
    pub(crate) fn swapchain_render_consumed(&self) -> bool {
        self.swapchain_sync.lock().consumed
    }

    /// Undo `take_swapchain_render_sync` after a failed queue submit.
    ///
    /// The failed submit enqueued nothing, so `image_available` still has its
    /// pending signal from the acquire and `image_render_finished` will never
    /// be signaled. Restoring the un-consumed state makes present fall back to
    /// waiting on `image_available` directly (the "nothing rendered" path),
    /// which drains the pending signal and keeps both semaphores reusable —
    /// instead of present blocking forever on `image_render_finished`.
    pub(crate) fn restore_swapchain_render_sync(&self, image_available: vk::Semaphore) {
        let mut sync = self.swapchain_sync.lock();
        sync.image_available = Some(image_available);
        sync.consumed = false;
    }

    /// Advance to the next frame.
    ///
    /// Frees command buffers from the oldest frame slot, advances the layout
    /// tracker, and resets the descriptor pool.
    ///
    /// This should be called after waiting on a frame fence to ensure
    /// the GPU has finished with resources from older frames.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the GPU has finished executing all commands
    /// from the oldest frame slot.
    pub unsafe fn advance_frame(&self) {
        let current = self.current_slot.load(Ordering::Relaxed);
        let oldest = (current + 1) % MAX_FRAMES_IN_FLIGHT;

        // Reset the oldest slot's frame command pools (graphics and, when
        // present, the secondary queues'), freeing all their command buffers
        // at once. Safe: ALL of the slot's fences have been waited (GPU done
        // on every queue).
        unsafe {
            let _ = self.device.reset_command_pool(
                self.frame_command_pools[oldest],
                vk::CommandPoolResetFlags::empty(),
            );
            for secondary in [&self.async_compute, &self.transfer_queue]
                .into_iter()
                .flatten()
            {
                let _ = self.device.reset_command_pool(
                    secondary.command_pools[oldest],
                    vk::CommandPoolResetFlags::empty(),
                );
            }
        }

        // Advance to next slot
        self.current_slot
            .store((current + 1) % MAX_FRAMES_IN_FLIGHT, Ordering::SeqCst);
        // Monotonic frame counter driving the BLAS-compaction lifecycle (#110
        // phase 3) — never wraps, unlike the slot index.
        self.frame_index.fetch_add(1, Ordering::SeqCst);

        // The layout tracker is global and persists across frames (see
        // TextureLayoutTracker docs) — no per-frame reset, so persistent
        // textures keep their contents.

        // Binding-group descriptor sets are no longer per-slot/transient: they
        // are created eagerly, cached for the group's lifetime, and freed on the
        // group's Drop — so there is no per-slot descriptor pool to reset here.

        // Return the oldest slot's staging-belt chunks to the pool — same
        // safety argument: the fence wait guarantees their copies completed.
        self.staging_belt
            .lock()
            .retire_slot(&self.device, &mut self.allocator.lock(), oldest);

        // Read back the retiring slot's GPU timestamps (#95). The same fence
        // wait that lets the staging chunks retire guarantees these queries are
        // available, so `vkGetQueryPoolResults` needs no WAIT bit.
        if let Some(m) = &self.timestamps {
            m.lock().read_slot(&self.device, oldest);
        }

        // Read back the retiring slot's BLAS compacted-size queries (#110 phase
        // 3) and deliver them to the awaiting BLASes — same fence-guaranteed
        // availability as the timestamps, so no WAIT bit.
        if let Some(m) = &self.compaction_queries {
            m.lock().read_slot(&self.device, oldest);
        }

        // Re-sample GPU-memory stats for this frame (#98). Driver budget/usage
        // may change at any moment, so this runs every frame — never cached
        // across frames. On the render thread, so the UI-thread reader
        // (`latest_memory_stats`) only ever clones a completed sample.
        {
            let stats = memory::sample(
                &self.instance,
                self.physical_device,
                self.device_caps.memory_budget,
                &self.allocator.lock(),
            );
            *self.memory_stats.lock() = stats;
        }

        // Retire the slot's crash breadcrumbs (#97): its fence signaled, so its
        // work completed and its breadcrumbs are no longer interesting. This
        // re-arms the marker reset for the slot's next use.
        if let Some(m) = &self.breadcrumbs {
            m.lock().retire_slot(oldest);
        }

        // Remove destroyed resources from the barrier trackers so the maps
        // stay bounded by live resources. Safe at any point: texture ids are
        // never reused, and a buffer handle reused before this drain merely
        // loses benign access state (next use counts as first).
        let retired = std::mem::take(&mut *self.retired_tracker_handles.lock());
        if !retired.textures.is_empty() {
            let mut tracker = self.layout_tracker.lock();
            let mut warned = self.async_decline_warned.lock();
            for id in retired.textures {
                tracker.remove(layout::TextureId::from_raw(id));
                warned.remove(&id);
            }
        }
        if !retired.buffers.is_empty() {
            let mut tracker = self.buffer_tracker.lock();
            for handle in retired.buffers {
                tracker.remove(BufferId::from_raw(handle));
            }
        }

        // Flush freshly compiled pipelines to the on-disk pipeline cache.
        // Done here (not only at teardown) so the cache survives abnormal
        // exits; a no-op on frames without pipeline compilation.
        self.pipeline_manager.persist_cache_if_dirty();
    }

    /// Monotonic frame counter (#110 phase 3): how many times
    /// [`advance_frame`](Self::advance_frame) has run. Drives the BLAS
    /// compaction lifecycle — work submitted this frame is guaranteed complete
    /// once this advances by `MAX_FRAMES_IN_FLIGHT`.
    pub fn frame_index(&self) -> u64 {
        self.frame_index.load(Ordering::SeqCst)
    }

    /// Get the layout tracker for direct access (for testing).
    pub fn layout_tracker(&self) -> &Mutex<TextureLayoutTracker> {
        &self.layout_tracker
    }

    /// Check if the current physical device supports presentation to a surface.
    pub fn is_surface_supported(&self, surface: vk::SurfaceKHR) -> bool {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_support(
                    self.physical_device,
                    self.graphics_queue_family,
                    surface,
                )
                .unwrap_or(false)
        }
    }

    /// Query surface capabilities for a given surface.
    pub fn get_surface_capabilities(
        &self,
        surface: vk::SurfaceKHR,
    ) -> Result<vk::SurfaceCapabilitiesKHR, GraphicsError> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, surface)
        }
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to get surface capabilities: {:?}",
                e
            ))
        })
    }

    /// Query surface formats for a given surface.
    pub fn get_surface_formats(
        &self,
        surface: vk::SurfaceKHR,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, GraphicsError> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, surface)
        }
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to get surface formats: {:?}", e))
        })
    }

    /// Query present modes for a given surface.
    pub fn get_surface_present_modes(
        &self,
        surface: vk::SurfaceKHR,
    ) -> Result<Vec<vk::PresentModeKHR>, GraphicsError> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, surface)
        }
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to get present modes: {:?}", e))
        })
    }

    /// Create an image view for a swapchain image.
    pub fn create_swapchain_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
    ) -> Result<vk::ImageView, GraphicsError> {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe { self.device.create_image_view(&view_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to create swapchain image view: {:?}",
                e
            ))
        })
    }
}

/// Validate a requested MSAA sample count against the device's queried
/// support and convert it to Vulkan flags (VK-M12: an unsupported count is an
/// error, not a silent downgrade to 1 sample that would break rendering when
/// the pipeline's sample state disagrees with the attachment's).
fn sample_count_flags(
    count: u32,
    caps: &crate::device::DeviceCapabilities,
) -> Result<vk::SampleCountFlags, GraphicsError> {
    if count == 1 {
        return Ok(vk::SampleCountFlags::TYPE_1);
    }
    if !caps.supports_sample_count(count) {
        return Err(GraphicsError::InvalidParameter(format!(
            "sample count {count} not supported by this device (supported mask: {:#x})",
            caps.sample_count_mask
        )));
    }
    // VkSampleCountFlagBits values equal the counts themselves.
    Ok(vk::SampleCountFlags::from_raw(count))
}

/// Aspect mask for whole-image operations, derived from the format.
///
/// Depth formats must not be addressed with `COLOR` (invalid usage);
/// image↔image copies may cover depth and stencil together.
fn image_aspect_mask(format: crate::types::TextureFormat) -> vk::ImageAspectFlags {
    if format.is_depth_stencil() {
        if format.has_stencil() {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

/// Resolved layer/z addressing of a texture copy location.
struct ResolvedLayerZ {
    base_layer: u32,
    layer_count: u32,
    z_offset: i32,
    /// Real z-depth of the copied region (1 for array-like textures — their
    /// "depth" moved into `layer_count`).
    extent_depth: u32,
}

/// Split a copy location's `origin.z` / `extent.depth` into array-layer and
/// z-offset terms.
///
/// For array-like dimensions (arrays, cubes) `origin.z` addresses the base
/// layer and `extent.depth` the number of layers — matching wgpu's
/// `Origin3d::z` / `depth_or_array_layers` semantics. For 1D/2D/3D they are a
/// real z offset and depth (Vulkan requires layer terms in the subresource,
/// not the offset).
fn resolve_layer_z(
    dimension: crate::types::TextureDimension,
    origin_z: u32,
    extent: crate::types::Extent3d,
) -> ResolvedLayerZ {
    use crate::types::TextureDimension;
    let array_like = matches!(
        dimension,
        TextureDimension::D1Array
            | TextureDimension::D2Array
            | TextureDimension::Cube
            | TextureDimension::CubeArray
    );
    if array_like {
        ResolvedLayerZ {
            base_layer: origin_z,
            layer_count: extent.depth.max(1),
            z_offset: 0,
            extent_depth: 1,
        }
    } else {
        ResolvedLayerZ {
            base_layer: 0,
            layer_count: 1,
            z_offset: origin_z as i32,
            extent_depth: extent.depth.max(1),
        }
    }
}

/// Build validated `vk::BufferImageCopy` regions for buffer↔image transfer
/// ops. Layout math (tight pitch, block conversion, alignment rules) comes
/// from the shared [`BufferTextureLayout::resolve`], so wgpu and Vulkan
/// accept and reject exactly the same graphs.
/// Image layout for a depth/stencil attachment. A fully read-only attachment
/// (`effective_read_only`) uses `DEPTH_STENCIL_READ_ONLY_OPTIMAL` so it can
/// share the image with simultaneous sampling and matches the layout the
/// barrier system transitions the image to; otherwise the writable attachment
/// layout. (#60)
fn ds_attachment_layout(attachment: &crate::graph::DepthStencilAttachment) -> vk::ImageLayout {
    if attachment.effective_read_only() {
        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
    } else {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    }
}

fn build_buffer_image_copies(
    format: crate::types::TextureFormat,
    dimension: crate::types::TextureDimension,
    regions: &[crate::graph::BufferTextureCopyRegion],
    op_name: &str,
) -> Result<Vec<vk::BufferImageCopy>, GraphicsError> {
    let aspect_mask = image_aspect_mask(format);
    regions
        .iter()
        .map(|r| {
            let layout = r
                .buffer_layout
                .resolve(format, r.extent)
                .map_err(|e| GraphicsError::InvalidParameter(format!("{op_name}: {e}")))?;
            let loc = resolve_layer_z(dimension, r.texture_location.origin.z, r.extent);
            Ok(vk::BufferImageCopy::default()
                .buffer_offset(layout.offset)
                .buffer_row_length(layout.row_length_texels)
                .buffer_image_height(layout.rows_per_image_texels)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask,
                    mip_level: r.texture_location.mip_level,
                    base_array_layer: loc.base_layer,
                    layer_count: loc.layer_count,
                })
                .image_offset(vk::Offset3D {
                    x: r.texture_location.origin.x as i32,
                    y: r.texture_location.origin.y as i32,
                    z: loc.z_offset,
                })
                .image_extent(vk::Extent3D {
                    width: r.extent.width,
                    height: r.extent.height,
                    depth: loc.extent_depth,
                }))
        })
        .collect()
}

/// Per-frame swapchain synchronization handoff between `acquire_next_image`,
/// the render submit (in `execute_graph`), and present.
#[derive(Default)]
struct SwapchainSync {
    /// Signaled by acquire; the swapchain-writing submit waits on it. `None`
    /// once consumed.
    image_available: Option<vk::Semaphore>,
    /// Signaled by the swapchain-writing submit; present waits on it.
    image_render_finished: Option<vk::Semaphore>,
    /// Whether a render submit this frame consumed `image_available` (i.e. wrote
    /// the swapchain image). Present uses this to pick its wait semaphore.
    consumed: bool,
    /// Whether the acquired image has been transitioned to
    /// `COLOR_ATTACHMENT_OPTIMAL` this frame. Only the FIRST surface-writing
    /// pass transitions from `UNDEFINED` (discarding stale presented
    /// contents); later passes emit a same-layout WAW barrier so their
    /// `LoadOp::Load` sees the earlier passes' output.
    surface_transitioned: bool,
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        // Serialized against creation of other backends (#93) — see
        // BACKEND_LIFECYCLE_LOCK.
        let _lifecycle = super::BACKEND_LIFECYCLE_LOCK.lock();
        unsafe {
            // Wait for device to be idle before cleanup
            let _ = self.device.device_wait_idle();

            // Destroy pipeline manager resources BEFORE destroying the device.
            // PipelineManager holds Vulkan handles (descriptor pool, pipelines, etc.)
            // that must be destroyed while the device is still valid.
            self.pipeline_manager.destroy();

            // Destroy command pools (frame pools free their buffers on destroy).
            for &pool in &self.frame_command_pools {
                self.device.destroy_command_pool(pool, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);

            // Destroy the queue timeline semaphores. Outstanding `GpuFence`s
            // cannot exist here: each holds an `Arc<GraphicsInstance>` that
            // keeps this backend alive (#50).
            self.device.destroy_semaphore(self.queue_timeline, None);

            // Secondary queue resources (async compute, transfer), if present.
            for secondary in [&self.async_compute, &self.transfer_queue]
                .into_iter()
                .flatten()
            {
                for &pool in &secondary.command_pools {
                    self.device.destroy_command_pool(pool, None);
                }
                self.device.destroy_semaphore(secondary.timeline, None);
            }

            // Destroy the timestamp query pools (#95), if any.
            if let Some(m) = &self.timestamps {
                m.lock().destroy(&self.device);
            }

            // Destroy the BLAS compacted-size query pools (#110 phase 3), if any.
            if let Some(m) = &self.compaction_queries {
                m.lock().destroy(&self.device);
            }

            // Destroy the breadcrumb marker buffers (#97), if any (the
            // device_wait_idle above guarantees the GPU is done with them).
            if let Some(m) = &self.breadcrumbs {
                m.lock().destroy(&self.device, &mut self.allocator.lock());
            }

            // Destroy the staging belt (the device_wait_idle above
            // guarantees the GPU is done with every chunk).
            self.staging_belt
                .lock()
                .destroy(&self.device, &mut self.allocator.lock());

            // Drop the allocator BEFORE destroying the device. gpu-allocator's
            // `Allocator::drop` frees its pooled memory blocks using the device,
            // so it must run while the device is still valid. Struct fields drop
            // *after* this method returns, so we drop it explicitly here.
            // SAFETY: `self.allocator` is never used again after this point.
            ManuallyDrop::drop(&mut self.allocator);

            // Destroy logical device
            self.device.destroy_device(None);

            // Destroy debug messenger
            if let (Some(debug_utils), Some(messenger)) = (&self.debug_utils, self.debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }

            // Destroy instance
            self.instance.destroy_instance(None);
        }
    }
}

impl VulkanBackend {
    /// Get the backend name.
    pub fn name(&self) -> &'static str {
        "Vulkan Backend (ash)"
    }

    /// Create a buffer resource.
    pub fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<GpuBuffer, GraphicsError> {
        // The AS/device-address roles are valid only with the ray-query
        // bundle enabled (#110): without `bufferDeviceAddress` the usage flags
        // themselves are a Vulkan error, so fail with the engine-level cause.
        if descriptor
            .usage
            .intersects(crate::types::BufferUsage::RAY_TRACING_FLAGS)
            && !self.device_caps.ray_query
        {
            return Err(GraphicsError::FeatureNotSupported(
                "acceleration-structure buffer usage requires DeviceCapabilities::ray_query \
                 (VK_KHR_acceleration_structure + VK_KHR_ray_query, #110)"
                    .to_string(),
            ));
        }
        let usage = convert_buffer_usage(descriptor.usage);

        // Memory location follows mappability, not copyability (ADR-021):
        // COPY_DST destinations are written by GPU-side copies and belong in
        // device-local memory — the old `COPY_DST => CpuToGpu` heuristic put
        // every mesh/uniform buffer in host-visible memory, so draws fetched
        // over PCIe on discrete GPUs. Buffers the CPU writes directly are
        // marked MAP_WRITE or RING (ring buffers are mapped-written every
        // frame; RING stays an engine-side flag because wgpu's MAP_WRITE has
        // different combination rules and rings use Queue::write_buffer
        // there).
        let location = if descriptor
            .usage
            .contains(crate::types::BufferUsage::MAP_READ)
        {
            gpu_allocator::MemoryLocation::GpuToCpu
        } else if descriptor
            .usage
            .intersects(crate::types::BufferUsage::MAP_WRITE | crate::types::BufferUsage::RING)
        {
            gpu_allocator::MemoryLocation::CpuToGpu
        } else {
            gpu_allocator::MemoryLocation::GpuOnly
        };

        // Guard the ADR-021 contract: a buffer meant for direct mapped writes
        // (RING every frame, or MAP_WRITE) must be host-visible. If a future
        // policy change let one land GpuOnly, every `write_buffer` on it
        // would fail (the blocking one-shot fallback was removed in #89) —
        // catch the miscreated buffer at creation instead.
        debug_assert!(
            !descriptor
                .usage
                .intersects(crate::types::BufferUsage::MAP_WRITE | crate::types::BufferUsage::RING)
                || location != gpu_allocator::MemoryLocation::GpuOnly,
            "RING/MAP_WRITE buffer must be host-visible (see ADR-021); got GpuOnly"
        );

        // Create buffer. CONCURRENT across graphics + async compute families
        // when the latter exists (see `concurrent_families`).
        let mut buffer_info = vk::BufferCreateInfo::default()
            .size(descriptor.size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let concurrent_families = self.concurrent_families();
        if let Some(families) = &concurrent_families {
            buffer_info = buffer_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(families.as_slice());
        }

        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to create buffer: {:?}", e))
        })?;
        // Name the object for RenderDoc / validation (#123); no-op when the
        // descriptor has no label or debug utils is absent.
        self.set_object_name(buffer, descriptor.label.as_deref().unwrap_or(""));

        // Get memory requirements
        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        // Allocate memory
        let allocation = {
            let mut allocator = self.allocator.lock();
            allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: descriptor.label.as_deref().unwrap_or("buffer"),
                    requirements: mem_requirements,
                    location,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| {
                    GraphicsError::ResourceCreationFailed(format!(
                        "Failed to allocate buffer memory: {}",
                        e
                    ))
                })?
        };

        // Bind memory to buffer
        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to bind buffer memory: {:?}", e))
        })?;

        Ok(GpuBuffer::Vulkan {
            device: self.device.clone(),
            buffer,
            allocation: Mutex::new(Some(allocation)),
            size: descriptor.size,
            allocator: Arc::clone(&self.allocator),
            retired: Arc::clone(&self.retired_tracker_handles),
        })
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        use crate::types::TextureDimension;

        let format = self.vk_texture_format(descriptor.format);
        let usage = convert_texture_usage(descriptor.usage, descriptor.format);

        // Determine image type, array layers, and flags based on dimension
        let (image_type, array_layers, extent, flags) = match descriptor.dimension {
            TextureDimension::D1 => (
                vk::ImageType::TYPE_1D,
                1,
                vk::Extent3D {
                    width: descriptor.size.width,
                    height: 1,
                    depth: 1,
                },
                vk::ImageCreateFlags::empty(),
            ),
            TextureDimension::D1Array => (
                vk::ImageType::TYPE_1D,
                descriptor.size.depth.max(1),
                vk::Extent3D {
                    width: descriptor.size.width,
                    height: 1,
                    depth: 1,
                },
                vk::ImageCreateFlags::empty(),
            ),
            TextureDimension::D2 => (
                vk::ImageType::TYPE_2D,
                1,
                vk::Extent3D {
                    width: descriptor.size.width,
                    height: descriptor.size.height,
                    depth: 1,
                },
                vk::ImageCreateFlags::empty(),
            ),
            TextureDimension::D2Array => (
                vk::ImageType::TYPE_2D,
                descriptor.size.depth.max(1),
                vk::Extent3D {
                    width: descriptor.size.width,
                    height: descriptor.size.height,
                    depth: 1,
                },
                vk::ImageCreateFlags::empty(),
            ),
            TextureDimension::D3 => (
                vk::ImageType::TYPE_3D,
                1,
                vk::Extent3D {
                    width: descriptor.size.width,
                    height: descriptor.size.height,
                    depth: descriptor.size.depth.max(1),
                },
                vk::ImageCreateFlags::empty(),
            ),
            TextureDimension::Cube => (
                vk::ImageType::TYPE_2D,
                6,
                vk::Extent3D {
                    width: descriptor.size.width,
                    height: descriptor.size.height,
                    depth: 1,
                },
                vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ),
            TextureDimension::CubeArray => (
                vk::ImageType::TYPE_2D,
                descriptor.size.depth * 6,
                vk::Extent3D {
                    width: descriptor.size.width,
                    height: descriptor.size.height,
                    depth: 1,
                },
                vk::ImageCreateFlags::CUBE_COMPATIBLE,
            ),
        };

        // Create image. CONCURRENT across graphics + async compute families
        // only when the texture is DECLARED cross-queue (#88) AND the
        // maintenance9 fast path is unavailable: with mutual implicit
        // ownership acquisition the declared texture stays EXCLUSIVE too
        // (compression retained, no transfers — the D3D12 model). Undeclared
        // images always stay EXCLUSIVE and `execute_graph` keeps them off
        // the async queue.
        let mut image_info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type)
            .format(format)
            .extent(extent)
            .mip_levels(descriptor.mip_level_count)
            .array_layers(array_layers)
            .samples(sample_count_flags(
                descriptor.sample_count,
                &self.device_caps,
            )?)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let concurrent_families = (descriptor.cross_queue && !self.implicit_cross_queue_images)
            .then(|| self.concurrent_families())
            .flatten();
        if let Some(families) = &concurrent_families {
            image_info = image_info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(families.as_slice());
        }

        let image = unsafe { self.device.create_image(&image_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to create image: {:?}", e))
        })?;
        self.set_object_name(image, descriptor.label.as_deref().unwrap_or(""));

        // Get memory requirements
        let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };

        // Allocate GPU-only memory for textures.
        //
        // Images get a DEDICATED allocation (their own `VkDeviceMemory`) rather
        // than a suballocation from a shared block. On unified-memory devices
        // (MoltenVK) buffers and images land in the same memory type, and a
        // suballocated image packed tightly next to a staging buffer trips
        // `VUID-vkCmdCopyBufferToImage-pRegions-00173` (the validation layer
        // reads the image's full `VkMemoryRequirements` footprint, which can
        // reach into the adjacent buffer). A dedicated image allocation makes a
        // buffer↔image memory overlap structurally impossible (#61).
        let allocation = {
            let mut allocator = self.allocator.lock();
            allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: descriptor.label.as_deref().unwrap_or("texture"),
                    requirements: mem_requirements,
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedImage(
                        image,
                    ),
                })
                .map_err(|e| {
                    GraphicsError::ResourceCreationFailed(format!(
                        "Failed to allocate texture memory: {}",
                        e
                    ))
                })?
        };

        // Bind memory to image
        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to bind image memory: {:?}", e))
        })?;

        // Create image view
        let aspect_mask = if descriptor.format.is_depth_stencil() {
            if descriptor.format.has_stencil() {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            } else {
                vk::ImageAspectFlags::DEPTH
            }
        } else {
            vk::ImageAspectFlags::COLOR
        };

        // Determine view type based on dimension
        let (view_type, layer_count) = match descriptor.dimension {
            TextureDimension::D1 => (vk::ImageViewType::TYPE_1D, 1),
            TextureDimension::D1Array => (vk::ImageViewType::TYPE_1D_ARRAY, array_layers),
            TextureDimension::D2 => (vk::ImageViewType::TYPE_2D, 1),
            TextureDimension::D2Array => (vk::ImageViewType::TYPE_2D_ARRAY, array_layers),
            TextureDimension::D3 => (vk::ImageViewType::TYPE_3D, 1),
            TextureDimension::Cube => (vk::ImageViewType::CUBE, 6),
            TextureDimension::CubeArray => (vk::ImageViewType::CUBE_ARRAY, array_layers),
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: descriptor.mip_level_count,
                base_array_layer: 0,
                layer_count,
            });

        let view = unsafe { self.device.create_image_view(&view_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to create image view: {:?}", e))
        })?;
        if let Some(label) = &descriptor.label {
            self.set_object_name(view, &format!("{label} view"));
        }

        Ok(GpuTexture::Vulkan {
            device: self.device.clone(),
            image,
            view,
            allocation: Mutex::new(Some(allocation)),
            format,
            extent,
            allocator: Arc::clone(&self.allocator),
            id: NEXT_TEXTURE_ID.fetch_add(1, Ordering::Relaxed),
            retired: Arc::clone(&self.retired_tracker_handles),
        })
    }

    /// Create a sampler resource.
    pub fn create_sampler(
        &self,
        descriptor: &SamplerDescriptor,
    ) -> Result<GpuSampler, GraphicsError> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(convert_filter_mode(descriptor.mag_filter))
            .min_filter(convert_filter_mode(descriptor.min_filter))
            .mipmap_mode(convert_mipmap_filter_mode(descriptor.mipmap_filter))
            .address_mode_u(convert_address_mode(descriptor.address_mode_u))
            .address_mode_v(convert_address_mode(descriptor.address_mode_v))
            .address_mode_w(convert_address_mode(descriptor.address_mode_w))
            .mip_lod_bias(0.0)
            // Clamp to the queried device limit; on devices without the
            // samplerAnisotropy feature the cap is 1 and this disables
            // anisotropy entirely instead of tripping validation (VK-M12).
            .anisotropy_enable(
                descriptor.anisotropy_clamp > 1 && self.device_caps.max_sampler_anisotropy > 1,
            )
            .max_anisotropy(
                descriptor
                    .anisotropy_clamp
                    .min(self.device_caps.max_sampler_anisotropy)
                    .max(1) as f32,
            )
            .compare_enable(descriptor.compare.is_some())
            .compare_op(
                descriptor
                    .compare
                    .map(convert_compare_function)
                    .unwrap_or(vk::CompareOp::ALWAYS),
            )
            .min_lod(descriptor.lod_min_clamp)
            .max_lod(descriptor.lod_max_clamp)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false);

        let sampler = unsafe { self.device.create_sampler(&sampler_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to create sampler: {:?}", e))
        })?;
        self.set_object_name(sampler, descriptor.label.as_deref().unwrap_or(""));

        Ok(GpuSampler::Vulkan {
            device: self.device.clone(),
            sampler,
        })
    }

    /// Create a binding group: allocate one descriptor set from the persistent
    /// pool for `layout`'s (deduped) `VkDescriptorSetLayout` and write it once.
    ///
    /// The written set is cached inside the returned handle and bound as-is on
    /// every draw — encoding performs zero descriptor allocations/writes.
    pub fn create_binding_group(
        &self,
        layout: &crate::materials::BindingLayout,
        descriptor: &crate::materials::BindingGroupDescriptor,
    ) -> Result<super::GpuBindingGroup, GraphicsError> {
        // Deduped layout: a group created against a `BindingLayout` gets the
        // same `VkDescriptorSetLayout` as any material with an equal-content
        // layout, so the cached set is compatible with those pipelines.
        let ds_layout = self.pipeline_manager.create_descriptor_set_layout(layout)?;

        let (descriptor_set, pool) = self
            .pipeline_manager
            .persistent_pools()
            .lock()
            .allocate(ds_layout)?;

        // Write the set once, now. Freed on the group's Drop.
        self.write_binding_group_set(descriptor_set, descriptor, layout);

        Ok(super::GpuBindingGroup::Vulkan {
            device: self.device.clone(),
            descriptor_set,
            pool,
            pools: Arc::clone(self.pipeline_manager.persistent_pools()),
        })
    }

    /// Write a freshly-allocated descriptor set from a binding group's
    /// descriptor. Called once at group-creation time (never per draw).
    ///
    /// # Image layout invariant
    ///
    /// A set written once at creation must record the layout each sampled image
    /// will be in **at draw time**, not the (possibly `UNDEFINED`) layout at
    /// creation time. A texture binding samples in `SHADER_READ_ONLY_OPTIMAL` by
    /// default; a binding declared
    /// [`SampledDepthLayout::DepthStencilReadOnly`](crate::SampledDepthLayout)
    /// (a depth texture co-used as a read-only depth attachment) samples in
    /// `DEPTH_STENCIL_READ_ONLY_OPTIMAL`. The render graph transitions the image
    /// to exactly that layout: `graph::pass::extract_material_resources` maps the
    /// same `sampled_depth_layout` to a `ShaderRead` / `DepthStencilReadOnly`
    /// access, so the descriptor and the barrier always agree. There are no
    /// storage-image bindings on this path.
    fn write_binding_group_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        descriptor: &crate::materials::BindingGroupDescriptor,
        layout: &crate::materials::BindingLayout,
    ) {
        use crate::materials::{BoundResource, SampledDepthLayout};

        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();

        for entry in &descriptor.entries {
            // The layout a sampled texture records is fixed at creation by the
            // entry's `sampled_depth_layout`; the barrier system transitions the
            // image to the same layout (see the doc-comment).
            let sampled_layout = match entry.sampled_depth_layout {
                SampledDepthLayout::ShaderReadOnly => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                SampledDepthLayout::DepthStencilReadOnly => {
                    vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                }
            };
            match &entry.resource {
                BoundResource::Buffer(buffer) => {
                    if let GpuBuffer::Vulkan {
                        buffer: vk_buffer,
                        size,
                        ..
                    } = buffer.gpu_handle()
                    {
                        buffer_infos.push(vk::DescriptorBufferInfo {
                            buffer: *vk_buffer,
                            offset: 0,
                            range: *size,
                        });
                    }
                }
                BoundResource::BufferRange {
                    buffer,
                    offset,
                    size,
                } => {
                    if let GpuBuffer::Vulkan {
                        buffer: vk_buffer, ..
                    } = buffer.gpu_handle()
                    {
                        buffer_infos.push(vk::DescriptorBufferInfo {
                            buffer: *vk_buffer,
                            offset: *offset,
                            range: *size,
                        });
                    }
                }
                BoundResource::Texture(texture) => {
                    if let GpuTexture::Vulkan { view, .. } = texture.gpu_handle() {
                        image_infos.push(vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: *view,
                            image_layout: sampled_layout,
                        });
                    }
                }
                BoundResource::Sampler(sampler) => {
                    if let GpuSampler::Vulkan {
                        sampler: vk_sampler,
                        ..
                    } = sampler.gpu_handle()
                    {
                        image_infos.push(vk::DescriptorImageInfo {
                            sampler: *vk_sampler,
                            image_view: vk::ImageView::null(),
                            image_layout: vk::ImageLayout::UNDEFINED,
                        });
                    }
                }
                BoundResource::CombinedTextureSampler { texture, sampler } => {
                    if let (
                        GpuTexture::Vulkan { view, .. },
                        GpuSampler::Vulkan {
                            sampler: vk_sampler,
                            ..
                        },
                    ) = (texture.gpu_handle(), sampler.gpu_handle())
                    {
                        image_infos.push(vk::DescriptorImageInfo {
                            sampler: *vk_sampler,
                            image_view: *view,
                            image_layout: sampled_layout,
                        });
                    }
                }
                // Acceleration structures need no buffer/image info; their
                // write is issued separately below (pNext-chained).
                BoundResource::AccelerationStructure(_) => {}
                // The bindless heap's set is written by registration (#117),
                // never here; user groups cannot even declare it (rejected
                // in create_binding_group).
                BoundResource::BindlessHeap(_) => {}
            }
        }

        // Build write descriptors referencing the info slices.
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        let mut buffer_idx = 0;
        let mut image_idx = 0;
        for entry in &descriptor.entries {
            // Layout is authoritative for the descriptor type. `create_binding_group`
            // already validated every binding is declared, so this always resolves.
            let binding_type = layout
                .entries
                .iter()
                .find(|e| e.binding == entry.binding)
                .map(|e| e.binding_type);

            let write = match &entry.resource {
                BoundResource::Buffer(_) | BoundResource::BufferRange { .. } => {
                    let info = &buffer_infos[buffer_idx..buffer_idx + 1];
                    buffer_idx += 1;
                    let descriptor_type = match binding_type {
                        Some(crate::materials::BindingType::StorageBuffer)
                        | Some(crate::materials::BindingType::StorageBufferReadOnly) => {
                            vk::DescriptorType::STORAGE_BUFFER
                        }
                        Some(crate::materials::BindingType::DynamicUniformBuffer) => {
                            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                        }
                        _ => vk::DescriptorType::UNIFORM_BUFFER,
                    };
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(entry.binding)
                        .descriptor_type(descriptor_type)
                        .buffer_info(info)
                }
                BoundResource::Texture(_) => {
                    let info = &image_infos[image_idx..image_idx + 1];
                    image_idx += 1;
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(entry.binding)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(info)
                }
                BoundResource::Sampler(_) => {
                    let info = &image_infos[image_idx..image_idx + 1];
                    image_idx += 1;
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(entry.binding)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(info)
                }
                BoundResource::CombinedTextureSampler { .. } => {
                    let info = &image_infos[image_idx..image_idx + 1];
                    image_idx += 1;
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(entry.binding)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(info)
                }
                BoundResource::AccelerationStructure(tlas) => {
                    // The AS handle rides in a pNext struct whose lifetime is
                    // this block, so its write is issued immediately instead
                    // of joining the batched `writes` (AS bindings are rare —
                    // one per ray-query material — so the extra call is noise).
                    let GpuAccelerationStructure::Vulkan { handle, .. } = tlas.gpu_handle() else {
                        log::warn!(
                            "binding {} references a non-Vulkan acceleration structure; \
                             descriptor left unwritten",
                            entry.binding
                        );
                        continue;
                    };
                    let handles = [*handle];
                    let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                        .acceleration_structures(&handles);
                    let mut write = vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(entry.binding)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .push_next(&mut as_write);
                    // Normally set by buffer_info/image_info; the AS count
                    // lives in the pNext struct, so set it explicitly.
                    write.descriptor_count = 1;
                    unsafe { self.device.update_descriptor_sets(&[write], &[]) };
                    continue;
                }
                // The bindless heap's descriptors are written by registration
                // (#117), never at group creation.
                BoundResource::BindlessHeap(_) => continue,
            };
            writes.push(write);
        }

        if !writes.is_empty() {
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
            }
        }
    }

    /// Device-clamped bindless heap capacities `(textures, samplers)` (#117);
    /// `(0, 0)` when `DeviceCapabilities::bindless` is false.
    pub fn bindless_capacities(&self) -> (u32, u32) {
        self.pipeline_manager.bindless_capacities()
    }

    /// Create the bindless heap's binding-group handle (#117): the single
    /// update-after-bind descriptor set, allocated from its dedicated pool.
    /// Called once per device by `GraphicsDevice::bindless_heap_group`.
    pub fn create_bindless_group(
        &self,
        layout: &crate::materials::BindingLayout,
    ) -> Result<super::GpuBindingGroup, GraphicsError> {
        let (descriptor_set, pool) = self.pipeline_manager.allocate_bindless_set(layout)?;
        Ok(super::GpuBindingGroup::Vulkan {
            device: self.device.clone(),
            descriptor_set,
            pool,
            // The heap pool is not part of the persistent chain, but Drop
            // frees the set into the specific `pool` handle — the chain's
            // mutex only provides the external synchronization.
            pools: Arc::clone(self.pipeline_manager.persistent_pools()),
        })
    }

    /// Write one sampled-texture descriptor into the bindless heap at array
    /// slot `index` (#117). Update-after-bind makes this legal while the
    /// heap is bound in flight, as long as slot `index` is not dynamically
    /// used by pending work — the engine-side allocator guarantees that
    /// (fresh or fence-recycled slots only).
    pub fn bindless_write_texture(
        &self,
        group: &super::GpuBindingGroup,
        binding: u32,
        index: u32,
        texture: &GpuTexture,
    ) -> Result<(), GraphicsError> {
        let super::GpuBindingGroup::Vulkan { descriptor_set, .. } = group else {
            return Err(GraphicsError::InvalidParameter(
                "bindless_write_texture: not a Vulkan binding group".to_string(),
            ));
        };
        let GpuTexture::Vulkan { view, .. } = texture else {
            return Err(GraphicsError::InvalidParameter(
                "bindless_write_texture: not a Vulkan texture".to_string(),
            ));
        };

        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(*view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(*descriptor_set)
            .dst_binding(binding)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&image_info);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        Ok(())
    }

    /// Write one sampler descriptor into the bindless heap at array slot
    /// `index` (#117). Same legality argument as
    /// [`bindless_write_texture`](Self::bindless_write_texture).
    pub fn bindless_write_sampler(
        &self,
        group: &super::GpuBindingGroup,
        binding: u32,
        index: u32,
        sampler: &super::GpuSampler,
    ) -> Result<(), GraphicsError> {
        let super::GpuBindingGroup::Vulkan { descriptor_set, .. } = group else {
            return Err(GraphicsError::InvalidParameter(
                "bindless_write_sampler: not a Vulkan binding group".to_string(),
            ));
        };
        let super::GpuSampler::Vulkan { sampler, .. } = sampler else {
            return Err(GraphicsError::InvalidParameter(
                "bindless_write_sampler: not a Vulkan sampler".to_string(),
            ));
        };

        let image_info = [vk::DescriptorImageInfo::default().sampler(*sampler)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(*descriptor_set)
            .dst_binding(binding)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&image_info);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        Ok(())
    }

    /// Create a GPU pipeline from a material descriptor.
    ///
    /// Compiles shaders, creates descriptor set layouts, pipeline layout,
    /// and the graphics or compute pipeline.
    pub fn create_pipeline(
        &self,
        descriptor: &crate::materials::MaterialDescriptor,
    ) -> Result<super::GpuPipeline, GraphicsError> {
        use crate::materials::ShaderStage;

        let is_compute = descriptor
            .shaders
            .iter()
            .any(|s| s.stage == ShaderStage::Compute);

        if is_compute {
            self.create_compute_pipeline_from_descriptor(descriptor)
        } else {
            self.create_graphics_pipeline_from_descriptor(descriptor)
        }
    }

    fn create_graphics_pipeline_from_descriptor(
        &self,
        descriptor: &crate::materials::MaterialDescriptor,
    ) -> Result<super::GpuPipeline, GraphicsError> {
        use crate::materials::ShaderStage;

        // Compile every graphics stage; `(vk stage, module, actual entry)`
        // in descriptor order. Stage-combination validity (vertex XOR mesh,
        // task implies mesh) was checked in `create_material`.
        let mut compiled: Vec<(vk::ShaderStageFlags, vk::ShaderModule, String)> = Vec::new();
        let compile_result: Result<(), GraphicsError> = (|| {
            for shader in &descriptor.shaders {
                let vk_stage = match shader.stage {
                    ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
                    ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
                    ShaderStage::Task => vk::ShaderStageFlags::TASK_EXT,
                    ShaderStage::Mesh => vk::ShaderStageFlags::MESH_EXT,
                    ShaderStage::Compute => continue,
                };
                let (module, actual_entry) = self.pipeline_manager.compile_shader(
                    &shader.source,
                    shader.stage,
                    &shader.entry_point,
                    shader.language,
                    &shader.defines,
                )?;
                compiled.push((vk_stage, module, actual_entry));
            }
            Ok(())
        })();
        // Modules compiled before a failing stage must not leak.
        if let Err(e) = compile_result {
            for (_, module, _) in &compiled {
                unsafe { self.device.destroy_shader_module(*module, None) };
            }
            return Err(e);
        }

        let is_mesh = compiled
            .iter()
            .any(|(stage, _, _)| *stage == vk::ShaderStageFlags::MESH_EXT);
        if !is_mesh
            && !compiled
                .iter()
                .any(|(stage, _, _)| *stage == vk::ShaderStageFlags::VERTEX)
        {
            for (_, module, _) in &compiled {
                unsafe { self.device.destroy_shader_module(*module, None) };
            }
            return Err(GraphicsError::ShaderCompilationFailed(
                "No vertex shader provided".into(),
            ));
        }

        let result = (|| {
            // Descriptor set layouts
            let descriptor_set_layouts: Vec<vk::DescriptorSetLayout> = descriptor
                .binding_layouts
                .iter()
                .map(|layout| self.pipeline_manager.create_descriptor_set_layout(layout))
                .collect::<Result<_, _>>()?;

            let pipeline_layout = self
                .pipeline_manager
                .create_pipeline_layout(&descriptor_set_layouts)?;

            let stages: Vec<(vk::ShaderStageFlags, vk::ShaderModule, &str)> = compiled
                .iter()
                .map(|(stage, module, entry)| (*stage, *module, entry.as_str()))
                .collect();
            // Mesh pipelines have no vertex input (#111); classic ones carry
            // the material's layout + topology.
            let vertex_input =
                (!is_mesh).then_some((&*descriptor.vertex_layout, descriptor.topology));

            let pipeline = self.pipeline_manager.create_graphics_pipeline(
                &stages,
                vertex_input,
                pipeline_layout,
                &descriptor.color_formats,
                descriptor.depth,
                descriptor.blend_state.as_ref(),
                descriptor.raster,
                descriptor.sample_count,
            )?;
            if let Some(label) = &descriptor.label {
                self.set_object_name(pipeline, label);
                self.set_object_name(pipeline_layout, &format!("{label} layout"));
            }

            Ok(super::GpuPipeline::Vulkan {
                device: self.device.clone(),
                pipeline,
                pipeline_layout,
                descriptor_set_layouts,
            })
        })();

        // Shader modules are baked into the pipeline; destroy them now
        // (success or failure alike).
        for (_, module, _) in &compiled {
            unsafe { self.device.destroy_shader_module(*module, None) };
        }

        result
    }

    fn create_compute_pipeline_from_descriptor(
        &self,
        descriptor: &crate::materials::MaterialDescriptor,
    ) -> Result<super::GpuPipeline, GraphicsError> {
        use crate::materials::ShaderStage;

        let mut compute_module = None;
        let mut compute_entry = String::from("main");

        for shader in &descriptor.shaders {
            if shader.stage == ShaderStage::Compute {
                let (module, actual_entry) = self.pipeline_manager.compile_shader(
                    &shader.source,
                    shader.stage,
                    &shader.entry_point,
                    shader.language,
                    &shader.defines,
                )?;
                compute_module = Some(module);
                compute_entry = actual_entry;
            }
        }

        let compute_module = compute_module.ok_or_else(|| {
            GraphicsError::ShaderCompilationFailed("No compute shader provided".into())
        })?;

        let descriptor_set_layouts: Vec<vk::DescriptorSetLayout> = descriptor
            .binding_layouts
            .iter()
            .map(|layout| self.pipeline_manager.create_descriptor_set_layout(layout))
            .collect::<Result<_, _>>()?;

        let pipeline_layout = self
            .pipeline_manager
            .create_pipeline_layout(&descriptor_set_layouts)?;

        let pipeline = self.pipeline_manager.create_compute_pipeline(
            compute_module,
            &compute_entry,
            pipeline_layout,
        )?;
        if let Some(label) = &descriptor.label {
            self.set_object_name(pipeline, label);
            self.set_object_name(pipeline_layout, &format!("{label} layout"));
        }

        // Shader module is baked into the pipeline; destroy it now.
        unsafe {
            self.device.destroy_shader_module(compute_module, None);
        }

        Ok(super::GpuPipeline::Vulkan {
            device: self.device.clone(),
            pipeline,
            pipeline_layout,
            descriptor_set_layouts,
        })
    }

    /// Create a fence for CPU-GPU synchronization.
    ///
    /// A "fence" is a wait value on a queue's timeline semaphore — no Vulkan
    /// object is created. `signaled` picks the initial value: `0` (every
    /// counter starts at 0, so trivially satisfied) or `u64::MAX` (never
    /// satisfied until `execute_graph` stamps a real submit value). The
    /// semaphore defaults to the graphics timeline; `execute_graph` re-stamps
    /// it with the routed queue's timeline at submit.
    pub fn create_fence(&self, signaled: bool) -> Result<GpuFence, GraphicsError> {
        Ok(GpuFence::Vulkan {
            device: self.device.clone(),
            semaphore: AtomicU64::new(self.queue_timeline.as_raw()),
            value: AtomicU64::new(if signaled { 0 } else { u64::MAX }),
        })
    }

    /// Wait for a fence to be signaled.
    ///
    /// Uses a 10-second timeout to prevent indefinite hangs on corrupted fences
    /// or GPU lockups. Timeout and device loss are returned as errors — the
    /// caller must not recycle resources guarded by this fence in that case.
    pub fn wait_fence(&self, fence: &GpuFence) -> Result<(), GraphicsError> {
        if let GpuFence::Vulkan {
            device,
            semaphore,
            value,
        } = fence
        {
            let semaphores = [vk::Semaphore::from_raw(semaphore.load(Ordering::Acquire))];
            let values = [value.load(Ordering::Acquire)];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            unsafe {
                match device.wait_semaphores(&wait_info, FENCE_WAIT_TIMEOUT_NS) {
                    Ok(()) => Ok(()),
                    Err(vk::Result::TIMEOUT) => Err(GraphicsError::Timeout(
                        "fence wait timed out after 10 s; GPU may be hung".into(),
                    )),
                    Err(vk::Result::ERROR_DEVICE_LOST) => {
                        self.report_device_lost();
                        Err(GraphicsError::DeviceLost)
                    }
                    Err(e) => Err(GraphicsError::Internal(format!(
                        "Fence wait failed: {:?}",
                        e
                    ))),
                }
            }
        } else {
            Ok(())
        }
    }

    /// Check if a fence is signaled (non-blocking).
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        if let GpuFence::Vulkan {
            device,
            semaphore,
            value,
        } = fence
        {
            let semaphore = vk::Semaphore::from_raw(semaphore.load(Ordering::Acquire));
            unsafe { device.get_semaphore_counter_value(semaphore) }
                .is_ok_and(|counter| counter >= value.load(Ordering::Acquire))
        } else {
            false
        }
    }

    /// Wait for a fence to be signaled with a timeout.
    ///
    /// Returns `Ok(true)` if the fence was signaled, `Ok(false)` on timeout,
    /// and an error on device loss or wait failure.
    pub fn wait_fence_timeout(
        &self,
        fence: &GpuFence,
        timeout: std::time::Duration,
    ) -> Result<bool, GraphicsError> {
        if let GpuFence::Vulkan {
            device,
            semaphore,
            value,
        } = fence
        {
            let semaphores = [vk::Semaphore::from_raw(semaphore.load(Ordering::Acquire))];
            let values = [value.load(Ordering::Acquire)];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            let timeout_ns = timeout.as_nanos() as u64;
            unsafe {
                match device.wait_semaphores(&wait_info, timeout_ns) {
                    Ok(()) => Ok(true),
                    Err(vk::Result::TIMEOUT) => Ok(false),
                    Err(vk::Result::ERROR_DEVICE_LOST) => {
                        self.report_device_lost();
                        Err(GraphicsError::DeviceLost)
                    }
                    Err(e) => Err(GraphicsError::Internal(format!(
                        "Fence wait failed: {:?}",
                        e
                    ))),
                }
            }
        } else {
            Ok(false)
        }
    }

    /// Signal a fence (for testing/dummy backend).
    pub fn signal_fence(&self, _fence: &GpuFence) {
        // Vulkan fences are signaled by the GPU, not the CPU
        // This is a no-op for the Vulkan backend
    }

    /// Resolve a graph's [`QueuePreference`] to an actual secondary queue,
    /// or `None` for the graphics queue (#47 phase 4, #89).
    ///
    /// The ladder: `Transfer` → dedicated transfer queue (transfer-only
    /// graphs — a transfer family cannot execute compute passes) → async
    /// compute → graphics; `AsyncCompute` → async compute (no graphics
    /// passes) → graphics. Each rung requires the queue to exist.
    ///
    /// Any secondary rung is additionally honored only when every texture
    /// the graph touches is declared cross-queue (#88): undeclared images
    /// are EXCLUSIVE, and accessing an EXCLUSIVE image from another queue
    /// family leaves its contents undefined per spec. Buffers are always
    /// CONCURRENT-shared and need no declaration. The fallback is correct
    /// but silently serializes work the caller wanted overlapped, so each
    /// offending texture warns once (the same graph legitimately runs
    /// this way on single-queue devices, hence no hard error).
    ///
    /// The transfer rung has one more gate ([`transfer_granularity_ok`]):
    /// on a transfer family coarser than 1×1×1 (AMD SDMA), only buffer copies
    /// and whole-subresource image copies are legal, so a graph with a
    /// partial image copy falls to async compute instead.
    ///
    /// [`transfer_granularity_ok`]: Self::transfer_granularity_ok
    fn transfer_granularity_ok(&self, graph: &RenderGraph) -> bool {
        let Some(g) = self.transfer_granularity else {
            return false; // no transfer queue; unreachable via route_graph
        };
        // A 1×1×1 family (NVIDIA copy engines) aligns every copy — nothing to
        // restrict. Only a coarse family needs the per-op check.
        if g.width <= 1 && g.height <= 1 && g.depth <= 1 {
            return true;
        }
        graph.passes().iter().all(|pass| {
            pass.as_transfer()
                .and_then(|p| p.transfer_config())
                .is_none_or(|cfg| cfg.operations.iter().all(op_ok_on_coarse_transfer))
        })
    }

    fn route_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
    ) -> Option<(QueueId, &SecondaryQueue)> {
        use crate::graph::QueuePreference;

        let target = match graph.queue_preference() {
            QueuePreference::Graphics => return None,
            _ if graph.has_graphics_passes() => return None,
            // A GenerateMipmaps op (vkCmdBlitImage) is legal only on a
            // graphics-capable queue — never transfer or async compute (#96).
            _ if graph.requires_graphics_queue() => return None,
            QueuePreference::AsyncCompute => self
                .async_compute
                .as_ref()
                .map(|q| (QueueId::AsyncCompute, q)),
            QueuePreference::Transfer => self
                .transfer_queue
                .as_ref()
                .filter(|_| graph.is_transfer_only() && self.transfer_granularity_ok(graph))
                .map(|q| (QueueId::Transfer, q))
                .or_else(|| {
                    self.async_compute
                        .as_ref()
                        .map(|q| (QueueId::AsyncCompute, q))
                }),
        };
        let (queue_id, queue) = target?;

        let mut eligible = true;
        for usage in compiled.pass_usages() {
            for decl in &usage.texture_usages {
                if decl.texture.descriptor().cross_queue {
                    continue;
                }
                eligible = false;
                let GpuTexture::Vulkan { id, .. } = decl.texture.gpu_handle() else {
                    continue;
                };
                if self.async_decline_warned.lock().insert(*id) {
                    log::warn!(
                        "{queue_id:?} queue hint declined: texture {:?} is not declared \
                         cross-queue (TextureDescriptor::with_cross_queue); the graph \
                         runs on the graphics queue instead",
                        decl.texture
                            .descriptor()
                            .label
                            .as_deref()
                            .unwrap_or("<unlabeled>"),
                    );
                }
            }
        }
        eligible.then_some((queue_id, queue))
    }

    /// Execute a compiled render graph.
    ///
    /// # Async Behavior
    ///
    /// This always returns immediately after `vkQueueSubmit` — it does **not**
    /// block on GPU completion in either case.
    ///
    /// Every submit signals the queue timeline semaphore's next value.
    ///
    /// - If `signal_fence` is provided: the submit's timeline value is stamped
    ///   into it, so the caller's waits/polls resolve against this submission.
    /// - If `signal_fence` is `None`: the value is signaled but recorded
    ///   nowhere. GPU lifetime of referenced resources is still guaranteed
    ///   because the `RenderGraph` holds `Arc`s to them until the slot's frame
    ///   fences are waited in
    ///   [`FramePipeline::begin_frame`](crate::pipeline::FramePipeline::begin_frame).
    ///
    /// # Queue routing (#47 phase 4, #89)
    ///
    /// A graph is routed to a secondary queue only when it asked for one
    /// ([`RenderGraph::set_queue_preference`]), its passes are legal on that
    /// queue, and the device actually has it — otherwise it walks down the
    /// fallback ladder (transfer → async compute → graphics) and ultimately
    /// runs on the graphics queue (the first-class fallback); see
    /// [`route_graph`](Self::route_graph). Ordering between submits is
    /// derived from resource usage by the persistent trackers: same-queue
    /// hazards become `vkCmdPipelineBarrier`s (valid across submits *within*
    /// one queue in submission order), cross-queue hazards become
    /// timeline-semaphore waits emitted on this submit (a pipeline barrier
    /// cannot cross queues). The remaining binary semaphores are the
    /// swapchain acquire/present handshake, on the graphics queue only.
    pub fn execute_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError> {
        profile_scope!("vulkan_execute_graph");

        // Route the graph. The swapchain handshake below is reachable only on
        // the graphics route: a swapchain write needs a graphics pass, which
        // disqualifies the graph from secondary routing.
        let route = self.route_graph(graph, compiled);
        let (queue_id, queue, queue_timeline, timeline_next) = match route {
            Some((id, q)) => (id, q.queue, q.timeline, &q.timeline_next),
            None => (
                QueueId::Graphics,
                self.graphics_queue,
                self.queue_timeline,
                &self.timeline_next,
            ),
        };

        // Allocate the frame's command buffer from the current slot's pool on
        // the routed queue (reset wholesale when the slot is recycled).
        let slot = self.current_slot.load(Ordering::Relaxed);
        let command_pool = match route {
            Some((_, q)) => q.command_pools[slot],
            None => self.frame_command_pools[slot],
        };
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| {
                GraphicsError::Internal(format!("Failed to allocate command buffer: {:?}", e))
            })?;

        let cmd = command_buffers[0];
        self.set_object_name(cmd, &format!("frame-cb {queue_id:?} slot{slot}"));

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { self.device.begin_command_buffer(cmd, &begin_info) }.map_err(|e| {
            GraphicsError::Internal(format!("Failed to begin command buffer: {:?}", e))
        })?;

        // Allocate this submit's timeline value up front: the trackers record
        // it as each resource's last-access value while barriers are
        // generated. Values allocated for failed submits leave a gap in the
        // signal sequence, which is fine — a later submit's higher signal
        // satisfies any wait on the gap value. (A gap value recorded in the
        // trackers makes a dependent submit wait for work that never ran —
        // over-synchronization, never corruption.)
        let timeline_value = timeline_next.fetch_add(1, Ordering::Relaxed);

        // Get all passes from the graph
        let passes = graph.passes();

        // Cross-queue timeline waits accumulated by the trackers: hazards
        // against work last touched by the OTHER queue, resolved by waiting
        // its timeline (pipeline barriers cannot cross queues).
        let mut waits = SubmitWaits::default();

        // Per-pass GPU timestamps (#95): reserve query indices and record the
        // pool reset + submit-begin marker before the passes. `None` when the
        // device lacks timestamp support or this queue has no query pool.
        let mut recording = self.timestamps.as_ref().and_then(|m| {
            m.lock().begin_submit(
                &self.device,
                &self.device,
                queue_id,
                slot,
                compiled.pass_order().len(),
                cmd,
            )
        });

        // Per-pass GPU crash breadcrumbs (#97): reserve markers and record the
        // buffer reset + submit-begin marker before the passes. `None` when
        // breadcrumbs are off or this queue has none.
        let mut breadcrumbs = self.breadcrumbs.as_ref().and_then(|m| {
            m.lock().begin_submit(
                queue_id,
                slot,
                compiled.pass_order().len(),
                timeline_value,
                cmd,
            )
        });

        // Process each pass in compiled order using pre-computed resource usages
        {
            profile_scope!("record_passes");
            let pass_usages = compiled.pass_usages();
            // One batch reused across passes: its merge maps keep their
            // capacity instead of two fresh HashMaps per pass.
            let mut barriers = BarrierBatch::new();
            for (i, handle) in compiled.pass_order().iter().enumerate() {
                let pass = &passes[handle.index()];

                // Generate barriers from pre-computed resource usage
                barriers.clear();
                self.generate_barriers_for_pass(
                    &mut barriers,
                    &pass_usages[i],
                    queue_id,
                    timeline_value,
                    &mut waits,
                );
                barriers.submit(&self.device, cmd);

                // Timestamp the pass's GPU work (begin before, end after).
                if let Some(rec) = recording.as_mut() {
                    rec.pass_begin(&self.device, cmd, i, pass.name());
                }
                // Breadcrumb the pass (begin before, end after).
                if let Some(bc) = breadcrumbs.as_mut() {
                    bc.pass_begin(cmd, i, pass.name());
                }
                if let Err(e) = self.encode_pass(cmd, pass) {
                    // The command buffer (with its reset + timestamp/breadcrumb
                    // writes) is abandoned; re-arm both so their next use resets.
                    if recording.is_some() {
                        self.abort_timestamps(queue_id, slot);
                    }
                    if breadcrumbs.is_some() {
                        self.abort_breadcrumbs(queue_id, slot);
                    }
                    self.abort_compaction(slot);
                    return Err(e);
                }
                if let Some(rec) = recording.as_mut() {
                    rec.pass_end(&self.device, cmd, i);
                }
                if let Some(bc) = breadcrumbs.as_ref() {
                    bc.pass_end(cmd, i);
                }
            }
        }

        // Submit-end marker, then hand the recording back for read-back at slot
        // retirement.
        if let Some(rec) = recording {
            rec.submit_end(&self.device, cmd);
            if let Some(m) = &self.timestamps {
                m.lock().finish_submit(slot, rec);
            }
        }

        // Submit-end breadcrumb, then hand the trace back so it survives for a
        // post-mortem if a later submit loses the device.
        if let Some(bc) = breadcrumbs {
            bc.submit_end(cmd);
            if let Some(m) = &self.breadcrumbs {
                m.lock().finish_submit(bc);
            }
        }

        // End command buffer
        if let Err(e) = unsafe { self.device.end_command_buffer(cmd) } {
            // Same as the encode error: the command buffer is abandoned, so
            // re-arm the query pool / breadcrumbs for a fresh reset.
            self.abort_timestamps(queue_id, slot);
            self.abort_breadcrumbs(queue_id, slot);
            self.abort_compaction(slot);
            return Err(GraphicsError::Internal(format!(
                "Failed to end command buffer: {:?}",
                e
            )));
        }

        if !waits.is_empty() {
            log::debug!(
                "submit {timeline_value} on {queue_id:?} waits: graphics={:?} async={:?} \
                 transfer={:?}",
                waits.get(QueueId::Graphics),
                waits.get(QueueId::AsyncCompute),
                waits.get(QueueId::Transfer),
            );
        }

        // The binary semaphores are the swapchain acquire/render-finished
        // handshake below (graphics queue only — a swapchain writer is never
        // routed async). The pair exists once per frame and is consumed
        // (`take`) by the single swapchain-writing graph — the scheduler
        // enforces at most one such graph per frame.
        //
        // If this graph writes the acquired swapchain image, this submit must
        // wait on `image_available` (so it does not write the image before the
        // presentation engine releases it) and signal `image_render_finished`
        // (so present transitions/presents only after rendering completes).
        let swapchain_pair = if graph.writes_swapchain() {
            self.take_swapchain_render_sync()
        } else {
            None
        };

        // Waits: the optional binary swapchain acquire, plus the cross-queue
        // timeline waits collected by the trackers. Under sync2 each wait is a
        // `VkSemaphoreSubmitInfo` carrying its own destination stage mask —
        // the swapchain acquire keeps COLOR_ATTACHMENT_OUTPUT, and cross-queue
        // timeline waits keep the ALL_COMMANDS destination scope that lets the
        // trackers treat a waited-on write as fully visible on this queue.
        let mut wait_infos: Vec<vk::SemaphoreSubmitInfo> =
            Vec::with_capacity(1 + barriers::QUEUE_COUNT);
        if let Some((image_available, _)) = swapchain_pair {
            wait_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(image_available)
                    .value(0) // binary semaphore: value ignored
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            );
        }
        let other_queue_timelines = [
            (QueueId::Graphics, self.queue_timeline),
            (
                QueueId::AsyncCompute,
                self.async_compute
                    .as_ref()
                    .map_or(vk::Semaphore::null(), |q| q.timeline),
            ),
            (
                QueueId::Transfer,
                self.transfer_queue
                    .as_ref()
                    .map_or(vk::Semaphore::null(), |q| q.timeline),
            ),
        ];
        for (wait_queue, timeline) in other_queue_timelines {
            if wait_queue == queue_id {
                continue; // own-queue hazards are pipeline barriers, not waits
            }
            if let Some(value) = waits.get(wait_queue) {
                debug_assert_ne!(
                    timeline,
                    vk::Semaphore::null(),
                    "tracker emitted a wait on a queue that does not exist"
                );
                wait_infos.push(
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(timeline)
                        .value(value)
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                );
            }
        }

        // Every submit signals its queue timeline's next value (in addition
        // to the binary `image_render_finished` when this graph writes the
        // swapchain). The value is stamped into `signal_fence` on success, so
        // the frontend `Fence` waits on (timeline, value). Signals use an
        // ALL_COMMANDS source scope: the semaphore signals after every stage.
        let mut signal_infos: Vec<vk::SemaphoreSubmitInfo> = Vec::with_capacity(2);
        signal_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(queue_timeline)
                .value(timeline_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
        if let Some((_, image_render_finished)) = swapchain_pair {
            signal_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(image_render_finished)
                    .value(0) // binary semaphore's value is ignored
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }

        // Submit the command buffer with the semaphores (vkQueueSubmit2).
        {
            profile_scope!("queue_submit");
            let cmd_buffer_infos: Vec<vk::CommandBufferSubmitInfo> = command_buffers
                .iter()
                .map(|&cb| vk::CommandBufferSubmitInfo::default().command_buffer(cb))
                .collect();
            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .command_buffer_infos(&cmd_buffer_infos)
                .signal_semaphore_infos(&signal_infos);

            let submit_result = unsafe {
                self.device
                    .queue_submit2(queue, &[submit_info], vk::Fence::null())
            };
            if let Err(e) = submit_result {
                // Nothing was enqueued: hand the swapchain semaphore pair back
                // so present can still consume `image_available` cleanly
                // instead of waiting forever on `image_render_finished`.
                if let Some((image_available, _)) = swapchain_pair {
                    self.restore_swapchain_render_sync(image_available);
                }
                // Device loss: read the breadcrumbs BEFORE the aborts below
                // clear this slot's traces, so the post-mortem sees them (#97).
                if e == vk::Result::ERROR_DEVICE_LOST {
                    self.report_device_lost();
                }
                // The command buffer never ran, so its query/breadcrumb
                // reset+writes didn't happen: drop this submit's records and
                // re-arm the pools so their next use resets before writing.
                self.abort_timestamps(queue_id, slot);
                self.abort_breadcrumbs(queue_id, slot);
                self.abort_compaction(slot);
                return Err(match e {
                    vk::Result::ERROR_DEVICE_LOST => GraphicsError::DeviceLost,
                    other => GraphicsError::Internal(format!(
                        "Failed to submit command buffer: {other:?}"
                    )),
                });
            }
        }

        // Submit enqueued: this fence is now satisfied exactly when the
        // routed queue's timeline reaches this submit's value.
        if let Some(GpuFence::Vulkan {
            semaphore, value, ..
        }) = signal_fence
        {
            semaphore.store(queue_timeline.as_raw(), Ordering::Release);
            value.store(timeline_value, Ordering::Release);
        }

        // No per-buffer freeing: the slot's pool is reset wholesale in
        // `advance_frame` once its fence signals.
        let _ = command_buffers;

        Ok(())
    }

    /// Generate barriers for a pass's resource usage into `batch`.
    ///
    /// This examines the texture and buffer usages declared by the pass, determines
    /// required layout transitions and memory barriers, and updates tracker state.
    /// The caller provides (and clears) the batch so its maps' capacity is
    /// reused across passes.
    fn generate_barriers_for_pass(
        &self,
        batch: &mut BarrierBatch,
        usage: &crate::graph::resource_usage::PassResourceUsage,
        queue: QueueId,
        submit_value: u64,
        waits: &mut SubmitWaits,
    ) {
        use crate::graph::resource_usage::TextureAccessMode;

        let mut tracker = self.layout_tracker.lock();

        // Generate texture (image) barriers
        for decl in &usage.texture_usages {
            // Get Vulkan image info from the texture
            let GpuTexture::Vulkan { image, id, .. } = decl.texture.gpu_handle() else {
                continue;
            };

            // Key by the stable id, not the raw image handle (handle reuse after
            // destruction would otherwise alias a stale tracked layout).
            let texture_id = TextureId::from_raw(*id);
            let required_layout = decl.access.to_layout();
            let (current_layout, cross_queue_write) =
                tracker.request_access(texture_id, required_layout, queue, submit_value, waits);

            // Determine aspect mask based on access mode and format
            let is_depth = matches!(
                decl.access,
                TextureAccessMode::DepthStencilWrite | TextureAccessMode::DepthStencilReadOnly
            ) || decl.texture.format().is_depth_stencil();

            let aspect_mask = if is_depth {
                if decl.texture.format().has_stencil() {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                } else {
                    vk::ImageAspectFlags::DEPTH
                }
            } else {
                vk::ImageAspectFlags::COLOR
            };

            // Add barrier if layout change is needed (`request_access`
            // already updated the tracked layout and recorded queue
            // ownership). When the previous write came from the OTHER queue
            // its availability comes from the timeline wait recorded in
            // `waits`, so the transition uses an empty source scope on this
            // queue — the previous layout's own stages may not even exist on
            // this queue's family (VUID 06461, caught on RDNA4 in #82).
            if cross_queue_write {
                batch.add_image_barrier_cross_queue(
                    texture_id,
                    *image,
                    current_layout,
                    required_layout,
                    aspect_mask,
                );
            } else {
                batch.add_image_barrier(
                    texture_id,
                    *image,
                    current_layout,
                    required_layout,
                    aspect_mask,
                );
            }
        }

        // Generate buffer barriers from per-buffer last-access tracking: the
        // tracker knows what wrote each buffer last (compute, transfer, …)
        // and which read scopes that write is already visible to, so reads
        // get a precise source scope and repeat reads emit nothing.
        let mut buffer_tracker = self.buffer_tracker.lock();
        for decl in &usage.buffer_usages {
            // Get Vulkan buffer info
            let GpuBuffer::Vulkan { buffer, .. } = decl.buffer.gpu_handle() else {
                continue;
            };

            let buffer_id = BufferId::from(*buffer);

            if let Some((src_stage, src_access)) =
                buffer_tracker.request_access(buffer_id, decl.access, queue, submit_value, waits)
            {
                batch.add_buffer_barrier(
                    buffer_id,
                    *buffer,
                    src_stage,
                    src_access,
                    // Tracker-augmented (task/mesh bits when enabled, #111) so
                    // the emitted destination scope matches what was recorded.
                    buffer_tracker.dst_stage(decl.access),
                    decl.access.dst_access_mask(),
                );
            }
        }
    }

    /// Write data to a host-visible (mapped) buffer.
    ///
    /// Returns an error if the buffer is not a Vulkan buffer or is not
    /// host-visible. There is deliberately NO device-local fallback (#89):
    /// GPU-only destinations must go through the frame graph
    /// (`TransferOperation::WriteBuffer`), which batches staging via the
    /// belt and lets the trackers derive synchronization.
    pub fn write_buffer(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), GraphicsError> {
        let GpuBuffer::Vulkan {
            allocation, size, ..
        } = buffer
        else {
            return Err(GraphicsError::Internal(
                "write_buffer called with non-Vulkan buffer".to_string(),
            ));
        };

        if data.is_empty() {
            return Ok(());
        }
        let end = offset.checked_add(data.len() as u64);
        if end.is_none_or(|end| end > *size) {
            return Err(GraphicsError::InvalidParameter(format!(
                "write_buffer range at offset {offset} ({} bytes) exceeds buffer size {size}",
                data.len()
            )));
        }

        // Host-visible buffer (MAP_WRITE/MAP_READ): direct mapped write.
        let guard = allocation.lock();
        let Some(alloc) = guard.as_ref() else {
            return Err(GraphicsError::Internal(
                "Buffer allocation is None".to_string(),
            ));
        };
        let Some(mapped_ptr) = alloc.mapped_ptr() else {
            return Err(GraphicsError::InvalidParameter(
                "write_buffer on a device-local buffer; upload through the frame graph \
                 via TransferOperation::write_buffer instead (#89)"
                    .to_string(),
            ));
        };
        unsafe {
            let dst = mapped_ptr.as_ptr().add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut u8, data.len());
        }
        Ok(())
    }

    /// Read a host-visible buffer's mapped memory. See the trait contract on
    /// [`GpuBackend::read_buffer`](crate::backend::GpuBackend::read_buffer).
    /// Non-blocking readback (Vulkan is native-only, so it fills `dst`
    /// synchronously — the caller drains after the frame fence, so the mapped
    /// memory is already GPU-complete — and clears the pending flag).
    pub fn read_buffer_async(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        size: u64,
        dst: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
        map_pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        match self.read_buffer(buffer, offset, size) {
            Ok(data) => *dst.lock().unwrap_or_else(|e| e.into_inner()) = data,
            Err(e) => log::error!("read_buffer_async failed: {e}"),
        }
        map_pending.store(false, std::sync::atomic::Ordering::Release);
    }

    pub fn read_buffer(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, GraphicsError> {
        let GpuBuffer::Vulkan {
            allocation,
            size: buf_size,
            ..
        } = buffer
        else {
            return Err(GraphicsError::Internal(
                "read_buffer called with non-Vulkan buffer".to_string(),
            ));
        };
        if size == 0 {
            return Ok(Vec::new());
        }
        if offset.checked_add(size).is_none_or(|end| end > *buf_size) {
            return Err(GraphicsError::InvalidParameter(format!(
                "read_buffer range at offset {offset} ({size} bytes) exceeds buffer size {buf_size}"
            )));
        }

        let guard = allocation.lock();
        let Some(allocation) = guard.as_ref() else {
            return Err(GraphicsError::Internal(
                "Buffer allocation is None".to_string(),
            ));
        };
        // No mapped pointer ⇒ device-local buffer. Reading it here would
        // silently return zeros; require a host-visible readback buffer.
        let Some(mapped_ptr) = allocation.mapped_ptr() else {
            return Err(GraphicsError::InvalidParameter(
                "read_buffer on a device-local buffer; copy it to a MAP_READ readback \
                 buffer via TransferOperation::ReadbackBuffer first"
                    .to_string(),
            ));
        };

        let mut result = vec![0u8; size as usize];
        unsafe {
            let src = mapped_ptr.as_ptr().add(offset as usize);
            std::ptr::copy_nonoverlapping(src as *const u8, result.as_mut_ptr(), size as usize);
        }
        Ok(result)
    }

    fn encode_pass(&self, cmd: vk::CommandBuffer, pass: &Pass) -> Result<(), GraphicsError> {
        profile_scope!("encode_pass");
        // Wrap the pass's GPU commands in a named, colour-coded debug region
        // (#123) so RenderDoc and validation output group them by frame-graph
        // pass name. A no-op without VK_EXT_debug_utils.
        self.begin_debug_label(cmd, pass.name(), pass_label_color(pass));
        let result = match pass {
            Pass::Graphics(graphics_pass) => self.encode_graphics_pass(cmd, graphics_pass),
            Pass::Transfer(transfer_pass) => self.encode_transfer_pass(cmd, transfer_pass),
            Pass::Compute(compute_pass) => self.encode_compute_pass(cmd, compute_pass),
            Pass::AccelerationStructureBuild(build_pass) => {
                self.encode_acceleration_structure_build_pass(cmd, build_pass)
            }
        };
        self.end_debug_label(cmd);
        result
    }

    fn encode_graphics_pass(
        &self,
        cmd: vk::CommandBuffer,
        pass: &crate::graph::GraphicsPass,
    ) -> Result<(), GraphicsError> {
        let Some(render_targets) = pass.render_targets() else {
            return Ok(());
        };

        // Build color attachments for dynamic rendering
        let color_attachments: Vec<vk::RenderingAttachmentInfo> = render_targets
            .color_attachments
            .iter()
            .filter_map(|attachment| {
                let (load_op, clear_value) =
                    conversion::convert_load_op_color(&attachment.load_op());
                let store_op = conversion::convert_store_op(&attachment.store_op());

                match &attachment.target {
                    RenderTarget::Texture { texture, .. } => {
                        let GpuTexture::Vulkan { view, .. } = texture.gpu_handle() else {
                            return None;
                        };

                        Some(
                            vk::RenderingAttachmentInfo::default()
                                .image_view(*view)
                                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .load_op(load_op)
                                .store_op(store_op)
                                .clear_value(clear_value),
                        )
                    }
                    RenderTarget::Surface { vulkan_view, .. } => {
                        // Use the Vulkan swapchain image view if available
                        if let Some(surface_view) = vulkan_view {
                            Some(
                                vk::RenderingAttachmentInfo::default()
                                    .image_view(surface_view.view())
                                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                    .load_op(load_op)
                                    .store_op(store_op)
                                    .clear_value(clear_value),
                            )
                        } else {
                            log::warn!(
                                "Pass '{}' has surface attachment but no Vulkan view available",
                                pass.name()
                            );
                            None
                        }
                    }
                }
            })
            .collect();

        // Build depth attachment if present
        let depth_attachment =
            render_targets
                .depth_stencil_attachment
                .as_ref()
                .and_then(|attachment| {
                    let (load_op, clear_value) =
                        conversion::convert_load_op_depth(&attachment.depth_load_op());
                    // A read-only aspect is not written at all: STORE_OP_NONE is
                    // the semantically correct op (#87) — storeOp=STORE is modeled
                    // as a write at LATE_FRAGMENT_TESTS and races same-pass
                    // sampling under sync validation. Unconditional: the enum is
                    // provided by VK_KHR_dynamic_rendering (core 1.3), which the
                    // baseline tier already requires of every selected device.
                    let store_op = if attachment.depth_read_only {
                        vk::AttachmentStoreOp::NONE
                    } else {
                        conversion::convert_store_op(&attachment.depth_store_op())
                    };
                    // A fully read-only attachment shares DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    // matching the layout the barrier system transitions it to (and any
                    // co-use sampling descriptor); otherwise it is a write target.
                    let ds_layout = ds_attachment_layout(attachment);

                    match &attachment.target {
                        RenderTarget::Texture { texture, .. } => {
                            let GpuTexture::Vulkan { view, .. } = texture.gpu_handle() else {
                                return None;
                            };

                            Some(
                                vk::RenderingAttachmentInfo::default()
                                    .image_view(*view)
                                    .image_layout(ds_layout)
                                    .load_op(load_op)
                                    .store_op(store_op)
                                    .clear_value(clear_value),
                            )
                        }
                        RenderTarget::Surface { vulkan_view, .. } => {
                            // Depth attachments are typically not surfaces, but handle for completeness
                            vulkan_view.as_ref().map(|surface_view| {
                                vk::RenderingAttachmentInfo::default()
                                    .image_view(surface_view.view())
                                    .image_layout(ds_layout)
                                    .load_op(load_op)
                                    .store_op(store_op)
                                    .clear_value(clear_value)
                            })
                        }
                    }
                });

        // Build stencil attachment if format has a stencil component
        let stencil_attachment = render_targets
            .depth_stencil_attachment
            .as_ref()
            .filter(|attachment| attachment.target.format().has_stencil())
            .and_then(|attachment| {
                let (load_op, clear_value) =
                    conversion::convert_load_op_stencil(&attachment.stencil_load_op());
                // Read-only stencil aspect: STORE_OP_NONE, mirroring the depth
                // aspect above (#87). Store ops are per-aspect, so a mixed
                // depth-read-only / stencil-write attachment keeps STORE here.
                let store_op = if attachment.stencil_read_only {
                    vk::AttachmentStoreOp::NONE
                } else {
                    conversion::convert_store_op(&attachment.stencil_store_op())
                };
                // Same image as the depth attachment, so it must carry the same layout.
                let ds_layout = ds_attachment_layout(attachment);

                match &attachment.target {
                    RenderTarget::Texture { texture, .. } => {
                        let GpuTexture::Vulkan { view, .. } = texture.gpu_handle() else {
                            return None;
                        };

                        Some(
                            vk::RenderingAttachmentInfo::default()
                                .image_view(*view)
                                .image_layout(ds_layout)
                                .load_op(load_op)
                                .store_op(store_op)
                                .clear_value(clear_value),
                        )
                    }
                    RenderTarget::Surface { vulkan_view, .. } => {
                        vulkan_view.as_ref().map(|surface_view| {
                            vk::RenderingAttachmentInfo::default()
                                .image_view(surface_view.view())
                                .image_layout(ds_layout)
                                .load_op(load_op)
                                .store_op(store_op)
                                .clear_value(clear_value)
                        })
                    }
                }
            });

        // Determine render area from the attachments (first color, else the
        // depth attachment for zero-color depth-only passes).
        let render_area = render_targets
            .dimensions()
            .map(|(w, h)| vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: w,
                    height: h,
                },
            })
            .unwrap_or_default();

        // NOTE: Layout transitions are now handled automatically by the barrier
        // generation system in execute_graph() before each pass is encoded.
        // Surface images (swapchain) are handled specially below.

        // Surface (swapchain) image barrier.
        //
        // Only the FIRST surface-writing pass of the frame transitions the
        // image, from `UNDEFINED` (valid from any actual layout; discards the
        // stale contents of the previously presented image — the engine always
        // renders full frames, so cross-frame `LoadOp::Load` on the surface is
        // not supported; use an offscreen target for accumulation). Later
        // passes emit a same-layout WAW barrier instead, so a UI overlay pass
        // with `LoadOp::Load` sees the scene pass's output rather than
        // re-discarding it.
        //
        // src stage is COLOR_ATTACHMENT_OUTPUT in both cases — the submit
        // waits the `image_available` semaphore at that stage, and a
        // `TOP_OF_PIPE` source would NOT be ordered after the semaphore wait
        // (the canonical WSI hazard: the transition could execute while the
        // presentation engine still reads the image).
        for attachment in &render_targets.color_attachments {
            if let RenderTarget::Surface {
                vulkan_view: Some(surface_view),
                ..
            } = &attachment.target
            {
                let first_write = {
                    let mut sync = self.swapchain_sync.lock();
                    let first = !sync.surface_transitioned;
                    sync.surface_transitioned = true;
                    first
                };
                let (old_layout, src_access) = if first_write {
                    (vk::ImageLayout::UNDEFINED, vk::AccessFlags2::NONE)
                } else {
                    (
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    )
                };

                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(src_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                    )
                    .old_layout(old_layout)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(surface_view.image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                let image_barriers = [barrier];
                let dependency_info =
                    vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
                unsafe {
                    self.device.cmd_pipeline_barrier2(cmd, &dependency_info);
                }
            }
        }

        // Create rendering info
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments);

        if let Some(ref depth) = depth_attachment {
            rendering_info = rendering_info.depth_attachment(depth);
        }

        if let Some(ref stencil) = stencil_attachment {
            rendering_info = rendering_info.stencil_attachment(stencil);
        }

        // Begin dynamic rendering
        unsafe {
            self.device.cmd_begin_rendering(cmd, &rendering_info);
        }

        // Set viewport with Y-flip and [0, 1] depth range to match wgpu/D3D conventions.
        // Vulkan's Y-axis points down (0=top, height=bottom), but wgpu/OpenGL use Y-up.
        // Using a negative height viewport flips the Y-axis, making the coordinate system
        // consistent with wgpu behavior. This requires VK_KHR_maintenance1 (Vulkan 1.1+).
        let viewport = if let Some(vp) = pass.viewport() {
            // Pass-level viewport override (apply Y-flip)
            vk::Viewport {
                x: vp.x,
                y: vp.y + vp.height, // Start at bottom of viewport
                width: vp.width,
                height: -vp.height, // Negative height flips Y
                min_depth: vp.min_depth,
                max_depth: vp.max_depth,
            }
        } else {
            vk::Viewport {
                x: 0.0,
                y: render_area.extent.height as f32, // Start at bottom
                width: render_area.extent.width as f32,
                height: -(render_area.extent.height as f32), // Negative height flips Y
                min_depth: 0.0,                              // Near plane maps to depth 0
                max_depth: 1.0,                              // Far plane maps to depth 1
            }
        };
        unsafe {
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
        }

        // Set scissor rect (use pass override or fall back to render area).
        // An explicit scissor is clamped to the render area — Vulkan rejects
        // negative offsets and out-of-bounds extents (VUID-vkCmdSetScissor).
        let default_scissor = if let Some(sr) = pass.scissor_rect() {
            let c = sr.clamped(render_area.extent.width, render_area.extent.height);
            vk::Rect2D {
                offset: vk::Offset2D {
                    x: c.x as i32,
                    y: c.y as i32,
                },
                extent: vk::Extent2D {
                    width: c.width,
                    height: c.height,
                },
            }
        } else {
            vk::Rect2D {
                offset: render_area.offset,
                extent: render_area.extent,
            }
        };
        unsafe {
            self.device.cmd_set_scissor(cmd, 0, &[default_scissor]);
        }

        // Encode draw commands
        for draw_cmd in pass.draw_commands() {
            self.encode_draw_command(cmd, draw_cmd, default_scissor, render_area.extent)?;
        }

        // Encode mesh-tasks draws (#111)
        for draw_cmd in pass.mesh_tasks_commands() {
            self.encode_mesh_tasks_command(cmd, draw_cmd, default_scissor, render_area.extent)?;
        }

        // End dynamic rendering
        unsafe {
            self.device.cmd_end_rendering(cmd);
        }

        // Note: Surface images are transitioned to PRESENT_SRC_KHR in present_vulkan_frame,
        // so we leave them in COLOR_ATTACHMENT_OPTIMAL here.

        Ok(())
    }

    fn encode_draw_command(
        &self,
        cmd: vk::CommandBuffer,
        draw_cmd: &crate::graph::DrawCommand,
        pass_scissor: vk::Rect2D,
        target_extent: vk::Extent2D,
    ) -> Result<(), GraphicsError> {
        let material_arc = draw_cmd.material.material();
        let mesh = &draw_cmd.mesh;

        // -- Pipeline: owned by Material, created at create_material() time --
        let super::GpuPipeline::Vulkan {
            pipeline,
            pipeline_layout,
            descriptor_set_layouts,
            ..
        } = material_arc.gpu_handle()
        else {
            log::warn!("Material has no Vulkan pipeline");
            return Ok(());
        };

        let pipeline = *pipeline;
        let pipeline_layout = *pipeline_layout;

        // Collect the cached descriptor sets — reuses scratch capacity.
        let scratch = &mut *self.encoder_scratch.lock();
        let VulkanEncoderScratch {
            descriptor_sets: scratch_ds_sets,
            ..
        } = scratch;

        let material_instance = &draw_cmd.material;
        let binding_groups = material_instance.binding_groups();

        // A zip would silently drop trailing groups on either side, drawing
        // with unbound descriptor sets (UB) — make the mismatch an error.
        if binding_groups.len() != descriptor_set_layouts.len() {
            return Err(GraphicsError::InvalidParameter(format!(
                "material instance provides {} binding group(s) but the material's pipeline \
                 layout declares {} descriptor set(s)",
                binding_groups.len(),
                descriptor_set_layouts.len()
            )));
        }

        // Pull each group's pre-built, cached descriptor set. Zero allocations,
        // zero writes — the sets were written once at group creation.
        scratch_ds_sets.clear();
        for (group_idx, group) in binding_groups.iter().enumerate() {
            // The set was allocated against the group's layout; validate it is
            // compatible with the material's declared set layout (deduped
            // content-equal layouts share the same VkDescriptorSetLayout, so
            // this holds by construction — the check guards against a group
            // built against the wrong layout).
            if let Some(mat_layout) = material_arc.binding_layouts().get(group_idx)
                && !Arc::ptr_eq(group.layout(), mat_layout)
                && !binding_layouts_compatible(group.layout(), mat_layout)
            {
                return Err(GraphicsError::InvalidParameter(format!(
                    "binding group {group_idx} was created against a layout incompatible \
                     with the material's descriptor set {group_idx}"
                )));
            }

            let super::GpuBindingGroup::Vulkan { descriptor_set, .. } = group.gpu_handle() else {
                return Err(GraphicsError::InvalidParameter(format!(
                    "binding group {group_idx} has no Vulkan descriptor set \
                     (resource from a different backend)"
                )));
            };
            scratch_ds_sets.push(*descriptor_set);
        }

        // Bind pipeline
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        }

        // Bind descriptor sets. Dynamic offsets are flattened across sets in
        // group order (matching `scratch_ds_sets`), one per dynamic binding.
        if !scratch_ds_sets.is_empty() {
            let dynamic_offsets: Vec<u32> =
                draw_cmd.dynamic_offsets.iter().flatten().copied().collect();
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    scratch_ds_sets,
                    &dynamic_offsets,
                );
            }
        }

        // Bind vertex buffers
        for (slot, buffer) in mesh.vertex_buffers().iter().enumerate() {
            if let GpuBuffer::Vulkan {
                buffer: vk_buffer, ..
            } = buffer.gpu_handle()
            {
                unsafe {
                    self.device.cmd_bind_vertex_buffers(
                        cmd,
                        slot as u32,
                        &[*vk_buffer],
                        &[mesh.vertex_offset(slot)],
                    );
                }
            }
        }

        // Set per-draw scissor rect if specified. Clamped to the target — a
        // negative/out-of-bounds rect (egui clip during resize) is a
        // validation error / UB on Vulkan otherwise.
        let custom_scissor = draw_cmd.scissor_rect.is_some();
        if let Some(scissor) = &draw_cmd.scissor_rect {
            let c = scissor.clamped(target_extent.width, target_extent.height);
            let vk_scissor = vk::Rect2D {
                offset: vk::Offset2D {
                    x: c.x as i32,
                    y: c.y as i32,
                },
                extent: vk::Extent2D {
                    width: c.width,
                    height: c.height,
                },
            };
            unsafe {
                self.device.cmd_set_scissor(cmd, 0, &[vk_scissor]);
            }
        }

        // Issue draw call
        if mesh.is_indexed() {
            // Bind index buffer
            if let Some(index_buffer) = mesh.index_buffer()
                && let GpuBuffer::Vulkan {
                    buffer: vk_buffer, ..
                } = index_buffer.gpu_handle()
            {
                let index_type = match mesh
                    .index_format()
                    .unwrap_or(crate::mesh::IndexFormat::Uint16)
                {
                    crate::mesh::IndexFormat::Uint16 => vk::IndexType::UINT16,
                    crate::mesh::IndexFormat::Uint32 => vk::IndexType::UINT32,
                };
                unsafe {
                    self.device.cmd_bind_index_buffer(
                        cmd,
                        *vk_buffer,
                        mesh.index_offset(),
                        index_type,
                    );
                }
            }

            unsafe {
                self.device.cmd_draw_indexed(
                    cmd,
                    mesh.index_count(),
                    draw_cmd.instance_count,
                    0,
                    0,
                    draw_cmd.first_instance,
                );
            }
        } else {
            unsafe {
                self.device.cmd_draw(
                    cmd,
                    mesh.vertex_count(),
                    draw_cmd.instance_count,
                    0,
                    draw_cmd.first_instance,
                );
            }
        }

        // Restore pass-level scissor if we set a per-draw one
        if custom_scissor {
            unsafe {
                self.device.cmd_set_scissor(cmd, 0, &[pass_scissor]);
            }
        }

        Ok(())
    }

    /// Encode a mesh-tasks draw (#111): bind the task?+mesh+fragment pipeline
    /// and its descriptor sets, then `vkCmdDrawMeshTasksEXT`. No vertex or
    /// index buffers exist — the mesh shader fetches geometry from the
    /// storage buffers bound through the material instance.
    fn encode_mesh_tasks_command(
        &self,
        cmd: vk::CommandBuffer,
        draw_cmd: &crate::graph::MeshTasksDrawCommand,
        pass_scissor: vk::Rect2D,
        target_extent: vk::Extent2D,
    ) -> Result<(), GraphicsError> {
        let Some(mesh_loader) = &self.mesh_loader else {
            return Err(GraphicsError::FeatureNotSupported(
                "mesh-tasks draws require DeviceCapabilities::mesh_shading \
                 (VK_EXT_mesh_shader, #111)"
                    .into(),
            ));
        };

        let material_arc = draw_cmd.material.material();
        let super::GpuPipeline::Vulkan {
            pipeline,
            pipeline_layout,
            descriptor_set_layouts,
            ..
        } = material_arc.gpu_handle()
        else {
            log::warn!("Material has no Vulkan pipeline");
            return Ok(());
        };
        let pipeline = *pipeline;
        let pipeline_layout = *pipeline_layout;

        // Collect the cached descriptor sets — reuses scratch capacity.
        let scratch = &mut *self.encoder_scratch.lock();
        let VulkanEncoderScratch {
            descriptor_sets: scratch_ds_sets,
            ..
        } = scratch;

        let material_instance = &draw_cmd.material;
        let binding_groups = material_instance.binding_groups();
        if binding_groups.len() != descriptor_set_layouts.len() {
            return Err(GraphicsError::InvalidParameter(format!(
                "material instance provides {} binding group(s) but the material's pipeline \
                 layout declares {} descriptor set(s)",
                binding_groups.len(),
                descriptor_set_layouts.len()
            )));
        }

        scratch_ds_sets.clear();
        for (group_idx, group) in binding_groups.iter().enumerate() {
            if let Some(mat_layout) = material_arc.binding_layouts().get(group_idx)
                && !Arc::ptr_eq(group.layout(), mat_layout)
                && !binding_layouts_compatible(group.layout(), mat_layout)
            {
                return Err(GraphicsError::InvalidParameter(format!(
                    "binding group {group_idx} was created against a layout incompatible \
                     with the material's descriptor set {group_idx}"
                )));
            }
            let super::GpuBindingGroup::Vulkan { descriptor_set, .. } = group.gpu_handle() else {
                return Err(GraphicsError::InvalidParameter(format!(
                    "binding group {group_idx} has no Vulkan descriptor set \
                     (resource from a different backend)"
                )));
            };
            scratch_ds_sets.push(*descriptor_set);
        }

        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        }

        if !scratch_ds_sets.is_empty() {
            let dynamic_offsets: Vec<u32> =
                draw_cmd.dynamic_offsets.iter().flatten().copied().collect();
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    scratch_ds_sets,
                    &dynamic_offsets,
                );
            }
        }

        // Per-draw scissor, clamped like the classic draw path.
        let custom_scissor = draw_cmd.scissor_rect.is_some();
        if let Some(scissor) = &draw_cmd.scissor_rect {
            let c = scissor.clamped(target_extent.width, target_extent.height);
            let vk_scissor = vk::Rect2D {
                offset: vk::Offset2D {
                    x: c.x as i32,
                    y: c.y as i32,
                },
                extent: vk::Extent2D {
                    width: c.width,
                    height: c.height,
                },
            };
            unsafe {
                self.device.cmd_set_scissor(cmd, 0, &[vk_scissor]);
            }
        }

        let [x, y, z] = draw_cmd.group_count;
        unsafe {
            mesh_loader.cmd_draw_mesh_tasks(cmd, x, y, z);
        }

        if custom_scissor {
            unsafe {
                self.device.cmd_set_scissor(cmd, 0, &[pass_scissor]);
            }
        }

        Ok(())
    }

    fn encode_transfer_pass(
        &self,
        cmd: vk::CommandBuffer,
        pass: &crate::graph::TransferPass,
    ) -> Result<(), GraphicsError> {
        let Some(config) = pass.transfer_config() else {
            return Ok(());
        };

        for operation in &config.operations {
            self.encode_transfer_operation(cmd, operation)?;
        }
        Ok(())
    }

    fn encode_transfer_operation(
        &self,
        cmd: vk::CommandBuffer,
        operation: &crate::graph::TransferOperation,
    ) -> Result<(), GraphicsError> {
        use crate::graph::{TransferOperation, validate_buffer_copy_alignment};

        match operation {
            TransferOperation::BufferToBuffer { src, dst, regions } => {
                let GpuBuffer::Vulkan {
                    buffer: src_buffer, ..
                } = src.gpu_handle()
                else {
                    return Ok(());
                };
                let GpuBuffer::Vulkan {
                    buffer: dst_buffer, ..
                } = dst.gpu_handle()
                else {
                    return Ok(());
                };

                validate_buffer_copy_alignment(regions)?;
                let copy_regions: Vec<vk::BufferCopy> = regions
                    .iter()
                    .map(|r| {
                        vk::BufferCopy::default()
                            .src_offset(r.src_offset)
                            .dst_offset(r.dst_offset)
                            .size(r.size)
                    })
                    .collect();

                unsafe {
                    self.device
                        .cmd_copy_buffer(cmd, *src_buffer, *dst_buffer, &copy_regions);
                }
            }
            TransferOperation::WriteBuffer {
                dst,
                dst_offset,
                data,
                src_range,
            } => {
                let bytes = data.get(src_range.clone()).ok_or_else(|| {
                    GraphicsError::InvalidParameter(format!(
                        "WriteBuffer src_range {:?} out of bounds (data len {})",
                        src_range,
                        data.len()
                    ))
                })?;
                let GpuBuffer::Vulkan {
                    buffer: dst_buffer, ..
                } = dst.gpu_handle()
                else {
                    return Ok(());
                };
                if bytes.is_empty() {
                    return Ok(());
                }
                // Same restriction as wgpu's copy path (COPY_BUFFER_ALIGNMENT);
                // enforced on both backends so a graph behaves identically.
                if !dst_offset.is_multiple_of(4) || !bytes.len().is_multiple_of(4) {
                    return Err(GraphicsError::InvalidParameter(format!(
                        "WriteBuffer requires 4-byte aligned dst_offset and size \
                         (got offset {}, size {})",
                        dst_offset,
                        bytes.len()
                    )));
                }
                // Copy via belt staging at THIS point in the command buffer,
                // so the write lands at the transfer pass's position in the
                // graph: passes ordered before it see the old contents,
                // passes after it see the new. A host memcpy here would
                // instead be visible to the whole frame — and race any
                // still-executing previous frame reading the same memory.
                // The belt chunk stays owned by this frame slot until its
                // fence signals (retired in `advance_frame`).
                let slot = self.current_slot.load(Ordering::Relaxed);
                let (staging, src_offset) = self.staging_belt.lock().write(
                    &self.device,
                    &mut self.allocator.lock(),
                    slot,
                    bytes,
                )?;
                let region = vk::BufferCopy::default()
                    .src_offset(src_offset)
                    .dst_offset(*dst_offset)
                    .size(bytes.len() as u64);
                unsafe {
                    self.device
                        .cmd_copy_buffer(cmd, staging, *dst_buffer, &[region]);
                }
            }
            // Drained by the frame pipeline after the fence (CPU read); nothing
            // to encode here.
            TransferOperation::ReadbackBuffer { .. } => {}
            TransferOperation::TextureToBuffer { src, dst, regions } => {
                let GpuTexture::Vulkan {
                    image: src_image, ..
                } = src.gpu_handle()
                else {
                    return Ok(());
                };
                let GpuBuffer::Vulkan {
                    buffer: dst_buffer, ..
                } = dst.gpu_handle()
                else {
                    return Ok(());
                };

                // NOTE: Layout transitions are now handled automatically by the barrier
                // generation system in execute_graph() before each pass is encoded.

                let copy_regions = build_buffer_image_copies(
                    src.format(),
                    src.dimension(),
                    regions,
                    "TextureToBuffer",
                )?;

                unsafe {
                    self.device.cmd_copy_image_to_buffer(
                        cmd,
                        *src_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        *dst_buffer,
                        &copy_regions,
                    );
                }
            }
            TransferOperation::BufferToTexture { src, dst, regions } => {
                let GpuBuffer::Vulkan {
                    buffer: src_buffer, ..
                } = src.gpu_handle()
                else {
                    return Ok(());
                };
                let GpuTexture::Vulkan {
                    image: dst_image, ..
                } = dst.gpu_handle()
                else {
                    return Ok(());
                };

                // NOTE: Layout transitions are now handled automatically by the barrier
                // generation system in execute_graph() before each pass is encoded.

                let copy_regions = build_buffer_image_copies(
                    dst.format(),
                    dst.dimension(),
                    regions,
                    "BufferToTexture",
                )?;

                unsafe {
                    self.device.cmd_copy_buffer_to_image(
                        cmd,
                        *src_buffer,
                        *dst_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &copy_regions,
                    );
                }
            }
            TransferOperation::TextureToTexture { src, dst, regions } => {
                let GpuTexture::Vulkan {
                    image: src_image, ..
                } = src.gpu_handle()
                else {
                    return Ok(());
                };
                let GpuTexture::Vulkan {
                    image: dst_image, ..
                } = dst.gpu_handle()
                else {
                    return Ok(());
                };

                // NOTE: Layout transitions are now handled automatically by the barrier
                // generation system in execute_graph() before each pass is encoded.

                // Aspect follows the format (a depth copy with COLOR aspect is
                // invalid); image-to-image copies may cover both depth and
                // stencil at once, unlike buffer<->image copies.
                let src_aspect = image_aspect_mask(src.format());
                let dst_aspect = image_aspect_mask(dst.format());

                let copy_regions: Vec<vk::ImageCopy> = regions
                    .iter()
                    .map(|r| {
                        // For array-like textures origin.z addresses the layer
                        // and extent.depth the layer count (matching wgpu);
                        // for 1D/2D/3D they are a real z offset and depth.
                        let src_loc = resolve_layer_z(src.dimension(), r.src.origin.z, r.extent);
                        let dst_loc = resolve_layer_z(dst.dimension(), r.dst.origin.z, r.extent);
                        // vk::ImageCopy has one extent; a 3D side keeps real
                        // depth, array<->array copies move layers instead.
                        let depth = src_loc.extent_depth.max(dst_loc.extent_depth);
                        vk::ImageCopy::default()
                            .src_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: src_aspect,
                                mip_level: r.src.mip_level,
                                base_array_layer: src_loc.base_layer,
                                layer_count: src_loc.layer_count,
                            })
                            .src_offset(vk::Offset3D {
                                x: r.src.origin.x as i32,
                                y: r.src.origin.y as i32,
                                z: src_loc.z_offset,
                            })
                            .dst_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: dst_aspect,
                                mip_level: r.dst.mip_level,
                                base_array_layer: dst_loc.base_layer,
                                layer_count: dst_loc.layer_count,
                            })
                            .dst_offset(vk::Offset3D {
                                x: r.dst.origin.x as i32,
                                y: r.dst.origin.y as i32,
                                z: dst_loc.z_offset,
                            })
                            .extent(vk::Extent3D {
                                width: r.extent.width,
                                height: r.extent.height,
                                depth,
                            })
                    })
                    .collect();

                unsafe {
                    self.device.cmd_copy_image(
                        cmd,
                        *src_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        *dst_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &copy_regions,
                    );
                }
            }
            TransferOperation::GenerateMipmaps { texture } => {
                let GpuTexture::Vulkan { image, .. } = texture.gpu_handle() else {
                    return Ok(());
                };
                let image = *image;
                let mip_count = texture.mip_level_count();
                // A single-mip texture (ineligible format / cap off) is a no-op.
                if mip_count <= 1 {
                    return Ok(());
                }

                let aspect = image_aspect_mask(texture.format());
                let extent = texture.size();
                let mut mip_w = extent.width.max(1) as i32;
                let mut mip_h = extent.height.max(1) as i32;

                // The whole image arrives in TRANSFER_DST (the tracker-declared
                // TransferWrite after the mip0 upload). Blit each mip i-1 → i
                // with a linear filter, transitioning the source mip to
                // TRANSFER_SRC immediately before it is read — that barrier is
                // also the write→read dependency on the blit that produced it.
                let subresource = |base: u32, count: u32| vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: base,
                    level_count: count,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                };
                for i in 1..mip_count {
                    let src_level = i - 1;
                    let to_src = vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image)
                        .subresource_range(subresource(src_level, 1));
                    let barriers = [to_src];
                    let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                    unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };

                    let dst_w = (mip_w / 2).max(1);
                    let dst_h = (mip_h / 2).max(1);
                    let blit = vk::ImageBlit::default()
                        .src_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: aspect,
                            mip_level: src_level,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .src_offsets([
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: mip_w,
                                y: mip_h,
                                z: 1,
                            },
                        ])
                        .dst_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: aspect,
                            mip_level: i,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .dst_offsets([
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: dst_w,
                                y: dst_h,
                                z: 1,
                            },
                        ]);
                    unsafe {
                        self.device.cmd_blit_image(
                            cmd,
                            image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[blit],
                            vk::Filter::LINEAR,
                        );
                    }
                    mip_w = dst_w;
                    mip_h = dst_h;
                }

                // CRITICAL (#96): the tracker models one layout per image, so the
                // op must END with every mip in the SAME layout it started in
                // (TRANSFER_DST). Mips 0..mip_count-1 were transitioned to
                // TRANSFER_SRC as blit sources; the last mip is still
                // TRANSFER_DST. Transition the sources back so the whole-image
                // tracker model stays truthful and the next barrier (→ SHADER_READ
                // before sampling) is correct.
                let restore = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(subresource(0, mip_count - 1));
                let barriers = [restore];
                let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };
            }
        }
        Ok(())
    }

    fn encode_compute_pass(
        &self,
        cmd: vk::CommandBuffer,
        pass: &crate::graph::ComputePass,
    ) -> Result<(), GraphicsError> {
        if !pass.has_dispatches() {
            return Ok(());
        }

        for dispatch_cmd in pass.dispatch_commands() {
            let material_arc = dispatch_cmd.material.material();

            // -- Pipeline: owned by Material, created at create_material() time --
            let super::GpuPipeline::Vulkan {
                pipeline,
                pipeline_layout,
                descriptor_set_layouts,
                ..
            } = material_arc.gpu_handle()
            else {
                log::warn!("Material has no Vulkan pipeline");
                continue;
            };

            let pipeline = *pipeline;
            let pipeline_layout = *pipeline_layout;

            let scratch = &mut *self.encoder_scratch.lock();
            let VulkanEncoderScratch {
                descriptor_sets: scratch_ds_sets,
                ..
            } = scratch;

            let material_instance = &dispatch_cmd.material;
            let binding_groups = material_instance.binding_groups();

            // A zip would silently drop trailing groups on either side,
            // dispatching with unbound descriptor sets (UB) — error instead.
            if binding_groups.len() != descriptor_set_layouts.len() {
                return Err(GraphicsError::InvalidParameter(format!(
                    "material instance provides {} binding group(s) but the material's \
                     pipeline layout declares {} descriptor set(s)",
                    binding_groups.len(),
                    descriptor_set_layouts.len()
                )));
            }

            // Pull each group's cached descriptor set (written once at creation).
            scratch_ds_sets.clear();
            for (group_idx, group) in binding_groups.iter().enumerate() {
                if let Some(mat_layout) = material_arc.binding_layouts().get(group_idx)
                    && !Arc::ptr_eq(group.layout(), mat_layout)
                    && !binding_layouts_compatible(group.layout(), mat_layout)
                {
                    return Err(GraphicsError::InvalidParameter(format!(
                        "binding group {group_idx} was created against a layout incompatible \
                         with the material's descriptor set {group_idx}"
                    )));
                }

                let super::GpuBindingGroup::Vulkan { descriptor_set, .. } = group.gpu_handle()
                else {
                    return Err(GraphicsError::InvalidParameter(format!(
                        "binding group {group_idx} has no Vulkan descriptor set \
                         (resource from a different backend)"
                    )));
                };
                scratch_ds_sets.push(*descriptor_set);
            }

            // Bind pipeline
            unsafe {
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            }

            // Bind descriptor sets
            if !scratch_ds_sets.is_empty() {
                unsafe {
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline_layout,
                        0,
                        scratch_ds_sets,
                        &[],
                    );
                }
            }

            // Dispatch
            unsafe {
                self.device.cmd_dispatch(
                    cmd,
                    dispatch_cmd.workgroup_count_x,
                    dispatch_cmd.workgroup_count_y,
                    dispatch_cmd.workgroup_count_z,
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        AccelerationStructureBuildPass, ComputePass, GraphicsPass, Pass, TransferPass,
    };

    /// #123: each pass type gets a distinct, stable debug-label colour so
    /// RenderDoc regions are visually separable.
    #[test]
    fn pass_label_colors_are_distinct_per_type() {
        let colors = [
            pass_label_color(&Pass::Graphics(GraphicsPass::new("g".into()))),
            pass_label_color(&Pass::Transfer(TransferPass::new("t".into()))),
            pass_label_color(&Pass::Compute(ComputePass::new("c".into()))),
            pass_label_color(&Pass::AccelerationStructureBuild(
                AccelerationStructureBuildPass::new("a".into()),
            )),
        ];
        for i in 0..colors.len() {
            assert_eq!(colors[i][3], 1.0, "alpha must be opaque");
            for j in (i + 1)..colors.len() {
                assert_ne!(colors[i], colors[j], "pass colours must differ by type");
            }
        }
    }
}
