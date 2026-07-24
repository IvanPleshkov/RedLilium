//! Vulkan swapchain implementation.
//!
//! This module contains the Vulkan-specific swapchain and surface texture handling.

use std::sync::Arc;

use ash::vk;

use super::conversion::{convert_present_mode, convert_texture_format};
use super::{VulkanBackend, VulkanImageView, VulkanSurfaceTextureView};
use crate::error::GraphicsError;
use crate::swapchain::SurfaceConfiguration;

/// Vulkan swapchain resources.
pub struct VulkanSwapchain {
    /// Number of frames in flight (from surface configuration).
    pub(crate) frames_in_flight: usize,
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) image_views: Vec<vk::ImageView>,
    /// Prebuilt shared wrappers over `image_views`, cloned per acquire —
    /// avoids an Arc allocation + device clone every frame. Destruction is
    /// owned by `image_views` (the wrapper's Drop is a no-op for swapchain
    /// views).
    image_view_wrappers: Vec<Arc<VulkanImageView>>,
    #[allow(dead_code)] // Reserved for future use
    pub(crate) format: vk::Format,
    #[allow(dead_code)] // Reserved for future use
    pub(crate) extent: vk::Extent2D,
    pub(crate) current_image_index: u32,
    /// Semaphores signaled when swapchain image is available (one per frame in flight).
    pub(crate) image_available_semaphores: Vec<vk::Semaphore>,
    /// Semaphores signaled by the render submit that writes the swapchain image,
    /// waited on by the present layout-transition submit (one per frame in
    /// flight). This is what orders "render finished" → "present barrier".
    pub(crate) image_render_finished_semaphores: Vec<vk::Semaphore>,
    /// Semaphores signaled when the present barrier is complete, waited on by
    /// `queue_present` — ONE PER SWAPCHAIN IMAGE, indexed by acquired image
    /// index. Present completion is not gated by the frame fence, so a per-frame
    /// semaphore would be re-signaled while a prior present still holds it
    /// (VUID-vkQueueSubmit2-semaphore-03868); a per-image semaphore is
    /// guaranteed idle by the time its image is re-acquired.
    pub(crate) render_finished_semaphores: Vec<vk::Semaphore>,
    /// Fences for CPU-GPU synchronization (one per frame in flight).
    pub(crate) in_flight_fences: Vec<vk::Fence>,
    /// Command buffers for presentation (one per frame in flight).
    pub(crate) present_command_buffers: Vec<vk::CommandBuffer>,
    /// Current frame index (cycles through frames in flight).
    pub(crate) current_frame: usize,
    /// Whether presents are id-tagged and paced with `vkWaitForPresentKHR`
    /// (`VK_KHR_present_wait` available and a blocking present mode). Without
    /// pacing the driver queues presents as deep as it likes; after any hitch
    /// the CPU then runs several frames ahead (near-zero deltas) and stalls
    /// whole refresh intervals catching up — a visible cadence oscillation.
    pub(crate) present_wait: bool,
    /// Next present id for pacing, allocated per acquire. Ids must only be
    /// monotonic **per swapchain object**, so every chain restarts at 1 —
    /// which also guarantees a wait never references an id from a previous
    /// (retired) chain: NVIDIA's `vkWaitForPresentKHR` has been observed to
    /// hang far past its timeout waiting on an id the new chain will never
    /// reach (frozen app → TDR → device lost during resize storms).
    next_present_id: u64,
    /// Device handle for cleanup.
    device: ash::Device,
    /// Swapchain loader for cleanup.
    swapchain_loader: ash::khr::swapchain::Device,
    /// Command pool for freeing command buffers.
    command_pool: vk::CommandPool,
    /// Whether resources have been destroyed.
    destroyed: bool,
}

impl VulkanSwapchain {
    /// Create a new Vulkan swapchain.
    ///
    /// `old_swapchain` (if not null) is passed to `vkCreateSwapchainKHR` so the
    /// driver can reuse resources and recreate seamlessly on resize. On success
    /// the old swapchain is retired and must be destroyed by the caller; on
    /// failure it remains valid.
    pub fn new(
        vulkan_backend: &VulkanBackend,
        surface: vk::SurfaceKHR,
        config: &SurfaceConfiguration,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<Self, GraphicsError> {
        // Get surface capabilities
        let capabilities = vulkan_backend.get_surface_capabilities(surface)?;

        // Choose format. The requested format must be supported exactly: all
        // pipelines are compiled against it, so silently substituting another
        // format (and color space) would mismatch every render pass.
        let formats = vulkan_backend.get_surface_formats(surface)?;
        if formats.is_empty() {
            return Err(GraphicsError::ResourceCreationFailed(
                "No surface formats available".to_string(),
            ));
        }
        let requested_format = convert_texture_format(config.format);
        // Float formats are HDR: the app writes linear values with 1.0 = SDR
        // white, which is the EXTENDED_SRGB_LINEAR_EXT contract (what wgpu
        // configures on Metal). Pairing them with SRGB_NONLINEAR makes the
        // display sRGB-decode linear data — a visibly too-dark picture.
        // Unorm formats keep preferring the standard sRGB color space over
        // e.g. Display-P3 variants on macOS.
        let wants_linear_extended = matches!(
            requested_format,
            vk::Format::R16G16B16A16_SFLOAT | vk::Format::R32G32B32A32_SFLOAT
        );
        let surface_format = formats
            .iter()
            .filter(|f| f.format == requested_format)
            .min_by_key(|f| match f.color_space {
                vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT if wants_linear_extended => 0,
                vk::ColorSpaceKHR::SRGB_NONLINEAR => 1,
                _ => 2,
            })
            .cloned()
            .ok_or_else(|| {
                GraphicsError::InvalidParameter(format!(
                    "surface does not support format {:?}; supported: {:?}",
                    config.format,
                    formats.iter().map(|f| f.format).collect::<Vec<_>>()
                ))
            })?;
        if wants_linear_extended
            && surface_format.color_space != vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT
        {
            log::warn!(
                "HDR swapchain format {:?} paired with {:?} (EXTENDED_SRGB_LINEAR_EXT \
                 unavailable — is VK_EXT_swapchain_colorspace enabled?); linear output \
                 will be decoded by the display and render too dark",
                surface_format.format,
                surface_format.color_space
            );
        }

        // Choose present mode
        let present_modes = vulkan_backend.get_surface_present_modes(surface)?;
        let present_mode = convert_present_mode(config.present_mode);
        let present_mode = if present_modes.contains(&present_mode) {
            present_mode
        } else {
            vk::PresentModeKHR::FIFO // Always available
        };

        // Choose extent
        let extent = if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: config.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: config.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        };

        // The spec forces `current_extent` when the surface reports one; a
        // caller that asked for a different size will record frames at a size
        // the swapchain images do not have — an out-of-bounds render area is
        // device-lost territory, so make the clamp loud.
        if extent.width != config.width || extent.height != config.height {
            log::warn!(
                "Swapchain clamped to the surface's current extent {}x{} (requested {}x{}); \
                 frames must be recorded at the actual extent",
                extent.width,
                extent.height,
                config.width,
                config.height
            );
        }

        // A zero extent (minimized window) is invalid for vkCreateSwapchainKHR.
        // Report it as a parameter error so the caller skips reconfiguring
        // until the window is restored.
        if extent.width == 0 || extent.height == 0 {
            return Err(GraphicsError::InvalidParameter(format!(
                "surface extent is {}x{} (window minimized?); swapchain creation skipped",
                extent.width, extent.height
            )));
        }

        // Choose image count (prefer triple buffering)
        let image_count =
            (capabilities.min_image_count + 1).min(if capabilities.max_image_count > 0 {
                capabilities.max_image_count
            } else {
                u32::MAX
            });

        // Pick a composite alpha mode the surface actually supports. OPAQUE is
        // preferred, but some platforms (Wayland, Android) only report
        // INHERIT or PRE_MULTIPLIED — hardcoding OPAQUE fails creation there.
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        ]
        .into_iter()
        .find(|&mode| capabilities.supported_composite_alpha.contains(mode))
        .ok_or_else(|| {
            GraphicsError::ResourceCreationFailed(format!(
                "surface reports no usable composite alpha modes: {:?}",
                capabilities.supported_composite_alpha
            ))
        })?;

        // COLOR_ATTACHMENT is mandatory for rendering; TRANSFER_SRC is added
        // opportunistically so the swapchain image can be read back
        // (screenshots, tests).
        if !capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        {
            return Err(GraphicsError::ResourceCreationFailed(
                "surface does not support COLOR_ATTACHMENT usage".to_string(),
            ));
        }
        let mut image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            image_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        }

        // Create swapchain
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(image_usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe {
            vulkan_backend
                .swapchain_loader()
                .create_swapchain(&swapchain_create_info, None)
        }
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to create swapchain: {:?}", e))
        })?;

        // Get swapchain images
        let images = unsafe {
            vulkan_backend
                .swapchain_loader()
                .get_swapchain_images(swapchain)
        }
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to get swapchain images: {:?}",
                e
            ))
        })?;

        for (i, &image) in images.iter().enumerate() {
            vulkan_backend.set_object_name(image, &format!("swapchain image[{i}]"));
        }

        // Create image views
        let image_views: Vec<vk::ImageView> = images
            .iter()
            .enumerate()
            .map(|(i, &image)| {
                let view =
                    vulkan_backend.create_swapchain_image_view(image, surface_format.format)?;
                vulkan_backend.set_object_name(view, &format!("swapchain image-view[{i}]"));
                Ok(view)
            })
            .collect::<Result<Vec<_>, GraphicsError>>()?;

        let image_view_wrappers: Vec<Arc<VulkanImageView>> = image_views
            .iter()
            .map(|&view| Arc::new(VulkanImageView::new(vulkan_backend.device().clone(), view)))
            .collect();

        // Create synchronization primitives for frames in flight
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let frames_in_flight = config.frames_in_flight;
        let mut image_available_semaphores = Vec::with_capacity(frames_in_flight);
        let mut image_render_finished_semaphores = Vec::with_capacity(frames_in_flight);
        // Present-wait semaphores are per swapchain image (see field doc), not
        // per frame in flight.
        let mut render_finished_semaphores = Vec::with_capacity(images.len());
        let mut in_flight_fences = Vec::with_capacity(frames_in_flight);

        for i in 0..frames_in_flight {
            let image_available = unsafe {
                vulkan_backend
                    .device()
                    .create_semaphore(&semaphore_info, None)
            }
            .map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to create image available semaphore: {:?}",
                    e
                ))
            })?;
            // Name the per-frame sync objects (#123): breadcrumb and validation
            // output is far more readable with named semaphores/fences.
            vulkan_backend.set_object_name(image_available, &format!("swapchain acquire sem[{i}]"));
            image_available_semaphores.push(image_available);

            let image_render_finished = unsafe {
                vulkan_backend
                    .device()
                    .create_semaphore(&semaphore_info, None)
            }
            .map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to create image render-finished semaphore: {:?}",
                    e
                ))
            })?;
            vulkan_backend.set_object_name(
                image_render_finished,
                &format!("swapchain render-finished sem[{i}]"),
            );
            image_render_finished_semaphores.push(image_render_finished);

            let fence = unsafe { vulkan_backend.device().create_fence(&fence_info, None) }
                .map_err(|e| {
                    GraphicsError::ResourceCreationFailed(format!(
                        "Failed to create in-flight fence: {:?}",
                        e
                    ))
                })?;
            vulkan_backend.set_object_name(fence, &format!("swapchain in-flight fence[{i}]"));
            in_flight_fences.push(fence);
        }

        // Present-wait semaphores: one per swapchain image (see field doc).
        for i in 0..images.len() {
            let render_finished = unsafe {
                vulkan_backend
                    .device()
                    .create_semaphore(&semaphore_info, None)
            }
            .map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to create render finished semaphore: {:?}",
                    e
                ))
            })?;
            vulkan_backend.set_object_name(render_finished, &format!("swapchain present sem[{i}]"));
            render_finished_semaphores.push(render_finished);
        }

        // Allocate command buffers for presentation (one per frame in flight)
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(vulkan_backend.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frames_in_flight as u32);

        let present_command_buffers = unsafe {
            vulkan_backend
                .device()
                .allocate_command_buffers(&alloc_info)
        }
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to allocate present command buffers: {:?}",
                e
            ))
        })?;
        for (i, cb) in present_command_buffers.iter().enumerate() {
            vulkan_backend.set_object_name(*cb, &format!("present-cb[{i}]"));
        }

        // Present pacing: only under a blocking present mode. FIFO can only
        // stretch an interval (VRR included), never show a frame for less
        // than one, so waiting on the previous present is always sound there;
        // IMMEDIATE/MAILBOX presents are not display-quantized and need no
        // bounding. REDLILIUM_NO_PRESENT_WAIT=1 is the diagnostic escape
        // hatch: pacing is a driver-facing behavior (NVIDIA and AMD differ),
        // and ruling it in or out of a repro must not require a rebuild.
        let present_wait = vulkan_backend.present_wait_loader().is_some()
            && matches!(
                present_mode,
                vk::PresentModeKHR::FIFO | vk::PresentModeKHR::FIFO_RELAXED
            )
            && !std::env::var("REDLILIUM_NO_PRESENT_WAIT").is_ok_and(|v| v == "1");

        log::info!(
            "Created Vulkan swapchain: {}x{} with {} images, {} frames in flight, {:?} / {:?}, \
             present pacing: {}",
            extent.width,
            extent.height,
            images.len(),
            frames_in_flight,
            surface_format.format,
            surface_format.color_space,
            if present_wait { "on" } else { "off" }
        );

        Ok(Self {
            frames_in_flight,
            swapchain,
            images,
            image_views,
            image_view_wrappers,
            format: surface_format.format,
            extent,
            current_image_index: 0,
            image_available_semaphores,
            image_render_finished_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            present_command_buffers,
            current_frame: 0,
            present_wait,
            next_present_id: 1,
            device: vulkan_backend.device().clone(),
            swapchain_loader: vulkan_backend.swapchain_loader().clone(),
            command_pool: vulkan_backend.command_pool(),
            destroyed: false,
        })
    }

    /// Destroy the swapchain and associated resources.
    ///
    /// Note: This is called automatically by Drop, but can be called explicitly
    /// if you need to control when destruction happens.
    pub fn destroy(&mut self) {
        // Check if already destroyed using state flag (more reliable than null checks)
        if self.destroyed {
            return;
        }

        // Mark as destroyed first to prevent re-entry even if something fails below
        self.destroyed = true;

        unsafe {
            let _ = self.device.device_wait_idle();

            // Free command buffers
            if !self.present_command_buffers.is_empty() {
                self.device
                    .free_command_buffers(self.command_pool, &self.present_command_buffers);
                self.present_command_buffers.clear();
            }

            // Destroy synchronization primitives
            for semaphore in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.image_render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }

            // Drop the shared wrappers first (their Drop does not destroy the
            // views), then destroy the views themselves.
            self.image_view_wrappers.clear();
            for view in self.image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }

            // Destroy swapchain (only if it was successfully created)
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
        }
    }

    /// Acquire the next swapchain image.
    ///
    /// Returns the surface texture view along with synchronization info needed for presentation.
    pub fn acquire_next_image(
        &mut self,
        vulkan_backend: &VulkanBackend,
    ) -> Result<VulkanSwapchainAcquireResult, GraphicsError> {
        let current_frame = self.current_frame;

        // Wait for the previous frame using this slot to complete. Bounded:
        // a hung GPU must become an error, not a permanent freeze.
        //
        // The fence is NOT reset here — it is reset in `present_vulkan_frame`
        // immediately before the submit that signals it. A frame that fails
        // anywhere between acquire and present (failed acquire, failed graph
        // submit, dropped surface texture) therefore leaves the fence
        // signaled, and the next frame on this slot proceeds instead of
        // timing out on a fence nobody will ever signal.
        let in_flight_fence = self.in_flight_fences[current_frame];
        unsafe {
            vulkan_backend.device().wait_for_fences(
                &[in_flight_fence],
                true,
                super::FENCE_WAIT_TIMEOUT_NS,
            )
        }
        .map_err(|e| match e {
            vk::Result::TIMEOUT => GraphicsError::Timeout(
                "in-flight fence wait timed out after 10 s; GPU may be hung".into(),
            ),
            vk::Result::ERROR_DEVICE_LOST => {
                vulkan_backend.report_device_lost();
                GraphicsError::DeviceLost
            }
            other => {
                GraphicsError::Internal(format!("Failed to wait for in-flight fence: {other:?}"))
            }
        })?;

        let image_available_semaphore = self.image_available_semaphores[current_frame];
        let image_render_finished_semaphore = self.image_render_finished_semaphores[current_frame];
        // The present-wait semaphore is chosen by acquired image index (below),
        // not frame slot — it must be unique per swapchain image.

        // Acquire next image with semaphore synchronization. OUT_OF_DATE is a
        // recoverable signal: the caller must reconfigure the surface and
        // retry. Nothing has been registered or reset yet, so failing here
        // leaves the slot fully reusable.
        let (image_index, suboptimal) = unsafe {
            vulkan_backend.swapchain_loader().acquire_next_image(
                self.swapchain,
                super::FENCE_WAIT_TIMEOUT_NS,
                image_available_semaphore,
                vk::Fence::null(),
            )
        }
        .map_err(|e| match e {
            vk::Result::ERROR_OUT_OF_DATE_KHR => GraphicsError::SurfaceOutdated,
            vk::Result::ERROR_SURFACE_LOST_KHR => GraphicsError::SurfaceLost,
            vk::Result::TIMEOUT | vk::Result::NOT_READY => GraphicsError::Timeout(
                "swapchain image acquire timed out after 10 s; GPU may be hung".into(),
            ),
            vk::Result::ERROR_DEVICE_LOST => {
                vulkan_backend.report_device_lost();
                GraphicsError::DeviceLost
            }
            other => GraphicsError::ResourceCreationFailed(format!(
                "Failed to acquire swapchain image: {other:?}"
            )),
        })?;

        // Register this frame's acquire/render-done semaphores with the backend
        // so the render submit that writes the swapchain image picks them up
        // (waits on `image_available`, signals `image_render_finished`). Done
        // after the acquire succeeded so a failed acquire leaves no stale
        // semaphores registered.
        vulkan_backend
            .begin_swapchain_frame(image_available_semaphore, image_render_finished_semaphore);

        self.current_image_index = image_index;

        // Validate image index to prevent out-of-bounds access
        let image_idx = image_index as usize;
        if image_idx >= self.images.len() || image_idx >= self.image_views.len() {
            return Err(GraphicsError::Internal(format!(
                "Invalid swapchain image index {} (have {} images)",
                image_index,
                self.images.len()
            )));
        }

        let image = self.images[image_idx];
        let swapchain_handle = self.swapchain;
        let present_cmd = self.present_command_buffers[current_frame];
        // Per-image present-wait semaphore (VUID-03868): indexed by the acquired
        // image, guaranteed idle since the image's prior present has completed.
        let render_finished_semaphore = self.render_finished_semaphores[image_idx];

        // Advance to next frame slot
        self.current_frame = (current_frame + 1) % self.frames_in_flight;

        let vulkan_view = VulkanSurfaceTextureView {
            image,
            view: Arc::clone(&self.image_view_wrappers[image_idx]),
        };

        // Present pacing: allocate this frame's present id (0 = pacing off).
        // A skipped or failed present leaves a gap in the sequence, which is
        // fine — `vkWaitForPresentKHR` waits on a watermark ("id at least
        // N"), not on an exact match.
        let present_id = if self.present_wait {
            let id = self.next_present_id;
            self.next_present_id += 1;
            id
        } else {
            0
        };

        Ok(VulkanSwapchainAcquireResult {
            view: vulkan_view,
            image_index,
            frame_index: current_frame,
            swapchain: swapchain_handle,
            image_available_semaphore,
            image_render_finished_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            present_command_buffer: present_cmd,
            suboptimal,
            present_id,
        })
    }
}

/// Result of acquiring a swapchain image.
pub struct VulkanSwapchainAcquireResult {
    /// The texture view for rendering.
    pub view: VulkanSurfaceTextureView,
    /// The swapchain image index.
    pub image_index: u32,
    /// The frame-in-flight index (for sync primitive lookup).
    pub frame_index: usize,
    /// The swapchain handle.
    pub swapchain: vk::SwapchainKHR,
    /// The image available semaphore for this frame.
    pub image_available_semaphore: vk::Semaphore,
    /// Semaphore signaled by the render submit that writes the swapchain image
    /// (waited on by the present layout-transition submit).
    pub image_render_finished_semaphore: vk::Semaphore,
    /// The render finished semaphore for this frame.
    pub render_finished_semaphore: vk::Semaphore,
    /// The in-flight fence for this frame.
    pub in_flight_fence: vk::Fence,
    /// The command buffer for this frame's presentation.
    pub present_command_buffer: vk::CommandBuffer,
    /// Whether the acquire reported the swapchain as suboptimal for the
    /// surface (e.g. after a resize on X11). The frame should still be
    /// rendered and presented, but the surface needs reconfiguration.
    pub suboptimal: bool,
    /// Present-pacing id for this frame's present, `0` when pacing is off
    /// (see [`VulkanSwapchain::present_wait`]). Restarts at 1 on every new
    /// swapchain so a pacing wait can never reference a retired chain.
    pub present_id: u64,
}

/// Present a Vulkan swapchain image.
#[allow(clippy::too_many_arguments)]
pub fn present_vulkan_frame(
    vulkan_backend: &VulkanBackend,
    view: &VulkanSurfaceTextureView,
    swapchain: vk::SwapchainKHR,
    image_index: u32,
    image_available_semaphore: vk::Semaphore,
    image_render_finished_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
    present_command_buffer: vk::CommandBuffer,
    _frame_index: u64,
    present_id: u64,
) -> Result<(), GraphicsError> {
    let cmd = present_command_buffer;

    // Reset and begin command buffer
    unsafe {
        vulkan_backend
            .device()
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
    }
    .map_err(|e| {
        GraphicsError::Internal(format!(
            "Failed to reset command buffer for present: {:?}",
            e
        ))
    })?;

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        vulkan_backend
            .device()
            .begin_command_buffer(cmd, &begin_info)
    }
    .map_err(|e| {
        GraphicsError::Internal(format!(
            "Failed to begin command buffer for present: {:?}",
            e
        ))
    })?;

    // Transition the image to PRESENT_SRC_KHR. The actual current layout
    // depends on whether anything rendered to the swapchain this frame: the
    // normal path left it in COLOR_ATTACHMENT_OPTIMAL, but a frame that
    // acquired without touching it (loading screen, headless tick) left it in
    // UNDEFINED/PRESENT_SRC — claiming color-attachment there is a layout
    // mismatch (validation error, UB on tilers). `UNDEFINED` is valid from
    // any layout; contents are discardable since nothing was drawn.
    let rendered = vulkan_backend.swapchain_render_consumed();
    let (old_layout, src_access) = if rendered {
        (
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        )
    } else {
        (vk::ImageLayout::UNDEFINED, vk::AccessFlags2::NONE)
    };
    // sync2: the COLOR_ATTACHMENT_OUTPUT → present transition. The
    // destination scope stays empty (`NONE`, formerly BOTTOM_OF_PIPE): the
    // present operation is ordered by the `render_finished` semaphore, not by
    // this barrier's second scope.
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(src_access)
        .dst_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::NONE)
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(view.image())
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let image_barriers = [barrier];
    let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
    unsafe {
        vulkan_backend
            .device()
            .cmd_pipeline_barrier2(cmd, &dependency_info);
    }

    // End command buffer
    unsafe { vulkan_backend.device().end_command_buffer(cmd) }.map_err(|e| {
        GraphicsError::Internal(format!("Failed to end command buffer for present: {:?}", e))
    })?;

    // Submit command buffer with synchronization.
    //
    // If the render submit that wrote the swapchain image consumed
    // `image_available` (the normal path), this barrier submit must instead wait
    // on `image_render_finished` (signaled when rendering completed) so the
    // COLOR_ATTACHMENT_OPTIMAL→PRESENT_SRC transition happens after the writes.
    // If nothing rendered to the swapchain this frame, fall back to waiting on
    // `image_available` directly (the image was acquired but never written).
    let wait_semaphore = if rendered {
        image_render_finished_semaphore
    } else {
        image_available_semaphore
    };
    // sync2: wait/signal are `VkSemaphoreSubmitInfo` lists carrying their own
    // stage masks. The binary present wait keeps COLOR_ATTACHMENT_OUTPUT; the
    // render-finished signal uses ALL_COMMANDS (signal after all work).
    let wait_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
        .semaphore(wait_semaphore)
        .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
    let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
        .semaphore(render_finished_semaphore)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let command_buffer_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];

    let submit_info = vk::SubmitInfo2::default()
        .wait_semaphore_infos(&wait_semaphore_infos)
        .command_buffer_infos(&command_buffer_infos)
        .signal_semaphore_infos(&signal_semaphore_infos);

    // Reset the in-flight fence only now, immediately before the submit that
    // signals it (see `acquire_next_image`): between acquire and this point
    // the fence stays signaled, so an abandoned frame never deadlocks the
    // slot.
    unsafe { vulkan_backend.device().reset_fences(&[in_flight_fence]) }
        .map_err(|e| GraphicsError::Internal(format!("Failed to reset in-flight fence: {e:?}")))?;

    // Submit and signal the fence
    let submit_result = unsafe {
        vulkan_backend.device().queue_submit2(
            vulkan_backend.graphics_queue(),
            &[submit_info],
            in_flight_fence,
        )
    };
    if let Err(e) = submit_result {
        // The fence was reset and now has no pending signal. Best effort:
        // re-signal it with an empty submit so the next acquire on this slot
        // does not stall out the full fence timeout. If that also fails the
        // device is almost certainly lost.
        let resignal = unsafe {
            vulkan_backend.device().queue_submit2(
                vulkan_backend.graphics_queue(),
                &[vk::SubmitInfo2::default()],
                in_flight_fence,
            )
        };
        if let Err(re) = resignal {
            log::error!("Failed to re-signal in-flight fence after failed present submit: {re:?}");
        }
        return Err(match e {
            vk::Result::ERROR_DEVICE_LOST => {
                vulkan_backend.report_device_lost();
                GraphicsError::DeviceLost
            }
            other => {
                GraphicsError::Internal(format!("Failed to submit presentation sync: {other:?}"))
            }
        });
    }

    // Present the swapchain image. `vkQueuePresentKHR` has no sync2 variant;
    // it waits on the plain binary semaphore signaled by the submit above.
    let swapchains = [swapchain];
    let image_indices = [image_index];
    let present_wait_semaphores = [render_finished_semaphore];
    let mut present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&present_wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    // Present pacing: tag this present with its per-swapchain id (allocated
    // at acquire; `0` = pacing off). Ids restart at 1 on every new chain, so
    // a pacing wait can never reference a retired chain's id.
    let pacing = present_id > 0;
    let present_ids = [present_id];
    let mut present_id_info = vk::PresentIdKHR::default().present_ids(&present_ids);
    if pacing {
        present_info = present_info.push_next(&mut present_id_info);
    }

    let result = unsafe {
        vulkan_backend
            .swapchain_loader()
            .queue_present(vulkan_backend.graphics_queue(), &present_info)
    };

    // Bound the present queue: block until the *previous* present has
    // actually reached the display, so at most two presents are ever in
    // flight. Under FIFO this pins the frame cadence to the refresh clock at
    // the cost of ~1 frame of run-ahead; without it the driver queues presents
    // several frames deep and any hitch turns into a burst/stall oscillation.
    //
    // Waited ONLY after a cleanly successful present: SUBOPTIMAL means a
    // resize/monitor transition is in flight, where NVIDIA's
    // `vkWaitForPresentKHR` has been observed to hang far past its timeout
    // (frozen app → TDR → device lost) — and pacing a dying chain buys
    // nothing anyway. `present_id == 1` is the chain's first present: there
    // is no previous id on this chain to wait for. The timeout is a safety
    // valve (compositor stall, occluded window) — pacing then degrades
    // gracefully instead of failing the frame.
    if pacing
        && present_id > 1
        && matches!(result, Ok(false))
        && let Some(wait_loader) = vulkan_backend.present_wait_loader()
    {
        const PRESENT_WAIT_TIMEOUT_NS: u64 = 100_000_000; // 100 ms
        let wait_result = unsafe {
            wait_loader.wait_for_present(swapchain, present_id - 1, PRESENT_WAIT_TIMEOUT_NS)
        };
        match wait_result {
            Ok(_) => {}
            Err(vk::Result::TIMEOUT) => {
                log::debug!("vkWaitForPresentKHR timed out; present pacing skipped this frame");
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                vulkan_backend.report_device_lost();
                return Err(GraphicsError::DeviceLost);
            }
            // OUT_OF_DATE/SURFACE_LOST surface here when the swapchain died
            // between present and wait; the present result below already
            // reports the reconfigure, so the wait just steps aside.
            Err(e) => {
                log::debug!("vkWaitForPresentKHR failed ({e:?}); present pacing skipped");
            }
        }
    }

    // SUBOPTIMAL presented the frame and OUT_OF_DATE did not, but both mean
    // the same thing to the caller: reconfigure the surface before the next
    // acquire. Per spec both results execute the semaphore wait, so the sync
    // objects stay clean either way.
    match result {
        Ok(false) => Ok(()),
        Ok(true) | Err(vk::Result::SUBOPTIMAL_KHR) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            Err(GraphicsError::SurfaceOutdated)
        }
        Err(vk::Result::ERROR_SURFACE_LOST_KHR) => Err(GraphicsError::SurfaceLost),
        Err(vk::Result::ERROR_DEVICE_LOST) => {
            vulkan_backend.report_device_lost();
            Err(GraphicsError::DeviceLost)
        }
        Err(e) => Err(GraphicsError::Internal(format!(
            "Failed to present swapchain image: {:?}",
            e
        ))),
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        self.destroy();
    }
}
