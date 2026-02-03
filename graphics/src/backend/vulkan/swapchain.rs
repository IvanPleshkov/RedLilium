//! Vulkan swapchain implementation.
//!
//! This module contains the Vulkan-specific swapchain and surface texture handling.

use std::sync::Arc;

use ash::vk;

use super::conversion::{convert_present_mode, convert_texture_format};
use super::{VulkanBackend, VulkanImageView, VulkanSurfaceTextureView};
use crate::error::GraphicsError;
use crate::swapchain::SurfaceConfiguration;

/// Maximum number of frames that can be in flight simultaneously.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Vulkan swapchain resources.
pub struct VulkanSwapchain {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) image_views: Vec<vk::ImageView>,
    #[allow(dead_code)] // Reserved for future use
    pub(crate) format: vk::Format,
    #[allow(dead_code)] // Reserved for future use
    pub(crate) extent: vk::Extent2D,
    pub(crate) current_image_index: u32,
    /// Semaphores signaled when swapchain image is available (one per frame in flight).
    pub(crate) image_available_semaphores: Vec<vk::Semaphore>,
    /// Semaphores signaled when rendering is complete (one per frame in flight).
    pub(crate) render_finished_semaphores: Vec<vk::Semaphore>,
    /// Fences for CPU-GPU synchronization (one per frame in flight).
    pub(crate) in_flight_fences: Vec<vk::Fence>,
    /// Command buffers for presentation (one per frame in flight).
    pub(crate) present_command_buffers: Vec<vk::CommandBuffer>,
    /// Current frame index (cycles through frames in flight).
    pub(crate) current_frame: usize,
    /// Device handle for cleanup.
    device: ash::Device,
    /// Swapchain loader for cleanup.
    swapchain_loader: ash::khr::swapchain::Device,
    /// Command pool for freeing command buffers.
    command_pool: vk::CommandPool,
}

impl VulkanSwapchain {
    /// Create a new Vulkan swapchain.
    pub fn new(
        vulkan_backend: &VulkanBackend,
        surface: vk::SurfaceKHR,
        config: &SurfaceConfiguration,
    ) -> Result<Self, GraphicsError> {
        // Get surface capabilities
        let capabilities = vulkan_backend.get_surface_capabilities(surface)?;

        // Choose format
        let formats = vulkan_backend.get_surface_formats(surface)?;
        let surface_format = formats
            .iter()
            .find(|f| f.format == convert_texture_format(config.format))
            .cloned()
            .unwrap_or(formats[0]);

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

        // Choose image count (prefer triple buffering)
        let image_count =
            (capabilities.min_image_count + 1).min(if capabilities.max_image_count > 0 {
                capabilities.max_image_count
            } else {
                u32::MAX
            });

        // Create swapchain
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

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

        // Create image views
        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| vulkan_backend.create_swapchain_image_view(image, surface_format.format))
            .collect::<Result<Vec<_>, _>>()?;

        // Create synchronization primitives for frames in flight
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
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
            image_available_semaphores.push(image_available);

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
            render_finished_semaphores.push(render_finished);

            let fence = unsafe { vulkan_backend.device().create_fence(&fence_info, None) }
                .map_err(|e| {
                    GraphicsError::ResourceCreationFailed(format!(
                        "Failed to create in-flight fence: {:?}",
                        e
                    ))
                })?;
            in_flight_fences.push(fence);
        }

        // Allocate command buffers for presentation (one per frame in flight)
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(vulkan_backend.command_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

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

        log::info!(
            "Created Vulkan swapchain: {}x{} with {} images, {} frames in flight",
            extent.width,
            extent.height,
            images.len(),
            MAX_FRAMES_IN_FLIGHT
        );

        Ok(Self {
            swapchain,
            images,
            image_views,
            format: surface_format.format,
            extent,
            current_image_index: 0,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            present_command_buffers,
            current_frame: 0,
            device: vulkan_backend.device().clone(),
            swapchain_loader: vulkan_backend.swapchain_loader().clone(),
            command_pool: vulkan_backend.command_pool(),
        })
    }

    /// Destroy the swapchain and associated resources.
    ///
    /// Note: This is called automatically by Drop, but can be called explicitly
    /// if you need to control when destruction happens.
    pub fn destroy(&mut self) {
        // Check if already destroyed (swapchain handle is null)
        if self.swapchain == vk::SwapchainKHR::null() {
            return;
        }

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
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }

            // Destroy image views
            for view in self.image_views.drain(..) {
                self.device.destroy_image_view(view, None);
            }

            // Destroy swapchain
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.swapchain = vk::SwapchainKHR::null();
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

        // Wait for the previous frame using this slot to complete
        let in_flight_fence = self.in_flight_fences[current_frame];
        unsafe {
            vulkan_backend
                .device()
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)
        }
        .map_err(|e| {
            GraphicsError::Internal(format!("Failed to wait for in-flight fence: {:?}", e))
        })?;

        // Reset the fence for this frame
        unsafe { vulkan_backend.device().reset_fences(&[in_flight_fence]) }.map_err(|e| {
            GraphicsError::Internal(format!("Failed to reset in-flight fence: {:?}", e))
        })?;

        // Acquire next image with semaphore synchronization
        let image_available_semaphore = self.image_available_semaphores[current_frame];
        let render_finished_semaphore = self.render_finished_semaphores[current_frame];
        let (image_index, _suboptimal) = unsafe {
            vulkan_backend.swapchain_loader().acquire_next_image(
                self.swapchain,
                u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        }
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to acquire swapchain image: {:?}",
                e
            ))
        })?;

        self.current_image_index = image_index;
        let image = self.images[image_index as usize];
        let view = self.image_views[image_index as usize];
        let swapchain_handle = self.swapchain;
        let present_cmd = self.present_command_buffers[current_frame];

        // Advance to next frame slot
        self.current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        let vulkan_view = VulkanSurfaceTextureView {
            image,
            view: Arc::new(VulkanImageView::new(vulkan_backend.device().clone(), view)),
        };

        Ok(VulkanSwapchainAcquireResult {
            view: vulkan_view,
            image_index,
            frame_index: current_frame,
            swapchain: swapchain_handle,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            present_command_buffer: present_cmd,
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
    /// The render finished semaphore for this frame.
    pub render_finished_semaphore: vk::Semaphore,
    /// The in-flight fence for this frame.
    pub in_flight_fence: vk::Fence,
    /// The command buffer for this frame's presentation.
    pub present_command_buffer: vk::CommandBuffer,
}

/// Present a Vulkan swapchain image.
#[allow(clippy::too_many_arguments)]
pub fn present_vulkan_frame(
    vulkan_backend: &VulkanBackend,
    view: &VulkanSurfaceTextureView,
    swapchain: vk::SwapchainKHR,
    image_index: u32,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
    present_command_buffer: vk::CommandBuffer,
    frame_index: u64,
) -> Result<(), GraphicsError> {
    let cmd = present_command_buffer;
    let command_buffers = [cmd];

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

    // Transition image from COLOR_ATTACHMENT_OPTIMAL to PRESENT_SRC_KHR
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
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
        })
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::empty());

    unsafe {
        vulkan_backend.device().cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    // End command buffer
    unsafe { vulkan_backend.device().end_command_buffer(cmd) }.map_err(|e| {
        GraphicsError::Internal(format!("Failed to end command buffer for present: {:?}", e))
    })?;

    // Submit command buffer with synchronization
    let wait_semaphores = [image_available_semaphore];
    let signal_semaphores = [render_finished_semaphore];
    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores);

    // Submit and signal the fence
    unsafe {
        vulkan_backend.device().queue_submit(
            vulkan_backend.graphics_queue(),
            &[submit_info],
            in_flight_fence,
        )
    }
    .map_err(|e| GraphicsError::Internal(format!("Failed to submit presentation sync: {:?}", e)))?;

    // Present the swapchain image
    let swapchains = [swapchain];
    let image_indices = [image_index];
    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(&signal_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    let result = unsafe {
        vulkan_backend
            .swapchain_loader()
            .queue_present(vulkan_backend.graphics_queue(), &present_info)
    };

    match result {
        Ok(_) => {
            log::trace!(
                "Presented Vulkan frame {}, image index: {}",
                frame_index,
                image_index
            );
            Ok(())
        }
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
            log::warn!("Swapchain out of date, needs recreation");
            Ok(())
        }
        Err(vk::Result::SUBOPTIMAL_KHR) => {
            log::trace!("Swapchain suboptimal");
            Ok(())
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
