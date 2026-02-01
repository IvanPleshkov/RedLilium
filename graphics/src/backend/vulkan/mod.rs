//! Native Vulkan backend implementation using ash.
//!
//! This backend provides direct Vulkan access for maximum performance and control.
//! It includes support for validation layers in debug builds.

mod allocator;
mod command;
mod conversion;
mod debug;
mod device;
mod instance;
mod sync;

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use parking_lot::Mutex;

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, Pass, RenderGraph};
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

use super::{GpuBuffer, GpuFence, GpuSampler, GpuTexture};

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
pub struct VulkanBackend {
    /// Vulkan entry points (function loader).
    #[allow(dead_code)]
    entry: ash::Entry,
    /// Vulkan instance.
    instance: ash::Instance,
    /// Debug messenger for validation layer output.
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Debug utils extension instance.
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    /// Selected physical device.
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    /// Logical device.
    device: ash::Device,
    /// Graphics queue.
    graphics_queue: vk::Queue,
    /// Graphics queue family index.
    #[allow(dead_code)]
    graphics_queue_family: u32,
    /// Memory allocator.
    allocator: Mutex<Allocator>,
    /// Command pool for graphics operations.
    command_pool: vk::CommandPool,
    /// Whether validation layers are enabled.
    #[allow(dead_code)]
    validation_enabled: bool,
    /// Dynamic rendering extension.
    dynamic_rendering: ash::khr::dynamic_rendering::Device,
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("validation_enabled", &self.validation_enabled)
            .finish()
    }
}

impl VulkanBackend {
    /// Create a new Vulkan backend.
    ///
    /// This initializes the Vulkan instance, selects a physical device,
    /// creates a logical device, and sets up the memory allocator.
    ///
    /// Validation layers are enabled in debug builds by default.
    pub fn new() -> Result<Self, GraphicsError> {
        // Load Vulkan entry points
        let entry = unsafe { ash::Entry::load() }.map_err(|e| {
            GraphicsError::InitializationFailed(format!("Failed to load Vulkan: {}", e))
        })?;

        // Enable validation in debug builds
        let validation_enabled = cfg!(debug_assertions);

        // Create instance with validation layers
        let (instance, debug_messenger, debug_utils) =
            instance::create_instance(&entry, validation_enabled)?;

        // Select physical device
        let physical_device = device::select_physical_device(&instance)?;

        // Find graphics queue family
        let graphics_queue_family = device::find_graphics_queue_family(&instance, physical_device)?;

        // Create logical device and queue
        let device =
            device::create_logical_device(&instance, physical_device, graphics_queue_family)?;

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

        // Create memory allocator
        let allocator = allocator::create_allocator(&instance, physical_device, device.clone())?;

        // Create command pool
        let command_pool = command::create_command_pool(&device, graphics_queue_family)?;

        // Load dynamic rendering extension
        let dynamic_rendering = ash::khr::dynamic_rendering::Device::new(&instance, &device);

        log::info!(
            "Vulkan backend initialized (validation: {})",
            validation_enabled
        );

        Ok(Self {
            entry,
            instance,
            debug_messenger,
            debug_utils,
            physical_device,
            device,
            graphics_queue,
            graphics_queue_family,
            allocator: Mutex::new(allocator),
            command_pool,
            validation_enabled,
            dynamic_rendering,
        })
    }

    /// Get the Vulkan device.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            // Wait for device to be idle before cleanup
            let _ = self.device.device_wait_idle();

            // Destroy command pool
            self.device.destroy_command_pool(self.command_pool, None);

            // Drop allocator before device
            // The allocator is behind a Mutex, so we need to take it
            // This happens automatically when VulkanBackend is dropped

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
        let usage = convert_buffer_usage(descriptor.usage);

        // Determine memory location based on usage flags
        let location = if descriptor
            .usage
            .contains(crate::types::BufferUsage::MAP_READ)
        {
            gpu_allocator::MemoryLocation::GpuToCpu
        } else if descriptor
            .usage
            .contains(crate::types::BufferUsage::MAP_WRITE)
        {
            gpu_allocator::MemoryLocation::CpuToGpu
        } else {
            gpu_allocator::MemoryLocation::GpuOnly
        };

        // Create buffer
        let buffer_info = vk::BufferCreateInfo::default()
            .size(descriptor.size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&buffer_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to create buffer: {:?}", e))
        })?;

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
        })
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        let format = convert_texture_format(descriptor.format);
        let usage = convert_texture_usage(descriptor.usage);

        // Create image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: descriptor.size.width,
                height: descriptor.size.height,
                depth: descriptor.size.depth.max(1),
            })
            .mip_levels(descriptor.mip_level_count)
            .array_layers(1)
            .samples(match descriptor.sample_count {
                1 => vk::SampleCountFlags::TYPE_1,
                2 => vk::SampleCountFlags::TYPE_2,
                4 => vk::SampleCountFlags::TYPE_4,
                8 => vk::SampleCountFlags::TYPE_8,
                _ => vk::SampleCountFlags::TYPE_1,
            })
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { self.device.create_image(&image_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to create image: {:?}", e))
        })?;

        // Get memory requirements
        let mem_requirements = unsafe { self.device.get_image_memory_requirements(image) };

        // Allocate GPU-only memory for textures
        let allocation = {
            let mut allocator = self.allocator.lock();
            allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: descriptor.label.as_deref().unwrap_or("texture"),
                    requirements: mem_requirements,
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
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

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: descriptor.mip_level_count,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { self.device.create_image_view(&view_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Failed to create image view: {:?}", e))
        })?;

        Ok(GpuTexture::Vulkan {
            device: self.device.clone(),
            image,
            view,
            allocation: Mutex::new(Some(allocation)),
            format,
            extent: vk::Extent3D {
                width: descriptor.size.width,
                height: descriptor.size.height,
                depth: descriptor.size.depth.max(1),
            },
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
            .anisotropy_enable(descriptor.anisotropy_clamp > 1)
            .max_anisotropy(descriptor.anisotropy_clamp as f32)
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

        Ok(GpuSampler::Vulkan {
            device: self.device.clone(),
            sampler,
        })
    }

    /// Create a fence for CPU-GPU synchronization.
    pub fn create_fence(&self, signaled: bool) -> GpuFence {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };

        let fence_info = vk::FenceCreateInfo::default().flags(flags);

        let fence =
            unsafe { self.device.create_fence(&fence_info, None) }.expect("Failed to create fence");

        GpuFence::Vulkan {
            device: self.device.clone(),
            fence,
        }
    }

    /// Wait for a fence to be signaled.
    pub fn wait_fence(&self, fence: &GpuFence) {
        if let GpuFence::Vulkan { device, fence } = fence {
            unsafe {
                let _ = device.wait_for_fences(&[*fence], true, u64::MAX);
            }
        }
    }

    /// Check if a fence is signaled (non-blocking).
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        if let GpuFence::Vulkan { device, fence } = fence {
            unsafe { device.get_fence_status(*fence).is_ok() }
        } else {
            false
        }
    }

    /// Signal a fence (for testing/dummy backend).
    pub fn signal_fence(&self, _fence: &GpuFence) {
        // Vulkan fences are signaled by the GPU, not the CPU
        // This is a no-op for the Vulkan backend
    }

    /// Execute a compiled render graph.
    pub fn execute_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError> {
        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| {
                GraphicsError::Internal(format!("Failed to allocate command buffer: {:?}", e))
            })?;

        let cmd = command_buffers[0];

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { self.device.begin_command_buffer(cmd, &begin_info) }.map_err(|e| {
            GraphicsError::Internal(format!("Failed to begin command buffer: {:?}", e))
        })?;

        // Get all passes from the graph
        let passes = graph.passes();

        // Process each pass in compiled order
        for handle in compiled.pass_order() {
            let pass = &passes[handle.index()];
            self.encode_pass(cmd, pass)?;
        }

        // End command buffer
        unsafe { self.device.end_command_buffer(cmd) }.map_err(|e| {
            GraphicsError::Internal(format!("Failed to end command buffer: {:?}", e))
        })?;

        // Get fence to signal
        let fence = signal_fence.and_then(|f| {
            if let GpuFence::Vulkan { fence, .. } = f {
                // Reset fence before use
                unsafe {
                    let _ = self.device.reset_fences(&[*fence]);
                }
                Some(*fence)
            } else {
                None
            }
        });

        // Submit command buffer
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        unsafe {
            self.device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                fence.unwrap_or(vk::Fence::null()),
            )
        }
        .map_err(|e| {
            GraphicsError::Internal(format!("Failed to submit command buffer: {:?}", e))
        })?;

        // If no fence provided, wait for queue to idle
        if signal_fence.is_none() {
            unsafe { self.device.queue_wait_idle(self.graphics_queue) }.map_err(|e| {
                GraphicsError::Internal(format!("Failed to wait for queue idle: {:?}", e))
            })?;
        }

        // Free command buffer
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &command_buffers);
        }

        Ok(())
    }

    /// Write data to a buffer.
    pub fn write_buffer(&self, buffer: &GpuBuffer, offset: u64, data: &[u8]) {
        if let GpuBuffer::Vulkan { allocation, .. } = buffer
            && let Some(allocation) = allocation.lock().as_ref()
            && let Some(mapped_ptr) = allocation.mapped_ptr()
        {
            unsafe {
                let dst = mapped_ptr.as_ptr().add(offset as usize);
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst as *mut u8, data.len());
            }
        }
    }

    /// Read data from a buffer.
    pub fn read_buffer(&self, buffer: &GpuBuffer, offset: u64, size: u64) -> Vec<u8> {
        if let GpuBuffer::Vulkan { allocation, .. } = buffer
            && let Some(allocation) = allocation.lock().as_ref()
            && let Some(mapped_ptr) = allocation.mapped_ptr()
        {
            let mut result = vec![0u8; size as usize];
            unsafe {
                let src = mapped_ptr.as_ptr().add(offset as usize);
                std::ptr::copy_nonoverlapping(src as *const u8, result.as_mut_ptr(), size as usize);
            }
            return result;
        }
        vec![0u8; size as usize]
    }

    fn encode_pass(&self, cmd: vk::CommandBuffer, pass: &Pass) -> Result<(), GraphicsError> {
        match pass {
            Pass::Graphics(graphics_pass) => self.encode_graphics_pass(cmd, graphics_pass),
            Pass::Transfer(transfer_pass) => self.encode_transfer_pass(cmd, transfer_pass),
            Pass::Compute(compute_pass) => self.encode_compute_pass(cmd, compute_pass),
        }
    }

    fn encode_graphics_pass(
        &self,
        cmd: vk::CommandBuffer,
        pass: &crate::graph::GraphicsPass,
    ) -> Result<(), GraphicsError> {
        let Some(render_targets) = pass.render_targets() else {
            log::trace!(
                "Skipping graphics pass '{}': no render targets",
                pass.name()
            );
            return Ok(());
        };

        // Build color attachments for dynamic rendering
        let color_attachments: Vec<vk::RenderingAttachmentInfo> = render_targets
            .color_attachments
            .iter()
            .filter_map(|attachment| {
                let GpuTexture::Vulkan { view, .. } = attachment.texture().gpu_handle() else {
                    return None;
                };

                let (load_op, clear_value) =
                    conversion::convert_load_op_color(&attachment.load_op());
                let store_op = conversion::convert_store_op(&attachment.store_op());

                Some(
                    vk::RenderingAttachmentInfo::default()
                        .image_view(*view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(load_op)
                        .store_op(store_op)
                        .clear_value(clear_value),
                )
            })
            .collect();

        // Build depth attachment if present
        let depth_attachment =
            render_targets
                .depth_stencil_attachment
                .as_ref()
                .and_then(|attachment| {
                    let GpuTexture::Vulkan { view, .. } = attachment.texture().gpu_handle() else {
                        return None;
                    };

                    let (load_op, clear_value) =
                        conversion::convert_load_op_depth(&attachment.depth_load_op());
                    let store_op = conversion::convert_store_op(&attachment.depth_store_op());

                    Some(
                        vk::RenderingAttachmentInfo::default()
                            .image_view(*view)
                            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                            .load_op(load_op)
                            .store_op(store_op)
                            .clear_value(clear_value),
                    )
                });

        // Determine render area from first attachment
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

        // Transition images to the appropriate layouts
        for attachment in &render_targets.color_attachments {
            if let GpuTexture::Vulkan { image, .. } = attachment.texture().gpu_handle() {
                self.transition_image_layout(
                    cmd,
                    *image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageAspectFlags::COLOR,
                );
            }
        }

        if let Some(attachment) = &render_targets.depth_stencil_attachment
            && let GpuTexture::Vulkan { image, .. } = attachment.texture().gpu_handle()
        {
            self.transition_image_layout(
                cmd,
                *image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
            );
        }

        // Create rendering info
        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments);

        if let Some(ref depth) = depth_attachment {
            rendering_info = rendering_info.depth_attachment(depth);
        }

        // Begin dynamic rendering
        unsafe {
            self.dynamic_rendering
                .cmd_begin_rendering(cmd, &rendering_info);
        }

        // Set viewport with [0, 1] depth range (D3D/wgpu convention)
        // This is the key coordinate system configuration that matches wgpu behavior.
        // Vulkan natively uses [0, 1] depth range, so we just need to set it explicitly.
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: render_area.extent.width as f32,
            height: render_area.extent.height as f32,
            min_depth: 0.0, // Near plane maps to depth 0
            max_depth: 1.0, // Far plane maps to depth 1
        };
        unsafe {
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
        }

        // Set scissor to match render area
        let scissor = vk::Rect2D {
            offset: render_area.offset,
            extent: render_area.extent,
        };
        unsafe {
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);
        }

        // TODO: Encode draw commands

        // End dynamic rendering
        unsafe {
            self.dynamic_rendering.cmd_end_rendering(cmd);
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
        use crate::graph::TransferOperation;

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
            TransferOperation::TextureToBuffer { src, dst, regions } => {
                let GpuTexture::Vulkan {
                    image: src_image,
                    format,
                    ..
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

                // Transition image to transfer src
                self.transition_image_layout(
                    cmd,
                    *src_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    if format.has_depth() {
                        vk::ImageAspectFlags::DEPTH
                    } else {
                        vk::ImageAspectFlags::COLOR
                    },
                );

                let block_size = src.format().block_size();

                let copy_regions: Vec<vk::BufferImageCopy> = regions
                    .iter()
                    .map(|r| {
                        // Compute bytes_per_row with 256-byte alignment for consistency with wgpu
                        // If bytes_per_row is not specified and we have multiple rows, align to 256 bytes
                        let bytes_per_row = r.buffer_layout.bytes_per_row.unwrap_or_else(|| {
                            if r.extent.height > 1 {
                                let unpadded = r.extent.width * block_size;
                                // Align to 256 bytes for wgpu compatibility
                                (unpadded + 255) & !255
                            } else {
                                0 // Single row - tight packing
                            }
                        });

                        // Vulkan's buffer_row_length is in texels (pixels), not bytes
                        // Convert from bytes to texels by dividing by block_size
                        let row_length_texels = if bytes_per_row > 0 {
                            bytes_per_row / block_size
                        } else {
                            0 // 0 means tightly packed
                        };

                        vk::BufferImageCopy::default()
                            .buffer_offset(r.buffer_layout.offset)
                            .buffer_row_length(row_length_texels)
                            .buffer_image_height(r.buffer_layout.rows_per_image.unwrap_or(0))
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: r.texture_location.mip_level,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .image_offset(vk::Offset3D {
                                x: r.texture_location.origin.x as i32,
                                y: r.texture_location.origin.y as i32,
                                z: r.texture_location.origin.z as i32,
                            })
                            .image_extent(vk::Extent3D {
                                width: r.extent.width,
                                height: r.extent.height,
                                depth: r.extent.depth.max(1),
                            })
                    })
                    .collect();

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

                // Transition image to transfer dst
                self.transition_image_layout(
                    cmd,
                    *dst_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageAspectFlags::COLOR,
                );

                let block_size = dst.format().block_size();

                let copy_regions: Vec<vk::BufferImageCopy> = regions
                    .iter()
                    .map(|r| {
                        // Compute bytes_per_row with 256-byte alignment for consistency with wgpu
                        let bytes_per_row = r.buffer_layout.bytes_per_row.unwrap_or_else(|| {
                            if r.extent.height > 1 {
                                let unpadded = r.extent.width * block_size;
                                (unpadded + 255) & !255
                            } else {
                                0
                            }
                        });

                        // Convert from bytes to texels for Vulkan
                        let row_length_texels = if bytes_per_row > 0 {
                            bytes_per_row / block_size
                        } else {
                            0
                        };

                        vk::BufferImageCopy::default()
                            .buffer_offset(r.buffer_layout.offset)
                            .buffer_row_length(row_length_texels)
                            .buffer_image_height(r.buffer_layout.rows_per_image.unwrap_or(0))
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: r.texture_location.mip_level,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .image_offset(vk::Offset3D {
                                x: r.texture_location.origin.x as i32,
                                y: r.texture_location.origin.y as i32,
                                z: r.texture_location.origin.z as i32,
                            })
                            .image_extent(vk::Extent3D {
                                width: r.extent.width,
                                height: r.extent.height,
                                depth: r.extent.depth.max(1),
                            })
                    })
                    .collect();

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

                // Transition images
                self.transition_image_layout(
                    cmd,
                    *src_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageAspectFlags::COLOR,
                );
                self.transition_image_layout(
                    cmd,
                    *dst_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageAspectFlags::COLOR,
                );

                let copy_regions: Vec<vk::ImageCopy> = regions
                    .iter()
                    .map(|r| {
                        vk::ImageCopy::default()
                            .src_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: r.src.mip_level,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .src_offset(vk::Offset3D {
                                x: r.src.origin.x as i32,
                                y: r.src.origin.y as i32,
                                z: r.src.origin.z as i32,
                            })
                            .dst_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: r.dst.mip_level,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .dst_offset(vk::Offset3D {
                                x: r.dst.origin.x as i32,
                                y: r.dst.origin.y as i32,
                                z: r.dst.origin.z as i32,
                            })
                            .extent(vk::Extent3D {
                                width: r.extent.width,
                                height: r.extent.height,
                                depth: r.extent.depth.max(1),
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
        }
        Ok(())
    }

    fn encode_compute_pass(
        &self,
        _cmd: vk::CommandBuffer,
        _pass: &crate::graph::ComputePass,
    ) -> Result<(), GraphicsError> {
        // TODO: Implement compute pass encoding
        Ok(())
    }

    fn transition_image_layout(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        let (src_access_mask, src_stage) = match old_layout {
            vk::ImageLayout::UNDEFINED => (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::TRANSFER,
            ),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            ),
        };

        let (dst_access_mask, dst_stage) = match new_layout {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::TRANSFER,
            ),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            ),
        };

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            })
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
}

// Helper trait for vk::Format to check depth
trait FormatExt {
    fn has_depth(&self) -> bool;
}

impl FormatExt for vk::Format {
    fn has_depth(&self) -> bool {
        matches!(
            *self,
            vk::Format::D16_UNORM
                | vk::Format::D32_SFLOAT
                | vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT
        )
    }
}
