//! Native Vulkan backend implementation using ash.
//!
//! This backend provides direct Vulkan access for maximum performance and control.
//! It includes support for validation layers in debug builds.

mod allocator;
pub mod barriers;
mod command;
pub(crate) mod conversion;
mod debug;
pub mod deferred;
mod device;
mod instance;
pub mod layout;
mod pipeline;
pub mod swapchain;

use std::sync::Arc;

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use parking_lot::Mutex;

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, Pass, RenderGraph, RenderTarget};
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

use super::{GpuBuffer, GpuFence, GpuSampler, GpuTexture};

pub use deferred::{DeferredDestructor, DeferredResource, MAX_FRAMES_IN_FLIGHT};
pub use layout::{TextureLayout, TextureLayoutTracker, TextureUsageGraph};

use self::barriers::{BarrierBatch, BufferId};
use self::layout::TextureId;

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
    /// Selected physical device.
    physical_device: vk::PhysicalDevice,
    /// Logical device.
    device: ash::Device,
    /// Graphics queue.
    graphics_queue: vk::Queue,
    /// Graphics queue family index.
    graphics_queue_family: u32,
    /// Memory allocator (wrapped in Arc for sharing with deferred destructor).
    allocator: Arc<Mutex<Allocator>>,
    /// Command pool for graphics operations.
    command_pool: vk::CommandPool,
    /// Whether validation layers are enabled.
    #[allow(dead_code)]
    validation_enabled: bool,
    /// Dynamic rendering extension.
    dynamic_rendering: ash::khr::dynamic_rendering::Device,
    /// Surface extension.
    surface_loader: ash::khr::surface::Instance,
    /// Swapchain extension.
    swapchain_loader: ash::khr::swapchain::Device,
    /// Deferred destructor for safe resource cleanup.
    deferred_destructor: Arc<DeferredDestructor>,
    /// Layout tracker for automatic barrier placement.
    /// Uses interior mutability since execute_graph takes &self.
    layout_tracker: Mutex<TextureLayoutTracker>,
    /// Pipeline manager for shader compilation and pipeline creation.
    pipeline_manager: pipeline::PipelineManager,
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

        // Create memory allocator (wrapped in Arc for sharing with deferred destructor)
        let allocator = Arc::new(Mutex::new(allocator::create_allocator(
            &instance,
            physical_device,
            device.clone(),
        )?));

        // Create deferred destructor
        let deferred_destructor = Arc::new(DeferredDestructor::new());
        deferred_destructor.set_allocator(Arc::downgrade(&allocator));

        // Create command pool
        let command_pool = command::create_command_pool(&device, graphics_queue_family)?;

        // Load dynamic rendering extension
        let dynamic_rendering = ash::khr::dynamic_rendering::Device::new(&instance, &device);

        // Load surface extension
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // Load swapchain extension
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        // Create layout tracker for automatic barrier placement
        let layout_tracker = Mutex::new(TextureLayoutTracker::new(MAX_FRAMES_IN_FLIGHT));

        // Create pipeline manager for shader compilation and graphics pipelines
        let pipeline_manager = pipeline::PipelineManager::new(device.clone())?;

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
            allocator,
            command_pool,
            validation_enabled,
            dynamic_rendering,
            surface_loader,
            swapchain_loader,
            deferred_destructor,
            layout_tracker,
            pipeline_manager,
        })
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

    /// Get the deferred destructor for safe resource cleanup.
    pub fn deferred_destructor(&self) -> &Arc<DeferredDestructor> {
        &self.deferred_destructor
    }

    /// Advance to the next frame.
    ///
    /// This advances both the deferred destructor (to clean up old resources)
    /// and the layout tracker (to reset layout state for the new frame).
    ///
    /// This should be called after waiting on a frame fence to ensure
    /// the GPU has finished with resources from older frames.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the GPU has finished executing all commands
    /// from `MAX_FRAMES_IN_FLIGHT` frames ago.
    pub unsafe fn advance_frame(&self) {
        // SAFETY: Caller guarantees GPU has finished with old resources
        unsafe { self.deferred_destructor.advance_frame() };

        // Advance layout tracker to new frame (resets layout state)
        self.layout_tracker.lock().advance_frame();
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

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            // Wait for device to be idle before cleanup
            let _ = self.device.device_wait_idle();

            // Flush all pending deferred resources now that the device is idle
            self.deferred_destructor.flush_all();

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

        // Determine memory location based on usage flags.
        // Buffers that need CPU access (MAP_READ, MAP_WRITE, or COPY_DST for CPU writes)
        // should use host-visible memory. UNIFORM and VERTEX buffers with COPY_DST are
        // commonly updated from CPU, so use CpuToGpu for those too.
        let location = if descriptor
            .usage
            .contains(crate::types::BufferUsage::MAP_READ)
        {
            gpu_allocator::MemoryLocation::GpuToCpu
        } else if descriptor
            .usage
            .contains(crate::types::BufferUsage::MAP_WRITE)
            || descriptor
                .usage
                .contains(crate::types::BufferUsage::COPY_DST)
        {
            // COPY_DST buffers are typically updated from CPU via write_buffer,
            // so they need host-visible memory for direct mapping
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
            deferred: Arc::clone(&self.deferred_destructor),
        })
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        use crate::types::TextureDimension;

        let format = convert_texture_format(descriptor.format);
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
            TextureDimension::D2 => (
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

        // Create image
        let image_info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(image_type)
            .format(format)
            .extent(extent)
            .mip_levels(descriptor.mip_level_count)
            .array_layers(array_layers)
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

        // Determine view type based on dimension
        let (view_type, layer_count) = match descriptor.dimension {
            TextureDimension::D1 => {
                if descriptor.size.depth > 1 {
                    (vk::ImageViewType::TYPE_1D_ARRAY, array_layers)
                } else {
                    (vk::ImageViewType::TYPE_1D, 1)
                }
            }
            TextureDimension::D2 => {
                if array_layers > 1 {
                    (vk::ImageViewType::TYPE_2D_ARRAY, array_layers)
                } else {
                    (vk::ImageViewType::TYPE_2D, 1)
                }
            }
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

        Ok(GpuTexture::Vulkan {
            device: self.device.clone(),
            image,
            view,
            allocation: Mutex::new(Some(allocation)),
            format,
            extent,
            deferred: Arc::clone(&self.deferred_destructor),
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
            deferred: Arc::clone(&self.deferred_destructor),
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
            deferred: Arc::clone(&self.deferred_destructor),
        }
    }

    /// Wait for a fence to be signaled.
    pub fn wait_fence(&self, fence: &GpuFence) {
        if let GpuFence::Vulkan { device, fence, .. } = fence {
            unsafe {
                let _ = device.wait_for_fences(&[*fence], true, u64::MAX);
            }
        }
    }

    /// Check if a fence is signaled (non-blocking).
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        if let GpuFence::Vulkan { device, fence, .. } = fence {
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
        // Reset descriptor pool at the start of each graph execution.
        // This is safe because execute_graph waits for the queue to idle at the end,
        // so any descriptor sets from previous executions are no longer in use.
        self.pipeline_manager.reset_descriptor_pool()?;

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

            // Infer resource usage and generate barriers
            let usage = pass.infer_resource_usage();
            let barriers = self.generate_barriers_for_pass(&usage);
            barriers.submit(&self.device, cmd);

            // Encode the pass
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

    /// Generate barriers for a pass's resource usage.
    ///
    /// This examines the texture and buffer usages declared by the pass, determines
    /// required layout transitions and memory barriers, and updates tracker state.
    fn generate_barriers_for_pass(
        &self,
        usage: &crate::graph::resource_usage::PassResourceUsage,
    ) -> BarrierBatch {
        use crate::graph::resource_usage::TextureAccessMode;

        let mut tracker = self.layout_tracker.lock();
        let mut batch = BarrierBatch::new();

        log::debug!(
            "Generating barriers: {} texture usages, {} buffer usages",
            usage.texture_usages.len(),
            usage.buffer_usages.len()
        );

        // Generate texture (image) barriers
        for decl in &usage.texture_usages {
            // Get Vulkan image info from the texture
            let GpuTexture::Vulkan { image, .. } = decl.texture.gpu_handle() else {
                continue;
            };

            let texture_id = TextureId::from(*image);
            let current_layout = tracker.get_layout(texture_id);
            let required_layout = decl.access.to_layout();

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

            // Add barrier if layout change is needed
            log::debug!(
                "Texture barrier: {:?} -> {:?} (label: {:?})",
                current_layout,
                required_layout,
                decl.texture.label()
            );
            batch.add_image_barrier(
                texture_id,
                *image,
                current_layout,
                required_layout,
                aspect_mask,
            );

            // Update tracked state
            tracker.set_layout(texture_id, required_layout);
        }

        // Generate buffer barriers
        // Note: Buffer barriers are needed for synchronization between passes that
        // write and read the same buffer. We track the previous access mode per buffer.
        for decl in &usage.buffer_usages {
            // Get Vulkan buffer info
            let GpuBuffer::Vulkan { buffer, .. } = decl.buffer.gpu_handle() else {
                continue;
            };

            let buffer_id = BufferId::from(*buffer);

            // For buffer barriers, we need to track previous access mode.
            // Since we don't have per-buffer tracking yet, we generate barriers
            // for all write operations and write-to-read transitions.
            // This is conservative but safe - a future optimization could add
            // buffer state tracking similar to texture layout tracking.
            if decl.access.is_write() {
                // Always barrier before writes to ensure previous reads complete
                batch.add_buffer_barrier(
                    buffer_id,
                    *buffer,
                    decl.access, // Use same access as src (will be optimized away if same read)
                    decl.access,
                    decl.offset,
                    decl.size,
                );
            } else {
                // For reads after potential writes, we need barriers.
                // Since we don't track previous state, use TransferWrite as conservative src.
                // This ensures any previous transfer/storage writes are visible.
                batch.add_buffer_barrier(
                    buffer_id,
                    *buffer,
                    crate::graph::resource_usage::BufferAccessMode::TransferWrite,
                    decl.access,
                    decl.offset,
                    decl.size,
                );
            }
        }

        batch
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
                    let store_op = conversion::convert_store_op(&attachment.depth_store_op());

                    match &attachment.target {
                        RenderTarget::Texture { texture, .. } => {
                            let GpuTexture::Vulkan { view, .. } = texture.gpu_handle() else {
                                return None;
                            };

                            Some(
                                vk::RenderingAttachmentInfo::default()
                                    .image_view(*view)
                                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
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
                                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                                    .load_op(load_op)
                                    .store_op(store_op)
                                    .clear_value(clear_value)
                            })
                        }
                    }
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

        log::debug!(
            "Graphics pass '{}': render_area={}x{}, color_attachments={}, depth={}",
            pass.name(),
            render_area.extent.width,
            render_area.extent.height,
            color_attachments.len(),
            depth_attachment.is_some()
        );

        // NOTE: Layout transitions are now handled automatically by the barrier
        // generation system in execute_graph() before each pass is encoded.
        // Surface images (swapchain) are handled specially below.

        // Transition surface images from UNDEFINED/PRESENT_SRC to COLOR_ATTACHMENT_OPTIMAL.
        // Using UNDEFINED as old_layout is valid from any actual layout (contents are discarded
        // but that's OK since we're clearing the render target).
        for attachment in &render_targets.color_attachments {
            if let RenderTarget::Surface {
                vulkan_view: Some(surface_view),
                ..
            } = &attachment.target
            {
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
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
                    })
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

                unsafe {
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
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

        // Begin dynamic rendering
        unsafe {
            self.dynamic_rendering
                .cmd_begin_rendering(cmd, &rendering_info);
        }

        // Set viewport with Y-flip and [0, 1] depth range to match wgpu/D3D conventions.
        // Vulkan's Y-axis points down (0=top, height=bottom), but wgpu/OpenGL use Y-up.
        // Using a negative height viewport flips the Y-axis, making the coordinate system
        // consistent with wgpu behavior. This requires VK_KHR_maintenance1 (Vulkan 1.1+).
        let viewport = vk::Viewport {
            x: 0.0,
            y: render_area.extent.height as f32, // Start at bottom
            width: render_area.extent.width as f32,
            height: -(render_area.extent.height as f32), // Negative height flips Y
            min_depth: 0.0,                              // Near plane maps to depth 0
            max_depth: 1.0,                              // Far plane maps to depth 1
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

        // Encode draw commands
        for draw_cmd in pass.draw_commands() {
            self.encode_draw_command(cmd, draw_cmd, render_targets)?;
        }

        // End dynamic rendering
        unsafe {
            self.dynamic_rendering.cmd_end_rendering(cmd);
        }

        // Note: Surface images are transitioned to PRESENT_SRC_KHR in present_vulkan_frame,
        // so we leave them in COLOR_ATTACHMENT_OPTIMAL here.

        Ok(())
    }

    fn encode_draw_command(
        &self,
        cmd: vk::CommandBuffer,
        draw_cmd: &crate::graph::DrawCommand,
        render_targets: &crate::graph::RenderTargetConfig,
    ) -> Result<(), GraphicsError> {
        use crate::materials::BoundResource;

        let material = draw_cmd.material.material();
        let mesh = &draw_cmd.mesh;

        // Get color target formats
        let color_formats: Vec<crate::types::TextureFormat> = render_targets
            .color_attachments
            .iter()
            .map(|a| a.target.format())
            .collect();

        // Get depth format if present
        let depth_format = render_targets
            .depth_stencil_attachment
            .as_ref()
            .map(|a| a.target.format());

        // Compile shaders
        let shaders = material.shaders();
        let mut vertex_module = None;
        let mut fragment_module = None;
        let mut vertex_entry = "vs_main".to_string();
        let mut fragment_entry = "fs_main".to_string();

        for shader in shaders {
            let module = self.pipeline_manager.compile_shader(
                &shader.source,
                shader.stage,
                &shader.entry_point,
            )?;

            match shader.stage {
                crate::materials::ShaderStage::Vertex => {
                    vertex_module = Some(module);
                    vertex_entry = shader.entry_point.clone();
                }
                crate::materials::ShaderStage::Fragment => {
                    fragment_module = Some(module);
                    fragment_entry = shader.entry_point.clone();
                }
                crate::materials::ShaderStage::Compute => {}
            }
        }

        let vertex_module = vertex_module.ok_or_else(|| {
            GraphicsError::ShaderCompilationFailed("No vertex shader provided".into())
        })?;

        // Create descriptor set layouts from material binding layouts
        let binding_layouts = material.binding_layouts();
        let mut descriptor_set_layouts = Vec::new();

        for layout in binding_layouts {
            let ds_layout = self.pipeline_manager.create_descriptor_set_layout(layout)?;
            descriptor_set_layouts.push(ds_layout);
        }

        // Create pipeline layout
        let pipeline_layout = self
            .pipeline_manager
            .create_pipeline_layout(&descriptor_set_layouts)?;

        // Create graphics pipeline
        let pipeline = self.pipeline_manager.create_graphics_pipeline(
            vertex_module,
            fragment_module,
            &vertex_entry,
            &fragment_entry,
            mesh,
            pipeline_layout,
            &color_formats,
            depth_format,
            &self.dynamic_rendering,
        )?;

        // Create and bind descriptor sets
        let material_instance = &draw_cmd.material;
        let binding_groups = material_instance.binding_groups();

        let mut descriptor_sets = Vec::new();
        for (group_idx, (group, ds_layout)) in binding_groups
            .iter()
            .zip(descriptor_set_layouts.iter())
            .enumerate()
        {
            let descriptor_set = self.pipeline_manager.allocate_descriptor_set(*ds_layout)?;

            // Get the corresponding binding layout to look up binding types
            let binding_layout = binding_layouts.get(group_idx);

            // Write descriptor set entries
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
            let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
            let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();

            for entry in &group.entries {
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
                    BoundResource::Texture(texture) => {
                        if let GpuTexture::Vulkan { view, .. } = texture.gpu_handle() {
                            image_infos.push(vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view: *view,
                                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
                                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            });
                        }
                    }
                }
            }

            // Build write descriptors
            let mut buffer_idx = 0;
            let mut image_idx = 0;
            for entry in &group.entries {
                // Look up the binding type from the layout
                let binding_type = binding_layout.and_then(|layout| {
                    layout
                        .entries
                        .iter()
                        .find(|e| e.binding == entry.binding)
                        .map(|e| e.binding_type)
                });

                let write = match &entry.resource {
                    BoundResource::Buffer(_) => {
                        let info = &buffer_infos[buffer_idx..buffer_idx + 1];
                        buffer_idx += 1;
                        // Use the binding type from layout, defaulting to UNIFORM_BUFFER
                        let descriptor_type =
                            if binding_type == Some(crate::materials::BindingType::StorageBuffer) {
                                vk::DescriptorType::STORAGE_BUFFER
                            } else {
                                vk::DescriptorType::UNIFORM_BUFFER
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
                };
                writes.push(write);
            }

            if !writes.is_empty() {
                log::debug!(
                    "Writing {} descriptors to set: {:?}",
                    writes.len(),
                    writes
                        .iter()
                        .map(|w| (w.dst_binding, w.descriptor_type))
                        .collect::<Vec<_>>()
                );
                unsafe {
                    self.device.update_descriptor_sets(&writes, &[]);
                }
            }

            descriptor_sets.push(descriptor_set);
        }

        // Bind pipeline
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        }

        // Bind descriptor sets
        if !descriptor_sets.is_empty() {
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets,
                    &[],
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
                    self.device
                        .cmd_bind_vertex_buffers(cmd, slot as u32, &[*vk_buffer], &[0]);
                }
            }
        }

        // Issue draw call
        log::debug!(
            "Issuing draw call: indexed={}, vertex_count={}, index_count={}, instance_count={}",
            mesh.is_indexed(),
            mesh.vertex_count(),
            mesh.index_count(),
            draw_cmd.instance_count
        );

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
                    self.device
                        .cmd_bind_index_buffer(cmd, *vk_buffer, 0, index_type);
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

        // Note: In a real implementation, we would cache pipelines and descriptor sets
        // and destroy them properly. For now, we'll leak them until a proper caching
        // mechanism is implemented.
        // TODO: Implement proper pipeline and descriptor set caching/cleanup

        // Clean up shader modules (not cached yet)
        unsafe {
            self.device.destroy_shader_module(vertex_module, None);
            if let Some(frag) = fragment_module {
                self.device.destroy_shader_module(frag, None);
            }
            // Note: We're leaking the pipeline and descriptor set layouts here
            // They should be cached and properly destroyed
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
                use crate::types::TextureDimension;

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

                let block_size = src.format().block_size();
                let dimension = src.dimension();

                // For cubemaps and 2D arrays, origin.z specifies the array layer, not the z offset.
                // Vulkan requires z offset to be 0 for 2D images, with layer specified in subresource.
                let uses_array_layers = matches!(
                    dimension,
                    TextureDimension::Cube | TextureDimension::CubeArray
                ) || (dimension == TextureDimension::D2 && src.depth() > 1);

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

                        // Determine array layer and z offset based on texture type
                        let (base_array_layer, z_offset) = if uses_array_layers {
                            (r.texture_location.origin.z, 0)
                        } else {
                            (0, r.texture_location.origin.z as i32)
                        };

                        vk::BufferImageCopy::default()
                            .buffer_offset(r.buffer_layout.offset)
                            .buffer_row_length(row_length_texels)
                            .buffer_image_height(r.buffer_layout.rows_per_image.unwrap_or(0))
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: r.texture_location.mip_level,
                                base_array_layer,
                                layer_count: 1,
                            })
                            .image_offset(vk::Offset3D {
                                x: r.texture_location.origin.x as i32,
                                y: r.texture_location.origin.y as i32,
                                z: z_offset,
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
                use crate::types::TextureDimension;

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

                let block_size = dst.format().block_size();
                let dimension = dst.dimension();

                // For cubemaps and 2D arrays, origin.z specifies the array layer, not the z offset.
                // Vulkan requires z offset to be 0 for 2D images, with layer specified in subresource.
                let uses_array_layers = matches!(
                    dimension,
                    TextureDimension::Cube | TextureDimension::CubeArray
                ) || (dimension == TextureDimension::D2 && dst.depth() > 1);

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

                        // Determine array layer and z offset based on texture type
                        let (base_array_layer, z_offset) = if uses_array_layers {
                            (r.texture_location.origin.z, 0)
                        } else {
                            (0, r.texture_location.origin.z as i32)
                        };

                        vk::BufferImageCopy::default()
                            .buffer_offset(r.buffer_layout.offset)
                            .buffer_row_length(row_length_texels)
                            .buffer_image_height(r.buffer_layout.rows_per_image.unwrap_or(0))
                            .image_subresource(vk::ImageSubresourceLayers {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                mip_level: r.texture_location.mip_level,
                                base_array_layer,
                                layer_count: 1,
                            })
                            .image_offset(vk::Offset3D {
                                x: r.texture_location.origin.x as i32,
                                y: r.texture_location.origin.y as i32,
                                z: z_offset,
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

                // NOTE: Layout transitions are now handled automatically by the barrier
                // generation system in execute_graph() before each pass is encoded.

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

    #[allow(dead_code)]
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
            vk::ImageLayout::PRESENT_SRC_KHR => (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
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
            vk::ImageLayout::PRESENT_SRC_KHR => (
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
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
