//! Vulkan backend implementation using ash
//!
//! This backend provides direct Vulkan API access for maximum control.

use crate::backend::traits::*;
use crate::backend::types::*;
use ash::khr::{surface, swapchain};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Arc;

/// Vulkan backend implementation
pub struct VulkanBackend {
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_fn: surface::Instance,
    swapchain_fn: swapchain::Device,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue: vk::Queue,
    graphics_queue_family: u32,
    allocator: Option<Arc<Mutex<Allocator>>>,

    // Swapchain
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    current_image_index: u32,

    // Synchronization
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,

    // Command pool and buffer
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    is_recording: bool,

    // Resource storage
    buffers: HashMap<u64, VkBuffer>,
    textures: HashMap<u64, VkTexture>,
    texture_views: HashMap<u64, vk::ImageView>,
    samplers: HashMap<u64, vk::Sampler>,
    descriptor_set_layouts: HashMap<u64, vk::DescriptorSetLayout>,
    descriptor_sets: HashMap<u64, vk::DescriptorSet>,
    render_pipelines: HashMap<u64, VkRenderPipeline>,
    compute_pipelines: HashMap<u64, VkComputePipeline>,
    _pipeline_layouts: HashMap<u64, vk::PipelineLayout>,

    // Descriptor pool
    descriptor_pool: vk::DescriptorPool,

    // Handle counters
    next_buffer_id: u64,
    next_texture_id: u64,
    next_view_id: u64,
    next_sampler_id: u64,
    next_layout_id: u64,
    next_bind_group_id: u64,
    next_render_pipeline_id: u64,
    next_compute_pipeline_id: u64,

    // VSync setting
    vsync: bool,

    // egui render pass
    egui_render_pass: vk::RenderPass,
}

struct VkBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
    _size: u64,
}

struct VkTexture {
    image: vk::Image,
    allocation: Allocation,
    format: vk::Format,
    _extent: vk::Extent3D,
}

struct VkRenderPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

struct VkComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl VulkanBackend {
    /// Get the Vulkan instance
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Get the physical device
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Get the Vulkan device
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Get the graphics queue
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// Get the command pool
    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    /// Get the allocator for egui-ash-renderer
    pub fn allocator(&self) -> Arc<Mutex<Allocator>> {
        self.allocator.clone().expect("Allocator already dropped")
    }

    /// Get the current command buffer (only valid during frame recording)
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    /// Get the current swapchain image view
    pub fn current_swapchain_image_view(&self) -> vk::ImageView {
        self.swapchain_image_views[self.current_image_index as usize]
    }

    /// Get the current swapchain image
    pub fn current_swapchain_image(&self) -> vk::Image {
        self.swapchain_images[self.current_image_index as usize]
    }

    /// Get the egui render pass
    pub fn egui_render_pass(&self) -> vk::RenderPass {
        self.egui_render_pass
    }

    /// Get the swapchain format (Vulkan format)
    pub fn vk_swapchain_format(&self) -> vk::Format {
        self.swapchain_format
    }

    /// Get the swapchain extent
    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.swapchain_extent
    }

    /// Create the egui render pass
    fn create_egui_render_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
        let attachment = vk::AttachmentDescription {
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR, // Clear to background color
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED, // Don't care about previous content
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &attachment_ref,
            ..Default::default()
        };

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &dependency,
            ..Default::default()
        };

        unsafe {
            device
                .create_render_pass(&render_pass_info, None)
                .expect("Failed to create egui render pass")
        }
    }

    fn convert_format(format: TextureFormat) -> vk::Format {
        match format {
            TextureFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            TextureFormat::Rgba8UnormSrgb => vk::Format::R8G8B8A8_SRGB,
            TextureFormat::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            TextureFormat::Bgra8UnormSrgb => vk::Format::B8G8R8A8_SRGB,
            TextureFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            TextureFormat::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            TextureFormat::Depth32Float => vk::Format::D32_SFLOAT,
            TextureFormat::Depth24PlusStencil8 => vk::Format::D24_UNORM_S8_UINT,
            TextureFormat::R32Float => vk::Format::R32_SFLOAT,
            TextureFormat::Rg32Float => vk::Format::R32G32_SFLOAT,
        }
    }

    fn convert_format_back(format: vk::Format) -> TextureFormat {
        match format {
            vk::Format::R8G8B8A8_UNORM => TextureFormat::Rgba8Unorm,
            vk::Format::R8G8B8A8_SRGB => TextureFormat::Rgba8UnormSrgb,
            vk::Format::B8G8R8A8_UNORM => TextureFormat::Bgra8Unorm,
            vk::Format::B8G8R8A8_SRGB => TextureFormat::Bgra8UnormSrgb,
            vk::Format::R16G16B16A16_SFLOAT => TextureFormat::Rgba16Float,
            vk::Format::R32G32B32A32_SFLOAT => TextureFormat::Rgba32Float,
            vk::Format::D32_SFLOAT => TextureFormat::Depth32Float,
            vk::Format::D24_UNORM_S8_UINT => TextureFormat::Depth24PlusStencil8,
            vk::Format::R32_SFLOAT => TextureFormat::R32Float,
            vk::Format::R32G32_SFLOAT => TextureFormat::Rg32Float,
            _ => TextureFormat::Rgba8Unorm,
        }
    }

    fn convert_compare_op(func: CompareFunction) -> vk::CompareOp {
        match func {
            CompareFunction::Never => vk::CompareOp::NEVER,
            CompareFunction::Less => vk::CompareOp::LESS,
            CompareFunction::Equal => vk::CompareOp::EQUAL,
            CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
            CompareFunction::Greater => vk::CompareOp::GREATER,
            CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
            CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
            CompareFunction::Always => vk::CompareOp::ALWAYS,
        }
    }

    fn convert_filter(mode: FilterMode) -> vk::Filter {
        match mode {
            FilterMode::Nearest => vk::Filter::NEAREST,
            FilterMode::Linear => vk::Filter::LINEAR,
        }
    }

    fn convert_address_mode(mode: AddressMode) -> vk::SamplerAddressMode {
        match mode {
            AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        }
    }

    fn find_queue_family(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_fn: &surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Option<u32> {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        for (index, family) in queue_families.iter().enumerate() {
            let supports_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let supports_surface = unsafe {
                surface_fn
                    .get_physical_device_surface_support(physical_device, index as u32, surface)
                    .unwrap_or(false)
            };

            if supports_graphics && supports_surface {
                return Some(index as u32);
            }
        }
        None
    }

    fn create_swapchain(&mut self, width: u32, height: u32) -> BackendResult<()> {
        unsafe {
            self.device.device_wait_idle().ok();

            // Clean up old swapchain resources
            for &view in &self.swapchain_image_views {
                self.device.destroy_image_view(view, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_fn.destroy_swapchain(self.swapchain, None);
            }

            // Query surface capabilities
            let capabilities = self
                .surface_fn
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .map_err(|e| BackendError::SwapchainCreationFailed(e.to_string()))?;

            let formats = self
                .surface_fn
                .get_physical_device_surface_formats(self.physical_device, self.surface)
                .map_err(|e| BackendError::SwapchainCreationFailed(e.to_string()))?;

            let present_modes = self
                .surface_fn
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
                .map_err(|e| BackendError::SwapchainCreationFailed(e.to_string()))?;

            // Choose format (prefer SRGB)
            let format = formats
                .iter()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_SRGB
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .unwrap_or(&formats[0]);

            // Choose present mode
            let present_mode = if self.vsync {
                vk::PresentModeKHR::FIFO
            } else {
                present_modes
                    .iter()
                    .copied()
                    .find(|&m| m == vk::PresentModeKHR::MAILBOX)
                    .unwrap_or(vk::PresentModeKHR::FIFO)
            };

            // Choose extent
            let extent = if capabilities.current_extent.width != u32::MAX {
                capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: width.clamp(
                        capabilities.min_image_extent.width,
                        capabilities.max_image_extent.width,
                    ),
                    height: height.clamp(
                        capabilities.min_image_extent.height,
                        capabilities.max_image_extent.height,
                    ),
                }
            };

            let image_count = (capabilities.min_image_count + 1).min(
                if capabilities.max_image_count > 0 {
                    capabilities.max_image_count
                } else {
                    u32::MAX
                },
            );

            let swapchain_info = vk::SwapchainCreateInfoKHR {
                surface: self.surface,
                min_image_count: image_count,
                image_format: format.format,
                image_color_space: format.color_space,
                image_extent: extent,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: capabilities.current_transform,
                composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                present_mode,
                clipped: vk::TRUE,
                ..Default::default()
            };

            self.swapchain = self
                .swapchain_fn
                .create_swapchain(&swapchain_info, None)
                .map_err(|e| BackendError::SwapchainCreationFailed(e.to_string()))?;

            self.swapchain_images = self
                .swapchain_fn
                .get_swapchain_images(self.swapchain)
                .map_err(|e| BackendError::SwapchainCreationFailed(e.to_string()))?;

            self.swapchain_format = format.format;
            self.swapchain_extent = extent;

            // Create image views
            self.swapchain_image_views = self
                .swapchain_images
                .iter()
                .map(|&image| {
                    let view_info = vk::ImageViewCreateInfo {
                        image,
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: format.format,
                        components: vk::ComponentMapping::default(),
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        ..Default::default()
                    };
                    self.device.create_image_view(&view_info, None)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| BackendError::SwapchainCreationFailed(e.to_string()))?;

            Ok(())
        }
    }

    fn _begin_single_time_commands(&self) -> vk::CommandBuffer {
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };

            let cmd = self.device.allocate_command_buffers(&alloc_info).unwrap()[0];

            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            self.device.begin_command_buffer(cmd, &begin_info).unwrap();

            cmd
        }
    }

    fn _end_single_time_commands(&self, cmd: vk::CommandBuffer) {
        unsafe {
            self.device.end_command_buffer(cmd).unwrap();

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            };

            self.device
                .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                .unwrap();
            self.device.queue_wait_idle(self.graphics_queue).unwrap();

            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }
    }
}

impl GraphicsBackend for VulkanBackend {
    fn new(window: Arc<winit::window::Window>, vsync: bool) -> BackendResult<Self> {
        unsafe {
            let entry = ash::Entry::load()
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;

            // Create instance
            let app_name = CStr::from_bytes_with_nul(b"Graphics Engine\0").unwrap();
            let engine_name = CStr::from_bytes_with_nul(b"Custom Engine\0").unwrap();

            let app_info = vk::ApplicationInfo {
                p_application_name: app_name.as_ptr(),
                application_version: vk::make_api_version(0, 1, 0, 0),
                p_engine_name: engine_name.as_ptr(),
                engine_version: vk::make_api_version(0, 1, 0, 0),
                api_version: vk::API_VERSION_1_2,
                ..Default::default()
            };

            let display_handle = window.display_handle()
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;
            let window_handle = window.window_handle()
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;

            let extensions = ash_window::enumerate_required_extensions(display_handle.as_raw())
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?
                .to_vec();

            let instance_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_extension_count: extensions.len() as u32,
                pp_enabled_extension_names: extensions.as_ptr(),
                ..Default::default()
            };

            let instance = entry
                .create_instance(&instance_info, None)
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;

            // Create surface
            let surface_fn = surface::Instance::new(&entry, &instance);
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
            .map_err(|e| BackendError::SurfaceCreationFailed(e.to_string()))?;

            // Select physical device
            let physical_devices = instance
                .enumerate_physical_devices()
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;

            let physical_device = physical_devices
                .into_iter()
                .find(|&pd| {
                    Self::find_queue_family(&instance, pd, &surface_fn, surface).is_some()
                })
                .ok_or_else(|| {
                    BackendError::InitializationFailed("No suitable physical device".into())
                })?;

            let graphics_queue_family =
                Self::find_queue_family(&instance, physical_device, &surface_fn, surface)
                    .ok_or_else(|| {
                        BackendError::InitializationFailed("No suitable queue family".into())
                    })?;

            // Create logical device
            let queue_priorities = [1.0f32];
            let queue_info = vk::DeviceQueueCreateInfo {
                queue_family_index: graphics_queue_family,
                queue_count: 1,
                p_queue_priorities: queue_priorities.as_ptr(),
                ..Default::default()
            };

            let device_extensions = [swapchain::NAME.as_ptr()];
            let device_features = vk::PhysicalDeviceFeatures::default();

            let device_info = vk::DeviceCreateInfo {
                queue_create_info_count: 1,
                p_queue_create_infos: &queue_info,
                enabled_extension_count: device_extensions.len() as u32,
                pp_enabled_extension_names: device_extensions.as_ptr(),
                p_enabled_features: &device_features,
                ..Default::default()
            };

            let device = instance
                .create_device(physical_device, &device_info, None)
                .map_err(|e| BackendError::DeviceCreationFailed(e.to_string()))?;

            let graphics_queue = device.get_device_queue(graphics_queue_family, 0);

            // Create allocator
            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;

            // Create swapchain loader
            let swapchain_fn = swapchain::Device::new(&instance, &device);

            // Create command pool
            let pool_info = vk::CommandPoolCreateInfo {
                queue_family_index: graphics_queue_family,
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            };

            let command_pool = device
                .create_command_pool(&pool_info, None)
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;

            // Allocate command buffer
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };

            let command_buffer = device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?[0];

            // Create synchronization objects
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let fence_info = vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            };

            let image_available_semaphore = device
                .create_semaphore(&semaphore_info, None)
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;
            let render_finished_semaphore = device
                .create_semaphore(&semaphore_info, None)
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;
            let in_flight_fence = device
                .create_fence(&fence_info, None)
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;

            // Create descriptor pool
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1000,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1000,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1000,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1000,
                },
            ];

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo {
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                max_sets: 1000,
                flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
                ..Default::default()
            };

            let descriptor_pool = device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;

            // Create egui render pass (will be recreated if format changes)
            let egui_render_pass = Self::create_egui_render_pass(&device, vk::Format::B8G8R8A8_SRGB);

            let mut backend = Self {
                _entry: entry,
                instance,
                surface_fn,
                swapchain_fn,
                surface,
                physical_device,
                device,
                graphics_queue,
                graphics_queue_family,
                allocator: Some(Arc::new(Mutex::new(allocator))),
                swapchain: vk::SwapchainKHR::null(),
                swapchain_images: Vec::new(),
                swapchain_image_views: Vec::new(),
                swapchain_format: vk::Format::B8G8R8A8_SRGB,
                swapchain_extent: vk::Extent2D { width: 0, height: 0 },
                current_image_index: 0,
                image_available_semaphore,
                render_finished_semaphore,
                in_flight_fence,
                command_pool,
                command_buffer,
                is_recording: false,
                buffers: HashMap::new(),
                textures: HashMap::new(),
                texture_views: HashMap::new(),
                samplers: HashMap::new(),
                descriptor_set_layouts: HashMap::new(),
                descriptor_sets: HashMap::new(),
                render_pipelines: HashMap::new(),
                compute_pipelines: HashMap::new(),
                _pipeline_layouts: HashMap::new(),
                descriptor_pool,
                next_buffer_id: 1,
                next_texture_id: 1,
                next_view_id: 1,
                next_sampler_id: 1,
                next_layout_id: 1,
                next_bind_group_id: 1,
                next_render_pipeline_id: 1,
                next_compute_pipeline_id: 1,
                vsync,
                egui_render_pass,
            };

            let size = window.inner_size();
            backend.create_swapchain(size.width.max(1), size.height.max(1))?;

            Ok(backend)
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            let _ = self.create_swapchain(width, height);
        }
    }

    fn surface_size(&self) -> (u32, u32) {
        (self.swapchain_extent.width, self.swapchain_extent.height)
    }

    fn begin_frame(&mut self) -> BackendResult<FrameContext> {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fence], true, u64::MAX)
                .map_err(|e| BackendError::AcquireImageFailed(e.to_string()))?;

            let (image_index, _) = self
                .swapchain_fn
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available_semaphore,
                    vk::Fence::null(),
                )
                .map_err(|e| match e {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => BackendError::SurfaceLost,
                    _ => BackendError::AcquireImageFailed(e.to_string()),
                })?;

            self.current_image_index = image_index;

            self.device
                .reset_fences(&[self.in_flight_fence])
                .map_err(|e| BackendError::AcquireImageFailed(e.to_string()))?;

            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| BackendError::AcquireImageFailed(e.to_string()))?;

            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(|e| BackendError::AcquireImageFailed(e.to_string()))?;

            self.is_recording = true;

            // Store swapchain view handle
            let view_id = self.next_view_id;
            self.next_view_id += 1;
            self.texture_views.insert(
                view_id,
                self.swapchain_image_views[image_index as usize],
            );

            Ok(FrameContext {
                swapchain_view: TextureViewHandle(view_id),
                width: self.swapchain_extent.width,
                height: self.swapchain_extent.height,
            })
        }
    }

    fn end_frame(&mut self) -> BackendResult<()> {
        unsafe {
            if self.is_recording {
                self.device
                    .end_command_buffer(self.command_buffer)
                    .map_err(|e| BackendError::PresentFailed(e.to_string()))?;
                self.is_recording = false;
            }

            let wait_semaphores = [self.image_available_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [self.render_finished_semaphore];
            let command_buffers = [self.command_buffer];

            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: wait_semaphores.as_ptr(),
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: command_buffers.as_ptr(),
                signal_semaphore_count: 1,
                p_signal_semaphores: signal_semaphores.as_ptr(),
                ..Default::default()
            };

            self.device
                .queue_submit(self.graphics_queue, &[submit_info], self.in_flight_fence)
                .map_err(|e| BackendError::PresentFailed(e.to_string()))?;

            let swapchains = [self.swapchain];
            let image_indices = [self.current_image_index];

            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: signal_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: swapchains.as_ptr(),
                p_image_indices: image_indices.as_ptr(),
                ..Default::default()
            };

            let _ = self.swapchain_fn.queue_present(self.graphics_queue, &present_info);

            Ok(())
        }
    }

    fn swapchain_format(&self) -> TextureFormat {
        Self::convert_format_back(self.swapchain_format)
    }

    fn create_buffer(&mut self, desc: &BufferDescriptor) -> BackendResult<BufferHandle> {
        unsafe {
            let mut usage = vk::BufferUsageFlags::empty();
            if desc.usage.contains(BufferUsage::VERTEX) {
                usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
            }
            if desc.usage.contains(BufferUsage::INDEX) {
                usage |= vk::BufferUsageFlags::INDEX_BUFFER;
            }
            if desc.usage.contains(BufferUsage::UNIFORM) {
                usage |= vk::BufferUsageFlags::UNIFORM_BUFFER;
            }
            if desc.usage.contains(BufferUsage::STORAGE) {
                usage |= vk::BufferUsageFlags::STORAGE_BUFFER;
            }
            if desc.usage.contains(BufferUsage::COPY_SRC) {
                usage |= vk::BufferUsageFlags::TRANSFER_SRC;
            }
            if desc.usage.contains(BufferUsage::COPY_DST) {
                usage |= vk::BufferUsageFlags::TRANSFER_DST;
            }
            if desc.usage.contains(BufferUsage::INDIRECT) {
                usage |= vk::BufferUsageFlags::INDIRECT_BUFFER;
            }

            let buffer_info = vk::BufferCreateInfo {
                size: desc.size,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|e| BackendError::BufferCreationFailed(e.to_string()))?;

            let requirements = self.device.get_buffer_memory_requirements(buffer);

            let location = if desc.usage.contains(BufferUsage::MAP_READ)
                || desc.usage.contains(BufferUsage::MAP_WRITE)
            {
                MemoryLocation::CpuToGpu
            } else {
                MemoryLocation::GpuOnly
            };

            let allocation = self
                .allocator
                .as_ref()
                .ok_or_else(|| BackendError::BufferCreationFailed("Allocator not available".into()))?
                .lock()
                .allocate(&AllocationCreateDesc {
                    name: desc.label.as_deref().unwrap_or("buffer"),
                    requirements,
                    location,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| BackendError::BufferCreationFailed(e.to_string()))?;

            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| BackendError::BufferCreationFailed(e.to_string()))?;

            let id = self.next_buffer_id;
            self.next_buffer_id += 1;
            self.buffers.insert(
                id,
                VkBuffer {
                    buffer,
                    allocation,
                    _size: desc.size,
                },
            );

            Ok(BufferHandle(id))
        }
    }

    fn create_buffer_init(
        &mut self,
        desc: &BufferDescriptor,
        data: &[u8],
    ) -> BackendResult<BufferHandle> {
        let handle = self.create_buffer(desc)?;
        self.write_buffer(handle, 0, data);
        Ok(handle)
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) {
        if let Some(vk_buffer) = self.buffers.get_mut(&buffer.0) {
            if let Some(mapped) = vk_buffer.allocation.mapped_slice_mut() {
                let start = offset as usize;
                let end = start + data.len();
                if end <= mapped.len() {
                    mapped[start..end].copy_from_slice(data);
                }
            }
        }
    }

    fn create_texture(&mut self, desc: &TextureDescriptor) -> BackendResult<TextureHandle> {
        unsafe {
            let format = Self::convert_format(desc.format);
            let is_depth = desc.format.is_depth();

            let mut usage = vk::ImageUsageFlags::empty();
            if desc.usage.contains(TextureUsage::COPY_SRC) {
                usage |= vk::ImageUsageFlags::TRANSFER_SRC;
            }
            if desc.usage.contains(TextureUsage::COPY_DST) {
                usage |= vk::ImageUsageFlags::TRANSFER_DST;
            }
            if desc.usage.contains(TextureUsage::TEXTURE_BINDING) {
                usage |= vk::ImageUsageFlags::SAMPLED;
            }
            if desc.usage.contains(TextureUsage::STORAGE_BINDING) {
                usage |= vk::ImageUsageFlags::STORAGE;
            }
            if desc.usage.contains(TextureUsage::RENDER_ATTACHMENT) {
                if is_depth {
                    usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
                } else {
                    usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
                }
            }

            let extent = vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth,
            };

            let image_info = vk::ImageCreateInfo {
                image_type: if desc.depth > 1 {
                    vk::ImageType::TYPE_3D
                } else {
                    vk::ImageType::TYPE_2D
                },
                extent,
                mip_levels: desc.mip_levels,
                array_layers: 1,
                format,
                tiling: vk::ImageTiling::OPTIMAL,
                initial_layout: vk::ImageLayout::UNDEFINED,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            let image = self
                .device
                .create_image(&image_info, None)
                .map_err(|e| BackendError::TextureCreationFailed(e.to_string()))?;

            let requirements = self.device.get_image_memory_requirements(image);

            let allocation = self
                .allocator
                .as_ref()
                .ok_or_else(|| BackendError::TextureCreationFailed("Allocator not available".into()))?
                .lock()
                .allocate(&AllocationCreateDesc {
                    name: desc.label.as_deref().unwrap_or("texture"),
                    requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| BackendError::TextureCreationFailed(e.to_string()))?;

            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(|e| BackendError::TextureCreationFailed(e.to_string()))?;

            let id = self.next_texture_id;
            self.next_texture_id += 1;
            self.textures.insert(
                id,
                VkTexture {
                    image,
                    allocation,
                    format,
                    _extent: extent,
                },
            );

            Ok(TextureHandle(id))
        }
    }

    fn create_texture_view(&mut self, texture: TextureHandle) -> BackendResult<TextureViewHandle> {
        let tex = self
            .textures
            .get(&texture.0)
            .ok_or_else(|| BackendError::TextureCreationFailed("Texture not found".into()))?;

        let is_depth = matches!(
            tex.format,
            vk::Format::D32_SFLOAT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D16_UNORM
        );

        let aspect_mask = if is_depth {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let view_info = vk::ImageViewCreateInfo {
            image: tex.image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: tex.format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .map_err(|e| BackendError::TextureCreationFailed(e.to_string()))?
        };

        let id = self.next_view_id;
        self.next_view_id += 1;
        self.texture_views.insert(id, view);

        Ok(TextureViewHandle(id))
    }

    fn write_texture(&mut self, _texture: TextureHandle, _data: &[u8], _width: u32, _height: u32) {
        // Simplified implementation - would need staging buffer
    }

    fn create_sampler(&mut self, desc: &SamplerDescriptor) -> BackendResult<SamplerHandle> {
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: Self::convert_filter(desc.mag_filter),
            min_filter: Self::convert_filter(desc.min_filter),
            mipmap_mode: match desc.mipmap_filter {
                FilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
                FilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
            },
            address_mode_u: Self::convert_address_mode(desc.address_mode_u),
            address_mode_v: Self::convert_address_mode(desc.address_mode_v),
            address_mode_w: Self::convert_address_mode(desc.address_mode_w),
            compare_enable: if desc.compare.is_some() { vk::TRUE } else { vk::FALSE },
            compare_op: desc.compare.map(Self::convert_compare_op).unwrap_or(vk::CompareOp::ALWAYS),
            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE,
            border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
            ..Default::default()
        };

        let sampler = unsafe {
            self.device
                .create_sampler(&sampler_info, None)
                .map_err(|e| BackendError::TextureCreationFailed(e.to_string()))?
        };

        let id = self.next_sampler_id;
        self.next_sampler_id += 1;
        self.samplers.insert(id, sampler);

        Ok(SamplerHandle(id))
    }

    fn create_bind_group_layout(
        &mut self,
        entries: &[BindGroupLayoutEntry],
    ) -> BackendResult<BindGroupLayoutHandle> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = entries
            .iter()
            .map(|e| {
                let descriptor_type = match &e.ty {
                    BindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                    BindingType::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
                    BindingType::Texture { .. } => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    BindingType::StorageTexture { .. } => vk::DescriptorType::STORAGE_IMAGE,
                    BindingType::Sampler { .. } => vk::DescriptorType::SAMPLER,
                };

                let mut stage_flags = vk::ShaderStageFlags::empty();
                if e.visibility.contains(ShaderStageFlags::VERTEX) {
                    stage_flags |= vk::ShaderStageFlags::VERTEX;
                }
                if e.visibility.contains(ShaderStageFlags::FRAGMENT) {
                    stage_flags |= vk::ShaderStageFlags::FRAGMENT;
                }
                if e.visibility.contains(ShaderStageFlags::COMPUTE) {
                    stage_flags |= vk::ShaderStageFlags::COMPUTE;
                }

                vk::DescriptorSetLayoutBinding {
                    binding: e.binding,
                    descriptor_type,
                    descriptor_count: 1,
                    stage_flags,
                    ..Default::default()
                }
            })
            .collect();

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        let layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| BackendError::PipelineCreationFailed(e.to_string()))?
        };

        let id = self.next_layout_id;
        self.next_layout_id += 1;
        self.descriptor_set_layouts.insert(id, layout);

        Ok(BindGroupLayoutHandle(id))
    }

    fn create_bind_group(
        &mut self,
        layout: BindGroupLayoutHandle,
        _entries: &[(u32, BindGroupEntry)],
    ) -> BackendResult<BindGroupHandle> {
        let layout_handle = self
            .descriptor_set_layouts
            .get(&layout.0)
            .ok_or_else(|| BackendError::PipelineCreationFailed("Layout not found".into()))?;

        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: layout_handle,
            ..Default::default()
        };

        let descriptor_set = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| BackendError::PipelineCreationFailed(e.to_string()))?[0]
        };

        let id = self.next_bind_group_id;
        self.next_bind_group_id += 1;
        self.descriptor_sets.insert(id, descriptor_set);

        Ok(BindGroupHandle(id))
    }

    fn create_render_pipeline(
        &mut self,
        desc: &RenderPipelineDescriptor,
    ) -> BackendResult<RenderPipelineHandle> {
        let layouts: Vec<vk::DescriptorSetLayout> = desc
            .bind_group_layouts
            .iter()
            .filter_map(|h| self.descriptor_set_layouts.get(&h.0).copied())
            .collect();

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| BackendError::PipelineCreationFailed(e.to_string()))?
        };

        let id = self.next_render_pipeline_id;
        self.next_render_pipeline_id += 1;
        self.render_pipelines.insert(
            id,
            VkRenderPipeline {
                pipeline: vk::Pipeline::null(),
                layout: pipeline_layout,
            },
        );

        Ok(RenderPipelineHandle(id))
    }

    fn create_compute_pipeline(
        &mut self,
        desc: &ComputePipelineDescriptor,
    ) -> BackendResult<ComputePipelineHandle> {
        let layouts: Vec<vk::DescriptorSetLayout> = desc
            .bind_group_layouts
            .iter()
            .filter_map(|h| self.descriptor_set_layouts.get(&h.0).copied())
            .collect();

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| BackendError::PipelineCreationFailed(e.to_string()))?
        };

        let id = self.next_compute_pipeline_id;
        self.next_compute_pipeline_id += 1;
        self.compute_pipelines.insert(
            id,
            VkComputePipeline {
                pipeline: vk::Pipeline::null(),
                layout: pipeline_layout,
            },
        );

        Ok(ComputePipelineHandle(id))
    }

    fn begin_render_pass(&mut self, _desc: &RenderPassDescriptor) {}
    fn end_render_pass(&mut self) {}
    fn begin_compute_pass(&mut self, _label: Option<&str>) {}
    fn end_compute_pass(&mut self) {}

    fn set_render_pipeline(&mut self, _pipeline: RenderPipelineHandle) {}
    fn set_compute_pipeline(&mut self, _pipeline: ComputePipelineHandle) {}
    fn set_bind_group(&mut self, _index: u32, _bind_group: BindGroupHandle) {}
    fn set_vertex_buffer(&mut self, _slot: u32, _buffer: BufferHandle, _offset: u64) {}
    fn set_index_buffer(&mut self, _buffer: BufferHandle, _offset: u64, _format: IndexFormat) {}

    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        let viewport = vk::Viewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        };
        unsafe {
            self.device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
        }
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: x as i32, y: y as i32 },
            extent: vk::Extent2D { width, height },
        };
        unsafe {
            self.device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);
        }
    }

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        unsafe {
            self.device.cmd_draw(
                self.command_buffer,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
        }
    }

    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        unsafe {
            self.device.cmd_draw_indexed(
                self.command_buffer,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            );
        }
    }

    fn dispatch_compute(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            self.device.cmd_dispatch(self.command_buffer, x, y, z);
        }
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        if let Some(vk_buffer) = self.buffers.remove(&buffer.0) {
            unsafe {
                self.device.destroy_buffer(vk_buffer.buffer, None);
                if let Some(ref allocator) = self.allocator {
                    let _ = allocator.lock().free(vk_buffer.allocation);
                }
            }
        }
    }

    fn destroy_texture(&mut self, texture: TextureHandle) {
        if let Some(vk_texture) = self.textures.remove(&texture.0) {
            unsafe {
                self.device.destroy_image(vk_texture.image, None);
                if let Some(ref allocator) = self.allocator {
                    let _ = allocator.lock().free(vk_texture.allocation);
                }
            }
        }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            // Free all buffer allocations
            if let Some(ref allocator) = self.allocator {
                for (_, buffer) in self.buffers.drain() {
                    self.device.destroy_buffer(buffer.buffer, None);
                    let _ = allocator.lock().free(buffer.allocation);
                }

                for (_, texture) in self.textures.drain() {
                    self.device.destroy_image(texture.image, None);
                    let _ = allocator.lock().free(texture.allocation);
                }
            }

            // Drop the allocator before destroying the device
            drop(self.allocator.take());

            // Don't destroy swapchain image views here - they're destroyed with the swapchain
            for (_, view) in self.texture_views.drain() {
                if !self.swapchain_image_views.contains(&view) {
                    self.device.destroy_image_view(view, None);
                }
            }

            for (_, sampler) in self.samplers.drain() {
                self.device.destroy_sampler(sampler, None);
            }

            for (_, layout) in self.descriptor_set_layouts.drain() {
                self.device.destroy_descriptor_set_layout(layout, None);
            }

            for (_, pipeline) in self.render_pipelines.drain() {
                if pipeline.pipeline != vk::Pipeline::null() {
                    self.device.destroy_pipeline(pipeline.pipeline, None);
                }
                self.device.destroy_pipeline_layout(pipeline.layout, None);
            }

            for (_, pipeline) in self.compute_pipelines.drain() {
                if pipeline.pipeline != vk::Pipeline::null() {
                    self.device.destroy_pipeline(pipeline.pipeline, None);
                }
                self.device.destroy_pipeline_layout(pipeline.layout, None);
            }

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_semaphore(self.image_available_semaphore, None);
            self.device.destroy_semaphore(self.render_finished_semaphore, None);
            self.device.destroy_fence(self.in_flight_fence, None);

            // Destroy egui render pass
            self.device.destroy_render_pass(self.egui_render_pass, None);

            for &view in &self.swapchain_image_views {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_fn.destroy_swapchain(self.swapchain, None);

            self.device.destroy_device(None);
            self.surface_fn.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
