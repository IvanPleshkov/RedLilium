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

    // Render pass and framebuffer caches
    render_passes: HashMap<RenderPassKey, vk::RenderPass>,
    framebuffers: HashMap<FramebufferKey, vk::Framebuffer>,

    // Framebuffers to destroy after the current frame is submitted
    framebuffers_to_destroy: Vec<vk::Framebuffer>,

    // Pending passes (command buffering)
    pending_render_pass: Option<PendingVkRenderPass>,
    pending_compute_pass: Option<PendingVkComputePass>,

    // Current pipeline for bind group binding
    current_render_pipeline: Option<RenderPipelineHandle>,
    current_compute_pipeline: Option<ComputePipelineHandle>,

    // Track texture view formats for render pass creation
    texture_view_formats: HashMap<u64, vk::Format>,
    texture_view_parents: HashMap<u64, TextureHandle>,

    // Track which views are for swapchain images
    swapchain_view_id: Option<u64>,
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

/// Buffered render pass command (deferred execution pattern matching wgpu)
#[derive(Clone)]
enum VkRenderCommand {
    SetPipeline(RenderPipelineHandle),
    SetBindGroup { index: u32, bind_group: BindGroupHandle },
    SetVertexBuffer { slot: u32, buffer: BufferHandle, offset: u64 },
    SetIndexBuffer { buffer: BufferHandle, offset: u64, format: IndexFormat },
    SetViewport { x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32 },
    SetScissorRect { x: u32, y: u32, width: u32, height: u32 },
    Draw { vertices: std::ops::Range<u32>, instances: std::ops::Range<u32> },
    DrawIndexed { indices: std::ops::Range<u32>, base_vertex: i32, instances: std::ops::Range<u32> },
}

/// Buffered compute pass command
#[derive(Clone)]
enum VkComputeCommand {
    SetPipeline(ComputePipelineHandle),
    SetBindGroup { index: u32, bind_group: BindGroupHandle },
    Dispatch { x: u32, y: u32, z: u32 },
}

/// Pending render pass with buffered commands
struct PendingVkRenderPass {
    descriptor: RenderPassDescriptor,
    commands: Vec<VkRenderCommand>,
}

/// Pending compute pass with buffered commands
struct PendingVkComputePass {
    _label: Option<String>,
    commands: Vec<VkComputeCommand>,
}

/// Key for render pass cache
#[derive(Clone, PartialEq, Eq, Hash)]
struct RenderPassKey {
    color_formats: Vec<vk::Format>,
    depth_format: Option<vk::Format>,
    color_load_ops: Vec<bool>, // true = clear, false = load
    depth_load_op: Option<bool>,
    color_store_ops: Vec<bool>, // true = store, false = discard
    depth_store_op: Option<bool>,
    is_present_pass: bool, // final layout is PRESENT_SRC_KHR
}

/// Key for framebuffer cache
#[derive(Clone, PartialEq, Eq, Hash)]
struct FramebufferKey {
    render_pass: vk::RenderPass,
    attachments: Vec<vk::ImageView>,
    width: u32,
    height: u32,
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
            load_op: vk::AttachmentLoadOp::LOAD, // Preserve scene content rendered before egui
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::PRESENT_SRC_KHR, // Lighting pass left it in present layout
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

    fn convert_topology(topology: PrimitiveTopology) -> vk::PrimitiveTopology {
        match topology {
            PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
            PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        }
    }

    fn convert_cull_mode(mode: CullMode) -> vk::CullModeFlags {
        match mode {
            CullMode::None => vk::CullModeFlags::NONE,
            CullMode::Front => vk::CullModeFlags::FRONT,
            CullMode::Back => vk::CullModeFlags::BACK,
        }
    }

    fn convert_front_face(face: FrontFace) -> vk::FrontFace {
        // Invert front face because we flip Y via negative viewport height
        // This is needed to match wgpu/WebGPU coordinate system
        match face {
            FrontFace::Ccw => vk::FrontFace::CLOCKWISE,
            FrontFace::Cw => vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }

    fn convert_blend_factor(factor: BlendFactor) -> vk::BlendFactor {
        match factor {
            BlendFactor::Zero => vk::BlendFactor::ZERO,
            BlendFactor::One => vk::BlendFactor::ONE,
            BlendFactor::Src => vk::BlendFactor::SRC_COLOR,
            BlendFactor::OneMinusSrc => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            BlendFactor::Dst => vk::BlendFactor::DST_COLOR,
            BlendFactor::OneMinusDst => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        }
    }

    fn convert_blend_op(op: BlendOperation) -> vk::BlendOp {
        match op {
            BlendOperation::Add => vk::BlendOp::ADD,
            BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
            BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
            BlendOperation::Min => vk::BlendOp::MIN,
            BlendOperation::Max => vk::BlendOp::MAX,
        }
    }

    fn convert_vertex_format(format: VertexFormat) -> vk::Format {
        match format {
            VertexFormat::Float32 => vk::Format::R32_SFLOAT,
            VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
            VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
            VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
            VertexFormat::Uint32 => vk::Format::R32_UINT,
            VertexFormat::Sint32 => vk::Format::R32_SINT,
        }
    }

    /// Compile WGSL shader to SPIR-V using naga
    fn compile_wgsl_to_spirv(wgsl_source: &str) -> BackendResult<Vec<u32>> {
        use naga::front::wgsl;
        use naga::back::spv;
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        // Parse WGSL
        let module = wgsl::parse_str(wgsl_source)
            .map_err(|e| BackendError::ShaderCreationFailed(format!("WGSL parse error: {}", e)))?;

        // Validate
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let info = validator.validate(&module)
            .map_err(|e| BackendError::ShaderCreationFailed(format!("Shader validation error: {:?}", e)))?;

        // Generate SPIR-V
        let options = spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        };

        let spv = spv::write_vec(&module, &info, &options, None)
            .map_err(|e| BackendError::ShaderCreationFailed(format!("SPIR-V generation error: {:?}", e)))?;

        Ok(spv)
    }

    /// Create a Vulkan shader module from SPIR-V bytecode
    fn create_shader_module(&self, spirv: &[u32]) -> BackendResult<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: spirv.len() * 4,
            p_code: spirv.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device
                .create_shader_module(&create_info, None)
                .map_err(|e| BackendError::ShaderCreationFailed(e.to_string()))
        }
    }

    /// Get or create a render pass for the given attachment configuration
    fn get_or_create_render_pass(
        &mut self,
        color_formats: &[vk::Format],
        depth_format: Option<vk::Format>,
        color_load_ops: &[bool],
        depth_load_op: Option<bool>,
        color_store_ops: &[bool],
        depth_store_op: Option<bool>,
        is_present_pass: bool,
    ) -> BackendResult<vk::RenderPass> {
        let key = RenderPassKey {
            color_formats: color_formats.to_vec(),
            depth_format,
            color_load_ops: color_load_ops.to_vec(),
            depth_load_op,
            color_store_ops: color_store_ops.to_vec(),
            depth_store_op,
            is_present_pass,
        };

        if let Some(&render_pass) = self.render_passes.get(&key) {
            return Ok(render_pass);
        }

        // Create new render pass
        let mut attachments = Vec::new();
        let mut color_refs = Vec::new();

        // Color attachments
        for (i, format) in color_formats.iter().enumerate() {
            let load_op = if color_load_ops.get(i).copied().unwrap_or(true) {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::LOAD
            };
            let store_op = if color_store_ops.get(i).copied().unwrap_or(true) {
                vk::AttachmentStoreOp::STORE
            } else {
                vk::AttachmentStoreOp::DONT_CARE
            };

            let final_layout = if is_present_pass && i == 0 {
                vk::ImageLayout::PRESENT_SRC_KHR
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            };

            attachments.push(vk::AttachmentDescription {
                format: *format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op,
                store_op,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: if load_op == vk::AttachmentLoadOp::LOAD {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                final_layout,
                ..Default::default()
            });

            color_refs.push(vk::AttachmentReference {
                attachment: i as u32,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            });
        }

        // Depth attachment
        let depth_ref = if let Some(format) = depth_format {
            let load_op = if depth_load_op.unwrap_or(true) {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::LOAD
            };
            let store_op = if depth_store_op.unwrap_or(true) {
                vk::AttachmentStoreOp::STORE
            } else {
                vk::AttachmentStoreOp::DONT_CARE
            };

            attachments.push(vk::AttachmentDescription {
                format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op,
                store_op,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: if load_op == vk::AttachmentLoadOp::LOAD {
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::UNDEFINED
                },
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            });

            Some(vk::AttachmentReference {
                attachment: attachments.len() as u32 - 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            })
        } else {
            None
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: color_refs.len() as u32,
            p_color_attachments: if color_refs.is_empty() { std::ptr::null() } else { color_refs.as_ptr() },
            p_depth_stencil_attachment: depth_ref
                .as_ref()
                .map(|r| r as *const _)
                .unwrap_or(std::ptr::null()),
            ..Default::default()
        };

        // Subpass dependencies
        let dependencies = [
            vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                dst_subpass: 0,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ..Default::default()
            },
        ];

        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: if attachments.is_empty() { std::ptr::null() } else { attachments.as_ptr() },
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: dependencies.len() as u32,
            p_dependencies: dependencies.as_ptr(),
            ..Default::default()
        };

        let render_pass = unsafe {
            self.device
                .create_render_pass(&render_pass_info, None)
                .map_err(|e| BackendError::PipelineCreationFailed(e.to_string()))?
        };

        self.render_passes.insert(key, render_pass);
        Ok(render_pass)
    }

    /// Get or create a framebuffer for the given render pass and attachments
    fn get_or_create_framebuffer(
        &mut self,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
        width: u32,
        height: u32,
    ) -> BackendResult<vk::Framebuffer> {
        let key = FramebufferKey {
            render_pass,
            attachments: attachments.to_vec(),
            width,
            height,
        };

        if let Some(&framebuffer) = self.framebuffers.get(&key) {
            return Ok(framebuffer);
        }

        let framebuffer_info = vk::FramebufferCreateInfo {
            render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width,
            height,
            layers: 1,
            ..Default::default()
        };

        let framebuffer = unsafe {
            self.device
                .create_framebuffer(&framebuffer_info, None)
                .map_err(|e| BackendError::PipelineCreationFailed(e.to_string()))?
        };

        self.framebuffers.insert(key, framebuffer);
        Ok(framebuffer)
    }

    /// Build vertex input state from vertex layouts
    fn build_vertex_input_state(layouts: &[VertexBufferLayout]) -> (
        Vec<vk::VertexInputBindingDescription>,
        Vec<vk::VertexInputAttributeDescription>,
    ) {
        let mut binding_descs = Vec::new();
        let mut attribute_descs = Vec::new();

        for (binding, layout) in layouts.iter().enumerate() {
            binding_descs.push(vk::VertexInputBindingDescription {
                binding: binding as u32,
                stride: layout.array_stride as u32,
                input_rate: match layout.step_mode {
                    VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                },
            });

            for attr in &layout.attributes {
                attribute_descs.push(vk::VertexInputAttributeDescription {
                    location: attr.location,
                    binding: binding as u32,
                    format: Self::convert_vertex_format(attr.format),
                    offset: attr.offset as u32,
                });
            }
        }

        (binding_descs, attribute_descs)
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

            // Clear framebuffer cache (they reference old swapchain views)
            for (_, fb) in self.framebuffers.drain() {
                self.device.destroy_framebuffer(fb, None);
            }

            // Also clear any deferred framebuffers
            for fb in self.framebuffers_to_destroy.drain(..) {
                self.device.destroy_framebuffer(fb, None);
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

            let mut extensions = ash_window::enumerate_required_extensions(display_handle.as_raw())
                .map_err(|e| BackendError::InitializationFailed(e.to_string()))?
                .to_vec();

            // Enable validation layers in debug builds
            #[cfg(debug_assertions)]
            let validation_layer_name = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();

            #[cfg(debug_assertions)]
            let layer_names = [validation_layer_name.as_ptr()];

            #[cfg(debug_assertions)]
            {
                // Add debug utils extension for validation messages
                extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            }

            #[cfg(debug_assertions)]
            let instance_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_extension_count: extensions.len() as u32,
                pp_enabled_extension_names: extensions.as_ptr(),
                enabled_layer_count: layer_names.len() as u32,
                pp_enabled_layer_names: layer_names.as_ptr(),
                ..Default::default()
            };

            #[cfg(not(debug_assertions))]
            let instance_info = vk::InstanceCreateInfo {
                p_application_info: &app_info,
                enabled_extension_count: extensions.len() as u32,
                pp_enabled_extension_names: extensions.as_ptr(),
                ..Default::default()
            };

            let instance = entry
                .create_instance(&instance_info, None)
                .map_err(|e| BackendError::InitializationFailed(format!("Failed to create Vulkan instance: {}. Make sure Vulkan drivers are installed.", e)))?;

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
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1000,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLER,
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
                render_passes: HashMap::new(),
                framebuffers: HashMap::new(),
                framebuffers_to_destroy: Vec::new(),
                pending_render_pass: None,
                pending_compute_pass: None,
                current_render_pipeline: None,
                current_compute_pipeline: None,
                texture_view_formats: HashMap::new(),
                texture_view_parents: HashMap::new(),
                swapchain_view_id: None,
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

            // Clean up framebuffers that were deferred from the previous frame
            for fb in self.framebuffers_to_destroy.drain(..) {
                self.device.destroy_framebuffer(fb, None);
            }

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

            // Store swapchain view handle and track it for render pass creation
            let view_id = self.next_view_id;
            self.next_view_id += 1;
            self.texture_views.insert(
                view_id,
                self.swapchain_image_views[image_index as usize],
            );
            self.texture_view_formats.insert(view_id, self.swapchain_format);
            self.swapchain_view_id = Some(view_id);

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
        self.texture_view_formats.insert(id, tex.format);
        self.texture_view_parents.insert(id, texture);

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
                    BindingType::Texture { .. } => vk::DescriptorType::SAMPLED_IMAGE,
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
        entries: &[(u32, BindGroupEntry)],
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

        // Build descriptor writes
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();

        // Pre-allocate to ensure pointers remain valid
        for (_, entry) in entries {
            match entry {
                BindGroupEntry::Buffer { .. } => buffer_infos.push(vk::DescriptorBufferInfo::default()),
                BindGroupEntry::Texture(_) | BindGroupEntry::StorageTexture(_) | BindGroupEntry::Sampler(_) => {
                    image_infos.push(vk::DescriptorImageInfo::default())
                }
            }
        }

        let mut buffer_idx = 0;
        let mut image_idx = 0;
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

        for (binding, entry) in entries {
            match entry {
                BindGroupEntry::Buffer { buffer, offset, size } => {
                    if let Some(vk_buffer) = self.buffers.get(&buffer.0) {
                        buffer_infos[buffer_idx] = vk::DescriptorBufferInfo {
                            buffer: vk_buffer.buffer,
                            offset: *offset,
                            range: size.unwrap_or(vk::WHOLE_SIZE),
                        };

                        writes.push(vk::WriteDescriptorSet {
                            dst_set: descriptor_set,
                            dst_binding: *binding,
                            dst_array_element: 0,
                            descriptor_count: 1,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            p_buffer_info: &buffer_infos[buffer_idx],
                            ..Default::default()
                        });
                        buffer_idx += 1;
                    }
                }
                BindGroupEntry::Texture(view) => {
                    if let Some(&image_view) = self.texture_views.get(&view.0) {
                        image_infos[image_idx] = vk::DescriptorImageInfo {
                            image_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            sampler: vk::Sampler::null(), // Not needed for SAMPLED_IMAGE
                        };

                        writes.push(vk::WriteDescriptorSet {
                            dst_set: descriptor_set,
                            dst_binding: *binding,
                            dst_array_element: 0,
                            descriptor_count: 1,
                            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                            p_image_info: &image_infos[image_idx],
                            ..Default::default()
                        });
                        image_idx += 1;
                    }
                }
                BindGroupEntry::StorageTexture(view) => {
                    if let Some(&image_view) = self.texture_views.get(&view.0) {
                        image_infos[image_idx] = vk::DescriptorImageInfo {
                            image_view,
                            image_layout: vk::ImageLayout::GENERAL,
                            sampler: vk::Sampler::null(),
                        };

                        writes.push(vk::WriteDescriptorSet {
                            dst_set: descriptor_set,
                            dst_binding: *binding,
                            dst_array_element: 0,
                            descriptor_count: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            p_image_info: &image_infos[image_idx],
                            ..Default::default()
                        });
                        image_idx += 1;
                    }
                }
                BindGroupEntry::Sampler(sampler) => {
                    if let Some(&vk_sampler) = self.samplers.get(&sampler.0) {
                        image_infos[image_idx] = vk::DescriptorImageInfo {
                            sampler: vk_sampler,
                            image_view: vk::ImageView::null(),
                            image_layout: vk::ImageLayout::UNDEFINED,
                        };

                        writes.push(vk::WriteDescriptorSet {
                            dst_set: descriptor_set,
                            dst_binding: *binding,
                            dst_array_element: 0,
                            descriptor_count: 1,
                            descriptor_type: vk::DescriptorType::SAMPLER,
                            p_image_info: &image_infos[image_idx],
                            ..Default::default()
                        });
                        image_idx += 1;
                    }
                }
            }
        }

        // Update descriptor sets
        if !writes.is_empty() {
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
            }
        }

        let id = self.next_bind_group_id;
        self.next_bind_group_id += 1;
        self.descriptor_sets.insert(id, descriptor_set);

        Ok(BindGroupHandle(id))
    }

    fn create_render_pipeline(
        &mut self,
        desc: &RenderPipelineDescriptor,
    ) -> BackendResult<RenderPipelineHandle> {
        // Compile shaders
        let vert_spirv = Self::compile_wgsl_to_spirv(&desc.vertex_shader)?;
        let vert_module = self.create_shader_module(&vert_spirv)?;

        let frag_module = if let Some(ref frag_src) = desc.fragment_shader {
            let frag_spirv = Self::compile_wgsl_to_spirv(frag_src)?;
            Some(self.create_shader_module(&frag_spirv)?)
        } else {
            None
        };

        // Entry point names - naga preserves original WGSL entry point names
        let vs_entry_point = std::ffi::CString::new("vs_main").unwrap();
        let fs_entry_point = std::ffi::CString::new("fs_main").unwrap();

        let mut shader_stages = vec![
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_module,
                p_name: vs_entry_point.as_ptr(),
                ..Default::default()
            },
        ];

        if let Some(frag) = frag_module {
            shader_stages.push(vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag,
                p_name: fs_entry_point.as_ptr(),
                ..Default::default()
            });
        }

        // Vertex input state
        let (binding_descs, attribute_descs) = Self::build_vertex_input_state(&desc.vertex_layouts);

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: binding_descs.len() as u32,
            p_vertex_binding_descriptions: if binding_descs.is_empty() { std::ptr::null() } else { binding_descs.as_ptr() },
            vertex_attribute_description_count: attribute_descs.len() as u32,
            p_vertex_attribute_descriptions: if attribute_descs.is_empty() { std::ptr::null() } else { attribute_descs.as_ptr() },
            ..Default::default()
        };

        // Input assembly
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: Self::convert_topology(desc.primitive_topology),
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        // Viewport and scissor (dynamic)
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        // Rasterization
        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: Self::convert_cull_mode(desc.cull_mode),
            front_face: Self::convert_front_face(desc.front_face),
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        // Multisampling
        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        // Depth stencil
        let depth_stencil = desc.depth_stencil.as_ref().map(|ds| {
            vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: if ds.depth_write_enabled { vk::TRUE } else { vk::FALSE },
                depth_compare_op: Self::convert_compare_op(ds.depth_compare),
                depth_bounds_test_enable: vk::FALSE,
                stencil_test_enable: vk::FALSE,
                ..Default::default()
            }
        });

        // Color blend attachments
        let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = desc
            .color_targets
            .iter()
            .map(|target| {
                if let Some(blend) = &target.blend {
                    vk::PipelineColorBlendAttachmentState {
                        blend_enable: vk::TRUE,
                        src_color_blend_factor: Self::convert_blend_factor(blend.color.src_factor),
                        dst_color_blend_factor: Self::convert_blend_factor(blend.color.dst_factor),
                        color_blend_op: Self::convert_blend_op(blend.color.operation),
                        src_alpha_blend_factor: Self::convert_blend_factor(blend.alpha.src_factor),
                        dst_alpha_blend_factor: Self::convert_blend_factor(blend.alpha.dst_factor),
                        alpha_blend_op: Self::convert_blend_op(blend.alpha.operation),
                        color_write_mask: vk::ColorComponentFlags::from_raw(target.write_mask.0),
                        ..Default::default()
                    }
                } else {
                    vk::PipelineColorBlendAttachmentState {
                        blend_enable: vk::FALSE,
                        color_write_mask: vk::ColorComponentFlags::from_raw(target.write_mask.0),
                        ..Default::default()
                    }
                }
            })
            .collect();

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            attachment_count: color_blend_attachments.len() as u32,
            p_attachments: if color_blend_attachments.is_empty() { std::ptr::null() } else { color_blend_attachments.as_ptr() },
            ..Default::default()
        };

        // Dynamic state
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        // Pipeline layout
        let layouts: Vec<vk::DescriptorSetLayout> = desc
            .bind_group_layouts
            .iter()
            .filter_map(|h| self.descriptor_set_layouts.get(&h.0).copied())
            .collect();

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: layouts.len() as u32,
            p_set_layouts: if layouts.is_empty() { std::ptr::null() } else { layouts.as_ptr() },
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| BackendError::PipelineCreationFailed(e.to_string()))?
        };

        // Create compatible render pass for pipeline
        let color_formats: Vec<vk::Format> = desc.color_targets
            .iter()
            .map(|t| Self::convert_format(t.format))
            .collect();
        let depth_format = desc.depth_stencil.as_ref().map(|ds| Self::convert_format(ds.format));
        let color_load_ops = vec![true; color_formats.len()]; // clear
        let color_store_ops = vec![true; color_formats.len()]; // store
        let depth_load_op = depth_format.map(|_| true);
        let depth_store_op = depth_format.map(|_| true);

        let render_pass = self.get_or_create_render_pass(
            &color_formats,
            depth_format,
            &color_load_ops,
            depth_load_op,
            &color_store_ops,
            depth_store_op,
            false, // not a present pass
        )?;

        // Create pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: depth_stencil
                .as_ref()
                .map(|ds| ds as *const _)
                .unwrap_or(std::ptr::null()),
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            ..Default::default()
        };

        let pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| BackendError::PipelineCreationFailed(e.to_string()))?[0]
        };

        // Cleanup shader modules
        unsafe {
            self.device.destroy_shader_module(vert_module, None);
            if let Some(frag) = frag_module {
                self.device.destroy_shader_module(frag, None);
            }
        }

        let id = self.next_render_pipeline_id;
        self.next_render_pipeline_id += 1;
        self.render_pipelines.insert(
            id,
            VkRenderPipeline {
                pipeline,
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

    fn begin_render_pass(&mut self, desc: &RenderPassDescriptor) {
        // Store the descriptor for deferred execution (command buffering pattern)
        self.pending_render_pass = Some(PendingVkRenderPass {
            descriptor: desc.clone(),
            commands: Vec::new(),
        });
    }

    fn end_render_pass(&mut self) {
        let Some(pending) = self.pending_render_pass.take() else {
            return;
        };

        let desc = &pending.descriptor;

        // Collect image views and formats for attachments
        let mut attachment_views: Vec<vk::ImageView> = Vec::new();
        let mut color_formats: Vec<vk::Format> = Vec::new();
        let mut color_load_ops: Vec<bool> = Vec::new();
        let mut color_store_ops: Vec<bool> = Vec::new();
        let mut clear_values: Vec<vk::ClearValue> = Vec::new();
        let mut is_present_pass = false;

        for att in &desc.color_attachments {
            // Check if this is the swapchain view
            if Some(att.view.0) == self.swapchain_view_id {
                attachment_views.push(self.swapchain_image_views[self.current_image_index as usize]);
                color_formats.push(self.swapchain_format);
                is_present_pass = true;
            } else if let Some(&view) = self.texture_views.get(&att.view.0) {
                attachment_views.push(view);
                let format = self.texture_view_formats.get(&att.view.0).copied()
                    .unwrap_or(vk::Format::R8G8B8A8_UNORM);
                color_formats.push(format);
            } else {
                continue;
            }

            let is_clear = matches!(&att.load_op, LoadOp::Clear(_));
            color_load_ops.push(is_clear);
            color_store_ops.push(matches!(att.store_op, StoreOp::Store));

            match &att.load_op {
                LoadOp::Clear(color) => {
                    clear_values.push(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: *color,
                        },
                    });
                }
                LoadOp::Load => {
                    clear_values.push(vk::ClearValue::default());
                }
            }
        }

        let mut depth_format = None;
        let mut depth_load_op = None;
        let mut depth_store_op = None;

        if let Some(depth_att) = &desc.depth_stencil_attachment {
            if let Some(&view) = self.texture_views.get(&depth_att.view.0) {
                attachment_views.push(view);
                let format = self.texture_view_formats.get(&depth_att.view.0).copied()
                    .unwrap_or(vk::Format::D32_SFLOAT);
                depth_format = Some(format);
                depth_load_op = Some(matches!(&depth_att.depth_load_op, LoadOp::Clear(_)));
                depth_store_op = Some(matches!(depth_att.depth_store_op, StoreOp::Store));

                clear_values.push(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: depth_att.depth_clear_value,
                        stencil: 0,
                    },
                });
            }
        }

        if attachment_views.is_empty() {
            return;
        }

        // Get or create render pass
        let render_pass = match self.get_or_create_render_pass(
            &color_formats,
            depth_format,
            &color_load_ops,
            depth_load_op,
            &color_store_ops,
            depth_store_op,
            is_present_pass,
        ) {
            Ok(rp) => rp,
            Err(_) => return,
        };

        // Get or create framebuffer
        let (width, height) = (self.swapchain_extent.width, self.swapchain_extent.height);
        let framebuffer = match self.get_or_create_framebuffer(render_pass, &attachment_views, width, height) {
            Ok(fb) => fb,
            Err(_) => return,
        };

        // Begin render pass
        let render_pass_begin = vk::RenderPassBeginInfo {
            render_pass,
            framebuffer,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.cmd_begin_render_pass(
                self.command_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );
        }

        // Execute buffered commands
        for cmd in &pending.commands {
            match cmd {
                VkRenderCommand::SetPipeline(handle) => {
                    if let Some(pipeline) = self.render_pipelines.get(&handle.0) {
                        unsafe {
                            self.device.cmd_bind_pipeline(
                                self.command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline.pipeline,
                            );
                        }
                        self.current_render_pipeline = Some(*handle);
                    }
                }
                VkRenderCommand::SetBindGroup { index, bind_group } => {
                    if let Some(&descriptor_set) = self.descriptor_sets.get(&bind_group.0) {
                        if let Some(pipeline_handle) = self.current_render_pipeline {
                            if let Some(pipeline) = self.render_pipelines.get(&pipeline_handle.0) {
                                unsafe {
                                    self.device.cmd_bind_descriptor_sets(
                                        self.command_buffer,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        pipeline.layout,
                                        *index,
                                        &[descriptor_set],
                                        &[],
                                    );
                                }
                            }
                        }
                    }
                }
                VkRenderCommand::SetVertexBuffer { slot, buffer, offset } => {
                    if let Some(vk_buffer) = self.buffers.get(&buffer.0) {
                        unsafe {
                            self.device.cmd_bind_vertex_buffers(
                                self.command_buffer,
                                *slot,
                                &[vk_buffer.buffer],
                                &[*offset],
                            );
                        }
                    }
                }
                VkRenderCommand::SetIndexBuffer { buffer, offset, format } => {
                    if let Some(vk_buffer) = self.buffers.get(&buffer.0) {
                        let index_type = match format {
                            IndexFormat::Uint16 => vk::IndexType::UINT16,
                            IndexFormat::Uint32 => vk::IndexType::UINT32,
                        };
                        unsafe {
                            self.device.cmd_bind_index_buffer(
                                self.command_buffer,
                                vk_buffer.buffer,
                                *offset,
                                index_type,
                            );
                        }
                    }
                }
                VkRenderCommand::SetViewport { x, y, width, height, min_depth, max_depth } => {
                    // Flip Y coordinate to match wgpu/WebGPU coordinate system
                    // Vulkan has Y pointing down, but wgpu expects Y pointing up
                    let viewport = vk::Viewport {
                        x: *x,
                        y: *y + *height, // Start from bottom
                        width: *width,
                        height: -*height, // Negative height to flip Y
                        min_depth: *min_depth,
                        max_depth: *max_depth,
                    };
                    unsafe {
                        self.device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
                    }
                }
                VkRenderCommand::SetScissorRect { x, y, width, height } => {
                    let scissor = vk::Rect2D {
                        offset: vk::Offset2D { x: *x as i32, y: *y as i32 },
                        extent: vk::Extent2D { width: *width, height: *height },
                    };
                    unsafe {
                        self.device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);
                    }
                }
                VkRenderCommand::Draw { vertices, instances } => {
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
                VkRenderCommand::DrawIndexed { indices, base_vertex, instances } => {
                    unsafe {
                        self.device.cmd_draw_indexed(
                            self.command_buffer,
                            indices.end - indices.start,
                            instances.end - instances.start,
                            indices.start,
                            *base_vertex,
                            instances.start,
                        );
                    }
                }
            }
        }

        // End render pass
        unsafe {
            self.device.cmd_end_render_pass(self.command_buffer);
        }

        self.current_render_pipeline = None;
    }

    fn begin_compute_pass(&mut self, label: Option<&str>) {
        self.pending_compute_pass = Some(PendingVkComputePass {
            _label: label.map(|s| s.to_string()),
            commands: Vec::new(),
        });
    }

    fn end_compute_pass(&mut self) {
        let Some(pending) = self.pending_compute_pass.take() else {
            return;
        };

        // Execute buffered compute commands
        for cmd in &pending.commands {
            match cmd {
                VkComputeCommand::SetPipeline(handle) => {
                    if let Some(pipeline) = self.compute_pipelines.get(&handle.0) {
                        unsafe {
                            self.device.cmd_bind_pipeline(
                                self.command_buffer,
                                vk::PipelineBindPoint::COMPUTE,
                                pipeline.pipeline,
                            );
                        }
                        self.current_compute_pipeline = Some(*handle);
                    }
                }
                VkComputeCommand::SetBindGroup { index, bind_group } => {
                    if let Some(&descriptor_set) = self.descriptor_sets.get(&bind_group.0) {
                        if let Some(pipeline_handle) = self.current_compute_pipeline {
                            if let Some(pipeline) = self.compute_pipelines.get(&pipeline_handle.0) {
                                unsafe {
                                    self.device.cmd_bind_descriptor_sets(
                                        self.command_buffer,
                                        vk::PipelineBindPoint::COMPUTE,
                                        pipeline.layout,
                                        *index,
                                        &[descriptor_set],
                                        &[],
                                    );
                                }
                            }
                        }
                    }
                }
                VkComputeCommand::Dispatch { x, y, z } => {
                    unsafe {
                        self.device.cmd_dispatch(self.command_buffer, *x, *y, *z);
                    }
                }
            }
        }

        self.current_compute_pipeline = None;
    }

    fn set_render_pipeline(&mut self, pipeline: RenderPipelineHandle) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(VkRenderCommand::SetPipeline(pipeline));
        }
    }

    fn set_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        if let Some(ref mut pending) = self.pending_compute_pass {
            pending.commands.push(VkComputeCommand::SetPipeline(pipeline));
        }
    }

    fn set_bind_group(&mut self, index: u32, bind_group: BindGroupHandle) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(VkRenderCommand::SetBindGroup { index, bind_group });
        } else if let Some(ref mut pending) = self.pending_compute_pass {
            pending.commands.push(VkComputeCommand::SetBindGroup { index, bind_group });
        }
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferHandle, offset: u64) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(VkRenderCommand::SetVertexBuffer { slot, buffer, offset });
        }
    }

    fn set_index_buffer(&mut self, buffer: BufferHandle, offset: u64, format: IndexFormat) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(VkRenderCommand::SetIndexBuffer { buffer, offset, format });
        }
    }

    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(VkRenderCommand::SetViewport { x, y, width, height, min_depth, max_depth });
        } else {
            // Direct command if not in render pass
            // Flip Y coordinate to match wgpu/WebGPU coordinate system
            let viewport = vk::Viewport {
                x,
                y: y + height, // Start from bottom
                width,
                height: -height, // Negative height to flip Y
                min_depth,
                max_depth
            };
            unsafe {
                self.device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
            }
        }
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(VkRenderCommand::SetScissorRect { x, y, width, height });
        } else {
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: x as i32, y: y as i32 },
                extent: vk::Extent2D { width, height },
            };
            unsafe {
                self.device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);
            }
        }
    }

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(VkRenderCommand::Draw { vertices, instances });
        } else {
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
    }

    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        if let Some(ref mut pending) = self.pending_render_pass {
            pending.commands.push(VkRenderCommand::DrawIndexed { indices, base_vertex, instances });
        } else {
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
    }

    fn dispatch_compute(&mut self, x: u32, y: u32, z: u32) {
        if let Some(ref mut pending) = self.pending_compute_pass {
            pending.commands.push(VkComputeCommand::Dispatch { x, y, z });
        } else {
            unsafe {
                self.device.cmd_dispatch(self.command_buffer, x, y, z);
            }
        }
    }

    fn transition_textures_for_sampling(&mut self, texture_views: &[TextureViewHandle]) {
        if texture_views.is_empty() {
            return;
        }

        let mut barriers = Vec::new();

        for view_handle in texture_views {
            // Get the parent texture
            let Some(&texture_handle) = self.texture_view_parents.get(&view_handle.0) else {
                continue;
            };
            let Some(texture) = self.textures.get(&texture_handle.0) else {
                continue;
            };
            let Some(&format) = self.texture_view_formats.get(&view_handle.0) else {
                continue;
            };

            let is_depth = matches!(
                format,
                vk::Format::D32_SFLOAT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D16_UNORM
            );

            let (old_layout, src_access_mask, src_stage, aspect_mask) = if is_depth {
                (
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::ImageAspectFlags::DEPTH,
                )
            } else {
                (
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::ImageAspectFlags::COLOR,
                )
            };

            barriers.push(vk::ImageMemoryBarrier {
                old_layout,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: texture.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            });
        }

        if !barriers.is_empty() {
            unsafe {
                self.device.cmd_pipeline_barrier(
                    self.command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            }
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
        // First, find all views that belong to this texture
        let views_to_remove: Vec<u64> = self.texture_view_parents
            .iter()
            .filter(|(_, &t)| t == texture)
            .map(|(&v, _)| v)
            .collect();

        // Remove framebuffers that use any of these views
        let views_set: std::collections::HashSet<_> = views_to_remove.iter().copied().collect();
        let framebuffers_to_remove: Vec<FramebufferKey> = self.framebuffers
            .keys()
            .filter(|key| {
                key.attachments.iter().any(|&view| {
                    // Check if this view corresponds to any of the texture views being destroyed
                    self.texture_views.iter().any(|(id, &v)| v == view && views_set.contains(id))
                })
            })
            .cloned()
            .collect();

        // Defer framebuffer destruction until after the frame is submitted
        for key in framebuffers_to_remove {
            if let Some(fb) = self.framebuffers.remove(&key) {
                self.framebuffers_to_destroy.push(fb);
            }
        }

        unsafe {
            // Destroy the views
            for view_id in &views_to_remove {
                if let Some(view) = self.texture_views.remove(view_id) {
                    self.device.destroy_image_view(view, None);
                }
                self.texture_view_formats.remove(view_id);
                self.texture_view_parents.remove(view_id);
            }

            // Finally destroy the texture itself
            if let Some(vk_texture) = self.textures.remove(&texture.0) {
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

            // Destroy cached framebuffers
            for (_, framebuffer) in self.framebuffers.drain() {
                self.device.destroy_framebuffer(framebuffer, None);
            }

            // Destroy cached render passes
            for (_, render_pass) in self.render_passes.drain() {
                self.device.destroy_render_pass(render_pass, None);
            }

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
