//! Vulkan pipeline management for shader compilation and graphics pipeline creation.

use std::collections::HashMap;
use std::ffi::CString;

use ash::vk;
use parking_lot::Mutex;

use crate::error::GraphicsError;
use crate::materials::{BindingLayout, BindingType, ShaderStage};
use crate::mesh::{Mesh, VertexAttributeFormat};
use crate::types::TextureFormat;

use super::conversion::{convert_blend_state, convert_texture_format};

/// Manages Vulkan pipelines and related resources.
pub struct PipelineManager {
    device: ash::Device,
    /// Cache of compiled shader modules.
    shader_cache: Mutex<HashMap<u64, vk::ShaderModule>>,
    /// Cache of descriptor set layouts.
    descriptor_set_layout_cache: Mutex<HashMap<u64, vk::DescriptorSetLayout>>,
    /// Cache of pipeline layouts.
    pipeline_layout_cache: Mutex<HashMap<u64, vk::PipelineLayout>>,
    /// Cache of graphics pipelines.
    pipeline_cache: Mutex<HashMap<u64, vk::Pipeline>>,
    /// Descriptor pool for allocating descriptor sets.
    descriptor_pool: vk::DescriptorPool,
    /// Whether resources have been explicitly destroyed.
    destroyed: bool,
}

impl PipelineManager {
    /// Create a new pipeline manager.
    pub fn new(device: ash::Device) -> Result<Self, GraphicsError> {
        // Create a descriptor pool with reasonable defaults
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
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
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 100,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 100,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1000,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(1000)
            .pool_sizes(&pool_sizes);

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&pool_info, None) }.map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to create descriptor pool: {:?}",
                    e
                ))
            })?;

        Ok(Self {
            device,
            shader_cache: Mutex::new(HashMap::new()),
            descriptor_set_layout_cache: Mutex::new(HashMap::new()),
            pipeline_layout_cache: Mutex::new(HashMap::new()),
            pipeline_cache: Mutex::new(HashMap::new()),
            descriptor_pool,
            destroyed: false,
        })
    }

    /// Compile WGSL shader to SPIR-V and create a shader module.
    pub fn compile_shader(
        &self,
        wgsl_source: &[u8],
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<vk::ShaderModule, GraphicsError> {
        // Parse WGSL
        let source = std::str::from_utf8(wgsl_source)
            .map_err(|e| GraphicsError::ShaderCompilationFailed(format!("Invalid UTF-8: {e}")))?;

        let module = naga::front::wgsl::parse_str(source).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("WGSL parse error: {e}"))
        })?;

        // Validate the module
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Validation error: {e}"))
        })?;

        // Convert to SPIR-V
        let naga_stage = match stage {
            ShaderStage::Vertex => naga::ShaderStage::Vertex,
            ShaderStage::Fragment => naga::ShaderStage::Fragment,
            ShaderStage::Compute => naga::ShaderStage::Compute,
        };

        // Find the entry point (verify it exists)
        let _entry_point_index = module
            .entry_points
            .iter()
            .position(|ep| ep.name == entry_point && ep.stage == naga_stage)
            .ok_or_else(|| {
                GraphicsError::ShaderCompilationFailed(format!(
                    "Entry point '{}' not found for stage {:?}",
                    entry_point, stage
                ))
            })?;

        let options = naga::back::spv::Options {
            lang_version: (1, 3),
            flags: naga::back::spv::WriterFlags::empty(),
            capabilities: None,
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
            binding_map: Default::default(),
            debug_info: None,
            zero_initialize_workgroup_memory:
                naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
        };

        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: naga_stage,
            entry_point: entry_point.to_string(),
        };

        let spv = naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options))
            .map_err(|e| {
                GraphicsError::ShaderCompilationFailed(format!("SPIR-V generation error: {e}"))
            })?;

        // Create Vulkan shader module
        let create_info = vk::ShaderModuleCreateInfo::default().code(&spv);

        let shader_module = unsafe { self.device.create_shader_module(&create_info, None) }
            .map_err(|e| {
                GraphicsError::ShaderCompilationFailed(format!(
                    "Failed to create shader module: {:?}",
                    e
                ))
            })?;

        Ok(shader_module)
    }

    /// Create a descriptor set layout from a binding layout.
    pub fn create_descriptor_set_layout(
        &self,
        layout: &BindingLayout,
    ) -> Result<vk::DescriptorSetLayout, GraphicsError> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = layout
            .entries
            .iter()
            .map(|entry| {
                let descriptor_type = match entry.binding_type {
                    BindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                    BindingType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
                    BindingType::Sampler => vk::DescriptorType::SAMPLER,
                    BindingType::Texture => vk::DescriptorType::SAMPLED_IMAGE,
                    BindingType::TextureCube => vk::DescriptorType::SAMPLED_IMAGE,
                    BindingType::CombinedTextureSampler => {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    }
                };

                let stage_flags = convert_shader_stage_flags(entry.visibility);

                vk::DescriptorSetLayoutBinding::default()
                    .binding(entry.binding)
                    .descriptor_type(descriptor_type)
                    .descriptor_count(1)
                    .stage_flags(stage_flags)
            })
            .collect();

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let layout = unsafe { self.device.create_descriptor_set_layout(&create_info, None) }
            .map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to create descriptor set layout: {:?}",
                    e
                ))
            })?;

        Ok(layout)
    }

    /// Create a pipeline layout from descriptor set layouts.
    pub fn create_pipeline_layout(
        &self,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<vk::PipelineLayout, GraphicsError> {
        let create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(descriptor_set_layouts);

        let layout =
            unsafe { self.device.create_pipeline_layout(&create_info, None) }.map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to create pipeline layout: {:?}",
                    e
                ))
            })?;

        Ok(layout)
    }

    /// Allocate a descriptor set.
    pub fn allocate_descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, GraphicsError> {
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);

        let sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to allocate descriptor set: {:?}",
                e
            ))
        })?;

        Ok(sets[0])
    }

    /// Create a graphics pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn create_graphics_pipeline(
        &self,
        vertex_module: vk::ShaderModule,
        fragment_module: Option<vk::ShaderModule>,
        vertex_entry: &str,
        fragment_entry: &str,
        mesh: &Mesh,
        pipeline_layout: vk::PipelineLayout,
        color_formats: &[TextureFormat],
        depth_format: Option<TextureFormat>,
        blend_state: Option<&crate::materials::BlendState>,
        _dynamic_rendering: &ash::khr::dynamic_rendering::Device,
    ) -> Result<vk::Pipeline, GraphicsError> {
        let vertex_entry_c = CString::new(vertex_entry).map_err(|e| {
            GraphicsError::InvalidParameter(format!(
                "Invalid vertex entry point name (contains null byte): {}",
                e
            ))
        })?;
        let fragment_entry_c = CString::new(fragment_entry).map_err(|e| {
            GraphicsError::InvalidParameter(format!(
                "Invalid fragment entry point name (contains null byte): {}",
                e
            ))
        })?;

        let mut shader_stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_module)
                .name(&vertex_entry_c),
        ];

        if let Some(frag_module) = fragment_module {
            shader_stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_module)
                    .name(&fragment_entry_c),
            );
        }

        // Build vertex input state from mesh layout
        let layout = mesh.layout();

        let binding_descriptions: Vec<vk::VertexInputBindingDescription> = layout
            .buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| {
                vk::VertexInputBindingDescription::default()
                    .binding(i as u32)
                    .stride(buffer.stride)
                    .input_rate(vk::VertexInputRate::VERTEX)
            })
            .collect();

        let attribute_descriptions: Vec<vk::VertexInputAttributeDescription> = layout
            .attributes
            .iter()
            .map(|attr| {
                vk::VertexInputAttributeDescription::default()
                    .location(attr.semantic.index())
                    .binding(attr.buffer_index)
                    .format(convert_vertex_format(attr.format))
                    .offset(attr.offset)
            })
            .collect();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Dynamic viewport and scissor
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Use CLOCKWISE front face to compensate for the viewport Y-flip.
        // When using negative viewport height to match wgpu/OpenGL coordinates,
        // the triangle winding order is effectively reversed, so we need to
        // flip the front face definition to match.
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_format.is_some())
            .depth_write_enable(depth_format.is_some())
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = color_formats
            .iter()
            .map(|_| {
                if let Some(state) = blend_state {
                    convert_blend_state(state)
                } else {
                    // Default: no blending (replace)
                    vk::PipelineColorBlendAttachmentState::default()
                        .color_write_mask(vk::ColorComponentFlags::RGBA)
                        .blend_enable(false)
                }
            })
            .collect();

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Set up dynamic rendering formats
        let color_attachment_formats: Vec<vk::Format> = color_formats
            .iter()
            .map(|f| convert_texture_format(*f))
            .collect();

        let depth_attachment_format = depth_format
            .map(convert_texture_format)
            .unwrap_or(vk::Format::UNDEFINED);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(depth_attachment_format);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .depth_stencil_state(&depth_stencil_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut rendering_info);

        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, e)| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to create graphics pipeline: {:?}",
                e
            ))
        })?;

        Ok(pipelines[0])
    }

    /// Get the descriptor pool.
    #[allow(dead_code)]
    pub fn descriptor_pool(&self) -> vk::DescriptorPool {
        self.descriptor_pool
    }

    /// Reset the descriptor pool, freeing all allocated descriptor sets.
    ///
    /// This should only be called when no descriptor sets from this pool
    /// are in use by the GPU (i.e., after waiting for the GPU to idle).
    pub fn reset_descriptor_pool(&self) -> Result<(), GraphicsError> {
        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
        }
        .map_err(|e| {
            GraphicsError::Internal(format!("Failed to reset descriptor pool: {:?}", e))
        })?;
        Ok(())
    }
}

impl PipelineManager {
    /// Explicitly destroy all resources.
    ///
    /// This must be called before the Vulkan device is destroyed.
    /// After calling this method, the PipelineManager should not be used.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The GPU is idle (no pending operations using these resources)
    /// - This is called before the Vulkan device is destroyed
    pub unsafe fn destroy(&mut self) {
        if self.destroyed {
            return;
        }

        // Destroy cached pipelines
        for (_, pipeline) in self.pipeline_cache.lock().drain() {
            // SAFETY: Caller guarantees GPU is idle and device is valid
            unsafe { self.device.destroy_pipeline(pipeline, None) };
        }

        // Destroy cached pipeline layouts
        for (_, layout) in self.pipeline_layout_cache.lock().drain() {
            // SAFETY: Caller guarantees GPU is idle and device is valid
            unsafe { self.device.destroy_pipeline_layout(layout, None) };
        }

        // Destroy cached descriptor set layouts
        for (_, layout) in self.descriptor_set_layout_cache.lock().drain() {
            // SAFETY: Caller guarantees GPU is idle and device is valid
            unsafe { self.device.destroy_descriptor_set_layout(layout, None) };
        }

        // Destroy cached shader modules
        for (_, module) in self.shader_cache.lock().drain() {
            // SAFETY: Caller guarantees GPU is idle and device is valid
            unsafe { self.device.destroy_shader_module(module, None) };
        }

        // Destroy descriptor pool
        // SAFETY: Caller guarantees GPU is idle and device is valid
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None)
        };

        self.destroyed = true;
    }
}

impl Drop for PipelineManager {
    fn drop(&mut self) {
        if self.destroyed {
            return;
        }

        // If destroy() was not called explicitly, we have a problem:
        // the device may already be destroyed. Log a warning but don't
        // attempt to use the device as it may cause undefined behavior.
        log::warn!(
            "PipelineManager::drop() called without explicit destroy(). \
             Resources may have leaked. Always call destroy() before dropping the device."
        );
    }
}

/// Convert our shader stage flags to Vulkan stage flags.
fn convert_shader_stage_flags(flags: crate::materials::ShaderStageFlags) -> vk::ShaderStageFlags {
    let mut result = vk::ShaderStageFlags::empty();
    if flags.contains(crate::materials::ShaderStageFlags::VERTEX) {
        result |= vk::ShaderStageFlags::VERTEX;
    }
    if flags.contains(crate::materials::ShaderStageFlags::FRAGMENT) {
        result |= vk::ShaderStageFlags::FRAGMENT;
    }
    if flags.contains(crate::materials::ShaderStageFlags::COMPUTE) {
        result |= vk::ShaderStageFlags::COMPUTE;
    }
    result
}

/// Convert vertex attribute format to Vulkan format.
fn convert_vertex_format(format: VertexAttributeFormat) -> vk::Format {
    match format {
        VertexAttributeFormat::Float => vk::Format::R32_SFLOAT,
        VertexAttributeFormat::Float2 => vk::Format::R32G32_SFLOAT,
        VertexAttributeFormat::Float3 => vk::Format::R32G32B32_SFLOAT,
        VertexAttributeFormat::Float4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexAttributeFormat::Int => vk::Format::R32_SINT,
        VertexAttributeFormat::Int2 => vk::Format::R32G32_SINT,
        VertexAttributeFormat::Int3 => vk::Format::R32G32B32_SINT,
        VertexAttributeFormat::Int4 => vk::Format::R32G32B32A32_SINT,
        VertexAttributeFormat::Uint => vk::Format::R32_UINT,
        VertexAttributeFormat::Uint2 => vk::Format::R32G32_UINT,
        VertexAttributeFormat::Uint3 => vk::Format::R32G32B32_UINT,
        VertexAttributeFormat::Uint4 => vk::Format::R32G32B32A32_UINT,
        VertexAttributeFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
        VertexAttributeFormat::Snorm8x4 => vk::Format::R8G8B8A8_SNORM,
    }
}
