//! Vulkan pipeline management for shader compilation and graphics pipeline creation.

use std::ffi::CString;

use ash::vk;

use crate::error::GraphicsError;
use crate::materials::{BindingLayout, BindingType, ShaderSourceLanguage, ShaderStage};
use crate::mesh::VertexAttributeFormat;
use crate::types::TextureFormat;
use redlilium_core::mesh::{PrimitiveTopology, VertexLayout};

use super::conversion::{convert_blend_state, convert_texture_format};

/// Manages Vulkan pipeline creation and descriptor pool resources.
pub struct PipelineManager {
    device: ash::Device,
    /// Per-slot descriptor pools — one per frame in flight.
    /// Each slot's pool is only reset after its fence signals,
    /// preventing resets while another slot's descriptors are in use.
    descriptor_pools: [vk::DescriptorPool; super::MAX_FRAMES_IN_FLIGHT],
    /// Whether resources have been explicitly destroyed.
    destroyed: bool,
}

impl PipelineManager {
    /// Create a new pipeline manager.
    pub fn new(device: ash::Device) -> Result<Self, GraphicsError> {
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

        // Create one descriptor pool per frame slot so each can be reset independently
        let mut descriptor_pools = [vk::DescriptorPool::null(); super::MAX_FRAMES_IN_FLIGHT];
        for pool in &mut descriptor_pools {
            *pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.map_err(|e| {
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to create descriptor pool: {:?}",
                    e
                ))
            })?;
        }

        Ok(Self {
            device,
            descriptor_pools,
            destroyed: false,
        })
    }

    /// Compile shader source to SPIR-V and create a Vulkan shader module.
    ///
    /// For GLSL sources: uses shaderc (glslang) for direct GLSL → SPIR-V.
    /// For WGSL sources: uses naga (WGSL → naga IR → SPIR-V) as fallback.
    pub fn compile_shader(
        &self,
        source: &[u8],
        stage: ShaderStage,
        entry_point: &str,
        language: ShaderSourceLanguage,
        defines: &[(String, String)],
    ) -> Result<vk::ShaderModule, GraphicsError> {
        let spv = match language {
            ShaderSourceLanguage::Glsl => {
                self.compile_glsl_to_spirv(source, stage, entry_point, defines)?
            }
            ShaderSourceLanguage::Wgsl => self.compile_wgsl_to_spirv(source, stage, entry_point)?,
        };

        // Create Vulkan shader module from SPIR-V
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

    /// Compile GLSL to SPIR-V using shaderc (glslang wrapper).
    fn compile_glsl_to_spirv(
        &self,
        source: &[u8],
        stage: ShaderStage,
        entry_point: &str,
        defines: &[(String, String)],
    ) -> Result<Vec<u32>, GraphicsError> {
        let source_str = std::str::from_utf8(source)
            .map_err(|e| GraphicsError::ShaderCompilationFailed(format!("Invalid UTF-8: {e}")))?;

        let compiler = shaderc::Compiler::new().ok_or_else(|| {
            GraphicsError::ShaderCompilationFailed("Failed to create shaderc compiler".into())
        })?;

        let mut options = shaderc::CompileOptions::new().ok_or_else(|| {
            GraphicsError::ShaderCompilationFailed(
                "Failed to create shaderc compile options".into(),
            )
        })?;

        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_3 as u32,
        );
        options.set_source_language(shaderc::SourceLanguage::GLSL);
        options.set_target_spirv(shaderc::SpirvVersion::V1_3);

        // Add all defines
        for (name, value) in defines {
            if value.is_empty() {
                options.add_macro_definition(name, None);
            } else {
                options.add_macro_definition(name, Some(value));
            }
        }

        let shaderc_stage = match stage {
            ShaderStage::Vertex => shaderc::ShaderKind::Vertex,
            ShaderStage::Fragment => shaderc::ShaderKind::Fragment,
            ShaderStage::Compute => shaderc::ShaderKind::Compute,
        };

        let result = compiler
            .compile_into_spirv(
                source_str,
                shaderc_stage,
                "shader.glsl",
                entry_point,
                Some(&options),
            )
            .map_err(|e| {
                GraphicsError::ShaderCompilationFailed(format!(
                    "shaderc GLSL compilation error: {e}"
                ))
            })?;

        if result.get_num_warnings() > 0 {
            log::warn!("Shader warnings: {}", result.get_warning_messages());
        }

        Ok(result.as_binary().to_vec())
    }

    /// Compile WGSL to SPIR-V using naga (fallback for WGSL-authored shaders).
    fn compile_wgsl_to_spirv(
        &self,
        wgsl_source: &[u8],
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<Vec<u32>, GraphicsError> {
        let source = std::str::from_utf8(wgsl_source)
            .map_err(|e| GraphicsError::ShaderCompilationFailed(format!("Invalid UTF-8: {e}")))?;

        let module = naga::front::wgsl::parse_str(source).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("WGSL parse error: {e}"))
        })?;

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator.validate(&module).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Validation error: {e}"))
        })?;

        let naga_stage = match stage {
            ShaderStage::Vertex => naga::ShaderStage::Vertex,
            ShaderStage::Fragment => naga::ShaderStage::Fragment,
            ShaderStage::Compute => naga::ShaderStage::Compute,
        };

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

        Ok(spv)
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
                    BindingType::Texture2DArray => vk::DescriptorType::SAMPLED_IMAGE,
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

    /// Allocate a descriptor set from the pool for the given frame slot.
    pub fn allocate_descriptor_set(
        &self,
        slot: usize,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, GraphicsError> {
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pools[slot])
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
        vertex_layout: &VertexLayout,
        topology: PrimitiveTopology,
        pipeline_layout: vk::PipelineLayout,
        color_formats: &[TextureFormat],
        depth_format: Option<TextureFormat>,
        blend_state: Option<&crate::materials::BlendState>,
        polygon_mode: crate::materials::PolygonMode,
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

        // Build vertex input state from material's vertex layout
        let binding_descriptions: Vec<vk::VertexInputBindingDescription> = vertex_layout
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

        let attribute_descriptions: Vec<vk::VertexInputAttributeDescription> = vertex_layout
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

        let vk_topology = match topology {
            PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
            PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
            PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk_topology)
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
            .polygon_mode(match polygon_mode {
                crate::materials::PolygonMode::Fill => vk::PolygonMode::FILL,
                crate::materials::PolygonMode::Line => vk::PolygonMode::LINE,
            })
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

    /// Create a compute pipeline.
    pub fn create_compute_pipeline(
        &self,
        compute_module: vk::ShaderModule,
        compute_entry: &str,
        pipeline_layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, GraphicsError> {
        let compute_entry_c = CString::new(compute_entry).map_err(|e| {
            GraphicsError::InvalidParameter(format!(
                "Invalid compute entry point name (contains null byte): {}",
                e
            ))
        })?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(compute_module)
            .name(&compute_entry_c);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, e)| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to create compute pipeline: {:?}",
                e
            ))
        })?;

        Ok(pipelines[0])
    }

    /// Get the descriptor pool for a given frame slot.
    #[allow(dead_code)]
    pub fn descriptor_pool(&self, slot: usize) -> vk::DescriptorPool {
        self.descriptor_pools[slot]
    }

    /// Reset the descriptor pool for a given frame slot, freeing all its descriptor sets.
    ///
    /// This should only be called after the slot's fence has signaled,
    /// ensuring no descriptor sets from this pool are in use by the GPU.
    pub fn reset_descriptor_pool(&self, slot: usize) -> Result<(), GraphicsError> {
        unsafe {
            self.device.reset_descriptor_pool(
                self.descriptor_pools[slot],
                vk::DescriptorPoolResetFlags::empty(),
            )
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

        // Pipelines are owned by Materials and destroyed when their last Arc is dropped.
        // We only need to destroy the descriptor pools here.

        // Destroy all per-slot descriptor pools
        // SAFETY: Caller guarantees GPU is idle and device is valid
        for pool in &self.descriptor_pools {
            unsafe {
                self.device.destroy_descriptor_pool(*pool, None);
            }
        }

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
