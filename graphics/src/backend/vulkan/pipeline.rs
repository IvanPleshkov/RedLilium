//! Vulkan pipeline management for shader compilation and graphics pipeline creation.

use std::ffi::CString;

use ash::vk;

use crate::error::GraphicsError;
use crate::materials::{BindingLayout, BindingType, ShaderSourceLanguage, ShaderStage};
use crate::mesh::VertexAttributeFormat;
use crate::types::TextureFormat;
use redlilium_core::mesh::{PrimitiveTopology, VertexLayout};

use super::conversion::{convert_blend_state, convert_texture_format};

/// Number of descriptor sets each individual descriptor pool can hand out
/// before a new pool is appended to the slot's chain.
const SETS_PER_POOL: u32 = 1000;

/// Manages Vulkan pipeline creation and descriptor pool resources.
pub struct PipelineManager {
    device: ash::Device,
    /// Device-resolved Vulkan format for `TextureFormat::Depth24PlusStencil8`
    /// (see `VulkanBackend::vk_texture_format`); pipeline attachment formats
    /// must match what `create_texture` actually created.
    depth24_stencil8_format: vk::Format,
    /// Pool-size template used to create each additional pool when a slot's
    /// current pools run out of descriptor sets.
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    /// Per-slot growable chain of descriptor pools — at least one per frame in
    /// flight. A slot's chain grows (an extra pool is appended) when allocation
    /// exhausts the current pools, removing the previous hard cap of
    /// [`SETS_PER_POOL`] sets per frame. The whole chain is reset together once
    /// the slot's fence signals, so resets never race in-flight descriptors.
    descriptor_pools: [parking_lot::Mutex<Vec<vk::DescriptorPool>>; super::MAX_FRAMES_IN_FLIGHT],
    /// Whether resources have been explicitly destroyed.
    destroyed: bool,
}

impl PipelineManager {
    /// Create a single descriptor pool sized by `pool_sizes`.
    fn create_pool(
        device: &ash::Device,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<vk::DescriptorPool, GraphicsError> {
        // No FREE_DESCRIPTOR_SET: sets are never freed individually, the whole
        // pool is bulk-reset once the slot's fence signals. This lets drivers
        // use a linear allocator and removes the fragmentation failure mode.
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(SETS_PER_POOL)
            .pool_sizes(pool_sizes);
        unsafe { device.create_descriptor_pool(&pool_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to create descriptor pool: {e:?}"
            ))
        })
    }

    /// Create a new pipeline manager.
    pub fn new(
        device: ash::Device,
        depth24_stencil8_format: vk::Format,
    ) -> Result<Self, GraphicsError> {
        let pool_sizes = vec![
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1000,
            },
            // DynamicUniformBuffer bindings allocate this type; conformant
            // drivers (mobile, MoltenVK) enforce per-type pool capacity, so
            // omitting it fails every dynamic-UBO allocation.
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
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

        // Create one descriptor pool per frame slot so each can be reset
        // independently; each slot starts with a single-pool chain.
        let mut slots: Vec<parking_lot::Mutex<Vec<vk::DescriptorPool>>> =
            Vec::with_capacity(super::MAX_FRAMES_IN_FLIGHT);
        for _ in 0..super::MAX_FRAMES_IN_FLIGHT {
            let pool = Self::create_pool(&device, &pool_sizes)?;
            slots.push(parking_lot::Mutex::new(vec![pool]));
        }
        let descriptor_pools: [parking_lot::Mutex<Vec<vk::DescriptorPool>>;
            super::MAX_FRAMES_IN_FLIGHT] = slots
            .try_into()
            .unwrap_or_else(|_| unreachable!("created exactly MAX_FRAMES_IN_FLIGHT pools"));

        Ok(Self {
            device,
            depth24_stencil8_format,
            pool_sizes,
            descriptor_pools,
            destroyed: false,
        })
    }

    /// Device-resolved attachment format (see field docs).
    fn vk_texture_format(&self, format: TextureFormat) -> vk::Format {
        if format == TextureFormat::Depth24PlusStencil8 {
            self.depth24_stencil8_format
        } else {
            convert_texture_format(format)
        }
    }

    /// Compile shader source to SPIR-V and create a Vulkan shader module.
    ///
    /// Returns `(shader_module, actual_entry_point)` where `actual_entry_point` is the
    /// entry point name as it appears in the compiled SPIR-V (may differ from the source
    /// function name — e.g. Slang with GLSL-compatible SPIR-V output uses `"main"`).
    ///
    /// For WGSL sources: uses naga (WGSL → naga IR → SPIR-V); preserves the original name.
    /// For Slang sources: uses the Slang compiler (Slang → SPIR-V); reads actual name from SPIR-V.
    pub fn compile_shader(
        &self,
        source: &[u8],
        stage: ShaderStage,
        entry_point: &str,
        language: ShaderSourceLanguage,
        defines: &[(String, String)],
    ) -> Result<(vk::ShaderModule, String), GraphicsError> {
        let (spv, actual_entry) = match language {
            ShaderSourceLanguage::Wgsl => {
                let spv = self.compile_wgsl_to_spirv(source, stage, entry_point)?;
                (spv, entry_point.to_string())
            }
            #[cfg(feature = "slang-shaders")]
            ShaderSourceLanguage::Slang => {
                let spv = self.compile_slang_to_spirv(source, entry_point, defines)?;
                let actual =
                    spirv_entry_point_name(&spv, stage).unwrap_or_else(|| entry_point.to_string());
                (spv, actual)
            }
            #[cfg(not(feature = "slang-shaders"))]
            ShaderSourceLanguage::Slang => {
                return Err(GraphicsError::FeatureNotSupported(
                    "Slang shaders require the 'slang-shaders' feature".into(),
                ));
            }
        };

        // Push constants have no engine-side binding path (pipeline layouts
        // declare no ranges); fail here with a clear message instead of a
        // confusing pipeline-validation error at draw time.
        if spirv_uses_push_constants(&spv) {
            return Err(GraphicsError::FeatureNotSupported(format!(
                "shader entry point '{entry_point}' declares push constants, which the \
                 engine's binding model does not support; use a uniform buffer binding instead"
            )));
        }

        // Create Vulkan shader module from SPIR-V
        let create_info = vk::ShaderModuleCreateInfo::default().code(&spv);

        let shader_module = unsafe { self.device.create_shader_module(&create_info, None) }
            .map_err(|e| {
                GraphicsError::ShaderCompilationFailed(format!(
                    "Failed to create shader module: {:?}",
                    e
                ))
            })?;

        Ok((shader_module, actual_entry))
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

    /// Compile Slang source to SPIR-V using the Slang compiler.
    #[cfg(feature = "slang-shaders")]
    fn compile_slang_to_spirv(
        &self,
        source: &[u8],
        entry_point: &str,
        defines: &[(String, String)],
    ) -> Result<Vec<u32>, GraphicsError> {
        let source_str = std::str::from_utf8(source)
            .map_err(|e| GraphicsError::ShaderCompilationFailed(format!("Invalid UTF-8: {e}")))?;

        let compiler = crate::shader::SlangCompiler::new()?;
        // Write standard library modules so `import math;` etc. resolve
        compiler.write_library_modules(&crate::shader::ShaderLibrary::standard_slang())?;
        let defines: Vec<(&str, &str)> = defines
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let spirv_bytes = compiler.compile_to_spirv(source_str, entry_point, &[], &defines)?;

        // Convert byte slice to u32 slice
        if spirv_bytes.len() % 4 != 0 {
            return Err(GraphicsError::ShaderCompilationFailed(
                "Slang SPIR-V output is not aligned to u32".into(),
            ));
        }
        let spv: Vec<u32> = spirv_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

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
                    BindingType::DynamicUniformBuffer => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    BindingType::StorageBuffer | BindingType::StorageBufferReadOnly => {
                        vk::DescriptorType::STORAGE_BUFFER
                    }
                    BindingType::Sampler | BindingType::ComparisonSampler => {
                        vk::DescriptorType::SAMPLER
                    }
                    BindingType::Texture
                    | BindingType::TextureCube
                    | BindingType::Texture2DArray
                    | BindingType::DepthTexture => vk::DescriptorType::SAMPLED_IMAGE,
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
        let mut chain = self.descriptor_pools[slot].lock();

        // Try the most-recently-added pool; if it's exhausted, append a fresh
        // pool and retry once. `grew` bounds this so a layout that can't fit
        // in any single pool errors instead of looping forever.
        // (FRAGMENTED_POOL is kept defensively: without FREE_DESCRIPTOR_SET
        // pools can't actually fragment.)
        let mut grew = false;
        loop {
            let pool = *chain
                .last()
                .expect("each slot always has at least one pool");
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);

            match unsafe { self.device.allocate_descriptor_sets(&alloc_info) } {
                Ok(sets) => return Ok(sets[0]),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY)
                | Err(vk::Result::ERROR_FRAGMENTED_POOL)
                    if !grew =>
                {
                    // Grow the chain and retry against the fresh pool.
                    let new_pool = Self::create_pool(&self.device, &self.pool_sizes)?;
                    chain.push(new_pool);
                    grew = true;
                }
                Err(e) => {
                    return Err(GraphicsError::ResourceCreationFailed(format!(
                        "Failed to allocate descriptor set: {e:?}"
                    )));
                }
            }
        }
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
                    .input_rate(match buffer.step_mode {
                        crate::mesh::VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                        crate::mesh::VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                    })
            })
            .collect();

        // Shader locations are sequential (0, 1, 2, ...) in VertexLayout attribute
        // declaration order — the same convention as the wgpu backend. This is
        // forced by Slang's WGSL output, which ignores [[vk::location(N)]] on
        // vertex inputs and numbers them sequentially, so shaders must declare
        // their inputs in the layout's attribute order on both backends.
        let attribute_descriptions: Vec<vk::VertexInputAttributeDescription> = vertex_layout
            .attributes
            .iter()
            .enumerate()
            .map(|(location, attr)| {
                vk::VertexInputAttributeDescription::default()
                    .location(location as u32)
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
            .map(|format| {
                // Blending is invalid on integer formats (matches the wgpu backend,
                // which also disables it for e.g. R32Uint picking/ID buffers).
                if let Some(state) = blend_state
                    && !format.is_integer()
                {
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
            .map(|f| self.vk_texture_format(*f))
            .collect();

        let depth_attachment_format = depth_format
            .map(|f| self.vk_texture_format(f))
            .unwrap_or(vk::Format::UNDEFINED);

        // Dynamic-rendering format matching: the encoder records a stencil
        // attachment whenever the depth format has a stencil aspect, and the
        // pipeline's declared stencil format must match it (for combined
        // formats it is the same vk::Format as the depth attachment).
        // Declaring UNDEFINED while an attachment is bound violates the
        // rendering VUIDs even with stencil testing disabled.
        let stencil_attachment_format = depth_format
            .filter(|f| f.has_stencil())
            .map(|f| self.vk_texture_format(f))
            .unwrap_or(vk::Format::UNDEFINED);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats)
            .depth_attachment_format(depth_attachment_format)
            .stencil_attachment_format(stencil_attachment_format);

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

    /// Reset every descriptor pool in a frame slot's chain, freeing all their
    /// descriptor sets. Pools are kept (not destroyed) so the chain — sized to
    /// the slot's peak usage — is reused next frame without reallocation.
    ///
    /// This should only be called after the slot's fence has signaled,
    /// ensuring no descriptor sets from these pools are in use by the GPU.
    pub fn reset_descriptor_pool(&self, slot: usize) -> Result<(), GraphicsError> {
        let chain = self.descriptor_pools[slot].lock();
        for &pool in chain.iter() {
            unsafe {
                self.device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
            }
            .map_err(|e| {
                GraphicsError::Internal(format!("Failed to reset descriptor pool: {e:?}"))
            })?;
        }
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

        // Destroy every pool in every per-slot chain.
        // SAFETY: Caller guarantees GPU is idle and device is valid
        for slot in &self.descriptor_pools {
            for &pool in slot.lock().iter() {
                unsafe {
                    self.device.destroy_descriptor_pool(pool, None);
                }
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

const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Iterate SPIR-V instructions as `(opcode, operand words)` pairs.
fn spirv_instructions(spirv: &[u32]) -> impl Iterator<Item = (u32, &[u32])> {
    let body = if spirv.len() >= 5 && spirv[0] == SPIRV_MAGIC {
        &spirv[5..]
    } else {
        &[]
    };
    let mut i = 0;
    std::iter::from_fn(move || {
        if i >= body.len() {
            return None;
        }
        let word = body[i];
        let opcode = word & 0xFFFF;
        let word_count = (word >> 16) as usize;
        if word_count == 0 || i + word_count > body.len() {
            return None;
        }
        let operands = &body[(i + 1)..(i + word_count)];
        i += word_count;
        Some((opcode, operands))
    })
}

/// Extract the `OpEntryPoint` name matching `stage` from a SPIR-V word slice.
///
/// Slang's SPIR-V output names entry points `"main"` (GLSL convention) rather than
/// preserving the original function name. This function reads the actual name so the
/// Vulkan pipeline can use the correct `pName`. Matching by execution model matters
/// for modules holding several entry points (e.g. vertex+fragment compiled together):
/// taking the first `OpEntryPoint` would return the wrong name for the second stage.
fn spirv_entry_point_name(spirv: &[u32], stage: ShaderStage) -> Option<String> {
    const OP_ENTRY_POINT: u32 = 15;
    // SPIR-V ExecutionModel values.
    let execution_model: u32 = match stage {
        ShaderStage::Vertex => 0,
        ShaderStage::Fragment => 4,
        ShaderStage::Compute => 5,
    };

    for (opcode, operands) in spirv_instructions(spirv) {
        // OpEntryPoint: | ExecutionModel | Id | Name (packed u32s) | interface vars |
        if opcode == OP_ENTRY_POINT && operands.len() >= 3 && operands[0] == execution_model {
            let mut bytes = Vec::new();
            'name: for &word in &operands[2..] {
                for byte in word.to_le_bytes() {
                    if byte == 0 {
                        break 'name;
                    }
                    bytes.push(byte);
                }
            }
            return String::from_utf8(bytes).ok();
        }
    }
    None
}

/// Whether the SPIR-V module declares a push-constant block.
///
/// The engine's binding model has no push-constant path (pipeline layouts
/// declare no ranges), so such a module would fail pipeline validation with a
/// confusing driver error. Detecting it up front turns that into a clear
/// compile-time diagnostic.
fn spirv_uses_push_constants(spirv: &[u32]) -> bool {
    const OP_VARIABLE: u32 = 59;
    const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;

    // OpVariable: | Result Type | Result Id | Storage Class | [Initializer] |
    spirv_instructions(spirv).any(|(opcode, operands)| {
        opcode == OP_VARIABLE && operands.len() >= 3 && operands[2] == STORAGE_CLASS_PUSH_CONSTANT
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_name(name: &str) -> Vec<u32> {
        // Pack a null-terminated string into little-endian words (SPIR-V
        // literal string encoding). Always emits the terminator, padding to a
        // word boundary.
        let mut bytes: Vec<u8> = name.as_bytes().to_vec();
        bytes.push(0);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn instruction(opcode: u32, operands: &[u32]) -> Vec<u32> {
        let mut words = vec![(((operands.len() + 1) as u32) << 16) | opcode];
        words.extend_from_slice(operands);
        words
    }

    fn module(instructions: &[Vec<u32>]) -> Vec<u32> {
        let mut spv = vec![SPIRV_MAGIC, 0x0001_0300, 0, 100, 0];
        for inst in instructions {
            spv.extend_from_slice(inst);
        }
        spv
    }

    fn entry_point(execution_model: u32, name: &str) -> Vec<u32> {
        let mut operands = vec![execution_model, 1];
        operands.extend(pack_name(name));
        instruction(15, &operands) // OpEntryPoint
    }

    #[test]
    fn entry_point_name_selected_by_stage() {
        // Vertex + fragment compiled into one module: each stage must get its
        // own name, not the first OpEntryPoint's.
        let spv = module(&[entry_point(0, "vsMain"), entry_point(4, "psMain")]);
        assert_eq!(
            spirv_entry_point_name(&spv, ShaderStage::Vertex).as_deref(),
            Some("vsMain")
        );
        assert_eq!(
            spirv_entry_point_name(&spv, ShaderStage::Fragment).as_deref(),
            Some("psMain")
        );
        assert_eq!(spirv_entry_point_name(&spv, ShaderStage::Compute), None);
    }

    #[test]
    fn entry_point_name_rejects_bad_magic() {
        let mut spv = module(&[entry_point(0, "main")]);
        spv[0] = 0xDEAD_BEEF;
        assert_eq!(spirv_entry_point_name(&spv, ShaderStage::Vertex), None);
    }

    #[test]
    fn push_constant_detection() {
        // OpVariable: | Result Type | Result Id | Storage Class |
        let uniform_var = instruction(59, &[2, 3, 2]); // StorageClass Uniform
        let push_var = instruction(59, &[2, 4, 9]); // StorageClass PushConstant

        let without = module(&[entry_point(0, "main"), uniform_var.clone()]);
        assert!(!spirv_uses_push_constants(&without));

        let with = module(&[entry_point(0, "main"), uniform_var, push_var]);
        assert!(spirv_uses_push_constants(&with));
    }
}
