//! Resource creation for the wgpu backend.

use std::sync::Mutex;

use crate::error::GraphicsError;
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

use super::super::{GpuBuffer, GpuFence, GpuSampler, GpuTexture};
use super::WgpuBackend;
use super::conversion::{
    convert_address_mode, convert_buffer_usage, convert_compare_function, convert_filter_mode,
    convert_mipmap_filter_mode, convert_texture_format, convert_texture_usage,
};

impl WgpuBackend {
    /// Get (or create + cache) the `wgpu::BindGroupLayout` for a binding layout,
    /// deduped by content. Shared between pipeline creation and binding-group
    /// creation so a group's layout is the same object the pipeline uses.
    pub(super) fn get_or_create_bind_group_layout(
        &self,
        layout: &crate::materials::BindingLayout,
    ) -> wgpu::BindGroupLayout {
        let key = super::bind_group_layout_key(layout);
        if let Some(cached) = self.bind_group_layout_cache.lock().get(&key) {
            return cached.clone();
        }
        let entries = super::conversion::binding_layout_entries(layout);
        let created = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: layout.label.as_deref(),
                entries: &entries,
            });
        // Another thread may have raced; keep one object per key.
        self.bind_group_layout_cache
            .lock()
            .entry(key)
            .or_insert(created)
            .clone()
    }

    /// Create a binding group: build the `wgpu::BindGroup` **once**, against the
    /// deduped bind group layout for `layout`. Encoding a draw then only calls
    /// `set_bind_group` with this cached group.
    pub fn create_binding_group(
        &self,
        layout: &crate::materials::BindingLayout,
        descriptor: &crate::materials::BindingGroupDescriptor,
    ) -> Result<super::super::GpuBindingGroup, GraphicsError> {
        let bg_layout = self.get_or_create_bind_group_layout(layout);
        let entries = build_wgpu_bind_group_entries(&descriptor.entries)?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: descriptor.label.as_deref(),
            layout: &bg_layout,
            entries: &entries,
        });
        Ok(super::super::GpuBindingGroup::Wgpu(bind_group))
    }

    /// Create a buffer resource.
    pub fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<GpuBuffer, GraphicsError> {
        let usage = convert_buffer_usage(descriptor.usage);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: descriptor.label.as_deref(),
            size: descriptor.size,
            usage,
            mapped_at_creation: false,
        });

        Ok(GpuBuffer::Wgpu(buffer))
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        use crate::types::TextureDimension;

        // wgpu has no BGRA-ordered 10-bit format; silently mapping to
        // Rgb10a2Unorm would swap the red/blue channels relative to the
        // Vulkan backend (A2R10G10B10). Fail loudly instead.
        if descriptor.format == crate::types::TextureFormat::Bgra10a2Unorm {
            return Err(GraphicsError::InvalidParameter(
                "Bgra10a2Unorm is not supported by the wgpu backend (no BGRA-ordered \
                 10-bit format in WebGPU); use Rgba10a2Unorm instead"
                    .into(),
            ));
        }

        let format = convert_texture_format(descriptor.format);

        // Feature-gated formats (BC/ETC2/ASTC compression, etc.) are requested
        // opportunistically at device creation; if this adapter lacks the
        // feature, creating the texture would raise an uncaptured validation
        // error. Fail with a clear error at the actual cause instead.
        let required = format.required_features();
        if !self.device.features().contains(required) {
            return Err(GraphicsError::FeatureNotSupported(format!(
                "texture format {:?} requires wgpu feature(s) {:?}, which this adapter \
                 does not support",
                descriptor.format,
                required.difference(self.device.features())
            )));
        }

        let usage = convert_texture_usage(descriptor.usage);

        // WebGPU has no arrayed 1D textures at all; the old mapping created a
        // multi-layer D1 texture with a D1 (non-array) view, which fails
        // bind-group validation on first use. Reject it up front.
        if descriptor.dimension == TextureDimension::D1Array {
            return Err(GraphicsError::FeatureNotSupported(
                "D1Array textures are not supported by the wgpu backend (WebGPU has no \
                 arrayed 1D textures); use a D2Array with height 1 instead"
                    .into(),
            ));
        }

        // Convert our texture dimension to wgpu's
        let (wgpu_dimension, depth_or_array_layers) = match descriptor.dimension {
            TextureDimension::D1 => (wgpu::TextureDimension::D1, descriptor.size.depth),
            TextureDimension::D1Array => (wgpu::TextureDimension::D1, descriptor.size.depth),
            TextureDimension::D2 => (wgpu::TextureDimension::D2, descriptor.size.depth),
            TextureDimension::D2Array => (wgpu::TextureDimension::D2, descriptor.size.depth),
            TextureDimension::D3 => (wgpu::TextureDimension::D3, descriptor.size.depth),
            TextureDimension::Cube => (wgpu::TextureDimension::D2, 6),
            TextureDimension::CubeArray => (wgpu::TextureDimension::D2, descriptor.size.depth * 6),
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: descriptor.label.as_deref(),
            size: wgpu::Extent3d {
                width: descriptor.size.width,
                height: descriptor.size.height,
                depth_or_array_layers,
            },
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: wgpu_dimension,
            format,
            usage,
            view_formats: &[],
        });

        // Create the appropriate view based on dimension
        let view_dimension = match descriptor.dimension {
            TextureDimension::D1 => wgpu::TextureViewDimension::D1,
            TextureDimension::D1Array => wgpu::TextureViewDimension::D1,
            TextureDimension::D2 => wgpu::TextureViewDimension::D2,
            TextureDimension::D2Array => wgpu::TextureViewDimension::D2Array,
            TextureDimension::D3 => wgpu::TextureViewDimension::D3,
            TextureDimension::Cube => wgpu::TextureViewDimension::Cube,
            TextureDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
        };

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(view_dimension),
            ..Default::default()
        });

        Ok(GpuTexture::Wgpu { texture, view })
    }

    /// Create a sampler resource.
    pub fn create_sampler(
        &self,
        descriptor: &SamplerDescriptor,
    ) -> Result<GpuSampler, GraphicsError> {
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: descriptor.label.as_deref(),
            address_mode_u: convert_address_mode(descriptor.address_mode_u),
            address_mode_v: convert_address_mode(descriptor.address_mode_v),
            address_mode_w: convert_address_mode(descriptor.address_mode_w),
            mag_filter: convert_filter_mode(descriptor.mag_filter),
            min_filter: convert_filter_mode(descriptor.min_filter),
            mipmap_filter: convert_mipmap_filter_mode(descriptor.mipmap_filter),
            lod_min_clamp: descriptor.lod_min_clamp,
            lod_max_clamp: descriptor.lod_max_clamp,
            compare: descriptor.compare.map(convert_compare_function),
            anisotropy_clamp: descriptor.anisotropy_clamp,
            border_color: None,
        });

        Ok(GpuSampler::Wgpu(sampler))
    }

    /// Create a GPU pipeline from a material descriptor.
    pub fn create_pipeline(
        &self,
        descriptor: &crate::materials::MaterialDescriptor,
    ) -> Result<super::super::GpuPipeline, GraphicsError> {
        use super::conversion::{
            convert_blend_state, convert_step_mode, convert_topology, convert_vertex_format,
        };
        use crate::materials::ShaderStage;

        let is_compute = descriptor
            .shaders
            .iter()
            .any(|s| s.stage == ShaderStage::Compute);

        if is_compute {
            return self.create_compute_pipeline_from_descriptor(descriptor);
        }

        // Compile shader modules
        let mut vertex_module = None;
        let mut fragment_module = None;
        let mut vertex_entry: &str = "vs_main";
        let mut fragment_entry: &str = "fs_main";

        for shader in &descriptor.shaders {
            let wgsl_source = self.compile_to_wgsl(shader)?;
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: descriptor.label.as_deref(),
                    source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
                });
            match shader.stage {
                ShaderStage::Vertex => {
                    vertex_module = Some(module);
                    vertex_entry = &shader.entry_point;
                }
                ShaderStage::Fragment => {
                    fragment_module = Some(module);
                    fragment_entry = &shader.entry_point;
                }
                ShaderStage::Compute => {}
            }
        }

        let Some(vertex_module) = vertex_module else {
            return Err(GraphicsError::ShaderCompilationFailed(
                "No vertex shader provided".into(),
            ));
        };

        let layout = &descriptor.vertex_layout;

        // Vertex attributes per buffer.
        // Use sequential shader_location values (0, 1, 2, ...) to match Slang's WGSL output,
        // which ignores [[vk::location(N)]] annotations on vertex inputs and assigns sequential
        // @location values in struct declaration order. The Vulkan backend follows the same
        // convention (ADR-019 in docs/DECISIONS.md).
        let buffer_count = layout.buffers.len();
        let mut vertex_attrs: Vec<Vec<wgpu::VertexAttribute>> = vec![Vec::new(); buffer_count];
        for (location, attr) in layout.attributes.iter().enumerate() {
            let idx = attr.buffer_index as usize;
            if idx < buffer_count {
                vertex_attrs[idx].push(wgpu::VertexAttribute {
                    format: convert_vertex_format(attr.format),
                    offset: attr.offset as u64,
                    shader_location: location as u32,
                });
            }
        }

        // Bind group layouts, pulled from the content-keyed dedup cache so a
        // binding group created against the same `BindingLayout` reuses the
        // exact same object (wgpu requires the group's layout be compatible).
        let mut bind_group_layouts = Vec::new();
        for bg_layout in &descriptor.binding_layouts {
            bind_group_layouts.push(self.get_or_create_bind_group_layout(bg_layout));
        }

        // Pipeline layout
        let pipeline_layout = {
            let refs: Vec<&wgpu::BindGroupLayout> = bind_group_layouts.iter().collect();
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Material Pipeline Layout"),
                    bind_group_layouts: &refs,
                    immediate_size: 0,
                })
        };

        // Color targets
        let wgpu_blend_state = descriptor
            .blend_state
            .as_ref()
            .map(convert_blend_state)
            .unwrap_or(wgpu::BlendState::REPLACE);

        let color_targets: Vec<Option<wgpu::ColorTargetState>> = descriptor
            .color_formats
            .iter()
            .map(|format| {
                // Integer formats (e.g. R32Uint) are not blendable — blend must be None.
                let blend = if format.is_integer() {
                    None
                } else {
                    Some(wgpu_blend_state)
                };
                Some(wgpu::ColorTargetState {
                    format: convert_texture_format(*format),
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect();

        let depth_format = descriptor.depth_format.map(convert_texture_format);

        // Build vertex buffer layouts
        let vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout> = layout
            .buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::VertexBufferLayout {
                array_stride: buffer.stride as u64,
                step_mode: convert_step_mode(buffer.step_mode),
                attributes: &vertex_attrs[i],
            })
            .collect();

        // Create render pipeline
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: descriptor.label.as_deref(),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex_module,
                    entry_point: Some(vertex_entry),
                    buffers: &vertex_buffer_layouts,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: fragment_module.as_ref().map(|module| wgpu::FragmentState {
                    module,
                    entry_point: Some(fragment_entry),
                    targets: &color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: convert_topology(descriptor.topology),
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: match descriptor.polygon_mode {
                        crate::materials::PolygonMode::Fill => wgpu::PolygonMode::Fill,
                        crate::materials::PolygonMode::Line => wgpu::PolygonMode::Line,
                    },
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        Ok(super::super::GpuPipeline::WgpuGraphics {
            pipeline,
            bind_group_layouts,
        })
    }

    fn create_compute_pipeline_from_descriptor(
        &self,
        descriptor: &crate::materials::MaterialDescriptor,
    ) -> Result<super::super::GpuPipeline, GraphicsError> {
        use crate::materials::ShaderStage;

        let mut compute_module = None;
        let mut compute_entry: &str = "main";

        for shader in &descriptor.shaders {
            if shader.stage == ShaderStage::Compute {
                let wgsl_source = self.compile_to_wgsl(shader)?;
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: descriptor.label.as_deref(),
                        source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
                    });
                compute_module = Some(module);
                compute_entry = &shader.entry_point;
            }
        }

        let Some(compute_module) = compute_module else {
            return Err(GraphicsError::ShaderCompilationFailed(
                "No compute shader provided".into(),
            ));
        };

        // Bind group layouts, pulled from the content-keyed dedup cache so a
        // binding group created against the same `BindingLayout` reuses the
        // exact same object (wgpu requires the group's layout be compatible).
        let mut bind_group_layouts = Vec::new();
        for bg_layout in &descriptor.binding_layouts {
            bind_group_layouts.push(self.get_or_create_bind_group_layout(bg_layout));
        }

        let pipeline_layout = {
            let refs: Vec<&wgpu::BindGroupLayout> = bind_group_layouts.iter().collect();
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &refs,
                    immediate_size: 0,
                })
        };

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: descriptor.label.as_deref(),
                layout: Some(&pipeline_layout),
                module: &compute_module,
                entry_point: Some(compute_entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(super::super::GpuPipeline::WgpuCompute {
            pipeline,
            bind_group_layouts,
        })
    }

    /// Compile a ShaderSource to WGSL string for wgpu consumption.
    ///
    /// For WGSL sources: returns the source as-is.
    /// For GLSL sources: parses through naga GLSL frontend, validates, and converts to WGSL.
    /// For Slang sources: compiles through the Slang compiler to WGSL.
    fn compile_to_wgsl(
        &self,
        shader: &crate::materials::ShaderSource,
    ) -> Result<String, GraphicsError> {
        use crate::materials::ShaderSourceLanguage;

        let source_str = std::str::from_utf8(&shader.source).map_err(|e| {
            GraphicsError::ShaderCompilationFailed(format!("Invalid UTF-8 in shader: {e}"))
        })?;

        match shader.language {
            ShaderSourceLanguage::Wgsl => Ok(source_str.to_string()),
            #[cfg(feature = "slang-shaders")]
            ShaderSourceLanguage::Slang => {
                let compiler = crate::shader::SlangCompiler::new()?;
                // Write standard library modules so `import math;` etc. resolve
                compiler.write_library_modules(&crate::shader::ShaderLibrary::standard_slang())?;
                let defines: Vec<(&str, &str)> = shader
                    .defines
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect();
                compiler.compile_to_wgsl(source_str, &shader.entry_point, &[], &defines)
            }
            #[cfg(not(feature = "slang-shaders"))]
            ShaderSourceLanguage::Slang => Err(GraphicsError::FeatureNotSupported(
                "Slang shaders require the 'slang-shaders' feature".into(),
            )),
        }
    }

    /// Create a fence for CPU-GPU synchronization.
    ///
    /// wgpu has no native binary fence, so the state is emulated (see
    /// [`WgpuFenceState`](crate::backend::WgpuFenceState)): the fence starts
    /// `Signaled` or `Unsignaled` exactly as requested — matching Vulkan — and
    /// becomes submission-tracked once `execute_graph` ties a submission to it.
    pub fn create_fence(&self, signaled: bool) -> Result<GpuFence, GraphicsError> {
        Ok(GpuFence::Wgpu {
            device: self.device.clone(),
            state: Mutex::new(if signaled {
                crate::backend::WgpuFenceState::Signaled
            } else {
                crate::backend::WgpuFenceState::Unsignaled
            }),
        })
    }

    /// Wait for a fence to be signaled.
    ///
    /// Bounded by a 10 s timeout; timeout and poll failures are returned as
    /// errors so callers never mistake a hung GPU for completed work.
    /// Waiting on an `Unsignaled` fence with no tied submission is an error:
    /// nothing will ever signal it (Vulkan would stall the full timeout).
    pub fn wait_fence(&self, fence: &GpuFence) -> Result<(), GraphicsError> {
        use crate::backend::WgpuFenceState;
        let GpuFence::Wgpu { device, state } = fence else {
            return Ok(());
        };
        let current = state
            .lock()
            .map_err(|_| GraphicsError::Internal("wgpu fence mutex poisoned".into()))?
            .clone();
        match current {
            WgpuFenceState::Signaled => Ok(()),
            WgpuFenceState::Unsignaled => Err(GraphicsError::Timeout(
                "waiting on an unsignaled wgpu fence with no pending submission — \
                 it can never be signaled"
                    .into(),
            )),
            WgpuFenceState::Submitted(idx) => {
                match device.poll(wgpu::PollType::Wait {
                    submission_index: Some(idx),
                    timeout: Some(std::time::Duration::from_secs(10)),
                }) {
                    Ok(status) if status.wait_finished() => Ok(()),
                    Ok(_) => Err(GraphicsError::Timeout(
                        "fence wait timed out after 10 s; GPU may be hung".into(),
                    )),
                    Err(e) => {
                        // The submission index was never validly submitted (a
                        // rejected submit still returns an index that never
                        // signals) or the device is lost — either way, waiting
                        // on it again would fail forever. Abandon it: mark the
                        // fence signaled so the next frame recovers the slot,
                        // and report this frame's failure. See #46.
                        Self::abandon_wgpu_fence(state);
                        Err(GraphicsError::Internal(format!(
                            "device poll failed during fence wait: {e}"
                        )))
                    }
                }
            }
        }
    }

    /// Mark a wgpu fence signaled after an unrecoverable poll failure, so a
    /// later wait on the same slot returns immediately instead of re-polling a
    /// submission index that will never signal (the #46 wedge).
    fn abandon_wgpu_fence(state: &std::sync::Mutex<crate::backend::WgpuFenceState>) {
        if let Ok(mut guard) = state.lock() {
            *guard = crate::backend::WgpuFenceState::Signaled;
        }
    }

    /// Check if a fence is signaled (non-blocking).
    ///
    /// `Unsignaled` fences poll as unsignaled (matching Vulkan) until
    /// `execute_graph` ties a submission to them.
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        use crate::backend::WgpuFenceState;
        let GpuFence::Wgpu { device, state } = fence else {
            return false; // Not a wgpu fence
        };

        let Ok(guard) = state.lock() else {
            return false; // Lock failed, assume not signaled (conservative)
        };

        match &*guard {
            WgpuFenceState::Signaled => true,
            WgpuFenceState::Unsignaled => false,
            // Poll without blocking to check completion status.
            // Note: wgpu's non-blocking poll checks if ALL queue work is done,
            // not a specific submission. This is conservative but correct.
            WgpuFenceState::Submitted(_) => match device.poll(wgpu::PollType::Poll) {
                Ok(status) => status.is_queue_empty(),
                Err(_) => false, // Poll failed, assume not signaled
            },
        }
    }

    /// Wait for a fence to be signaled with a timeout.
    ///
    /// Returns `Ok(true)` if the fence was signaled, `Ok(false)` on timeout
    /// (including an untied `Unsignaled` fence, which can never signal), and
    /// an error on poll failure.
    pub fn wait_fence_timeout(
        &self,
        fence: &GpuFence,
        timeout: std::time::Duration,
    ) -> Result<bool, GraphicsError> {
        use crate::backend::WgpuFenceState;
        let GpuFence::Wgpu { device, state } = fence else {
            return Ok(false);
        };
        let current = state
            .lock()
            .map_err(|_| GraphicsError::Internal("wgpu fence mutex poisoned".into()))?
            .clone();
        match current {
            WgpuFenceState::Signaled => Ok(true),
            // Nothing will ever signal it — report "not yet" immediately
            // instead of sleeping out the timeout.
            WgpuFenceState::Unsignaled => Ok(false),
            WgpuFenceState::Submitted(idx) => {
                // `wait_finished()` is true for both QueueEmpty and WaitSucceeded;
                // `is_queue_empty()` would report a spurious timeout whenever other
                // submissions are still in flight.
                match device.poll(wgpu::PollType::Wait {
                    submission_index: Some(idx),
                    timeout: Some(timeout),
                }) {
                    Ok(status) => Ok(status.wait_finished()),
                    Err(e) => {
                        // Unrecoverable — abandon so the slot recovers (#46).
                        Self::abandon_wgpu_fence(state);
                        Err(GraphicsError::Internal(format!(
                            "device poll failed during fence wait: {e}"
                        )))
                    }
                }
            }
        }
    }

    /// Signal a fence (for testing/dummy backend).
    pub fn signal_fence(&self, _fence: &GpuFence) {
        // wgpu fences are signaled automatically when GPU work completes
    }

    /// Write data to a buffer.
    pub fn write_buffer(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), crate::error::GraphicsError> {
        if let GpuBuffer::Wgpu(wgpu_buffer) = buffer {
            self.queue.write_buffer(wgpu_buffer, offset, data);
            Ok(())
        } else {
            Err(crate::error::GraphicsError::Internal(
                "write_buffer called with non-Wgpu buffer".to_string(),
            ))
        }
    }

    /// Write tightly-packed data covering mip 0 of every layer of a texture.
    ///
    /// Mirrors the Vulkan backend's contract: block-compressed pitch is
    /// computed in blocks (not texels), the data size is validated against
    /// the tightly-packed image size, textures with `mip_level_count > 1`
    /// and combined depth-stencil formats are rejected — upload those with
    /// explicit regions via `TransferOperation::upload_texture`.
    pub fn write_texture(
        &self,
        texture: &GpuTexture,
        data: &[u8],
        descriptor: &TextureDescriptor,
    ) -> Result<(), crate::error::GraphicsError> {
        use crate::error::GraphicsError;

        let GpuTexture::Wgpu {
            texture: wgpu_texture,
            ..
        } = texture
        else {
            return Err(GraphicsError::Internal(
                "write_texture called with non-Wgpu texture".to_string(),
            ));
        };

        if data.is_empty() {
            return Ok(());
        }

        let format = descriptor.format;
        if format.is_depth_stencil() && format.has_stencil() {
            return Err(GraphicsError::InvalidParameter(format!(
                "write_texture does not support combined depth-stencil formats ({format:?})"
            )));
        }
        if descriptor.mip_level_count > 1 {
            return Err(GraphicsError::InvalidParameter(format!(
                "write_texture uploads mip 0 only, but the texture has {} mip levels — upload \
                 each mip with an explicit region via TransferOperation::upload_texture",
                descriptor.mip_level_count
            )));
        }

        // Tight pitch in BLOCKS: width in texels / block width, times bytes
        // per block (multiplying texel width by block bytes is 4x too large
        // for BC/ETC/ASTC).
        let (block_w, block_h) = format.block_dimensions();
        let block_size = format.block_size();
        let row_blocks = descriptor.size.width.div_ceil(block_w);
        let col_blocks = descriptor.size.height.div_ceil(block_h);
        let bytes_per_row = row_blocks * block_size;
        // Same layer interpretation as create_texture (Cube -> 6, CubeArray
        // -> cubes x 6, arrays -> size.depth, D3 -> real depth).
        let (layer_count, depth) = descriptor.layers_and_depth();
        let depth_or_array_layers = (layer_count * depth).max(1);

        let expected =
            bytes_per_row as usize * col_blocks as usize * depth_or_array_layers as usize;
        if data.len() != expected {
            return Err(GraphicsError::InvalidParameter(format!(
                "write_texture data size {} does not match the texture's tightly-packed size \
                 {expected} ({row_blocks}x{col_blocks} blocks x {block_size} bytes x \
                 {depth_or_array_layers} layers/depth)",
                data.len(),
            )));
        }

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: wgpu_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                // In block rows, not texel rows.
                rows_per_image: Some(col_blocks),
            },
            wgpu::Extent3d {
                width: descriptor.size.width,
                height: descriptor.size.height,
                depth_or_array_layers,
            },
        );

        Ok(())
    }

    /// Read data from a buffer.
    /// Read a host-visible buffer's mapped memory. See the trait contract on
    /// [`GpuBackend::read_buffer`](crate::backend::GpuBackend::read_buffer).
    pub fn read_buffer(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, GraphicsError> {
        let GpuBuffer::Wgpu(wgpu_buffer) = buffer else {
            return Err(GraphicsError::Internal(
                "read_buffer called with non-wgpu buffer".to_string(),
            ));
        };
        if size == 0 {
            return Ok(Vec::new());
        }
        if offset
            .checked_add(size)
            .is_none_or(|end| end > wgpu_buffer.size())
        {
            return Err(GraphicsError::InvalidParameter(format!(
                "read_buffer range at offset {offset} ({size} bytes) exceeds buffer size {}",
                wgpu_buffer.size()
            )));
        }
        // Check MAP_READ up front rather than letting `map_async` fail: a
        // failed map is a device validation error (logged by the uncaptured
        // handler) on every call, and the old staging-copy fallback then
        // panicked in `get_mapped_range`. Require a readback buffer instead —
        // matching the Vulkan backend, which also does not stage here.
        if !wgpu_buffer.usage().contains(wgpu::BufferUsages::MAP_READ) {
            return Err(GraphicsError::InvalidParameter(
                "read_buffer on a buffer without MAP_READ; copy device-local data to a \
                 readback buffer via TransferOperation::ReadbackBuffer first"
                    .to_string(),
            ));
        }

        let slice = wgpu_buffer.slice(offset..offset + size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        // Drive the map callback. The caller guarantees GPU completion
        // (post-fence), so this returns promptly.
        if let Err(e) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            return Err(GraphicsError::Internal(format!(
                "device poll failed during read_buffer map: {e}"
            )));
        }
        match rx.recv() {
            Ok(Ok(())) => {
                // `get_mapped_range()` borrows the mapping; `.to_vec()` copies
                // it out and drops the view before `unmap()`.
                let data = slice.get_mapped_range().to_vec();
                wgpu_buffer.unmap();
                Ok(data)
            }
            Ok(Err(e)) => Err(GraphicsError::Internal(format!(
                "read_buffer map_async failed: {e}"
            ))),
            Err(_) => Err(GraphicsError::Internal(
                "read_buffer map callback was dropped".to_string(),
            )),
        }
    }
}

/// Build the `wgpu::BindGroupEntry` list for a binding group descriptor.
///
/// Factored out of the per-draw encode paths: it now runs **once**, at
/// `create_binding_group` time. The returned entries borrow the resources'
/// wgpu handles (via the `Arc`s inside `entries`), so they must be consumed
/// while `entries` is alive.
///
/// `CombinedTextureSampler` is one material binding but wgpu (via Slang's WGSL
/// reflection) expects a texture at binding N **plus** a sampler at N + 1, so it
/// expands to two entries.
fn build_wgpu_bind_group_entries(
    entries: &[crate::materials::BindingEntry],
) -> Result<Vec<wgpu::BindGroupEntry<'_>>, GraphicsError> {
    use crate::materials::BoundResource;

    let mut out: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(entries.len());
    for entry in entries {
        if let BoundResource::CombinedTextureSampler { texture, sampler } = &entry.resource {
            if let GpuTexture::Wgpu { view, .. } = texture.gpu_handle() {
                out.push(wgpu::BindGroupEntry {
                    binding: entry.binding,
                    resource: wgpu::BindingResource::TextureView(view),
                });
            }
            if let GpuSampler::Wgpu(wgpu_sampler) = sampler.gpu_handle() {
                out.push(wgpu::BindGroupEntry {
                    binding: entry.binding + 1,
                    resource: wgpu::BindingResource::Sampler(wgpu_sampler),
                });
            }
            continue;
        }

        // A non-wgpu GPU handle here means a resource from another backend was
        // bound. Fail loudly at the actual cause rather than desyncing the bind
        // group from its layout.
        let mismatch = |kind: &str| {
            GraphicsError::InvalidParameter(format!(
                "wgpu: {kind} bound at binding {} has a non-wgpu GPU handle \
                 (resource from a different backend)",
                entry.binding
            ))
        };
        let resource = match &entry.resource {
            BoundResource::Buffer(buffer) => {
                let GpuBuffer::Wgpu(wgpu_buffer) = buffer.gpu_handle() else {
                    return Err(mismatch("buffer"));
                };
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: wgpu_buffer,
                    offset: 0,
                    size: None,
                })
            }
            BoundResource::BufferRange {
                buffer,
                offset,
                size,
            } => {
                let GpuBuffer::Wgpu(wgpu_buffer) = buffer.gpu_handle() else {
                    return Err(mismatch("buffer"));
                };
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: wgpu_buffer,
                    offset: *offset,
                    size: std::num::NonZeroU64::new(*size),
                })
            }
            BoundResource::Texture(texture) => {
                let GpuTexture::Wgpu { view, .. } = texture.gpu_handle() else {
                    return Err(mismatch("texture"));
                };
                wgpu::BindingResource::TextureView(view)
            }
            BoundResource::Sampler(sampler) => {
                let GpuSampler::Wgpu(wgpu_sampler) = sampler.gpu_handle() else {
                    return Err(mismatch("sampler"));
                };
                wgpu::BindingResource::Sampler(wgpu_sampler)
            }
            BoundResource::CombinedTextureSampler { .. } => {
                unreachable!("CombinedTextureSampler handled above")
            }
        };
        out.push(wgpu::BindGroupEntry {
            binding: entry.binding,
            resource,
        });
    }
    Ok(out)
}
