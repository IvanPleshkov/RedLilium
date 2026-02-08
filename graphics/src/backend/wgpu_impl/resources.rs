//! Resource creation for the wgpu backend.

use std::sync::Mutex;

use crate::error::GraphicsError;
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

use super::super::{GpuBuffer, GpuFence, GpuSampler, GpuSemaphore, GpuTexture};
use super::WgpuBackend;
use super::conversion::{
    convert_address_mode, convert_buffer_usage, convert_compare_function, convert_filter_mode,
    convert_mipmap_filter_mode, convert_texture_format, convert_texture_usage,
};

impl WgpuBackend {
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

        let format = convert_texture_format(descriptor.format);
        let usage = convert_texture_usage(descriptor.usage);

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
            convert_binding_type, convert_blend_state, convert_shader_stages, convert_step_mode,
            convert_topology, convert_vertex_format,
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
            let source = std::str::from_utf8(&shader.source).map_err(|e| {
                GraphicsError::ShaderCompilationFailed(format!("Invalid UTF-8 in shader: {e}"))
            })?;
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: descriptor.label.as_deref(),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
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

        // Vertex attributes per buffer
        let buffer_count = layout.buffers.len();
        let mut vertex_attrs: Vec<Vec<wgpu::VertexAttribute>> = vec![Vec::new(); buffer_count];
        for attr in &layout.attributes {
            let idx = attr.buffer_index as usize;
            if idx < buffer_count {
                vertex_attrs[idx].push(wgpu::VertexAttribute {
                    format: convert_vertex_format(attr.format),
                    offset: attr.offset as u64,
                    shader_location: attr.semantic.index(),
                });
            }
        }

        // Bind group layouts
        let mut bind_group_layouts = Vec::new();
        for bg_layout in &descriptor.binding_layouts {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = bg_layout
                .entries
                .iter()
                .map(|entry| wgpu::BindGroupLayoutEntry {
                    binding: entry.binding,
                    visibility: convert_shader_stages(entry.visibility),
                    ty: convert_binding_type(entry.binding_type),
                    count: None,
                })
                .collect();
            bind_group_layouts.push(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: bg_layout.label.as_deref(),
                    entries: &entries,
                },
            ));
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
                Some(wgpu::ColorTargetState {
                    format: convert_texture_format(*format),
                    blend: Some(wgpu_blend_state),
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
                    polygon_mode: wgpu::PolygonMode::Fill,
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
        use super::conversion::{convert_binding_type, convert_shader_stages};
        use crate::materials::ShaderStage;

        let mut compute_module = None;
        let mut compute_entry: &str = "main";

        for shader in &descriptor.shaders {
            if shader.stage == ShaderStage::Compute {
                let source = std::str::from_utf8(&shader.source).map_err(|e| {
                    GraphicsError::ShaderCompilationFailed(format!("Invalid UTF-8 in shader: {e}"))
                })?;
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: descriptor.label.as_deref(),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
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

        // Bind group layouts
        let mut bind_group_layouts = Vec::new();
        for bg_layout in &descriptor.binding_layouts {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = bg_layout
                .entries
                .iter()
                .map(|entry| wgpu::BindGroupLayoutEntry {
                    binding: entry.binding,
                    visibility: convert_shader_stages(entry.visibility),
                    ty: convert_binding_type(entry.binding_type),
                    count: None,
                })
                .collect();
            bind_group_layouts.push(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: bg_layout.label.as_deref(),
                    entries: &entries,
                },
            ));
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

    /// Create a fence for CPU-GPU synchronization.
    ///
    /// Note: wgpu fences work differently from Vulkan fences. Instead of a binary
    /// signaled/unsignaled state, wgpu tracks submission indices. A fence with no
    /// submission (None) is considered "signaled" (no work to wait for).
    ///
    /// The `signaled` parameter is acknowledged but has limited effect:
    /// - `signaled=true`: Fence starts with no submission (effectively signaled)
    /// - `signaled=false`: Same as above - wgpu cannot represent an unsignaled fence
    ///   without pending work. The fence becomes meaningful only after `execute_graph`
    ///   stores a submission index.
    pub fn create_fence(&self, _signaled: bool) -> GpuFence {
        // Note: wgpu fences track submissions, not binary state.
        // Fence will appear signaled until work is submitted.
        GpuFence::Wgpu {
            device: self.device.clone(),
            submission_index: Mutex::new(None),
        }
    }

    /// Create a GPU semaphore (no-op for wgpu; synchronization is implicit).
    pub fn create_semaphore(&self) -> GpuSemaphore {
        GpuSemaphore::Wgpu
    }

    /// Wait for a fence to be signaled.
    pub fn wait_fence(&self, fence: &GpuFence) {
        if let GpuFence::Wgpu {
            device,
            submission_index,
        } = fence
            && let Ok(guard) = submission_index.lock()
            && let Some(idx) = guard.clone()
        {
            // Wait for the specific submission
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: Some(std::time::Duration::from_secs(10)),
            });
        }
    }

    /// Check if a fence is signaled (non-blocking).
    ///
    /// Returns `true` if:
    /// - No work has been submitted yet (fence is in initial state)
    /// - All submitted work has completed
    ///
    /// Returns `false` if:
    /// - Work is still pending on the GPU
    /// - Lock acquisition failed (conservative assumption)
    /// - Not a wgpu fence
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        let GpuFence::Wgpu {
            device,
            submission_index,
        } = fence
        else {
            return false; // Not a wgpu fence
        };

        let Ok(guard) = submission_index.lock() else {
            return false; // Lock failed, assume not signaled (conservative)
        };

        // No submission yet means fence is in initial "signaled" state
        if guard.is_none() {
            return true;
        }

        // Poll without blocking to check completion status.
        // Note: wgpu's non-blocking poll checks if ALL queue work is done,
        // not a specific submission. This is conservative but correct.
        match device.poll(wgpu::PollType::Poll) {
            Ok(status) => status.is_queue_empty(),
            Err(_) => false, // Poll failed, assume not signaled
        }
    }

    /// Wait for a fence to be signaled with a timeout.
    ///
    /// Returns `true` if the fence was signaled, `false` if the timeout elapsed.
    pub fn wait_fence_timeout(&self, fence: &GpuFence, timeout: std::time::Duration) -> bool {
        if let GpuFence::Wgpu {
            device,
            submission_index,
        } = fence
            && let Ok(guard) = submission_index.lock()
            && let Some(idx) = guard.clone()
        {
            // Wait for the specific submission with user-specified timeout
            match device.poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: Some(timeout),
            }) {
                Ok(status) => status.is_queue_empty(),
                Err(_) => false,
            }
        } else {
            // No submission or not a wgpu fence - treat as signaled
            true
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

    /// Write data to a texture.
    pub fn write_texture(
        &self,
        texture: &GpuTexture,
        data: &[u8],
        descriptor: &TextureDescriptor,
    ) -> Result<(), crate::error::GraphicsError> {
        use crate::types::TextureDimension;

        let GpuTexture::Wgpu {
            texture: wgpu_texture,
            ..
        } = texture
        else {
            return Err(crate::error::GraphicsError::Internal(
                "write_texture called with non-Wgpu texture".to_string(),
            ));
        };

        let format = convert_texture_format(descriptor.format);
        let block_size = format.block_copy_size(None).unwrap_or(4);
        let bytes_per_row = descriptor.size.width * block_size;

        let depth_or_array_layers = match descriptor.dimension {
            TextureDimension::Cube => 6,
            TextureDimension::CubeArray => descriptor.size.depth * 6,
            TextureDimension::D1
            | TextureDimension::D1Array
            | TextureDimension::D2
            | TextureDimension::D2Array
            | TextureDimension::D3 => descriptor.size.depth,
        };

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
                rows_per_image: Some(descriptor.size.height),
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
    pub fn read_buffer(&self, buffer: &GpuBuffer, offset: u64, size: u64) -> Vec<u8> {
        if let GpuBuffer::Wgpu(wgpu_buffer) = buffer {
            // Try to map the buffer directly first (works if buffer has MAP_READ)
            let slice = wgpu_buffer.slice(offset..offset + size);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                // Direct mapping succeeded
                let data = slice.get_mapped_range().to_vec();
                let _ = slice;
                wgpu_buffer.unmap();
                return data;
            }

            // Direct mapping failed - use staging buffer approach
            // This requires the source buffer to have COPY_SRC
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Read Staging Buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy from source to staging
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Read Buffer Encoder"),
                });
            encoder.copy_buffer_to_buffer(wgpu_buffer, offset, &staging, 0, size);

            let idx = self.queue.submit(std::iter::once(encoder.finish()));

            // Wait for copy to complete
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: Some(std::time::Duration::from_secs(10)),
            });

            // Map and read
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            let _ = rx.recv();

            let data = slice.get_mapped_range().to_vec();
            let _ = slice;
            staging.unmap();

            data
        } else {
            vec![0u8; size as usize]
        }
    }
}
