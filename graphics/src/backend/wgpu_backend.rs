//! wgpu GPU backend implementation.
//!
//! This backend uses wgpu for cross-platform GPU access, supporting
//! Vulkan, Metal, DX12, and WebGPU.

use std::sync::{Arc, Mutex};

use crate::error::GraphicsError;
use crate::graph::{CompiledGraph, DrawCommand, Pass, RenderGraph};
use crate::materials::ShaderStage;
use crate::mesh::{IndexFormat, PrimitiveTopology, VertexAttributeFormat, VertexStepMode};
use crate::types::{
    AddressMode, BufferDescriptor, BufferUsage, CompareFunction, FilterMode, SamplerDescriptor,
    TextureDescriptor, TextureFormat, TextureUsage,
};

use super::{GpuBuffer, GpuFence, GpuSampler, GpuTexture};

/// wgpu-based GPU backend.
pub struct WgpuBackend {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBackend")
            .field("adapter", &self.adapter.get_info().name)
            .finish()
    }
}

impl WgpuBackend {
    /// Create a new wgpu backend.
    pub fn new() -> Result<Self, GraphicsError> {
        // Create instance with all backends
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        });

        // Request adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("No compatible GPU adapter: {e}"))
        })?;

        log::info!("wgpu adapter: {:?}", adapter.get_info());

        // Request device
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RedLilium Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("Device creation failed: {e}"))
        })?;

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /// Get the wgpu device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Get the wgpu queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
}

impl WgpuBackend {
    /// Get the backend name.
    pub fn name(&self) -> &'static str {
        "wgpu Backend"
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

        Ok(GpuBuffer::Wgpu(Arc::new(buffer)))
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        let format = convert_texture_format(descriptor.format);
        let usage = convert_texture_usage(descriptor.usage);

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: descriptor.label.as_deref(),
            size: wgpu::Extent3d {
                width: descriptor.size.width,
                height: descriptor.size.height,
                depth_or_array_layers: descriptor.size.depth,
            },
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(GpuTexture::Wgpu {
            texture: Arc::new(texture),
            view: Arc::new(view),
        })
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

        Ok(GpuSampler::Wgpu(Arc::new(sampler)))
    }

    /// Create a fence for CPU-GPU synchronization.
    pub fn create_fence(&self, _signaled: bool) -> GpuFence {
        GpuFence::Wgpu {
            device: self.device.clone(),
            submission_index: Mutex::new(None),
        }
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
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        if let GpuFence::Wgpu {
            device,
            submission_index,
        } = fence
            && let Ok(guard) = submission_index.lock()
            && guard.is_some()
            && let Ok(status) = device.poll(wgpu::PollType::Poll)
        {
            return status.is_queue_empty();
        }
        // No submission yet or not wgpu fence means "done" or default
        matches!(fence, GpuFence::Wgpu { .. })
    }

    /// Signal a fence (for testing/dummy backend).
    pub fn signal_fence(&self, _fence: &GpuFence) {
        // wgpu fences are signaled automatically when GPU work completes
    }

    /// Execute a compiled render graph.
    pub fn execute_graph(
        &self,
        graph: &RenderGraph,
        compiled: &CompiledGraph,
        signal_fence: Option<&GpuFence>,
    ) -> Result<(), GraphicsError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderGraph Encoder"),
            });

        // Get all passes from the graph
        let passes = graph.passes();

        // Process each pass in compiled order
        for handle in compiled.pass_order() {
            let pass = &passes[handle.index()];
            self.encode_pass(&mut encoder, pass)?;
        }

        // Submit commands
        let command_buffer = encoder.finish();
        let submission_index = self.queue.submit(std::iter::once(command_buffer));

        // Store submission index in fence for polling
        if let Some(GpuFence::Wgpu {
            submission_index: fence_idx,
            ..
        }) = signal_fence
            && let Ok(mut guard) = fence_idx.lock()
        {
            *guard = Some(submission_index.clone());
        }

        // Wait for GPU to complete before returning
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: Some(std::time::Duration::from_secs(10)),
        });

        Ok(())
    }

    /// Write data to a buffer.
    pub fn write_buffer(&self, buffer: &GpuBuffer, offset: u64, data: &[u8]) {
        if let GpuBuffer::Wgpu(wgpu_buffer) = buffer {
            self.queue.write_buffer(wgpu_buffer, offset, data);
        }
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

    fn encode_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass: &Pass,
    ) -> Result<(), GraphicsError> {
        match pass {
            Pass::Graphics(graphics_pass) => {
                self.encode_graphics_pass(encoder, graphics_pass)?;
            }
            Pass::Transfer(transfer_pass) => {
                self.encode_transfer_pass(encoder, transfer_pass)?;
            }
            Pass::Compute(compute_pass) => {
                self.encode_compute_pass(encoder, compute_pass)?;
            }
        }
        Ok(())
    }

    fn encode_graphics_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass: &crate::graph::GraphicsPass,
    ) -> Result<(), GraphicsError> {
        // Get render targets configuration
        let Some(render_targets) = pass.render_targets() else {
            // No render targets configured, skip this pass
            log::trace!(
                "Skipping graphics pass '{}': no render targets",
                pass.name()
            );
            return Ok(());
        };

        // Build color attachments
        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = render_targets
            .color_attachments
            .iter()
            .map(|attachment| {
                let GpuTexture::Wgpu { view, .. } = attachment.texture().gpu_handle() else {
                    return None;
                };

                let load_op = convert_load_op(&attachment.load_op());
                let store_op = convert_store_op(&attachment.store_op());

                Some(wgpu::RenderPassColorAttachment {
                    view: view.as_ref(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: load_op,
                        store: store_op,
                    },
                    depth_slice: None,
                })
            })
            .collect();

        // Build depth stencil attachment if present
        let depth_stencil_attachment =
            render_targets
                .depth_stencil_attachment
                .as_ref()
                .map(|attachment| {
                    let GpuTexture::Wgpu { view, .. } = attachment.texture().gpu_handle() else {
                        panic!("Invalid depth texture GPU handle");
                    };

                    wgpu::RenderPassDepthStencilAttachment {
                        view: view.as_ref(),
                        depth_ops: Some(wgpu::Operations {
                            load: convert_depth_load_op(&attachment.depth_load_op()),
                            store: convert_store_op(&attachment.depth_store_op()),
                        }),
                        stencil_ops: None, // TODO: Add stencil support
                    }
                });

        // Create render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(pass.name()),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Encode draw commands
        for draw_cmd in pass.draw_commands() {
            self.encode_draw_command(&mut render_pass, draw_cmd, render_targets)?;
        }

        Ok(())
    }

    fn encode_draw_command<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        draw_cmd: &'a DrawCommand,
        render_targets: &crate::graph::RenderTargetConfig,
    ) -> Result<(), GraphicsError> {
        let material = draw_cmd.material.material();
        let mesh = &draw_cmd.mesh;

        // Get color target formats
        let color_formats: Vec<Option<wgpu::TextureFormat>> = render_targets
            .color_attachments
            .iter()
            .map(|a| Some(convert_texture_format(a.texture().format())))
            .collect();

        // Get depth format if present
        let depth_format = render_targets
            .depth_stencil_attachment
            .as_ref()
            .map(|a| convert_texture_format(a.texture().format()));

        // Create shader modules from material
        let shaders = material.shaders();
        let mut vertex_module = None;
        let mut fragment_module = None;
        let mut vertex_entry = "vs_main";
        let mut fragment_entry = "fs_main";

        for shader in shaders {
            let source = std::str::from_utf8(&shader.source).map_err(|e| {
                GraphicsError::ShaderCompilationFailed(format!("Invalid UTF-8 in shader: {e}"))
            })?;

            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: material.label(),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });

            match shader.stage {
                ShaderStage::Vertex => {
                    vertex_module = Some(module);
                    vertex_entry = Box::leak(shader.entry_point.clone().into_boxed_str());
                }
                ShaderStage::Fragment => {
                    fragment_module = Some(module);
                    fragment_entry = Box::leak(shader.entry_point.clone().into_boxed_str());
                }
                ShaderStage::Compute => {}
            }
        }

        let vertex_module = vertex_module.ok_or_else(|| {
            GraphicsError::ShaderCompilationFailed("No vertex shader provided".into())
        })?;

        // Build vertex buffer layouts
        let layout = mesh.layout();
        let vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout> = layout
            .buffers
            .iter()
            .enumerate()
            .map(|(buffer_idx, buffer)| {
                let attributes: Vec<wgpu::VertexAttribute> = layout
                    .attributes
                    .iter()
                    .filter(|attr| attr.buffer_index == buffer_idx as u32)
                    .map(|attr| wgpu::VertexAttribute {
                        format: convert_vertex_format(attr.format),
                        offset: attr.offset as u64,
                        shader_location: attr.semantic.index(),
                    })
                    .collect();

                wgpu::VertexBufferLayout {
                    array_stride: buffer.stride as u64,
                    step_mode: match buffer.step_mode {
                        VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
                        VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
                    },
                    attributes: Box::leak(attributes.into_boxed_slice()),
                }
            })
            .collect();

        // Create pipeline layout (empty for now - no bind groups)
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Draw Pipeline Layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        // Build color targets
        let color_targets: Vec<Option<wgpu::ColorTargetState>> = color_formats
            .iter()
            .map(|format| {
                format.map(|f| wgpu::ColorTargetState {
                    format: f,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect();

        // Create render pipeline
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: material.label(),
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
                    topology: convert_topology(mesh.topology()),
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

        // Set the pipeline
        render_pass.set_pipeline(&pipeline);

        // Bind vertex buffers
        for (slot, buffer) in mesh.vertex_buffers().iter().enumerate() {
            if let GpuBuffer::Wgpu(wgpu_buffer) = buffer.gpu_handle() {
                render_pass.set_vertex_buffer(slot as u32, wgpu_buffer.slice(..));
            }
        }

        // Issue draw call
        if mesh.is_indexed() {
            // Bind index buffer
            if let Some(index_buffer) = mesh.index_buffer()
                && let GpuBuffer::Wgpu(wgpu_buffer) = index_buffer.gpu_handle()
            {
                let index_format = match mesh.index_format().unwrap_or(IndexFormat::Uint16) {
                    IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                    IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                };
                render_pass.set_index_buffer(wgpu_buffer.slice(..), index_format);
            }
            render_pass.draw_indexed(
                0..mesh.index_count(),
                0,
                draw_cmd.first_instance..(draw_cmd.first_instance + draw_cmd.instance_count),
            );
        } else {
            render_pass.draw(
                0..mesh.vertex_count(),
                draw_cmd.first_instance..(draw_cmd.first_instance + draw_cmd.instance_count),
            );
        }

        Ok(())
    }

    fn encode_transfer_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass: &crate::graph::TransferPass,
    ) -> Result<(), GraphicsError> {
        let Some(config) = pass.transfer_config() else {
            return Ok(());
        };

        for operation in &config.operations {
            self.encode_transfer_operation(encoder, operation)?;
        }
        Ok(())
    }

    fn encode_transfer_operation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        operation: &crate::graph::TransferOperation,
    ) -> Result<(), GraphicsError> {
        use crate::graph::TransferOperation;

        match operation {
            TransferOperation::BufferToBuffer { src, dst, regions } => {
                let GpuBuffer::Wgpu(src_buffer) = src.gpu_handle() else {
                    return Ok(());
                };
                let GpuBuffer::Wgpu(dst_buffer) = dst.gpu_handle() else {
                    return Ok(());
                };

                for region in regions {
                    encoder.copy_buffer_to_buffer(
                        src_buffer,
                        region.src_offset,
                        dst_buffer,
                        region.dst_offset,
                        region.size,
                    );
                }
            }
            TransferOperation::TextureToBuffer { src, dst, regions } => {
                let GpuTexture::Wgpu {
                    texture: src_texture,
                    ..
                } = src.gpu_handle()
                else {
                    return Ok(());
                };
                let GpuBuffer::Wgpu(dst_buffer) = dst.gpu_handle() else {
                    return Ok(());
                };

                let format = src.format();
                let block_size = format.block_size();

                for region in regions {
                    // Compute bytes_per_row if not specified (and align to 256 bytes as required by wgpu)
                    let bytes_per_row =
                        region
                            .buffer_layout
                            .bytes_per_row
                            .or(if region.extent.height > 1 {
                                let unpadded = region.extent.width * block_size;
                                // wgpu requires 256-byte alignment for bytes_per_row
                                Some((unpadded + 255) & !255)
                            } else {
                                None
                            });

                    let rows_per_image =
                        region
                            .buffer_layout
                            .rows_per_image
                            .or(if region.extent.depth > 1 {
                                Some(region.extent.height)
                            } else {
                                None
                            });

                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo {
                            texture: src_texture,
                            mip_level: region.texture_location.mip_level,
                            origin: wgpu::Origin3d {
                                x: region.texture_location.origin.x,
                                y: region.texture_location.origin.y,
                                z: region.texture_location.origin.z,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyBufferInfo {
                            buffer: dst_buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: region.buffer_layout.offset,
                                bytes_per_row,
                                rows_per_image,
                            },
                        },
                        wgpu::Extent3d {
                            width: region.extent.width,
                            height: region.extent.height,
                            depth_or_array_layers: region.extent.depth,
                        },
                    );
                }
            }
            TransferOperation::BufferToTexture { src, dst, regions } => {
                let GpuBuffer::Wgpu(src_buffer) = src.gpu_handle() else {
                    return Ok(());
                };
                let GpuTexture::Wgpu {
                    texture: dst_texture,
                    ..
                } = dst.gpu_handle()
                else {
                    return Ok(());
                };

                let format = dst.format();
                let block_size = format.block_size();

                for region in regions {
                    // Compute bytes_per_row if not specified (and align to 256 bytes as required by wgpu)
                    let bytes_per_row =
                        region
                            .buffer_layout
                            .bytes_per_row
                            .or(if region.extent.height > 1 {
                                let unpadded = region.extent.width * block_size;
                                // wgpu requires 256-byte alignment for bytes_per_row
                                Some((unpadded + 255) & !255)
                            } else {
                                None
                            });

                    let rows_per_image =
                        region
                            .buffer_layout
                            .rows_per_image
                            .or(if region.extent.depth > 1 {
                                Some(region.extent.height)
                            } else {
                                None
                            });

                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: src_buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: region.buffer_layout.offset,
                                bytes_per_row,
                                rows_per_image,
                            },
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: dst_texture,
                            mip_level: region.texture_location.mip_level,
                            origin: wgpu::Origin3d {
                                x: region.texture_location.origin.x,
                                y: region.texture_location.origin.y,
                                z: region.texture_location.origin.z,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: region.extent.width,
                            height: region.extent.height,
                            depth_or_array_layers: region.extent.depth,
                        },
                    );
                }
            }
            TransferOperation::TextureToTexture { src, dst, regions } => {
                let GpuTexture::Wgpu {
                    texture: src_texture,
                    ..
                } = src.gpu_handle()
                else {
                    return Ok(());
                };
                let GpuTexture::Wgpu {
                    texture: dst_texture,
                    ..
                } = dst.gpu_handle()
                else {
                    return Ok(());
                };

                for region in regions {
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: src_texture,
                            mip_level: region.src.mip_level,
                            origin: wgpu::Origin3d {
                                x: region.src.origin.x,
                                y: region.src.origin.y,
                                z: region.src.origin.z,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: dst_texture,
                            mip_level: region.dst.mip_level,
                            origin: wgpu::Origin3d {
                                x: region.dst.origin.x,
                                y: region.dst.origin.y,
                                z: region.dst.origin.z,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: region.extent.width,
                            height: region.extent.height,
                            depth_or_array_layers: region.extent.depth,
                        },
                    );
                }
            }
        }
        Ok(())
    }

    fn encode_compute_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass: &crate::graph::ComputePass,
    ) -> Result<(), GraphicsError> {
        let _compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass.name()),
            timestamp_writes: None,
        });

        // TODO: Encode compute dispatches
        Ok(())
    }
}

// ============================================================================
// Conversion Functions
// ============================================================================

fn convert_buffer_usage(usage: BufferUsage) -> wgpu::BufferUsages {
    let mut result = wgpu::BufferUsages::empty();

    if usage.contains(BufferUsage::VERTEX) {
        result |= wgpu::BufferUsages::VERTEX;
    }
    if usage.contains(BufferUsage::INDEX) {
        result |= wgpu::BufferUsages::INDEX;
    }
    if usage.contains(BufferUsage::UNIFORM) {
        result |= wgpu::BufferUsages::UNIFORM;
    }
    if usage.contains(BufferUsage::STORAGE) {
        result |= wgpu::BufferUsages::STORAGE;
    }
    if usage.contains(BufferUsage::INDIRECT) {
        result |= wgpu::BufferUsages::INDIRECT;
    }
    if usage.contains(BufferUsage::COPY_SRC) {
        result |= wgpu::BufferUsages::COPY_SRC;
    }
    if usage.contains(BufferUsage::COPY_DST) {
        result |= wgpu::BufferUsages::COPY_DST;
    }
    if usage.contains(BufferUsage::MAP_READ) {
        result |= wgpu::BufferUsages::MAP_READ;
    }
    if usage.contains(BufferUsage::MAP_WRITE) {
        result |= wgpu::BufferUsages::MAP_WRITE;
    }

    result
}

fn convert_texture_format(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        // 8-bit formats
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        TextureFormat::R8Snorm => wgpu::TextureFormat::R8Snorm,
        TextureFormat::R8Uint => wgpu::TextureFormat::R8Uint,
        TextureFormat::R8Sint => wgpu::TextureFormat::R8Sint,

        // 16-bit formats
        TextureFormat::R16Unorm => wgpu::TextureFormat::R16Unorm,
        TextureFormat::R16Float => wgpu::TextureFormat::R16Float,
        TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,

        // 32-bit formats
        TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
        TextureFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        TextureFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,

        // 64-bit formats
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,

        // 128-bit formats
        TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,

        // Depth/stencil formats
        TextureFormat::Depth16Unorm => wgpu::TextureFormat::Depth16Unorm,
        TextureFormat::Depth24Plus => wgpu::TextureFormat::Depth24Plus,
        TextureFormat::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        TextureFormat::Depth32FloatStencil8 => wgpu::TextureFormat::Depth32FloatStencil8,
    }
}

fn convert_texture_usage(usage: TextureUsage) -> wgpu::TextureUsages {
    let mut result = wgpu::TextureUsages::empty();

    if usage.contains(TextureUsage::COPY_SRC) {
        result |= wgpu::TextureUsages::COPY_SRC;
    }
    if usage.contains(TextureUsage::COPY_DST) {
        result |= wgpu::TextureUsages::COPY_DST;
    }
    if usage.contains(TextureUsage::TEXTURE_BINDING) {
        result |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if usage.contains(TextureUsage::STORAGE_BINDING) {
        result |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
        result |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }

    result
}

fn convert_address_mode(mode: AddressMode) -> wgpu::AddressMode {
    match mode {
        AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        AddressMode::Repeat => wgpu::AddressMode::Repeat,
        AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
        AddressMode::ClampToBorder => wgpu::AddressMode::ClampToBorder,
    }
}

fn convert_filter_mode(mode: FilterMode) -> wgpu::FilterMode {
    match mode {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

fn convert_mipmap_filter_mode(mode: FilterMode) -> wgpu::MipmapFilterMode {
    match mode {
        FilterMode::Nearest => wgpu::MipmapFilterMode::Nearest,
        FilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

fn convert_compare_function(func: CompareFunction) -> wgpu::CompareFunction {
    match func {
        CompareFunction::Never => wgpu::CompareFunction::Never,
        CompareFunction::Less => wgpu::CompareFunction::Less,
        CompareFunction::Equal => wgpu::CompareFunction::Equal,
        CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        CompareFunction::Greater => wgpu::CompareFunction::Greater,
        CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        CompareFunction::Always => wgpu::CompareFunction::Always,
    }
}

fn convert_load_op(op: &crate::graph::LoadOp) -> wgpu::LoadOp<wgpu::Color> {
    match op {
        crate::graph::LoadOp::Load => wgpu::LoadOp::Load,
        crate::graph::LoadOp::DontCare => wgpu::LoadOp::Load, // wgpu doesn't have DontCare for color
        crate::graph::LoadOp::Clear(clear_value) => {
            if let crate::types::ClearValue::Color { r, g, b, a } = clear_value {
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: *r as f64,
                    g: *g as f64,
                    b: *b as f64,
                    a: *a as f64,
                })
            } else {
                wgpu::LoadOp::Load
            }
        }
    }
}

fn convert_depth_load_op(op: &crate::graph::LoadOp) -> wgpu::LoadOp<f32> {
    match op {
        crate::graph::LoadOp::Load => wgpu::LoadOp::Load,
        crate::graph::LoadOp::DontCare => wgpu::LoadOp::Load, // wgpu doesn't have DontCare for depth
        crate::graph::LoadOp::Clear(clear_value) => {
            if let crate::types::ClearValue::Depth(depth) = clear_value {
                wgpu::LoadOp::Clear(*depth)
            } else if let crate::types::ClearValue::DepthStencil { depth, .. } = clear_value {
                wgpu::LoadOp::Clear(*depth)
            } else {
                wgpu::LoadOp::Load
            }
        }
    }
}

fn convert_store_op(op: &crate::graph::StoreOp) -> wgpu::StoreOp {
    match op {
        crate::graph::StoreOp::Store => wgpu::StoreOp::Store,
        crate::graph::StoreOp::DontCare => wgpu::StoreOp::Discard,
    }
}

fn convert_vertex_format(format: VertexAttributeFormat) -> wgpu::VertexFormat {
    match format {
        VertexAttributeFormat::Float => wgpu::VertexFormat::Float32,
        VertexAttributeFormat::Float2 => wgpu::VertexFormat::Float32x2,
        VertexAttributeFormat::Float3 => wgpu::VertexFormat::Float32x3,
        VertexAttributeFormat::Float4 => wgpu::VertexFormat::Float32x4,
        VertexAttributeFormat::Int => wgpu::VertexFormat::Sint32,
        VertexAttributeFormat::Int2 => wgpu::VertexFormat::Sint32x2,
        VertexAttributeFormat::Int3 => wgpu::VertexFormat::Sint32x3,
        VertexAttributeFormat::Int4 => wgpu::VertexFormat::Sint32x4,
        VertexAttributeFormat::Uint => wgpu::VertexFormat::Uint32,
        VertexAttributeFormat::Uint2 => wgpu::VertexFormat::Uint32x2,
        VertexAttributeFormat::Uint3 => wgpu::VertexFormat::Uint32x3,
        VertexAttributeFormat::Uint4 => wgpu::VertexFormat::Uint32x4,
        VertexAttributeFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
        VertexAttributeFormat::Snorm8x4 => wgpu::VertexFormat::Snorm8x4,
    }
}

fn convert_topology(topology: PrimitiveTopology) -> wgpu::PrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
        PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}
