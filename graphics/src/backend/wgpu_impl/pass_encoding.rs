//! Pass encoding for the wgpu backend.

use crate::error::GraphicsError;
use crate::graph::{DrawCommand, Pass, RenderTargetConfig};
use crate::materials::ShaderStage;
use crate::mesh::IndexFormat;

use super::super::{GpuBuffer, GpuTexture};
use super::WgpuBackend;
use super::conversion::{
    convert_depth_load_op, convert_load_op, convert_step_mode, convert_store_op,
    convert_texture_format, convert_topology, convert_vertex_format,
};

impl WgpuBackend {
    pub(super) fn encode_pass(
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

        // Set viewport with [0, 1] depth range (wgpu/D3D convention)
        // This is the standard coordinate system used by wgpu and matches the Vulkan backend.
        if let Some((width, height)) = render_targets.dimensions() {
            render_pass.set_viewport(
                0.0,           // x
                0.0,           // y
                width as f32,  // width
                height as f32, // height
                0.0,           // min_depth (near plane)
                1.0,           // max_depth (far plane)
            );

            // Set scissor to match render area
            render_pass.set_scissor_rect(0, 0, width, height);
        }

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
        render_targets: &RenderTargetConfig,
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
                    step_mode: convert_step_mode(buffer.step_mode),
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
