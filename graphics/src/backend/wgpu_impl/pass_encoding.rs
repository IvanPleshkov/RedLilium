//! Pass encoding for the wgpu backend.

use redlilium_core::profile_scope;

use crate::error::GraphicsError;
use crate::graph::Pass;
use crate::materials::ShaderStage;
use crate::mesh::IndexFormat;

use super::super::{GpuBuffer, GpuTexture};
use super::WgpuBackend;
use super::conversion::{
    convert_binding_type, convert_blend_state, convert_depth_load_op, convert_load_op,
    convert_shader_stages, convert_step_mode, convert_store_op, convert_texture_format,
    convert_topology, convert_vertex_format,
};

impl WgpuBackend {
    pub(super) fn encode_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass: &Pass,
    ) -> Result<(), GraphicsError> {
        profile_scope!("encode_pass");
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
        use crate::graph::RenderTarget;

        // Get render targets configuration
        let Some(render_targets) = pass.render_targets() else {
            log::trace!(
                "Skipping graphics pass '{}': no render targets",
                pass.name()
            );
            return Ok(());
        };

        // Build color attachments (1 alloc per pass — acceptable)
        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = render_targets
            .color_attachments
            .iter()
            .map(|attachment| match &attachment.target {
                RenderTarget::Texture { texture, .. } => {
                    let GpuTexture::Wgpu { view, .. } = texture.gpu_handle() else {
                        return None;
                    };
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: convert_load_op(&attachment.load_op()),
                            store: convert_store_op(&attachment.store_op()),
                        },
                        depth_slice: None,
                    })
                }
                RenderTarget::Surface { view, .. } => {
                    if let Some(surface_view) = view {
                        Some(wgpu::RenderPassColorAttachment {
                            view: surface_view.view(),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: convert_load_op(&attachment.load_op()),
                                store: convert_store_op(&attachment.store_op()),
                            },
                            depth_slice: None,
                        })
                    } else {
                        log::warn!(
                            "Pass '{}' has surface attachment but no texture view available",
                            pass.name()
                        );
                        None
                    }
                }
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
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: convert_depth_load_op(&attachment.depth_load_op()),
                            store: convert_store_op(&attachment.depth_store_op()),
                        }),
                        stencil_ops: None, // TODO: Add stencil support
                    }
                });

        // Check if we have any valid attachments - wgpu requires at least one
        let has_valid_color = color_attachments.iter().any(|a| a.is_some());
        let has_depth = depth_stencil_attachment.is_some();
        if !has_valid_color && !has_depth {
            log::trace!(
                "Skipping graphics pass '{}': no valid attachments",
                pass.name()
            );
            return Ok(());
        }

        // Create render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(pass.name()),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Set viewport
        if let Some((width, height)) = render_targets.dimensions() {
            render_pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
            render_pass.set_scissor_rect(0, 0, width, height);
        }

        // Lock scratch ONCE for all draw commands in this pass.
        // Destructure to allow independent field borrows.
        let scratch = &mut *self.encoder_scratch.lock().unwrap();
        let super::WgpuEncoderScratch {
            color_formats: scratch_color_formats,
            color_targets: scratch_color_targets,
            bind_group_layout_entries: scratch_bgl_entries,
            vertex_attributes: scratch_vertex_attrs,
            bind_group_layouts: scratch_bind_group_layouts,
            bind_groups: scratch_bind_groups,
        } = scratch;

        // Reusable Vec for types with Rust lifetimes (can't go in scratch).
        // Allocated once, cleared per bind group — after the first draw, capacity is warm.
        let mut bind_group_entries: Vec<wgpu::BindGroupEntry> = Vec::new();

        // Encode each draw command (inlined to reuse all Vecs across draws)
        for draw_cmd in pass.draw_commands() {
            let material = draw_cmd.material.material();
            let mesh = &draw_cmd.mesh;

            // -- Color formats --
            scratch_color_formats.clear();
            scratch_color_formats.extend(
                render_targets
                    .color_attachments
                    .iter()
                    .map(|a| Some(convert_texture_format(a.target.format()))),
            );

            let depth_format = render_targets
                .depth_stencil_attachment
                .as_ref()
                .map(|a| convert_texture_format(a.target.format()));

            // -- Shader modules --
            let shaders = material.shaders();
            let mut vertex_module = None;
            let mut fragment_module = None;
            let mut vertex_entry: &str = "vs_main";
            let mut fragment_entry: &str = "fs_main";

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

            // -- Vertex buffer layouts --
            let layout = mesh.layout();
            let buffer_count = layout.buffers.len();

            for attr_vec in scratch_vertex_attrs.iter_mut() {
                attr_vec.clear();
            }
            scratch_vertex_attrs.resize_with(buffer_count, Vec::new);

            for attr in &layout.attributes {
                let idx = attr.buffer_index as usize;
                if idx < buffer_count {
                    scratch_vertex_attrs[idx].push(wgpu::VertexAttribute {
                        format: convert_vertex_format(attr.format),
                        offset: attr.offset as u64,
                        shader_location: attr.semantic.index(),
                    });
                }
            }

            // -- Bind group layouts --
            let binding_layouts = material.binding_layouts();
            scratch_bind_group_layouts.clear();

            for bg_layout in binding_layouts {
                scratch_bgl_entries.clear();
                scratch_bgl_entries.extend(bg_layout.entries.iter().map(|entry| {
                    wgpu::BindGroupLayoutEntry {
                        binding: entry.binding,
                        visibility: convert_shader_stages(entry.visibility),
                        ty: convert_binding_type(entry.binding_type),
                        count: None,
                    }
                }));

                scratch_bind_group_layouts.push(self.device.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
                        label: bg_layout.label.as_deref(),
                        entries: scratch_bgl_entries,
                    },
                ));
            }

            // -- Pipeline layout (scoped refs to release borrow before bind group creation) --
            let pipeline_layout = {
                let refs: Vec<&wgpu::BindGroupLayout> = scratch_bind_group_layouts.iter().collect();
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Draw Pipeline Layout"),
                        bind_group_layouts: &refs,
                        immediate_size: 0,
                    })
            };

            // -- Bind groups --
            let material_instance = &draw_cmd.material;
            scratch_bind_groups.clear();

            for (binding_group, bg_layout) in material_instance
                .binding_groups()
                .iter()
                .zip(scratch_bind_group_layouts.iter())
            {
                bind_group_entries.clear();
                for entry in &binding_group.entries {
                    let resource = match &entry.resource {
                        crate::materials::BoundResource::Buffer(buffer) => {
                            if let GpuBuffer::Wgpu(wgpu_buffer) = buffer.gpu_handle() {
                                Some(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: wgpu_buffer,
                                    offset: 0,
                                    size: None,
                                }))
                            } else {
                                None
                            }
                        }
                        crate::materials::BoundResource::Texture(texture) => {
                            if let GpuTexture::Wgpu { view, .. } = texture.gpu_handle() {
                                Some(wgpu::BindingResource::TextureView(view))
                            } else {
                                None
                            }
                        }
                        crate::materials::BoundResource::Sampler(sampler) => {
                            if let crate::backend::GpuSampler::Wgpu(wgpu_sampler) =
                                sampler.gpu_handle()
                            {
                                Some(wgpu::BindingResource::Sampler(wgpu_sampler))
                            } else {
                                None
                            }
                        }
                        crate::materials::BoundResource::CombinedTextureSampler {
                            texture, ..
                        } => {
                            if let GpuTexture::Wgpu { view, .. } = texture.gpu_handle() {
                                Some(wgpu::BindingResource::TextureView(view))
                            } else {
                                None
                            }
                        }
                    };
                    if let Some(r) = resource {
                        bind_group_entries.push(wgpu::BindGroupEntry {
                            binding: entry.binding,
                            resource: r,
                        });
                    }
                }

                scratch_bind_groups.push(self.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: binding_group.label.as_deref(),
                        layout: bg_layout,
                        entries: &bind_group_entries,
                    },
                ));
            }

            // -- Color targets --
            let wgpu_blend_state = material
                .blend_state()
                .map(convert_blend_state)
                .unwrap_or(wgpu::BlendState::REPLACE);

            scratch_color_targets.clear();
            scratch_color_targets.extend(scratch_color_formats.iter().map(|format| {
                format.map(|f| wgpu::ColorTargetState {
                    format: f,
                    blend: Some(wgpu_blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })
            }));

            // -- Create render pipeline (scoped vertex_buffer_layouts to release borrow
            //    of scratch_vertex_attrs before next iteration mutates it) --
            let pipeline = {
                let vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout> = layout
                    .buffers
                    .iter()
                    .enumerate()
                    .map(|(i, buffer)| wgpu::VertexBufferLayout {
                        array_stride: buffer.stride as u64,
                        step_mode: convert_step_mode(buffer.step_mode),
                        attributes: &scratch_vertex_attrs[i],
                    })
                    .collect();

                self.device
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
                            targets: scratch_color_targets,
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
                    })
            };

            // -- Record into render pass --
            render_pass.set_pipeline(&pipeline);

            for (index, bind_group) in scratch_bind_groups.iter().enumerate() {
                render_pass.set_bind_group(index as u32, bind_group, &[]);
            }

            for (slot, buffer) in mesh.vertex_buffers().iter().enumerate() {
                if let GpuBuffer::Wgpu(wgpu_buffer) = buffer.gpu_handle() {
                    render_pass.set_vertex_buffer(slot as u32, wgpu_buffer.slice(..));
                }
            }

            if mesh.is_indexed() {
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
                    let bytes_per_row =
                        region
                            .buffer_layout
                            .bytes_per_row
                            .or(if region.extent.height > 1 {
                                let unpadded = region.extent.width * block_size;
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
                    let bytes_per_row =
                        region
                            .buffer_layout
                            .bytes_per_row
                            .or(if region.extent.height > 1 {
                                let unpadded = region.extent.width * block_size;
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
