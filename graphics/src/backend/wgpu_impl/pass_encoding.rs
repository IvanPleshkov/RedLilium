//! Pass encoding for the wgpu backend.

use redlilium_core::profile_scope;

use crate::error::GraphicsError;
use crate::graph::Pass;
use crate::mesh::IndexFormat;

use super::super::{GpuBuffer, GpuTexture};
use super::WgpuBackend;
use super::conversion::{
    convert_depth_load_op, convert_load_op, convert_stencil_load_op, convert_store_op,
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
            return Ok(());
        };

        // Build color attachments on stack (8 = wgpu max_color_attachments default).
        let color_count = render_targets.color_attachments.len().min(8);
        debug_assert!(
            render_targets.color_attachments.len() <= 8,
            "More than 8 color attachments exceeds wgpu default limits"
        );
        let mut color_attachments = [const { None }; 8];
        for (i, attachment) in render_targets.color_attachments.iter().enumerate().take(8) {
            color_attachments[i] = match &attachment.target {
                RenderTarget::Texture { texture, .. } => {
                    let GpuTexture::Wgpu { view, .. } = texture.gpu_handle() else {
                        continue;
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
            };
        }
        let color_attachments = &color_attachments[..color_count];

        // Build depth stencil attachment if present
        let depth_stencil_attachment =
            render_targets
                .depth_stencil_attachment
                .as_ref()
                .map(|attachment| {
                    let GpuTexture::Wgpu { view, .. } = attachment.texture().gpu_handle() else {
                        panic!("Invalid depth texture GPU handle");
                    };
                    let stencil_ops = if attachment.target.format().has_stencil() {
                        Some(wgpu::Operations {
                            load: convert_stencil_load_op(&attachment.stencil_load_op()),
                            store: convert_store_op(&attachment.stencil_store_op()),
                        })
                    } else {
                        None
                    };
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: convert_depth_load_op(&attachment.depth_load_op()),
                            store: convert_store_op(&attachment.depth_store_op()),
                        }),
                        stencil_ops,
                    }
                });

        // Check if we have any valid attachments - wgpu requires at least one
        let has_valid_color = color_attachments.iter().any(|a| a.is_some());
        let has_depth = depth_stencil_attachment.is_some();
        if !has_valid_color && !has_depth {
            return Ok(());
        }

        // Create render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(pass.name()),
            color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Set viewport (use pass override or fall back to full target dimensions)
        let default_dims = render_targets.dimensions();
        if let Some(vp) = pass.viewport() {
            render_pass.set_viewport(vp.x, vp.y, vp.width, vp.height, vp.min_depth, vp.max_depth);
        } else if let Some((width, height)) = default_dims {
            render_pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        }

        // Set scissor rect (use pass override or fall back to full target dimensions)
        let pass_scissor = pass.scissor_rect();
        if let Some(sr) = pass_scissor {
            render_pass.set_scissor_rect(sr.x as u32, sr.y as u32, sr.width, sr.height);
        } else if let Some((width, height)) = default_dims {
            render_pass.set_scissor_rect(0, 0, width, height);
        }

        // Lock scratch ONCE for all draw commands in this pass.
        // Destructure to allow independent field borrows.
        let scratch = &mut *self.encoder_scratch.lock().unwrap();
        let super::WgpuEncoderScratch {
            bind_group_layouts: scratch_bind_group_layouts,
            bind_groups: scratch_bind_groups,
            ..
        } = scratch;

        // Reusable Vec for types with Rust lifetimes (can't go in scratch).
        // Allocated once, cleared per bind group — after the first draw, capacity is warm.
        let mut bind_group_entries: Vec<wgpu::BindGroupEntry> = Vec::new();

        // Encode each draw command
        for draw_cmd in pass.draw_commands() {
            let material_arc = draw_cmd.material.material();
            let mesh = &draw_cmd.mesh;

            // -- Pipeline: owned by Material, created at create_material() time --
            let super::super::GpuPipeline::WgpuGraphics {
                pipeline,
                bind_group_layouts,
            } = material_arc.gpu_handle()
            else {
                log::warn!("Material has no wgpu graphics pipeline");
                continue;
            };

            scratch_bind_group_layouts.clear();
            scratch_bind_group_layouts.extend(bind_group_layouts.iter().cloned());

            // -- Bind groups (always per draw: resources may change each frame) --
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

            // -- Record into render pass --
            render_pass.set_pipeline(pipeline);

            for (index, bind_group) in scratch_bind_groups.iter().enumerate() {
                render_pass.set_bind_group(index as u32, bind_group, &[]);
            }

            for (slot, buffer) in mesh.vertex_buffers().iter().enumerate() {
                if let GpuBuffer::Wgpu(wgpu_buffer) = buffer.gpu_handle() {
                    render_pass.set_vertex_buffer(slot as u32, wgpu_buffer.slice(..));
                }
            }

            // Set per-draw scissor rect if specified
            let custom_scissor = draw_cmd.scissor_rect.is_some();
            if let Some(scissor) = &draw_cmd.scissor_rect {
                render_pass.set_scissor_rect(
                    scissor.x as u32,
                    scissor.y as u32,
                    scissor.width,
                    scissor.height,
                );
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

            // Restore pass-level scissor after per-draw override
            if custom_scissor {
                if let Some(sr) = pass_scissor {
                    render_pass.set_scissor_rect(sr.x as u32, sr.y as u32, sr.width, sr.height);
                } else if let Some((width, height)) = default_dims {
                    render_pass.set_scissor_rect(0, 0, width, height);
                }
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
        if !pass.has_dispatches() {
            return Ok(());
        }

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(pass.name()),
            timestamp_writes: None,
        });

        let scratch = &mut *self.encoder_scratch.lock().unwrap();
        let super::WgpuEncoderScratch {
            bind_group_layouts: scratch_bind_group_layouts,
            bind_groups: scratch_bind_groups,
            ..
        } = scratch;

        let mut bind_group_entries: Vec<wgpu::BindGroupEntry> = Vec::new();

        for dispatch_cmd in pass.dispatch_commands() {
            let material_arc = dispatch_cmd.material.material();

            // -- Pipeline: owned by Material, created at create_material() time --
            let super::super::GpuPipeline::WgpuCompute {
                pipeline,
                bind_group_layouts,
            } = material_arc.gpu_handle()
            else {
                log::warn!("Material has no wgpu compute pipeline");
                continue;
            };

            scratch_bind_group_layouts.clear();
            scratch_bind_group_layouts.extend(bind_group_layouts.iter().cloned());

            // -- Bind groups (always per dispatch: resources may change each frame) --
            let material_instance = &dispatch_cmd.material;
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

            // Record into compute pass
            compute_pass.set_pipeline(pipeline);

            for (index, bind_group) in scratch_bind_groups.iter().enumerate() {
                compute_pass.set_bind_group(index as u32, bind_group, &[]);
            }

            compute_pass.dispatch_workgroups(
                dispatch_cmd.workgroup_count_x,
                dispatch_cmd.workgroup_count_y,
                dispatch_cmd.workgroup_count_z,
            );
        }

        Ok(())
    }
}
