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
            Pass::AccelerationStructureBuild(build_pass) => {
                return Err(GraphicsError::FeatureNotSupported(format!(
                    "pass '{}': acceleration-structure builds are not supported on the wgpu \
                     backend (DeviceCapabilities::ray_query is false, #110)",
                    build_pass.name()
                )));
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

        // Build color attachments on stack (8 = wgpu max_color_attachments
        // default). Exceeding it must be an error: silently dropping trailing
        // attachments would render without them and corrupt MRT output.
        let color_count = render_targets.color_attachments.len();
        if color_count > 8 {
            return Err(GraphicsError::InvalidParameter(format!(
                "pass '{}' declares {} color attachments; wgpu supports at most 8",
                pass.name(),
                color_count
            )));
        }
        let mut color_attachments = [const { None }; 8];
        for (i, attachment) in render_targets.color_attachments.iter().enumerate() {
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

        // Build depth stencil attachment if present. Mismatched handles are
        // errors, not panics — matching the color/buffer handling in this
        // file (a panic here also poisons the encoder scratch mutex).
        let depth_stencil_attachment = match &render_targets.depth_stencil_attachment {
            Some(attachment) => {
                let view = match &attachment.target {
                    RenderTarget::Texture { texture, .. } => {
                        let GpuTexture::Wgpu { view, .. } = texture.gpu_handle() else {
                            return Err(GraphicsError::InvalidParameter(format!(
                                "wgpu: depth attachment of pass '{}' has a non-wgpu GPU \
                                 handle (resource from a different backend)",
                                pass.name()
                            )));
                        };
                        view
                    }
                    RenderTarget::Surface { .. } => {
                        return Err(GraphicsError::InvalidParameter(format!(
                            "wgpu: pass '{}' uses the surface as a depth attachment; \
                             surfaces are color-only",
                            pass.name()
                        )));
                    }
                };
                // wgpu expresses a read-only depth/stencil attachment as
                // `ops: None` (WebGPU `depthReadOnly` / `stencilReadOnly`). This
                // is what lets the same depth texture be a depth attachment and
                // be sampled in one pass (#60); WebGPU rejects a writable depth
                // attachment that is also bound for sampling.
                let depth_ops = if attachment.depth_read_only {
                    None
                } else {
                    Some(wgpu::Operations {
                        load: convert_depth_load_op(&attachment.depth_load_op()),
                        store: convert_store_op(&attachment.depth_store_op()),
                    })
                };
                let stencil_ops =
                    if attachment.target.format().has_stencil() && !attachment.stencil_read_only {
                        Some(wgpu::Operations {
                            load: convert_stencil_load_op(&attachment.stencil_load_op()),
                            store: convert_store_op(&attachment.stencil_store_op()),
                        })
                    } else {
                        None
                    };
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops,
                    stencil_ops,
                })
            }
            None => None,
        };

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

        // Set scissor rect (use pass override or fall back to full target
        // dimensions). An explicit scissor is clamped to the target — wgpu
        // requires the rect ⊆ target, and a negative origin cast to u32
        // becomes ~4 billion and panics.
        let pass_scissor = pass.scissor_rect();
        if let (Some(sr), Some((width, height))) = (pass_scissor, default_dims) {
            let c = sr.clamped(width, height);
            render_pass.set_scissor_rect(c.x, c.y, c.width, c.height);
        } else if let Some((width, height)) = default_dims {
            render_pass.set_scissor_rect(0, 0, width, height);
        }

        // Mesh-tasks draws are Vulkan-only (#111); their materials cannot
        // even be created on wgpu, so a command here is a graph-construction
        // bug — fail loudly rather than render an incomplete frame.
        if !pass.mesh_tasks_commands().is_empty() {
            return Err(GraphicsError::FeatureNotSupported(format!(
                "pass '{}' contains mesh-tasks draws, which require \
                 DeviceCapabilities::mesh_shading (Vulkan-only, #111)",
                pass.name()
            )));
        }

        // Encode each draw command
        for draw_cmd in pass.draw_commands() {
            let material_arc = draw_cmd.material.material();
            let mesh = &draw_cmd.mesh;

            // -- Pipeline: owned by Material, created at create_material() time --
            let super::super::GpuPipeline::WgpuGraphics { pipeline, .. } =
                material_arc.gpu_handle()
            else {
                log::warn!("Material has no wgpu graphics pipeline");
                continue;
            };

            // -- Record into render pass --
            render_pass.set_pipeline(pipeline);

            // Bind each group's pre-built, cached bind group — no per-draw
            // creation. Compatibility with the pipeline is guaranteed by the
            // shared, content-deduped bind group layout.
            for (index, group) in draw_cmd.material.binding_groups().iter().enumerate() {
                let super::super::GpuBindingGroup::Wgpu(bind_group) = group.gpu_handle() else {
                    return Err(GraphicsError::InvalidParameter(format!(
                        "wgpu: binding group {index} has no wgpu bind group \
                         (resource from a different backend)"
                    )));
                };
                let offsets: &[u32] = draw_cmd
                    .dynamic_offsets
                    .get(index)
                    .map_or(&[], |v| v.as_slice());
                render_pass.set_bind_group(index as u32, bind_group, offsets);
            }

            for (slot, buffer) in mesh.vertex_buffers().iter().enumerate() {
                if let GpuBuffer::Wgpu(wgpu_buffer) = buffer.gpu_handle() {
                    render_pass.set_vertex_buffer(
                        slot as u32,
                        wgpu_buffer.slice(mesh.vertex_offset(slot)..),
                    );
                }
            }

            // Set per-draw scissor rect if specified (clamped to the target,
            // as above).
            let custom_scissor = draw_cmd.scissor_rect.is_some();
            if let (Some(scissor), Some((width, height))) = (&draw_cmd.scissor_rect, default_dims) {
                let c = scissor.clamped(width, height);
                render_pass.set_scissor_rect(c.x, c.y, c.width, c.height);
            }

            if mesh.is_indexed() {
                if let Some(index_buffer) = mesh.index_buffer()
                    && let GpuBuffer::Wgpu(wgpu_buffer) = index_buffer.gpu_handle()
                {
                    let index_format = match mesh.index_format().unwrap_or(IndexFormat::Uint16) {
                        IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                        IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                    };
                    render_pass
                        .set_index_buffer(wgpu_buffer.slice(mesh.index_offset()..), index_format);
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

            // Restore pass-level scissor after per-draw override (clamped, as
            // when it was first set).
            if custom_scissor {
                if let (Some(sr), Some((width, height))) = (pass_scissor, default_dims) {
                    let c = sr.clamped(width, height);
                    render_pass.set_scissor_rect(c.x, c.y, c.width, c.height);
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

                crate::graph::validate_buffer_copy_alignment(regions)?;
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
            TransferOperation::WriteBuffer {
                dst,
                dst_offset,
                data,
                src_range,
            } => {
                let GpuBuffer::Wgpu(dst_buffer) = dst.gpu_handle() else {
                    return Ok(());
                };
                let bytes = data.get(src_range.clone()).ok_or_else(|| {
                    GraphicsError::InvalidParameter(format!(
                        "WriteBuffer src_range {:?} out of bounds (data len {})",
                        src_range,
                        data.len()
                    ))
                })?;
                if bytes.is_empty() {
                    return Ok(());
                }
                if !dst_offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
                    || !(bytes.len() as u64).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
                {
                    return Err(GraphicsError::InvalidParameter(format!(
                        "WriteBuffer requires 4-byte aligned dst_offset and size \
                         (got offset {}, size {})",
                        dst_offset,
                        bytes.len()
                    )));
                }
                // Copy via a transient staging buffer at THIS point in the
                // encoder, so the write lands at the transfer pass's position
                // in the graph. queue.write_buffer would instead execute at the
                // start of the submission — before ALL passes, regardless of
                // where the transfer pass sits. The staging buffer is dropped
                // after encoding; wgpu keeps it alive until the submission
                // completes.
                let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("write_buffer_staging"),
                    size: bytes.len() as u64,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                staging
                    .slice(..)
                    .get_mapped_range_mut()
                    .copy_from_slice(bytes);
                staging.unmap();
                encoder.copy_buffer_to_buffer(
                    &staging,
                    0,
                    dst_buffer,
                    *dst_offset,
                    bytes.len() as u64,
                );
            }
            // Drained by the frame pipeline after the fence (CPU read); nothing
            // to encode here.
            TransferOperation::ReadbackBuffer { .. } => {}
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

                for region in regions {
                    // Shared resolver: same tight-packing/block math and
                    // alignment validation as the Vulkan backend.
                    let layout = region
                        .buffer_layout
                        .resolve(format, region.extent)
                        .map_err(|e| {
                            GraphicsError::InvalidParameter(format!("TextureToBuffer: {e}"))
                        })?;

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
                                offset: layout.offset,
                                bytes_per_row: layout.multi_row.then_some(layout.bytes_per_row),
                                rows_per_image: (region.extent.depth > 1)
                                    .then_some(layout.rows_per_image_blocks),
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

                for region in regions {
                    // Shared resolver: same tight-packing/block math and
                    // alignment validation as the Vulkan backend.
                    let layout = region
                        .buffer_layout
                        .resolve(format, region.extent)
                        .map_err(|e| {
                            GraphicsError::InvalidParameter(format!("BufferToTexture: {e}"))
                        })?;

                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: src_buffer,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: layout.offset,
                                bytes_per_row: layout.multi_row.then_some(layout.bytes_per_row),
                                rows_per_image: (region.extent.depth > 1)
                                    .then_some(layout.rows_per_image_blocks),
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
            // No blit path in wgpu (#96): DeviceCapabilities.mip_generation is
            // false, so the loader never emits this op for a wgpu device and the
            // texture stays single-mip. Encode nothing if one arrives anyway.
            TransferOperation::GenerateMipmaps { .. } => {}
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

        for dispatch_cmd in pass.dispatch_commands() {
            let material_arc = dispatch_cmd.material.material();

            // -- Pipeline: owned by Material, created at create_material() time --
            let super::super::GpuPipeline::WgpuCompute { pipeline, .. } = material_arc.gpu_handle()
            else {
                log::warn!("Material has no wgpu compute pipeline");
                continue;
            };

            // Record into compute pass
            compute_pass.set_pipeline(pipeline);

            // Bind each group's cached, pre-built bind group (no per-dispatch
            // creation).
            for (index, group) in dispatch_cmd.material.binding_groups().iter().enumerate() {
                let super::super::GpuBindingGroup::Wgpu(bind_group) = group.gpu_handle() else {
                    return Err(GraphicsError::InvalidParameter(format!(
                        "wgpu: binding group {index} has no wgpu bind group \
                         (resource from a different backend)"
                    )));
                };
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
