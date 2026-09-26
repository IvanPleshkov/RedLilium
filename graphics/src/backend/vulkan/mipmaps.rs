//! Cross-face cubemap reduction. All commands are lowered from a graph op.

use super::{MAX_FRAMES_IN_FLIGHT, VulkanBackend};
use crate::backend::{GpuPipeline, GpuTexture};
use crate::{
    BindingLayout, CullMode, GraphicsError, MaterialDescriptor, ShaderSource, Texture,
    TextureDescriptor, TextureFormat, TextureUsage,
};
use ash::vk;
use std::{
    collections::HashMap,
    sync::{Arc, atomic::Ordering},
};

#[derive(Default)]
pub(super) struct CubeMipState {
    pipelines: HashMap<TextureFormat, GpuPipeline>,
    slots: [Vec<Scratch>; MAX_FRAMES_IN_FLIGHT],
}
impl CubeMipState {
    pub(super) fn retire_slot(&mut self, slot: usize) {
        self.slots[slot].clear();
    }
    pub(super) fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.clear();
        }
        self.pipelines.clear();
    }
}

struct Scratch {
    device: ash::Device,
    _image: GpuTexture,
    views: Vec<vk::ImageView>,
    pool: vk::DescriptorPool,
}
impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.pool, None);
            for view in self.views.drain(..) {
                self.device.destroy_image_view(view, None);
            }
        }
        // image drops after all its views; the owning slot's fences have retired.
    }
}
fn creation(error: vk::Result) -> GraphicsError {
    GraphicsError::ResourceCreationFailed(format!("cubemap mip resources: {error:?}"))
}

impl VulkanBackend {
    pub(super) fn encode_cube_mipmaps(
        &self,
        cmd: vk::CommandBuffer,
        texture: &Texture,
    ) -> Result<(), GraphicsError> {
        let levels = texture.mip_level_count();
        if levels <= 1 {
            return Ok(());
        }
        let GpuTexture::Vulkan { image: target, .. } = texture.gpu_handle() else {
            return Ok(());
        };
        let layers = if texture.dimension() == crate::TextureDimension::Cube {
            6
        } else {
            texture.depth().checked_mul(6).ok_or_else(|| {
                GraphicsError::InvalidParameter("cube layer count overflows".into())
            })?
        };
        let mut state = self.cube_mips.lock();
        if !state.pipelines.contains_key(&texture.format()) {
            let code = include_bytes!("../mipmaps_cube.wgsl");
            let mut descriptor = MaterialDescriptor::new()
                .with_shader(ShaderSource::vertex(code.to_vec(), "vs"))
                .with_shader(ShaderSource::fragment(code.to_vec(), "fs"));
            descriptor.binding_layouts =
                vec![Arc::new(BindingLayout::new().with_texture_2d_array(0))];
            descriptor.color_formats = vec![texture.format()];
            descriptor.raster.cull_mode = CullMode::None;
            state
                .pipelines
                .insert(texture.format(), self.create_pipeline(&descriptor)?);
        }
        let GpuPipeline::Vulkan {
            pipeline,
            pipeline_layout,
            descriptor_set_layouts,
            ..
        } = &state.pipelines[&texture.format()]
        else {
            unreachable!()
        };
        let (pipeline, pipeline_layout, set_layout) =
            (*pipeline, *pipeline_layout, descriptor_set_layouts[0]);
        let image = self.create_texture(
            &TextureDescriptor::new_2d_array(
                texture.width(),
                texture.height(),
                layers,
                texture.format(),
                TextureUsage::COPY_SRC
                    | TextureUsage::COPY_DST
                    | TextureUsage::TEXTURE_BINDING
                    | TextureUsage::RENDER_ATTACHMENT,
            )
            .with_mip_levels(levels),
        )?;
        let GpuTexture::Vulkan {
            image: scratch_image,
            ..
        } = &image
        else {
            unreachable!()
        };
        let scratch_image = *scratch_image;
        let mut scratch = Scratch {
            device: self.device.clone(),
            _image: image,
            views: Vec::new(),
            pool: vk::DescriptorPool::null(),
        };
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: levels - 1,
        }];
        scratch.pool = unsafe {
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(levels - 1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(creation)?;
        let layouts = vec![set_layout; (levels - 1) as usize];
        let sets = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(scratch.pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(creation)?;
        let view = |scratch: &mut Scratch,
                    mip,
                    layer,
                    count,
                    view_type|
         -> Result<vk::ImageView, GraphicsError> {
            let info = vk::ImageViewCreateInfo::default()
                .image(scratch_image)
                .format(self.vk_texture_format(texture.format()))
                .view_type(view_type)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip,
                    level_count: 1,
                    base_array_layer: layer,
                    layer_count: count,
                });
            let view = unsafe { self.device.create_image_view(&info, None) }.map_err(creation)?;
            scratch.views.push(view);
            Ok(view)
        };
        let sub = |mip| vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: mip,
            base_array_layer: 0,
            layer_count: layers,
        };
        let copy = |cmd, src, dst, mip| unsafe {
            self.device.cmd_copy_image(
                cmd,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageCopy::default()
                    .src_subresource(sub(mip))
                    .dst_subresource(sub(mip))
                    .extent(vk::Extent3D {
                        width: (texture.width() >> mip).max(1),
                        height: (texture.height() >> mip).max(1),
                        depth: 1,
                    })],
            );
        };
        self.mip_barrier(
            cmd,
            scratch_image,
            0,
            levels,
            layers,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        self.mip_barrier(
            cmd,
            *target,
            0,
            1,
            layers,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        copy(cmd, *target, scratch_image, 0);
        self.mip_barrier(
            cmd,
            *target,
            0,
            1,
            layers,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        for mip in 1..levels {
            self.mip_barrier(
                cmd,
                scratch_image,
                mip - 1,
                1,
                layers,
                if mip == 1 {
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL
                } else {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                },
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            self.mip_barrier(
                cmd,
                scratch_image,
                mip,
                1,
                layers,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            let source = view(
                &mut scratch,
                mip - 1,
                0,
                layers,
                vk::ImageViewType::TYPE_2D_ARRAY,
            )?;
            let image_info = [vk::DescriptorImageInfo::default()
                .image_view(source)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let set = sets[(mip - 1) as usize];
            unsafe {
                self.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&image_info)],
                    &[],
                );
            }
            let width = (texture.width() >> mip).max(1);
            let height = (texture.height() >> mip).max(1);
            let area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D { width, height },
            };
            for layer in 0..layers {
                let destination = view(&mut scratch, mip, layer, 1, vk::ImageViewType::TYPE_2D)?;
                let color = [vk::RenderingAttachmentInfo::default()
                    .image_view(destination)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE)];
                unsafe {
                    self.device.cmd_begin_rendering(
                        cmd,
                        &vk::RenderingInfo::default()
                            .render_area(area)
                            .layer_count(1)
                            .color_attachments(&color),
                    );
                    self.device.cmd_set_viewport(
                        cmd,
                        0,
                        &[vk::Viewport {
                            x: 0.0,
                            y: height as f32,
                            width: width as f32,
                            height: -(height as f32),
                            min_depth: 0.0,
                            max_depth: 1.0,
                        }],
                    );
                    self.device.cmd_set_scissor(cmd, 0, &[area]);
                    self.device
                        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &[set],
                        &[],
                    );
                    self.device.cmd_draw(cmd, 3, 1, 0, layer);
                    self.device.cmd_end_rendering(cmd);
                }
            }
            self.mip_barrier(
                cmd,
                scratch_image,
                mip,
                1,
                layers,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            copy(cmd, scratch_image, *target, mip);
        }
        // Native scratch, views and descriptor sets survive all queued work.
        state.slots[self.current_slot.load(Ordering::SeqCst)].push(scratch);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn mip_barrier(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        mip: u32,
        count: u32,
        layers: u32,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
    ) {
        let access = |layout| match layout {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
                vk::PipelineStageFlags2::ALL_TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            ),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            _ => (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
        };
        let (src_stage, src_access) = access(old);
        let (dst_stage, dst_access) = access(new);
        let barriers = [vk::ImageMemoryBarrier2::default()
            .image(image)
            .old_layout(old)
            .new_layout(new)
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip,
                level_count: count,
                base_array_layer: 0,
                layer_count: layers,
            })];
        unsafe {
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
        }
    }
}
