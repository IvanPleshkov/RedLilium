//! Barrier batch and generation for Vulkan.
//!
//! This module provides efficient barrier batching for the Vulkan backend.
//! Barriers are collected for all resources needed by a pass, then submitted
//! as a single pipeline barrier command.

use std::collections::HashMap;

use ash::vk;

use super::layout::{TextureId, TextureLayout, TextureLayoutTracker};
use crate::graph::resource_usage::PassResourceUsage;

/// A batch of image memory barriers to submit together.
///
/// Barriers are collected from all resource usages in a pass, then
/// submitted as a single `vkCmdPipelineBarrier` call for efficiency.
#[derive(Debug, Default)]
pub struct BarrierBatch {
    /// Image barriers keyed by image handle (to avoid duplicates).
    image_barriers: HashMap<TextureId, ImageBarrierInfo>,
    /// Source pipeline stage mask (union of all barriers).
    src_stage_mask: vk::PipelineStageFlags,
    /// Destination pipeline stage mask (union of all barriers).
    dst_stage_mask: vk::PipelineStageFlags,
}

/// Information for a single image barrier.
#[derive(Debug, Clone)]
struct ImageBarrierInfo {
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
    aspect_mask: vk::ImageAspectFlags,
}

impl BarrierBatch {
    /// Create a new empty barrier batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an image layout transition barrier.
    ///
    /// If a barrier for the same image already exists, it will be replaced.
    /// Barriers where `old_layout == new_layout` are skipped.
    pub fn add_image_barrier(
        &mut self,
        id: TextureId,
        image: vk::Image,
        old_layout: TextureLayout,
        new_layout: TextureLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        // Skip if no transition needed
        if old_layout == new_layout {
            return;
        }

        let info = ImageBarrierInfo {
            image,
            old_layout: old_layout.to_vk(),
            new_layout: new_layout.to_vk(),
            src_access_mask: old_layout.src_access_mask(),
            dst_access_mask: new_layout.dst_access_mask(),
            aspect_mask,
        };

        self.image_barriers.insert(id, info);
        self.src_stage_mask |= old_layout.src_stage();
        self.dst_stage_mask |= new_layout.dst_stage();
    }

    /// Check if the batch has any barriers.
    pub fn is_empty(&self) -> bool {
        self.image_barriers.is_empty()
    }

    /// Get the number of barriers in the batch.
    pub fn len(&self) -> usize {
        self.image_barriers.len()
    }

    /// Submit all barriers in a single pipeline barrier command.
    ///
    /// Does nothing if the batch is empty.
    pub fn submit(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        if self.is_empty() {
            return;
        }

        let barriers: Vec<vk::ImageMemoryBarrier> = self
            .image_barriers
            .values()
            .map(|info| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(info.old_layout)
                    .new_layout(info.new_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(info.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: info.aspect_mask,
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    })
                    .src_access_mask(info.src_access_mask)
                    .dst_access_mask(info.dst_access_mask)
            })
            .collect();

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                self.src_stage_mask,
                self.dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
    }

    /// Clear all barriers from the batch.
    pub fn clear(&mut self) {
        self.image_barriers.clear();
        self.src_stage_mask = vk::PipelineStageFlags::empty();
        self.dst_stage_mask = vk::PipelineStageFlags::empty();
    }
}

impl TextureLayoutTracker {
    /// Generate barriers for a pass's resource usage.
    ///
    /// This examines each texture usage declaration, determines if a layout
    /// transition is needed, and adds the appropriate barrier to the batch.
    /// After generating barriers, the tracker's state is updated to reflect
    /// the new layouts.
    ///
    /// # Arguments
    ///
    /// * `usage` - The resource usage declarations for the pass
    /// * `get_image_info` - A closure that returns `(vk::Image, vk::Format)` for a texture,
    ///   or `None` if the texture should be skipped (e.g., non-Vulkan backend)
    ///
    /// # Returns
    ///
    /// A `BarrierBatch` containing all necessary image memory barriers.
    pub fn generate_barriers<F>(
        &mut self,
        usage: &PassResourceUsage,
        get_image_info: F,
    ) -> BarrierBatch
    where
        F: Fn(&crate::resources::Texture) -> Option<(vk::Image, vk::Format, bool)>,
    {
        let mut batch = BarrierBatch::new();

        for decl in &usage.texture_usages {
            // Get image info from the texture
            let Some((image, format, is_depth)) = get_image_info(&decl.texture) else {
                continue;
            };

            let texture_id = TextureId::from(image);
            let current_layout = self.get_layout(texture_id);
            let required_layout = decl.access.to_layout();

            // Determine aspect mask based on format
            let aspect_mask = if is_depth {
                if format_has_stencil(format) {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                } else {
                    vk::ImageAspectFlags::DEPTH
                }
            } else {
                vk::ImageAspectFlags::COLOR
            };

            // Add barrier if layout change is needed
            batch.add_image_barrier(
                texture_id,
                image,
                current_layout,
                required_layout,
                aspect_mask,
            );

            // Update tracked state
            self.set_layout(texture_id, required_layout);
        }

        batch
    }
}

/// Check if a Vulkan format has a stencil component.
fn format_has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
            | vk::Format::S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    #[test]
    fn test_barrier_batch_empty() {
        let batch = BarrierBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_barrier_batch_skip_same_layout() {
        let mut batch = BarrierBatch::new();
        let id = TextureId::from_raw(12345);
        let image = vk::Image::from_raw(12345);

        // Adding a barrier with same old and new layout should be skipped
        batch.add_image_barrier(
            id,
            image,
            TextureLayout::ColorAttachment,
            TextureLayout::ColorAttachment,
            vk::ImageAspectFlags::COLOR,
        );

        assert!(batch.is_empty());
    }

    #[test]
    fn test_barrier_batch_adds_transition() {
        let mut batch = BarrierBatch::new();
        let id = TextureId::from_raw(12345);
        let image = vk::Image::from_raw(12345);

        batch.add_image_barrier(
            id,
            image,
            TextureLayout::Undefined,
            TextureLayout::ColorAttachment,
            vk::ImageAspectFlags::COLOR,
        );

        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_barrier_batch_deduplicates() {
        let mut batch = BarrierBatch::new();
        let id = TextureId::from_raw(12345);
        let image = vk::Image::from_raw(12345);

        // Add first barrier
        batch.add_image_barrier(
            id,
            image,
            TextureLayout::Undefined,
            TextureLayout::ColorAttachment,
            vk::ImageAspectFlags::COLOR,
        );

        // Add second barrier for same image (should replace)
        batch.add_image_barrier(
            id,
            image,
            TextureLayout::ColorAttachment,
            TextureLayout::ShaderReadOnly,
            vk::ImageAspectFlags::COLOR,
        );

        // Should still only have 1 barrier
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_barrier_batch_multiple_images() {
        let mut batch = BarrierBatch::new();

        let id1 = TextureId::from_raw(11111);
        let image1 = vk::Image::from_raw(11111);
        let id2 = TextureId::from_raw(22222);
        let image2 = vk::Image::from_raw(22222);

        batch.add_image_barrier(
            id1,
            image1,
            TextureLayout::Undefined,
            TextureLayout::ColorAttachment,
            vk::ImageAspectFlags::COLOR,
        );

        batch.add_image_barrier(
            id2,
            image2,
            TextureLayout::ColorAttachment,
            TextureLayout::ShaderReadOnly,
            vk::ImageAspectFlags::COLOR,
        );

        assert_eq!(batch.len(), 2);
    }
}
