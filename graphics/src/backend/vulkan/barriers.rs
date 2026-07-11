//! Barrier batch and generation for Vulkan.
//!
//! This module provides efficient barrier batching for the Vulkan backend.
//! Barriers are collected for all resources needed by a pass, then submitted
//! as a single pipeline barrier command.

use std::collections::HashMap;

use ash::vk;
use ash::vk::Handle;

use super::layout::{TextureId, TextureLayout};
use crate::graph::resource_usage::BufferAccessMode;

/// Unique identifier for a Vulkan buffer within the barrier batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

impl From<vk::Buffer> for BufferId {
    fn from(buffer: vk::Buffer) -> Self {
        Self(buffer.as_raw())
    }
}

impl BufferId {
    /// Create a buffer ID from a raw Vulkan buffer handle.
    pub fn from_raw(handle: u64) -> Self {
        Self(handle)
    }

    /// Get the raw handle value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Synchronization state of a single buffer on the GPU timeline.
///
/// Tracks the last write (stage + access) plus which read scopes that write
/// has already been made visible to, so repeated readers don't re-emit
/// barriers (VK-P7) and writers get a precise source scope instead of a
/// hardcoded `TRANSFER` guess (VK-C3).
#[derive(Debug, Default, Clone, Copy)]
struct BufferAccessState {
    /// Pipeline stages of the last tracked write.
    last_write_stage: vk::PipelineStageFlags,
    /// Access mask of the last tracked write.
    last_write_access: vk::AccessFlags,
    /// Read stages the last write has been made visible to (via a barrier).
    visible_stages: vk::PipelineStageFlags,
    /// Read access types the last write has been made visible to.
    visible_access: vk::AccessFlags,
    /// All read stages seen since the last write (for write-after-read
    /// execution dependencies).
    reads_since_write: vk::PipelineStageFlags,
}

/// Tracks the last access to every buffer used by render graphs, mirroring
/// [`super::layout::TextureLayoutTracker`] for buffers.
///
/// The tracker is global and persists across frames.
///
/// INVARIANT (single queue): correctness relies on every tracked access being
/// on the one graphics queue. A `vkCmdPipelineBarrier` synchronizes across
/// submissions only *within a queue*, so a write recorded in one submit is a
/// valid source scope for a read in a later submit — whether that is the next
/// graph of the same frame or a graph of the next frame — only because all
/// submits are on that queue in submission order. This tracker has no concept
/// of queue ownership: a second queue (async compute) would need semaphores +
/// queue-ownership tracking instead, and adding one without that would
/// silently drop synchronization. See #47 before introducing a second queue.
///
/// Keyed by the raw `vk::Buffer` handle. Unlike texture layouts, stale state
/// after handle reuse is benign: it can only produce an unnecessary or
/// overly-broad barrier, never skip a needed one (a new buffer has no GPU
/// writes to synchronize until a tracked write records one, which resets the
/// entry). Destroyed buffers are removed via the retirement queue drained in
/// `advance_frame`, keeping the map bounded by live resources.
#[derive(Debug, Default)]
pub struct BufferAccessTracker {
    states: HashMap<BufferId, BufferAccessState>,
}

impl BufferAccessTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an access and return the source scope for a barrier, if one is
    /// needed before it.
    ///
    /// Returns `Some((src_stage, src_access))` when a barrier must be placed
    /// before the pass performing `access`, or `None` when the access is
    /// already synchronized (first GPU use, host-written buffers, or a read
    /// scope the last write is already visible to).
    pub fn request_access(
        &mut self,
        id: BufferId,
        access: BufferAccessMode,
    ) -> Option<(vk::PipelineStageFlags, vk::AccessFlags)> {
        let state = self.states.entry(id).or_default();
        let stage = access.dst_stage();
        let access_mask = access.dst_access_mask();

        if access.is_write() {
            // Writes must wait for the previous write (WAW) and for all reads
            // issued since it (WAR — execution dependency only, so reads
            // contribute no source access mask).
            let src_stage = state.last_write_stage | state.reads_since_write;
            let src_access = state.last_write_access;

            *state = BufferAccessState {
                last_write_stage: stage,
                last_write_access: access_mask,
                ..Default::default()
            };

            (!src_stage.is_empty()).then_some((src_stage, src_access))
        } else {
            state.reads_since_write |= stage;

            if state.last_write_access.is_empty() {
                // No tracked GPU write: nothing to make visible. Host writes
                // are made visible by queue submission itself.
                return None;
            }
            if state.visible_stages.contains(stage) && state.visible_access.contains(access_mask) {
                // The last write is already visible to this read scope.
                return None;
            }
            state.visible_stages |= stage;
            state.visible_access |= access_mask;
            Some((state.last_write_stage, state.last_write_access))
        }
    }

    /// Remove a destroyed buffer's entry (called from the retirement drain
    /// in `advance_frame`).
    pub fn remove(&mut self, id: BufferId) {
        self.states.remove(&id);
    }
}

/// A batch of memory barriers (both image and buffer) to submit together.
///
/// Barriers are collected from all resource usages in a pass, then
/// submitted as a single `vkCmdPipelineBarrier` call for efficiency.
#[derive(Debug, Default)]
pub struct BarrierBatch {
    /// Image barriers keyed by image handle (to avoid duplicates).
    image_barriers: HashMap<TextureId, ImageBarrierInfo>,
    /// Buffer barriers keyed by buffer handle (to avoid duplicates).
    buffer_barriers: HashMap<BufferId, BufferBarrierInfo>,
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

/// Information for a single buffer barrier.
///
/// Barriers always cover the whole buffer: the access tracker works at
/// whole-buffer granularity, and a range-limited barrier would only make a
/// write visible for that range while the tracker marks the whole buffer as
/// synchronized.
#[derive(Debug, Clone)]
struct BufferBarrierInfo {
    buffer: vk::Buffer,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
}

impl BarrierBatch {
    /// Create a new empty barrier batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an image layout transition barrier.
    ///
    /// Same-layout uses are skipped only for read-only layouts: two
    /// consecutive passes writing an image in the same layout (color target
    /// rendered twice, storage image in `General` across dispatches) have a
    /// WAW/RAW hazard that dynamic rendering does not implicitly order, so a
    /// memory barrier without a layout transition is emitted.
    ///
    /// If a barrier for the same image already exists in the batch, the
    /// transitions are collapsed into one chain: the existing `old_layout`
    /// is kept, the new `new_layout` wins, and access scopes are unioned —
    /// replacing the entry outright would submit a barrier whose
    /// `old_layout` no longer matches the image's actual layout.
    pub fn add_image_barrier(
        &mut self,
        id: TextureId,
        image: vk::Image,
        old_layout: TextureLayout,
        new_layout: TextureLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        // Same read-only layout: no hazard, nothing to do.
        if old_layout == new_layout && !new_layout.is_write() {
            return;
        }

        match self.image_barriers.entry(id) {
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                let info = occupied.get_mut();
                info.new_layout = new_layout.to_vk();
                info.src_access_mask |= old_layout.src_access_mask();
                info.dst_access_mask |= new_layout.dst_access_mask();
            }
            std::collections::hash_map::Entry::Vacant(vacant) => {
                vacant.insert(ImageBarrierInfo {
                    image,
                    old_layout: old_layout.to_vk(),
                    new_layout: new_layout.to_vk(),
                    src_access_mask: old_layout.src_access_mask(),
                    dst_access_mask: new_layout.dst_access_mask(),
                    aspect_mask,
                });
            }
        }
        self.src_stage_mask |= old_layout.src_stage();
        self.dst_stage_mask |= new_layout.dst_stage();
    }

    /// Add a buffer memory barrier with explicit stage/access scopes.
    ///
    /// Buffer barriers ensure memory coherence between different access types.
    /// If a barrier for the same buffer already exists in the batch, the
    /// scopes are merged (mask union) — a pass that both reads and writes a
    /// buffer gets one barrier covering both dependencies, not whichever
    /// happened to be added last.
    pub fn add_buffer_barrier(
        &mut self,
        id: BufferId,
        buffer: vk::Buffer,
        src_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_stage: vk::PipelineStageFlags,
        dst_access: vk::AccessFlags,
    ) {
        let info = self
            .buffer_barriers
            .entry(id)
            .or_insert_with(|| BufferBarrierInfo {
                buffer,
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::empty(),
            });
        info.src_access_mask |= src_access;
        info.dst_access_mask |= dst_access;
        self.src_stage_mask |= src_stage;
        self.dst_stage_mask |= dst_stage;
    }

    /// Check if the batch has any barriers.
    pub fn is_empty(&self) -> bool {
        self.image_barriers.is_empty() && self.buffer_barriers.is_empty()
    }

    /// Get the number of image barriers in the batch.
    pub fn image_barrier_count(&self) -> usize {
        self.image_barriers.len()
    }

    /// Get the number of buffer barriers in the batch.
    pub fn buffer_barrier_count(&self) -> usize {
        self.buffer_barriers.len()
    }

    /// Get the total number of barriers in the batch.
    pub fn len(&self) -> usize {
        self.image_barriers.len() + self.buffer_barriers.len()
    }

    /// Submit all barriers in a single pipeline barrier command.
    ///
    /// Does nothing if the batch is empty.
    pub fn submit(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        if self.is_empty() {
            return;
        }

        let image_barriers: Vec<vk::ImageMemoryBarrier> = self
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

        let buffer_barriers: Vec<vk::BufferMemoryBarrier> = self
            .buffer_barriers
            .values()
            .map(|info| {
                vk::BufferMemoryBarrier::default()
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(info.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
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
                &buffer_barriers,
                &image_barriers,
            );
        }
    }

    /// Clear all barriers from the batch.
    pub fn clear(&mut self) {
        self.image_barriers.clear();
        self.buffer_barriers.clear();
        self.src_stage_mask = vk::PipelineStageFlags::empty();
        self.dst_stage_mask = vk::PipelineStageFlags::empty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrier_batch_empty() {
        let batch = BarrierBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.image_barrier_count(), 0);
        assert_eq!(batch.buffer_barrier_count(), 0);
    }

    #[test]
    fn test_barrier_batch_skip_same_readonly_layout() {
        let mut batch = BarrierBatch::new();
        let id = TextureId::from_raw(12345);
        let image = vk::Image::from_raw(12345);

        // Same read-only layout: read-after-read, no hazard, no barrier.
        batch.add_image_barrier(
            id,
            image,
            TextureLayout::ShaderReadOnly,
            TextureLayout::ShaderReadOnly,
            vk::ImageAspectFlags::COLOR,
        );

        assert!(batch.is_empty());
    }

    #[test]
    fn test_barrier_batch_same_write_layout_emits_barrier() {
        let mut batch = BarrierBatch::new();
        let id = TextureId::from_raw(12345);
        let image = vk::Image::from_raw(12345);

        // Rendering to the same color target in two consecutive passes is a
        // WAW hazard even though the layout doesn't change (VK-H1): a memory
        // barrier without a transition must be emitted.
        batch.add_image_barrier(
            id,
            image,
            TextureLayout::ColorAttachment,
            TextureLayout::ColorAttachment,
            vk::ImageAspectFlags::COLOR,
        );

        assert_eq!(batch.image_barrier_count(), 1);
        let info = batch.image_barriers.values().next().unwrap();
        assert_eq!(info.old_layout, info.new_layout);
        assert!(
            info.dst_access_mask
                .contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        );
    }

    #[test]
    fn test_barrier_batch_duplicate_collapses_chain() {
        let mut batch = BarrierBatch::new();
        let id = TextureId::from_raw(12345);
        let image = vk::Image::from_raw(12345);

        // One pass declaring two transitions for the same image: the batch
        // must collapse them into old(first) -> new(last) — replacing the
        // entry would submit a barrier whose old_layout doesn't match the
        // image's actual layout (VK-L1).
        batch.add_image_barrier(
            id,
            image,
            TextureLayout::Undefined,
            TextureLayout::TransferDst,
            vk::ImageAspectFlags::COLOR,
        );
        batch.add_image_barrier(
            id,
            image,
            TextureLayout::TransferDst,
            TextureLayout::ShaderReadOnly,
            vk::ImageAspectFlags::COLOR,
        );

        assert_eq!(batch.image_barrier_count(), 1);
        let info = batch.image_barriers.values().next().unwrap();
        assert_eq!(info.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(info.new_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert!(info.dst_access_mask.contains(vk::AccessFlags::SHADER_READ));
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
        assert_eq!(batch.image_barrier_count(), 1);
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
        assert_eq!(batch.image_barrier_count(), 2);
    }

    // Buffer barrier tests

    /// Shorthand: barrier scopes for "transfer write -> vertex read".
    fn transfer_to_vertex() -> (
        vk::PipelineStageFlags,
        vk::AccessFlags,
        vk::PipelineStageFlags,
        vk::AccessFlags,
    ) {
        (
            BufferAccessMode::TransferWrite.src_stage(),
            BufferAccessMode::TransferWrite.src_access_mask(),
            BufferAccessMode::VertexBuffer.dst_stage(),
            BufferAccessMode::VertexBuffer.dst_access_mask(),
        )
    }

    #[test]
    fn test_buffer_barrier_adds() {
        let mut batch = BarrierBatch::new();
        let id = BufferId::from_raw(12345);
        let buffer = vk::Buffer::from_raw(12345);

        let (src_stage, src_access, dst_stage, dst_access) = transfer_to_vertex();
        batch.add_buffer_barrier(id, buffer, src_stage, src_access, dst_stage, dst_access);

        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.buffer_barrier_count(), 1);
        assert_eq!(batch.image_barrier_count(), 0);
    }

    #[test]
    fn test_buffer_barrier_merges_scopes() {
        let mut batch = BarrierBatch::new();
        let id = BufferId::from_raw(12345);
        let buffer = vk::Buffer::from_raw(12345);

        // Same buffer, two different dependencies in one pass: the scopes
        // must merge into one barrier covering both, not replace each other.
        let (src_stage, src_access, dst_stage, dst_access) = transfer_to_vertex();
        batch.add_buffer_barrier(id, buffer, src_stage, src_access, dst_stage, dst_access);
        batch.add_buffer_barrier(
            id,
            buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::VERTEX_SHADER,
            vk::AccessFlags::UNIFORM_READ,
        );

        assert_eq!(batch.buffer_barrier_count(), 1);
        let info = batch.buffer_barriers.values().next().unwrap();
        assert!(
            info.src_access_mask
                .contains(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
        );
        assert!(
            info.dst_access_mask
                .contains(vk::AccessFlags::VERTEX_ATTRIBUTE_READ | vk::AccessFlags::UNIFORM_READ)
        );
        assert!(
            batch.src_stage_mask.contains(
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER
            )
        );
    }

    // Buffer access tracker tests (VK-C3 / VK-P7)

    #[test]
    fn tracker_first_read_needs_no_barrier() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        // Host-written (or never-written) buffer: reads need no barrier.
        assert_eq!(
            tracker.request_access(id, BufferAccessMode::VertexBuffer),
            None
        );
        assert_eq!(
            tracker.request_access(id, BufferAccessMode::UniformRead),
            None
        );
    }

    #[test]
    fn tracker_compute_write_then_read_uses_shader_src_scope() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        // First write: nothing earlier to synchronize against.
        assert_eq!(
            tracker.request_access(id, BufferAccessMode::StorageReadWrite),
            None
        );

        // Read after the compute write: src scope must be the shader write —
        // the old hardcoded TRANSFER source would miss it entirely (VK-C3).
        let (src_stage, src_access) = tracker
            .request_access(id, BufferAccessMode::VertexBuffer)
            .expect("read after write needs a barrier");
        assert!(src_stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(src_access.contains(vk::AccessFlags::SHADER_WRITE));
    }

    #[test]
    fn tracker_repeated_read_is_skipped() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        tracker.request_access(id, BufferAccessMode::TransferWrite);
        assert!(
            tracker
                .request_access(id, BufferAccessMode::VertexBuffer)
                .is_some()
        );
        // Same read scope again (next pass, next frame): already visible,
        // no redundant barrier (VK-P7).
        assert_eq!(
            tracker.request_access(id, BufferAccessMode::VertexBuffer),
            None
        );
        // A *different* read scope still needs its own visibility.
        assert!(
            tracker
                .request_access(id, BufferAccessMode::IndirectRead)
                .is_some()
        );
    }

    #[test]
    fn tracker_write_after_read_has_execution_dependency() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        tracker.request_access(id, BufferAccessMode::VertexBuffer);
        // Write after read: needs an execution dependency on the read stage,
        // with no source access (reads make nothing available).
        let (src_stage, src_access) = tracker
            .request_access(id, BufferAccessMode::TransferWrite)
            .expect("write after read needs a barrier");
        assert!(src_stage.contains(vk::PipelineStageFlags::VERTEX_INPUT));
        assert_eq!(src_access, vk::AccessFlags::empty());
    }

    #[test]
    fn tracker_write_after_write_chains_scopes() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        tracker.request_access(id, BufferAccessMode::TransferWrite);
        let (src_stage, src_access) = tracker
            .request_access(id, BufferAccessMode::StorageWrite)
            .expect("write after write needs a barrier");
        assert!(src_stage.contains(vk::PipelineStageFlags::TRANSFER));
        assert!(src_access.contains(vk::AccessFlags::TRANSFER_WRITE));

        // And a read now synchronizes against the SECOND write.
        let (src_stage, src_access) = tracker
            .request_access(id, BufferAccessMode::UniformRead)
            .expect("read after write needs a barrier");
        assert!(src_stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(src_access.contains(vk::AccessFlags::SHADER_WRITE));
    }

    #[test]
    fn test_mixed_barriers() {
        let mut batch = BarrierBatch::new();

        // Add image barrier
        let tex_id = TextureId::from_raw(11111);
        let image = vk::Image::from_raw(11111);
        batch.add_image_barrier(
            tex_id,
            image,
            TextureLayout::Undefined,
            TextureLayout::ColorAttachment,
            vk::ImageAspectFlags::COLOR,
        );

        // Add buffer barrier
        let buf_id = BufferId::from_raw(22222);
        let buffer = vk::Buffer::from_raw(22222);
        let (src_stage, src_access, dst_stage, dst_access) = transfer_to_vertex();
        batch.add_buffer_barrier(buf_id, buffer, src_stage, src_access, dst_stage, dst_access);

        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.image_barrier_count(), 1);
        assert_eq!(batch.buffer_barrier_count(), 1);
    }

    #[test]
    fn test_clear_all_barriers() {
        let mut batch = BarrierBatch::new();

        // Add image barrier
        let tex_id = TextureId::from_raw(11111);
        let image = vk::Image::from_raw(11111);
        batch.add_image_barrier(
            tex_id,
            image,
            TextureLayout::Undefined,
            TextureLayout::ColorAttachment,
            vk::ImageAspectFlags::COLOR,
        );

        // Add buffer barrier
        let buf_id = BufferId::from_raw(22222);
        let buffer = vk::Buffer::from_raw(22222);
        let (src_stage, src_access, dst_stage, dst_access) = transfer_to_vertex();
        batch.add_buffer_barrier(buf_id, buffer, src_stage, src_access, dst_stage, dst_access);

        assert_eq!(batch.len(), 2);

        // Clear all
        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.image_barrier_count(), 0);
        assert_eq!(batch.buffer_barrier_count(), 0);
    }
}
