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

/// Identifies a queue for cross-queue synchronization tracking (#47 phase 4).
///
/// Also used as an index into per-queue tracker arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueId {
    /// The graphics queue (always present).
    Graphics = 0,
    /// The async compute queue (present only when the device exposes one).
    AsyncCompute = 1,
}

/// Number of queues a resource can be tracked across.
pub const QUEUE_COUNT: usize = 2;

/// Timeline-semaphore waits a submit must perform before it executes,
/// accumulated by the trackers while barriers are generated.
///
/// At most one wait per source queue: timeline waits are monotonic, so the
/// maximum value subsumes all smaller ones.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SubmitWaits {
    values: [Option<u64>; QUEUE_COUNT],
}

impl SubmitWaits {
    /// Require this submit to wait until `queue`'s timeline reaches `value`.
    pub fn add(&mut self, queue: QueueId, value: u64) {
        let slot = &mut self.values[queue as usize];
        *slot = Some(slot.map_or(value, |v| v.max(value)));
    }

    /// The wait value required on `queue`'s timeline, if any.
    pub fn get(&self, queue: QueueId) -> Option<u64> {
        self.values[queue as usize]
    }

    /// Whether no waits were accumulated.
    pub fn is_empty(&self) -> bool {
        self.values.iter().all(Option::is_none)
    }
}

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
///
/// # Queue ownership (#47 phase 4)
///
/// The write additionally records which queue performed it and that submit's
/// timeline value, and reads record the highest submit value per queue. An
/// access from a different queue than the last writer emits a **submit-level
/// timeline wait** instead of a pipeline barrier (which cannot cross queues):
/// the semaphore wait, performed with an all-commands destination scope,
/// makes the write available *and* visible on the waiting queue, so no
/// additional barrier is needed there. The stage/access/visibility fields
/// remain meaningful only on the writing queue itself.
#[derive(Debug, Default, Clone, Copy)]
struct BufferAccessState {
    /// Pipeline stages of the last tracked write.
    last_write_stage: vk::PipelineStageFlags,
    /// Access mask of the last tracked write.
    last_write_access: vk::AccessFlags,
    /// Queue and submit timeline value of the last tracked write.
    last_write_submit: Option<(QueueId, u64)>,
    /// Read stages the last write has been made visible to (via a barrier)
    /// **on the writing queue**. On other queues visibility comes from the
    /// semaphore wait itself.
    visible_stages: vk::PipelineStageFlags,
    /// Read access types the last write has been made visible to.
    visible_access: vk::AccessFlags,
    /// All read stages seen since the last write **on the writing queue**
    /// (for same-queue write-after-read execution dependencies). Cross-queue
    /// reads are recorded in `read_submits` instead.
    reads_since_write: vk::PipelineStageFlags,
    /// Highest submit timeline value that read this buffer since the last
    /// write, per queue (write-after-read across queues waits on these).
    read_submits: [Option<u64>; QUEUE_COUNT],
}

/// Tracks the last access to every buffer used by render graphs, mirroring
/// [`super::layout::TextureLayoutTracker`] for buffers.
///
/// The tracker is global and persists across frames.
///
/// INVARIANT (queue ownership): a `vkCmdPipelineBarrier` synchronizes across
/// submissions only *within a queue*, so a write recorded in one submit is a
/// valid source scope for a later access only when both submits are on the
/// same queue in submission order. The tracker therefore records which queue
/// performed each write (and read) and resolves cross-queue hazards with
/// submit-level timeline waits instead of barriers — see
/// [`request_access`](BufferAccessTracker::request_access) and #47. Every
/// access MUST be reported with the correct `QueueId`; reporting an async
/// submit as graphics (or vice versa) silently drops synchronization.
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
    /// `queue` and `submit_value` identify the submit performing the access;
    /// cross-queue hazards are pushed into `waits` as timeline waits instead
    /// of barriers (see [`BufferAccessState`]).
    ///
    /// Returns `Some((src_stage, src_access))` when a pipeline barrier must be
    /// placed before the pass performing `access`, or `None` when the access
    /// is already synchronized (first GPU use, host-written buffers, a read
    /// scope the last write is already visible to, or a hazard covered by an
    /// emitted cross-queue wait).
    pub fn request_access(
        &mut self,
        id: BufferId,
        access: BufferAccessMode,
        queue: QueueId,
        submit_value: u64,
        waits: &mut SubmitWaits,
    ) -> Option<(vk::PipelineStageFlags, vk::AccessFlags)> {
        let state = self.states.entry(id).or_default();
        let stage = access.dst_stage();
        let access_mask = access.dst_access_mask();

        // Whether the last write came from another queue: its hazard is
        // resolved by a timeline wait, and its stage/access scopes are
        // meaningless on this queue.
        let cross_queue_write = match state.last_write_submit {
            Some((write_queue, write_value)) if write_queue != queue => {
                waits.add(write_queue, write_value);
                true
            }
            _ => false,
        };

        if access.is_write() {
            // Cross-queue reads since the last write: WAR resolved by waits.
            for (queue_index, read) in state.read_submits.iter().enumerate() {
                if let Some(read_value) = read
                    && queue_index != queue as usize
                {
                    let read_queue = if queue_index == QueueId::Graphics as usize {
                        QueueId::Graphics
                    } else {
                        QueueId::AsyncCompute
                    };
                    waits.add(read_queue, *read_value);
                }
            }

            // Same-queue WAW/WAR: barrier from the previous write's scope and
            // the reads issued since it (execution dependency only, so reads
            // contribute no source access mask). Skipped when the previous
            // write was cross-queue — the wait already covers it, and its
            // scopes belong to the other queue.
            let (src_stage, src_access) = if cross_queue_write {
                (vk::PipelineStageFlags::empty(), vk::AccessFlags::empty())
            } else {
                (
                    state.last_write_stage | state.reads_since_write,
                    state.last_write_access,
                )
            };

            *state = BufferAccessState {
                last_write_stage: stage,
                last_write_access: access_mask,
                last_write_submit: Some((queue, submit_value)),
                ..Default::default()
            };

            (!src_stage.is_empty()).then_some((src_stage, src_access))
        } else {
            state.read_submits[queue as usize] = Some(
                state.read_submits[queue as usize].map_or(submit_value, |v| v.max(submit_value)),
            );

            if cross_queue_write {
                // The timeline wait (all-commands scope) makes the write both
                // available and visible on this queue: no barrier. Same-queue
                // WAR tracking (`reads_since_write`) stays untouched — it
                // belongs to the writing queue.
                return None;
            }

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
        self.add_image_barrier_with_src_scope(
            id,
            image,
            old_layout,
            new_layout,
            aspect_mask,
            old_layout.src_stage(),
            old_layout.src_access_mask(),
        );
    }

    /// Add an image layout transition whose previous access ran on a
    /// DIFFERENT queue and is ordered by a tracker-emitted timeline wait
    /// (#47 phase 4).
    ///
    /// The wait (all-commands destination scope) already made the other
    /// queue's access available and visible here, so the transition's source
    /// scope is empty on this queue: `TOP_OF_PIPE` and no access mask. Using
    /// the previous layout's own scopes instead would name stages of the
    /// OTHER queue, which may not exist on this queue's family — e.g.
    /// `COLOR_ATTACHMENT_OUTPUT` on a dedicated compute family
    /// (VUID-vkCmdPipelineBarrier-srcStageMask-06461).
    pub fn add_image_barrier_cross_queue(
        &mut self,
        id: TextureId,
        image: vk::Image,
        old_layout: TextureLayout,
        new_layout: TextureLayout,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        self.add_image_barrier_with_src_scope(
            id,
            image,
            old_layout,
            new_layout,
            aspect_mask,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn add_image_barrier_with_src_scope(
        &mut self,
        id: TextureId,
        image: vk::Image,
        old_layout: TextureLayout,
        new_layout: TextureLayout,
        aspect_mask: vk::ImageAspectFlags,
        src_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
    ) {
        // Same read-only layout: no hazard, nothing to do.
        if old_layout == new_layout && !new_layout.is_write() {
            return;
        }

        match self.image_barriers.entry(id) {
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                let info = occupied.get_mut();
                info.new_layout = new_layout.to_vk();
                info.src_access_mask |= src_access;
                info.dst_access_mask |= new_layout.dst_access_mask();
            }
            std::collections::hash_map::Entry::Vacant(vacant) => {
                vacant.insert(ImageBarrierInfo {
                    image,
                    old_layout: old_layout.to_vk(),
                    new_layout: new_layout.to_vk(),
                    src_access_mask: src_access,
                    dst_access_mask: new_layout.dst_access_mask(),
                    aspect_mask,
                });
            }
        }
        self.src_stage_mask |= src_stage;
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

    /// Same-queue access on the graphics queue; asserts no cross-queue wait
    /// is emitted (single-queue behavior must be byte-for-byte what it was
    /// before queue ownership existed).
    fn gfx_access(
        tracker: &mut BufferAccessTracker,
        id: BufferId,
        access: BufferAccessMode,
    ) -> Option<(vk::PipelineStageFlags, vk::AccessFlags)> {
        let mut waits = SubmitWaits::default();
        let result = tracker.request_access(id, access, QueueId::Graphics, 1, &mut waits);
        assert!(
            waits.is_empty(),
            "same-queue access must not emit cross-queue waits"
        );
        result
    }

    #[test]
    fn tracker_first_read_needs_no_barrier() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        // Host-written (or never-written) buffer: reads need no barrier.
        assert_eq!(
            gfx_access(&mut tracker, id, BufferAccessMode::VertexBuffer),
            None
        );
        assert_eq!(
            gfx_access(&mut tracker, id, BufferAccessMode::UniformRead),
            None
        );
    }

    #[test]
    fn tracker_compute_write_then_read_uses_shader_src_scope() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        // First write: nothing earlier to synchronize against.
        assert_eq!(
            gfx_access(&mut tracker, id, BufferAccessMode::StorageReadWrite),
            None
        );

        // Read after the compute write: src scope must be the shader write —
        // the old hardcoded TRANSFER source would miss it entirely (VK-C3).
        let (src_stage, src_access) = gfx_access(&mut tracker, id, BufferAccessMode::VertexBuffer)
            .expect("read after write needs a barrier");
        assert!(src_stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(src_access.contains(vk::AccessFlags::SHADER_WRITE));
    }

    #[test]
    fn tracker_repeated_read_is_skipped() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        gfx_access(&mut tracker, id, BufferAccessMode::TransferWrite);
        assert!(gfx_access(&mut tracker, id, BufferAccessMode::VertexBuffer).is_some());
        // Same read scope again (next pass, next frame): already visible,
        // no redundant barrier (VK-P7).
        assert_eq!(
            gfx_access(&mut tracker, id, BufferAccessMode::VertexBuffer),
            None
        );
        // A *different* read scope still needs its own visibility.
        assert!(gfx_access(&mut tracker, id, BufferAccessMode::IndirectRead).is_some());
    }

    #[test]
    fn tracker_write_after_read_has_execution_dependency() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        gfx_access(&mut tracker, id, BufferAccessMode::VertexBuffer);
        // Write after read: needs an execution dependency on the read stage,
        // with no source access (reads make nothing available).
        let (src_stage, src_access) = gfx_access(&mut tracker, id, BufferAccessMode::TransferWrite)
            .expect("write after read needs a barrier");
        assert!(src_stage.contains(vk::PipelineStageFlags::VERTEX_INPUT));
        assert_eq!(src_access, vk::AccessFlags::empty());
    }

    #[test]
    fn tracker_write_after_write_chains_scopes() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        gfx_access(&mut tracker, id, BufferAccessMode::TransferWrite);
        let (src_stage, src_access) = gfx_access(&mut tracker, id, BufferAccessMode::StorageWrite)
            .expect("write after write needs a barrier");
        assert!(src_stage.contains(vk::PipelineStageFlags::TRANSFER));
        assert!(src_access.contains(vk::AccessFlags::TRANSFER_WRITE));

        // And a read now synchronizes against the SECOND write.
        let (src_stage, src_access) = gfx_access(&mut tracker, id, BufferAccessMode::UniformRead)
            .expect("read after write needs a barrier");
        assert!(src_stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(src_access.contains(vk::AccessFlags::SHADER_WRITE));
    }

    // Cross-queue tracker tests (#47 phase 4)

    #[test]
    fn tracker_cross_queue_read_waits_instead_of_barrier() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        // Async compute writes at timeline value 5.
        let mut waits = SubmitWaits::default();
        tracker.request_access(
            id,
            BufferAccessMode::StorageWrite,
            QueueId::AsyncCompute,
            5,
            &mut waits,
        );
        assert!(waits.is_empty());

        // Graphics reads: no pipeline barrier (the timeline wait covers both
        // availability and visibility), but a wait on compute's value 5.
        let mut waits = SubmitWaits::default();
        let barrier = tracker.request_access(
            id,
            BufferAccessMode::VertexBuffer,
            QueueId::Graphics,
            9,
            &mut waits,
        );
        assert_eq!(barrier, None);
        assert_eq!(waits.get(QueueId::AsyncCompute), Some(5));
        assert_eq!(waits.get(QueueId::Graphics), None);
    }

    #[test]
    fn tracker_cross_queue_write_after_read_waits_on_readers() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);

        // Graphics reads the (host-written) buffer in submits 3 and 7.
        let mut waits = SubmitWaits::default();
        tracker.request_access(
            id,
            BufferAccessMode::VertexBuffer,
            QueueId::Graphics,
            3,
            &mut waits,
        );
        tracker.request_access(
            id,
            BufferAccessMode::UniformRead,
            QueueId::Graphics,
            7,
            &mut waits,
        );
        assert!(waits.is_empty());

        // Async compute writes: WAR across queues — must wait graphics
        // reaching the highest reading submit (7).
        let mut waits = SubmitWaits::default();
        tracker.request_access(
            id,
            BufferAccessMode::StorageWrite,
            QueueId::AsyncCompute,
            2,
            &mut waits,
        );
        assert_eq!(waits.get(QueueId::Graphics), Some(7));
    }

    #[test]
    fn tracker_same_queue_after_cross_queue_write_still_gets_barrier() {
        let mut tracker = BufferAccessTracker::new();
        let id = BufferId::from_raw(1);
        let mut waits = SubmitWaits::default();

        // Graphics writes (value 1), async compute overwrites (value 2, gets
        // a wait), then async compute reads its OWN write: that read needs a
        // normal same-queue barrier — the earlier cross-queue wait must not
        // have poisoned same-queue visibility tracking.
        tracker.request_access(
            id,
            BufferAccessMode::TransferWrite,
            QueueId::Graphics,
            1,
            &mut waits,
        );
        let mut waits = SubmitWaits::default();
        let barrier = tracker.request_access(
            id,
            BufferAccessMode::StorageWrite,
            QueueId::AsyncCompute,
            2,
            &mut waits,
        );
        // Cross-queue WAW: wait emitted, no barrier (scopes belong to the
        // graphics queue).
        assert_eq!(waits.get(QueueId::Graphics), Some(1));
        assert_eq!(barrier, None);

        let mut waits = SubmitWaits::default();
        let barrier = tracker
            .request_access(
                id,
                BufferAccessMode::UniformRead,
                QueueId::AsyncCompute,
                3,
                &mut waits,
            )
            .expect("same-queue read after write needs a barrier");
        assert!(waits.is_empty());
        assert!(barrier.0.contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(barrier.1.contains(vk::AccessFlags::SHADER_WRITE));
    }

    #[test]
    fn submit_waits_keep_max_value_per_queue() {
        let mut waits = SubmitWaits::default();
        assert!(waits.is_empty());
        waits.add(QueueId::Graphics, 4);
        waits.add(QueueId::Graphics, 9);
        waits.add(QueueId::Graphics, 6);
        assert_eq!(waits.get(QueueId::Graphics), Some(9));
        assert_eq!(waits.get(QueueId::AsyncCompute), None);
        assert!(!waits.is_empty());
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
