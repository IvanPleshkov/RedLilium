//! Per-frame render-graph execution.
//!
//! Each frame builds and submits **one or more** render graphs, each as its
//! own queue submit on the single graphics queue. Ordering between graphs is
//! submission order: the backend's persistent barrier trackers emit
//! `vkCmdPipelineBarrier`s that synchronize across submits on one queue
//! exactly as they do across frames, so no cross-graph GPU semaphores are
//! needed (that changes with a second queue — see #47). Cross-frame CPU/GPU
//! overlap is provided by the frames-in-flight machinery in the pipeline.
//!
//! # Architecture
//!
//! `FrameSchedule` is the middle layer of the rendering architecture:
//!
//! | Layer | Type | Purpose |
//! |-------|------|---------|
//! | Pipeline | [`FramePipeline`](crate::pipeline::FramePipeline) | Multiple frames in flight |
//! | **Schedule** | [`FrameSchedule`] | Submits the frame's graphs (this module) |
//! | Graph | [`RenderGraph`](crate::graph::RenderGraph) | Passes + their dependencies |
//! | Pass | [`GraphicsPass`](crate::graph::GraphicsPass), etc. | Single GPU operation |
//!
//! For the full architecture documentation, see `docs/ARCHITECTURE.md`.
//!
//! # Module Contents
//!
//! - [`FrameSchedule`] - Submits the frame's render graphs
//! - [`SubmitHandle`] - Identifies one submitted graph within a frame
//! - [`Fence`] - CPU-GPU synchronization for frame completion
//!
//! # Example
//!
//! ```ignore
//! // FrameSchedule is created by FramePipeline::begin_frame()
//! let mut schedule = pipeline.begin_frame();
//!
//! // An independent offscreen pre-pass, submitted on its own...
//! let mut prepass = schedule.acquire_graph();
//! prepass.add_graphics_pass(shadow_pass);
//! schedule.submit(prepass);
//!
//! // ...then the main graph (at most one graph per frame may write the
//! // swapchain). Ordering is submission order; shared resources are
//! // synchronized automatically by the barrier trackers.
//! let mut main = schedule.acquire_graph();
//! main.add_graphics_pass(main_pass);
//! schedule.submit(main);
//!
//! pipeline.end_frame(schedule);
//! ```

mod sync;

pub use sync::{Fence, FenceStatus};

use std::sync::Arc;

use crate::device::GraphicsDevice;
use crate::graph::{RenderGraph, RenderGraphCompilationMode};
use crate::resources::{RingAllocation, RingBuffer};
use redlilium_core::profiling::profile_scope;

/// Identifies one graph submitted to a [`FrameSchedule`] within a frame.
///
/// Handles are only meaningful within the frame they were issued for; they
/// carry the submission index (0 for the first `submit` of the frame, 1 for
/// the second, ...). Ordering between submits is submission order — there is
/// no dependency API, and none is planned: cross-graph dependencies are
/// derived automatically from resource usage (#47).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubmitHandle {
    index: usize,
}

impl SubmitHandle {
    /// Zero-based submission index within the frame.
    pub fn index(&self) -> usize {
        self.index
    }
}

/// Frame schedule for streaming graph submission.
///
/// Allows submitting render graphs immediately as they're built,
/// rather than batching all submissions at frame end. This maximizes
/// CPU-GPU parallelism.
///
/// # Async Behavior
///
/// With GPU-backed fences, [`submit`](Self::submit) returns immediately
/// after handing the work to the GPU. The per-submit fences track when the
/// GPU actually completes, enabling true async rendering where the CPU can
/// build the next frame while the GPU renders the current one.
///
/// # Creation
///
/// `FrameSchedule` is created by [`FramePipeline::begin_frame`](crate::pipeline::FramePipeline::begin_frame).
/// Do not create it directly.
///
/// # Lifecycle
///
/// ```ignore
/// // Each frame:
/// let mut schedule = pipeline.begin_frame();
///
/// // Submit graphs as they're ready (ordering = submission order).
/// schedule.submit(prepass_graph);
/// schedule.submit(main_graph); // at most one graph writes the swapchain
///
/// // Return schedule to pipeline (stores fences for later waiting)
/// pipeline.end_frame(schedule);
/// ```
pub struct FrameSchedule {
    /// Device for executing the graphs.
    device: Arc<GraphicsDevice>,
    /// One fence per submit, signaled when that submit completes.
    fences: Vec<Fence>,
    /// The frame slot index (for per-frame resource management).
    frame_slot: usize,
    /// Ring buffer for this frame (if configured in FramePipeline).
    ring_buffer: Option<RingBuffer>,
    /// Pool of reusable render graphs (moved from FramePipeline each frame).
    graph_pool: Vec<RenderGraph>,
    /// The graphs executed this frame, kept for recycling in end_frame. Their
    /// `Arc` references keep GPU resources alive until the slot's fence wait.
    submitted_graphs: Vec<RenderGraph>,
    /// Whether a swapchain-writing graph has already been submitted this
    /// frame. The acquire/present semaphore pair exists once per frame, so a
    /// second swapchain writer would silently miss synchronization.
    swapchain_writer_submitted: bool,
}

impl std::fmt::Debug for FrameSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameSchedule")
            .field("device", &self.device.name())
            .field("frame_slot", &self.frame_slot)
            .field("fences", &self.fences)
            .finish()
    }
}

impl FrameSchedule {
    /// Create a new frame schedule.
    ///
    /// This is called internally by [`FramePipeline::begin_frame`](crate::pipeline::FramePipeline::begin_frame).
    pub(crate) fn new(
        device: Arc<GraphicsDevice>,
        frame_slot: usize,
        ring_buffer: Option<RingBuffer>,
        graph_pool: Vec<RenderGraph>,
    ) -> Self {
        Self {
            device,
            fences: Vec::new(),
            frame_slot,
            ring_buffer,
            graph_pool,
            submitted_graphs: Vec::new(),
            swapchain_writer_submitted: false,
        }
    }

    /// Get the frame slot index for this schedule.
    ///
    /// The slot index cycles from 0 to `frames_in_flight - 1`.
    pub fn frame_slot(&self) -> usize {
        self.frame_slot
    }

    /// Check if this schedule has a ring buffer configured.
    pub fn has_ring_buffer(&self) -> bool {
        self.ring_buffer.is_some()
    }

    /// Get read-only access to the ring buffer (if configured).
    pub fn ring_buffer(&self) -> Option<&RingBuffer> {
        self.ring_buffer.as_ref()
    }

    /// Get mutable access to the ring buffer (if configured).
    pub fn ring_buffer_mut(&mut self) -> Option<&mut RingBuffer> {
        self.ring_buffer.as_mut()
    }

    /// Allocate space from the ring buffer.
    ///
    /// Returns `None` if no ring buffer is configured or if there isn't
    /// enough space remaining.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the allocation in bytes
    pub fn allocate(&mut self, size: u64) -> Option<RingAllocation> {
        self.ring_buffer.as_mut()?.allocate(size)
    }

    /// Allocate space from the ring buffer with custom alignment.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the allocation in bytes
    /// * `alignment` - Required alignment (must be power of 2)
    pub fn allocate_aligned(&mut self, size: u64, alignment: u64) -> Option<RingAllocation> {
        self.ring_buffer.as_mut()?.allocate_aligned(size, alignment)
    }

    /// Take ownership of the ring buffer (called by FramePipeline::end_frame).
    pub(crate) fn take_ring_buffer(&mut self) -> Option<RingBuffer> {
        self.ring_buffer.take()
    }

    /// Acquire a render graph from the pool.
    ///
    /// Returns a graph from the pool if available, or creates a new one.
    /// The graph is cleared and ready for use.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut graph = schedule.acquire_graph();
    /// graph.add_graphics_pass(pass);
    /// let handle = schedule.submit("name", graph, &[]);
    /// ```
    pub fn acquire_graph(&mut self) -> RenderGraph {
        self.graph_pool.pop().unwrap_or_else(RenderGraph::new)
    }

    /// Take ownership of the graph pool (called by FramePipeline::end_frame).
    pub(crate) fn take_graph_pool(&mut self) -> Vec<RenderGraph> {
        std::mem::take(&mut self.graph_pool)
    }

    /// Take ownership of the submitted graphs (called by FramePipeline::end_frame).
    pub(crate) fn take_submitted_graphs(&mut self) -> Vec<RenderGraph> {
        std::mem::take(&mut self.submitted_graphs)
    }

    /// Submit a render graph for execution as its own queue submit.
    ///
    /// May be called any number of times per frame; graphs execute in
    /// submission order on the single graphics queue. Resources shared
    /// between graphs are synchronized automatically: the backend's
    /// persistent barrier trackers emit pipeline barriers that are valid
    /// across submits on one queue, exactly as across frames. There are no
    /// cross-graph GPU semaphores (that changes with a second queue, #47).
    ///
    /// A fence per submit is signalled on completion; the pipeline waits on
    /// all of them before recycling the slot. The graph is kept for
    /// recycling — its `Arc` references keep GPU resources alive until that
    /// wait.
    ///
    /// Takes ownership of the graph for pooling. Must be called at least once
    /// before [`FramePipeline::end_frame`](crate::pipeline::FramePipeline::end_frame).
    ///
    /// # Panics
    ///
    /// Panics if a swapchain-writing graph was already submitted this frame:
    /// the acquire/present semaphore pair exists once per frame, so a second
    /// swapchain writer would run unsynchronized against the presentation
    /// engine. Route all swapchain-writing passes into one graph.
    pub fn submit(&mut self, mut graph: RenderGraph) -> SubmitHandle {
        profile_scope!("submit_graph");

        // INVARIANT: all submits go to the single graphics queue, in
        // submission order. Cross-graph and cross-frame synchronization both
        // rest on the persistent barrier trackers, whose pipeline barriers
        // only work across submits *within one queue*. A second queue needs
        // semaphores + queue-ownership tracking first — see #47.
        if graph.writes_swapchain() {
            assert!(
                !self.swapchain_writer_submitted,
                "a swapchain-writing graph was already submitted this frame — the \
                 acquire/present handshake supports exactly one swapchain writer per frame \
                 (put all swapchain passes in one graph; see #47)"
            );
            self.swapchain_writer_submitted = true;
        }

        let handle = SubmitHandle {
            index: self.fences.len(),
        };

        // Fence signalled by this submit. Any failure below (fence creation,
        // compile, submit) means no GPU work is in flight for THIS submit, so
        // its fence must read as signaled — the CPU fence guarantees that
        // regardless of backend fence semantics. Earlier submits' fences are
        // unaffected.
        let mut fence = match Fence::new_gpu(Arc::clone(self.device.instance())) {
            Ok(fence) => fence,
            Err(e) => {
                log::error!("Failed to create submit fence, skipping submit: {e}");
                self.submitted_graphs.push(graph);
                self.fences.push(Fence::new_signaled());
                return handle;
            }
        };

        #[cfg(debug_assertions)]
        debug_assert_no_write_to_mapped(&graph);

        #[cfg(debug_assertions)]
        debug_assert_pipeline_state_matches_targets(&graph);

        match graph.compile(RenderGraphCompilationMode::Strict) {
            Ok(_) => {
                profile_scope!("execute_graph");
                let compiled = graph.compiled().unwrap();
                let backend = self.device.instance().backend();
                if let Err(e) = backend.execute_graph(&graph, compiled, fence.gpu_fence()) {
                    log::error!("Failed to execute frame graph: {e}");
                    fence = Fence::new_signaled();
                }
            }
            Err(e) => {
                log::error!("Failed to compile frame graph: {e}");
                fence = Fence::new_signaled();
            }
        }

        // Keep the graph for recycling at end of frame.
        self.submitted_graphs.push(graph);
        self.fences.push(fence);
        handle
    }

    /// Submit the graph for execution — equivalent to [`submit`](Self::submit).
    ///
    /// Retained for callers from the one-graph-per-frame era; new code should
    /// call `submit` directly.
    pub fn render(&mut self, graph: RenderGraph) {
        self.submit(graph);
    }

    /// Extract the per-submit fences from this schedule.
    ///
    /// This is called internally by [`FramePipeline::end_frame`](crate::pipeline::FramePipeline::end_frame).
    ///
    /// # Panics
    ///
    /// Panics if [`submit`](Self::submit) was never called.
    pub(crate) fn take_fences(&mut self) -> Vec<Fence> {
        assert!(
            !self.fences.is_empty(),
            "submit() must be called at least once before end_frame()"
        );
        std::mem::take(&mut self.fences)
    }
}

/// Debug-only guard (#33): a transfer must not write into a buffer whose async
/// readback `map_async` is still in flight — that is UB / a wgpu validation
/// error ("buffer already mapped").
///
/// Async maps are only ever issued in the frame pipeline's `process_readbacks`
/// (during `begin_frame`), never mid-encode, so at execute time the buffer's
/// `is_map_pending` flag deterministically reflects any map still outstanding
/// from an earlier frame — exactly the hazard we want to catch here, before the
/// backend encodes the copy.
///
/// The flag may clear between this check and actual GPU execution (the map
/// completion callback runs on device poll), so a firing assert is *slightly*
/// conservative — a rare false positive is acceptable for a dev-only guard; do
/// not weaken it to silence such a case. A consumer that trips this every frame
/// is doing a per-frame readback into a single buffer and genuinely needs
/// double-buffering.
#[cfg(debug_assertions)]
fn debug_assert_no_write_to_mapped(graph: &RenderGraph) {
    use crate::graph::TransferOperation;
    for pass in graph.passes() {
        let Some(transfer) = pass.as_transfer() else {
            continue;
        };
        let Some(config) = transfer.transfer_config() else {
            continue;
        };
        for op in &config.operations {
            let dst = match op {
                TransferOperation::BufferToBuffer { dst, .. }
                | TransferOperation::WriteBuffer { dst, .. }
                | TransferOperation::TextureToBuffer { dst, .. } => dst,
                _ => continue,
            };
            debug_assert!(
                !dst.is_map_pending(),
                "transfer writes into buffer {:?} while an async readback map of it is still \
                 in flight (write-while-mapped, #33) — this consumer must wait the map out or \
                 double-buffer the readback target",
                dst.label()
            );
        }
    }
}

/// Debug-only guard (#39): a draw's material pipeline must agree with the pass's
/// render-target attachments on MSAA sample count and depth-attachment presence.
///
/// These are two independent sources of truth now that `MaterialDescriptor`
/// carries `sample_count` and `Option<DepthState>` (XB-L6): a mismatch is a
/// validation error on wgpu and undefined-to-fatal on raw Vulkan. All attachments
/// in a pass share one sample count, so it is read from the first color
/// attachment (surface swapchains are single-sample), falling back to depth.
#[cfg(debug_assertions)]
fn debug_assert_pipeline_state_matches_targets(graph: &RenderGraph) {
    use crate::graph::RenderTarget;

    fn target_samples(t: &RenderTarget) -> u32 {
        match t {
            RenderTarget::Texture { texture, .. } => texture.sample_count(),
            RenderTarget::Surface { .. } => 1,
        }
    }

    for pass in graph.passes() {
        let Some(gp) = pass.as_graphics() else {
            continue;
        };
        let Some(targets) = gp.render_targets() else {
            continue;
        };

        let pass_samples = targets
            .color_attachments
            .first()
            .map(|c| target_samples(&c.target))
            .or_else(|| {
                targets
                    .depth_stencil_attachment
                    .as_ref()
                    .map(|d| target_samples(&d.target))
            })
            .unwrap_or(1);
        let pass_has_depth = targets.depth_stencil_attachment.is_some();

        for cmd in gp.draw_commands() {
            let desc = cmd.material.material().descriptor();
            let label = desc.label.as_deref().unwrap_or("<unlabeled>");
            debug_assert_eq!(
                desc.sample_count,
                pass_samples,
                "pass '{}': material '{label}' has sample_count {} but the target attachments \
                 are {}-sample (#39, XB-L6)",
                gp.name(),
                desc.sample_count,
                pass_samples,
            );
            debug_assert_eq!(
                desc.depth.is_some(),
                pass_has_depth,
                "pass '{}': material '{label}' depth state ({}) disagrees with the pass depth \
                 attachment ({}) — a pipeline must have a depth-stencil state iff its pass has a \
                 depth attachment (#39)",
                gp.name(),
                if desc.depth.is_some() { "Some" } else { "None" },
                if pass_has_depth { "present" } else { "absent" },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphicsPass, RenderGraph};
    use crate::instance::GraphicsInstance;

    fn make_test_graph(name: &str) -> RenderGraph {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(GraphicsPass::new(name.into()));
        graph
    }

    fn make_test_schedule() -> FrameSchedule {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        FrameSchedule::new(device, 0, None, Vec::new())
    }

    /// A graph whose single pass renders to the swapchain surface.
    fn make_surface_graph(name: &str) -> RenderGraph {
        use crate::graph::{ColorAttachment, RenderTarget, RenderTargetConfig};
        use crate::types::TextureFormat;

        let target = RenderTarget::Surface {
            format: TextureFormat::Bgra8UnormSrgb,
            width: 4,
            height: 4,
            #[cfg(feature = "wgpu-backend")]
            view: None,
            #[cfg(feature = "vulkan-backend")]
            vulkan_view: None,
        };
        let mut pass = GraphicsPass::new(name.into());
        pass.set_render_targets(RenderTargetConfig::new().with_color(ColorAttachment::new(target)));
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(pass);
        graph
    }

    #[test]
    fn submit_signals_fence() {
        let mut schedule = make_test_schedule();
        let handle = schedule.submit(make_test_graph("main"));
        assert_eq!(handle.index(), 0);

        let fences = schedule.take_fences();
        assert_eq!(fences.len(), 1);
        fences[0].wait().unwrap();
        assert_eq!(fences[0].status(), FenceStatus::Signaled);
    }

    #[test]
    fn submit_multi_pass_single_graph() {
        // One graph may carry many passes; ordering/barriers within it are
        // the compiler's job. Just verify it executes and signals.
        let mut schedule = make_test_schedule();
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(GraphicsPass::new("shadow".into()));
        graph.add_graphics_pass(GraphicsPass::new("main".into()));
        schedule.submit(graph);

        let fences = schedule.take_fences();
        assert_eq!(fences.len(), 1);
        fences[0].wait().unwrap();
    }

    #[test]
    fn submit_multiple_graphs_signals_all_fences() {
        // Multiple graphs per frame, each its own submit on the single queue.
        // Ordering is submission order; every submit gets its own fence.
        let mut schedule = make_test_schedule();
        let a = schedule.submit(make_test_graph("prepass"));
        let b = schedule.submit(make_test_graph("main"));
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);

        let fences = schedule.take_fences();
        assert_eq!(fences.len(), 2);
        for fence in &fences {
            fence.wait().unwrap();
            assert_eq!(fence.status(), FenceStatus::Signaled);
        }
    }

    #[test]
    fn render_is_submit_alias() {
        // render() survives as a thin wrapper; calling it twice is now two
        // submits, not a panic.
        let mut schedule = make_test_schedule();
        schedule.render(make_test_graph("a"));
        schedule.render(make_test_graph("b"));

        assert_eq!(schedule.take_fences().len(), 2);
    }

    #[test]
    fn one_swapchain_writer_is_accepted() {
        let mut schedule = make_test_schedule();
        schedule.submit(make_test_graph("offscreen"));
        schedule.submit(make_surface_graph("present"));

        assert_eq!(schedule.take_fences().len(), 2);
    }

    #[test]
    #[should_panic(expected = "swapchain-writing graph was already submitted")]
    fn second_swapchain_writer_panics() {
        let mut schedule = make_test_schedule();
        schedule.submit(make_surface_graph("present_a"));
        schedule.submit(make_surface_graph("present_b")); // Panics
    }

    #[test]
    #[should_panic(expected = "submit() must be called at least once before end_frame()")]
    fn take_fences_without_submit_panics() {
        let mut schedule = make_test_schedule();
        schedule.take_fences(); // Panics
    }
}
