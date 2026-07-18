//! Per-frame render-graph execution.
//!
//! Each frame builds and submits **one or more** render graphs, each as its
//! own queue submit. Graphs run on the graphics queue unless they opt into
//! secondary-queue routing ([`RenderGraph::set_queue_preference`] — async
//! compute #47, dedicated transfer #89) — an explicit placement hint,
//! honored only for compute/transfer-only graphs on devices that expose the
//! queue; otherwise everything runs on the graphics queue, the first-class
//! fallback. Ordering is always automatic:
//! same-queue hazards become pipeline barriers from the backend's persistent
//! trackers (valid across submits within one queue, exactly as across
//! frames), cross-queue hazards become timeline-semaphore waits emitted at
//! submit (#47 phase 4). Cross-frame CPU/GPU overlap is provided by the
//! frames-in-flight machinery in the pipeline.
//!
//! Dependency edges between the frame's graphs are additionally **derived
//! automatically** at graph granularity from their declared resource usage
//! (the `deps` module, #47 phase 3) and exposed for diagnostics via
//! [`FrameSchedule::derived_dependencies`]. There is deliberately no manual
//! dependency API.
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

mod deps;
mod sync;

use deps::GraphUsage;
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
    /// Aggregated resource usage of each submitted graph, index-aligned with
    /// `submitted_graphs`/`fences` (empty usage for graphs that failed to
    /// compile). Source data for cross-graph dependency derivation.
    submitted_usages: Vec<GraphUsage>,
    /// Dependency edges `(from, to)` derived from overlapping resource usage:
    /// the `to` submit accesses a resource the earlier `from` submit wrote
    /// (or writes one it read). On the single graphics queue these are
    /// satisfied by submission order and change nothing at runtime; phase 4
    /// of #47 turns cross-queue edges into timeline-semaphore waits.
    derived_edges: Vec<(SubmitHandle, SubmitHandle)>,
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
            submitted_usages: Vec::new(),
            derived_edges: Vec::new(),
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

        // Ordering: same-queue submits execute in submission order and are
        // synchronized by the trackers' pipeline barriers; a graph routed to
        // the async compute queue (an explicit opt-in hint, honored only for
        // compute/transfer-only graphs on devices that have the queue) is
        // synchronized by tracker-emitted timeline waits (#47 phase 4).
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

        // Aggregated usage of this graph, for cross-graph dependency
        // derivation (empty if compilation fails — no GPU work, no edges).
        let mut usage = GraphUsage::default();

        // Strict compilation surfaces ambiguous WAW pairs (two writers, no
        // explicit edge, no derivable order). Dropping the graph over that
        // renders NOTHING with one log line to explain it — brutal to debug
        // (#141). Instead fall back to Automatic resolution (addition order,
        // which is what the graph author almost always meant) and say so
        // loudly with pass names. The ERROR repeats every submit on purpose:
        // ambiguity is a graph-construction bug to fix with an explicit
        // edge, not a supported steady state.
        let compiled_ok = match graph.compile(RenderGraphCompilationMode::Strict) {
            Ok(_) => true,
            Err(e @ crate::compiler::GraphError::AmbiguousOrder { .. }) => {
                log::error!("frame graph: {e}; falling back to addition-order resolution (#141)");
                match graph.compile(RenderGraphCompilationMode::Automatic) {
                    Ok(_) => true,
                    Err(e) => {
                        log::error!("Failed to compile frame graph: {e}");
                        false
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to compile frame graph: {e}");
                false
            }
        };
        if compiled_ok {
            profile_scope!("execute_graph");
            let compiled = graph.compiled().unwrap();
            usage = GraphUsage::from_compiled(compiled);
            let backend = self.device.instance().backend();
            if let Err(e) = backend.execute_graph(&graph, compiled, fence.gpu_fence()) {
                log::error!("Failed to execute frame graph: {e}");
                fence = Fence::new_signaled();
            }
        } else {
            fence = Fence::new_signaled();
        }

        // Derive dependency edges against every earlier submit of this frame
        // (diagnostics/tests). Correctness does not rest on these: same-queue
        // edges are satisfied by submission order, and cross-queue hazards
        // are resolved by the backend trackers' timeline waits — which also
        // cover cross-FRAME hazards this per-frame view cannot see.
        for (index, prev) in self.submitted_usages.iter().enumerate() {
            if prev.conflicts_with(&usage) {
                self.derived_edges.push((SubmitHandle { index }, handle));
            }
        }
        self.submitted_usages.push(usage);

        // Keep the graph for recycling at end of frame.
        self.submitted_graphs.push(graph);
        self.fences.push(fence);
        handle
    }

    /// Dependency edges `(from, to)` derived this frame from overlapping
    /// resource usage: the `to` submit reads or overwrites something the
    /// earlier `from` submit wrote (or writes something it read).
    ///
    /// Derivation is fully automatic — there is no way to declare an edge by
    /// hand (#47 design decision). The edges are observability for tests and
    /// diagnostics: same-queue edges are satisfied by submission order, and
    /// cross-queue hazards are enforced by the backend trackers' timeline
    /// waits (which additionally cover cross-frame hazards outside this
    /// per-frame view).
    pub fn derived_dependencies(&self) -> &[(SubmitHandle, SubmitHandle)] {
        &self.derived_edges
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
            debug_assert_eq!(
                desc.color_formats.len(),
                targets.color_attachments.len(),
                "pass '{}': material '{label}' declares {} color format(s) but the pass has {} \
                 color attachment(s) — counts must match (zero-color depth-only passes need \
                 zero-color materials, #129)",
                gp.name(),
                desc.color_formats.len(),
                targets.color_attachments.len(),
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

    /// A graph with a single whole-buffer copy pass (`src` read, `dst` write).
    fn make_copy_graph(
        name: &str,
        src: std::sync::Arc<crate::resources::Buffer>,
        dst: std::sync::Arc<crate::resources::Buffer>,
    ) -> RenderGraph {
        use crate::graph::{TransferConfig, TransferOperation, TransferPass};

        let mut pass = TransferPass::new(name.into());
        pass.set_transfer_config(
            TransferConfig::new().with_operation(TransferOperation::copy_buffer_whole(src, dst)),
        );
        let mut graph = RenderGraph::new();
        graph.add_transfer_pass(pass);
        graph
    }

    fn make_test_buffer(
        device: &Arc<crate::device::GraphicsDevice>,
    ) -> std::sync::Arc<crate::resources::Buffer> {
        use crate::types::{BufferDescriptor, BufferUsage};
        device
            .create_buffer(&BufferDescriptor::new(
                256,
                BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            ))
            .unwrap()
    }

    #[test]
    fn derives_edge_from_cross_graph_hazard() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let mut schedule = FrameSchedule::new(Arc::clone(&device), 0, None, Vec::new());

        let x = make_test_buffer(&device);
        let y = make_test_buffer(&device);
        let z = make_test_buffer(&device);

        // Graph A writes Y (copy X -> Y); graph B reads Y (copy Y -> Z): RAW.
        let a = schedule.submit(make_copy_graph("a", x, Arc::clone(&y)));
        let b = schedule.submit(make_copy_graph("b", y, z));

        assert_eq!(schedule.derived_dependencies(), &[(a, b)]);
        schedule.take_fences();
    }

    #[test]
    fn no_edge_between_disjoint_graphs() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let mut schedule = FrameSchedule::new(Arc::clone(&device), 0, None, Vec::new());

        let a_src = make_test_buffer(&device);
        let a_dst = make_test_buffer(&device);
        let b_src = make_test_buffer(&device);
        let b_dst = make_test_buffer(&device);

        schedule.submit(make_copy_graph("a", a_src, a_dst));
        schedule.submit(make_copy_graph("b", b_src, b_dst));

        assert!(schedule.derived_dependencies().is_empty());
        schedule.take_fences();
    }

    #[test]
    fn shared_read_only_source_derives_no_edge() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let mut schedule = FrameSchedule::new(Arc::clone(&device), 0, None, Vec::new());

        // Both graphs read the same source (read-after-read): no hazard.
        let src = make_test_buffer(&device);
        let a_dst = make_test_buffer(&device);
        let b_dst = make_test_buffer(&device);

        schedule.submit(make_copy_graph("a", Arc::clone(&src), a_dst));
        schedule.submit(make_copy_graph("b", src, b_dst));

        assert!(schedule.derived_dependencies().is_empty());
        schedule.take_fences();
    }

    #[test]
    fn edges_derive_against_every_earlier_conflicting_submit() {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        let mut schedule = FrameSchedule::new(Arc::clone(&device), 0, None, Vec::new());

        let x = make_test_buffer(&device);
        let y = make_test_buffer(&device);
        let z = make_test_buffer(&device);

        // A writes Y; B writes Y (WAW with A); C reads Y (RAW with A and B).
        let a = schedule.submit(make_copy_graph("a", Arc::clone(&x), Arc::clone(&y)));
        let b = schedule.submit(make_copy_graph("b", x, Arc::clone(&y)));
        let c = schedule.submit(make_copy_graph("c", y, z));

        assert_eq!(schedule.derived_dependencies(), &[(a, b), (a, c), (b, c)]);
        schedule.take_fences();
    }
}
