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
//! let mut schedule = pipeline.begin_frame()?;
//!
//! // An independent offscreen pre-pass, submitted on its own...
//! let mut prepass = schedule.acquire_graph();
//! prepass.add_graphics_pass(shadow_pass);
//! schedule.submit(prepass)?;
//!
//! // ...then the main graph (at most one graph per frame may write the
//! // swapchain). Ordering is submission order; shared resources are
//! // synchronized automatically by the barrier trackers.
//! let mut main = schedule.acquire_graph();
//! main.add_graphics_pass(main_pass);
//! schedule.submit(main)?;
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
/// let mut schedule = pipeline.begin_frame()?;
///
/// // Submit graphs as they're ready (ordering = submission order).
/// schedule.submit(prepass_graph)?;
/// schedule.submit(main_graph)?; // at most one graph writes the swapchain
///
/// // Return schedule to pipeline (stores fences for later waiting)
/// pipeline.end_frame(schedule);
/// ```
pub struct FrameSchedule {
    pub(crate) owner: Option<Arc<crate::pipeline::FrameOwner>>,
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
            owner: None,
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

    pub(crate) fn take_resources(&mut self) -> crate::pipeline::FrameResources {
        crate::pipeline::FrameResources {
            fences: std::mem::take(&mut self.fences),
            graphs: std::mem::take(&mut self.submitted_graphs),
            pool: std::mem::take(&mut self.graph_pool),
            ring: self.ring_buffer.take(),
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
    /// let handle = schedule.submit(graph)?;
    /// ```
    pub fn acquire_graph(&mut self) -> RenderGraph {
        self.graph_pool.pop().unwrap_or_else(RenderGraph::new)
    }

    /// Submit a render graph for execution as its own queue submit.
    ///
    /// May be called any number of times per frame. Shared resources are
    /// synchronized automatically: pipeline barriers order same-queue uses;
    /// timeline semaphore waits order graphs routed to different queues.
    ///
    /// A fence per submit is signalled on completion; the pipeline waits on
    /// all of them before recycling the slot. The graph is kept for
    /// recycling — its `Arc` references keep GPU resources alive until that
    /// wait.
    ///
    /// Takes ownership of the graph for pooling. Automatic compilation orders
    /// ambiguous writers by addition order. Use `submit_with_mode` for Strict.
    ///
    /// # Errors
    ///
    /// Returns an error if a swapchain-writing graph was already submitted this frame:
    /// the acquire/present semaphore pair exists once per frame, so a second
    /// swapchain writer would run unsynchronized against the presentation
    /// engine. Route all swapchain-writing passes into one graph.
    pub fn submit(&mut self, graph: RenderGraph) -> Result<SubmitHandle, crate::GraphicsError> {
        self.submit_with_mode(graph, RenderGraphCompilationMode::Automatic)
    }

    /// Submit with an explicit conflict-resolution policy. `Strict` reports
    /// ambiguous writers as an error; it never falls back to addition order.
    pub fn submit_with_mode(
        &mut self,
        mut graph: RenderGraph,
        mode: RenderGraphCompilationMode,
    ) -> Result<SubmitHandle, crate::GraphicsError> {
        profile_scope!("submit_graph");
        let writes_swapchain = graph.writes_swapchain();
        if writes_swapchain && self.swapchain_writer_submitted {
            return Err(crate::GraphicsError::InvalidParameter(
                "a swapchain-writing graph was already submitted this frame".into(),
            ));
        }
        let fence = Fence::new_gpu(Arc::clone(self.device.instance()))?;

        #[cfg(debug_assertions)]
        debug_assert_no_write_to_mapped(&graph);
        #[cfg(debug_assertions)]
        debug_assert_pipeline_state_matches_targets(&graph);

        let compiled = graph.compile(mode)?;
        let usage = GraphUsage::from_compiled(compiled);
        {
            let backend = self.device.instance().backend();
            backend.execute_graph(&graph, graph.compiled().unwrap(), fence.gpu_fence())?;
        }
        let handle = SubmitHandle {
            index: self.fences.len(),
        };
        self.swapchain_writer_submitted |= writes_swapchain;
        for (index, prev) in self.submitted_usages.iter().enumerate() {
            if prev.conflicts_with(&usage) {
                self.derived_edges.push((SubmitHandle { index }, handle));
            }
        }
        self.submitted_usages.push(usage);
        self.submitted_graphs.push(graph);
        self.fences.push(fence);
        Ok(handle)
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
    pub fn render(&mut self, graph: RenderGraph) -> Result<SubmitHandle, crate::GraphicsError> {
        self.submit(graph)
    }

    /// Extract the per-submit fences from this schedule.
    ///
    /// This is called internally by [`FramePipeline::end_frame`](crate::pipeline::FramePipeline::end_frame).
    ///
    /// # Panics
    ///
    /// Panics if [`submit`](Self::submit) was never called.
    #[cfg(test)]
    pub(crate) fn take_fences(&mut self) -> Vec<Fence> {
        assert!(
            !self.fences.is_empty(),
            "submit() must be called at least once before end_frame()"
        );
        std::mem::take(&mut self.fences)
    }
}

impl Drop for FrameSchedule {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            owner.return_frame(self.take_resources());
        }
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
        t.sample_count()
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

    #[test]
    fn dropped_schedule_keeps_pending_resources_and_pipeline_drop_waits() {
        let schedule = make_test_schedule();
        let device = Arc::clone(&schedule.device);
        let mut pipeline = device.create_pipeline(1);
        drop(schedule);
        let mut schedule = pipeline.begin_frame().unwrap();
        let buffer = device
            .create_buffer(&crate::BufferDescriptor::new(
                16,
                crate::BufferUsage::COPY_DST,
            ))
            .unwrap();
        let weak = Arc::downgrade(&buffer);
        let mut graph = RenderGraph::new();
        let mut pass = crate::TransferPass::new("pending".into());
        pass.set_transfer_config(crate::TransferConfig::new().with_operation(
            crate::TransferOperation::write_buffer(buffer, 0, Arc::from([0u8; 16].as_slice())),
        ));
        graph.add_transfer_pass(pass);
        let fence = Fence::new_unsignaled();
        let completion = fence.clone();
        schedule.submitted_graphs.push(graph);
        schedule.fences.push(fence);
        drop(schedule);
        assert!(weak.upgrade().is_some());
        assert!(!pipeline.is_idle());
        let check = weak.clone();
        let worker = std::thread::spawn(move || {
            assert!(check.upgrade().is_some());
            completion.signal();
        });
        drop(pipeline);
        worker.join().unwrap();
        assert!(weak.upgrade().is_none());
        // All owners retired; the backend lease can be acquired again.
        drop(device.create_pipeline(1));
    }

    #[test]
    fn strict_submit_reports_ambiguity_and_preserves_prior_submits() {
        let mut schedule = make_test_schedule();
        schedule.submit(make_test_graph("prior")).unwrap();
        let mut graph = make_surface_graph("a");
        let mut second = GraphicsPass::new("b".into());
        second.set_render_targets(
            graph.passes()[0]
                .as_graphics()
                .unwrap()
                .render_targets()
                .unwrap()
                .clone(),
        );
        graph.add_graphics_pass(second);
        graph
            .compile(RenderGraphCompilationMode::Automatic)
            .unwrap();
        assert!(
            schedule
                .submit_with_mode(graph, RenderGraphCompilationMode::Strict)
                .is_err()
        );
        assert_eq!(schedule.fences.len(), 1);
        assert!(!schedule.swapchain_writer_submitted);
    }

    fn make_test_graph(name: &str) -> RenderGraph {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(GraphicsPass::new(name.into()));
        graph
    }

    fn make_test_schedule() -> FrameSchedule {
        let instance = GraphicsInstance::with_parameters(
            crate::InstanceParameters::new().with_backend(crate::BackendType::Dummy),
        )
        .unwrap();
        let device = instance.create_device().unwrap();
        FrameSchedule::new(device, 0, None, Vec::new())
    }

    /// A graph whose single pass renders to the swapchain surface.
    fn make_surface_graph(name: &str) -> RenderGraph {
        use crate::graph::{ColorAttachment, RenderTarget, RenderTargetConfig};
        use crate::types::TextureFormat;

        let target = RenderTarget::test_surface(TextureFormat::Bgra8UnormSrgb, 4, 4);
        let mut pass = GraphicsPass::new(name.into());
        pass.set_render_targets(RenderTargetConfig::new().with_color(ColorAttachment::new(target)));
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(pass);
        graph
    }

    #[test]
    fn submit_signals_fence() {
        let mut schedule = make_test_schedule();
        let handle = schedule
            .submit(make_test_graph("main"))
            .expect("graph submission failed");
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
        schedule.submit(graph).expect("graph submission failed");

        let fences = schedule.take_fences();
        assert_eq!(fences.len(), 1);
        fences[0].wait().unwrap();
    }

    #[test]
    fn submit_multiple_graphs_signals_all_fences() {
        // Multiple graphs per frame, each its own submit on the single queue.
        // Ordering is submission order; every submit gets its own fence.
        let mut schedule = make_test_schedule();
        let a = schedule
            .submit(make_test_graph("prepass"))
            .expect("graph submission failed");
        let b = schedule
            .submit(make_test_graph("main"))
            .expect("graph submission failed");
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
        schedule
            .render(make_test_graph("a"))
            .expect("graph submission failed");
        schedule
            .render(make_test_graph("b"))
            .expect("graph submission failed");

        assert_eq!(schedule.take_fences().len(), 2);
    }

    #[test]
    fn one_swapchain_writer_is_accepted() {
        let mut schedule = make_test_schedule();
        schedule
            .submit(make_test_graph("offscreen"))
            .expect("graph submission failed");
        schedule
            .submit(make_surface_graph("present"))
            .expect("graph submission failed");

        assert_eq!(schedule.take_fences().len(), 2);
    }

    #[test]
    fn second_swapchain_writer_returns_error() {
        let mut schedule = make_test_schedule();
        schedule
            .submit(make_surface_graph("present_a"))
            .expect("graph submission failed");
        assert!(schedule.submit(make_surface_graph("present_b")).is_err());
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
        let instance = GraphicsInstance::with_parameters(
            crate::InstanceParameters::new().with_backend(crate::BackendType::Dummy),
        )
        .unwrap();
        let device = instance.create_device().unwrap();
        let mut schedule = FrameSchedule::new(Arc::clone(&device), 0, None, Vec::new());

        let x = make_test_buffer(&device);
        let y = make_test_buffer(&device);
        let z = make_test_buffer(&device);

        // Graph A writes Y (copy X -> Y); graph B reads Y (copy Y -> Z): RAW.
        let a = schedule
            .submit(make_copy_graph("a", x, Arc::clone(&y)))
            .expect("graph submission failed");
        let b = schedule
            .submit(make_copy_graph("b", y, z))
            .expect("graph submission failed");

        assert_eq!(schedule.derived_dependencies(), &[(a, b)]);
        schedule.take_fences();
    }

    #[test]
    fn no_edge_between_disjoint_graphs() {
        let instance = GraphicsInstance::with_parameters(
            crate::InstanceParameters::new().with_backend(crate::BackendType::Dummy),
        )
        .unwrap();
        let device = instance.create_device().unwrap();
        let mut schedule = FrameSchedule::new(Arc::clone(&device), 0, None, Vec::new());

        let a_src = make_test_buffer(&device);
        let a_dst = make_test_buffer(&device);
        let b_src = make_test_buffer(&device);
        let b_dst = make_test_buffer(&device);

        schedule
            .submit(make_copy_graph("a", a_src, a_dst))
            .expect("graph submission failed");
        schedule
            .submit(make_copy_graph("b", b_src, b_dst))
            .expect("graph submission failed");

        assert!(schedule.derived_dependencies().is_empty());
        schedule.take_fences();
    }

    #[test]
    fn shared_read_only_source_derives_no_edge() {
        let instance = GraphicsInstance::with_parameters(
            crate::InstanceParameters::new().with_backend(crate::BackendType::Dummy),
        )
        .unwrap();
        let device = instance.create_device().unwrap();
        let mut schedule = FrameSchedule::new(Arc::clone(&device), 0, None, Vec::new());

        // Both graphs read the same source (read-after-read): no hazard.
        let src = make_test_buffer(&device);
        let a_dst = make_test_buffer(&device);
        let b_dst = make_test_buffer(&device);

        schedule
            .submit(make_copy_graph("a", Arc::clone(&src), a_dst))
            .expect("graph submission failed");
        schedule
            .submit(make_copy_graph("b", src, b_dst))
            .expect("graph submission failed");

        assert!(schedule.derived_dependencies().is_empty());
        schedule.take_fences();
    }

    #[test]
    fn edges_derive_against_every_earlier_conflicting_submit() {
        let instance = GraphicsInstance::with_parameters(
            crate::InstanceParameters::new().with_backend(crate::BackendType::Dummy),
        )
        .unwrap();
        let device = instance.create_device().unwrap();
        let mut schedule = FrameSchedule::new(Arc::clone(&device), 0, None, Vec::new());

        let x = make_test_buffer(&device);
        let y = make_test_buffer(&device);
        let z = make_test_buffer(&device);

        // A writes Y; B writes Y (WAW with A); C reads Y (RAW with A and B).
        let a = schedule
            .submit(make_copy_graph("a", Arc::clone(&x), Arc::clone(&y)))
            .expect("graph submission failed");
        let b = schedule
            .submit(make_copy_graph("b", x, Arc::clone(&y)))
            .expect("graph submission failed");
        let c = schedule
            .submit(make_copy_graph("c", y, z))
            .expect("graph submission failed");

        assert_eq!(schedule.derived_dependencies(), &[(a, b), (a, c), (b, c)]);
        schedule.take_fences();
    }
}
