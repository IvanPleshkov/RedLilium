//! Per-frame render-graph execution.
//!
//! Each frame builds and submits **exactly one** render graph. Cross-frame
//! CPU/GPU overlap is provided by the frames-in-flight machinery in the
//! pipeline, so there are no cross-graph dependencies or GPU semaphores —
//! multi-pass work lives inside the single graph and is ordered by the compiler.
//!
//! # Architecture
//!
//! `FrameSchedule` is the middle layer of the rendering architecture:
//!
//! | Layer | Type | Purpose |
//! |-------|------|---------|
//! | Pipeline | [`FramePipeline`](crate::pipeline::FramePipeline) | Multiple frames in flight |
//! | **Schedule** | [`FrameSchedule`] | Executes the frame's single graph (this module) |
//! | Graph | [`RenderGraph`](crate::graph::RenderGraph) | Passes + their dependencies |
//! | Pass | [`GraphicsPass`](crate::graph::GraphicsPass), etc. | Single GPU operation |
//!
//! For the full architecture documentation, see `docs/ARCHITECTURE.md`.
//!
//! # Module Contents
//!
//! - [`FrameSchedule`] - Executes the single render graph for one frame
//! - [`Fence`] - CPU-GPU synchronization for frame completion
//!
//! # Example
//!
//! ```ignore
//! // FrameSchedule is created by FramePipeline::begin_frame()
//! let mut schedule = pipeline.begin_frame();
//!
//! // Build the frame's single graph (it may contain many passes).
//! let mut graph = schedule.acquire_graph();
//! graph.add_graphics_pass(shadow_pass);
//! graph.add_graphics_pass(main_pass);
//!
//! // Execute it (signals the frame fence), then hand back to the pipeline.
//! schedule.render(graph);
//! pipeline.end_frame(schedule);
//! ```

mod sync;

pub use sync::{Fence, FenceStatus};

use std::sync::Arc;

use crate::device::GraphicsDevice;
use crate::graph::{RenderGraph, RenderGraphCompilationMode};
use crate::resources::{RingAllocation, RingBuffer};
use redlilium_core::profiling::profile_scope;

/// Frame schedule for streaming graph submission.
///
/// Allows submitting render graphs immediately as they're built,
/// rather than batching all submissions at frame end. This maximizes
/// CPU-GPU parallelism.
///
/// # Async Behavior
///
/// With GPU-backed fences, `present()` and `finish()` return immediately
/// after submitting work to the GPU. The fence tracks when the GPU actually
/// completes, enabling true async rendering where the CPU can build the
/// next frame while the GPU renders the current one.
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
/// // Submit graphs as they're ready
/// let a = schedule.submit("graph_a", graph_a, &[]);
/// let b = schedule.submit("graph_b", graph_b, &[a]);
///
/// // Present to screen (returns immediately - GPU works async)
/// schedule.present("present", final_graph, &[b]);
///
/// // Return schedule to pipeline (stores fence for later waiting)
/// pipeline.end_frame(schedule);
/// ```
pub struct FrameSchedule {
    /// Device for executing the graph.
    device: Arc<GraphicsDevice>,
    /// Fence signaled when this frame's single submit completes (set by `render`).
    fence: Option<Fence>,
    /// The frame slot index (for per-frame resource management).
    frame_slot: usize,
    /// Ring buffer for this frame (if configured in FramePipeline).
    ring_buffer: Option<RingBuffer>,
    /// Pool of reusable render graphs (moved from FramePipeline each frame).
    graph_pool: Vec<RenderGraph>,
    /// The graph executed this frame, kept for recycling in end_frame. Its `Arc`
    /// references keep GPU resources alive until the slot's fence wait.
    submitted_graphs: Vec<RenderGraph>,
}

impl std::fmt::Debug for FrameSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameSchedule")
            .field("device", &self.device.name())
            .field("frame_slot", &self.frame_slot)
            .field("fence", &self.fence)
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
            fence: None,
            frame_slot,
            ring_buffer,
            graph_pool,
            submitted_graphs: Vec::new(),
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

    /// Execute the frame's single render graph and signal the frame fence.
    ///
    /// Exactly **one** render graph is submitted per frame. Cross-frame CPU/GPU
    /// overlap is provided by the frames-in-flight machinery in the pipeline, so
    /// there are no cross-graph dependencies or semaphores. The created fence is
    /// signalled when this submit completes; the pipeline waits on it before
    /// recycling the slot. The graph is kept for recycling — its `Arc`
    /// references keep GPU resources alive until that fence wait.
    ///
    /// Takes ownership of the graph for pooling. Must be called exactly once,
    /// before [`FramePipeline::end_frame`](crate::pipeline::FramePipeline::end_frame).
    ///
    /// # Panics
    ///
    /// Panics if called more than once on the same schedule.
    pub fn render(&mut self, mut graph: RenderGraph) {
        profile_scope!("render_graph");

        assert!(
            self.fence.is_none(),
            "render() has already been called on this schedule"
        );

        // Fence signalled by this frame's single submit. Any failure below
        // (fence creation, compile, submit) means NO GPU work is in flight for
        // this frame, so the slot fence must read as signaled — a GPU fence
        // that was reset but never submitted would stall every subsequent
        // `begin_frame` until its wait times out.
        let mut fence = match Fence::new_gpu(Arc::clone(self.device.instance())) {
            Ok(fence) => fence,
            Err(e) => {
                log::error!("Failed to create frame fence, skipping submit: {e}");
                self.submitted_graphs.push(graph);
                self.fence = Some(Fence::new_signaled());
                return;
            }
        };

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
        self.fence = Some(fence);
    }

    /// Extract the fence from this schedule.
    ///
    /// This is called internally by [`FramePipeline::end_frame`](crate::pipeline::FramePipeline::end_frame).
    ///
    /// # Panics
    ///
    /// Panics if [`render`](Self::render) was not called.
    pub(crate) fn take_fence(&mut self) -> Fence {
        self.fence
            .take()
            .expect("render() must be called before end_frame()")
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

    #[test]
    fn render_signals_fence() {
        let mut schedule = make_test_schedule();
        schedule.render(make_test_graph("main"));

        let fence = schedule.take_fence();
        fence.wait().unwrap();
        assert_eq!(fence.status(), FenceStatus::Signaled);
    }

    #[test]
    fn render_multi_pass_single_graph() {
        // One graph per frame may still carry many passes; ordering/barriers are
        // the compiler's job. Just verify it executes and signals.
        let mut schedule = make_test_schedule();
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(GraphicsPass::new("shadow".into()));
        graph.add_graphics_pass(GraphicsPass::new("main".into()));
        schedule.render(graph);

        let fence = schedule.take_fence();
        fence.wait().unwrap();
        assert_eq!(fence.status(), FenceStatus::Signaled);
    }

    #[test]
    #[should_panic(expected = "render() has already been called")]
    fn double_render_panics() {
        let mut schedule = make_test_schedule();
        schedule.render(make_test_graph("a"));
        schedule.render(make_test_graph("b")); // Panics
    }

    #[test]
    #[should_panic(expected = "render() must be called before end_frame()")]
    fn take_fence_without_render_panics() {
        let mut schedule = make_test_schedule();
        schedule.take_fence(); // Panics
    }
}
