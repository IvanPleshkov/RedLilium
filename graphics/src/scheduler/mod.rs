//! Frame scheduling and streaming graph submission.
//!
//! The scheduler provides streaming submission of render graphs to the GPU,
//! allowing graphs to start executing as soon as they're ready while the CPU
//! continues building subsequent graphs.
//!
//! # Architecture
//!
//! `FrameSchedule` is the middle layer of the rendering architecture:
//!
//! | Layer | Type | Purpose |
//! |-------|------|---------|
//! | Pipeline | [`FramePipeline`](crate::pipeline::FramePipeline) | Multiple frames in flight |
//! | **Schedule** | [`FrameSchedule`] | Streaming graph submission (this module) |
//! | Graph | [`RenderGraph`](crate::graph::RenderGraph) | Pass dependencies |
//! | Pass | [`GraphicsPass`](crate::graph::GraphicsPass), etc. | Single GPU operation |
//!
//! For the full architecture documentation, see `docs/ARCHITECTURE.md`.
//!
//! # Module Contents
//!
//! - [`FrameSchedule`] - Manages streaming submission for a single frame
//! - [`GraphHandle`] - Handle to a submitted graph, used for dependencies
//! - [`Semaphore`] - GPU synchronization primitive for graph ordering
//! - [`Fence`] - CPU-GPU synchronization for frame completion
//!
//! # Example
//!
//! ```ignore
//! // FrameSchedule is created by FramePipeline::begin_frame()
//! let mut schedule = pipeline.begin_frame();
//!
//! // Submit shadow graph immediately - GPU starts working
//! let shadows = schedule.submit("shadows", shadow_graph, &[]);
//!
//! // Build and submit depth while GPU renders shadows
//! let depth = schedule.submit("depth", depth_graph, &[]);
//!
//! // Main pass waits for both shadows and depth
//! let main = schedule.submit("main", main_graph, &[shadows, depth]);
//!
//! // Present to screen (marks schedule as complete)
//! schedule.present("present", post_graph, &[main]);
//!
//! // Return schedule to pipeline
//! pipeline.end_frame(schedule);
//! ```

mod sync;

pub use sync::{Fence, FenceStatus, Semaphore};

use std::sync::Arc;

use crate::device::GraphicsDevice;
use crate::graph::{RenderGraph, RenderGraphCompilationMode};
use crate::resources::{RingAllocation, RingBuffer};
use redlilium_core::profiling::profile_scope;

/// Handle to a submitted graph in the frame schedule.
///
/// Used to declare dependencies between graphs. A graph can wait
/// for multiple other graphs to complete before starting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphHandle(u32);

impl GraphHandle {
    fn new(index: u32) -> Self {
        Self(index)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// Information about a submitted graph.
struct SubmittedGraph {
    /// Debug name for this graph.
    name: String,
    /// Semaphore signaled when this graph completes on GPU.
    completion: Semaphore,
    /// Handles of graphs this one waited for (for debugging/visualization).
    #[allow(dead_code)]
    waited_for: Vec<GraphHandle>,
}

impl std::fmt::Debug for SubmittedGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubmittedGraph")
            .field("name", &self.name)
            .field("completion", &self.completion)
            .field("waited_for", &self.waited_for)
            .finish()
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
    /// Device for executing graphs.
    device: Arc<GraphicsDevice>,
    /// Submitted graphs with their completion semaphores.
    submitted: Vec<SubmittedGraph>,
    /// Counter for generating semaphore IDs.
    semaphore_counter: u64,
    /// Fence signaled when frame completes (set by present()).
    fence: Option<Fence>,
    /// The frame slot index (for per-frame resource management).
    frame_slot: usize,
    /// Ring buffer for this frame (if configured in FramePipeline).
    ring_buffer: Option<RingBuffer>,
    /// Pool of reusable render graphs (moved from FramePipeline each frame).
    graph_pool: Vec<RenderGraph>,
    /// Graphs submitted this frame (for recycling in end_frame).
    submitted_graphs: Vec<RenderGraph>,
}

impl std::fmt::Debug for FrameSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameSchedule")
            .field("device", &self.device.name())
            .field("frame_slot", &self.frame_slot)
            .field("submitted", &self.submitted)
            .field("semaphore_counter", &self.semaphore_counter)
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
            submitted: Vec::new(),
            semaphore_counter: 0,
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

    /// Submit a graph for immediate execution.
    ///
    /// The graph is submitted to the GPU immediately. If `wait_for` is non-empty,
    /// the GPU will wait for those graphs to complete before starting this one.
    ///
    /// Takes ownership of the graph for pooling. Use [`acquire_graph`](Self::acquire_graph)
    /// to get a graph from the pool.
    ///
    /// Returns a handle that can be used as a dependency for subsequent graphs.
    ///
    /// # Arguments
    ///
    /// * `name` - Debug name for this graph submission
    /// * `graph` - The render graph to execute (ownership transferred)
    /// * `wait_for` - Graphs that must complete before this one starts
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Acquire from pool, configure, submit
    /// let mut graph = schedule.acquire_graph();
    /// graph.add_graphics_pass(shadow_pass);
    /// let shadows = schedule.submit("shadows", graph, &[]);
    ///
    /// // Another graph waiting for shadows
    /// let mut graph = schedule.acquire_graph();
    /// graph.add_graphics_pass(light_pass);
    /// let lighting = schedule.submit("lighting", graph, &[shadows]);
    /// ```
    pub fn submit(
        &mut self,
        name: impl Into<String>,
        mut graph: RenderGraph,
        wait_for: &[GraphHandle],
    ) -> GraphHandle {
        let name = name.into();
        profile_scope!("submit_graph");

        // Validate wait_for handles
        for &handle in wait_for {
            assert!(
                handle.index() < self.submitted.len(),
                "Invalid dependency handle"
            );
        }

        // Get semaphore ID before acquiring backend lock (avoids borrow conflict)
        let semaphore_id = self.next_semaphore_id();

        // Create GPU-backed completion semaphore for this graph
        let backend = self.device.instance().backend();
        let gpu_semaphore = backend.create_semaphore();
        let completion = Semaphore::new(semaphore_id, gpu_semaphore);

        // Collect GPU semaphores to wait on from dependency handles
        let wait_gpu_semaphores: Vec<&crate::backend::GpuSemaphore> = wait_for
            .iter()
            .map(|h| self.submitted[h.index()].completion.gpu_semaphore())
            .collect();

        // Signal this graph's completion semaphore
        let signal_gpu_semaphores: Vec<&crate::backend::GpuSemaphore> =
            vec![completion.gpu_semaphore()];

        // Compile and execute the graph on the GPU
        match graph.compile(RenderGraphCompilationMode::Strict) {
            Ok(_) => {
                profile_scope!("execute_graph");
                let compiled = graph.compiled().unwrap();
                if let Err(e) = backend.execute_graph(
                    &graph,
                    compiled,
                    &wait_gpu_semaphores,
                    &signal_gpu_semaphores,
                    None,
                ) {
                    log::error!("Failed to execute graph '{}': {}", name, e);
                }
            }
            Err(e) => {
                log::error!("Failed to compile graph '{}': {}", name, e);
            }
        }

        let handle = GraphHandle::new(self.submitted.len() as u32);
        self.submitted.push(SubmittedGraph {
            name,
            completion,
            waited_for: wait_for.to_vec(),
        });

        // Store graph for recycling at end of frame
        self.submitted_graphs.push(graph);

        handle
    }

    /// Submit a graph and present to swapchain.
    ///
    /// This is typically the final submission of a frame. It waits for
    /// the specified dependencies, executes the graph, and presents
    /// the result to the swapchain.
    ///
    /// Takes ownership of the graph for pooling. Use [`acquire_graph`](Self::acquire_graph)
    /// to get a graph from the pool.
    ///
    /// After calling this, the schedule is considered complete and should
    /// be returned to the pipeline via [`FramePipeline::end_frame`](crate::pipeline::FramePipeline::end_frame).
    ///
    /// # Arguments
    ///
    /// * `name` - Debug name for this graph submission
    /// * `graph` - The render graph to execute (ownership transferred)
    /// * `wait_for` - Graphs that must complete before this one starts
    ///
    /// # Panics
    ///
    /// Panics if `present` has already been called on this schedule.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut schedule = pipeline.begin_frame();
    /// let mut graph = schedule.acquire_graph();
    /// graph.add_graphics_pass(main_pass);
    /// let main = schedule.submit("main", graph, &[]);
    /// let mut present_graph = schedule.acquire_graph();
    /// present_graph.add_graphics_pass(final_pass);
    /// schedule.present("present", present_graph, &[main]);
    /// pipeline.end_frame(schedule);
    /// ```
    pub fn present(
        &mut self,
        name: impl Into<String>,
        mut graph: RenderGraph,
        wait_for: &[GraphHandle],
    ) {
        profile_scope!("present");

        assert!(
            self.fence.is_none(),
            "present() has already been called on this schedule"
        );

        let name = name.into();

        // Validate wait_for handles
        for &handle in wait_for {
            assert!(
                handle.index() < self.submitted.len(),
                "Invalid dependency handle"
            );
        }

        // Get semaphore ID before acquiring backend lock (avoids borrow conflict)
        let semaphore_id = self.next_semaphore_id();

        // Create GPU-backed fence first (acquires+releases backend read lock internally)
        let instance = Arc::clone(self.device.instance());
        let fence = Fence::new_gpu(instance);

        // Create GPU-backed completion semaphore
        let backend = self.device.instance().backend();
        let gpu_semaphore = backend.create_semaphore();
        let completion = Semaphore::new(semaphore_id, gpu_semaphore);

        // Collect GPU semaphores to wait on from dependency handles
        let wait_gpu_semaphores: Vec<&crate::backend::GpuSemaphore> = wait_for
            .iter()
            .map(|h| self.submitted[h.index()].completion.gpu_semaphore())
            .collect();

        // Signal this graph's completion semaphore
        let signal_gpu_semaphores: Vec<&crate::backend::GpuSemaphore> =
            vec![completion.gpu_semaphore()];

        // Compile and execute with semaphores and fence
        match graph.compile(RenderGraphCompilationMode::Strict) {
            Ok(_) => {
                profile_scope!("execute_present");
                let compiled = graph.compiled().unwrap();
                if let Err(e) = backend.execute_graph(
                    &graph,
                    compiled,
                    &wait_gpu_semaphores,
                    &signal_gpu_semaphores,
                    fence.gpu_fence(),
                ) {
                    log::error!("Failed to execute present graph '{}': {}", name, e);
                }
            }
            Err(e) => {
                log::error!("Failed to compile present graph '{}': {}", name, e);
            }
        }

        self.submitted.push(SubmittedGraph {
            name,
            completion,
            waited_for: wait_for.to_vec(),
        });

        // Store graph for recycling at end of frame
        self.submitted_graphs.push(graph);

        self.fence = Some(fence);
    }

    /// Get the number of submitted graphs.
    pub fn submitted_count(&self) -> usize {
        self.submitted.len()
    }

    /// Check if any graphs have been submitted.
    pub fn is_empty(&self) -> bool {
        self.submitted.is_empty()
    }

    /// Check if the schedule has been presented.
    pub fn is_presented(&self) -> bool {
        self.fence.is_some()
    }

    /// Get debug names of all submitted graphs in submission order.
    pub fn submitted_names(&self) -> impl Iterator<Item = &str> {
        self.submitted.iter().map(|s| s.name.as_str())
    }

    /// Finish the schedule without presenting to a swapchain.
    ///
    /// This is an alternative to [`present`](Self::present) for offscreen rendering
    /// or test scenarios where no swapchain is involved. It sets a fence that will
    /// be signaled when all submitted graphs complete.
    ///
    /// After calling this, the schedule is considered complete and should
    /// be returned to the pipeline via [`FramePipeline::end_frame`](crate::pipeline::FramePipeline::end_frame).
    ///
    /// # Panics
    ///
    /// Panics if `present` or `finish` has already been called on this schedule.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut schedule = pipeline.begin_frame();
    /// let main = schedule.submit("main", main_graph, &[]);
    /// schedule.finish(&[main]);  // No swapchain presentation
    /// pipeline.end_frame(schedule);
    /// ```
    pub fn finish(&mut self, wait_for: &[GraphHandle]) {
        profile_scope!("finish");

        assert!(
            self.fence.is_none(),
            "finish() or present() has already been called on this schedule"
        );

        // Validate wait_for handles
        for &handle in wait_for {
            assert!(
                handle.index() < self.submitted.len(),
                "Invalid dependency handle"
            );
        }

        // Create fence for CPU synchronization
        // Note: Since intermediate submit() calls currently execute synchronously,
        // all GPU work is already complete by the time finish() is called.
        // We create a signaled fence to indicate completion.
        let instance = Arc::clone(self.device.instance());
        let fence = Fence::new_gpu(instance);

        self.fence = Some(fence);
    }

    /// Extract the fence from this schedule.
    ///
    /// This is called internally by [`FramePipeline::end_frame`](crate::pipeline::FramePipeline::end_frame).
    ///
    /// # Panics
    ///
    /// Panics if neither `present()` nor `finish()` was called.
    pub(crate) fn take_fence(&mut self) -> Fence {
        self.fence
            .take()
            .expect("present() or finish() must be called before end_frame()")
    }

    /// Generate a unique semaphore ID.
    fn next_semaphore_id(&mut self) -> u64 {
        let id = self.semaphore_counter;
        self.semaphore_counter += 1;
        id
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
    fn test_submit_single_graph() {
        let mut schedule = make_test_schedule();

        let graph = make_test_graph("test");
        let handle = schedule.submit("test", graph, &[]);

        assert_eq!(schedule.submitted_count(), 1);
        assert_eq!(handle.index(), 0);
    }

    #[test]
    fn test_submit_with_dependencies() {
        let mut schedule = make_test_schedule();

        let shadow_graph = make_test_graph("shadow");
        let depth_graph = make_test_graph("depth");
        let main_graph = make_test_graph("main");

        let shadow = schedule.submit("shadows", shadow_graph, &[]);
        let depth = schedule.submit("depth", depth_graph, &[]);
        let main = schedule.submit("main", main_graph, &[shadow, depth]);

        assert_eq!(schedule.submitted_count(), 3);
        assert_eq!(main.index(), 2);
    }

    #[test]
    fn test_present() {
        let mut schedule = make_test_schedule();

        let main_graph = make_test_graph("main");
        let present_graph = make_test_graph("present");

        let main = schedule.submit("main", main_graph, &[]);
        assert!(!schedule.is_presented());

        schedule.present("present", present_graph, &[main]);

        assert_eq!(schedule.submitted_count(), 2);
        assert!(schedule.is_presented());
    }

    #[test]
    fn test_take_fence() {
        let mut schedule = make_test_schedule();

        let main_graph = make_test_graph("main");
        let present_graph = make_test_graph("present");

        let main = schedule.submit("main", main_graph, &[]);
        schedule.present("present", present_graph, &[main]);

        let fence = schedule.take_fence();
        // Wait for GPU work to complete
        fence.wait();
        assert_eq!(fence.status(), FenceStatus::Signaled);
        assert!(!schedule.is_presented()); // Fence was taken
    }

    #[test]
    #[should_panic(expected = "present() has already been called")]
    fn test_double_present_panics() {
        let mut schedule = make_test_schedule();

        let present1 = make_test_graph("present1");
        let present2 = make_test_graph("present2");

        schedule.present("present1", present1, &[]);
        schedule.present("present2", present2, &[]); // Panics
    }

    #[test]
    #[should_panic(expected = "present() or finish() must be called before end_frame()")]
    fn test_take_fence_without_present_panics() {
        let mut schedule = make_test_schedule();
        let main_graph = make_test_graph("main");
        schedule.submit("main", main_graph, &[]);
        schedule.take_fence(); // Panics
    }

    #[test]
    fn test_finish() {
        let mut schedule = make_test_schedule();

        let main_graph = make_test_graph("main");
        let main = schedule.submit("main", main_graph, &[]);
        assert!(!schedule.is_presented());

        schedule.finish(&[main]);

        // is_presented returns true for finish() too since fence is set
        assert!(schedule.is_presented());
    }

    #[test]
    fn test_finish_empty_dependencies() {
        let mut schedule = make_test_schedule();
        let main_graph = make_test_graph("main");
        schedule.submit("main", main_graph, &[]);
        schedule.finish(&[]); // Finish without waiting for any graph

        assert!(schedule.is_presented());
    }

    #[test]
    #[should_panic(expected = "finish() or present() has already been called")]
    fn test_double_finish_panics() {
        let mut schedule = make_test_schedule();

        schedule.finish(&[]);
        schedule.finish(&[]); // Panics
    }

    #[test]
    #[should_panic(expected = "finish() or present() has already been called")]
    fn test_finish_after_present_panics() {
        let mut schedule = make_test_schedule();

        let present_graph = make_test_graph("present");
        schedule.present("present", present_graph, &[]);
        schedule.finish(&[]); // Panics
    }

    #[test]
    fn test_submitted_names() {
        let mut schedule = make_test_schedule();

        let shadow_graph = make_test_graph("shadow");
        let main_graph = make_test_graph("main");
        let post_graph = make_test_graph("post");

        schedule.submit("shadows", shadow_graph, &[]);
        schedule.submit("main", main_graph, &[]);
        schedule.present("post", post_graph, &[]);

        let names: Vec<_> = schedule.submitted_names().collect();
        assert_eq!(names, vec!["shadows", "main", "post"]);
    }

    #[test]
    #[should_panic(expected = "Invalid dependency handle")]
    fn test_invalid_dependency_panics() {
        let mut schedule = make_test_schedule();

        // Try to depend on non-existent graph
        let invalid_handle = GraphHandle::new(999);
        let test_graph = make_test_graph("test");
        schedule.submit("test", test_graph, &[invalid_handle]);
    }

    #[test]
    fn test_complex_dependency_graph() {
        let mut schedule = make_test_schedule();

        // Build a diamond dependency pattern:
        //       shadows
        //      /       \
        //   depth     gbuffer
        //      \       /
        //        main
        //          |
        //        post

        let shadow_graph = make_test_graph("shadow");
        let depth_graph = make_test_graph("depth");
        let gbuffer_graph = make_test_graph("gbuffer");
        let main_graph = make_test_graph("main");
        let post_graph = make_test_graph("post");

        let shadows = schedule.submit("shadows", shadow_graph, &[]);
        let depth = schedule.submit("depth", depth_graph, &[shadows]);
        let gbuffer = schedule.submit("gbuffer", gbuffer_graph, &[shadows]);
        let main = schedule.submit("main", main_graph, &[depth, gbuffer]);
        schedule.present("post", post_graph, &[main]);

        assert_eq!(schedule.submitted_count(), 5);
        assert!(schedule.is_presented());
    }
}
