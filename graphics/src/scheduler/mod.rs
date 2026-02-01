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

use crate::graph::CompiledGraph;

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
#[derive(Debug)]
struct SubmittedGraph {
    /// Debug name for this graph.
    name: String,
    /// Semaphore signaled when this graph completes on GPU.
    completion: Semaphore,
    /// Handles of graphs this one waited for (for debugging/visualization).
    #[allow(dead_code)]
    waited_for: Vec<GraphHandle>,
}

/// Frame schedule for streaming graph submission.
///
/// Allows submitting render graphs immediately as they're built,
/// rather than batching all submissions at frame end. This maximizes
/// CPU-GPU parallelism.
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
/// // Present to screen (marks schedule as complete)
/// schedule.present("present", final_graph, &[b]);
///
/// // Return schedule to pipeline
/// pipeline.end_frame(schedule);
/// ```
#[derive(Debug)]
pub struct FrameSchedule {
    /// Submitted graphs with their completion semaphores.
    submitted: Vec<SubmittedGraph>,
    /// Counter for generating semaphore IDs.
    semaphore_counter: u64,
    /// Fence signaled when frame completes (set by present()).
    fence: Option<Fence>,
}

impl FrameSchedule {
    /// Create a new frame schedule.
    ///
    /// This is called internally by [`FramePipeline::begin_frame`](crate::pipeline::FramePipeline::begin_frame).
    pub(crate) fn new() -> Self {
        Self {
            submitted: Vec::new(),
            semaphore_counter: 0,
            fence: None,
        }
    }

    /// Submit a graph for immediate execution.
    ///
    /// The graph is submitted to the GPU immediately. If `wait_for` is non-empty,
    /// the GPU will wait for those graphs to complete before starting this one.
    ///
    /// Returns a handle that can be used as a dependency for subsequent graphs.
    ///
    /// # Arguments
    ///
    /// * `name` - Debug name for this graph submission
    /// * `graph` - The compiled render graph to execute
    /// * `wait_for` - Graphs that must complete before this one starts
    ///
    /// # Example
    ///
    /// ```ignore
    /// // No dependencies - starts immediately
    /// let shadows = schedule.submit("shadows", shadow_graph, &[]);
    ///
    /// // Waits for shadows to complete
    /// let lighting = schedule.submit("lighting", light_graph, &[shadows]);
    /// ```
    pub fn submit(
        &mut self,
        name: impl Into<String>,
        graph: CompiledGraph,
        wait_for: &[GraphHandle],
    ) -> GraphHandle {
        let name = name.into();

        // Validate wait_for handles
        for &handle in wait_for {
            assert!(
                handle.index() < self.submitted.len(),
                "Invalid dependency handle"
            );
        }

        // Create completion semaphore for this graph
        let completion = Semaphore::new(self.next_semaphore_id());

        // Collect semaphores to wait on
        let wait_semaphores: Vec<&Semaphore> = wait_for
            .iter()
            .map(|h| &self.submitted[h.index()].completion)
            .collect();

        // Submit to GPU (this would be the actual GPU submission)
        self.submit_to_gpu(&name, &graph, &wait_semaphores, &completion);

        log::trace!(
            "Submitted graph '{}' (waiting for {} dependencies)",
            name,
            wait_for.len()
        );

        let handle = GraphHandle::new(self.submitted.len() as u32);
        self.submitted.push(SubmittedGraph {
            name,
            completion,
            waited_for: wait_for.to_vec(),
        });
        handle
    }

    /// Submit a graph and present to swapchain.
    ///
    /// This is typically the final submission of a frame. It waits for
    /// the specified dependencies, executes the graph, and presents
    /// the result to the swapchain.
    ///
    /// After calling this, the schedule is considered complete and should
    /// be returned to the pipeline via [`FramePipeline::end_frame`](crate::pipeline::FramePipeline::end_frame).
    ///
    /// # Arguments
    ///
    /// * `name` - Debug name for this graph submission
    /// * `graph` - The compiled render graph to execute
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
    /// let main = schedule.submit("main", main_graph, &[]);
    /// schedule.present("present", final_graph, &[main]);
    /// pipeline.end_frame(schedule);
    /// ```
    pub fn present(
        &mut self,
        name: impl Into<String>,
        graph: CompiledGraph,
        wait_for: &[GraphHandle],
    ) {
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

        // Create completion semaphore and fence
        let completion = Semaphore::new(self.next_semaphore_id());
        let fence = Fence::new_unsignaled();

        // Collect semaphores to wait on
        let wait_semaphores: Vec<&Semaphore> = wait_for
            .iter()
            .map(|h| &self.submitted[h.index()].completion)
            .collect();

        // Submit to GPU with fence for CPU synchronization
        self.submit_to_gpu_with_present(&name, &graph, &wait_semaphores, &completion, &fence);

        log::trace!(
            "Submitted present graph '{}' (waiting for {} dependencies)",
            name,
            wait_for.len()
        );

        self.submitted.push(SubmittedGraph {
            name,
            completion,
            waited_for: wait_for.to_vec(),
        });

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
        let fence = Fence::new_unsignaled();

        // Collect semaphores to wait on (GPU waits for these before signaling fence)
        let _wait_semaphores: Vec<&Semaphore> = wait_for
            .iter()
            .map(|h| &self.submitted[h.index()].completion)
            .collect();

        // TODO: Actual GPU submission with fence signal would happen here
        // This would submit a command buffer that:
        // 1. Waits on wait_semaphores
        // 2. Signals the fence when complete
        log::trace!(
            "Finish schedule (waiting for {} dependencies)",
            wait_for.len()
        );

        // For the dummy backend (no actual GPU), signal the fence immediately
        // to simulate instant completion. Real backends will signal this
        // when GPU work actually completes.
        fence.signal();

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

    /// Submit graph to GPU (placeholder for actual implementation).
    fn submit_to_gpu(
        &self,
        name: &str,
        graph: &CompiledGraph,
        wait_semaphores: &[&Semaphore],
        signal_semaphore: &Semaphore,
    ) {
        // TODO: Actual GPU submission would happen here
        // This would:
        // 1. Record commands from the compiled graph into a command buffer
        // 2. Submit the command buffer with:
        //    - wait_semaphores: semaphores to wait on before execution
        //    - signal_semaphore: semaphore to signal after completion
        log::trace!(
            "GPU submit '{}': {} passes, wait={}, signal={}",
            name,
            graph.pass_order().len(),
            wait_semaphores
                .iter()
                .map(|s| s.id().to_string())
                .collect::<Vec<_>>()
                .join(","),
            signal_semaphore.id()
        );
    }

    /// Submit graph to GPU with present (placeholder for actual implementation).
    fn submit_to_gpu_with_present(
        &self,
        name: &str,
        graph: &CompiledGraph,
        wait_semaphores: &[&Semaphore],
        signal_semaphore: &Semaphore,
        fence: &Fence,
    ) {
        // TODO: Actual GPU submission with present would happen here
        // This would:
        // 1. Acquire swapchain image (with semaphore)
        // 2. Record commands including final blit to swapchain
        // 3. Submit with fence for CPU synchronization
        // 4. Queue present operation
        log::trace!(
            "GPU submit+present '{}': {} passes, wait={}, signal={}, fence={:?}",
            name,
            graph.pass_order().len(),
            wait_semaphores
                .iter()
                .map(|s| s.id().to_string())
                .collect::<Vec<_>>()
                .join(","),
            signal_semaphore.id(),
            fence.status()
        );

        // For the dummy backend (no actual GPU), signal the fence immediately
        // to simulate instant completion. Real backends will signal this
        // when GPU work actually completes.
        fence.signal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphicsPass, RenderGraph};

    fn make_test_graph(name: &str) -> CompiledGraph {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(GraphicsPass::new(name.into()));
        graph.compile().unwrap()
    }

    #[test]
    fn test_submit_single_graph() {
        let mut schedule = FrameSchedule::new();

        let graph = make_test_graph("test");
        let handle = schedule.submit("test", graph, &[]);

        assert_eq!(schedule.submitted_count(), 1);
        assert_eq!(handle.index(), 0);
    }

    #[test]
    fn test_submit_with_dependencies() {
        let mut schedule = FrameSchedule::new();

        let shadow = schedule.submit("shadows", make_test_graph("shadow"), &[]);
        let depth = schedule.submit("depth", make_test_graph("depth"), &[]);
        let main = schedule.submit("main", make_test_graph("main"), &[shadow, depth]);

        assert_eq!(schedule.submitted_count(), 3);
        assert_eq!(main.index(), 2);
    }

    #[test]
    fn test_present() {
        let mut schedule = FrameSchedule::new();

        let main = schedule.submit("main", make_test_graph("main"), &[]);
        assert!(!schedule.is_presented());

        schedule.present("present", make_test_graph("present"), &[main]);

        assert_eq!(schedule.submitted_count(), 2);
        assert!(schedule.is_presented());
    }

    #[test]
    fn test_take_fence() {
        let mut schedule = FrameSchedule::new();

        let main = schedule.submit("main", make_test_graph("main"), &[]);
        schedule.present("present", make_test_graph("present"), &[main]);

        let fence = schedule.take_fence();
        // In the dummy backend, fence is signaled immediately after present()
        assert_eq!(fence.status(), FenceStatus::Signaled);
        assert!(!schedule.is_presented()); // Fence was taken
    }

    #[test]
    #[should_panic(expected = "present() has already been called")]
    fn test_double_present_panics() {
        let mut schedule = FrameSchedule::new();

        schedule.present("present1", make_test_graph("present1"), &[]);
        schedule.present("present2", make_test_graph("present2"), &[]); // Panics
    }

    #[test]
    #[should_panic(expected = "present() or finish() must be called before end_frame()")]
    fn test_take_fence_without_present_panics() {
        let mut schedule = FrameSchedule::new();
        schedule.submit("main", make_test_graph("main"), &[]);
        schedule.take_fence(); // Panics
    }

    #[test]
    fn test_finish() {
        let mut schedule = FrameSchedule::new();

        let main = schedule.submit("main", make_test_graph("main"), &[]);
        assert!(!schedule.is_presented());

        schedule.finish(&[main]);

        // is_presented returns true for finish() too since fence is set
        assert!(schedule.is_presented());
    }

    #[test]
    fn test_finish_empty_dependencies() {
        let mut schedule = FrameSchedule::new();
        schedule.submit("main", make_test_graph("main"), &[]);
        schedule.finish(&[]); // Finish without waiting for any graph

        assert!(schedule.is_presented());
    }

    #[test]
    #[should_panic(expected = "finish() or present() has already been called")]
    fn test_double_finish_panics() {
        let mut schedule = FrameSchedule::new();

        schedule.finish(&[]);
        schedule.finish(&[]); // Panics
    }

    #[test]
    #[should_panic(expected = "finish() or present() has already been called")]
    fn test_finish_after_present_panics() {
        let mut schedule = FrameSchedule::new();

        schedule.present("present", make_test_graph("present"), &[]);
        schedule.finish(&[]); // Panics
    }

    #[test]
    fn test_submitted_names() {
        let mut schedule = FrameSchedule::new();

        schedule.submit("shadows", make_test_graph("shadow"), &[]);
        schedule.submit("main", make_test_graph("main"), &[]);
        schedule.present("post", make_test_graph("post"), &[]);

        let names: Vec<_> = schedule.submitted_names().collect();
        assert_eq!(names, vec!["shadows", "main", "post"]);
    }

    #[test]
    #[should_panic(expected = "Invalid dependency handle")]
    fn test_invalid_dependency_panics() {
        let mut schedule = FrameSchedule::new();

        // Try to depend on non-existent graph
        let invalid_handle = GraphHandle::new(999);
        schedule.submit("test", make_test_graph("test"), &[invalid_handle]);
    }

    #[test]
    fn test_complex_dependency_graph() {
        let mut schedule = FrameSchedule::new();

        // Build a diamond dependency pattern:
        //       shadows
        //      /       \
        //   depth     gbuffer
        //      \       /
        //        main
        //          |
        //        post

        let shadows = schedule.submit("shadows", make_test_graph("shadow"), &[]);
        let depth = schedule.submit("depth", make_test_graph("depth"), &[shadows]);
        let gbuffer = schedule.submit("gbuffer", make_test_graph("gbuffer"), &[shadows]);
        let main = schedule.submit("main", make_test_graph("main"), &[depth, gbuffer]);
        schedule.present("post", make_test_graph("post"), &[main]);

        assert_eq!(schedule.submitted_count(), 5);
        assert!(schedule.is_presented());
    }
}
