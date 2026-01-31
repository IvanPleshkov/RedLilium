//! Frame scheduling and streaming graph submission.
//!
//! The scheduler provides streaming submission of render graphs to the GPU,
//! allowing graphs to start executing as soon as they're ready while the CPU
//! continues building subsequent graphs.
//!
//! # Architecture
//!
//! - [`FrameSchedule`] - Manages streaming submission for a single frame
//! - [`GraphHandle`] - Handle to a submitted graph, used for dependencies
//! - [`Semaphore`] - GPU synchronization primitive for graph ordering
//! - [`Fence`] - CPU-GPU synchronization for frame completion
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::scheduler::FrameSchedule;
//!
//! let mut schedule = FrameSchedule::new();
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
//! // Present to screen
//! let fence = schedule.submit_and_present(post_graph, &[main], &mut swapchain);
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
/// # Lifecycle
///
/// ```ignore
/// // Each frame:
/// let mut schedule = FrameSchedule::new();
///
/// // Submit graphs as they're ready
/// let a = schedule.submit("graph_a", graph_a, &[]);
/// let b = schedule.submit("graph_b", graph_b, &[a]);
///
/// // Present and get fence
/// let fence = schedule.submit_and_present(final_graph, &[b], swapchain);
///
/// // Store fence for later synchronization
/// frame_fences[current_frame] = fence;
/// ```
#[derive(Debug, Default)]
pub struct FrameSchedule {
    /// Submitted graphs with their completion semaphores.
    submitted: Vec<SubmittedGraph>,
    /// Counter for generating semaphore IDs.
    semaphore_counter: u64,
}

impl FrameSchedule {
    /// Create a new frame schedule.
    pub fn new() -> Self {
        Self::default()
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
    /// Returns a fence that will be signaled when the GPU completes this frame.
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
    /// let fence = schedule.submit_and_present(
    ///     "present",
    ///     final_graph,
    ///     &[main_pass],
    /// );
    /// ```
    pub fn submit_and_present(
        &mut self,
        name: impl Into<String>,
        graph: CompiledGraph,
        wait_for: &[GraphHandle],
    ) -> Fence {
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

        fence
    }

    /// Get the number of submitted graphs.
    pub fn submitted_count(&self) -> usize {
        self.submitted.len()
    }

    /// Check if any graphs have been submitted.
    pub fn is_empty(&self) -> bool {
        self.submitted.is_empty()
    }

    /// Get debug names of all submitted graphs in submission order.
    pub fn submitted_names(&self) -> impl Iterator<Item = &str> {
        self.submitted.iter().map(|s| s.name.as_str())
    }

    /// Reset the schedule for a new frame.
    ///
    /// This clears all submitted graphs. Call this at the start of each frame
    /// after ensuring the previous frame's work is complete.
    pub fn reset(&mut self) {
        self.submitted.clear();
        // Note: semaphore_counter intentionally not reset for unique IDs
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
    fn test_submit_and_present() {
        let mut schedule = FrameSchedule::new();

        let main = schedule.submit("main", make_test_graph("main"), &[]);
        let fence = schedule.submit_and_present("present", make_test_graph("present"), &[main]);

        assert_eq!(schedule.submitted_count(), 2);
        assert_eq!(fence.status(), FenceStatus::Unsignaled);
    }

    #[test]
    fn test_reset() {
        let mut schedule = FrameSchedule::new();

        schedule.submit("test", make_test_graph("test"), &[]);
        assert_eq!(schedule.submitted_count(), 1);

        schedule.reset();
        assert_eq!(schedule.submitted_count(), 0);
        assert!(schedule.is_empty());
    }

    #[test]
    fn test_submitted_names() {
        let mut schedule = FrameSchedule::new();

        schedule.submit("shadows", make_test_graph("shadow"), &[]);
        schedule.submit("main", make_test_graph("main"), &[]);
        schedule.submit("post", make_test_graph("post"), &[]);

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
        let _fence = schedule.submit_and_present("post", make_test_graph("post"), &[main]);

        assert_eq!(schedule.submitted_count(), 5);
    }
}
