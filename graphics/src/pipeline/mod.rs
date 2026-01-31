//! Frame pipeline for managing multiple frames in flight.
//!
//! This module provides [`FramePipeline`], which coordinates CPU-GPU synchronization
//! across multiple frames, enabling efficient frame overlap (the CPU prepares frame N+1
//! while the GPU renders frame N).
//!
//! # Rendering Architecture Overview
//!
//! The rendering system is organized in layers, from low-level to high-level:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                          FramePipeline                                  │
//! │  Manages multiple frames in flight. Handles CPU-GPU synchronization     │
//! │  via fences. Enables frame overlap for maximum throughput.              │
//! │                                                                         │
//! │  Responsibilities:                                                      │
//! │  - Track fences for N frames in flight                                  │
//! │  - Wait for frame slot availability (begin_frame)                       │
//! │  - Graceful shutdown (wait_idle)                                        │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                          FrameSchedule                                  │
//! │  Orchestrates multiple render graphs within ONE frame. Enables          │
//! │  streaming submission (submit graphs as they're ready).                 │
//! │                                                                         │
//! │  Responsibilities:                                                      │
//! │  - Accept compiled graphs and submit immediately to GPU                 │
//! │  - Track dependencies between graphs via semaphores                     │
//! │  - Return fence for frame completion                                    │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                           RenderGraph                                   │
//! │  Describes a set of passes and their dependencies. Represents one       │
//! │  logical rendering task (e.g., "shadow rendering", "main scene").       │
//! │                                                                         │
//! │  Responsibilities:                                                      │
//! │  - Store passes (graphics, transfer, compute)                           │
//! │  - Track pass-to-pass dependencies                                      │
//! │  - Compile to execution order                                           │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                              Pass                                       │
//! │  A single unit of GPU work (draw calls, copies, dispatches).            │
//! │                                                                         │
//! │  Types:                                                                 │
//! │  - GraphicsPass: vertex/fragment shaders, rasterization                 │
//! │  - TransferPass: buffer/texture copies                                  │
//! │  - ComputePass: compute shaders                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Synchronization Model
//!
//! Different synchronization primitives are used at different levels:
//!
//! | Level | Primitive | Purpose |
//! |-------|-----------|---------|
//! | Pass → Pass | Barriers | Resource state transitions within a graph |
//! | Graph → Graph | Semaphores | GPU-GPU sync within a frame |
//! | Frame → Frame | Fences | CPU-GPU sync across frames |
//!
//! # Frame Overlap (Pipelining)
//!
//! With 2 frames in flight, the CPU and GPU work in parallel:
//!
//! ```text
//! Frame 0: [CPU build] [submit] ─────────────────────────────────────────────►
//!                               [GPU execute frame 0] ───────────────────────►
//!
//! Frame 1:              [CPU build] [submit] ────────────────────────────────►
//!                                            [GPU execute frame 1] ──────────►
//!
//! Frame 2:                          [wait F0] [CPU build] [submit] ──────────►
//!                                                         [GPU execute F2] ──►
//!
//! Time ──────────────────────────────────────────────────────────────────────►
//! ```
//!
//! - CPU doesn't wait for GPU unless it's reusing a frame slot
//! - GPU processes frames in order via semaphores
//! - Fences ensure we don't overwrite in-use resources
//!
//! # Graceful Shutdown
//!
//! When the application exits (e.g., window close), call [`FramePipeline::wait_idle`]
//! to ensure all GPU work completes before destroying resources:
//!
//! ```text
//! [Window Close Event]
//!         │
//!         ▼
//! ┌───────────────────┐
//! │  Stop rendering   │  Don't start new frames
//! └─────────┬─────────┘
//!           │
//!           ▼
//! ┌───────────────────┐
//! │ pipeline.wait_idle│  Wait for all in-flight GPU work
//! └─────────┬─────────┘
//!           │
//!           ▼
//! ┌───────────────────┐
//! │  Drop resources   │  Safe to destroy GPU objects
//! └───────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::pipeline::FramePipeline;
//! use redlilium_graphics::scheduler::FrameSchedule;
//!
//! // Create pipeline with 2 frames in flight
//! let mut pipeline = FramePipeline::new(2);
//!
//! // Main loop
//! while !window.should_close() {
//!     // Wait for frame slot (blocks if GPU is behind)
//!     pipeline.begin_frame();
//!
//!     // Build and submit this frame's work
//!     let mut schedule = FrameSchedule::new();
//!     let shadows = schedule.submit("shadows", shadow_graph, &[]);
//!     let main = schedule.submit("main", main_graph, &[shadows]);
//!     let fence = schedule.submit_and_present("present", post_graph, &[main]);
//!
//!     // Record fence for this frame
//!     pipeline.end_frame(fence);
//! }
//!
//! // Graceful shutdown - wait for all GPU work
//! pipeline.wait_idle();
//!
//! // Now safe to destroy device, resources, etc.
//! ```
//!
//! # Choosing Frames in Flight
//!
//! | Count | Behavior |
//! |-------|----------|
//! | 1 | CPU waits for GPU every frame. Simple but slow. |
//! | 2 | Good balance. CPU can work on N+1 while GPU renders N. |
//! | 3 | More overlap, higher latency. Useful for VR or heavy CPU work. |
//!
//! More frames = more throughput but higher input latency and memory usage
//! (each frame needs its own resources like uniform buffers).

use crate::scheduler::Fence;
use std::time::Duration;

/// Manages multiple frames in flight for CPU-GPU parallelism.
///
/// `FramePipeline` coordinates the overlap between CPU frame preparation and
/// GPU frame execution. It tracks fences for each frame slot and ensures
/// the CPU doesn't overwrite resources that the GPU is still using.
///
/// # Frame Slots
///
/// With N frames in flight, there are N "slots". Each slot can hold one frame's
/// worth of work. When all slots are full, [`begin_frame`](Self::begin_frame)
/// blocks until the oldest frame completes.
///
/// ```text
/// frames_in_flight = 2
///
/// Slot 0: [Frame 0] ──► [Frame 2] ──► [Frame 4] ──►
/// Slot 1: [Frame 1] ──► [Frame 3] ──► [Frame 5] ──►
/// ```
///
/// # Thread Safety
///
/// `FramePipeline` is **not thread-safe**. It should be owned by a single
/// thread (typically the main/render thread).
#[derive(Debug)]
pub struct FramePipeline {
    /// Fences for each frame slot. `None` if slot hasn't been used yet.
    frame_fences: Vec<Option<Fence>>,

    /// Current frame slot index (0 to frames_in_flight - 1).
    current_slot: usize,

    /// Total number of frames in flight.
    frames_in_flight: usize,

    /// Total frames started (for debugging/profiling).
    frame_count: u64,
}

impl FramePipeline {
    /// Create a new frame pipeline.
    ///
    /// # Arguments
    ///
    /// * `frames_in_flight` - Number of frames that can be in flight simultaneously.
    ///   Typically 2 or 3. Must be at least 1.
    ///
    /// # Panics
    ///
    /// Panics if `frames_in_flight` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use redlilium_graphics::pipeline::FramePipeline;
    ///
    /// let pipeline = FramePipeline::new(2);
    /// assert_eq!(pipeline.frames_in_flight(), 2);
    /// ```
    pub fn new(frames_in_flight: usize) -> Self {
        assert!(frames_in_flight > 0, "frames_in_flight must be at least 1");

        Self {
            frame_fences: (0..frames_in_flight).map(|_| None).collect(),
            current_slot: 0,
            frames_in_flight,
            frame_count: 0,
        }
    }

    /// Begin a new frame.
    ///
    /// This waits for the current frame slot to become available. If the GPU
    /// is still processing a previous frame in this slot, this call blocks
    /// until that work completes.
    ///
    /// Call this at the start of each frame, before building render graphs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// loop {
    ///     pipeline.begin_frame();  // May block if GPU is behind
    ///
    ///     // Build and submit frame...
    ///     let fence = schedule.submit_and_present(...);
    ///
    ///     pipeline.end_frame(fence);
    /// }
    /// ```
    pub fn begin_frame(&mut self) {
        // Wait for previous work in this slot to complete
        if let Some(fence) = &self.frame_fences[self.current_slot] {
            fence.wait();
        }

        self.frame_count += 1;

        log::trace!(
            "Begin frame {} (slot {})",
            self.frame_count,
            self.current_slot
        );
    }

    /// Begin a new frame with a timeout.
    ///
    /// Like [`begin_frame`](Self::begin_frame), but returns `false` if the
    /// timeout elapses before the frame slot becomes available.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for the frame slot.
    ///
    /// # Returns
    ///
    /// `true` if the frame slot is ready, `false` if timeout elapsed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::time::Duration;
    ///
    /// if !pipeline.begin_frame_timeout(Duration::from_millis(100)) {
    ///     log::warn!("GPU is falling behind!");
    ///     // Handle the timeout (skip frame, reduce quality, etc.)
    /// }
    /// ```
    pub fn begin_frame_timeout(&mut self, timeout: Duration) -> bool {
        if let Some(fence) = &self.frame_fences[self.current_slot]
            && !fence.wait_timeout(timeout)
        {
            return false;
        }

        self.frame_count += 1;

        log::trace!(
            "Begin frame {} (slot {})",
            self.frame_count,
            self.current_slot
        );

        true
    }

    /// End the current frame.
    ///
    /// Records the fence for this frame and advances to the next frame slot.
    /// Call this after submitting all work for the frame.
    ///
    /// # Arguments
    ///
    /// * `fence` - The fence returned from [`FrameSchedule::submit_and_present`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// pipeline.begin_frame();
    ///
    /// let mut schedule = FrameSchedule::new();
    /// // ... submit graphs ...
    /// let fence = schedule.submit_and_present("present", graph, &[deps]);
    ///
    /// pipeline.end_frame(fence);  // Records fence, advances slot
    /// ```
    ///
    /// [`FrameSchedule::submit_and_present`]: crate::scheduler::FrameSchedule::submit_and_present
    pub fn end_frame(&mut self, fence: Fence) {
        log::trace!(
            "End frame {} (slot {})",
            self.frame_count,
            self.current_slot
        );

        // Store fence for this slot
        self.frame_fences[self.current_slot] = Some(fence);

        // Advance to next slot
        self.current_slot = (self.current_slot + 1) % self.frames_in_flight;
    }

    /// Wait for all in-flight GPU work to complete.
    ///
    /// This blocks until every frame slot's fence is signaled. Call this
    /// before destroying GPU resources to ensure they're not in use.
    ///
    /// # When to Call
    ///
    /// - Application shutdown (window close)
    /// - Before hot-reloading shaders
    /// - Before resizing swapchain
    /// - Any time you need to ensure GPU is completely idle
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Main loop
    /// while !window.should_close() {
    ///     pipeline.begin_frame();
    ///     // ... render ...
    ///     pipeline.end_frame(fence);
    /// }
    ///
    /// // Shutdown
    /// pipeline.wait_idle();  // Wait for ALL GPU work
    /// drop(device);          // Safe to destroy
    /// ```
    pub fn wait_idle(&self) {
        log::trace!("Waiting for GPU idle ({} slots)", self.frames_in_flight);

        for (i, fence) in self.frame_fences.iter().enumerate() {
            if let Some(f) = fence {
                log::trace!("Waiting for slot {}...", i);
                f.wait();
            }
        }

        log::trace!("GPU idle");
    }

    /// Wait for all in-flight GPU work with a timeout.
    ///
    /// Like [`wait_idle`](Self::wait_idle), but returns `false` if the
    /// timeout elapses before all work completes.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum total time to wait across all fences.
    ///
    /// # Returns
    ///
    /// `true` if GPU is idle, `false` if timeout elapsed.
    pub fn wait_idle_timeout(&self, timeout: Duration) -> bool {
        let start = std::time::Instant::now();

        for fence in self.frame_fences.iter().flatten() {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return false;
            }

            let remaining = timeout - elapsed;
            if !fence.wait_timeout(remaining) {
                return false;
            }
        }

        true
    }

    /// Get the number of frames in flight.
    ///
    /// This is the value passed to [`new`](Self::new).
    pub fn frames_in_flight(&self) -> usize {
        self.frames_in_flight
    }

    /// Get the current frame slot index.
    ///
    /// Returns a value from 0 to `frames_in_flight - 1`.
    pub fn current_slot(&self) -> usize {
        self.current_slot
    }

    /// Get the total number of frames started.
    ///
    /// This counter increments each time [`begin_frame`](Self::begin_frame) is called.
    /// Useful for debugging and profiling.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Check if a specific frame slot is ready (non-blocking).
    ///
    /// Returns `true` if the slot's fence is signaled or if the slot
    /// hasn't been used yet.
    pub fn is_slot_ready(&self, slot: usize) -> bool {
        assert!(slot < self.frames_in_flight, "Invalid slot index");

        match &self.frame_fences[slot] {
            Some(fence) => fence.is_signaled(),
            None => true, // Never used, so it's ready
        }
    }

    /// Check if all frame slots are ready (non-blocking).
    ///
    /// Returns `true` if [`wait_idle`](Self::wait_idle) would return immediately.
    pub fn is_idle(&self) -> bool {
        self.frame_fences
            .iter()
            .all(|f| f.as_ref().is_none_or(|fence| fence.is_signaled()))
    }
}

impl Default for FramePipeline {
    /// Creates a pipeline with 2 frames in flight (recommended default).
    fn default() -> Self {
        Self::new(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let pipeline = FramePipeline::new(2);
        assert_eq!(pipeline.frames_in_flight(), 2);
        assert_eq!(pipeline.current_slot(), 0);
        assert_eq!(pipeline.frame_count(), 0);
    }

    #[test]
    fn test_default() {
        let pipeline = FramePipeline::default();
        assert_eq!(pipeline.frames_in_flight(), 2);
    }

    #[test]
    #[should_panic(expected = "frames_in_flight must be at least 1")]
    fn test_zero_frames_panics() {
        FramePipeline::new(0);
    }

    #[test]
    fn test_begin_frame_increments_count() {
        let mut pipeline = FramePipeline::new(2);
        assert_eq!(pipeline.frame_count(), 0);

        pipeline.begin_frame();
        assert_eq!(pipeline.frame_count(), 1);

        pipeline.begin_frame();
        assert_eq!(pipeline.frame_count(), 2);
    }

    #[test]
    fn test_end_frame_advances_slot() {
        let mut pipeline = FramePipeline::new(3);
        assert_eq!(pipeline.current_slot(), 0);

        pipeline.begin_frame();
        pipeline.end_frame(Fence::default());
        assert_eq!(pipeline.current_slot(), 1);

        pipeline.begin_frame();
        pipeline.end_frame(Fence::default());
        assert_eq!(pipeline.current_slot(), 2);

        pipeline.begin_frame();
        pipeline.end_frame(Fence::default());
        assert_eq!(pipeline.current_slot(), 0); // Wraps around
    }

    #[test]
    fn test_is_slot_ready_unused() {
        let pipeline = FramePipeline::new(2);
        assert!(pipeline.is_slot_ready(0));
        assert!(pipeline.is_slot_ready(1));
    }

    #[test]
    fn test_is_idle_initial() {
        let pipeline = FramePipeline::new(2);
        assert!(pipeline.is_idle());
    }

    #[test]
    fn test_wait_idle_no_fences() {
        let pipeline = FramePipeline::new(2);
        // Should return immediately when no fences
        pipeline.wait_idle();
    }

    #[test]
    fn test_wait_idle_with_signaled_fences() {
        let mut pipeline = FramePipeline::new(2);

        // Create already-signaled fences
        let fence1 = Fence::default();
        fence1.signal();
        let fence2 = Fence::default();
        fence2.signal();

        pipeline.begin_frame();
        pipeline.end_frame(fence1);

        pipeline.begin_frame();
        pipeline.end_frame(fence2);

        // Should return immediately since fences are signaled
        pipeline.wait_idle();
        assert!(pipeline.is_idle());
    }

    #[test]
    fn test_frame_lifecycle() {
        let mut pipeline = FramePipeline::new(2);

        // Frame 0
        pipeline.begin_frame();
        assert_eq!(pipeline.frame_count(), 1);
        let fence0 = Fence::default();
        fence0.signal(); // Simulate GPU completion
        pipeline.end_frame(fence0);
        assert_eq!(pipeline.current_slot(), 1);

        // Frame 1
        pipeline.begin_frame();
        assert_eq!(pipeline.frame_count(), 2);
        let fence1 = Fence::default();
        fence1.signal();
        pipeline.end_frame(fence1);
        assert_eq!(pipeline.current_slot(), 0);

        // Frame 2 (reuses slot 0)
        pipeline.begin_frame(); // Would wait for fence0 if not signaled
        assert_eq!(pipeline.frame_count(), 3);
        assert_eq!(pipeline.current_slot(), 0);
    }

    #[test]
    fn test_begin_frame_timeout_ready() {
        let mut pipeline = FramePipeline::new(2);

        // Should succeed immediately (no fence to wait on)
        assert!(pipeline.begin_frame_timeout(Duration::from_millis(1)));
        assert_eq!(pipeline.frame_count(), 1);
    }

    #[test]
    fn test_wait_idle_timeout_ready() {
        let pipeline = FramePipeline::new(2);

        // Should succeed immediately (no fences)
        assert!(pipeline.wait_idle_timeout(Duration::from_millis(1)));
    }

    #[test]
    #[should_panic(expected = "Invalid slot index")]
    fn test_is_slot_ready_invalid() {
        let pipeline = FramePipeline::new(2);
        pipeline.is_slot_ready(5); // Invalid
    }
}
