//! Frame pipeline for managing multiple frames in flight.
//!
//! This module provides [`FramePipeline`], which coordinates CPU-GPU synchronization
//! across multiple frames, enabling efficient frame overlap (the CPU prepares frame N+1
//! while the GPU renders frame N).
//!
//! # Architecture
//!
//! `FramePipeline` is the top layer of the rendering architecture:
//!
//! | Layer | Type | Purpose |
//! |-------|------|---------|
//! | **Pipeline** | [`FramePipeline`] | Multiple frames in flight (this module) |
//! | Schedule | [`FrameSchedule`](crate::scheduler::FrameSchedule) | Streaming graph submission |
//! | Graph | [`RenderGraph`](crate::graph::RenderGraph) | Pass dependencies |
//! | Pass | [`GraphicsPass`](crate::graph::GraphicsPass), etc. | Single GPU operation |
//!
//! For the full architecture documentation, see `docs/ARCHITECTURE.md`.
//!
//! # Creation
//!
//! `FramePipeline` is created by [`GraphicsDevice::create_pipeline`](crate::device::GraphicsDevice::create_pipeline).
//!
//! # Synchronization
//!
//! - **Fences** (this level): CPU waits for GPU across frames
//! - **Semaphores** (schedule level): GPU-GPU sync within a frame
//! - **Barriers** (graph level): Resource transitions within a graph
//!
//! # Example
//!
//! ```ignore
//! use redlilium_graphics::{GraphicsInstance, FramePipeline};
//!
//! let instance = GraphicsInstance::new()?;
//! let device = instance.create_device()?;
//! let mut pipeline = device.create_pipeline(2);  // 2 frames in flight
//!
//! while !window.should_close() {
//!     let mut schedule = pipeline.begin_frame();  // Wait + get schedule
//!
//!     let shadows = schedule.submit("shadows", shadow_graph, &[]);
//!     let main = schedule.submit("main", main_graph, &[shadows]);
//!     schedule.present("present", post_graph, &[main]);
//!
//!     pipeline.end_frame(schedule);  // Store fence, advance slot
//! }
//!
//! pipeline.wait_idle();  // Graceful shutdown
//! ```
//!
//! # Choosing Frames in Flight
//!
//! | Count | Behavior |
//! |-------|----------|
//! | 1 | CPU waits for GPU every frame. Simple but slow. |
//! | 2 | Good balance. CPU can work on N+1 while GPU renders N. (Recommended) |
//! | 3 | More overlap, higher latency. Useful for VR or heavy CPU work. |
//!
//! More frames = higher throughput but more input latency and memory usage.

use std::sync::Arc;
use std::time::Duration;

use crate::device::GraphicsDevice;
use crate::error::GraphicsError;
use crate::graph::RenderGraph;
use crate::profiling::{frame_mark, profile_scope};
use crate::resources::RingBuffer;
use crate::scheduler::{Fence, FrameSchedule};
use crate::types::BufferUsage;

/// Manages multiple frames in flight for CPU-GPU parallelism.
///
/// `FramePipeline` coordinates the overlap between CPU frame preparation and
/// GPU frame execution. It tracks fences for each frame slot and ensures
/// the CPU doesn't overwrite resources that the GPU is still using.
///
/// # Creation
///
/// Created via [`GraphicsDevice::create_pipeline`](crate::device::GraphicsDevice::create_pipeline):
///
/// ```ignore
/// let pipeline = device.create_pipeline(2);
/// ```
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
pub struct FramePipeline {
    /// Device for executing graphs.
    device: Option<Arc<GraphicsDevice>>,

    /// Fences for each frame slot. `None` if slot hasn't been used yet.
    frame_fences: Vec<Option<Fence>>,

    /// Current frame slot index (0 to frames_in_flight - 1).
    current_slot: usize,

    /// Total number of frames in flight.
    frames_in_flight: usize,

    /// Total frames started (for debugging/profiling).
    frame_count: u64,

    /// Per-frame ring buffers for streaming data. `None` if not configured or
    /// temporarily moved to FrameSchedule during a frame.
    ring_buffers: Vec<Option<RingBuffer>>,

    /// Pool of reusable render graphs. Moved to FrameSchedule each frame,
    /// then recycled back in end_frame.
    graph_pool: Vec<RenderGraph>,
}

impl std::fmt::Debug for FramePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FramePipeline")
            .field("device", &self.device.as_ref().map(|d| d.name()))
            .field("frame_fences", &self.frame_fences)
            .field("current_slot", &self.current_slot)
            .field("frames_in_flight", &self.frames_in_flight)
            .field("frame_count", &self.frame_count)
            .field("has_ring_buffers", &self.has_ring_buffers())
            .finish()
    }
}

impl FramePipeline {
    /// Create a new frame pipeline without a device (for testing).
    ///
    /// This is called internally by [`GraphicsDevice::create_pipeline`](crate::device::GraphicsDevice::create_pipeline).
    ///
    /// # Arguments
    ///
    /// * `frames_in_flight` - Number of frames that can be in flight simultaneously.
    ///   Typically 2 or 3. Must be at least 1.
    ///
    /// # Panics
    ///
    /// Panics if `frames_in_flight` is 0.
    #[allow(dead_code)]
    pub(crate) fn new(frames_in_flight: usize) -> Self {
        assert!(frames_in_flight > 0, "frames_in_flight must be at least 1");

        Self {
            device: None,
            frame_fences: (0..frames_in_flight).map(|_| None).collect(),
            current_slot: 0,
            frames_in_flight,
            frame_count: 0,
            ring_buffers: Vec::new(),
            graph_pool: Vec::new(),
        }
    }

    /// Create a new frame pipeline with a device for GPU execution.
    ///
    /// This is called internally by [`GraphicsDevice::create_pipeline`](crate::device::GraphicsDevice::create_pipeline).
    ///
    /// # Arguments
    ///
    /// * `device` - The graphics device for executing graphs.
    /// * `frames_in_flight` - Number of frames that can be in flight simultaneously.
    ///
    /// # Panics
    ///
    /// Panics if `frames_in_flight` is 0.
    pub(crate) fn new_with_device(device: Arc<GraphicsDevice>, frames_in_flight: usize) -> Self {
        assert!(frames_in_flight > 0, "frames_in_flight must be at least 1");

        Self {
            device: Some(device),
            frame_fences: (0..frames_in_flight).map(|_| None).collect(),
            current_slot: 0,
            frames_in_flight,
            frame_count: 0,
            ring_buffers: Vec::new(),
            graph_pool: Vec::new(),
        }
    }

    /// Begin a new frame and return a schedule for graph submission.
    ///
    /// This waits for the current frame slot to become available. If the GPU
    /// is still processing a previous frame in this slot, this call blocks
    /// until that work completes.
    ///
    /// Returns a [`FrameSchedule`] for submitting render graphs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// loop {
    ///     let mut schedule = pipeline.begin_frame();  // Wait + get schedule
    ///
    ///     let main = schedule.submit("main", &main_graph, &[]);
    ///     schedule.present("present", &post_graph, &[main]);
    ///
    ///     pipeline.end_frame(schedule);
    /// }
    /// ```
    pub fn begin_frame(&mut self) -> FrameSchedule {
        profile_scope!("begin_frame");

        // Wait for previous work in this slot to complete
        if let Some(fence) = &self.frame_fences[self.current_slot] {
            profile_scope!("wait_fence");
            fence.wait();
        }

        // Now that the GPU has finished with the old frame, advance the deferred
        // destruction system to clean up resources that were dropped
        if let Some(device) = &self.device {
            profile_scope!("advance_deferred");
            device.advance_deferred_destruction();
        }

        self.frame_count += 1;

        log::trace!(
            "Begin frame {} (slot {})",
            self.frame_count,
            self.current_slot
        );

        // Take ring buffer for this slot (if configured) and reset it
        let ring_buffer = if !self.ring_buffers.is_empty() {
            self.ring_buffers[self.current_slot].take().map(|mut rb| {
                rb.reset();
                rb
            })
        } else {
            None
        };

        // Take graph pool for this frame
        let graph_pool = std::mem::take(&mut self.graph_pool);

        // Create schedule with device if available
        match &self.device {
            Some(device) => FrameSchedule::new_with_device(
                device.clone(),
                self.current_slot,
                ring_buffer,
                graph_pool,
            ),
            None => FrameSchedule::new_with_ring_buffer(ring_buffer),
        }
    }

    /// Begin a new frame with a timeout.
    ///
    /// Like [`begin_frame`](Self::begin_frame), but returns `None` if the
    /// timeout elapses before the frame slot becomes available.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for the frame slot.
    ///
    /// # Returns
    ///
    /// `Some(schedule)` if the frame slot is ready, `None` if timeout elapsed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::time::Duration;
    ///
    /// match pipeline.begin_frame_timeout(Duration::from_millis(100)) {
    ///     Some(schedule) => {
    ///         // Normal frame processing
    ///         schedule.present("present", &graph, &[]);
    ///         pipeline.end_frame(schedule);
    ///     }
    ///     None => {
    ///         log::warn!("GPU is falling behind!");
    ///         // Handle the timeout (skip frame, reduce quality, etc.)
    ///     }
    /// }
    /// ```
    pub fn begin_frame_timeout(&mut self, timeout: Duration) -> Option<FrameSchedule> {
        profile_scope!("begin_frame_timeout");

        if let Some(fence) = &self.frame_fences[self.current_slot]
            && !fence.wait_timeout(timeout)
        {
            return None;
        }

        // Now that the GPU has finished with the old frame, advance the deferred
        // destruction system to clean up resources that were dropped
        if let Some(device) = &self.device {
            profile_scope!("advance_deferred");
            device.advance_deferred_destruction();
        }

        self.frame_count += 1;

        log::trace!(
            "Begin frame {} (slot {})",
            self.frame_count,
            self.current_slot
        );

        // Take ring buffer for this slot (if configured) and reset it
        let ring_buffer = if !self.ring_buffers.is_empty() {
            self.ring_buffers[self.current_slot].take().map(|mut rb| {
                rb.reset();
                rb
            })
        } else {
            None
        };

        // Take graph pool for this frame
        let graph_pool = std::mem::take(&mut self.graph_pool);

        // Create schedule with device if available
        let schedule = match &self.device {
            Some(device) => FrameSchedule::new_with_device(
                device.clone(),
                self.current_slot,
                ring_buffer,
                graph_pool,
            ),
            None => FrameSchedule::new_with_ring_buffer(ring_buffer),
        };

        Some(schedule)
    }

    /// End the current frame.
    ///
    /// Takes ownership of the schedule, extracts its fence, and advances
    /// to the next frame slot.
    ///
    /// # Arguments
    ///
    /// * `schedule` - The schedule returned from [`begin_frame`](Self::begin_frame),
    ///   after calling [`present`](FrameSchedule::present) on it.
    ///
    /// # Panics
    ///
    /// Panics if [`present`](FrameSchedule::present) was not called on the schedule.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut schedule = pipeline.begin_frame();
    /// let main = schedule.submit("main", main_graph, &[]);
    /// schedule.present("present", post_graph, &[main]);
    /// pipeline.end_frame(schedule);  // Takes ownership
    /// ```
    pub fn end_frame(&mut self, mut schedule: FrameSchedule) {
        profile_scope!("end_frame");

        let fence = schedule.take_fence();

        // Return ring buffer to its slot (if configured)
        if !self.ring_buffers.is_empty()
            && let Some(ring) = schedule.take_ring_buffer()
        {
            self.ring_buffers[self.current_slot] = Some(ring);
        }

        // Recycle graphs: clear submitted graphs and merge back to pool
        let mut pool = schedule.take_graph_pool();
        for mut graph in schedule.take_submitted_graphs() {
            graph.clear(); // Clears passes, edges, and compiled cache
            pool.push(graph);
        }
        self.graph_pool = pool;

        log::trace!(
            "End frame {} (slot {})",
            self.frame_count,
            self.current_slot
        );

        // Store fence for this slot
        self.frame_fences[self.current_slot] = Some(fence);

        // Advance to next slot
        self.current_slot = (self.current_slot + 1) % self.frames_in_flight;

        // Mark frame end for Tracy
        frame_mark!();
    }

    /// Wait for the current frame slot to be ready.
    ///
    /// This blocks until the current slot's fence is signaled. This is more
    /// efficient than [`wait_idle`](Self::wait_idle) when you only need to
    /// ensure one frame slot is available.
    ///
    /// # When to Call
    ///
    /// - Before resizing swapchain (only current slot needed)
    /// - Before modifying resources used by current frame slot
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Resize handling - only wait for current slot
    /// if let Some(event) = resize_manager.update() {
    ///     pipeline.wait_current_slot();
    ///     surface.resize(event.width, event.height);
    /// }
    /// ```
    pub fn wait_current_slot(&self) {
        if let Some(fence) = &self.frame_fences[self.current_slot] {
            log::trace!("Waiting for current slot {}...", self.current_slot);
            fence.wait();
        }
    }

    /// Wait for the current frame slot with a timeout.
    ///
    /// Like [`wait_current_slot`](Self::wait_current_slot), but returns `false`
    /// if the timeout elapses before the slot becomes available.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait.
    ///
    /// # Returns
    ///
    /// `true` if the slot is ready, `false` if timeout elapsed.
    pub fn wait_current_slot_timeout(&self, timeout: std::time::Duration) -> bool {
        if let Some(fence) = &self.frame_fences[self.current_slot] {
            fence.wait_timeout(timeout)
        } else {
            true // No fence means slot is ready
        }
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
    /// - Any time you need to ensure GPU is completely idle
    ///
    /// For swapchain resize, prefer [`wait_current_slot`](Self::wait_current_slot)
    /// which is faster (waits for 1 frame instead of all frames).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Main loop
    /// while !window.should_close() {
    ///     let mut schedule = pipeline.begin_frame();
    ///     // ... render ...
    ///     pipeline.end_frame(schedule);
    /// }
    ///
    /// // Shutdown
    /// pipeline.wait_idle();  // Wait for ALL GPU work
    /// drop(device);          // Safe to destroy
    /// ```
    pub fn wait_idle(&self) {
        profile_scope!("wait_idle");

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
    /// This is the value passed to [`GraphicsDevice::create_pipeline`](crate::device::GraphicsDevice::create_pipeline).
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

    /// Create ring buffers for per-frame uniform/vertex data streaming.
    ///
    /// This creates one ring buffer per frame slot, allowing efficient CPU-GPU
    /// streaming without synchronization overhead. Call this once during initialization.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Size of each ring buffer in bytes
    /// * `usage` - Buffer usage flags (RING flag is added automatically)
    /// * `label` - Debug label for the buffers (slot number is appended)
    ///
    /// # Example
    ///
    /// ```ignore
    /// pipeline.create_ring_buffers(
    ///     64 * 1024,  // 64 KB per frame
    ///     BufferUsage::UNIFORM | BufferUsage::COPY_DST,
    ///     "per_frame_uniforms",
    /// )?;
    /// ```
    pub fn create_ring_buffers(
        &mut self,
        capacity: u64,
        usage: BufferUsage,
        label: &str,
    ) -> Result<(), GraphicsError> {
        self.create_ring_buffers_with_alignment(
            capacity,
            usage,
            label,
            RingBuffer::DEFAULT_ALIGNMENT,
        )
    }

    /// Create ring buffers with custom alignment.
    ///
    /// Like [`create_ring_buffers`](Self::create_ring_buffers), but allows
    /// specifying custom alignment for allocations.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Size of each ring buffer in bytes
    /// * `usage` - Buffer usage flags
    /// * `label` - Debug label for the buffers
    /// * `alignment` - Allocation alignment (must be power of 2)
    pub fn create_ring_buffers_with_alignment(
        &mut self,
        capacity: u64,
        usage: BufferUsage,
        label: &str,
        alignment: u64,
    ) -> Result<(), GraphicsError> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| GraphicsError::InvalidParameter("No device configured".to_string()))?;

        self.ring_buffers = Vec::with_capacity(self.frames_in_flight);
        for slot in 0..self.frames_in_flight {
            let slot_label = format!("{}_slot{}", label, slot);
            let ring = RingBuffer::with_alignment(device, capacity, usage, &slot_label, alignment)?;
            self.ring_buffers.push(Some(ring));
        }

        log::info!(
            "FramePipeline: created {} ring buffers of {} bytes each for '{}'",
            self.frames_in_flight,
            capacity,
            label
        );

        Ok(())
    }

    /// Check if ring buffers have been configured.
    pub fn has_ring_buffers(&self) -> bool {
        !self.ring_buffers.is_empty()
    }

    /// Get the ring buffer capacity (if configured).
    pub fn ring_buffer_capacity(&self) -> Option<u64> {
        self.ring_buffers
            .first()
            .and_then(|opt| opt.as_ref().map(|rb| rb.capacity()))
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

    /// Signal all pending fences (for testing).
    ///
    /// This simulates GPU completion for all frame slots.
    #[cfg(test)]
    pub(crate) fn signal_all_fences(&self) {
        for fence in self.frame_fences.iter().flatten() {
            fence.signal();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphicsPass, RenderGraph};

    fn make_test_graph(name: &str) -> RenderGraph {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(GraphicsPass::new(name.into()));
        graph
    }

    #[test]
    fn test_new() {
        let pipeline = FramePipeline::new(2);
        assert_eq!(pipeline.frames_in_flight(), 2);
        assert_eq!(pipeline.current_slot(), 0);
        assert_eq!(pipeline.frame_count(), 0);
    }

    #[test]
    #[should_panic(expected = "frames_in_flight must be at least 1")]
    fn test_zero_frames_panics() {
        FramePipeline::new(0);
    }

    #[test]
    fn test_begin_frame_returns_schedule() {
        let mut pipeline = FramePipeline::new(2);
        let schedule = pipeline.begin_frame();

        assert_eq!(pipeline.frame_count(), 1);
        assert!(schedule.is_empty());
        assert!(!schedule.is_presented());
    }

    #[test]
    fn test_end_frame_advances_slot() {
        let mut pipeline = FramePipeline::new(3);
        assert_eq!(pipeline.current_slot(), 0);

        let mut schedule = pipeline.begin_frame();
        schedule.present("present", make_test_graph("present"), &[]);
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 1);

        // Signal fences so next begin_frame doesn't block
        pipeline.signal_all_fences();

        let mut schedule = pipeline.begin_frame();
        schedule.present("present", make_test_graph("present"), &[]);
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 2);

        pipeline.signal_all_fences();

        let mut schedule = pipeline.begin_frame();
        schedule.present("present", make_test_graph("present"), &[]);
        pipeline.end_frame(schedule);
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
    fn test_frame_lifecycle() {
        let mut pipeline = FramePipeline::new(2);

        // Frame 0
        let mut schedule = pipeline.begin_frame();
        assert_eq!(pipeline.frame_count(), 1);
        schedule.present("present", make_test_graph("present"), &[]);
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 1);

        // Simulate GPU completing frame 0
        pipeline.signal_all_fences();

        // Frame 1
        let mut schedule = pipeline.begin_frame();
        assert_eq!(pipeline.frame_count(), 2);
        schedule.present("present", make_test_graph("present"), &[]);
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 0);

        // Simulate GPU completing frame 1
        pipeline.signal_all_fences();

        // Frame 2 (reuses slot 0)
        let schedule = pipeline.begin_frame();
        assert_eq!(pipeline.frame_count(), 3);
        assert_eq!(pipeline.current_slot(), 0);
        assert!(schedule.is_empty()); // Fresh schedule
    }

    #[test]
    fn test_begin_frame_timeout_ready() {
        let mut pipeline = FramePipeline::new(2);

        // Should succeed immediately (no fence to wait on)
        let schedule = pipeline.begin_frame_timeout(Duration::from_millis(1));
        assert!(schedule.is_some());
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

    #[test]
    #[should_panic(expected = "present() or finish() must be called before end_frame()")]
    fn test_end_frame_without_present_panics() {
        let mut pipeline = FramePipeline::new(2);
        let schedule = pipeline.begin_frame();
        pipeline.end_frame(schedule); // Panics - no present() or finish() called
    }

    #[test]
    fn test_full_frame_with_finish() {
        let mut pipeline = FramePipeline::new(2);

        let mut schedule = pipeline.begin_frame();

        // Build a simple offscreen render
        let main_graph = make_test_graph("main");
        let main = schedule.submit("main", main_graph, &[]);
        schedule.finish(&[main]); // Use finish instead of present

        assert!(schedule.is_presented()); // is_presented returns true for finish too

        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 1);
    }

    #[test]
    fn test_full_frame_with_graphs() {
        let mut pipeline = FramePipeline::new(2);

        let mut schedule = pipeline.begin_frame();

        // Build a simple dependency chain
        let shadow_graph = make_test_graph("shadow");
        let main_graph = make_test_graph("main");
        let present_graph = make_test_graph("present");

        let shadows = schedule.submit("shadows", shadow_graph, &[]);
        let main = schedule.submit("main", main_graph, &[shadows]);
        schedule.present("present", present_graph, &[main]);

        assert_eq!(schedule.submitted_count(), 3);
        assert!(schedule.is_presented());

        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 1);
    }
}
