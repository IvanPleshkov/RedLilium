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
//!     let mut schedule = pipeline.begin_frame()?;  // Wait + get schedule
//!
//!     // Graphs execute in submission order on the single graphics queue;
//!     // shared resources are synchronized automatically.
//!     schedule.submit(shadow_graph);
//!     schedule.submit(main_graph);  // at most one graph writes the swapchain
//!
//!     pipeline.end_frame(schedule);  // Store fences, advance slot
//! }
//!
//! pipeline.wait_idle()?;  // Graceful shutdown
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

use redlilium_core::pool::Poolable;

use crate::device::GraphicsDevice;
use crate::error::GraphicsError;
use crate::graph::RenderGraph;
use crate::resources::RingBuffer;
use crate::scheduler::{Fence, FrameSchedule};
use crate::types::BufferUsage;
use redlilium_core::profiling::{frame_mark, profile_scope};

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
    /// Per-submit fences for each frame slot (one per graph submitted that
    /// frame). Empty if the slot hasn't been used yet. The slot is ready to
    /// recycle only when ALL of its fences are signaled — with per-submit
    /// fences, waiting just the last one would recycle too early if a late
    /// submit failed and was replaced by an already-signaled CPU fence.
    frame_fences: Vec<Vec<Fence>>,

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

    /// Per-slot submitted graphs kept alive until fence wait guarantees GPU is done.
    /// Resetting these after fence wait drops Arc references to GPU resources safely.
    slot_graphs: Vec<Vec<RenderGraph>>,

    /// Device for executing graphs.
    ///
    /// Declared last deliberately: fields drop in declaration order, and this
    /// keep-alive must outlive the fences/ring buffers/graphs above — their
    /// `Drop`s destroy GPU objects and need the backend (owned by the
    /// device→instance chain) to still be alive. See #50.
    device: Arc<GraphicsDevice>,
}

impl std::fmt::Debug for FramePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FramePipeline")
            .field("device", &self.device.name())
            .field("frame_fences", &self.frame_fences)
            .field("current_slot", &self.current_slot)
            .field("frames_in_flight", &self.frames_in_flight)
            .field("frame_count", &self.frame_count)
            .field("has_ring_buffers", &self.has_ring_buffers())
            .finish()
    }
}

impl FramePipeline {
    /// Create a new frame pipeline.
    ///
    /// This is called internally by [`GraphicsDevice::create_pipeline`](crate::device::GraphicsDevice::create_pipeline).
    ///
    /// # Arguments
    ///
    /// * `device` - The graphics device for executing graphs.
    /// * `frames_in_flight` - Number of frames that can be in flight simultaneously.
    ///   Typically 2 or 3. Must be at least 1.
    ///
    /// # Panics
    ///
    /// Panics if `frames_in_flight` is 0.
    pub(crate) fn new(device: Arc<GraphicsDevice>, frames_in_flight: usize) -> Self {
        assert!(frames_in_flight > 0, "frames_in_flight must be at least 1");

        Self {
            device,
            frame_fences: (0..frames_in_flight).map(|_| Vec::new()).collect(),
            current_slot: 0,
            frames_in_flight,
            frame_count: 0,
            ring_buffers: Vec::new(),
            graph_pool: Vec::new(),
            slot_graphs: (0..frames_in_flight).map(|_| Vec::new()).collect(),
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
    ///     let mut schedule = pipeline.begin_frame()?;  // Wait + get schedule
    ///
    ///     schedule.submit(prepass_graph);
    ///     schedule.submit(main_graph);
    ///
    ///     pipeline.end_frame(schedule);
    /// }
    /// ```
    /// Issue async readback of every `ReadbackBuffer` op in a slot's retired
    /// graphs.
    ///
    /// Called only after the slot's fence has signalled (GPU finished writing
    /// the readback sources). Each `src[src_range]` is mapped non-blockingly;
    /// the map's completion callback copies it into the op's CPU `dst` a tick
    /// later (next `device.poll` on native, next event-loop turn on wasm) — so
    /// this never blocks and is safe on the single browser thread (#33).
    /// Consumers poll `dst` for readiness (it stays as-is until filled).
    ///
    /// Two-gate recycling: the map holds its own clones of the wgpu buffer and
    /// the `dst`/`map_pending` handles, so it outlives this graph's recycle. The
    /// `map_pending` flag on `src` prevents an overlapping map of the same
    /// buffer.
    fn process_readbacks(&self, slot: usize) {
        use crate::graph::TransferOperation;
        for graph in &self.slot_graphs[slot] {
            for pass in graph.passes() {
                let Some(transfer) = pass.as_transfer() else {
                    continue;
                };
                let Some(config) = transfer.transfer_config() else {
                    continue;
                };
                for op in &config.operations {
                    if let TransferOperation::ReadbackBuffer {
                        src,
                        src_range,
                        dst,
                    } = op
                    {
                        src.read_mapped_async(
                            src_range.start as u64,
                            (src_range.end - src_range.start) as u64,
                            Arc::clone(dst),
                        );
                    }
                }
            }
        }
    }

    /// Begin a frame, blocking until this slot's previous GPU work completes.
    ///
    /// Native only in spirit — a browser thread cannot block. It is a thin
    /// blocking wrapper over the canonical non-blocking [`try_begin_frame`]: it
    /// waits the slot's fences, then runs the exact same recycle-and-schedule
    /// body.
    ///
    /// On timeout or device loss the GPU may still be using this slot's
    /// resources — the error propagates WITHOUT advancing frame state or
    /// recycling anything. The fences stay in the slot, so a later call retries.
    pub fn begin_frame(&mut self) -> Result<FrameSchedule, GraphicsError> {
        profile_scope!("begin_frame");
        {
            profile_scope!("wait_fence");
            for fence in &self.frame_fences[self.current_slot] {
                fence.wait()?;
            }
        }
        Ok(self.recycle_and_schedule())
    }

    /// Canonical non-blocking begin-frame: `Ok(None)` when this slot's previous
    /// GPU work is not yet done (the caller should retry next tick), `Ok(Some)`
    /// when the slot is ready and recycled. This is the shape the wasm frame
    /// loop uses — it never blocks; completion arrives via each fence's
    /// `on_submitted_work_done` callback and is observed by `is_signaled`.
    pub fn try_begin_frame(&mut self) -> Result<Option<FrameSchedule>, GraphicsError> {
        profile_scope!("try_begin_frame");
        if !self.frame_fences[self.current_slot]
            .iter()
            .all(|fence| fence.is_signaled())
        {
            return Ok(None);
        }
        Ok(Some(self.recycle_and_schedule()))
    }

    /// The shared body of [`begin_frame`]/[`try_begin_frame`], run once the
    /// slot's fence is known signaled: drain readbacks, advance backend frame
    /// state, recycle the slot's graphs, and hand out a fresh [`FrameSchedule`].
    /// Keeping it in one place means the blocking (native) and non-blocking
    /// (wasm) entry points exercise identical recycling logic.
    fn recycle_and_schedule(&mut self) -> FrameSchedule {
        // Fence passed: the GPU finished this slot's work, so readback source
        // buffers are safe to read. Drain them before recycling the graphs.
        self.process_readbacks(self.current_slot);

        // Advance backend frame state (layout tracker, descriptor pool,
        // command buffer cleanup)
        {
            profile_scope!("advance_frame");
            self.device.advance_frame();
        }

        // Now safe: reset old graphs from this slot. The fence guarantees the
        // GPU is done, so dropping Arc references to GPU resources is safe.
        {
            profile_scope!("recycle_slot_graphs");
            for mut graph in self.slot_graphs[self.current_slot].drain(..) {
                graph.reset();
                self.graph_pool.push(graph);
            }
        }

        self.frame_count += 1;

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

        FrameSchedule::new(
            self.device.clone(),
            self.current_slot,
            ring_buffer,
            graph_pool,
        )
    }

    /// Begin a new frame with a timeout.
    ///
    /// Like [`begin_frame`](Self::begin_frame), but returns `Ok(None)` if the
    /// timeout elapses before the frame slot becomes available.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for the frame slot.
    ///
    /// # Returns
    ///
    /// `Ok(Some(schedule))` if the frame slot is ready, `Ok(None)` if the
    /// timeout elapsed, `Err` on device loss or fence-wait failure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::time::Duration;
    ///
    /// match pipeline.begin_frame_timeout(Duration::from_millis(100))? {
    ///     Some(mut schedule) => {
    ///         // Normal frame processing
    ///         schedule.submit(graph);
    ///         pipeline.end_frame(schedule);
    ///     }
    ///     None => {
    ///         log::warn!("GPU is falling behind!");
    ///         // Handle the timeout (skip frame, reduce quality, etc.)
    ///     }
    /// }
    /// ```
    pub fn begin_frame_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<FrameSchedule>, GraphicsError> {
        profile_scope!("begin_frame_timeout");

        // Wait every submit of the slot, sharing one deadline. On timeout the
        // fences stay in the slot so a later call retries.
        let start = web_time::Instant::now();
        for fence in &self.frame_fences[self.current_slot] {
            let elapsed = start.elapsed();
            if elapsed >= timeout || !fence.wait_timeout(timeout - elapsed)? {
                return Ok(None);
            }
        }

        Ok(Some(self.recycle_and_schedule()))
    }

    /// End the current frame.
    ///
    /// Takes ownership of the schedule, extracts its per-submit fences, and
    /// advances to the next frame slot.
    ///
    /// # Arguments
    ///
    /// * `schedule` - The schedule returned from [`begin_frame`](Self::begin_frame),
    ///   after submitting the frame's graphs via [`submit`](FrameSchedule::submit).
    ///
    /// # Panics
    ///
    /// Panics if [`submit`](FrameSchedule::submit) was never called on the schedule.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut schedule = pipeline.begin_frame()?;
    /// schedule.submit(main_graph);
    /// pipeline.end_frame(schedule);  // Takes ownership
    /// ```
    pub fn end_frame(&mut self, mut schedule: FrameSchedule) {
        profile_scope!("end_frame");

        let fences = schedule.take_fences();

        // Return ring buffer to its slot (if configured)
        if !self.ring_buffers.is_empty()
            && let Some(ring) = schedule.take_ring_buffer()
        {
            self.ring_buffers[self.current_slot] = Some(ring);
        }

        // Store submitted graphs per-slot (DON'T reset — GPU may still be using them).
        // Their Arc references keep GPU resources alive until begin_frame() resets them
        // after the fence wait guarantees the GPU is done.
        self.slot_graphs[self.current_slot] = schedule.take_submitted_graphs();

        // Return unused graphs to the pool directly
        self.graph_pool.extend(schedule.take_graph_pool());

        // Store the per-submit fences for this slot
        self.frame_fences[self.current_slot] = fences;

        // Advance to next slot
        self.current_slot = (self.current_slot + 1) % self.frames_in_flight;

        // Mark frame end for Tracy
        frame_mark!();
    }

    /// Wait for the current frame slot to be ready.
    ///
    /// This blocks until all of the current slot's fences are signaled. This
    /// is more efficient than [`wait_idle`](Self::wait_idle) when you only
    /// need to ensure one frame slot is available.
    ///
    /// # When to Call
    ///
    /// - Before modifying resources used only by the current frame slot
    ///
    /// For swapchain resize, use [`wait_idle`](Self::wait_idle) instead — the
    /// surface is shared across all slots, so all in-flight work must complete.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Wait for current slot before modifying slot-local resources
    /// pipeline.wait_current_slot()?;
    /// update_slot_local_buffer(pipeline.current_slot());
    /// ```
    pub fn wait_current_slot(&self) -> Result<(), GraphicsError> {
        for fence in &self.frame_fences[self.current_slot] {
            fence.wait()?;
        }
        Ok(())
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
    /// `Ok(true)` if the slot is ready, `Ok(false)` if the timeout elapsed,
    /// `Err` on device loss or fence-wait failure.
    pub fn wait_current_slot_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> Result<bool, GraphicsError> {
        let start = web_time::Instant::now();
        for fence in &self.frame_fences[self.current_slot] {
            let elapsed = start.elapsed();
            if elapsed >= timeout || !fence.wait_timeout(timeout - elapsed)? {
                return Ok(false);
            }
        }
        Ok(true) // No fences means slot is ready
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
    ///     let mut schedule = pipeline.begin_frame()?;
    ///     // ... render ...
    ///     pipeline.end_frame(schedule);
    /// }
    ///
    /// // Shutdown
    /// pipeline.wait_idle()?;  // Wait for ALL GPU work
    /// drop(device);          // Safe to destroy
    /// ```
    /// # Errors
    ///
    /// Returns an error on fence-wait timeout or device loss. In that case the
    /// GPU may still be executing — do NOT destroy resources or call
    /// [`recycle_all_graphs`](Self::recycle_all_graphs).
    pub fn wait_idle(&self) -> Result<(), GraphicsError> {
        profile_scope!("wait_idle");

        for fence in self.frame_fences.iter().flatten() {
            fence.wait()?;
        }
        Ok(())
    }

    /// Recycle all submitted graphs across every frame slot.
    ///
    /// This releases all `Arc` references (e.g. swapchain texture views) held
    /// by graphs that were kept alive for GPU safety. **Must only be called
    /// after [`wait_idle`](Self::wait_idle)** — the fence wait guarantees the
    /// GPU is done with these resources.
    ///
    /// # When to Call
    ///
    /// Before reconfiguring the surface: render graphs store
    /// `Arc<wgpu::TextureView>` clones of the swapchain back buffers. On DX12,
    /// `ResizeBuffers` requires all back-buffer references to be released first.
    pub fn recycle_all_graphs(&mut self) {
        for slot_graphs in &mut self.slot_graphs {
            for mut graph in slot_graphs.drain(..) {
                graph.reset();
                self.graph_pool.push(graph);
            }
        }
    }

    /// Wait for all in-flight GPU work with a timeout.
    ///
    /// Like [`wait_idle`](Self::wait_idle), but returns `Ok(false)` if the
    /// timeout elapses before all work completes.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum total time to wait across all fences.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if GPU is idle, `Ok(false)` if the timeout elapsed, `Err`
    /// on device loss or fence-wait failure.
    pub fn wait_idle_timeout(&self, timeout: Duration) -> Result<bool, GraphicsError> {
        let start = web_time::Instant::now();

        for fence in self.frame_fences.iter().flatten() {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Ok(false);
            }

            let remaining = timeout - elapsed;
            if !fence.wait_timeout(remaining)? {
                return Ok(false);
            }
        }

        Ok(true)
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
        self.ring_buffers = Vec::with_capacity(self.frames_in_flight);
        for slot in 0..self.frames_in_flight {
            let slot_label = format!("{}_slot{}", label, slot);
            let ring =
                RingBuffer::with_alignment(&self.device, capacity, usage, &slot_label, alignment)?;
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
    /// Returns `true` if all of the slot's fences are signaled or if the slot
    /// hasn't been used yet (no fences).
    pub fn is_slot_ready(&self, slot: usize) -> bool {
        assert!(slot < self.frames_in_flight, "Invalid slot index");

        self.frame_fences[slot]
            .iter()
            .all(|fence| fence.is_signaled())
    }

    /// Check if all frame slots are ready (non-blocking).
    ///
    /// Returns `true` if [`wait_idle`](Self::wait_idle) would return immediately.
    pub fn is_idle(&self) -> bool {
        self.frame_fences
            .iter()
            .flatten()
            .all(|fence| fence.is_signaled())
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
    use crate::instance::GraphicsInstance;

    fn make_test_graph(name: &str) -> RenderGraph {
        let mut graph = RenderGraph::new();
        graph.add_graphics_pass(GraphicsPass::new(name.into()));
        graph
    }

    fn make_test_pipeline(frames_in_flight: usize) -> FramePipeline {
        let instance = GraphicsInstance::new().unwrap();
        let device = instance.create_device().unwrap();
        FramePipeline::new(device, frames_in_flight)
    }

    #[test]
    fn test_new() {
        let pipeline = make_test_pipeline(2);
        assert_eq!(pipeline.frames_in_flight(), 2);
        assert_eq!(pipeline.current_slot(), 0);
        assert_eq!(pipeline.frame_count(), 0);
    }

    #[test]
    fn test_begin_frame_returns_schedule() {
        let mut pipeline = make_test_pipeline(2);
        let _schedule = pipeline.begin_frame().unwrap();

        assert_eq!(pipeline.frame_count(), 1);
    }

    #[test]
    fn test_end_frame_advances_slot() {
        let mut pipeline = make_test_pipeline(3);
        assert_eq!(pipeline.current_slot(), 0);

        let mut schedule = pipeline.begin_frame().unwrap();
        schedule.render(make_test_graph("present"));
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 1);

        // Signal fences so next begin_frame doesn't block
        pipeline.signal_all_fences();

        let mut schedule = pipeline.begin_frame().unwrap();
        schedule.render(make_test_graph("present"));
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 2);

        pipeline.signal_all_fences();

        let mut schedule = pipeline.begin_frame().unwrap();
        schedule.render(make_test_graph("present"));
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 0); // Wraps around
    }

    #[test]
    fn test_is_slot_ready_unused() {
        let pipeline = make_test_pipeline(2);
        assert!(pipeline.is_slot_ready(0));
        assert!(pipeline.is_slot_ready(1));
    }

    #[test]
    fn test_is_idle_initial() {
        let pipeline = make_test_pipeline(2);
        assert!(pipeline.is_idle());
    }

    #[test]
    fn test_wait_idle_no_fences() {
        let pipeline = make_test_pipeline(2);
        // Should return immediately when no fences
        pipeline.wait_idle().unwrap();
    }

    #[test]
    fn test_frame_lifecycle() {
        let mut pipeline = make_test_pipeline(2);

        // Frame 0
        let mut schedule = pipeline.begin_frame().unwrap();
        assert_eq!(pipeline.frame_count(), 1);
        schedule.render(make_test_graph("present"));
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 1);

        // Simulate GPU completing frame 0
        pipeline.signal_all_fences();

        // Frame 1
        let mut schedule = pipeline.begin_frame().unwrap();
        assert_eq!(pipeline.frame_count(), 2);
        schedule.render(make_test_graph("present"));
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 0);

        // Simulate GPU completing frame 1
        pipeline.signal_all_fences();

        // Frame 2 (reuses slot 0)
        let _schedule = pipeline.begin_frame().unwrap();
        assert_eq!(pipeline.frame_count(), 3);
        assert_eq!(pipeline.current_slot(), 0);
    }

    #[test]
    fn test_begin_frame_timeout_ready() {
        let mut pipeline = make_test_pipeline(2);

        // Should succeed immediately (no fence to wait on)
        let schedule = pipeline
            .begin_frame_timeout(Duration::from_millis(1))
            .unwrap();
        assert!(schedule.is_some());
        assert_eq!(pipeline.frame_count(), 1);
    }

    #[test]
    fn test_wait_idle_timeout_ready() {
        let pipeline = make_test_pipeline(2);

        // Should succeed immediately (no fences)
        assert!(
            pipeline
                .wait_idle_timeout(Duration::from_millis(1))
                .unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "Invalid slot index")]
    fn test_is_slot_ready_invalid() {
        let pipeline = make_test_pipeline(2);
        pipeline.is_slot_ready(5); // Invalid
    }

    #[test]
    #[should_panic(expected = "submit() must be called at least once before end_frame()")]
    fn test_end_frame_without_submit_panics() {
        let mut pipeline = make_test_pipeline(2);
        let schedule = pipeline.begin_frame().unwrap();
        pipeline.end_frame(schedule); // Panics - submit() not called
    }

    #[test]
    fn test_full_frame_with_render() {
        let mut pipeline = make_test_pipeline(2);

        let mut schedule = pipeline.begin_frame().unwrap();
        schedule.render(make_test_graph("main"));

        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 1);
    }

    #[test]
    fn test_full_frame_with_multiple_submits() {
        let mut pipeline = make_test_pipeline(2);

        // Two graphs in one frame, each its own submit; the slot must hold
        // one fence per submit and recycle only when all are signaled.
        let mut schedule = pipeline.begin_frame().unwrap();
        schedule.submit(make_test_graph("prepass"));
        schedule.submit(make_test_graph("main"));
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 1);

        pipeline.signal_all_fences();
        pipeline.wait_idle().unwrap();
        assert!(pipeline.is_idle());

        // The slot recycles cleanly into the next frame.
        pipeline.signal_all_fences();
        let mut schedule = pipeline.begin_frame().unwrap();
        schedule.submit(make_test_graph("main"));
        pipeline.end_frame(schedule);
        assert_eq!(pipeline.current_slot(), 0);
    }
}
