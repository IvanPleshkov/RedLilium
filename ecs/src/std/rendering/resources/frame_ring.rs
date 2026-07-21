//! A per-frame dynamic-uniform ring buffer, as an ECS resource.

use std::sync::Arc;

use redlilium_graphics::{
    Buffer, BufferUsage, GraphicsDevice, GraphicsError, MAX_FRAMES_IN_FLIGHT, RingAllocation,
    RingBuffer,
};

/// Alignment every allocation is rounded up to: a dynamic uniform offset must
/// satisfy the device's `minUniformBufferOffsetAlignment`, 256 bytes on every
/// backend the engine targets.
const ALIGNMENT: u64 = RingBuffer::DEFAULT_ALIGNMENT;

/// Render-side scratch buffer for per-draw dynamic-uniform data. Render systems
/// `push` their uniforms each frame and bind the ring's [`buffer`](Self::buffer)
/// at the returned dynamic offset (group with a dynamic-uniform binding).
///
/// # Why the buffer is split into regions
///
/// The GPU reads these bytes for as long as the frame that referenced them is in
/// flight, so a frame must never write where a still-pending frame is reading.
/// The buffer is therefore divided into [`MAX_FRAMES_IN_FLIGHT`] equal regions
/// and [`begin_frame`](Self::begin_frame) advances to the next one each frame.
/// A region is reused only every `MAX_FRAMES_IN_FLIGHT` frames, by which point
/// the host's own frame-slot fence has already been waited — so reuse is safe by
/// construction rather than by hope. Using the maximum (not the configured)
/// frames-in-flight keeps that true for any swapchain configuration.
///
/// This replaced a single monotonic ring that called `RingBuffer::reset()` —
/// i.e. `write_offset = 0` — the moment it filled, in the middle of whatever
/// frame happened to be recording, with no fence. That silently rewrote the
/// bytes at offset 0 onward, where the camera uniforms (including
/// `prev_view_projection`) live, while passes already recorded into the graph
/// still pointed at them. The corruption surfaced as reprojection error, whose
/// magnitude scales with screen-space velocity: geometry sweeping past a
/// rotating camera juddered while whatever the camera tracked looked fine.
///
/// One instance can back many bind groups: they all bind the same buffer, each
/// at its own dynamic offset.
pub struct FrameRing {
    ring: RingBuffer,
    /// Bytes reserved for a single frame.
    region: u64,
    /// Region the current frame writes into.
    slot: u64,
    /// Bytes consumed so far in the current frame's region.
    used: u64,
    /// Whether the current frame has already exhausted its region — the report
    /// is once per frame, not once per starved push.
    overflowed: bool,
    /// Offset of the last allocation that fit. Handed out again once the region
    /// is exhausted, so a starved draw reads valid (if wrong) uniforms instead
    /// of writing outside its region and corrupting another frame.
    last_offset: u64,
}

impl FrameRing {
    /// Create a ring able to serve `per_frame_capacity` bytes to each frame
    /// (`UNIFORM | COPY_DST`). The allocation is `MAX_FRAMES_IN_FLIGHT` times
    /// that, one region per in-flight frame.
    ///
    /// Size it for the busiest frame: one `ModelUniforms` per visible renderable
    /// per camera, plus each camera's pass uniforms, every one rounded up to
    /// [`ALIGNMENT`]. Overshooting costs only uniform memory; undershooting is
    /// reported loudly and renders that frame with stale uniforms.
    pub fn new(
        device: &Arc<GraphicsDevice>,
        per_frame_capacity: u64,
        label: &str,
    ) -> Result<Self, GraphicsError> {
        let region = per_frame_capacity.max(ALIGNMENT);
        Ok(Self {
            ring: RingBuffer::new(
                device,
                region * MAX_FRAMES_IN_FLIGHT as u64,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                label,
            )?,
            region,
            slot: 0,
            used: 0,
            overflowed: false,
            last_offset: 0,
        })
    }

    /// Move to the next frame's region. Call exactly once per frame, before any
    /// [`push`](Self::push) and after the host has waited the frame-slot fence
    /// (the render dispatcher does this alongside the temporal history rotation).
    pub fn begin_frame(&mut self) {
        self.slot = (self.slot + 1) % MAX_FRAMES_IN_FLIGHT as u64;
        self.used = 0;
        self.overflowed = false;
        self.last_offset = self.slot * self.region;
    }

    /// Allocate and write `data` inside this frame's region, returning its byte
    /// offset for use as a dynamic uniform offset.
    ///
    /// If the region is exhausted the write is dropped and the previous offset
    /// is returned — wrong on screen, but it can never scribble over a frame the
    /// GPU is still reading. The shortfall is logged once per frame.
    pub fn push(&mut self, data: &[u8]) -> u32 {
        let size = data.len() as u64;
        let aligned = self.used.next_multiple_of(ALIGNMENT);
        if aligned + size > self.region {
            if !self.overflowed {
                self.overflowed = true;
                log::error!(
                    "FrameRing region exhausted: {} bytes per frame, needed at least {} — \
                     this frame renders with stale uniforms. Raise the ring's per-frame capacity.",
                    self.region,
                    aligned + size
                );
            }
            return self.last_offset as u32;
        }

        let offset = self.slot * self.region + aligned;
        self.used = aligned + size;
        self.last_offset = offset;
        let _ = self.ring.write(&RingAllocation::new(offset, size), data);
        offset as u32
    }

    /// The underlying GPU buffer — bind this at the offsets returned by
    /// [`push`](Self::push).
    pub fn buffer(&self) -> &Arc<Buffer> {
        self.ring.buffer()
    }

    /// Bytes this frame has consumed of its region — for diagnostics and for the
    /// tests that pin the region discipline.
    pub fn used(&self) -> u64 {
        self.used
    }

    /// Bytes available to a single frame.
    pub fn region_capacity(&self) -> u64 {
        self.region
    }

    /// Byte range this frame writes into — the invariant the tests pin.
    #[cfg(test)]
    fn region_bounds(&self) -> (u64, u64) {
        (self.slot * self.region, (self.slot + 1) * self.region)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_graphics::GraphicsInstance;

    fn ring(per_frame: u64) -> FrameRing {
        let device = GraphicsInstance::new()
            .expect("graphics instance")
            .create_device()
            .expect("graphics device");
        FrameRing::new(&device, per_frame, "test_frame_ring").expect("frame ring")
    }

    /// The point of the whole design: consecutive frames must not share bytes,
    /// because the GPU is still reading the previous frame's while the next one
    /// records. The old ring reset to offset 0 mid-frame and violated this.
    #[test]
    fn consecutive_frames_write_disjoint_regions() {
        let mut ring = ring(4096);
        let mut seen: Vec<(u64, u64)> = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            ring.begin_frame();
            let offset = ring.push(&[0u8; 64]) as u64;
            let (start, end) = ring.region_bounds();
            assert!(
                (start..end).contains(&offset),
                "offset {offset} escaped its region {start}..{end}"
            );
            for (prev_start, prev_end) in &seen {
                assert!(
                    end <= *prev_start || start >= *prev_end,
                    "region {start}..{end} overlaps {prev_start}..{prev_end}"
                );
            }
            seen.push((start, end));
        }
    }

    /// A region is reused only after every other in-flight frame has had one,
    /// which is exactly the window the host's frame fence covers.
    #[test]
    fn regions_are_reused_only_after_a_full_cycle() {
        let mut ring = ring(4096);
        let mut first = None;
        for frame in 0..=MAX_FRAMES_IN_FLIGHT {
            ring.begin_frame();
            let start = ring.region_bounds().0;
            match first {
                None => first = Some(start),
                Some(first) if frame == MAX_FRAMES_IN_FLIGHT => {
                    assert_eq!(start, first, "the cycle must close on itself")
                }
                Some(first) => assert_ne!(start, first, "frame {frame} reused a live region"),
            }
        }
    }

    /// Allocations advance and stay aligned, and a fresh frame starts over —
    /// so a frame's own pushes never alias each other either.
    #[test]
    fn pushes_advance_and_realign_each_frame() {
        let mut ring = ring(4096);
        ring.begin_frame();
        let a = ring.push(&[0u8; 8]) as u64;
        let b = ring.push(&[0u8; 8]) as u64;
        assert_eq!(b - a, ALIGNMENT, "pushes must not overlap");
        assert_eq!(a % ALIGNMENT, 0);

        let used = ring.used();
        ring.begin_frame();
        assert!(ring.used() < used, "a new frame restarts its region");
    }

    /// Overflow must stay inside the region rather than wrapping into another
    /// frame's bytes: the draw renders wrong, but nothing is corrupted.
    #[test]
    fn overflow_stays_inside_the_region() {
        let mut ring = ring(ALIGNMENT * 2);
        ring.begin_frame();
        let (start, end) = ring.region_bounds();
        let mut last = 0;
        for _ in 0..8 {
            last = ring.push(&[0u8; 16]) as u64;
            assert!(
                (start..end).contains(&last),
                "overflow escaped the region {start}..{end}"
            );
        }
        // Past capacity the ring keeps handing back the last good offset.
        assert_eq!(last, start + ALIGNMENT);
    }
}
