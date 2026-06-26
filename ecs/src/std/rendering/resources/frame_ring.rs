//! A per-frame dynamic-uniform ring buffer, as an ECS resource.

use std::sync::Arc;

use redlilium_graphics::{Buffer, BufferUsage, GraphicsDevice, GraphicsError, RingBuffer};

/// Render-side scratch buffer for per-draw dynamic-uniform data. Render systems
/// `push` their uniforms each frame and bind the ring's [`buffer`](Self::buffer)
/// at the returned dynamic offset (group with a dynamic-uniform binding).
///
/// Allocations are monotonic and wrap when the ring fills (the [`RingBuffer`]'s
/// own frames-in-flight safety applies — never a synchronous overwrite of data a
/// pending frame still reads). One instance can back many bind groups: they all
/// bind the same buffer, each at its own dynamic offset.
pub struct FrameRing {
    ring: RingBuffer,
}

impl FrameRing {
    /// Create a ring of `capacity` bytes (`UNIFORM | COPY_DST`).
    pub fn new(
        device: &Arc<GraphicsDevice>,
        capacity: u64,
        label: &str,
    ) -> Result<Self, GraphicsError> {
        Ok(Self {
            ring: RingBuffer::new(
                device,
                capacity,
                BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                label,
            )?,
        })
    }

    /// Allocate and write `data`, returning its byte offset for use as a dynamic
    /// uniform offset. Wraps the ring if `data` doesn't fit (assumes a single
    /// element fits in an empty ring).
    pub fn push(&mut self, data: &[u8]) -> u32 {
        let size = data.len() as u64;
        let alloc = self.ring.allocate(size).unwrap_or_else(|| {
            self.ring.reset();
            self.ring
                .allocate(size)
                .expect("FrameRing too small for a single element")
        });
        let _ = self.ring.write(&alloc, data);
        alloc.offset as u32
    }

    /// The underlying GPU buffer — bind this at the offsets returned by
    /// [`push`](Self::push).
    pub fn buffer(&self) -> &Arc<Buffer> {
        self.ring.buffer()
    }
}
