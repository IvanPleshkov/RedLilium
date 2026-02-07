//! Ring buffer for efficient GPU data streaming.
//!
//! A ring buffer (also called a circular buffer) is an optimization for streaming
//! data to the GPU each frame. Instead of creating new buffers or waiting for the
//! GPU to finish, a ring buffer pre-allocates a large buffer and writes to
//! consecutive regions, wrapping around when reaching the end.
//!
//! # Usage
//!
//! Ring buffers are ideal for:
//! - Per-frame uniform data (camera matrices, time, etc.)
//! - Dynamic vertex data (particles, UI elements)
//! - Indirect draw arguments
//!
//! # Example
//!
//! ```ignore
//! // Create a ring buffer for uniform data
//! let mut ring = RingBuffer::new(
//!     &device,
//!     64 * 1024, // 64 KB
//!     BufferUsage::UNIFORM | BufferUsage::COPY_DST,
//!     "uniform_ring",
//! )?;
//!
//! // Each frame, allocate space and write data
//! if let Some(alloc) = ring.allocate(std::mem::size_of::<CameraUniforms>() as u64, 256) {
//!     device.write_buffer(ring.buffer(), alloc.offset, bytemuck::bytes_of(&camera_data))?;
//!     // Use alloc.offset as the dynamic offset in bind groups
//! }
//!
//! // After N frames, reset the ring buffer
//! ring.reset();
//! ```

use std::sync::Arc;

use crate::device::GraphicsDevice;
use crate::error::GraphicsError;
use crate::resources::Buffer;
use crate::types::{BufferDescriptor, BufferUsage};

/// A sub-allocation from a ring buffer.
///
/// Contains the offset and size of the allocated region within the ring buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RingAllocation {
    /// Byte offset into the ring buffer.
    pub offset: u64,
    /// Size of the allocation in bytes.
    pub size: u64,
}

impl RingAllocation {
    /// Create a new ring allocation.
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    /// Get the end offset (offset + size).
    pub fn end(&self) -> u64 {
        self.offset + self.size
    }
}

/// A ring buffer for efficient GPU data streaming.
///
/// Ring buffers enable CPU-GPU overlap by allowing the CPU to write to different
/// regions of the buffer while the GPU reads from previously written regions.
///
/// # Alignment
///
/// Allocations are aligned to the specified alignment (default: 256 bytes).
/// This is important for:
/// - Uniform buffer dynamic offsets (typically require 256-byte alignment)
/// - Optimal memory access patterns
///
/// # Thread Safety
///
/// `RingBuffer` is NOT thread-safe. If you need concurrent access, wrap it in
/// a mutex or use separate ring buffers per thread.
pub struct RingBuffer {
    buffer: Arc<Buffer>,
    capacity: u64,
    write_offset: u64,
    default_alignment: u64,
    wrap_count: u64,
}

impl RingBuffer {
    /// Default alignment for allocations (256 bytes).
    ///
    /// This matches the typical minimum uniform buffer offset alignment
    /// required by most GPUs.
    pub const DEFAULT_ALIGNMENT: u64 = 256;

    /// Create a new ring buffer with the given capacity.
    ///
    /// # Arguments
    ///
    /// * `device` - The graphics device to create the buffer on
    /// * `capacity` - Total size of the ring buffer in bytes
    /// * `usage` - Buffer usage flags (RING flag is added automatically)
    /// * `label` - Debug label for the buffer
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ring = RingBuffer::new(
    ///     &device,
    ///     1024 * 1024, // 1 MB
    ///     BufferUsage::VERTEX | BufferUsage::COPY_DST,
    ///     "vertex_stream",
    /// )?;
    /// ```
    pub fn new(
        device: &Arc<GraphicsDevice>,
        capacity: u64,
        usage: BufferUsage,
        label: &str,
    ) -> Result<Self, GraphicsError> {
        Self::with_alignment(device, capacity, usage, label, Self::DEFAULT_ALIGNMENT)
    }

    /// Create a new ring buffer with custom alignment.
    ///
    /// # Arguments
    ///
    /// * `device` - The graphics device to create the buffer on
    /// * `capacity` - Total size of the ring buffer in bytes
    /// * `usage` - Buffer usage flags (RING flag is added automatically)
    /// * `label` - Debug label for the buffer
    /// * `alignment` - Alignment for allocations (must be power of 2)
    pub fn with_alignment(
        device: &Arc<GraphicsDevice>,
        capacity: u64,
        usage: BufferUsage,
        label: &str,
        alignment: u64,
    ) -> Result<Self, GraphicsError> {
        if !alignment.is_power_of_two() {
            return Err(GraphicsError::InvalidParameter(format!(
                "alignment must be a power of 2, got {alignment}"
            )));
        }

        if capacity == 0 {
            return Err(GraphicsError::InvalidParameter(
                "ring buffer capacity cannot be zero".to_string(),
            ));
        }

        // Ensure capacity is aligned
        let aligned_capacity = align_up(capacity, alignment);

        let descriptor = BufferDescriptor::new(aligned_capacity, usage | BufferUsage::RING)
            .with_label(format!("{label}_ring"));

        let buffer = device.create_buffer(&descriptor)?;

        Ok(Self {
            buffer,
            capacity: aligned_capacity,
            write_offset: 0,
            default_alignment: alignment,
            wrap_count: 0,
        })
    }

    /// Get the underlying GPU buffer.
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Get the total capacity of the ring buffer.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Get the current write offset.
    pub fn write_offset(&self) -> u64 {
        self.write_offset
    }

    /// Get the number of times the buffer has wrapped around.
    pub fn wrap_count(&self) -> u64 {
        self.wrap_count
    }

    /// Get the amount of space used since the last reset.
    pub fn used(&self) -> u64 {
        self.write_offset
    }

    /// Get the amount of space remaining before wrapping.
    pub fn remaining(&self) -> u64 {
        self.capacity - self.write_offset
    }

    /// Check if the buffer can accommodate an allocation of the given size.
    ///
    /// Returns true if the allocation would fit without wrapping.
    pub fn can_allocate(&self, size: u64) -> bool {
        let aligned_offset = align_up(self.write_offset, self.default_alignment);
        aligned_offset + size <= self.capacity
    }

    /// Allocate space from the ring buffer with default alignment.
    ///
    /// Returns `None` if the allocation would wrap around the buffer.
    /// Call [`reset`] to start a new frame and reclaim space.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the allocation in bytes
    ///
    /// # Returns
    ///
    /// A [`RingAllocation`] containing the offset and size, or `None` if
    /// there isn't enough contiguous space.
    ///
    /// [`reset`]: Self::reset
    pub fn allocate(&mut self, size: u64) -> Option<RingAllocation> {
        self.allocate_aligned(size, self.default_alignment)
    }

    /// Allocate space from the ring buffer with custom alignment.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the allocation in bytes
    /// * `alignment` - Required alignment (must be power of 2)
    ///
    /// # Returns
    ///
    /// A [`RingAllocation`] containing the offset and size, or `None` if
    /// there isn't enough contiguous space.
    pub fn allocate_aligned(&mut self, size: u64, alignment: u64) -> Option<RingAllocation> {
        debug_assert!(alignment.is_power_of_two(), "alignment must be power of 2");

        if size == 0 {
            return Some(RingAllocation::new(self.write_offset, 0));
        }

        // Align the current offset
        let aligned_offset = align_up(self.write_offset, alignment);

        // Check if we have enough space
        if aligned_offset + size > self.capacity {
            // Not enough space - would need to wrap
            return None;
        }

        // Update the write offset
        self.write_offset = aligned_offset + size;

        Some(RingAllocation::new(aligned_offset, size))
    }

    /// Reset the ring buffer to the beginning.
    ///
    /// Call this at the start of each frame (or after GPU synchronization)
    /// to reclaim the entire buffer space.
    ///
    /// # Warning
    ///
    /// Make sure the GPU has finished reading from the buffer before calling
    /// this, or you may overwrite data that's still in use. Typically this
    /// means waiting for the frame fence before resetting.
    pub fn reset(&mut self) {
        if self.write_offset > 0 {
            self.wrap_count += 1;
            self.write_offset = 0;
        }
    }

    /// Force wrap to the beginning of the buffer.
    ///
    /// This is similar to [`reset`] but doesn't increment the wrap counter.
    /// Use this when you need to wrap mid-frame and are managing synchronization
    /// manually.
    ///
    /// [`reset`]: Self::reset
    pub fn wrap(&mut self) {
        self.write_offset = 0;
    }
}

impl std::fmt::Debug for RingBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingBuffer")
            .field("capacity", &self.capacity)
            .field("write_offset", &self.write_offset)
            .field("default_alignment", &self.default_alignment)
            .field("wrap_count", &self.wrap_count)
            .field("buffer", &self.buffer.label())
            .finish()
    }
}

/// Align a value up to the given alignment.
#[inline]
fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;

    fn create_test_device() -> Arc<GraphicsDevice> {
        let instance = GraphicsInstance::new().unwrap();
        instance.create_device().unwrap()
    }

    #[test]
    fn test_ring_buffer_creation() {
        let device = create_test_device();
        let ring = RingBuffer::new(&device, 4096, BufferUsage::UNIFORM, "test").unwrap();

        assert_eq!(ring.capacity(), 4096);
        assert_eq!(ring.write_offset(), 0);
        assert_eq!(ring.wrap_count(), 0);
        assert_eq!(ring.used(), 0);
        assert_eq!(ring.remaining(), 4096);
    }

    #[test]
    fn test_ring_buffer_allocation() {
        let device = create_test_device();
        let mut ring =
            RingBuffer::with_alignment(&device, 1024, BufferUsage::UNIFORM, "test", 64).unwrap();

        // First allocation
        let alloc1 = ring.allocate(128).unwrap();
        assert_eq!(alloc1.offset, 0);
        assert_eq!(alloc1.size, 128);
        assert_eq!(ring.write_offset(), 128);

        // Second allocation (should be aligned to 64)
        let alloc2 = ring.allocate(64).unwrap();
        assert_eq!(alloc2.offset, 128); // Already aligned
        assert_eq!(alloc2.size, 64);
        assert_eq!(ring.write_offset(), 192);

        // Third allocation
        let alloc3 = ring.allocate(100).unwrap();
        assert_eq!(alloc3.offset, 192);
        assert_eq!(alloc3.size, 100);
    }

    #[test]
    fn test_ring_buffer_alignment() {
        let device = create_test_device();
        let mut ring =
            RingBuffer::with_alignment(&device, 1024, BufferUsage::UNIFORM, "test", 256).unwrap();

        // Allocate 100 bytes
        let alloc1 = ring.allocate(100).unwrap();
        assert_eq!(alloc1.offset, 0);
        assert_eq!(ring.write_offset(), 100);

        // Next allocation should be aligned to 256
        let alloc2 = ring.allocate(50).unwrap();
        assert_eq!(alloc2.offset, 256); // Aligned up from 100
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let device = create_test_device();
        let mut ring =
            RingBuffer::with_alignment(&device, 512, BufferUsage::UNIFORM, "test", 64).unwrap();

        // Allocate most of the buffer
        let alloc1 = ring.allocate(400).unwrap();
        assert_eq!(alloc1.offset, 0);
        // write_offset is now 400

        // Try to allocate more than remaining (aligned_offset would be 448, + 200 = 648 > 512)
        let alloc2 = ring.allocate(200);
        assert!(alloc2.is_none());

        // 64 bytes can still fit (aligned_offset = 448, + 64 = 512 == capacity)
        let alloc3 = ring.allocate(64);
        assert!(alloc3.is_some());
        assert_eq!(alloc3.unwrap().offset, 448); // aligned up from 400

        // Now buffer is full
        let alloc4 = ring.allocate(1);
        assert!(alloc4.is_none());
    }

    #[test]
    fn test_ring_buffer_reset() {
        let device = create_test_device();
        let mut ring =
            RingBuffer::with_alignment(&device, 512, BufferUsage::UNIFORM, "test", 64).unwrap();

        ring.allocate(256).unwrap();
        assert_eq!(ring.wrap_count(), 0);

        ring.reset();
        assert_eq!(ring.write_offset(), 0);
        assert_eq!(ring.wrap_count(), 1);

        // Can allocate again
        let alloc = ring.allocate(256).unwrap();
        assert_eq!(alloc.offset, 0);
    }

    #[test]
    fn test_ring_buffer_zero_allocation() {
        let device = create_test_device();
        let mut ring =
            RingBuffer::with_alignment(&device, 512, BufferUsage::UNIFORM, "test", 64).unwrap();

        let alloc = ring.allocate(0);
        assert!(alloc.is_some());
        assert_eq!(alloc.unwrap().size, 0);
    }

    #[test]
    fn test_ring_buffer_can_allocate() {
        let device = create_test_device();
        let mut ring =
            RingBuffer::with_alignment(&device, 512, BufferUsage::UNIFORM, "test", 64).unwrap();

        assert!(ring.can_allocate(400));
        ring.allocate(400).unwrap();
        assert!(!ring.can_allocate(200));
        assert!(ring.can_allocate(48)); // 400 + padding to 448, then 48 more = 496 < 512
    }

    #[test]
    fn test_ring_buffer_invalid_alignment() {
        let device = create_test_device();
        let result = RingBuffer::with_alignment(&device, 512, BufferUsage::UNIFORM, "test", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(100, 64), 128);
    }
}
