//! Pooled staging-buffer belt for frame-graph uploads (ADR-021).
//!
//! `TransferOperation::WriteBuffer` previously created and destroyed a
//! dedicated staging buffer per operation per frame. The belt sub-allocates
//! from pooled chunks instead: a chunk is bump-allocated during a frame,
//! owned by that frame's slot while the GPU may still read it, and returned
//! to the free pool once the slot's fence has signaled — the same lifetime
//! guarantee the per-operation buffers had, without the churn.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::error::GraphicsError;

use super::MAX_FRAMES_IN_FLIGHT;

/// Size of a pooled staging chunk. Uploads larger than this get a dedicated
/// chunk that is destroyed (not pooled) when its frame slot retires, so a
/// single huge upload does not pin its memory forever.
const STAGING_CHUNK_SIZE: u64 = 4 * 1024 * 1024;

/// Sub-allocation alignment. Buffer→buffer copies need none, but
/// buffer→image copies require texel-block-size alignment on the source
/// offset; 256 covers every format cheaply.
const STAGING_ALIGN: u64 = 256;

/// One host-visible `TRANSFER_SRC` buffer the belt bump-allocates from.
struct StagingChunk {
    buffer: vk::Buffer,
    /// `Option` so destruction can move it into `Allocator::free`.
    allocation: Option<Allocation>,
    capacity: u64,
    /// Bump offset of the next sub-allocation (always `STAGING_ALIGN`ed).
    used: u64,
}

impl StagingChunk {
    fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(allocation) = self.allocation.take()
            && let Err(e) = allocator.free(allocation)
        {
            log::error!("Failed to free staging chunk allocation: {e}");
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

/// Pooled staging memory, partitioned by frame slot.
#[derive(Default)]
pub(crate) struct StagingBelt {
    /// Chunks written during each in-flight frame; the GPU may still be
    /// reading them until the slot's fence signals.
    in_use: [Vec<StagingChunk>; MAX_FRAMES_IN_FLIGHT],
    /// Retired chunks ready for reuse.
    free: Vec<StagingChunk>,
}

impl StagingBelt {
    /// Copy `bytes` into belt memory owned by `slot`.
    ///
    /// Returns the staging buffer and the offset the caller must copy from.
    /// The memory stays valid until `retire_slot(slot)` runs — i.e. until the
    /// slot's frame fence has signaled.
    pub fn write(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        slot: usize,
        bytes: &[u8],
    ) -> Result<(vk::Buffer, u64), GraphicsError> {
        let need = bytes.len() as u64;

        let fits = self.in_use[slot]
            .last()
            .is_some_and(|chunk| chunk.capacity - chunk.used >= need);
        if !fits {
            let chunk = self.take_or_create(device, allocator, need)?;
            self.in_use[slot].push(chunk);
        }

        let chunk = self.in_use[slot]
            .last_mut()
            .expect("chunk pushed or verified above");
        let offset = chunk.used;
        let mapped = chunk
            .allocation
            .as_ref()
            .and_then(|a| a.mapped_ptr())
            .ok_or_else(|| {
                GraphicsError::Internal("staging chunk memory is not host-mapped".into())
            })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (mapped.as_ptr() as *mut u8).add(offset as usize),
                bytes.len(),
            );
        }
        chunk.used = (offset + need)
            .next_multiple_of(STAGING_ALIGN)
            .min(chunk.capacity);

        Ok((chunk.buffer, offset))
    }

    /// Reuse a free chunk that can hold `need` bytes, or create a new one.
    fn take_or_create(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        need: u64,
    ) -> Result<StagingChunk, GraphicsError> {
        if let Some(pos) = self.free.iter().position(|c| c.capacity >= need) {
            return Ok(self.free.swap_remove(pos));
        }

        let capacity = need.max(STAGING_CHUNK_SIZE);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!(
                "Failed to create staging belt chunk: {e:?}"
            ))
        })?;

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "staging_belt_chunk",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| {
                unsafe {
                    device.destroy_buffer(buffer, None);
                }
                GraphicsError::ResourceCreationFailed(format!(
                    "Failed to allocate staging belt chunk memory: {e}"
                ))
            })?;

        if let Err(e) =
            unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
        {
            let _ = allocator.free(allocation);
            unsafe {
                device.destroy_buffer(buffer, None);
            }
            return Err(GraphicsError::ResourceCreationFailed(format!(
                "Failed to bind staging belt chunk memory: {e:?}"
            )));
        }

        Ok(StagingChunk {
            buffer,
            allocation: Some(allocation),
            capacity,
            used: 0,
        })
    }

    /// Return `slot`'s chunks to the pool. Must only be called after the
    /// slot's frame fence has signaled (the GPU is done reading them).
    /// Oversized chunks are destroyed instead of pooled.
    pub fn retire_slot(&mut self, device: &ash::Device, allocator: &mut Allocator, slot: usize) {
        for mut chunk in self.in_use[slot].drain(..) {
            if chunk.capacity > STAGING_CHUNK_SIZE {
                chunk.destroy(device, allocator);
            } else {
                chunk.used = 0;
                self.free.push(chunk);
            }
        }
    }

    /// Destroy every chunk. The GPU must be idle.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for slot in &mut self.in_use {
            for chunk in slot.drain(..) {
                chunk.destroy(device, allocator);
            }
        }
        for chunk in self.free.drain(..) {
            chunk.destroy(device, allocator);
        }
    }
}
