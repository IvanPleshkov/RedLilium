//! GPU memory allocator integration using gpu-allocator.

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::error::GraphicsError;

/// Create a memory allocator for the Vulkan device.
///
/// `buffer_device_address` must be true exactly when the device enabled the
/// Vulkan 1.2 `bufferDeviceAddress` feature (the ray-query bundle, #110):
/// gpu-allocator then tags every allocation with
/// `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS`, which `vkGetBufferDeviceAddress`
/// requires of the backing memory — and which is invalid without the feature.
pub fn create_allocator(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    buffer_device_address: bool,
) -> Result<Allocator, GraphicsError> {
    let allocator = Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device,
        physical_device,
        debug_settings: Default::default(),
        buffer_device_address,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .map_err(|e| {
        GraphicsError::InitializationFailed(format!("Failed to create memory allocator: {}", e))
    })?;

    Ok(allocator)
}
