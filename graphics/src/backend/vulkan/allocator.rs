//! GPU memory allocator integration using gpu-allocator.

use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::error::GraphicsError;

/// Create a memory allocator for the Vulkan device.
pub fn create_allocator(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
) -> Result<Allocator, GraphicsError> {
    let allocator = Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device,
        physical_device,
        debug_settings: Default::default(),
        buffer_device_address: false,
        allocation_sizes: gpu_allocator::AllocationSizes::default(),
    })
    .map_err(|e| {
        GraphicsError::InitializationFailed(format!("Failed to create memory allocator: {}", e))
    })?;

    Ok(allocator)
}
