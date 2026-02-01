//! Vulkan command pool and buffer management.

use ash::vk;

use crate::error::GraphicsError;

/// Create a command pool for graphics operations.
pub fn create_command_pool(
    device: &ash::Device,
    queue_family_index: u32,
) -> Result<vk::CommandPool, GraphicsError> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let pool = unsafe { device.create_command_pool(&pool_info, None) }.map_err(|e| {
        GraphicsError::InitializationFailed(format!("Failed to create command pool: {:?}", e))
    })?;

    Ok(pool)
}
